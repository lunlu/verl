# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Megatron Actor.
In megatron actor, the differences are:
1. We only make minibatch

Note that our model doesn't have to be `MegatronModule` because we don't share embedding in the last layer
"""

import itertools
import logging
import os
from functools import partial
from typing import Iterable

import torch
import torch.distributed
from megatron.core import parallel_state as mpu

# from megatron.core.optimizer import DistributedOptimizer
from megatron.core.optimizer import DistributedOptimizer
from megatron.core.pipeline_parallel import get_forward_backward_func
from torch import nn

from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
from verl.utils.device import get_device_id, get_torch_device
from verl.utils.megatron.pipeline_parallel import make_batch_generator
from verl.utils.megatron.tensor_parallel import vocab_parallel_entropy, vocab_parallel_log_probs_from_logits
from verl.utils.megatron_utils import get_model_config
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.profiler.profile import Profiler
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches
from verl.utils.torch_functional import broadcast_dict_tensor
from verl.workers.actor import BasePPOActor

__all__ = ["MegatronPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class MegatronPPOActor(BasePPOActor):
    def __init__(
        self,
        config,
        model_config,
        hf_config,
        tf_config,
        actor_module: nn.ModuleList,
        actor_optimizer: DistributedOptimizer,
    ):
        """MeagtronPPOActor class. This class implements the simple PPO logics when the model is built with Megatron.

        Args:
            config (OmegaConf): the basic config that contains the hyper-parameters of PPO Actor. It must contain

                ``ppo_micro_batch_size_per_gpu``: micro batch size when updating ppo.

                ``ppo_mini_batch_size``: minibatch size when updating ppo using the batch data.

                ``ppo_epochs``: number of epochs to update the actor using the batch data.

                ``shuffle``: whether to shuffle the data after each ppo epoch.

                ``clip_ratio``: clip ratio of the ppo algorithm. See https://arxiv.org/abs/1707.06347.

                ``entropy_coeff``: entropy coefficient of the PPO loss. See https://arxiv.org/abs/1707.06347.
            model_config (OmegaConf): model configuration. It must contains ``model_config.vocab_size`` and
                ``model_config.hidden_size``
            hf_config (PretrainedConfig): huggingface config
            tf_config (TransformerConfig): mcore transformer config
            actor_module (nn.ModuleList): actor module is a ModuleList that contains a list of nn.Module in this
                pp stage.
                each nn.Module in this rank holds a vpp module chunk. See https://arxiv.org/pdf/2104.04473.pdf for
                more details.
                The actor module has some constraints to follow in order to use the updating logics implemented here

                1. It must implement unpad_input before any computation and pad_input after all the computation.
                Remove padding is an
                optimization that removes the padding tokens. See unpad_input and pad_input function in flash-attn
                (https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/bert_padding.py).

                2. Each pp stage must return the hidden state with the same shape [total_nnz, 1, hidden_size],
                where total_nnz is the number of valid tokens in this batch. If sequence parallel is enabled, the size
                of the hidden state is [total_nnz // tp, 1, hidden_size].
            actor_optimizer (DistributedOptimizer): currently, we only support DistributedOptimizer in Megatron.
                It implements
                zero1 optimizer that shards the optimizer state across dp ranks.

        >>> from megatron.training import get_model
        >>> from megatron.optimizer import get_megatron_optimizer
        >>> actor_module = get_model(megatron_actor_model_provider, wrap_with_ddp=True)
        >>> actor_module = nn.ModuleList(actor_module)
        >>> actor_optimizer = get_megatron_optimizer(actor_module)
        >>> actor = MegatronPPOActor(config=config,
        >>>                          model_config=actor_model_config,
        >>>                          hf_config=hf_config,
        >>>                          tf_config=tf_config,
        >>>                          actor_module=actor_module,
        >>>                          actor_optimizer=actor_optimizer)
        """
        super().__init__(config)
        self._validate_config(config)
        self.model_config = model_config
        self.hf_config = hf_config
        self.tf_config = tf_config
        self.actor_module = actor_module
        self.actor_optimizer: DistributedOptimizer = actor_optimizer
        self.use_torch_profiler = self.config.profiler.get("tool") == "torch"
        if self.use_torch_profiler:
            self.prof = Profiler(
                self.config.profiler, tool_config=self.config.profiler.get("tool_config", {}).get("torch", {})
            )
        else:
            self.prof = None
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        if self.use_fused_kernels:
            from verl.models.mcore.model_forward_fused import patch_fused_forward

            for model in self.actor_module:
                patch_fused_forward(model)

        config = get_model_config(self.actor_module[0])
        if torch.distributed.get_rank() == 0:
            print(config)
            
        # Check if MoE logging is enabled (default: False if not present)
        self.enable_moe_logging = self.config.get("enable_moe_logging", False)

        # Check if IcePop is enabled (default: False if not present)
        self.enable_icepop = self.config.get("enable_icepop", False)

        # Check if TIS is enabled (default: False if not present)
        self.enable_tis = self.config.get("enable_tis", False)
        
        self.current_global_step = "unknown"

    def _validate_config(self, config) -> None:
        """Validate config options not implemented for Megatron backend"""
        assert config.get("ulysses_sequence_parallel_size", 1) == 1
        if config.get("shuffle", False):
            assert config.data_loader_seed is not None, "If shuffle dataloader, seed must be manually set"
        if config.megatron.tensor_model_parallel_size == 1:
            print("[Warining] Because actor tp size == 1, set sp to False")
            config.megatron.sequence_parallel = False
        self.config = config
        
    def _compute_icepop_mask(self, inf_log_probs, old_log_probs, alpha=0.5, beta=2.0):
        """
        Compute IcePop double-sided masking to filter out noisy gradient updates.

        The masking function M(k) where k = p_train / p_infer:
        M(k) = { k  if k ∈ [α, β]
               { 0  otherwise

        Args:
            inf_log_probs: Log probabilities from inference engine (vLLM), shape [batch_size, response_length]
            old_log_probs: Log probabilities from training engine (FSDP old policy), shape [batch_size, response_length]
            alpha: Lower bound for probability ratio (default: 0.5)
            beta: Upper bound for probability ratio (default: 2.0)

        Returns:
            icepop_mask: Binary mask tensor, 1.0 for healthy updates, 0.0 for clipped tokens
            clipping_stats: Dictionary with clipping statistics for logging
        """
        with torch.no_grad():
            # Filter for model-generated tokens only (inf_log_probs != 0)
            # inf_log_probs is padded with 0.0 for padding tokens
            generated_token_mask = (inf_log_probs != 0.0)

            # Convert log probabilities to probabilities
            # p_train (old_log_probs) / p_infer (inf_log_probs) = exp(log_p_train - log_p_infer)
            log_ratio = old_log_probs - inf_log_probs
            prob_ratio = torch.exp(log_ratio)

            # Apply double-sided clipping: keep only ratios in [alpha, beta]
            # Mask out tokens where:
            # 1. prob_ratio < alpha (training prob << inference prob - huge divergence)
            # 2. prob_ratio > beta (training prob >> inference prob - overconfident)
            icepop_mask = ((prob_ratio >= alpha) & (prob_ratio <= beta)).float()

            # Compute clipping statistics only on model-generated tokens
            total_tokens = generated_token_mask.sum().item()
            clipped_tokens = ((icepop_mask == 0.0) & generated_token_mask).sum().item()
            valid_tokens = (icepop_mask == 1.0).sum().item()
            clipping_ratio = clipped_tokens / total_tokens if total_tokens > 0 else 0.0

            # Analyze clipped vs non-clipped tokens (only model-generated)
            clipped_lower = ((prob_ratio < alpha) & generated_token_mask).sum().item()  # Training prob too low
            clipped_upper = ((prob_ratio > beta) & generated_token_mask).sum().item()   # Training prob too high

            # Calculate mean probability ratios separately for <1 and >1 ratios
            # Only compute on model-generated tokens
            valid_mask = (icepop_mask == 1.0) & generated_token_mask
            clipped_mask = (icepop_mask == 0.0) & generated_token_mask

            # Separate underconfident (ratio < 1) and overconfident (ratio > 1) tokens
            underconfident_mask = (prob_ratio < 1.0) & generated_token_mask
            overconfident_mask = (prob_ratio >= 1.0) & generated_token_mask

            # Valid ratios split
            valid_underconfident = prob_ratio[(prob_ratio < 1.0) & valid_mask]
            valid_overconfident = prob_ratio[(prob_ratio >= 1.0) & valid_mask]

            # Clipped ratios split
            clipped_underconfident = prob_ratio[(prob_ratio < 1.0) & clipped_mask]
            clipped_overconfident = prob_ratio[(prob_ratio >= 1.0) & clipped_mask]

            # Overall ratios split
            all_underconfident = prob_ratio[underconfident_mask]
            all_overconfident = prob_ratio[overconfident_mask]

            clipping_stats = {
                'icepop/clipped_ratio': clipping_ratio,
                'icepop/clipped_lower': clipped_lower,
                'icepop/clipped_upper': clipped_upper,
                'icepop/clipped_tokens': clipped_tokens,
                'icepop/total_tokens': total_tokens,
                'icepop/valid_ratio_underconfident_mean': valid_underconfident.mean().item() if len(valid_underconfident) > 0 else 0.0,
                'icepop/valid_ratio_overconfident_mean': valid_overconfident.mean().item() if len(valid_overconfident) > 0 else 0.0,
                'icepop/clipped_ratio_underconfident_mean': clipped_underconfident.mean().item() if len(clipped_underconfident) > 0 else 0.0,
                'icepop/clipped_ratio_overconfident_mean': clipped_overconfident.mean().item() if len(clipped_overconfident) > 0 else 0.0,
                'icepop/prob_ratio_underconfident_mean': all_underconfident.mean().item() if len(all_underconfident) > 0 else 0.0,
                'icepop/prob_ratio_overconfident_mean': all_overconfident.mean().item() if len(all_overconfident) > 0 else 0.0,
            }

        return icepop_mask, clipping_stats
        
    def _compute_tis_token_ratio(self, inf_log_probs, old_log_probs, response_mask, clip_threshold=1.2):
        """
        Compute Truncated Importance Sampling (TIS) ratio at TOKEN level for vanilla DAPO.
        TIS corrects the mismatch between sampler (vLLM/inference) and learner (FSDP/training)
        by computing per-token importance weights with upper-bound truncation.
        Mathematical formulation:
            ρ_t = π_sampler(a_t, θ_old) / π_learner(a_t, θ_old)
            ρ̃_t = min(ρ_t, C)  (truncated at upper bound only)
        This will be applied as a multiplicative factor to vanilla PPO's objective:
            J_PPO+TIS = E[ρ̃_t * min(r_t*A_t, clip(r_t)*A_t)]
        Key differences from sequence-level TIS (for GSPO):
            - Operates at TOKEN level (one ratio per token, not per sequence)
            - NO length-normalization (applied directly to each token)
            - Applied outside PPO's min() operation, not inside
        Paper references:
            - TIS: https://fengyao.notion.site/off-policy-rl (TIS blog post)
            - PPO: https://arxiv.org/abs/1707.06347 (PPO paper)
        Args:
            inf_log_probs: Log probs from inference engine (vLLM/sampler)
                           Shape: [batch_size, response_length]
            old_log_probs: Log probs from training engine (FSDP/learner old policy)
                           Shape: [batch_size, response_length]
            response_mask: Mask for valid tokens (1=valid, 0=padding)
                           Shape: [batch_size, response_length]
            clip_threshold: C parameter - truncate token ratios above this
                            Default: 1.2 (more conservative than GSPO's 2.0-8.0)
        Returns:
            tis_token_ratio: Token-level TIS ratios (with stop-grad)
                             Shape: [batch_size, response_length]
            tis_stats: Dictionary with TIS statistics for logging
        """
        with torch.no_grad():
            # 1. Token-level log ratio: log(π_sampler / π_learner)
            #    inf_log_probs = sampler (vLLM inference backend)
            #    old_log_probs = learner (FSDP training backend old policy)
            log_ratio = inf_log_probs - old_log_probs  # [batch_size, response_length]

            # 2. Convert to probability ratio
            token_prob_ratio = torch.exp(log_ratio)  # [batch_size, response_length]

            # 3. TIS: Truncate upper bound ONLY
            #    - Keep ratios < 1 unchanged (unbiased for low-prob rollouts)
            #    - Clip ratios > C to prevent gradient explosion
            #    This is critical: unlike IcePop's double-sided clipping,
            #    TIS only clips the upper bound
            tis_token_ratio = torch.clamp(token_prob_ratio, max=clip_threshold)  # [batch_size, response_length]

            # 4. Apply response mask (set padding tokens to 0)
            tis_token_ratio = tis_token_ratio * response_mask

            # 5. Compute statistics for monitoring (only on valid tokens)
            valid_tokens = response_mask.sum().item()
            clipped_tokens = ((token_prob_ratio > clip_threshold) * response_mask).sum().item()
            clipping_freq = clipped_tokens / valid_tokens if valid_tokens > 0 else 0.0

            # Mask-aware statistics
            from verl.utils.torch_functional import masked_mean

            # Ratio distribution
            ratio_mean = masked_mean(token_prob_ratio, response_mask).item()
            ratio_std_val = torch.sqrt(masked_mean((token_prob_ratio - ratio_mean) ** 2, response_mask)).item() if valid_tokens > 0 else 0.0
            ratio_min = token_prob_ratio[response_mask.bool()].min().item() if valid_tokens > 0 else 0.0
            ratio_max = token_prob_ratio[response_mask.bool()].max().item() if valid_tokens > 0 else 0.0
            truncated_ratio_mean = masked_mean(tis_token_ratio, response_mask).item()

            # Direction of mismatch (only count valid tokens)
            ratio_lt_1 = ((token_prob_ratio < 1.0) * response_mask).sum().item()  # Sampler > Learner
            ratio_gt_1 = ((token_prob_ratio >= 1.0) * response_mask).sum().item()  # Sampler <= Learner
            ratio_gt_C = clipped_tokens  # Extreme mismatch (clipped)

            # Track extreme mismatches
            ratio_lt_0_5 = ((token_prob_ratio < 0.5) * response_mask).sum().item()  # Severe under-estimation
            ratio_gt_1_5 = ((token_prob_ratio > 1.5) * response_mask).sum().item()  # Severe over-estimation

            # Log-space statistics
            masked_log_ratio = log_ratio * response_mask
            log_ratio_mean = masked_log_ratio.sum().item() / valid_tokens if valid_tokens > 0 else 0.0
            log_ratio_std = torch.sqrt(masked_mean((log_ratio - log_ratio_mean) ** 2, response_mask)).item() if valid_tokens > 0 else 0.0
            log_ratio_min = log_ratio[response_mask.bool()].min().item() if valid_tokens > 0 else 0.0
            log_ratio_max = log_ratio[response_mask.bool()].max().item() if valid_tokens > 0 else 0.0

            tis_stats = {
                # Clipping statistics
                'tis_dapo/clipping_freq': clipping_freq,
                'tis_dapo/clipped_tokens': clipped_tokens,
                'tis_dapo/total_tokens': valid_tokens,

                # Ratio distribution
                'tis_dapo/ratio_mean': ratio_mean,
                'tis_dapo/ratio_std': ratio_std_val,
                'tis_dapo/ratio_min': ratio_min,
                'tis_dapo/ratio_max': ratio_max,
                'tis_dapo/truncated_ratio_mean': truncated_ratio_mean,

                # Direction of mismatch
                'tis_dapo/ratio_lt_1_count': ratio_lt_1,
                'tis_dapo/ratio_gt_1_count': ratio_gt_1,
                'tis_dapo/ratio_gt_C_count': ratio_gt_C,

                # Extreme cases
                'tis_dapo/ratio_lt_0.5_count': ratio_lt_0_5,
                'tis_dapo/ratio_gt_1.5_count': ratio_gt_1_5,

                # Log-space statistics (useful for debugging numerical issues)
                'tis_dapo/log_ratio_mean': log_ratio_mean,
                'tis_dapo/log_ratio_std': log_ratio_std,
                'tis_dapo/log_ratio_min': log_ratio_min,
                'tis_dapo/log_ratio_max': log_ratio_max,
            }

        return tis_token_ratio, tis_stats

    @GPUMemoryLogger(role="megatron actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            DataProto: torch.Tensor: the log_prob tensor
        """
        use_dynamic_bsz = data.meta_info.get("use_dynamic_bsz", False)
        micro_batch_size = data.meta_info.get("micro_batch_size", None)
        max_token_len = data.meta_info.get("max_token_len", None)
        if use_dynamic_bsz:
            assert max_token_len is not None, "max_token_len must be set when use_dynamic_bsz is True"
            max_token_len = max_token_len * self.config.megatron.context_parallel_size
        else:
            assert micro_batch_size is not None, (
                "micro batch size is needed for forward compute when use_dynamic_bsz is False"
            )

        # We make recompute_old_log_prob by default here.
        # TODO (zhangchi.usc1992): actually, this function should only return log_prob and this logic should be
        # handled by user outside
        entropys = torch.Tensor()

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        batch = data.select(batch_keys=select_keys).batch
        input_ids = batch["input_ids"]
        batch_size = input_ids.size(0)
        response = batch["responses"]
        response_length = response.size(1)
        with torch.no_grad():
            output = self.forward_backward_batch(
                data,
                forward_only=True,
                calculate_entropy=calculate_entropy,
                use_dynamic_bsz=use_dynamic_bsz,
                micro_batch_size=micro_batch_size,
                max_token_len=max_token_len,
            )
            if mpu.is_pipeline_last_stage(ignore_virtual=True):
                # only on last rank. It should be on every tp rank
                log_probs = [o["log_probs"] for o in output["output"]]  # (bs, seq_size)
                log_probs = torch.cat(log_probs, dim=0).to(torch.float32)

                if calculate_entropy:
                    entropys = torch.cat([o["entropy"] for o in output["output"]], dim=0)
                    entropys = entropys.to(torch.float32)

                if use_dynamic_bsz:
                    indices = output["indices"]
                    indices = list(itertools.chain.from_iterable(indices))
                    assert len(indices) == log_probs.size(0), f"{len(indices)} vs. {log_probs.size()}"
                    revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
                    log_probs = log_probs[revert_indices]
                    if calculate_entropy:
                        assert len(indices) == entropys.size(0), f"{len(indices)} vs. {entropys.size()}"
                        entropys = entropys[revert_indices]
            else:
                # other pp ranks
                log_probs = torch.empty(
                    size=(batch_size, response_length), dtype=torch.float32, device=input_ids.device
                )
                if calculate_entropy:
                    entropys = torch.empty(
                        size=(batch_size, response_length), dtype=torch.float32, device=input_ids.device
                    )

            log_probs = log_probs.to(get_device_id())
            # broadcast across pp ranks
            torch.distributed.broadcast(
                tensor=log_probs,
                src=mpu.get_pipeline_model_parallel_last_rank(),
                group=mpu.get_pipeline_model_parallel_group(),
                async_op=False,
            )
            log_probs = log_probs.to("cpu")

            if calculate_entropy:
                entropys = entropys.to(get_device_id())
                torch.distributed.broadcast(
                    tensor=entropys,
                    src=mpu.get_pipeline_model_parallel_last_rank(),
                    group=mpu.get_pipeline_model_parallel_group(),
                    async_op=False,
                )
                entropys = entropys.to("cpu")

        # add empty cache after each compute
        get_torch_device().empty_cache()

        return log_probs, entropys

    def make_minibatch_iterator(self, data: DataProto) -> Iterable[DataProto]:
        """Make minibatch iterator for updating the actor

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64, where
                ``sequence_length = prompt_length + response_length``

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64

                ``responses``: tensor of shape [batch_size, response_length]. torch.int64. Note that
                responses = input_ids[:, -response_length:]

                ``old_log_probs``: tensor of shape [batch_size, response_length]. torch.float32. The log probability
                of responses.

                ``advantages``: tensor of shape [batch_size, response_length]. torch.float32. The advantages of
                responses.
                See PPO paper for details. https://arxiv.org/abs/1707.06347

        Returns:

        """
        select_keys = [
            "responses",
            "input_ids",
            "attention_mask",
            "response_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
        ]
        if "traj_mask" in data.batch:
            print("[TrainingUpdate] current using traj mask !!!")
            select_keys.append("traj_mask")
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        self.has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        if self.has_multi_modal_inputs:
            data = data.select(select_keys, ["multi_modal_inputs"])
        else:
            data = data.select(batch_keys=select_keys)
        return data.make_iterator(
            mini_batch_size=self.config.ppo_mini_batch_size,
            epochs=self.config.ppo_epochs,
            seed=self.config.data_loader_seed,
            dataloader_kwargs={"shuffle": self.config.shuffle},
        )

    def compute_ppo_loss(self, model_output, data):
        log_prob = model_output["log_probs"]
        entropy = model_output.get("entropy", None)
        entropy_value = model_output.get("entropy_value", None)
        model_inputs = data

        metrics = {}
        
        if "traj_mask" in data:
            print("[TrainingUpdate] current using traj mask in ppo loss !!!")
            response_mask = data["traj_mask"].to(bool)
        else:
            response_mask = data["response_mask"].to(bool)
        # compute policy loss
        old_log_prob = data["old_log_probs"]
        advantages = data["advantages"]
        
        # Mismatch correction configuration
        # IcePop: Token-level double-sided masking
        USE_ICEPOP = self.enable_icepop  # Controlled by config
        ICEPOP_ALPHA = self.config.get("icepop_alpha", 0.5)  # Lower bound (symmetric 2x tolerance)
        ICEPOP_BETA = self.config.get("icepop_beta", 2.0)   # Upper bound (symmetric 2x tolerance)

        # TIS: Sequence-level importance weighting
        USE_TIS = self.enable_tis  # Controlled by config
        TIS_CLIP_THRESHOLD = self.config.get("tis_clip_threshold", 1.2)  # C parameter for TIS

        icepop_stats = {}
        tis_stats = {}


        if 'inf_log_probs' in model_inputs:
            inf_log_probs = model_inputs['inf_log_probs']
            inf_log_probs_list.append(inf_log_probs.to("cpu").detach())
            # Apply IcePop (token-level masking)
            if USE_ICEPOP:
                icepop_mask, icepop_stats = self._compute_icepop_mask(
                    inf_log_probs=inf_log_probs,
                    old_log_probs=old_log_prob,
                    alpha=ICEPOP_ALPHA,
                    beta=ICEPOP_BETA
                )

                # Combine IcePop mask with response_mask (element-wise AND)
                response_mask = response_mask * icepop_mask

                if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
                    print(f"[IcePop] Step {self.current_global_step} - "
                          f"α={ICEPOP_ALPHA}, β={ICEPOP_BETA}, "
                          f"clipping_ratio={icepop_stats['icepop/clipped_ratio']:.4f} "
                          f"({icepop_stats['icepop/clipped_tokens']}/{icepop_stats['icepop/total_tokens']} tokens), "
                          f"lower={icepop_stats['icepop/clipped_lower']}, "
                          f"upper={icepop_stats['icepop/clipped_upper']}")
            else:
                # No IcePop, create a dummy all-ones mask
                icepop_mask = None
                
        # Compute TIS based on loss_mode
        tis_seq_ratio = None   # For GSPO
        tis_token_ratio = None  # For vanilla DAPO

        if USE_TIS and 'inf_log_probs' in model_inputs:
            if "vanilla" in loss_mode:
                # Token-level TIS for vanilla DAPO
                tis_token_ratio, tis_stats = self._compute_tis_token_ratio(
                    inf_log_probs=inf_log_probs,
                    old_log_probs=old_log_prob,
                    response_mask=response_mask,
                    clip_threshold=TIS_CLIP_THRESHOLD
                )

                if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
                    print(f"[TIS-DAPO] Step {self.current_global_step} - "
                          f"C={TIS_CLIP_THRESHOLD}, "
                          f"clipping_freq={tis_stats['tis_dapo/clipping_freq']:.4f} "
                          f"({tis_stats['tis_dapo/clipped_tokens']}/{tis_stats['tis_dapo/total_tokens']} tokens), "
                          f"ratio_mean={tis_stats['tis_dapo/ratio_mean']:.4f}±{tis_stats['tis_dapo/ratio_std']:.4f}")


        loss_agg_mode = self.config.loss_agg_mode

        loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")

        policy_loss_fn = get_policy_loss_fn(loss_mode)
        pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = policy_loss_fn(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=advantages,
            response_mask=response_mask,
            loss_agg_mode=loss_agg_mode,
            config=self.config,
            tis_token_ratio=tis_token_ratio
        )

        metrics.update(
            {
                "actor/pg_loss": pg_loss.detach().item(),
                "actor/pg_clipfrac": pg_clipfrac.detach().item(),
                "actor/ppo_kl": ppo_kl.detach().item(),
                "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
            }
        )
        policy_loss = pg_loss

        # add entropy loss
        if entropy is not None:
            entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
            entropy_coeff = self.config.entropy_coeff
            policy_loss -= entropy_coeff * entropy_loss
        entropy_value_loss = None
        if entropy_value is not None:
            with torch.no_grad():
                entropy_value_loss = agg_loss(loss_mat=entropy_value, loss_mask=response_mask, loss_agg_mode='token-mean')

        # add kl loss
        if self.config.use_kl_loss:
            ref_log_prob = data["ref_log_prob"]
            # compute kl loss
            kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type)
            kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=self.config.loss_agg_mode)

            policy_loss += kl_loss * self.config.kl_loss_coef
            metrics["actor/kl_loss"] = kl_loss.detach().item()
            metrics["actor/kl_coef"] = self.config.kl_loss_coef
        metrics["actor/entropy_token_mean_loss"] = entropy_value_loss.detach().item(),

        return policy_loss, metrics

    def forward_backward_batch(
        self,
        data: DataProto,
        forward_only=False,
        calculate_entropy=False,
        use_dynamic_bsz=False,
        micro_batch_size=None,
        max_token_len=None,
    ):
        """
        We assume:
        - The model takes input: (input_ids, attention_mask, position_ids). No rmpad for the input
        - The communication shape is (total_nnz_pad_to_sp // tp_size, 1, hidden_size) if sequence parallel is enabled
        """
        # broadcast from last pp rank to all other pp ranks
        # TODO: actually, we just need to control the sampling order.
        data.to(get_device_id())
        data.batch = data.batch.contiguous()
        mini_batch = data
        broadcast_dict_tensor(
            mini_batch.batch,
            src=mpu.get_pipeline_model_parallel_last_rank(),
            group=mpu.get_pipeline_model_parallel_group(),
        )
        mini_batch.to("cpu")
        # split into micro-batches
        mini_batch.batch["attention_mask"] = mini_batch.batch["attention_mask"].to(bool)
        self.has_multi_modal_inputs = "multi_modal_inputs" in mini_batch.non_tensor_batch.keys()
        if self.has_multi_modal_inputs:
            mini_batch.batch["multi_modal_inputs"] = mini_batch.non_tensor_batch["multi_modal_inputs"]
            mini_batch.batch["multi_modal_inputs_idx"] = torch.Tensor(
                list(range(len(mini_batch.non_tensor_batch["multi_modal_inputs"])))
            ).to(torch.int64)

        if mini_batch.batch["position_ids"].dim() == 3:  # qwen2vl mrope [bs, 3, seq_len]
            mini_batch.batch["position_ids"] = mini_batch.batch["position_ids"][
                :, 0
            ]  # mcore patch recompute qwen2vl's pos ids during forward

        indices = None
        temperature = data.meta_info["temperature"]
        if use_dynamic_bsz:
            assert max_token_len is not None, "max_token_len must be set when use_dynamic_bsz is True"
            vpp_size = mpu.get_virtual_pipeline_model_parallel_world_size()
            if vpp_size is not None and vpp_size > 1:
                microbatch_group_size_per_vp_stage = self.tf_config.microbatch_group_size_per_vp_stage
                micro_batches, indices = rearrange_micro_batches(
                    batch=mini_batch.batch,
                    num_batches_divided_by=microbatch_group_size_per_vp_stage,
                    max_token_len=max_token_len,
                )
                assert len(micro_batches) % self.tf_config.microbatch_group_size_per_vp_stage == 0, (
                    f"micro_batches {micro_batches} must be divisible by microbatch_group_size_per_vp_stage "
                    f"{microbatch_group_size_per_vp_stage} for megatron backend"
                )
            else:
                micro_batches, indices = rearrange_micro_batches(batch=mini_batch.batch, max_token_len=max_token_len)
        else:
            assert micro_batch_size is not None, (
                "micro_batch_size is needed to be passed in when not using dynamic batch size"
            )
            micro_batches = mini_batch.batch.split(micro_batch_size)
        # compute input shapes for pp stages
        n_micro_batch = len(micro_batches)

        forward_backward_func = get_forward_backward_func()

        def loss_func(output, data):
            # For memory efficiency
            # We move calculation of entropy to compute_log_probs, forward_only == True
            device = output["log_probs"].device

            responses = data["responses"]
            response_length = responses.size(1)

            log_prob = output["log_probs"][:, -response_length - 1 : -1].contiguous()
            model_output = {"log_probs": log_prob}
            entropy = output["entropy"][:, -response_length - 1 : -1].contiguous()
            if calculate_entropy:
                model_output["entropy"] = entropy
            model_output["entropy_value"] = entropy

            if forward_only:
                # for inference
                return torch.tensor(1.0, device=device), model_output

            # for training
            # note that this loss function can be swapped with other loss functions such as SFT
            policy_loss, metrics = self.compute_ppo_loss(model_output, data)

            # return loss and stats
            return policy_loss, metrics

        def forward_step(batch_iter, model):
            batch = next(batch_iter)
            batch = batch.to(get_device_id())
            batch = batch.contiguous()

            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"].to(bool)
            position_ids = batch["position_ids"]

            multi_modal_inputs = {}
            if "multi_modal_inputs" in batch:
                for key in batch["multi_modal_inputs"][0].keys():
                    idxs = batch["multi_modal_inputs_idx"]
                    mmi = batch["multi_modal_inputs"]
                    multi_modal_inputs[key] = torch.cat(
                        [mmi[idx].get(key) for idx in idxs if mmi[idx].get(key) is not None], dim=0
                    )
            responses = batch["responses"]
            response_length = responses.size(1)
            label = position_ids.clone()
            label[:, -response_length - 1 : -1] = responses
            label_mask = attention_mask.clone()
            label_mask[:, : -response_length - 1] = False
            label_mask[:, -1] = False

            from verl.models.mcore import get_mcore_forward_fn, get_mcore_forward_fused_fn

            if self.use_fused_kernels:
                forward_fn = get_mcore_forward_fused_fn(self.hf_config)
                # return dict of [logits, entropy]
                output = forward_fn(
                    model,
                    input_ids,
                    position_ids,
                    attention_mask,
                    sequence_parallel=self.tf_config.sequence_parallel,
                    multi_modal_inputs=multi_modal_inputs,
                    labels=label,
                    labels_mask=label_mask,
                    temperature=temperature,
                )
            else:
                forward_fn = get_mcore_forward_fn(self.hf_config)

                def logits_processor(logits, label, label_mask):
                    assert logits.shape[:2] == label.shape[:2]
                    assert label.shape == label_mask.shape
                    logits.div_(temperature)
                    ret = {}
                    if calculate_entropy:
                        logits_bak = logits.clone()
                        if torch.distributed.get_rank() == 0:
                            logger.warning_once(
                                "For memory-efficient computation, enable fused kernels via "
                                "`actor_rollout_ref.model.use_fused_kernels=True`. "
                                "The current `clone()` operation ensures correctness but increases memory usage."
                            )
                        entropy = vocab_parallel_entropy(logits)
                        ret["entropy"] = entropy
                    else:
                        logits_bak = logits
                    log_probs = vocab_parallel_log_probs_from_logits(logits_bak, label)
                    log_probs = log_probs.masked_fill(~label_mask, 0.0)
                    ret["log_probs"] = log_probs
                    return ret

                logits_processor_args = {"label": label, "label_mask": label_mask}
                output = forward_fn(
                    model,
                    input_ids,
                    attention_mask,
                    position_ids,
                    sequence_parallel=self.tf_config.sequence_parallel,
                    multi_modal_inputs=multi_modal_inputs,
                    logits_processor=logits_processor,
                    logits_processor_args=logits_processor_args,
                )

            return output, partial(loss_func, data=batch)

        # batch should be a list of batches inside micro-batches
        batch_generator = make_batch_generator(micro_batches, vpp_size=len(self.actor_module))

        # TODO: we may use the new schedule instead
        # for flash-attn: (seq_len, batch_size, hidden_size) = (mbs*seq_len, 1, hidden_size)
        losses_reduced = forward_backward_func(
            forward_step_func=forward_step,
            data_iterator=batch_generator,
            model=self.actor_module,
            num_microbatches=n_micro_batch,
            seq_length=1,  # the communication shape is obtained via p2p comm
            micro_batch_size=1,  # the communication shape is obtained via p2p comm
            forward_only=forward_only,
        )
        # loss_reduces contains the stats returned from loss_func

        if self.has_multi_modal_inputs:
            data.batch.pop("multi_modal_inputs")
            data.batch.pop("multi_modal_inputs_idx")
            data.non_tensor_batch.pop("multi_modal_inputs")

        losses_reduced = {"output": losses_reduced}
        if use_dynamic_bsz:
            losses_reduced["indices"] = indices
        return losses_reduced

    @GPUMemoryLogger(role="megatron actor", logger=logger)
    def update_policy(self, dataloader: Iterable[DataProto]) -> dict:
        """Update the policy with an iterator of DataProto

        Args:
            dataloader (Iterable[DataProto]): an iterator over the DataProto that returns by ``make_minibatch_iterator``
                The keys of each data batch is described in the make_minibatch_iterator.

        Returns:
            Dict: a dictionary containing the statistics. Note that the statistics are only valid in the last pp stage
            and users have to combine the output in each dp rank manually.

        """
        metrics = {}
        if self.use_torch_profiler and self.prof and self.prof.enable:
            self.prof.start()
        for data in dataloader:
            self.actor_optimizer.zero_grad()
            # use use_contiguous_buffers_in_local_ddp and no overlap_dp_param_comm
            for chunk in self.actor_module:
                # if use distributed optimizer, zero grad buffer will be handled by optimizer
                chunk.zero_grad_buffer()

            calculate_entropy = self.config.entropy_coeff != 0
            if data.meta_info.get("micro_batch_size", None) is not None:
                micro_batch_size = data.meta_info["micro_batch_size"]
            else:
                micro_batch_size = self.config.ppo_micro_batch_size_per_gpu
            max_token_len = None
            if self.config.use_dynamic_bsz:
                max_token_len = self.config.ppo_max_token_len_per_gpu * self.config.megatron.context_parallel_size
            metric_micro_batch = self.forward_backward_batch(
                data,
                calculate_entropy=calculate_entropy,
                use_dynamic_bsz=self.config.use_dynamic_bsz,
                micro_batch_size=micro_batch_size,
                max_token_len=max_token_len,
            )
            metric_micro_batch = metric_micro_batch["output"]
            for metric in metric_micro_batch:
                # Note that o[0] is metrics, o[1] is entropy, o[2] is response_mask
                append_to_dict(metrics, metric)  # append the metric from this micro-batch to global metrics.

            update_successful, grad_norm, num_zeros_in_grad = self.actor_optimizer.step()
            data = {"actor/grad_norm": grad_norm}
            append_to_dict(metrics, data)

            if update_successful:
                # allgather already execute in optimizer.step in new megatron
                pass
            else:
                raise NotImplementedError
            if self.use_torch_profiler and self.prof and self.prof.enable:
                self.prof.step()
        # add empty cache after each compute
        if self.use_torch_profiler and self.prof and self.prof.enable:
            self.prof.stop_and_save()
            self.prof.stop_trace()
        get_torch_device().empty_cache()
        return metrics
