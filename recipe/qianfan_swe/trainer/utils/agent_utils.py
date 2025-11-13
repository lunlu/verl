"""
Agent Utilities for SWE Training Framework.

This module provides utility functions for creating and managing software engineering
agents in a distributed training environment. It handles agent initialization,
configuration loading, and integration with pod management systems.

Key Components:
- Agent creation and configuration management
- YAML configuration file loading and parsing
- Pod manager integration for containerized execution
- Tool integration and system prompt generation
- Environment setup and resource management

The utilities support both R2E (Repository to Environment) and SWE-bench agent
configurations, providing a unified interface for different agent types and
execution environments.

Functions:
- load_config_from_yaml: Load and parse YAML configuration files
- _create_agent: Create and initialize agent instances with pod management

Features:
- Flexible configuration management through YAML files
- Integration with Kubernetes pod management
- Support for custom tool descriptions and system prompts
- Comprehensive error handling and logging
- Resource cleanup and lifecycle management

Usage:
    config = load_config_from_yaml("config.yaml")
    agent = _create_agent(0, env_args, config)
"""

import yaml
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import PodManager for pod management
from ..environment import PodManager

# Import R2E configurations
from workers.tools.r2e_configs import (
    parse_xml_action_custom,
    generate_custom_system_prompt
)
from workers.agents.swe_agent import SweAgent
from workers.agents.swe_agent_tools_icepop_messages import SweToolsIcepopMessagesAgent
from workers.core import create_tool

# Map agent class names to actual classes
AGENT_CLASSES = {
    "SweAgent": SweAgent,
    "SweToolsIcepopMessagesAgent": SweToolsIcepopMessagesAgent
}
MAX_RETRIES = 100


def load_config_from_yaml(config_path: str):
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Dictionary containing the configuration
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Failed to load configuration from {config_path}: {e}")
        raise

def _create_agent(i, env_args, config):
    """
    Create an agent instance with pod management and tool configuration.

    This function creates a software engineering agent with an associated Kubernetes pod
    for isolated execution. It handles pod creation, tool setup, and agent initialization
    with proper resource management and error handling.

    Args:
        i (int): Index of the agent to create (used for accessing env_args)
        env_args (list): List of environment argument dictionaries, where env_args[i]
                        contains configuration for this specific agent including:
                        - docker_image: Container image to use for the pod
                        - repo: Repository information
                        - Additional environment-specific settings
        config: Configuration object containing:
                - agent.working_dir: Working directory path (defaults to "/testbed")
                - agent.system_yaml: Optional path to YAML file with system prompt
                - agent.max_steps: Maximum number of steps for the agent
                - agent.termination_tool_names: List of tool names that can terminate execution
                - agent.kubeconfig_path: Path to Kubernetes configuration file
                - agent.namespace: Kubernetes namespace for pod creation
                - agent.rollout_agent: Agent class name to use (defaults to "SweAgent")
                                       Supported values: "SweAgent", "SweToolsAgent", etc.

    Returns:
        tuple: A tuple containing:
            - i (int): The original agent index
            - pod_name (str): Name of the created Kubernetes pod
            - image (str): Docker image used for the pod
            - agent: Configured agent instance ready for execution
            - pod_manager (PodManager): Pod manager instance for this agent

    Raises:
        Exception: If pod creation fails or agent initialization encounters errors
        ValueError: If the specified agent class is not supported

    Note:
        - Each agent gets its own independent PodManager and Kubernetes pod
        - The function supports both YAML-based system prompts and dynamic tool-based prompts
        - Tools are automatically configured with Kubernetes execution context
        - Pod names are generated with "swetrainer" prefix for identification
        - Agent class is determined by config.agent.rollout_agent (defaults to "SweAgent")

    Example:
        >>> config = load_config_from_yaml("config.yaml")
        >>> env_args = [{"docker_image": "ubuntu:20.04", "repo": "test-repo"}]
        >>> idx, pod_name, image, agent, pod_mgr = _create_agent(0, env_args, config)
    """
    _args = env_args[i]
    app_id = _args.get("app_id", None)
    requirement_type = _args.get("requirement_type", None)
    working_dir = _args.get("working_dir", "/testbed")
    sample_type = _args.get("sample_source", "r2e")
    print(f"[AgentUtils] current working_dir is {working_dir}, sample_type is {sample_type}")
    
    # Determine which agent class to use
    rollout_agent = config.agent.get("rollout_agent", "SweAgent")
    print(f"[AgentUtilsLogs] Using agent class: {rollout_agent}")

    if rollout_agent not in AGENT_CLASSES:
        raise ValueError(f"Unknown agent class: {rollout_agent}. Supported classes: {list(AGENT_CLASSES.keys())}")

    AgentFactory = AGENT_CLASSES[rollout_agent]
    
    # Initialize pod based on environment type using PodManager
    if "docker_image" in _args:
        docker_image = _args.get("docker_image", "")
    elif "docker" in _args:
        docker_image = _args.get("docker", "")
    else:
        raise Exception("[AgentUtilsLogs] please provide docker or docker_image in sample!")
    
    # Check if the environment is SWE-bench
    swebench_verified = "sweb" in docker_image
    #秒哒
    miaoda_task = "miaoda" in docker_image

    for attempt in range(MAX_RETRIES):
        try:
            # Create independent PodManager for this agent
            pod_manager = PodManager(config=config)

            # Create pod using PodManager
            pod_name, pod_info = pod_manager.create_pod(_args, pod_prefix="mdtrain")
            break
        except Exception as e:
            print(f"[WARNING] Agent {i} create failed (attempt {attempt+1}): {e}")
            time.sleep(1 + attempt * 2)
            if attempt == MAX_RETRIES - 1:
                raise RuntimeError(f"[PodManager] Agent {i} failed after retries, exception: {e}")
    
    image = pod_info["image"]

    # Generate custom system prompt
    if config.agent.get("system_yaml", ""):
        # Load configuration from YAML file
        config_path = config.agent.system_yaml
        config_yaml = load_config_from_yaml(config_path)
        custom_system_prompt = config_yaml.get("system_prompt", "")
        print("[TrainingLogs] Using system prompt from system_yaml!")

        # Create agent using the configured agent class
        agent = AgentFactory(
            max_rounds=config.agent.max_steps,
            debug=config.agent.get("debug", False),
            termination_tool_names=config.agent.termination_tool_names,
            action_parser=parse_xml_action_custom,
            profiler=None,
            system_prompt=custom_system_prompt,
            extra_info=_args,
            kubeconfig_path=config.agent.kubeconfig_path,
            namespace=config.agent.namespace,
            working_dir=working_dir
        )
    else:
        # Create k8s_config for tools
        k8s_config = {
            "execution_mode": "k8s",
            "pod_name": pod_name,
            "namespace": config.agent.namespace,
            "kubeconfig_path": config.agent.kubeconfig_path,
            "working_dir": working_dir  # Important: R2E tools need to know the working directory
        }

        base_tools = {
            "r2e_bash_executor": create_tool(k8s_config),
            "r2e_file_editor": create_tool(k8s_config),
            "r2e_search": create_tool(k8s_config),
            "r2e_submit": create_tool(k8s_config)
        }

        custom_system_prompt = generate_custom_system_prompt(
            base_tools,
            task_description="analyze and fix the reported issue in the repository",
            working_directory=working_dir,
            additional_instructions="\n- Focus on the specific issue described\n- Make minimal changes to fix the issue\n- Ensure your changes don't break existing functionality"
        )

        # Create agent using the configured agent class
        agent = AgentFactory(
            max_rounds=config.agent.max_steps,
            debug=config.agent.debug,
            termination_tool_names=config.agent.termination_tool_names,
            action_parser=parse_xml_action_custom,
            profiler=None,
            system_prompt=custom_system_prompt,
            extra_info=_args,
            kubeconfig_path=config.agent.kubeconfig_path,
            namespace=config.agent.namespace,
            working_dir=working_dir
        )

        agent.set_tools(base_tools)
    
    # Initialize pod based on environment type using PodManager
    pod_manager.initialize_pod(pod_name, sample_type=sample_type, requirement_type=requirement_type, app_id=app_id)

    # Return the agent, pod name, image, and pod manager
    return i, pod_name, image, agent, pod_manager
