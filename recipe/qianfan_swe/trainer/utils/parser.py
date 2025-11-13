"""
This module defines chat template parsers for formatting messages for different LLMs.

Classes:
    ChatTemplateParser: Base class for parsing chat messages using a tokenizer's chat template.
    LlamaChatTemplateParser: Parser for Llama-style chat templates with custom tokens.

Usage:
    Instantiate a parser with a tokenizer and use the `parse` method to convert a list of messages
    into a formatted string suitable for LLM input. The parser supports system, user, assistant,
    and tool message roles, and can optionally add a generation prompt.

    The `verify_equivalence` method checks that parsing a batch of messages is equivalent to
    parsing them individually and concatenating the results.

Example:
    parser = LlamaChatTemplateParser(tokenizer)
    formatted = parser.parse(messages, add_generation_prompt=True)
"""
import re

PARSER_TEST_MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Search for information about Python."},
    {"role": "assistant", "content": "I'll search for that.", "tool_calls": [{"function": {"name": "search", "arguments": '{"query": "Python programming"}'}}]},
    # {"role": "tool", "content": "Python is a high-level programming language."},
    {"role": "user", "content": "What about Java?"},
    {"role": "assistant", "content": "Let me search for Java information.", "tool_calls": [{"function": {"name": "search", "arguments": '{"query": "Java programming"}'}}]},
]


class ChatTemplateParser:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.assistant_token = ""

    def parse(self, messages, add_generation_prompt=False, is_first_msg=False, **kwargs) -> str:
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)

    def verify_equivalence(self, messages, verbose=True):
        """Verify that parsing messages together is equivalent to parsing them individually.

        Args:
            messages (list): List of message dictionaries to test
            verbose (bool): Whether to print detailed information about the test

        Returns:
            bool: True if the equivalence check passes, False otherwise

        Raises:
            AssertionError: If the equivalence check fails and verbose is True
        """
        # Parse all messages together
        batch_result = self.parse(messages)

        # Parse each message individually and concatenate
        individual_results = []
        for message in messages:
            individual_results.append(self.parse([message]))

        concatenated_result = "".join(individual_results)

        # Check if results are equivalent
        is_equivalent = batch_result == concatenated_result

        if verbose and not is_equivalent:
            print("Equivalence check failed!")
            print("Batch parsing result:")
            print(batch_result)
            print("\nConcatenated individual parsing result:")
            print(concatenated_result)
            raise AssertionError("Parser failed equivalence check. See above for details.")

        return is_equivalent

    @classmethod
    def get_parser(cls, tokenizer, disable_thinking=False, parser_class=None) -> "ChatTemplateParser":
        """Factory method to get the appropriate parser based on a string identifier.

        Args:
            tokenizer: The tokenizer to use with the parser
            disable_thinking: Whether generation prompt will disable thinking.
            parser_class (str, optional): Explicit parser class name to use.
                Supported values: "DeepseekV31TerminusSWEChatTemplateParser",
                                 "DeepseekQwenChatTemplateParser",
                                 "QwenChatTemplateParser",
                                 "LlamaChatTemplateParser",
                                 "ChatTemplateParser"
                If None, auto-detect based on model name.

        Returns:
            ChatTemplateParser: An instance of the requested parser

        Raises:
            ValueError: If the parser_class is not recognized
        """
        # If parser_class is explicitly specified, use it
        if parser_class:
            parser_class_lower = parser_class.lower()

            if parser_class_lower == "deepseekv31terminusswechattemplateparser":
                print(f"Using explicitly specified DeepseekV31TerminusSWEChatTemplateParser")
                return DeepseekV31TerminusSWEChatTemplateParser(tokenizer, disable_thinking=disable_thinking)
            elif parser_class_lower == "deepseekqwenchattemplateparser":
                print(f"Using explicitly specified DeepseekQwenChatTemplateParser")
                model_name = tokenizer.name_or_path.lower() if isinstance(tokenizer.name_or_path, str) else ""
                return DeepseekQwenChatTemplateParser(tokenizer, disable_thinking=disable_thinking, model_name=model_name)
            elif parser_class_lower == "qwenchattemplateparser":
                print(f"Using explicitly specified QwenChatTemplateParser")
                return QwenChatTemplateParser(tokenizer, disable_thinking=disable_thinking)
            elif parser_class_lower == "llamachattemplateparser":
                print(f"Using explicitly specified LlamaChatTemplateParser")
                return LlamaChatTemplateParser(tokenizer)
            elif parser_class_lower == "chattemplateparser":
                print(f"Using explicitly specified ChatTemplateParser")
                return ChatTemplateParser(tokenizer)
            else:
                raise ValueError(f"Unknown parser_class: {parser_class}. Supported values: "
                               "DeepseekV31TerminusSWEChatTemplateParser, DeepseekQwenChatTemplateParser, "
                               "QwenChatTemplateParser, LlamaChatTemplateParser, ChatTemplateParser")

        # Auto-detect based on tokenizer name or path
        if isinstance(tokenizer.name_or_path, str):
            model_name = tokenizer.name_or_path.lower()
            tokenizer_cls = tokenizer.__class__.__name__.lower()
            print(f"model_name: {model_name}, tokenizer_cls: {tokenizer_cls}")
            if any(x in model_name for x in ("deepseek", "deepscaler", "deepcoder")) and "llama" in tokenizer_cls:
                print(f"Using DeepseekQwenChatTemplateParser for {tokenizer.name_or_path}")
                return DeepseekQwenChatTemplateParser(tokenizer, disable_thinking=disable_thinking, model_name=model_name)
            elif "qwen" in model_name or "r2e" in model_name or "deepswe" in model_name or "qwen" in tokenizer_cls:
                print(f"Using QwenChatTemplateParser for {tokenizer.name_or_path}")
                return QwenChatTemplateParser(tokenizer, disable_thinking=disable_thinking)
            elif "llama" in model_name:
                print(f"Using LlamaChatTemplateParser for {tokenizer.name_or_path}")
                return LlamaChatTemplateParser(tokenizer)

        # Default to the standard parser if no specific match
        parser = ChatTemplateParser(tokenizer)
        print(f"No custom parser found. Using default ChatTemplateParser for {tokenizer.name_or_path}")
        assert parser.verify_equivalence(PARSER_TEST_MESSAGES), "Parser failed equivalence check"
        return parser


class DeepseekQwenChatTemplateParser(ChatTemplateParser):
    def __init__(self, tokenizer, disable_thinking=True, model_name=""):
        super().__init__(tokenizer)
        self.bos_token = tokenizer.bos_token
        self.eos_token = tokenizer.eos_token
        self.system_token = ""
        self.user_token = "<｜User｜>"
        self.assistant_token = "<｜Assistant｜>"
        self.generation_prompt = self.eos_token + self.assistant_token
        if disable_thinking:
            if "v3.1" in model_name.lower():
                self.generation_prompt += "</think>"
        else:
            self.generation_prompt += "<think>"

    def parse(self, messages, add_generation_prompt=False, is_first_msg=False, **kwargs) -> str:
        result = ""

        if is_first_msg:
            result += self.bos_token
            generation_prompt = self.generation_prompt

        for message in messages:
            if message["role"] == "system":
                result += self.parse_system(message)
            elif message["role"] == "user":
                result += self.parse_user(message)
            elif message["role"] == "assistant":
                result += self.parse_assistant(message)
            else:
                raise NotImplementedError(f"Unsupported message role: {message['role']}")

        if add_generation_prompt:
            result += self.generation_prompt
        return result

    def parse_system(self, message):
        return self.system_token + message["content"]

    def parse_user(self, message):
        return self.user_token + message["content"]

    def parse_assistant(self, message):
        return self.assistant_token + message["content"] + self.eos_token


class DeepseekV31TerminusSWEChatTemplateParser(ChatTemplateParser):
    """Parser for DeepSeek-V3.1-Terminus model following the official chat template.

    This parser implements the chat template from tokenizer_config.json which includes:
    - System messages collected at the beginning
    - User/Assistant conversation flow
    - Tool calls with special markers
    - Tool outputs handling
    - Thinking tag support
    """
    def __init__(self, tokenizer, disable_thinking=True):
        super().__init__(tokenizer)
        self.bos_token = "<｜begin▁of▁sentence｜>"
        self.eos_token = "<｜end▁of▁sentence｜>"
        self.user_token = "<｜User｜>"
        self.assistant_token = "<｜Assistant｜>"

        # Tool-related tokens
        self.tool_calls_begin = "<｜tool▁calls▁begin｜>"
        self.tool_calls_end = "<｜tool▁calls▁end｜>"
        self.tool_call_begin = "<｜tool▁call▁begin｜>"
        self.tool_call_end = "<｜tool▁call▁end｜>"
        self.tool_sep = "<｜tool▁sep｜>"
        self.tool_output_begin = "<｜tool▁output▁begin｜>"
        self.tool_output_end = "<｜tool▁output▁end｜>"

        # Thinking tag
        self.disable_thinking = disable_thinking
        if disable_thinking:
            self.generation_prompt = self.assistant_token + "</think>"
        else:
            self.generation_prompt = self.assistant_token + "<think>"

    def _parse_tool_calls_from_content(self, content):
        """
        Parse tool calls from content using regex.

        Returns:
            tuple: (has_tools, text_before_tools)
                - has_tools: bool indicating if valid tool calls were found and parsed
                - text_before_tools: text content before tool calls (or entire content if no tools)
        """
        # Pattern to match the complete tool calls section
        tool_calls_pattern = re.escape(self.tool_calls_begin) + r'(.*?)' + re.escape(self.tool_calls_end)
        match = re.search(tool_calls_pattern, content, re.DOTALL)

        if not match:
            return False, content

        # Extract text before tool calls
        tool_calls_start = match.start()
        text_before = content[:tool_calls_start].strip()

        # Extract the tool calls section
        tool_section = match.group(1)

        # Verify we have at least one valid tool call inside
        # Pattern: <｜tool▁call▁begin｜>function_name<｜tool▁sep｜>arguments<｜tool▁call▁end｜>
        tool_call_pattern = re.escape(self.tool_call_begin) + r'(.*?)' + re.escape(self.tool_sep) + r'(.*?)' + re.escape(self.tool_call_end)
        tool_calls = re.findall(tool_call_pattern, tool_section, re.DOTALL)

        if tool_calls:
            # Successfully found valid tool calls
            return True, text_before if text_before else None

        return False, content

    def parse(self, messages, add_generation_prompt=False, is_first_msg=False, **kwargs) -> str:
        result = ""

        # Collect all system messages at the beginning
        system_prompt = ""
        is_first_sp = True
        for message in messages:
            if message["role"] == "system":
                if is_first_sp:
                    system_prompt = message["content"]
                    is_first_sp = False
                else:
                    system_prompt += "\n\n" + message["content"]

        # Add BOS token and system prompt if this is the first message
        if is_first_msg:
            result += self.bos_token + system_prompt

        # Track state
        is_last_user = False
        is_tool = False
        expect_tool_response = False  # Track if previous assistant had tool calls

        # Process messages
        for message in messages:
            role = message["role"]

            if role == "system":
                # System messages already handled
                continue

            elif role == "user":
                # If previous assistant had tool calls, treat this user message as tool response
                if expect_tool_response:
                    is_last_user = False
                    is_tool = True
                    result += self.tool_output_begin + message["content"] + self.tool_output_end
                    expect_tool_response = False
                else:
                    is_tool = False
                    is_last_user = True
                    result += self.user_token + message["content"]

            elif role == "assistant":
                content = message.get("content", "")

                # First check if tool_calls field exists (parsed structure)
                tool_calls = message.get("tool_calls")

                # If no tool_calls field, try to parse from content
                has_tools_in_content = False
                text_before_tools = None
                if not tool_calls:
                    has_tools_in_content, text_before_tools = self._parse_tool_calls_from_content(content)

                if tool_calls:
                    # Assistant message with parsed tool_calls structure
                    if is_last_user:
                        result += self.assistant_token + "</think>"
                    is_last_user = False
                    is_tool = False

                    # Add content if present, then tool calls
                    if content:
                        result += content

                    result += self.tool_calls_begin
                    for tool in tool_calls:
                        func_name = tool["function"]["name"]
                        func_args = tool["function"]["arguments"]
                        result += self.tool_call_begin + func_name + self.tool_sep + func_args + self.tool_call_end
                    result += self.tool_calls_end + self.eos_token

                    # Mark that we expect a tool response next
                    expect_tool_response = True

                elif has_tools_in_content:
                    # Successfully parsed tool calls from content
                    if is_last_user:
                        result += self.assistant_token + "</think>"
                    is_last_user = False
                    is_tool = False

                    # Content already contains the tool calls, just append it
                    result += content

                    # Add eos_token if not already present at the end
                    if not content.endswith(self.eos_token):
                        result += self.eos_token

                    # Mark that we expect a tool response next
                    expect_tool_response = True

                else:
                    # Regular assistant message without tool calls
                    if is_last_user:
                        result += self.assistant_token
                        # Handle thinking tag based on prefix or disable_thinking
                        if message.get("prefix") and not self.disable_thinking:
                            result += "<think>"
                        else:
                            result += "</think>"
                    is_last_user = False

                    # Remove </think> tag if present in content
                    if "</think>" in content:
                        content = content.split("</think>", 1)[1]

                    result += content + self.eos_token
                    is_tool = False
                    expect_tool_response = False

            elif role == "tool":
                is_last_user = False
                is_tool = True
                result += self.tool_output_begin + message["content"] + self.tool_output_end
                expect_tool_response = False

            else:
                raise NotImplementedError(f"Unsupported message role: {role}")

        # Add generation prompt if requested
        if add_generation_prompt and is_last_user and not is_tool:
            result += self.generation_prompt

        return result


class QwenChatTemplateParser(ChatTemplateParser):
    def __init__(self, tokenizer, disable_thinking=True):
        super().__init__(tokenizer)
        self.bos_token = tokenizer.bos_token
        self.eos_token = tokenizer.eos_token
        self.eot_token = "<|im_end|>\n"
        self.system_token = "<|im_start|>system\n"
        self.user_token = "<|im_start|>user\n"
        self.assistant_token = "<|im_start|>assistant\n"
        if disable_thinking:
            self.assistant_token += "<think>\\n\\n</think>\\n\\n"
        self.generation_prompt = self.assistant_token

        self.tool_start_token = "\n<tool_call>\n"
        self.tool_end_token = "\n</tool_call>"

        self.tool_response_start_token = "<tool_response>\n"
        self.tool_response_end_token = "\n</tool_response>"

    def parse(self, messages, add_generation_prompt=False, is_first_msg=False, **kwargs) -> str:
        result = ""

        # if the first message is not a system message, add the system message
        if is_first_msg and messages[0]["role"] != "system":
            result += self.system_token + "You are Qwen, created by Alibaba Cloud. You are a helpful assistant." + self.eot_token

        for message in messages:
            if message["role"] == "system":
                result += self.parse_system(message)
            elif message["role"] == "user":
                result += self.parse_user(message)
            elif message["role"] == "assistant":
                result += self.parse_assistant(message)
            elif message["role"] == "tool":
                result += self.parse_tool(message)
            else:
                raise NotImplementedError(f"Unsupported message role: {message['role']}")

        if add_generation_prompt:
            result += self.generation_prompt
        return result

    def parse_system(self, message):
        return self.system_token + message["content"] + self.eot_token

    def parse_user(self, message):
        return self.user_token + message["content"] + self.eot_token

    def parse_assistant(self, message):
        result = self.assistant_token + message["content"] + self.eot_token
        return result

    def parse_tool(self, message):
        return self.user_token + self.tool_response_start_token + message["content"] + self.tool_response_end_token + self.eot_token


class LlamaChatTemplateParser(ChatTemplateParser):
    def __init__(self, tokenizer):
        super().__init__(tokenizer)
        self.bos_token = "<|begin_of_text|>"
        self.system_token = "<|start_header_id|>system<|end_header_id|>\n\n"
        self.user_token = "<|start_header_id|>user<|end_header_id|>\n\n"
        self.assistant_token = "<|start_header_id|>assistant<|end_header_id|>\n\n"
        self.eot_token = "<|eot_id|>"
        self.generation_prompt = self.assistant_token

        # took tokens
        self.tool_start_token = "<|start_header_id|>tool<|end_header_id|>\n\n"
        self.tool_end_token = "<|eot_id|>"
        self.tool_response_start_token = "<|start_header_id|>tool_response<|end_header_id|>\n\n"
        self.tool_response_end_token = "<|eot_id|>"

    def parse(self, messages, add_generation_prompt=False, is_first_msg=False, **kwargs) -> str:
        result = ""

        if is_first_msg:
            result += self.bos_token

        for message in messages:
            if message["role"] == "system":
                result += self.parse_system(message)
            elif message["role"] == "user":
                result += self.parse_user(message)
            elif message["role"] == "assistant":
                result += self.parse_assistant(message)
            elif message["role"] == "tool":
                result += self.parse_tool(message)
            else:
                raise NotImplementedError(f"Unsupported message role: {message['role']}")

        if add_generation_prompt:
            result += self.generation_prompt
        return result

    def parse_system(self, message):
        return self.system_token + message["content"] + self.eot_token

    def parse_user(self, message):
        return self.user_token + message["content"] + self.eot_token

    def parse_assistant(self, message):
        return self.assistant_token + message["content"] + self.eot_token

    def parse_tool(self, message):
        return self.user_token + self.tool_response_start_token + message["content"] + self.tool_response_end_token + self.eot_token