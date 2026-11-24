# Copyright 2022 Adobe
# All Rights Reserved.

# NOTICE: Adobe permits you to use, modify, and distribute this file in
# accordance with the terms of the Adobe license agreement accompanying
# it.

import asyncio
from openai import AsyncAzureOpenAI, AsyncOpenAI, RateLimitError
import tiktoken

import time
import httpx 

from functools import cache

from transformers import AutoTokenizer
from typing import Union, List, Optional

# The Gemini (vertexai, google.genai) and AWS (langchain_aws) SDKs are imported where those providers are
# used: they took about 8 s to import on every run, including runs of other providers

from tenacity import retry, wait_random_exponential, stop_after_delay, retry_if_exception_type, retry_if_exception_message, wait_random

from request_cache import RequestCache, request_key

DEFAULT_TOKENIZER_MAPPING = {
    "gemini": "google",
    "vllm": "huggingface",
    "openai": "tiktoken",
    "azure-openai": "tiktoken",
    "aws": "huggingface"
}

OPENAI_COMPATIBLE_PROVIDERS = ("openai", "vllm", "azure-openai")
# Request record fields that select the endpoint rather than being arguments of the SDK call
ENDPOINT_FIELDS = ("api_provider", "api_url")


def build_messages(
    system_prompt: str,
    user_prompt: Union[str, List[str]],
    default_system_prompt: str,
    add_default_system_prompt: bool = True
) -> list:
    """
    Chat messages for a request: the default system prompt (from the model config) when
    requested and non-empty, else the given system prompt when non-empty, then the user prompt(s).
    """
    if add_default_system_prompt and default_system_prompt != "":
        messages = [{"role": "system", "content": default_system_prompt}]
    elif system_prompt != "":
        messages = [{"role": "system", "content": system_prompt}]
    else:
        messages = []
    for prompt in (user_prompt if isinstance(user_prompt, list) else [user_prompt]):
        messages.append({"role": "user", "content": prompt})
    return messages


def build_request(
    api_provider: str,
    api_url: str,
    model: str,
    model_config: dict,
    default_system_prompt: str,
    messages: list,
    max_tokens: int,
    temperature: float,
    top_p: float
) -> dict:
    """
    The request record: everything that decides the model's answer (provider, endpoint, model and
    the exact arguments of the SDK call). It is the request cache key, so it never holds API keys,
    timeouts, retry or tokenizer settings. Numbers are normalized so 0 and 0.0 give the same key.
    """
    request = {
        "api_provider": api_provider,
        "api_url": api_url.rstrip("/") if isinstance(api_url, str) else api_url,
        "model": model,
    }
    if api_provider in OPENAI_COMPATIBLE_PROVIDERS:
        request["messages"] = messages
        request["seed"] = 43
        if model_config.get("openai_thinking_model", False):
            request["max_completion_tokens"] = int(max_tokens)
        else:
            request["max_tokens"] = int(max_tokens)
            request["temperature"] = float(temperature)
            request["top_p"] = float(top_p)
        # OpenRouter provider routing (e.g. pin to OpenAI) and reasoning settings (e.g. thinking off). Both change
        # which backend answers or how, so they are part of the key.
        extra_body = {}
        if "openrouter_provider" in model_config:
            extra_body["provider"] = model_config["openrouter_provider"]
        if "openrouter_reasoning" in model_config:
            extra_body["reasoning"] = model_config["openrouter_reasoning"]
        if extra_body:
            request["extra_body"] = extra_body
    elif api_provider == "gemini":
        request["system_instruction"] = default_system_prompt
        request["contents"] = messages[-1]["content"]
        request["max_output_tokens"] = int(max_tokens)
        request["temperature"] = float(temperature)
        request["top_p"] = float(top_p)
        request["thinking_budget"] = model_config.get("thinking_budget")
        request["seed"] = 43
    elif api_provider == "aws":
        # Bedrock sampling is fixed when the client is created (from the model config), not per call
        request["messages"] = messages
        request["max_tokens"] = int(model_config["max_tokens"])
        request["temperature"] = float(model_config["temperature"])
        request["top_p"] = float(model_config["top_p"])
        request["region"] = model_config["region"]
    else:
        raise ValueError(f"Invalid API provider: {api_provider}")
    return request


class APIConnector:
    def __init__(
        self,
        api_key: str,
        api_url: str,
        api_provider: str,
        model: str,
        cache: Optional[RequestCache] = None,
        **kwargs
    ) -> None:
        """
        APIConnector class to unify the API calls for different API providers

        Parameters:
            api_key (`str`): API Key for the API
            api_url (`str`): API URL for the API
            api_provider (`str`): API Provider (openai, gemini, aws, vllm, azure-openai, etc.)
            model (`str`): Model name to use for the API
            cache (`RequestCache`, optional): Serve identical requests from this cache instead of the API
        """
        self.api_provider = api_provider
        self.api_url = api_url
        self.cache = cache

        if "tokenizer_type" in kwargs:
            self.tokenizer_type = kwargs["tokenizer_type"]
        else:
            if api_provider in DEFAULT_TOKENIZER_MAPPING:
                self.tokenizer_type = DEFAULT_TOKENIZER_MAPPING[api_provider]
            else:
                raise ValueError(f"Invalid API provider: {api_provider}")
            
        self.tokenizer_model = kwargs["tokenizer_model"] if "tokenizer_model" in kwargs else model
            
        if self.tokenizer_type == "huggingface":
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_model, use_fast=True)
            self._hf_fast_decode = self.tokenizer.is_fast and not self.tokenizer.clean_up_tokenization_spaces
        elif self.tokenizer_type == "tiktoken":
            self.tokenizer = tiktoken.get_encoding(self.tokenizer_model)
        elif self.tokenizer_type == "google":
            from vertexai.preview.tokenization import get_tokenizer_for_model
            self.tokenizer = get_tokenizer_for_model(self.tokenizer_model)
        else:
            raise ValueError(f"Invalid tokenizer type: {self.tokenizer_type}")
        
        self.SYSTEM_PROMPT = kwargs.get("system_prompt", "You are a helpful assistant")

        if api_provider == "openai":
            self.api = AsyncOpenAI(
                api_key=api_key,
                base_url=api_url,
            )
        elif api_provider == "azure-openai":
            self.api = AsyncAzureOpenAI(
                api_key=api_key,
                azure_endpoint=api_url,
                api_version=kwargs["azure_api_version"]
            )
        elif api_provider == "vllm":
            self.api = AsyncOpenAI(
                api_key=api_key,
                base_url=api_url,
                max_retries=kwargs["max_retries"],
                timeout=kwargs["timeout"]
            )
        elif api_provider == "gemini":
            from google import genai
            from google.genai import types
            self.api = genai.Client(
                vertexai=True, project=kwargs["project_ID"], location=kwargs["location"], http_options=types.HttpOptions(timeout=kwargs["http_timeout"] * 1000)
            )

            self.gemini_retry_config ={
                "initial": kwargs["retry_delay"],
                "maximum": kwargs["retry_max_delay"],
                "multiplier": kwargs["retry_multiplier"],
                "timeout": kwargs["retry_timeout"]
            }

        elif api_provider == "aws":
            from langchain_aws import ChatBedrockConverse
            self.api = ChatBedrockConverse(
                model=model,
                temperature=kwargs["temperature"],
                max_tokens=kwargs["max_tokens"],
                region_name=kwargs["region"],
                top_p=kwargs["top_p"],
            )

        self.model = model
        self.model_config = {k: v for k, v in kwargs.items()}

    def encode(self, text: str) -> list:
        """
        Encodes the given text using the tokenizer
        
        Parameters:
            text (`str`): Text to encode
        
        Returns:
            `dict`: Encoded text
        """
        if self.tokenizer_type == "huggingface":
            return self.tokenizer.encode(text, add_special_tokens=False)
        elif self.tokenizer_type == "tiktoken":
            return self.tokenizer.encode(text)
        elif self.tokenizer_type == "google":
            return self.tokenizer._sentencepiece_adapter._tokenizer.encode(text)
        else:
            raise ValueError(f"Invalid tokenizer type: {self.tokenizer_type}")

    def decode(self, tokens: list) -> str:
        """
        Decodes the given tokens using the tokenizer
        
        Parameters:
            tokens (`list`): Tokens to decode
        
        Returns:
            `str`: Decoded text
        """
        if self.tokenizer_type == "huggingface":
            # The Rust tokenizer directly: transformers' decode first type-checks every token id in Python
            # (to_py_obj), which took over 90% of a fully cached run. Same text unless the tokenizer cleans
            # up spaces, so those keep the transformers path (a different prompt would miss the request cache).
            if self._hf_fast_decode:
                return self.tokenizer.backend_tokenizer.decode(tokens, skip_special_tokens=False)
            return self.tokenizer.decode(tokens)
        elif self.tokenizer_type == "tiktoken":
            return self.tokenizer.decode(tokens)
        elif self.tokenizer_type == "google":
            return self.tokenizer._sentencepiece_adapter._tokenizer.decode(tokens)
        else:
            raise ValueError(f"Invalid tokenizer type: {self.tokenizer_type}")

    @cache
    def token_count(self, text: str) -> int:
        """
        Returns the token count of the given text
        
        Parameters:
            text (`str`): Text to count tokens
            use_cache (`bool`, optional): Use cache for token count. Defaults to True.
        
        Returns:
            `int`: Token count of the text
        """
        if self.tokenizer_type == "tiktoken":
            return len(self.tokenizer.encode(text))
        elif self.tokenizer_type == "huggingface":
            return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])
        elif self.tokenizer_type == "google":
            return self.tokenizer.count_tokens(text).total_tokens
        else:
            raise ValueError(f"Invalid tokenizer type: {self.tokenizer_type}")
            
    async def generate_response(
        self, 
        system_prompt: str,
        user_prompt: Union[str, List[str]],
        max_tokens: int = 100, 
        temperature: float = 0.0, 
        top_p: float = 1.0,
        add_default_system_prompt: bool = True
    ) -> dict:
        """
        Generates a response using the API with the given prompts
        
        Parameters:
            system_prompt (`str`): System prompt
            user_prompt (`str`): User prompt
            max_tokens (`int`, optional): Maximum tokens to generate. Defaults to 100.
            temperature (`float`, optional): Temperature. Defaults to 0.0 (Greedy Sampling).
            top_p (`float`, optional): Top-p. Defaults to 1.0.
        
        Returns:
            `dict`: Response from the API that includes the response, prompt tokens count, completion tokens count, total tokens count, and stopping reason
        """
        messages = build_messages(system_prompt, user_prompt, self.SYSTEM_PROMPT, add_default_system_prompt)
        request = build_request(
            self.api_provider, self.api_url, self.model, self.model_config, self.SYSTEM_PROMPT,
            messages, max_tokens, temperature, top_p
        )
        if self.cache is not None:
            output, cache_hit = await self.cache.fetch(request, self._send)
        else:
            output, cache_hit = await self._send(request), False
        return {**output, "request_hash": request_key(request), "request_cache_hit": cache_hit}

    async def _send(self, request: dict) -> dict:
        """
        Sends a request record (see `build_request`) to the provider

        Returns:
            `dict`: Response from the API that includes the response, prompt tokens count, completion tokens count, total tokens count, and stopping reason
        """
        if self.api_provider in OPENAI_COMPATIBLE_PROVIDERS:
            @retry(reraise=True, wait=wait_random(1, 20), retry=retry_if_exception_type(RateLimitError), stop=stop_after_delay(300))
            async def generate_content():
                params = {k: v for k, v in request.items() if k not in ENDPOINT_FIELDS}

                completion = await self.api.chat.completions.create(
                    **params
                )
                return completion
            
            completion = await generate_content()
            

            output = {
                # content is None when reasoning uses the whole max_tokens budget (finish_reason "length");
                # treat it as an empty (wrong) answer instead of crashing the scorer
                "response": completion.choices[0].message.content or "",
                "prompt_tokens": completion.usage.prompt_tokens,
                "completion_tokens": completion.usage.completion_tokens,
                "total_tokens": completion.usage.total_tokens,
                "finish_reason": completion.choices[0].finish_reason,
                "cached_tokens": completion.usage.prompt_tokens_details.cached_tokens if completion.usage.prompt_tokens_details else None,
                "reasoning_tokens": completion.usage.completion_tokens_details.reasoning_tokens if completion.usage.completion_tokens_details else None,
                # USD cost; only some providers (e.g. OpenRouter) report it
                "cost": getattr(completion.usage, "cost", None),
                # Backend that served the request (OpenRouter only), to check provider pinning held
                "provider": getattr(completion, "provider", None),
                "system_fingerprint": getattr(completion, "system_fingerprint", None),
            }
            return output

        elif self.api_provider == "gemini":
            from google.genai import types
            from google.genai.errors import ClientError

            @retry(reraise=True, wait=wait_random_exponential(multiplier=self.gemini_retry_config["multiplier"], max=self.gemini_retry_config["maximum"]), stop=stop_after_delay(self.gemini_retry_config["timeout"]), retry=retry_if_exception_type((ClientError, httpx.ConnectTimeout, httpx.TimeoutException)))
            async def generate_content():
                completion = await self.api.aio.models.generate_content(
                    model=request["model"],
                    contents=request["contents"],
                    config=types.GenerateContentConfig(
                        system_instruction=request["system_instruction"],
                        temperature=request["temperature"],
                        top_p=request["top_p"],
                        max_output_tokens=request["max_output_tokens"],
                        thinking_config=types.ThinkingConfig(thinking_budget=request["thinking_budget"]) if request["thinking_budget"] is not None else None,
                        seed=request["seed"]

                    )
                )
                return completion

            completion = await generate_content()

            return {
                "response": completion.text if completion.text is not None else "",
                "prompt_tokens": completion.usage_metadata.prompt_token_count,
                "completion_tokens": completion.usage_metadata.candidates_token_count,
                "total_tokens": completion.usage_metadata.total_token_count,
                "finish_reason": completion.candidates[0].finish_reason,
                "cached_tokens": completion.usage_metadata.cached_content_token_count if completion.usage_metadata.cached_content_token_count is not None else None,
                "reasoning_tokens": completion.usage_metadata.thoughts_token_count if completion.usage_metadata.thoughts_token_count is not None else None
            }
        elif self.api_provider == "aws":
            @retry(reraise=True, wait=wait_random(5, 20), retry=retry_if_exception_message(match=r".*ThrottlingException.*"))
            async def generate_content():
                completion = await self.api.ainvoke(request["messages"])
                return completion
            
            completion = await generate_content()

            return {
                "response": completion.content,
                "prompt_tokens": completion.usage_metadata["input_tokens"],
                "completion_tokens": completion.usage_metadata["output_tokens"],
                "total_tokens": completion.usage_metadata["total_tokens"],
                "finish_reason": completion.response_metadata["stopReason"]
            }
        else:
            raise ValueError(f"Invalid API provider: {self.api_provider}")
