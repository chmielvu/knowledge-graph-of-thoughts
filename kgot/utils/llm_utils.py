# Copyright (c) 2025 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# Main author: Lorenzo Paleari

import json
import os
import logging
import traceback
from typing import Type

import httpx
try:
    from langchain_ollama import ChatOllama
except ImportError:
    ChatOllama = None
from langchain_openai import ChatOpenAI
from openai import APIConnectionError, InternalServerError, OpenAI
from pydantic import BaseModel
from tenacity import (
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

CONFIG_LLM_PATH = ''
NUM_LLM_RETRIES = 1

logger = logging.getLogger("Controller.LLMUtils")

def init_llm_utils(config_path: str = CONFIG_LLM_PATH, 
                   num_retries: int = NUM_LLM_RETRIES):
    """
    Initialize the LLM utils with the given configuration path.
    """
    global CONFIG_LLM_PATH
    CONFIG_LLM_PATH = config_path
    global NUM_LLM_RETRIES
    NUM_LLM_RETRIES = num_retries
    logger.info(f"LLM utils initialized with config path: {CONFIG_LLM_PATH} and num retries: {NUM_LLM_RETRIES}")


def _get_llm_retries():
    """
    Get the number of retries for LLM requests.
    """
    global NUM_LLM_RETRIES
    return NUM_LLM_RETRIES


def _retry_call(fn):
    try: 
        for attempt in Retrying(
            wait=wait_random_exponential(min=1, max=60), 
            stop=stop_after_attempt(_get_llm_retries()), 
            reraise=True,
            retry=(
                retry_if_exception_type(InternalServerError) |
                retry_if_exception_type(APIConnectionError) |
                retry_if_exception_type(httpx.ConnectTimeout) |
                retry_if_exception_type(httpx.ReadTimeout)
            )
        ):
            with attempt:
                try:
                    return fn()
                except InternalServerError as e:
                    logger.error(f"Internal Server Error when invoking the chain: {str(e)} - Type of error: {type(e)}")
                    raise  # Re-raise the exception to trigger retry
                except APIConnectionError as e:
                    logger.error(f"API Connection Error when invoking the chain: {str(e)} - Type of error: {type(e)}")
                    raise
                except Exception as e:
                    logger.error(f"Error when invoking the chain: {str(e)} - Type of error: {traceback.format_exc()}")
                    raise
    except Exception as e:
        logger.error(f"Failed to invoke the chain after {NUM_LLM_RETRIES} attempts: {str(e)}")
        raise


def invoke_with_retry(chain, *args, **kwargs):
    return _retry_call(lambda: chain.invoke(*args, **kwargs))


def _should_use_json_object_structured_output(llm) -> bool:
    base_url = getattr(llm, "openai_api_base", None)
    return isinstance(llm, ChatOpenAI) and isinstance(base_url, str) and "nano-gpt.com" in base_url


def _prompt_to_text(prompt) -> str:
    if isinstance(prompt, str):
        return prompt
    if hasattr(prompt, "to_string"):
        return prompt.to_string()
    if hasattr(prompt, "text"):
        return prompt.text
    return str(prompt)


def _clean_json_content(content: str) -> str:
    text = (content or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text.strip("`").strip()


def _invoke_structured_via_json_object(llm, schema_model: Type[BaseModel], prompt):
    client = OpenAI(
        api_key=llm.openai_api_key.get_secret_value() if llm.openai_api_key is not None else None,
        base_url=getattr(llm, "openai_api_base", None),
        organization=getattr(llm, "openai_organization", None),
    )
    schema = schema_model.model_json_schema()
    required_keys = schema.get("required") or list(schema.get("properties", {}).keys())
    key_instruction = ", ".join(required_keys) if required_keys else "the schema-defined keys"
    prompt_text = _prompt_to_text(prompt)
    response = client.chat.completions.create(
        model=llm.model_name,
        temperature=llm.temperature,
        max_tokens=llm.max_tokens,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    "Return only a valid JSON object matching this schema. "
                    f"Use exactly these top-level keys: {key_instruction}. "
                    "Do not rename keys, omit required keys, add extra keys, "
                    "or include markdown, code fences, or explanatory text.\n"
                    f"{json.dumps(schema, ensure_ascii=False)}"
                ),
            },
            {"role": "user", "content": prompt_text},
        ],
    )
    content = _clean_json_content(response.choices[0].message.content)
    return schema_model.model_validate_json(content)


def invoke_structured_with_retry(llm, schema_model: Type[BaseModel], prompt, method: str = "json_schema"):
    if method == "json_schema" and _should_use_json_object_structured_output(llm):
        return _retry_call(lambda: _invoke_structured_via_json_object(llm, schema_model, prompt))

    chain = llm.with_structured_output(schema_model, method=method)
    return invoke_with_retry(chain, prompt)


def get_model_configurations(model_name: str) -> dict:
    global CONFIG_LLM_PATH
    with open(CONFIG_LLM_PATH, 'r') as config_file:
        config = json.load(config_file)

    if model_name is None:
        raise ValueError(f"Model '{model_name}' not found in the supported models. Update the config_llms.json file")

    model_config = config[model_name]
    for key, value in model_config.items():
        if isinstance(value, str):
            model_config[key] = os.path.expandvars(value)

    return model_config


def get_llm(model_name: str, temperature: float = None, max_tokens: int = None):
    # Set up the LLMs
    model_config = get_model_configurations(model_name)
    if temperature is not None:
        model_config["temperature"] = temperature
    if not (0 <= model_config["temperature"] <= 1):
        raise ValueError(
            f"LLM models temperature needs to be in the range [0, 1], but given {model_config['temperature']}")
    model_config["max_tokens"] = max_tokens

    llm_to_return = None
    if model_config["model_family"] == "OpenAI":
        llm_to_return = ChatOpenAI(
            model=model_config["model"],
            api_key=model_config["api_key"],
            base_url=model_config.get("base_url"),
            max_tokens=model_config["max_tokens"],
            organization=model_config["organization"],
            **{key: model_config[key] for key in 
               ["temperature", "reasoning_effort"] if key in model_config}
        )
    elif model_config["model_family"] == "Ollama":
        if ChatOllama is None:
            raise ImportError("langchain_ollama is required for Ollama models")
        llm_to_return = ChatOllama(
            model=model_config["model"],
            temperature=model_config["temperature"],
            base_url=model_config["base_url"] if "base_url" in model_config else "localhost:11434",
            num_ctx=model_config["num_ctx"],
            num_predict=model_config["num_predict"],
            num_batch=model_config["num_batch"],
            keep_alive=-1)
    else:
        raise ValueError(f"Model family '{model_config['model_family']}' not supported, check the config_llms.json file")

    return llm_to_return
