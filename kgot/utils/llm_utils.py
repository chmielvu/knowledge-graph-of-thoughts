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
from langchain_openai import ChatOpenAI
from openai import APIConnectionError, InternalServerError
from pydantic import BaseModel
from tenacity import (
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from kgot.utils.circuit_breaker import (
    CircuitBreakerOpenError,
    get_llm_circuit_breaker,
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
    """
    Invoke a chain with retry logic and circuit breaker protection.

    The circuit breaker prevents cascading failures when the LLM service is degraded.
    After 5 consecutive failures, the circuit opens and blocks calls for 60 seconds,
    then transitions to half-open to test recovery.
    """
    cb = get_llm_circuit_breaker()

    if not cb.is_call_allowed():
        raise CircuitBreakerOpenError(
            f"LLM service circuit breaker is {cb.state}. "
            f"Failures: {cb._failure_count}/{cb.failure_threshold}. "
            f"Will retry after recovery timeout."
        )

    try:
        result = _retry_call(lambda: chain.invoke(*args, **kwargs))
        cb.record_success()
        return result
    except Exception as e:
        cb.record_failure()
        raise


def invoke_structured_with_retry(llm, schema_model: Type[BaseModel], prompt):
    """Provider-enforced structured output over the strict tool-call contract.

    Binds the Pydantic model as a single strict function tool and forces its
    invocation with tool_choice="required" - the form OpenAI-compatible
    routers honor reliably, unlike with_structured_output's object-form
    tool_choice. The returned tool arguments are validated by Pydantic.
    Fails fast when the transport returns text instead of the tool call.
    """
    bound = llm.bind_tools(
        [schema_model],
        tool_choice="required",
        parallel_tool_calls=False,
        strict=True,
    )
    message = invoke_with_retry(bound, prompt)
    tool_calls = getattr(message, "tool_calls", None)
    if not tool_calls:
        raise ValueError(
            f"Structured output failed: transport did not emit the required "
            f"tool call for schema {schema_model.__name__}. "
            f"Content: {str(getattr(message, 'content', ''))[:200]}"
        )
    return schema_model.model_validate(tool_calls[0]["args"])


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
    else:
        raise ValueError(f"Model family '{model_config['model_family']}' not supported. Only 'OpenAI' is supported. Check the config_llms.json file")

    return llm_to_return
