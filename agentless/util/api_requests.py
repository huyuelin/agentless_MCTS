import signal
import time
from typing import Dict, Union
from typing import List, Optional
import requests
import os

import openai
import tiktoken
import logging

# Configuration
GPT4V_KEY = "52f7817a4f964a4db1dc57b7be25427e"
GPT4V_ENDPOINT = "https://gcrgpt4aoai5c.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2024-02-15-preview"

# GPT4V_KEY = "eb5c8ad7dda2424aa0996e4cf6ce637e"
# GPT4V_ENDPOINT = "https://gcrgpt4aoai5.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2024-02-15-preview"

headers = {
    "Content-Type": "application/json",
    "api-key": GPT4V_KEY,
}

class ChatCompletionMessage:
    def __init__(self, content, role, function_call=None, tool_calls=None):
        self.content = content
        self.role = role
        self.function_call = function_call
        self.tool_calls = tool_calls

class Choice:
    def __init__(self, finish_reason, index, logprobs, message):
        self.finish_reason = finish_reason
        self.index = index
        self.logprobs = logprobs
        self.message = message

class CompletionUsage:
    def __init__(self, completion_tokens, prompt_tokens, total_tokens):
        self.completion_tokens = completion_tokens
        self.prompt_tokens = prompt_tokens
        self.total_tokens = total_tokens

class ChatCompletion:
    def __init__(self, id, choices, created, model, object, system_fingerprint, usage):
        self.id = id
        self.choices = choices
        self.created = created
        self.model = model
        self.object = object
        self.system_fingerprint = system_fingerprint
        self.usage = usage



#client = openai.OpenAI()


def num_tokens_from_messages(message, model="gpt-3.5-turbo-0301"):
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    if isinstance(message, list):
        # use last message.
        num_tokens = len(encoding.encode(message[0]["content"]))
    else:
        num_tokens = len(encoding.encode(message))
    return num_tokens


def create_chatgpt_config(
    message: Union[str, list],
    max_tokens: int,
    temperature: float = 1,
    batch_size: int = 1,
    system_message: str = "You are a helpful assistant.",
    model: str = "gpt-4o",
) -> Dict:
    if isinstance(message, list):
        messages = [{"role": "system", "content": [{"type": "text", "text": system_message}]}] + message
    else:
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_message}]},
            {"role": "user", "content": [{"type": "text", "text": message}]}
        ]

    payload = {
        "messages": messages,
        "temperature": temperature,
        "top_p": 0.95,
        "n": batch_size,
        "max_tokens": max_tokens,
    }
    return payload


def handler(signum, frame):
    # swallow signum and frame
    raise Exception("end of time")


def request_chatgpt_engine(config):
    ret = None
    
    count_fo_wait=0
    
    while ret is None:
        try:
            # signal.signal(signal.SIGALRM, handler)
            # signal.alarm(100)
            # ret = client.chat.completions.create(**config)
            # signal.alarm(0)
            signal.signal(signal.SIGALRM, handler)
            signal.alarm(100)
            response = requests.post(GPT4V_ENDPOINT, headers=headers, json=config)
            #logging.info("GPT4V_KEY: ", GPT4V_KEY)
            print("GPT4V_KEY: 52f7817a4f964a4db1dc57b7be25427e")
            #print("GPT4V_KEY: eb5c8ad7dda2424aa0996e4cf6ce637e")
            response.raise_for_status()
            json_response = response.json()
            
            choices = [
                Choice(
                    finish_reason=choice['finish_reason'],
                    index=choice['index'],
                    logprobs=choice.get('logprobs'),
                    message=ChatCompletionMessage(
                        content=choice['message']['content'],
                        role=choice['message']['role']
                    )
                )
                for choice in json_response['choices']
            ]
            
            usage = CompletionUsage(
                completion_tokens=json_response['usage']['completion_tokens'],
                prompt_tokens=json_response['usage']['prompt_tokens'],
                total_tokens=json_response['usage']['total_tokens']
            )
            
            ret = ChatCompletion(
                id=json_response['id'],
                choices=choices,
                created=json_response['created'],
                model=json_response['model'],
                object=json_response['object'],
                system_fingerprint=json_response.get('system_fingerprint'),
                usage=usage
            )
            
            signal.alarm(0)
        except openai._exceptions.BadRequestError as e:
            print(e)
            signal.alarm(0)
        except openai._exceptions.RateLimitError as e:
            print("Rate limit exceeded. Waiting...")
            print(e)
            signal.alarm(0)
            time.sleep(5)
        except openai._exceptions.APIConnectionError as e:
            print("API connection error. Waiting...")
            signal.alarm(0)
            time.sleep(5)
        except Exception as e:
            count_fo_wait =  count_fo_wait + 1;   
            print("Unknown error. Waiting..." ,count_fo_wait)
            print(e)
            signal.alarm(0)
            time.sleep(2)
    return ret


# def create_anthropic_config(
#     message: str,
#     prefill_message: str,
#     max_tokens: int,
#     temperature: float = 1,
#     batch_size: int = 1,
#     system_message: str = "You are a helpful assistant.",
#     model: str = "claude-2.1",
# ) -> Dict:
#     if isinstance(message, list):
#         config = {
#             "model": model,
#             "temperature": temperature,
#             "max_tokens": max_tokens,
#             "system": system_message,
#             "messages": message,
#         }
#     else:
#         config = {
#             "model": model,
#             "temperature": temperature,
#             "max_tokens": max_tokens,
#             "system": system_message,
#             "messages": [
#                 {"role": "user", "content": message},
#                 {"role": "assistant", "content": prefill_message},
#             ],
#         }
#     return config


# def request_anthropic_engine(client, config):
#     ret = None
#     while ret is None:
#         try:
#             signal.signal(signal.SIGALRM, handler)
#             signal.alarm(100)
#             ret = client.messages.create(**config)
#             signal.alarm(0)
#         except Exception as e:
#             print("Unknown error. Waiting...")
#             print(e)
#             signal.alarm(0)
#             time.sleep(10)
#     return ret
