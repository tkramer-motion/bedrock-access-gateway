import base64
import json
import logging
import os
import re
import time
from abc import ABC
from copy import deepcopy
from functools import cache
from typing import Iterable, Iterator, Literal, Any
from urllib.parse import quote, urlparse

import boto3

# boto3.setup_default_session(profile_name='mldc')
import numpy as np
import requests
import tiktoken
from botocore.config import Config
from fastapi import HTTPException

from api.models.base import BaseChatModel, BaseEmbeddingsModel
from api.schema import (
    # Chat
    ChatResponse,
    ChatRequest,
    Choice,
    ChatResponseMessage,
    Usage,
    ChatStreamResponse,
    ImageContent,
    TextContent,
    ToolCall,
    ChoiceDelta,
    SystemMessage,
    UserMessage,
    AssistantMessage,
    ToolMessage,
    Function,
    ResponseFunction,
    # Embeddings
    EmbeddingsRequest,
    EmbeddingsResponse,
    EmbeddingsUsage,
    Embedding, )
from api.setting import DEBUG, AWS_REGION

logger = logging.getLogger(__name__)

config = Config(connect_timeout=60, read_timeout=2000, retries={"max_attempts": 5})

lambda_client = boto3.client('lambda', config=config, region_name=AWS_REGION, )

bedrock_runtime = boto3.client(
    service_name="bedrock-runtime",
    region_name=AWS_REGION,
    config=config,
)
bedrock_client = boto3.client(
    service_name='bedrock',
    region_name=AWS_REGION,
    config=config,
)

bedrock_agent_client = boto3.client('bedrock-agent', region_name=AWS_REGION, config=config)
bedrock_agent_runtime = boto3.client('bedrock-agent-runtime', region_name=AWS_REGION, config=config)


def get_inference_region_prefix():
    if AWS_REGION.startswith('ap-'):
        return 'apac'
    return AWS_REGION[:2]


# https://docs.aws.amazon.com/bedrock/latest/userguide/inference-profiles-support.html
cr_inference_prefix = get_inference_region_prefix()

SUPPORTED_BEDROCK_EMBEDDING_MODELS = {
    "cohere.embed-multilingual-v3": "Cohere Embed Multilingual",
    "cohere.embed-english-v3": "Cohere Embed English",
    # Disable Titan embedding.
    # "amazon.titan-embed-text-v1": "Titan Embeddings G1 - Text",
    # "amazon.titan-embed-image-v1": "Titan Multimodal Embeddings G1"
}

ENCODER = tiktoken.get_encoding("cl100k_base")

# Claude Opus 4.8 and Claude 5 models think adaptively: they reject temperature/top_p, budget_tokens thinking and assistant prefill
ADAPTIVE_THINKING_MODELS = ("anthropic.claude-opus-5-5", "anthropic.claude-opus-4-8")

THINKING_TAG = re.compile(r"(?<![\w@])@thinking\b")

# Open WebUI's background tasks (titles, tags, follow-ups) quote the chat history, and with it any @tools or @thinking.
# Its RAG template also starts with "### Task:", but wraps a real user message around retrieved documents.
TASK_PROMPT = re.compile(r"### Task:(?!\s*Respond to the user query using the provided context).*<chat_history>", re.DOTALL)

# Bedrock's stop reasons when the model's safety classifiers decline a request
REFUSAL_STOP_REASONS = ("content_filtered", "refusal")

# Claude Opus 5.5's biology classifier can decline drug discovery data. Bedrock has no server-side fallback,
# so the gateway retries a declined turn on the model Anthropic recommends, with a notice for the user.
REFUSAL_FALLBACKS = {
    "global.anthropic.claude-opus-5-5": (
        "global.anthropic.claude-opus-4-8",
        "Claude Opus 5.5's safety filter declined this request (it flags some biology and chemistry data), so Claude Opus 4.8 is answering instead.",
    ),
}


def uses_adaptive_thinking(model_id: str) -> bool:
    return any(model in model_id for model in ADAPTIVE_THINKING_MODELS)


def list_bedrock_models() -> dict:
    """Automatically getting a list of supported models.

    Returns a model list combines:
        - ON_DEMAND models.
        - Cross-Region Inference Profiles (if enabled via Env)
    """
    # model_list = {}
    # try:
    #     profile_list = []
    #     if ENABLE_CROSS_REGION_INFERENCE:
    #         # List system defined inference profile IDs
    #         response = bedrock_client.list_inference_profiles(
    #             maxResults=1000,
    #             typeEquals='SYSTEM_DEFINED'
    #         )
    #         profile_list = [p['inferenceProfileId'] for p in response['inferenceProfileSummaries']]
    #
    #     # List foundation models, only cares about text outputs here.
    #     response = bedrock_client.list_foundation_models(
    #         byOutputModality='TEXT'
    #     )
    #
    #     for model in response['modelSummaries']:
    #         model_id = model.get('modelId', 'N/A')
    #         stream_supported = model.get('responseStreamingSupported', True)
    #         status = model['modelLifecycle'].get('status', 'ACTIVE')
    #
    #         # currently, use this to filter out rerank models and legacy models
    #         if not stream_supported or status != "ACTIVE":
    #             continue
    #
    #         inference_types = model.get('inferenceTypesSupported', [])
    #         input_modalities = model['inputModalities']
    #         # Add on-demand model list
    #         if 'ON_DEMAND' in inference_types:
    #             model_list[model_id] = {
    #                 'modalities': input_modalities
    #             }
    #
    #         # Add cross-region inference model list.
    #         profile_id = cr_inference_prefix + '.' + model_id
    #         if profile_id in profile_list:
    #             model_list[profile_id] = {
    #                 'modalities': input_modalities
    #             }
    #
    # except Exception as e:
    #     logger.error(f"Unable to list models: {str(e)}")
    #
    # if not model_list:
    #     # In case stack not updated.
    #     model_list[DEFAULT_MODEL] = {
    #         'modalities': ["TEXT", "IMAGE"]
    #     }

    # return model_list
    return {'global.anthropic.claude-opus-5-5': {'modalities': ['TEXT', 'IMAGE']}, 'global.anthropic.claude-opus-4-8': {'modalities': ['TEXT', 'IMAGE']}, 'global.anthropic.claude-opus-4-5-20251101-v1:0': {'modalities': ['TEXT', 'IMAGE']}, 'us.anthropic.claude-3-7-sonnet-20250219-v1:0': {'modalities': ['TEXT', 'IMAGE']}, 'us.anthropic.claude-opus-4-20250514-v1:0': {'modalities': ['TEXT', 'IMAGE']}, 'us.anthropic.claude-sonnet-4-20250514-v1:0': {'modalities': ['TEXT', 'IMAGE']},
            'us.meta.llama4-maverick-17b-instruct-v1:0': {'modalities': ['TEXT', 'IMAGE']}, 'us.deepseek.r1-v1:0': {'modalities': ['TEXT']}}


# Initialize the model list.
bedrock_model_list = list_bedrock_models()


class BedrockModel(BaseChatModel):

    def list_models(self) -> list[str]:
        """Always refresh the latest model list"""
        global bedrock_model_list
        bedrock_model_list = list_bedrock_models()
        return list(bedrock_model_list.keys())

    def validate(self, chat_request: ChatRequest):
        """Perform basic validation on requests"""
        error = ""
        # check if model is supported
        if chat_request.model not in bedrock_model_list.keys():
            error = f"Unsupported model {chat_request.model}, please use models API to get a list of supported models"

        if error:
            raise HTTPException(
                status_code=400,
                detail=error,
            )

    def _invoke_bedrock(self, chat_request: ChatRequest, stream=False):
        """Common logic for invoke bedrock models"""
        # Image tool results and redacted thinking carry raw bytes, which model_dump_json() cannot encode
        raw_request = json.dumps(chat_request.model_dump(), default=str)
        if DEBUG:
            logger.info("Raw request: " + raw_request)

        logger.info(f"LENGTH OF REQUEST: {len(raw_request)}")

        # convert OpenAI chat request to Bedrock SDK request
        args = self._parse_request(chat_request)
        if DEBUG:
            logger.info("Bedrock request: " + json.dumps(str(args)))

        kbs = [{"name": row["name"], "knowledgeBaseId": row["knowledgeBaseId"]} for row in
               bedrock_agent_client.list_knowledge_bases()["knowledgeBaseSummaries"] if
               row["status"] in ("ACTIVE", "UPDATING")]
        last_content = args["messages"][-1]["content"] if args["messages"] else []
        message = last_content[0].get("text", "") if last_content else ""

        reference_data = dict()
        references = dict()

        for kb in kbs:
            if f'@{kb["name"]}' in message.split():
                logger.info(f"Using knowledge base {kb['name']} for text message: {message}")
                retrieve_response = bedrock_agent_runtime.retrieve(
                    retrievalQuery={
                        'text': message.replace(f'@{kb["name"]}', '')[:998]
                    },
                    knowledgeBaseId=kb["knowledgeBaseId"],
                    retrievalConfiguration={
                        'vectorSearchConfiguration': {
                            'numberOfResults': 50,
                        }
                    }
                )
                if DEBUG:
                    logger.info(f"Got search results of {[row['content']['text'] for row in retrieve_response['retrievalResults']]}")

                for i, row in enumerate(retrieve_response['retrievalResults']):
                    if row["score"] >= .5:
                        if "metadata" in row and "x-amz-kendra-document-title" in row["metadata"]:
                            references[row["metadata"]["x-amz-kendra-document-title"]] = {"title": row["metadata"]["x-amz-kendra-document-title"], "url": row["location"]['kendraDocumentLocation']["uri"]}
                            reference_data[row["metadata"]["x-amz-kendra-document-title"]] = row['content']['text'].strip()

                if len(reference_data) <= 5:
                    for title, rows in reference_data.items():
                        args["messages"][-1]["content"].append({"document": {
                            'format': 'txt',
                            'name': title,
                            'source': {
                                'bytes': "\n".join(rows).encode()
                            }
                        }
                        })
                else:
                    args["messages"][-1]["content"].append({"document": {
                        'format': 'txt',
                        'name': "combined",
                        'source': {
                            'bytes': "\n".join(reference_data.values()).encode()
                        }
                    }})

        if "GUARDRAIL_IDENTIFIER" in os.environ:
            args["guardrailConfig"] = {
                'guardrailIdentifier': os.environ["GUARDRAIL_IDENTIFIER"],
                'guardrailVersion': os.environ["GUARDRAIL_VERSION"],
                'trace': 'enabled'
            }

        try:
            if stream:
                response = bedrock_runtime.converse_stream(**args)
            else:
                response = bedrock_runtime.converse(**args)
        except bedrock_runtime.exceptions.ValidationException as e:
            logger.error("Validation Error: " + str(e))
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(e)
            raise HTTPException(status_code=500, detail=str(e))
        return response, list(references.values())

    def chat(self, chat_request: ChatRequest) -> ChatResponse:
        """Default implementation for Chat API."""

        message_id = self.generate_message_id()
        response, references = self._invoke_bedrock(chat_request)

        output_message = response["output"]["message"]
        input_tokens = response["usage"]["inputTokens"]
        output_tokens = response["usage"]["outputTokens"]
        finish_reason = response["stopReason"]

        if finish_reason in REFUSAL_STOP_REASONS and chat_request.model in REFUSAL_FALLBACKS:
            fallback_model, _ = REFUSAL_FALLBACKS[chat_request.model]
            stop_details = response.get("additionalModelResponseFields", {}).get("stop_details")
            logger.warning(f"{chat_request.model} declined the request ({stop_details}); retrying on {fallback_model}")
            # The conversation goes over unchanged: Bedrock drops thinking blocks the fallback model can't read
            return self.chat(chat_request.model_copy(update={"model": fallback_model}))

        if finish_reason == "tool_use":
            # Every toolUse block needs a toolResult in the next turn
            tool_results = [
                self._tool_message(part["toolUse"]["toolUseId"], self._invoke_tool(part["toolUse"]["name"], part["toolUse"]["input"]))
                for part in output_message["content"] if "toolUse" in part
            ]
            args = chat_request.model_dump()
            args["messages"] = chat_request.messages + [output_message] + tool_results
            return self.chat(ChatRequest(**args))

        chat_response = self._create_response(
            model=chat_request.model,
            message_id=message_id,
            content=output_message["content"],
            finish_reason=finish_reason,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        if DEBUG:
            logger.info("Proxy response :" + chat_response.model_dump_json())

        return chat_response

    def chat_stream(self, chat_request: ChatRequest) -> Iterator[bytes]:
        """Default implementation for Chat Stream API.

        Bedrock is called right away; only reading its stream is deferred to the returned generator.
        """
        response, references = self._invoke_bedrock(chat_request, stream=True)
        return self._stream_response(chat_request, response, references)

    def _stream_response(self, chat_request: ChatRequest, response: dict, references: list[dict]) -> Iterator[bytes]:
        message_id = self.generate_message_id()

        # The assistant turn's Converse content blocks by contentBlockIndex, replayed verbatim if it ends in tool use
        blocks: dict[int, dict] = {}

        tool_use = False
        # The messageStop of a turn the model declined, when another model can take it over
        declined = None
        stream = response.get("stream")
        for chunk in stream:
            self._collect_content_block(blocks, chunk)
            stop_reason = chunk.get("messageStop", {}).get("stopReason")
            if stop_reason in REFUSAL_STOP_REASONS and chat_request.model in REFUSAL_FALLBACKS:
                # Read the rest of this stream, then retry on the fallback model
                declined = chunk["messageStop"]
                continue
            stream_response = self._create_response_stream(
                model_id=chat_request.model, message_id=message_id, chunk=chunk
            )
            if not stream_response:
                continue
            if DEBUG:
                logger.info("Proxy response :" + stream_response.model_dump_json())

            if stream_response.choices:
                if stream_response.choices[0].finish_reason == "stop":
                    if references:
                        s = "\n\n##### References:\n"
                        for reference in references:
                            parsed_url = urlparse(reference['url'])
                            url = f"{parsed_url.scheme}://{parsed_url.netloc}{quote(parsed_url.path)}"
                            if parsed_url.query:
                                url += f"?{parsed_url.query}"
                            s += f"  * [{reference['title']}]({url})\n"
                        stream_response.choices[0].delta.content = s

                if stop_reason in REFUSAL_STOP_REASONS:
                    # Open WebUI shows nothing for a content_filter finish
                    stream_response.choices[0].delta.content = "\n\nThe model's safety filter declined this request."

                if stream_response.choices[0].finish_reason == "tool_calls":
                    # Read the rest of this stream (its usage metadata) so the connection is released before the tools run
                    tool_use = True
                    continue

                yield self.stream_response_to_bytes(stream_response)
            elif (
                    chat_request.stream_options
                    and chat_request.stream_options.include_usage
                    and not tool_use
                    and declined is None
            ):
                # An empty choices for Usage as per OpenAI doc below:
                # if you set stream_options: {"include_usage": true}.
                # an additional chunk will be streamed before the data: [DONE] message.
                # The usage field on this chunk shows the token usage statistics for the entire request,
                # and the choices field will always be an empty array.
                # All other chunks will also include a usage field, but with a null value.
                yield self.stream_response_to_bytes(stream_response)

        if declined is not None:
            fallback_model, notice = REFUSAL_FALLBACKS[chat_request.model]
            stop_details = declined.get("additionalModelResponseFields", {}).get("stop_details")
            logger.warning(f"{chat_request.model} declined the request ({stop_details}); retrying on {fallback_model}")
            # Close a code block the declined answer left open, or the notice and the new answer render inside it
            streamed = "".join("".join(block["text"]) for block in blocks.values() if "text" in block)
            fence = "\n```" if streamed.count("```") % 2 else ""
            yield self._stream_text(message_id, chat_request.model, f"{fence}\n\n_{notice}_\n\n")
            # The conversation goes over unchanged: Bedrock drops thinking blocks the fallback model can't read
            yield from self.chat_stream(chat_request.model_copy(update={"model": fallback_model}))
            return

        if tool_use:
            # The gateway runs the tools itself, so tool calls are never streamed to the client
            yield from self._run_tools(chat_request, message_id, blocks)
            return

        # return an [DONE] message at the end.
        yield self.stream_response_to_bytes()

    @staticmethod
    def _collect_content_block(blocks: dict[int, dict], chunk: dict):
        if "contentBlockStart" in chunk:
            start = chunk["contentBlockStart"]["start"]
            if "toolUse" in start:
                blocks[chunk["contentBlockStart"]["contentBlockIndex"]] = {
                    "toolUse": {"toolUseId": start["toolUse"]["toolUseId"], "name": start["toolUse"]["name"], "input": []}
                }
        elif "contentBlockDelta" in chunk:
            index = chunk["contentBlockDelta"]["contentBlockIndex"]
            delta = chunk["contentBlockDelta"]["delta"]
            if "text" in delta:
                blocks.setdefault(index, {"text": []})["text"].append(delta["text"])
            elif "toolUse" in delta:
                blocks[index]["toolUse"]["input"].append(delta["toolUse"]["input"])
            elif "reasoningContent" in delta:
                reasoning = blocks.setdefault(index, {"reasoningContent": {"text": [], "signature": None, "redactedContent": None}})["reasoningContent"]
                reasoning["text"].append(delta["reasoningContent"].get("text", ""))
                if "signature" in delta["reasoningContent"]:
                    reasoning["signature"] = delta["reasoningContent"]["signature"]
                if "redactedContent" in delta["reasoningContent"]:
                    reasoning["redactedContent"] = delta["reasoningContent"]["redactedContent"]

    def _run_tools(self, chat_request: ChatRequest, message_id: str, blocks: dict[int, dict]) -> Iterator[bytes]:
        """Run the tools the model asked for, then stream its next turn."""
        assistant_content = []
        tool_results = []

        for index in sorted(blocks):
            block = blocks[index]
            if "text" in block:
                text = "".join(block["text"])
                if text.strip():
                    assistant_content.append({"text": text})
            elif "reasoningContent" in block:
                # Thinking is passed back unmodified so the model continues from it instead of thinking again
                reasoning = block["reasoningContent"]
                if reasoning["redactedContent"] is not None:
                    assistant_content.append({"reasoningContent": {"redactedContent": reasoning["redactedContent"]}})
                elif reasoning["signature"]:
                    assistant_content.append({"reasoningContent": {"reasoningText": {"text": "".join(reasoning["text"]), "signature": reasoning["signature"]}}})
            elif "toolUse" in block:
                tool_name = block["toolUse"]["name"]
                toolUseId = block["toolUse"]["toolUseId"]
                # A tool without parameters streams its input as "" rather than "{}"
                tool_args = "".join(block["toolUse"]["input"])

                try:
                    function_args = json.loads(tool_args) if tool_args.strip() else {}
                except Exception:
                    logger.exception(f"Error parsing tool_args for {tool_name}")
                    yield self._stream_text(message_id, chat_request.model, f'\n```text\nAttempted to call tool {tool_name} with arguments: {tool_args}\nI was unable to parse the JSON.\n```\n', "stop")
                    yield self.stream_response_to_bytes()
                    return

                try:
                    results = self._invoke_tool(tool_name, function_args)

                    # Add toolUse to assistant message content
                    assistant_content.append({'toolUse': {'input': function_args, 'name': tool_name, 'toolUseId': toolUseId}})

                    # Collect tool result
                    tool_results.append(self._tool_message(toolUseId, results))

                    if results.get("success", False) and results.get("data_type") != "image":
                        markdown_format = results.get("markdown_format", "json")
                        rendered = json.dumps(results["results"], indent=2, default=str) if markdown_format == "json" else results["results"]
                        if markdown_format != "json" or len(rendered) < 4000:
                            logger.info(f"Returning tool response of size {len(rendered)}")
                            yield self._stream_text(message_id, chat_request.model, f'\n```{markdown_format}\n{rendered}\n```\n')

                except Exception as e:
                    logger.exception(f"Error invoking tool {tool_name}")
                    yield self._stream_text(message_id, chat_request.model, f'\n```text\nAttempted to call tool {tool_name} with arguments: {function_args}\nI got an exception {e}.\n```\n', "stop")
                    yield self.stream_response_to_bytes()
                    return

        # Build the new messages with all tool uses and results
        args = chat_request.model_dump()
        args["messages"] = chat_request.messages + [{'content': assistant_content, 'role': 'assistant'}] + tool_results

        if DEBUG:
            logger.info(f"Calling chat_stream with ********{args}*********")

        # Separates this turn's text from the next one, and hands control back before the next Bedrock call,
        # so a client that left while the tools ran stops the loop here
        yield self._stream_text(message_id, chat_request.model, "\n\n")
        yield from self.chat_stream(ChatRequest(**args))

    def _invoke_tool(self, tool_name: str, tool_input: dict) -> dict:
        t0 = time.time()
        logger.info(f"Invoking tool {tool_name} with {tool_input} and lambda {self.get_tool_map().get(tool_name)}")

        response = lambda_client.invoke(
            FunctionName=self.get_tool_map()[tool_name],
            InvocationType='RequestResponse',
            Payload=json.dumps(tool_input).encode(),
        )

        logger.info(f"Finished tool {tool_name} in {time.time() - t0} seconds")
        results = json.loads(response["Payload"].read().decode())
        if "FunctionError" in response:
            # The Lambda raised; its payload is the error, which the model gets back as a failed tool result
            logger.error(f"Tool {tool_name} failed: {results}")
            return {"success": False, "message": f"The tool failed with {results.get('errorType')}: {results.get('errorMessage')}"}
        return results

    @staticmethod
    def _tool_message(tool_use_id: str, results: dict) -> ToolMessage:
        if not results.get("success"):
            content = results.get("message", "The tool did not return a result.")
        elif results.get("data_type", "json") == "json":
            content = {"results": results["results"]}
        else:
            content = results["results"]
            if results.get("data_type") == "image":
                # Copy rather than mutate, so results stays JSON-serializable for the caller
                content = {**content, "source": {**content["source"], "bytes": base64.b64decode(content["source"]["bytes"])}}

        return ToolMessage(
            tool_call_id=tool_use_id,
            content=content,
            status=None if results.get("success") else "error",
            data_type=results.get("data_type", "json"))

    def _stream_text(self, message_id: str, model: str, content: str, finish_reason: str | None = None) -> bytes:
        return self.stream_response_to_bytes(ChatStreamResponse(
            id=message_id,
            model=model,
            choices=[
                ChoiceDelta(
                    index=0,
                    delta=ChatResponseMessage(role="assistant", content=content),
                    logprobs=None,
                    finish_reason=finish_reason,
                )
            ],
        ))

    def _parse_system_prompts(self, chat_request: ChatRequest) -> list[dict[str, str]]:
        """Create system prompts.
        Note that not all models support system prompts.

        example output: [{"text" : system_prompt}]

        See example:
        https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference.html#message-inference-examples
        """

        system_prompts = []
        for message in chat_request.messages:
            if not isinstance(message, SystemMessage):
                continue
            if isinstance(message.content, str):
                texts = [message.content]
            else:
                texts = [part.text for part in message.content]
            system_prompts.extend({"text": text} for text in texts if text.strip())

        return system_prompts

    def _parse_messages(self, chat_request: ChatRequest) -> list[dict]:
        """
        Converse API only support user and assistant messages.

        example output: [{
            "role": "user",
            "content": [{"text": input_text}]
        }]

        See example:
        https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference.html#message-inference-examples
        """
        messages = []
        for message in chat_request.messages:
            if isinstance(message, UserMessage):
                messages.append(
                    {
                        "role": message.role,
                        "content": self._parse_content_parts(
                            message, chat_request.model
                        ),
                    }
                )
            elif isinstance(message, AssistantMessage):
                if isinstance(message.content, str):
                    text = message.content
                else:
                    text = "".join(part.text for part in message.content or [] if isinstance(part, TextContent))
                if text.startswith("This prompt goes against Relay Acceptable Use policy."):
                    logger.info("Popping message to avoid poisoning thread")
                    if messages:
                        messages.pop()
                    continue

                # Text message, followed by any tool uses. Bedrock rejects blank text blocks.
                content = []
                if message.content:
                    content = [part for part in self._parse_content_parts(message, chat_request.model)
                               if "text" not in part or part["text"].strip()]
                for tool_call in message.tool_calls or []:
                    try:
                        tool_input = json.loads(tool_call.function.arguments or "{}")
                    except json.JSONDecodeError:
                        logger.warning(f"Unable to parse arguments of tool call {tool_call.id}")
                        tool_input = {}
                    content.append(
                        {
                            "toolUse": {
                                "toolUseId": tool_call.id,
                                "name": tool_call.function.name,
                                "input": tool_input if isinstance(tool_input, dict) else {}
                            }
                        }
                    )
                if content:
                    messages.append({"role": message.role, "content": content})
            elif isinstance(message, ToolMessage):
                # Bedrock does not support tool role,
                # Add toolResult to content
                # https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ToolResultBlock.html

                tool_result = {
                    "toolUseId": message.tool_call_id,
                }

                if message.status:
                    tool_result["status"] = message.status
                    tool_result["content"] = [{"text": message.content}]
                else:
                    tool_result["content"] = [{message.data_type: message.content}]

                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "toolResult": tool_result
                            }
                        ],
                    }
                )
            elif isinstance(message, dict):
                # Handle dict messages with 'tool' role - convert to Bedrock format
                if message.get("role") == "tool":
                    tool_result = {
                        "toolUseId": message.get("tool_call_id"),
                    }
                    content = message.get("content")
                    status = message.get("status")

                    if status:
                        tool_result["status"] = status
                        tool_result["content"] = [{"text": content if isinstance(content, str) else json.dumps(content)}]
                    else:
                        # Parse string content as JSON if possible
                        if isinstance(content, str):
                            try:
                                content = json.loads(content)
                            except (json.JSONDecodeError, TypeError):
                                pass
                        # Bedrock "json" type requires a dict; use "text" for strings/other types
                        if isinstance(content, dict):
                            tool_result["content"] = [{"json": content}]
                        else:
                            tool_result["content"] = [{"text": content if isinstance(content, str) else json.dumps(content)}]

                    messages.append({
                        "role": "user",
                        "content": [{"toolResult": tool_result}]
                    })
                else:
                    messages.append(message)
            else:
                # ignore others, such as system messages
                continue
        return self._reframe_multi_payloard(messages)

    def _reframe_multi_payloard(self, messages: list) -> list:
        """ Receive messages and reformat them to comply with the Claude format

        With OpenAI format requests, it's not a problem to repeatedly receive messages from the same role, but
        with Claude format requests, you cannot repeatedly receive messages from the same role.

        This method searches through the OpenAI format messages in order and reformats them to the Claude format.

        ```
        openai_format_messages=[
            {"role": "user", "content": "Hello"},
            {"role": "user", "content": "Who are you?"},
        ]

        bedrock_format_messages=[
            {
                "role": "user",
                "content": [
                    {"text": "Hello"},
                    {"text": "Who are you?"}
                ]
            },
        ]
        """
        reformatted_messages = []
        current_role = None
        current_content = []

        # Search through the list of messages and combine messages from the same role into one list
        for message in messages:
            next_role = message['role']
            next_content = message['content']

            # If the next role is different from the previous message, add the previous role's messages to the list
            if next_role != current_role:
                if current_content:
                    reformatted_messages.append({
                        "role": current_role,
                        "content": current_content
                    })
                # Switch to the new role
                current_role = next_role
                current_content = []

            # Add the message content to current_content
            if isinstance(next_content, str):
                current_content.append({"text": next_content})
            elif isinstance(next_content, list):
                current_content.extend(next_content)

        # Add the last role's messages to the list
        if current_content:
            reformatted_messages.append({
                "role": current_role,
                "content": current_content
            })

        return reformatted_messages

    @cache
    def get_tools(self) -> list[dict[str, Any]]:
        s3 = boto3.resource("s3", config=config)
        obj = s3.Object(os.environ.get("RELAY_AI_TOOLS_BUCKET"), os.environ.get("RELAY_AI_TOOLS_KEY"))
        return json.load(obj.get()['Body'])

    @cache
    def get_tools_config(self):
        tools = deepcopy(self.get_tools())
        for tool in tools:
            del tool["toolSpec"]["lambda_arn"]
        return tools

    @cache
    def get_tool_map(self):
        return {tool["toolSpec"]["name"]: tool["toolSpec"]["lambda_arn"] for tool in self.get_tools()}

    def _parse_request(self, chat_request: ChatRequest) -> dict:
        """Create default converse request body.

        Also perform validations to tool call etc.

        Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html
        """

        messages = self._parse_messages(chat_request)
        system_prompts = self._parse_system_prompts(chat_request)

        last_user_text = self._last_user_text(chat_request)
        task = bool(TASK_PROMPT.match(last_user_text))
        adaptive_thinking = uses_adaptive_thinking(chat_request.model)
        thinking = "anthropic" in chat_request.model and not task and bool(THINKING_TAG.search(last_user_text))

        if adaptive_thinking and messages and messages[-1]["role"] == "assistant" and not any("toolUse" in part for part in messages[-1]["content"]):
            # Claude 5 rejects assistant prefill, which is how Open WebUI's "Continue response" asks for more
            messages.append({"role": "user", "content": [{"text": "Continue your previous response exactly where it stopped, without repeating anything."}]})

        # Base inference parameters.

        tokens = 32768
        if "claude-3-7" in chat_request.model:
            tokens = 131072
        if "llama4" in chat_request.model:
            tokens = 8192
        if adaptive_thinking:
            tokens = 64000

        inference_config = {
            "maxTokens": tokens,
            # "topP": chat_request.top_p
        }
        # Claude 5 rejects sampling parameters, and extended thinking requires the default temperature
        if not adaptive_thinking and not thinking:
            inference_config["temperature"] = chat_request.temperature

        if chat_request.stop is not None:
            stop = chat_request.stop
            if isinstance(stop, str):
                stop = [stop]
            inference_config["stopSequences"] = stop

        config = {"modelId": chat_request.model, "messages": messages, "system": system_prompts, "inferenceConfig": inference_config}
        if not task and any('@tools' in part.get("text", "") for message in messages for part in message["content"]):
            config["toolConfig"] = {
                "tools": self.get_tools_config()
            }
        if chat_request.model in REFUSAL_FALLBACKS:
            # The refusal category, logged when the model declines (Bedrock returns it to non-streaming calls only)
            config["additionalModelResponseFieldPaths"] = ["/stop_details"]
        if thinking:
            if adaptive_thinking:
                # @thinking raises the effort and streams a summary of the reasoning (Claude 5 thinks either way)
                config["additionalModelRequestFields"] = {
                    "thinking": {"type": "adaptive", "display": "summarized"},
                    "output_config": {"effort": "high"},
                }
            else:
                config["additionalModelRequestFields"] = {
                    "thinking": {
                        "type": "enabled",
                        "budget_tokens": 10000
                    }
                }
        return config

    @staticmethod
    def _last_user_text(chat_request: ChatRequest) -> str:
        """Text of the last message the user typed. Tool continuations end with tool results, which are skipped."""
        for message in reversed(chat_request.messages):
            if isinstance(message, UserMessage):
                if isinstance(message.content, str):
                    text = message.content
                else:
                    text = " ".join(part.text for part in message.content if isinstance(part, TextContent))
                if text.strip():
                    return text
        return ""

    def _create_response(
            self,
            model: str,
            message_id: str,
            content: list[dict] = None,
            finish_reason: str | None = None,
            input_tokens: int = 0,
            output_tokens: int = 0,
    ) -> ChatResponse:

        message = ChatResponseMessage(
            role="assistant",
        )
        if finish_reason == "tool_use":
            # https://docs.aws.amazon.com/bedrock/latest/userguide/tool-use.html#tool-use-examples
            tool_calls = []
            for part in content:
                if "toolUse" in part:
                    tool = part["toolUse"]
                    tool_calls.append(
                        ToolCall(
                            id=tool["toolUseId"],
                            type="function",
                            function=ResponseFunction(
                                name=tool["name"],
                                arguments=json.dumps(tool["input"]),
                            ),
                        )
                    )
            message.tool_calls = tool_calls
            message.content = None
        else:
            # Thinking models put a reasoning block ahead of the text
            message.content = "".join(part["text"] for part in content or [] if "text" in part)

        response = ChatResponse(
            id=message_id,
            model=model,
            choices=[
                Choice(
                    index=0,
                    message=message,
                    finish_reason=self._convert_finish_reason(finish_reason),
                    logprobs=None,
                )
            ],
            usage=Usage(
                prompt_tokens=input_tokens,
                completion_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
            ),
        )
        response.system_fingerprint = "fp"
        response.object = "chat.completion"
        response.created = int(time.time())
        return response

    def _create_response_stream(
            self, model_id: str, message_id: str, chunk: dict
    ) -> ChatStreamResponse | None:
        """Parsing the Bedrock stream response chunk.

        Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference.html#message-inference-examples
        """
        if DEBUG:
            logger.info("Bedrock response chunk: " + str(chunk))

        finish_reason = None
        message = None
        usage = None
        if "messageStart" in chunk:
            message = ChatResponseMessage(
                role=chunk["messageStart"]["role"],
                content="",
            )
        # Tool use blocks are not streamed: the gateway runs the tools itself (see _collect_content_block)
        if "contentBlockDelta" in chunk:
            delta = chunk["contentBlockDelta"]["delta"]
            if "text" in delta:
                # stream content
                message = ChatResponseMessage(
                    content=delta["text"],
                )
            elif delta.get("reasoningContent", {}).get("text"):
                # Open WebUI shows reasoning_content as a collapsible thinking block
                message = ChatResponseMessage(
                    reasoning_content=delta["reasoningContent"]["text"],
                )
        if "messageStop" in chunk:
            message = ChatResponseMessage()
            finish_reason = chunk["messageStop"]["stopReason"]

        if "metadata" in chunk:
            # usage information in metadata.
            metadata = chunk["metadata"]
            if "usage" in metadata:
                # token usage
                return ChatStreamResponse(
                    id=message_id,
                    model=model_id,
                    choices=[],
                    usage=Usage(
                        prompt_tokens=metadata["usage"]["inputTokens"],
                        completion_tokens=metadata["usage"]["outputTokens"],
                        total_tokens=metadata["usage"]["totalTokens"],
                    ),
                )
        if message:
            return ChatStreamResponse(
                id=message_id,
                model=model_id,
                choices=[
                    ChoiceDelta(
                        index=0,
                        delta=message,
                        logprobs=None,
                        finish_reason=self._convert_finish_reason(finish_reason),
                    )
                ],
                usage=usage,
            )

        return None

    def _parse_image(self, image_url: str) -> tuple[bytes, str]:
        """Try to get the raw data from an image url.

        Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ImageSource.html
        returns a tuple of (Image Data, Content Type)
        """
        pattern = r"^data:(image/[a-z]*);base64,\s*"
        content_type = re.search(pattern, image_url)
        # if already base64 encoded.
        # Only supports 'image/jpeg', 'image/png', 'image/gif' or 'image/webp'
        if content_type:
            image_data = re.sub(pattern, "", image_url)
            return base64.b64decode(image_data), content_type.group(1)

        # Send a request to the image URL
        response = requests.get(image_url)
        # Check if the request was successful
        if response.status_code == 200:

            content_type = response.headers.get("Content-Type")
            if not content_type.startswith("image"):
                content_type = "image/jpeg"
            # Get the image content
            image_content = response.content
            return image_content, content_type
        else:
            raise HTTPException(
                status_code=500, detail="Unable to access the image url"
            )

    def _parse_content_parts(
            self,
            message: UserMessage,
            model_id: str,
    ) -> list[dict]:
        if isinstance(message.content, str):
            return [
                {
                    "text": message.content,
                }
            ]
        content_parts = []
        for part in message.content:
            if isinstance(part, TextContent):
                content_parts.append(
                    {
                        "text": part.text,
                    }
                )
            elif isinstance(part, ImageContent):
                if not self.is_supported_modality(model_id, modality="IMAGE"):
                    raise HTTPException(
                        status_code=400,
                        detail=f"Multimodal message is currently not supported by {model_id}",
                    )
                image_data, content_type = self._parse_image(part.image_url.url)
                content_parts.append(
                    {
                        "image": {
                            "format": content_type[6:],  # image/
                            "source": {"bytes": image_data},
                        },
                    }
                )
            else:
                # Ignore..
                continue
        return content_parts

    @staticmethod
    def is_supported_modality(model_id: str, modality: str = "IMAGE") -> bool:
        model = bedrock_model_list.get(model_id)
        modalities = model.get('modalities', [])
        if modality in modalities:
            return True
        return False

    def _convert_tool_spec(self, func: Function) -> dict:
        return {
            "toolSpec": {
                "name": func.name,
                "description": func.description,
                "inputSchema": {
                    "json": func.parameters,
                },
            }
        }

    def _convert_finish_reason(self, finish_reason: str | None) -> str | None:
        """
        Below is a list of finish reason according to OpenAI doc:

        - stop: if the model hit a natural stop point or a provided stop sequence,
        - length: if the maximum number of tokens specified in the request was reached,
        - content_filter: if content was omitted due to a flag from our content filters,
        - tool_calls: if the model called a tool
        """
        if finish_reason:
            finish_reason_mapping = {
                "tool_use": "tool_calls",
                "finished": "stop",
                "end_turn": "stop",
                "max_tokens": "length",
                "stop_sequence": "stop",
                "complete": "stop",
                "content_filtered": "content_filter",
                "guardrail_intervened": "content_filter",
                "refusal": "content_filter",
                "model_context_window_exceeded": "length",
            }
            return finish_reason_mapping.get(finish_reason.lower(), finish_reason.lower())
        return None


class BedrockEmbeddingsModel(BaseEmbeddingsModel, ABC):
    accept = "application/json"
    content_type = "application/json"

    def _invoke_model(self, args: dict, model_id: str):
        body = json.dumps(args)
        if DEBUG:
            logger.info("Invoke Bedrock Model: " + model_id)
            logger.info("Bedrock request body: " + body)
        try:
            return bedrock_runtime.invoke_model(
                body=body,
                modelId=model_id,
                accept=self.accept,
                contentType=self.content_type,
            )
        except bedrock_runtime.exceptions.ValidationException as e:
            logger.error("Validation Error: " + str(e))
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(e)
            raise HTTPException(status_code=500, detail=str(e))

    def _create_response(
            self,
            embeddings: list[float],
            model: str,
            input_tokens: int = 0,
            output_tokens: int = 0,
            encoding_format: Literal["float", "base64"] = "float",
    ) -> EmbeddingsResponse:
        data = []
        for i, embedding in enumerate(embeddings):
            if encoding_format == "base64":
                arr = np.array(embedding, dtype=np.float32)
                arr_bytes = arr.tobytes()
                encoded_embedding = base64.b64encode(arr_bytes)
                data.append(Embedding(index=i, embedding=encoded_embedding))
            else:
                data.append(Embedding(index=i, embedding=embedding))
        response = EmbeddingsResponse(
            data=data,
            model=model,
            usage=EmbeddingsUsage(
                prompt_tokens=input_tokens,
                total_tokens=input_tokens + output_tokens,
            ),
        )
        if DEBUG:
            logger.info("Proxy response :" + response.model_dump_json())
        return response


class CohereEmbeddingsModel(BedrockEmbeddingsModel):

    def _parse_args(self, embeddings_request: EmbeddingsRequest) -> dict:
        texts = []
        if isinstance(embeddings_request.input, str):
            texts = [embeddings_request.input]
        elif isinstance(embeddings_request.input, list):
            texts = embeddings_request.input
        elif isinstance(embeddings_request.input, Iterable):
            # For encoded input
            # The workaround is to use tiktoken to decode to get the original text.
            encodings = []
            for inner in embeddings_request.input:
                if isinstance(inner, int):
                    # Iterable[int]
                    encodings.append(inner)
                else:
                    # Iterable[Iterable[int]]
                    text = ENCODER.decode(list(inner))
                    texts.append(text)
            if encodings:
                texts.append(ENCODER.decode(encodings))

        # Maximum of 2048 characters
        args = {
            "texts": texts,
            "input_type": "search_document",
            "truncate": "END",  # "NONE|START|END"
        }
        return args

    def embed(self, embeddings_request: EmbeddingsRequest) -> EmbeddingsResponse:
        response = self._invoke_model(
            args=self._parse_args(embeddings_request), model_id=embeddings_request.model
        )
        response_body = json.loads(response.get("body").read())
        if DEBUG:
            logger.info("Bedrock response body: " + str(response_body))

        return self._create_response(
            embeddings=response_body["embeddings"],
            model=embeddings_request.model,
            encoding_format=embeddings_request.encoding_format,
        )


class TitanEmbeddingsModel(BedrockEmbeddingsModel):

    def _parse_args(self, embeddings_request: EmbeddingsRequest) -> dict:
        if isinstance(embeddings_request.input, str):
            input_text = embeddings_request.input
        elif (
                isinstance(embeddings_request.input, list)
                and len(embeddings_request.input) == 1
        ):
            input_text = embeddings_request.input[0]
        else:
            raise ValueError(
                "Amazon Titan Embeddings models support only single strings as input."
            )
        args = {
            "inputText": input_text,
            # Note: inputImage is not supported!
        }
        if embeddings_request.model == "amazon.titan-embed-image-v1":
            args["embeddingConfig"] = (
                embeddings_request.embedding_config
                if embeddings_request.embedding_config
                else {"outputEmbeddingLength": 1024}
            )
        return args

    def embed(self, embeddings_request: EmbeddingsRequest) -> EmbeddingsResponse:
        response = self._invoke_model(
            args=self._parse_args(embeddings_request), model_id=embeddings_request.model
        )
        response_body = json.loads(response.get("body").read())
        if DEBUG:
            logger.info("Bedrock response body: " + str(response_body))

        return self._create_response(
            embeddings=[response_body["embedding"]],
            model=embeddings_request.model,
            input_tokens=response_body["inputTextTokenCount"],
        )


def get_embeddings_model(model_id: str) -> BedrockEmbeddingsModel:
    model_name = SUPPORTED_BEDROCK_EMBEDDING_MODELS.get(model_id, "")
    if DEBUG:
        logger.info("model name is " + model_name)
    match model_name:
        case "Cohere Embed Multilingual" | "Cohere Embed English":
            return CohereEmbeddingsModel()
        case _:
            logger.error("Unsupported model id " + model_id)
            raise HTTPException(
                status_code=400,
                detail="Unsupported embedding model id " + model_id,
            )