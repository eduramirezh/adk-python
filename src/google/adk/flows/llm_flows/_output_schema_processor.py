# Copyright 2025 Google LLC
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

"""Handles output schema when tools are also present."""

from __future__ import annotations

from typing import AsyncGenerator

from typing_extensions import override

from ...agents.invocation_context import InvocationContext
from ...events.event import Event
from ...models.llm_request import LlmRequest
from ...models.llm_response import LlmResponse
from ...tools.set_model_response_tool import MODEL_JSON_RESPONSE_KEY
from ...tools.set_model_response_tool import SetModelResponseTool
from ._base_llm_processor import BaseLlmRequestProcessor
from ._base_llm_processor import BaseLlmResponseProcessor


class _OutputSchemaRequestProcessor(BaseLlmRequestProcessor):
  """Processor that handles output schema for agents with tools."""

  @override
  async def run_async(
      self, invocation_context: InvocationContext, llm_request: LlmRequest
  ) -> AsyncGenerator[Event, None]:
    from ...agents.llm_agent import LlmAgent

    agent = invocation_context.agent
    if not isinstance(agent, LlmAgent):
      return

    # Check if we need the processor: output_schema + tools
    if not agent.output_schema or not agent.tools:
      return

    # Add the set_model_response tool to handle structured output
    set_response_tool = SetModelResponseTool(agent.output_schema)
    llm_request.append_tools([set_response_tool])

    # Add instruction about using the set_model_response tool
    instruction = (
        'IMPORTANT: You have access to other tools, but you must provide '
        'your final response using the set_model_response tool with the '
        'required structured format. After using any other tools needed '
        'to complete the task, always call set_model_response with your '
        'final answer in the specified schema format.'
    )
    llm_request.append_instructions([instruction])

    return
    yield  # Generator requires yield statement in function body.


class _OutputSchemaResponseProcessor(BaseLlmResponseProcessor):
  """Processor that extracts structured response from set_model_response tool calls."""

  @override
  async def run_async(
      self, invocation_context: InvocationContext, llm_response: LlmResponse
  ) -> AsyncGenerator[Event, None]:
    from ...agents.llm_agent import LlmAgent

    agent = invocation_context.agent
    if not isinstance(agent, LlmAgent):
      return

    # Check if we need the processor: output_schema + tools
    if not agent.output_schema or not agent.tools:
      return

    # Check if the model response was set via our special tool
    model_response_json = invocation_context.session.state.get(
        MODEL_JSON_RESPONSE_KEY
    )
    if model_response_json:
      # Replace the response content with the structured response
      from google.genai import types

      llm_response.content = types.Content(
          role='model', parts=[types.Part(text=model_response_json)]
      )

      # Clear the special key
      del invocation_context.session.state[MODEL_JSON_RESPONSE_KEY]

    return
    yield  # Generator requires yield statement in function body.


# Export the processors
request_processor = _OutputSchemaRequestProcessor()
response_processor = _OutputSchemaResponseProcessor()
