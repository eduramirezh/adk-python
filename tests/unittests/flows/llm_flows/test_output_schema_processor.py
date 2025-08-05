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

"""Tests for output schema processor functionality."""

import json

from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.llm_agent import LlmAgent
from google.adk.agents.run_config import RunConfig
from google.adk.flows.llm_flows.single_flow import SingleFlow
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.tools.function_tool import FunctionTool
from pydantic import BaseModel
from pydantic import Field
import pytest


class PersonSchema(BaseModel):
  """Test schema for structured output."""

  name: str = Field(description="A person's name")
  age: int = Field(description="A person's age")
  city: str = Field(description='The city they live in')


def dummy_tool(query: str) -> str:
  """A dummy tool for testing."""
  return f'Searched for: {query}'


async def _create_invocation_context(agent: LlmAgent) -> InvocationContext:
  """Helper to create InvocationContext for testing."""
  session_service = InMemorySessionService()
  session = await session_service.create_session(
      app_name='test_app', user_id='test_user'
  )
  return InvocationContext(
      invocation_id='test-id',
      agent=agent,
      session=session,
      session_service=session_service,
      run_config=RunConfig(),
  )


@pytest.mark.asyncio
async def test_output_schema_with_tools_validation_removed():
  """Test that LlmAgent now allows output_schema with tools."""
  # This should not raise an error anymore
  agent = LlmAgent(
      name='test_agent',
      model='gemini-1.5-flash',
      output_schema=PersonSchema,
      tools=[FunctionTool(func=dummy_tool)],
  )

  assert agent.output_schema == PersonSchema
  assert len(agent.tools) == 1


@pytest.mark.asyncio
async def test_basic_processor_skips_output_schema_with_tools():
  """Test that basic processor doesn't set output_schema when tools are present."""
  from google.adk.flows.llm_flows.basic import _BasicLlmRequestProcessor

  agent = LlmAgent(
      name='test_agent',
      model='gemini-1.5-flash',
      output_schema=PersonSchema,
      tools=[FunctionTool(func=dummy_tool)],
  )

  invocation_context = await _create_invocation_context(agent)

  llm_request = LlmRequest()
  processor = _BasicLlmRequestProcessor()

  # Process the request
  events = []
  async for event in processor.run_async(invocation_context, llm_request):
    events.append(event)

  # Should not have set response_schema since agent has tools
  assert llm_request.config.response_schema is None
  assert llm_request.config.response_mime_type != 'application/json'


@pytest.mark.asyncio
async def test_basic_processor_sets_output_schema_without_tools():
  """Test that basic processor still sets output_schema when no tools are present."""
  from google.adk.flows.llm_flows.basic import _BasicLlmRequestProcessor

  agent = LlmAgent(
      name='test_agent',
      model='gemini-1.5-flash',
      output_schema=PersonSchema,
      tools=[],  # No tools
  )

  invocation_context = await _create_invocation_context(agent)

  llm_request = LlmRequest()
  processor = _BasicLlmRequestProcessor()

  # Process the request
  events = []
  async for event in processor.run_async(invocation_context, llm_request):
    events.append(event)

  # Should have set response_schema since agent has no tools
  assert llm_request.config.response_schema == PersonSchema
  assert llm_request.config.response_mime_type == 'application/json'


@pytest.mark.asyncio
async def test_output_schema_request_processor():
  """Test that output schema processor adds set_model_response tool."""
  from google.adk.flows.llm_flows._output_schema_processor import _OutputSchemaRequestProcessor

  agent = LlmAgent(
      name='test_agent',
      model='gemini-1.5-flash',
      output_schema=PersonSchema,
      tools=[FunctionTool(func=dummy_tool)],
  )

  invocation_context = await _create_invocation_context(agent)

  llm_request = LlmRequest()
  processor = _OutputSchemaRequestProcessor()

  # Process the request
  events = []
  async for event in processor.run_async(invocation_context, llm_request):
    events.append(event)

  # Should have added set_model_response tool
  assert 'set_model_response' in llm_request.tools_dict

  # Should have added instruction about using set_model_response
  assert 'set_model_response' in llm_request.config.system_instruction


@pytest.mark.asyncio
async def test_set_model_response_tool():
  """Test the set_model_response tool functionality."""
  from google.adk.tools.set_model_response_tool import SetModelResponseTool
  from google.adk.tools.tool_context import ToolContext

  tool = SetModelResponseTool(PersonSchema)

  agent = LlmAgent(name='test_agent', model='gemini-1.5-flash')
  invocation_context = await _create_invocation_context(agent)
  tool_context = ToolContext(invocation_context)

  # Call the tool with valid data
  result = await tool.run_async(
      args={'name': 'John Doe', 'age': 30, 'city': 'New York'},
      tool_context=tool_context,
  )

  assert result == 'Response set successfully.'

  # Check that the response was stored in session state
  stored_response = invocation_context.session.state.get(
      'temp:__adk_model_response__'
  )
  assert stored_response is not None

  # Parse and validate the stored response
  parsed_response = json.loads(stored_response)
  assert parsed_response['name'] == 'John Doe'
  assert parsed_response['age'] == 30
  assert parsed_response['city'] == 'New York'


@pytest.mark.asyncio
async def test_output_schema_response_processor():
  """Test that output schema response processor extracts structured response."""
  from google.adk.flows.llm_flows._output_schema_processor import _OutputSchemaResponseProcessor
  from google.genai import types

  agent = LlmAgent(
      name='test_agent',
      model='gemini-1.5-flash',
      output_schema=PersonSchema,
      tools=[FunctionTool(func=dummy_tool)],
  )

  invocation_context = await _create_invocation_context(agent)

  # Simulate that set_model_response tool was called
  test_response = {'name': 'Jane Smith', 'age': 25, 'city': 'Los Angeles'}
  invocation_context.session.state['temp:__adk_model_response__'] = json.dumps(
      test_response
  )

  # Create a dummy LLM response
  llm_response = LlmResponse(
      content=types.Content(
          role='model', parts=[types.Part(text='Original response')]
      )
  )

  processor = _OutputSchemaResponseProcessor()

  # Process the response
  events = []
  async for event in processor.run_async(invocation_context, llm_response):
    events.append(event)

  # Should have replaced the content with structured response
  assert llm_response.content.parts[0].text == json.dumps(test_response)

  # Should have cleared the special marker
  assert 'temp:__adk_model_response__' not in invocation_context.session.state


@pytest.mark.asyncio
async def test_end_to_end_integration():
  """Test the complete output schema with tools integration."""
  agent = LlmAgent(
      name='test_agent',
      model='gemini-1.5-flash',
      output_schema=PersonSchema,
      tools=[FunctionTool(func=dummy_tool)],
  )

  invocation_context = await _create_invocation_context(agent)

  # Create a flow and test the processors
  flow = SingleFlow()
  llm_request = LlmRequest()

  # Run all request processors
  async for event in flow._preprocess_async(invocation_context, llm_request):
    pass

  # Verify set_model_response tool was added
  assert 'set_model_response' in llm_request.tools_dict

  # Verify instruction was added
  assert 'set_model_response' in llm_request.config.system_instruction

  # Verify output_schema was NOT set on the model config
  assert llm_request.config.response_schema is None
