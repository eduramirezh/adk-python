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

"""Tests for LlmRequest functionality."""

from google.adk.models.llm_request import LlmRequest
from google.adk.tools.function_tool import FunctionTool
from google.genai import types


def dummy_tool(query: str) -> str:
  """A dummy tool for testing."""
  return f'Searched for: {query}'


class TestLlmRequest:
  """Test class for LlmRequest."""

  def test_append_tools_with_none_config_tools(self):
    """Test that append_tools initializes config.tools when it's None."""
    request = LlmRequest()

    # Initially config.tools should be None
    assert request.config.tools is None

    # Create a tool to append
    tool = FunctionTool(func=dummy_tool)

    # This should not raise an AttributeError
    request.append_tools([tool])

    # Now config.tools should be initialized and contain the tool
    assert request.config.tools is not None
    assert len(request.config.tools) == 1
    assert len(request.config.tools[0].function_declarations) == 1
    assert request.config.tools[0].function_declarations[0].name == 'dummy_tool'

    # Tool should also be in tools_dict
    assert 'dummy_tool' in request.tools_dict
    assert request.tools_dict['dummy_tool'] == tool

  def test_append_tools_with_existing_tools(self):
    """Test that append_tools works correctly when config.tools already exists."""
    request = LlmRequest()

    # Pre-initialize config.tools with an existing tool
    existing_declaration = types.FunctionDeclaration(
        name='existing_tool', description='An existing tool', parameters={}
    )
    request.config.tools = [
        types.Tool(function_declarations=[existing_declaration])
    ]

    # Create a new tool to append
    tool = FunctionTool(func=dummy_tool)

    # Append the new tool
    request.append_tools([tool])

    # Should now have 2 tools
    assert len(request.config.tools) == 2
    assert (
        request.config.tools[0].function_declarations[0].name == 'existing_tool'
    )
    assert request.config.tools[1].function_declarations[0].name == 'dummy_tool'

  def test_append_tools_empty_list(self):
    """Test that append_tools handles empty list correctly."""
    request = LlmRequest()

    # This should not modify anything
    request.append_tools([])

    # config.tools should still be None
    assert request.config.tools is None
    assert len(request.tools_dict) == 0

  def test_append_tools_tool_with_no_declaration(self):
    """Test append_tools with a BaseTool that returns None from _get_declaration."""
    from google.adk.tools.base_tool import BaseTool

    request = LlmRequest()

    # Create a mock tool that inherits from BaseTool and returns None for declaration
    class NoDeclarationTool(BaseTool):

      def __init__(self):
        super().__init__(
            name='no_decl_tool', description='A tool with no declaration'
        )

      def _get_declaration(self):
        return None

    tool = NoDeclarationTool()

    # This should not add anything to config.tools but should handle gracefully
    request.append_tools([tool])

    # config.tools should still be None since no declarations were added
    assert request.config.tools is None
    # tools_dict should be empty since no valid declaration
    assert len(request.tools_dict) == 0
