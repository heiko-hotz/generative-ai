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

"""
Tool execution and handling for Gemini Multimodal Live Proxy Server
"""

# Replace the entire contents of server/core/tool_handler.py with this:

import logging
import aiohttp
import asyncio
from typing import Dict, Any, Optional
from config.config import CLOUD_FUNCTIONS
from urllib.parse import urlencode

from computer_agent.runner import agent_stop_event, run_computer_agent_task

logger = logging.getLogger(__name__)

# --- Global State for the Background Task ---
background_task = None

async def run_and_log_task(coro, query: str, genai_session: Any):
    """
    A wrapper to run the agent task, send its final summary back to the main
    conversation, and then clean up.
    """
    global background_task
    try:
        logger.info(f"Background task for '{query}' is now running.")
        result = await coro
        logger.info(f"Background task for '{query}' finished with result: {result}")

        if result and result.get("status") == "success":
            summary = result.get("summary", "The task is complete.")
            logger.info(f"Sending final summary to main assistant: '{summary}'")
            await genai_session.send(input=summary, end_of_turn=True)
            logger.info("Final summary sent.")

    except asyncio.CancelledError:
        logger.warning(f"Background task for '{query}' was cancelled.")
        await genai_session.send(input="The task has been cancelled.", end_of_turn=True)
    except Exception as e:
        logger.error(f"Background task for '{query}' failed with an exception: {e}")
        await genai_session.send(input="An error occurred during the task.", end_of_turn=True)
    finally:
        logger.info(f"Cleaning up background task for '{query}'.")
        background_task = None

async def execute_tool(tool_name: str, params: Dict[str, Any], genai_session: Optional[Any] = None) -> Dict[str, Any]:
    """Execute a tool based on name and parameters."""
    global background_task

    if tool_name == "stop_computer_task":
        logger.info("Received request to stop computer agent.")
        if background_task and not background_task.done():
            agent_stop_event.set()
            background_task.cancel()
            return {"status": "success", "summary": "Stop signal sent."}
        else:
            return {"status": "no_task", "summary": "No task is running."}

    if tool_name == "execute_computer_task":
        if background_task and not background_task.done():
            return {"status": "already_running", "summary": "A task is in progress."}

        query = params.get("query")
        if not query: return {"error": "Missing query"}
        if not genai_session: return {"error": "Missing session"}

        logger.info(f"Scheduling computer task for query: {query}")
        loop = asyncio.get_running_loop()
        executor_coro = loop.run_in_executor(None, run_computer_agent_task, query)

        background_task = asyncio.create_task(
            run_and_log_task(executor_coro, query, genai_session)
        )

        return {"status": "started", "summary": f"Task '{query}' has started."}

    try:
        if tool_name not in CLOUD_FUNCTIONS:
            logger.error(f"Tool not found: {tool_name}")
            return {"error": f"Unknown tool: {tool_name}"}

        base_url = CLOUD_FUNCTIONS[tool_name]
        # Convert params to URL query parameters
        query_string = urlencode(params)
        function_url = f"{base_url}?{query_string}" if params else base_url

        logger.debug(f"Calling cloud function for {tool_name}")
        logger.debug(f"URL with params: {function_url}")

        async with aiohttp.ClientSession() as session:
            async with session.get(function_url) as response:
                response_text = await response.text()
                logger.debug(f"Response status: {response.status}")
                logger.debug(f"Response headers: {dict(response.headers)}")
                logger.debug(f"Response body: {response_text}")

                if response.status != 200:
                    logger.error(f"Cloud function error: {response_text}")
                    return {
                        "error": f"Cloud function returned status {response.status}"
                    }

                try:
                    return await response.json()
                except Exception as e:
                    logger.error(f"Failed to parse JSON response: {response_text}")
                    return {
                        "error": f"Invalid JSON response from cloud function: {str(e)}"
                    }

    except aiohttp.ClientError as e:
        logger.error(f"Network error calling cloud function for {tool_name}: {str(e)}")
        return {"error": f"Failed to call cloud function: {str(e)}"}
    except Exception as e:
        logger.error(f"Error executing tool {tool_name}: {str(e)}")
        return {"error": f"Tool execution failed: {str(e)}"}
