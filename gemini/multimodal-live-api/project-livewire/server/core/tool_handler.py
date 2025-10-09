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

import logging
from typing import Any, Dict
from urllib.parse import urlencode

import aiohttp
import asyncio
from config.config import CLOUD_FUNCTIONS

background_task = None


logger = logging.getLogger(__name__)

async def run_and_log_task(coro, query):
    """
    A wrapper to run the agent task, log the result, and clean up the global task variable.
    """
    global background_task
    try:
        logger.info(f"Background task for '{query}' is now running.")
        result = await coro
        logger.info(f"Background task for '{query}' finished with result: {result}")
    except asyncio.CancelledError:
        logger.warning(f"Background task for '{query}' was cancelled.")
    except Exception as e:
        logger.error(f"Background task for '{query}' failed with an exception: {e}")
    finally:
        # Clean up the global reference so a new task can be started.
        logger.info(f"Cleaning up background task for '{query}'.")
        background_task = None


async def execute_tool(tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a tool based on name and parameters by calling the corresponding cloud function"""

    global background_task

    if tool_name == "stop_computer_task":
        from computer_agent.runner import agent_stop_event
        logger.info("Received request to stop computer agent.")
        if background_task and not background_task.done():
            agent_stop_event.set()
            background_task.cancel()
            # The background_task variable will be set to None by the finally block in run_and_log_task
            return {"status": "success", "summary": "Stop signal sent to the running task."}
        else:
            return {"status": "no_task", "summary": "There is no computer task currently running."}


    # Handle the new local computer task
    if tool_name == "execute_computer_task":
        from computer_agent.runner import run_computer_agent_task
        if background_task and not background_task.done():
            return {"status": "already_running", "summary": "A computer task is already in progress. Please stop it before starting a new one."}

        query = params.get("query")
        if not query:
            return {"error": "Missing query parameter"}

        logger.info(f"Scheduling computer task in the background for query: {query}")

        loop = asyncio.get_running_loop()

        # Create a coroutine for the executor task
        executor_coro = loop.run_in_executor(None, run_computer_agent_task, query)

        # Create the background task using our wrapper
        background_task = asyncio.create_task(
            run_and_log_task(executor_coro, query)
        )

        # Immediately return confirmation. This unblocks the main assistant.
        return {"status": "started", "summary": f"The computer task '{query}' has been started in the background."}

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
