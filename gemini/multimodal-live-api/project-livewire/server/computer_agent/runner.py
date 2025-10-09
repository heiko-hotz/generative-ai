# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import threading
from .agent import BrowserAgent
from .computers.playwright.playwright import PlaywrightComputer

# This variable will hold our single, persistent browser instance.
browser_env = None
# This Event will act as our interrupt signal for the agent's thread.
agent_stop_event = threading.Event()

def run_computer_agent_task(query: str) -> dict:
    """
    Uses a global browser instance to run a BrowserAgent task.
    The browser will remain open after the task is complete.
    """
    global browser_env 

    print(f"Computer Agent Task Started for query: '{query}'")
    final_summary = "Task completed, but no final summary was generated."

    try:
        agent_stop_event.clear()
        # If the browser isn't open yet, create and initialize it.
        if browser_env is None:
            print("No existing browser found. Creating a new persistent instance...")
            USER_DATA_DIR = os.path.join(os.path.dirname(__file__), 'playwright_user_data')
            os.makedirs(USER_DATA_DIR, exist_ok=True)

            browser_env = PlaywrightComputer(
                screen_size=(1440, 900),
                initial_url="https://www.google.com",
                highlight_mouse=True,
                persistent_user_data_dir=USER_DATA_DIR,
            )
            # Manually "enter" the context to launch the browser.
            # It will stay open because we will not call __exit__.
            browser_env.__enter__()
            print("New browser instance created.")
        else:
            print("Reusing existing browser instance.")

        # Now, `browser_env` is guaranteed to be our active browser computer.
        agent = BrowserAgent(
            browser_computer=browser_env,
            query=query,
            model_name='computer-use-exp',
            verbose=True,
            stop_event=agent_stop_event
        )
        agent.agent_loop()

        if agent.final_reasoning:
            final_summary = agent.final_reasoning

        print(f"Computer Agent Task Finished. Summary: {final_summary}")
        # Bring the browser window to the front for the user.
        browser_env._page.bring_to_front()

        return {"status": "success", "summary": final_summary}

    except Exception as e:
        print(f"An error occurred in the computer agent task: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "summary": str(e)}

    finally:
        print("Clearing stop event flag.")
        agent_stop_event.clear()
