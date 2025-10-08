# file: server/computer_agent/runner.py

import os
from .agent import BrowserAgent
from .computers import PlaywrightComputer

# Define a constant screen size, or make it configurable
PLAYWRIGHT_SCREEN_SIZE = (1440, 900)

def run_computer_agent_task(query: str) -> dict:
    """
    Initializes and runs the BrowserAgent for a given query.
    This function is blocking and should be run in a separate thread.
    """
    print(f"Computer Agent Task Started for query: '{query}'")
    final_summary = "Task completed, but no final summary was generated."

    # Configure Playwright to run with a visible browser window
    # Ensure PLAYWRIGHT_HEADLESS is not set or is 'false'
    os.environ['PLAYWRIGHT_HEADLESS'] = 'false'

    try:
        # We will use the PlaywrightComputer for local, visible automation
        env = PlaywrightComputer(
            screen_size=PLAYWRIGHT_SCREEN_SIZE,
            initial_url="https://www.google.com",
            highlight_mouse=True, # Make it easy to see what the agent is doing
        )

        with env as browser_computer:
            agent = BrowserAgent(
                browser_computer=browser_computer,
                query=query,
                # Ensure you use the correct model name for the computer use agent
                # model_name='gemini-2.5-computer-use-preview-10-2025',
                model_name='computer-use-exp',
                verbose=True
            )
            agent.agent_loop()
            if agent.final_reasoning:
                final_summary = agent.final_reasoning

        print(f"Computer Agent Task Finished. Summary: {final_summary}")
        return {"status": "success", "summary": final_summary}
    except Exception as e:
        print(f"An error occurred in the computer agent task: {e}")
        return {"status": "error", "summary": str(e)}
