import os
import asyncio
import sys
import time # Keep time if needed for delays, though not strictly required now
from RealtimeSTT import AudioToTextRecorder
from langchain_google_genai import ChatGoogleGenerativeAI
from browser_use import Agent # Assuming browser_use library is installed and configured
# from browser_use import Agent, Browser, BrowserConfig # Uncomment if BrowserConfig is needed
from pydantic import SecretStr
from dotenv import load_dotenv

# --- Configuration ---
load_dotenv()

API_KEY_ENV_VAR_NAME = "GEMINI_API_KEY"
MODEL_NAME = 'gemini-1.5-flash-latest'
# Optional: Specify Chrome path if needed by browser_use
# CHROME_PATH = "C:\Program Files\Google\Chrome\Application\chrome.exe"

# --- Global State ---
is_recording = False
recorder = None # Initialize recorder variable
llm = None      # Initialize LLM variable

# --- Agent Task Function ---
async def run_agent_task(task_description):
    """
    Initializes and runs the browser agent with the given task.
    """
    if not llm:
        print("\nError: LLM not initialized. Cannot run agent.")
        return

    print("\n--- Initializing Agent ---")
   
    try:
        agent = Agent(task=task_description, llm=llm)
        print(f"--- Running Agent with Task: \"{task_description[:100]}...\" ---") # Print truncated task
        result = await agent.run()
        print("\n--- Agent Result ---")
        print(result)
        print("--------------------")

    except Exception as e:
        print("\n--- Error running agent ---")
        import traceback
        traceback.print_exc()
        print("-------------------------")

# --- Main Application Logic ---
def main():
    """
    Main function to handle user commands for recording and agent execution.
    """
    global is_recording, recorder, llm # Declare intent to modify globals

    # --- LLM Setup ---
    api_key_value = os.getenv(API_KEY_ENV_VAR_NAME)
    if not api_key_value:
        print(f"Error: Environment variable '{API_KEY_ENV_VAR_NAME}' not found.")
        return # Exit if no API key

    try:
        llm = ChatGoogleGenerativeAI(
            model=MODEL_NAME,
            api_key=SecretStr(api_key_value)
        )
        print("LLM Initialized.")
    except Exception as e:
        print(f"Error initializing ChatGoogleGenerativeAI: {e}")
        return # Exit if LLM fails to initialize

    # --- Recorder Setup ---
    try:
        print("Initializing recorder...")
        # Initialize the recorder (consider adding specific model, language if needed)
        recorder = AudioToTextRecorder(
            spinner=False, # Disable default spinner
            # model="tiny.en", # Example: Specifying a model
            # language="en",   # Example: Specifying language
        )
        print("Recorder initialized.")
    except Exception as e:
        print(f"Error initializing AudioToTextRecorder: {e}")
        return # Exit if recorder fails

    # --- Command Loop ---
    print("\n--- Controls ---")
    print("Press 's' then Enter to START recording.")
    print("Press 'p' then Enter to STOP recording and run agent with captured speech.")
    print("Press 'q' then Enter to QUIT the program.")
    print("-----------------")

    while True:
        try:
            command = input("Enter command (s/p/q): ").lower().strip()

            if command == 's':
                if not is_recording:
                    try:
                        print(">>> Starting recording... Speak your task clearly.")
                        recorder.start()
                        is_recording = True
                        print("--- Recording ACTIVE --- (Press 'p' then Enter to stop)")
                    except Exception as e:
                        print(f"Error starting recording: {e}")
                        is_recording = False # Ensure state is correct
                else:
                    print(">>> Already recording.")

            elif command == 'p':
                if is_recording:
                    try:
                        print(">>> Stopping recording...")
                        recorder.stop()
                        is_recording = False
                        print("--- Recording STOPPED. Processing text... ---")

                        # Process the recorded audio
                        # This call blocks until transcription is complete
                        transcribed_text = recorder.text()

                        if transcribed_text:
                            print("\n--- Transcribed Text ---")
                            print(f">>> You said: {transcribed_text}")
                            print("------------------------")

                            # --- Run Agent Task ---
                            # Use asyncio.run() to execute the async agent function
                            # from this synchronous part of the code.
                            print(">>> Processing task with Agent...")
                            asyncio.run(run_agent_task(transcribed_text))
                            print(">>> Agent task finished processing.")


                        else:
                            print("--- No text transcribed (perhaps silence or too short). ---")
                            print("--- Agent not run. ---")

                    except Exception as e:
                        print(f"Error stopping, processing, or running agent: {e}")
                        import traceback
                        traceback.print_exc()
                        is_recording = False # Ensure state reflects stop attempt
                else:
                    print(">>> Not currently recording. Press 's' to start.")

            elif command == 'q':
                print(">>> Quitting...")
                if is_recording:
                    try:
                        print("Stopping active recording before quitting...")
                        recorder.stop()
                        is_recording = False
                    except Exception as e:
                        print(f"Error stopping recording during quit: {e}")
                break # Exit the main loop

            else:
                print("Unknown command. Use 's' (start), 'p' (stop/process/run agent), or 'q' (quit).")

        except KeyboardInterrupt:
            print("\nKeyboard interrupt detected.")
            if is_recording and recorder:
                 try:
                     print("Stopping active recording...")
                     recorder.stop()
                     is_recording = False
                 except Exception as e:
                     print(f"Error stopping recording during interrupt: {e}")
            break # Exit loop on Ctrl+C


# --- Entry Point ---
if __name__ == "__main__":
    try:
        main() # Run the main synchronous function
    except Exception as e:
        print(f"\nAn unexpected critical error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Optional: Any other cleanup if needed
        print("\nProgram finished.")