from langchain_google_genai import ChatGoogleGenerativeAI
from browser_use import Agent, Browser, BrowserConfig
from pydantic import SecretStr
import os
from dotenv import load_dotenv
import asyncio 
load_dotenv()
API_KEY_ENV_VAR_NAME = "GEMINI_API_KEY"
api_key_value = os.getenv(API_KEY_ENV_VAR_NAME)

if not api_key_value:
    raise ValueError(f"Environment variable '{API_KEY_ENV_VAR_NAME}' not found.")

MODEL_NAME = 'gemini-1.5-flash-latest'
try:
    llm = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        api_key=SecretStr(api_key_value)
    )
except Exception as e:
    raise RuntimeError(f"Error initializing ChatGoogleGenerativeAI: {e}")

# Define the main asynchronous function 
async def main():

    # browser = Browser(
    #     config=BrowserConfig(
    #     # Specify the path to your Chrome executable
    #         browser_binary_path="C:\Program Files\Google\Chrome\Application\chrome.exe",
    #     )
    # )

    agent = Agent(
        task='''"Hi! Please search the web for 3 to 5 of the most important AI news stories from the last 24 hours. Make sure they're from good, trustworthy websites. 
        For each story, just write a short sentence or two saying what it's about and why it matters, and give me the link to the actual article. 
        Please put all this together like it's the body of an email I can send out, starting with something like 'Here's your AI news update:'" ''',
        llm=llm
    )
    try:
        result = await agent.run()

    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())