## agents will be gemini, and chatgpt backup
import sys
import os


sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv  # pip install python-dotenv

load_dotenv()

# try importing gemini async client
try:
    from google import genai
    from google.genai import types

    class Agent:
        def __init__(self):
            self.client = genai.Client()

        # calls the client using google genai kwargs
        def invoke(self, prompt: str, **kwargs):
            response = self.client.models.generate_content(
                model="gemini-2.5-flash", contents=prompt, **kwargs
            )
            return response.text


except ImportError:
    print("google-genai not installed")
    sys.exit(1)

if __name__ == "__main__":
    agent = Agent()
    response = agent.invoke("Are we in an AI revolution?")
    print(response)
