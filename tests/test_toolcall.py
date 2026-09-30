import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.environ["GROQ_API_KEY"])


from tools import TOOL_SCHEMAS

resp = client.chat.completions.create(
    model="openai/gpt-oss-120b",
    messages=[{"role": "user", "content": "What caused the 2010 flash crash?"}],
    tools=TOOL_SCHEMAS,
    tool_choice="auto",
)

msg = resp.choices[0].message
print("tool_calls:", msg.tool_calls)