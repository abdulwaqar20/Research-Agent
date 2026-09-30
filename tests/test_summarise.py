import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.environ["GROQ_API_KEY"])

# paste a chunk of the flash crash text you just fetched
text = """
The flash crash of May 6, 2010, also known as the crash of 2:45 or simply the flash crash,
was a United States trillion-dollar flash crash which started at 2:32 p.m. EDT and lasted
approximately 36 minutes. A variety of explanations have been forwarded for the crash,
including errors in automatic trading systems and deliberate market manipulation.
"""

question = "What caused the 2010 flash crash?"

resp = client.chat.completions.create(
    model="openai/gpt-oss-120b",
    messages=[{
        "role": "user",
        "content": f"Summarise this in 3 bullet points, focused on: {question}\n\n{text}"
    }],
)
print(resp.choices[0].message.content)