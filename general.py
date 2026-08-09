import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

# Global conversation history
conversation_history = [
    {
        "role": "system",
        "content": "Reply in a detail. Do not use bold, symbols, asterisks, hashtags, or any special formatting characters."
    }
]

def general_questions(question):
    global conversation_history

    conversation_history.append({
        "role": "user",
        "content": question
    })

    system_message = conversation_history[0]
    recent_messages = conversation_history[1:][-20:] 
    conversation_history = [system_message] + recent_messages

    completion = client.chat.completions.create(
        model="openai/gpt-5.2",
        max_tokens=1000,
        messages=conversation_history
    )

    reply = completion.choices[0].message.content.strip()

    conversation_history.append({
        "role": "assistant",
        "content": reply
    })

    return reply
