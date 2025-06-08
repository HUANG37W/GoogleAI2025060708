from configparser import ConfigParser

# Set up the config parser
config = ConfigParser()
config.read("config.ini")

from langchain_google_genai import ChatGoogleGenerativeAI

llm_gemini = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-lite", 
    google_api_key=config["Gemini"]["API_KEY"]
)

user_input = "人生的意義是什麼？"

role_description = """
你是一個哲學家，請用繁體中文回答。
"""

messages = [
    ("system", role_description),
    ("human", user_input),
]

response_gemini = llm_gemini.invoke(messages)

print(f"問 : {user_input}")
print(f"Gemini : {response_gemini.content}")