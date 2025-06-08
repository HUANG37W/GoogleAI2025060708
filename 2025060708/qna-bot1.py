from configparser import ConfigParser

# Set up the config parser
config = ConfigParser()
config.read("config.ini")

from langchain_google_genai import ChatGoogleGenerativeAI

llm_gemini = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-lite", google_api_key=config["Gemini"]["API_KEY"]
)

from langchain_community.vectorstores import FAISS
from langchain.embeddings import SentenceTransformerEmbeddings
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
# load the vector store

prebuilt_faiss = FAISS.load_local(
    "faiss_db",
    embedding_function,
    "animal-fun-facts",
    allow_dangerous_deserialization=True
)

prebuilt_faiss.similarity_search_with_score("What is ship of the desert?", 3)
prebuilt_faiss.similarity_search_with_score("What's fun in Taipei city?", 3)

question = "What is ship of the desert?"
# question = "What's fun in Taipei city?"
results = prebuilt_faiss.similarity_search_with_score(question, 1)
if results[0][1] > 1:
    print(f"Q: {question}")
    # print(f"A: 與動物無關")
    messages = [
        ("system", "請用繁體中文回答，不超過50字"),
        ("human", question),
    ]
    response_gemini = llm_gemini.invoke(messages)
    print(f"A: {response_gemini.content}")
else:
    print(f"Q: {question}")
    print(f"A: {results[0][0].page_content}")