from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from IPython.display import Image, display
from configparser import ConfigParser
import base64

config = ConfigParser()
config.read("config.ini")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=config["Gemini"]["API_KEY"],
    max_tokens=8192,
)


# def image4LangChain(image_url):
#     if "http" in image_url:
#         return {"url": image_url}
#     else:
#         with open(image_url, "rb") as image_file:
#             image_data = base64.b64encode(image_file.read()).decode("utf-8")
#         return {"url": f"data:image/jpeg;base64,{image_data}"}
def image4LangChain(image_url):
    if "http" in image_url:
        return image_url
    else:
        with open(image_url, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode("utf-8")
        return f"data:image/jpeg;base64,{image_data}"


user_messages = []
# append user input question
# user_input = "這兩隻哪一隻比較好養？為什麼。"
user_input = "請給我這張截圖的完整的可執行的HTML程式碼，並且包含CSS和JavaScript，讓我可以直接在瀏覽器中打開。"
user_messages.append({"type": "text", "text": user_input})

# append images
# image_url = "https://i.ibb.co/KyNtMw5/IMG-20240321-172354614-AE.jpg"
# image_url = "https://images.squarespace-cdn.com/content/v1/607f89e638219e13eee71b1e/1684821560422-SD5V37BAG28BURTLIXUQ/michael-sum-LEpfefQf4rU-unsplash.jpg"
# image_url = "dog.jpg"
# image_url = "https://knect365.imgix.net/uploads/bd64cf28-498e-44df-be77-3e74cd079783-featured-49f3a3c3ecd5bdb5a10037c1b80de2ff.jpg"

image_url = "web.png"


user_messages.append({"type": "image_url", "image_url": image4LangChain(image_url)})

# image_url_2 = "dog.jpg"
# user_messages.append({"type": "image_url", "image_url": image4LangChain(image_url_2)})

human_messages = HumanMessage(content=user_messages)
result = llm.invoke([human_messages])

print("Q: " + user_input)
print("A: " + result.content)

# Display the image
display(Image(url=image_url))
# display(Image(url=image_url_2))