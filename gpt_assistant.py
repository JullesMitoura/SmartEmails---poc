import base64
import requests
import os
from langchain_openai.embeddings import AzureOpenAIEmbeddings
from langchain_openai.chat_models import AzureChatOpenAI

class AzureService:
    def __init__(self):
        self.llm = AzureChatOpenAI(openai_api_version="2023-07-01-preview",
                               azure_endpoint=os.getenv("OPENAI_URL"),
                               openai_api_key=os.getenv("OPEN_AI_KEY"),
                               azure_deployment=os.getenv("OPENAI_DEPLOY"),
                               temperature=0)
        self.embedding = AzureOpenAIEmbeddings(api_key = os.getenv("OPEN_AI_KEY"),
                                     api_version="2024-02-01",
                                     deployment=os.getenv("EMBEDDING_DEPLOY"),
                                     azure_endpoint=os.getenv("OPENAI_URL"))

    def context_optimization(self, prompt):
        messages = [{"role": "user", "content": prompt}]
        response = self.llm.invoke(messages, max_tokens=500, temperature=0.7)
        return response.content

    def get_embeddings(self, text):
        return self.embedding.embed_query(text.replace('\n', ' '))
    

class ImageChat:
    def __init__(self, image_path):
        self.image_path = image_path
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.model = "gpt-4o-mini"

    def encode_image(self):
        with open(self.image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
        
    def read_img(self, tokens):
        image = self.encode_image()
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "You are an image analyzer. You will receive images that are reports or technical documents and you need to return in words what the image contains. Return only the information described in the image."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": tokens
        }
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        try:
            return response.json()['choices'][0]['message']['content']
        except ValueError as e:
            print("Error decoding JSON:", e)
            return None