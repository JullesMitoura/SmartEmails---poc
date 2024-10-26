from gpt_assistant import AzureService
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np

class Chat:
    def __init__(self, prompt: str):
        self.prompt = prompt
        self.helper = AzureService()

    def find_similar_embeddings(self, json_file: str = 'database/database.json', similarity_threshold: float = 0.74) -> list:
        """Find indices of similar embeddings based on the prompt."""
        try:
            df = pd.read_json(json_file, orient='records')
        except ValueError as e:
            raise ValueError(f"Error reading JSON file: {e}")

        if 'Embeddings' not in df.columns:
            raise ValueError("The 'Embeddings' column is missing from the JSON file.")

        prompt_embedding = self.helper.get_embeddings(self.prompt)

        similar_idx = [
            index for index, row in df.iterrows()
            if cosine_similarity([prompt_embedding], [np.array(row['Embeddings'])])[0][0] > similarity_threshold
        ]
        
        return similar_idx

    def get_similar_rows(self, json_file: str = 'database/database.json', indices: list = None) -> tuple:
        """Retrieve similar rows from the DataFrame based on indices."""
        if indices is None or not indices:
            return '', pd.Series(dtype='object')

        try:
            df = pd.read_json(json_file, orient='records')
        except ValueError as e:
            raise ValueError(f"Error reading JSON file: {e}")

        similar_rows = df.iloc[indices]
        contexts = " ".join(similar_rows["Optimized Context"].tolist())
        
        return contexts, similar_rows['Image']

    def check_relevance(self, contexts: str) -> bool:
        """Check if the similar contexts are relevant to the user's question."""
        combined_context_embedding = self.helper.get_embeddings(contexts)
        prompt_embedding = self.helper.get_embeddings(self.prompt)
        relevance_score = cosine_similarity([prompt_embedding], [combined_context_embedding])[0][0]
        return relevance_score > 0.7  # Set a threshold for relevance

    def openai_request(self) -> str:
        """Prepare and send the request to the Azure service."""
        idx = self.find_similar_embeddings()
        similar_contexts, img = self.get_similar_rows(indices=idx)

        if not idx:
            final_prompt = f"""Your name is Celia, a consultation assistant at Celanese. Please answer the following user question: {self.prompt}"""
        else:
            if self.check_relevance(similar_contexts):
                final_prompt = f"""
                    User's Question: {self.prompt}
                    Based on the following information, provide an answer: {similar_contexts}.
                """
            else:
                final_prompt = f"""Your name is Celia, a consultation assistant at Celanese. Please answer the following user question: {self.prompt}"""
        
        return self.helper.context_optimization(prompt=final_prompt)

    def get_img(self) -> list:
        """Retrieve images related to similar rows."""
        idx = self.find_similar_embeddings()
        if not idx:
            return []
        
        _, images = self.get_similar_rows(indices=idx)
        return images.tolist() if not images.empty else []