import os
import pandas as pd
import json
import base64
from gpt_assistant import ImageChat, AzureService

class FirstProcessing:
    def __init__(self, database_path='database/database.json', emails_path='emails'):
        self.database_path = database_path
        self.emails_path = emails_path

        # Verifica se o arquivo JSON existe e tenta carregá-lo
        if os.path.exists(self.database_path):
            try:
                with open(self.database_path, 'r') as f:
                    self.df = pd.json_normalize(json.load(f))
            except json.JSONDecodeError:
                print("Erro ao decodificar o JSON. O arquivo pode estar corrompido.")
                self.df = pd.DataFrame()  # Cria um DataFrame vazio
        else:
            self.df = pd.DataFrame()  # Cria um DataFrame vazio se o arquivo não existir

        self.existing_files = self.df['File'].unique() if not self.df.empty else []

    @staticmethod
    def image_to_base64(image_path):
        """Converts an image to base64 format."""
        with open(image_path, 'rb') as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return encoded_string

    def process_emails(self):
        """Processes email folders to extract information."""
        data = []
        file_processed = False

        for email_folder in os.listdir(self.emails_path):
            if file_processed:
                break

            folder_path = os.path.join(self.emails_path, email_folder)

            if os.path.isdir(folder_path) and email_folder not in self.existing_files:
                json_file_path = os.path.join(folder_path, 'infos.json')

                if os.path.exists(json_file_path):
                    with open(json_file_path, 'r') as json_file:
                        infos = json.load(json_file)

                    text_file_path = os.path.join(folder_path, 'text.txt')
                    if os.path.exists(text_file_path):
                        with open(text_file_path, 'r') as text_file:
                            text_content = text_file.read()

                        image_path = os.path.join(folder_path, 'img.png')
                        if os.path.exists(image_path):
                            image_helper = ImageChat(image_path)
                            for info in infos:
                                data.append({
                                    'File': email_folder,
                                    'Title': info['title'],
                                    'Date':info['date'],
                                    'Sender': info['sender'],
                                    'Type': '',
                                    'Text Message': text_content,
                                    'Image': self.image_to_base64(image_path),
                                    'Optimized Context': image_helper.read_img(tokens=1000),
                                    'Reference':'',
                                    'Embeddings': ''
                                })
                            file_processed = True

        if data:
            new_df = pd.DataFrame(data)
            self.df = pd.concat([self.df, new_df], ignore_index=True)
            with open(self.database_path, 'w') as f:
                json.dump(self.df.to_dict(orient='records'), f, indent=4)
            print("Data successfully added to the JSON file.")
        else:
            print("No new files to process.")

class SecondProcessing:
    def __init__(self, database_path='database/database.json'):
        self.database_path = database_path
        
        # Carrega o arquivo JSON ou cria um novo se não existir
        if os.path.exists(self.database_path):
            try:
                with open(self.database_path, 'r') as f:
                    self.df = pd.json_normalize(json.load(f))
            except json.JSONDecodeError:
                print("Erro ao decodificar o JSON. O arquivo pode estar corrompido.")
                self.df = pd.DataFrame(columns=['File', 'Title', 'Sender', 'Type', 'Text Message', 'Image', 'Optimized Context', 'Embeddings'])
        else:
            self.df = pd.DataFrame(columns=['File', 'Title', 'Sender', 'Type', 'Text Message', 'Image', 'Optimized Context', 'Embeddings'])
            # Salva um novo arquivo JSON vazio
            with open(self.database_path, 'w') as f:
                json.dump(self.df.to_dict(orient='records'), f, indent=4)
        
        self.helper = AzureService()

    def select_null_type_row(self):
        """Selects the first row where the 'Type' column is null."""
        null_type_row = self.df[self.df['Type'].isnull() | (self.df['Type'] == '')].head(1)
        return null_type_row

    def update_row(self):
        """Updates the 'Optimized Context', 'Type', and 'Embeddings' columns for a single row."""
        null_row = self.select_null_type_row()

        if not null_row.empty:
            index = null_row.index[0]
            row = null_row.iloc[0]
            
            new_context = f"""
                Email Description
                Title: {row['Title']}
                Date: {row['Date']}
                Textual Message: {row["Text Message"]}
                Other informations: {row["Optimized Context"]}
                {self.helper.context_optimization(
                f'Combine the information provided and create a unified output: {row["Text Message"]} and {row["Optimized Context"]}')}
                           """
            
            embedding = self.helper.get_embeddings(new_context)
            
            new_type = self.helper.context_optimization(
                f'Based on this text {new_context}, classify it as "Processes", "Management", or "HR". Return only the Type.'
            )
            
            self.df.at[index, 'Optimized Context'] = new_context
            self.df.at[index, 'Type'] = new_type
            self.df.at[index, 'Embeddings'] = embedding

            with open(self.database_path, 'w') as f:
                json.dump(self.df.to_dict(orient='records'), f, indent=4)
            print("Updated 'Optimized Context', 'Type', and 'Embeddings' for one null 'Type' row.")
        else:
            print("No rows found with null 'Type'.")