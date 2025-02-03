import os
import os
import subprocess
import json
import random
from collabllm.datasets.dataset import ChatDataset


class MTBP(ChatDataset):

    def __init__(self, 
                 repo_url='https://github.com/salesforce/CodeGen.git',
                 file_path='codegen1/benchmark/mtpb.jsonl',
                 train_ratio=0.8):
        """
        Initializes the MTBP dataset by cloning the repository and loading the specified file.

        Parameters:
        repo_url (str): The URL of the GitHub repository to clone.
        file_path (str): The path to the file within the cloned repository.
        """
        self.repo_url = repo_url
        self.file_path = file_path
        self.repo_dir = 'data'
        self.train_ratio = train_ratio

        # Clone the repository if it doesn't already exist
        if not os.path.exists(self.repo_dir):
            self.clone_repo()

        # Check if the file exists after cloning
        full_file_path = os.path.join(self.repo_dir, self.file_path)
        if not os.path.exists(full_file_path):
            raise FileNotFoundError(f'File not found: {full_file_path}')
        
        # Load the data from the file
        data = self.load_data(full_file_path)
        processed_data = self.preprocess(data)
        super().__init__(processed_data)

    def preprocess(self, raw_data):
        """
        Processes the raw MATH data into the format expected by ChatDataset.

        Parameters:
        raw_data (dict): The raw MATH data to be processed.

        Returns:
        list: A list of processed chats with metadata.
        """
        
        processed_data = []
        splits = ['train'] * int(len(raw_data) * self.train_ratio) + \
                 ['test'] * (len(raw_data) - int(len(raw_data) * self.train_ratio))
        random.seed(42)
        random.shuffle(splits)
        for entry, split in zip(raw_data, splits):
            metadata = {
                    'split': split,
                    'id': entry.get('id'),
                    'category': entry.get('category'),
                    'name': entry.get('name')
                }
            query = 'I need a Python function with description: ' + entry.get('description') + \
                    'The function has the following steps: ' + \
                    '\n'.join([f'({i+1}) {step}' for i, step in enumerate(entry.get('prompts'))])
            
            turns = [
                {'role': 'user', 'content': query},
                {'role': 'assistant', 'content': str({'inputs': entry.get('inputs'), 
                                                      'outputs': entry.get('outputs')})}
            ]
            processed_data.append({"metadata": metadata, "chat": turns})

        return processed_data
    
    def clone_repo(self):
        """
        Clones the repository from the provided URL.
        """
        print(f'Cloning repository from {self.repo_url}...')
        subprocess.run(['git', 'clone', self.repo_url, self.repo_dir], check=True)
        print('Repository cloned successfully.')

    def load_data(self, file_path):
        """
        Load the data from the specified file.

        Parameters:
        file_path (str): The path to the file to load.

        Returns:
        data: The loaded data from the file.
        """
        with open(file_path, 'r') as file:
            data = [json.loads(line) for line in file]
        print(f'Data loaded successfully from {file_path}')
        return data
