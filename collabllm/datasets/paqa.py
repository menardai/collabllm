import os.path as osp
import pandas as pd
from collabllm.datasets.dataset import ChatDataset


class PAQA(ChatDataset):

    def __init__(self, root):
        """
        Initializes the PAQA dataset with raw data.

        Parameters:
       root (dict): The raw PAQA data to be processed.
        """
        raw_data = {}
        for split in ['train', 'dev', 'test']:
            raw_data_path = osp.join(root, f'paqa/{split}.csv')
            raw_data[split] = pd.read_csv(raw_data_path)
        processed_data = self.preprocess(raw_data)
        super().__init__(processed_data)

    def preprocess(self, raw_data):
        """
        Processes the raw PAQA data into the format expected by ChatDataset.

        Parameters:
        raw_data (dict): The raw PAQA data to be processed.

        Returns:
        list: A list of processed chats with metadata.
        """
        processed_data = []

        for set_type, csv in raw_data.items():
            for idx, row in csv.iterrows():
                # row to dictionary
                row = row.to_dict()
                metadata = {
                    'split': set_type,
                    'ambiguity': bool(row['ambig'])
                }
                if metadata['ambiguity']:
                    answers = eval(row['answer'])
                    choices = [chr(i + 65) for i in range(len(answers))]
                    for choice, answer in zip(choices, answers):
                        turns = [
                            {'role': 'user', 'content': row['question']},
                            {'role': 'assistant', 'content': row['clar'] + f' Please choose from {choices}'},
                            {'role': 'user', 'content': choice},
                            {'role': 'assistant', 'content': answer}
                        ]
                        processed_data.append({"metadata": metadata, "chat": turns})
                else:
                    turns = [
                        {'role': 'user', 'content': row['question']},
                        {'role': 'user', 'content': row['answer']}
                    ]
                    processed_data.append({"metadata": metadata, "chat": turns})

        return processed_data