import json
import os.path as osp
from collabllm.datasets.dataset import ChatDataset


class ASQA(ChatDataset):

    def __init__(self, root):
        """
        Initializes the ASQA dataset with raw data.

        Parameters:
        raw_data (dict): The raw ASQA data to be processed.
        """
        raw_data = json.load(open(osp.join(root, 'asqa/ASQA.json'), 'r'))
        processed_data = self.preprocess(raw_data)
        super().__init__(processed_data)

    def preprocess(self, raw_data):
        """
        Processes the raw ASQA data into the format expected by ChatDataset.

        Parameters:
        raw_data (dict): The raw ASQA data to be processed.

        Returns:
        list: A list of processed chats with metadata.
        """
        processed_data = []

        for set_type, entries in raw_data.items():
            for key, entry in entries.items():
                qa_pairs = entry.get("qa_pairs", [])
                for qa in qa_pairs:
                    metadata = {
                        'split': set_type,
                        'wikipage': qa.get('wikipage'),
                        'ambiguity': True
                    }
                    turns = [
                        {'role': 'user', 'content': 'Context:\n' + qa.get('context') + '\n\n' + qa['question']},
                        {'role': 'assistant', 'content': ';'.join(qa['short_answers'])}
                    ]
                    processed_data.append({"metadata": metadata, "chat": turns})

        return processed_data