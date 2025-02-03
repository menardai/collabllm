from datasets import load_dataset
from collabllm.datasets.dataset import ChatDataset


class PPC(ChatDataset):
    hf_repo = 'erbacher/personalized-collabllmive-conversations'
    
    def __init__(self):
        """
        Initializes the PPC dataset with raw data.

        Parameters:
        raw_data (dict): The raw PPC data to be processed.
        """
        raw_data = load_dataset(self.hf_repo)
        processed_data = self.preprocess(raw_data)
        super().__init__(processed_data)

    def preprocess(self, raw_data):
        """
        Processes the raw PPC data into the format expected by ChatDataset.

        Parameters:
        raw_data (dict): The raw PPC data to be processed.

        Returns:
        list: A list of processed chats with metadata.
        """
        processed_data = []

        for set_type in raw_data.keys():
            for entry in raw_data[set_type]:
                metadata = {
                    'split': set_type
                }
                turns = []
                for conv in entry["conversation"]:
                    turns.append({'role': 'user', 'content':  conv['user']})
                    turns.append({'role': 'assistant', 'content': conv['assistant']})
                
                turns[0]['content'] = entry.get('user') + '\n' + turns[0]['content']
                processed_data.append({"metadata": metadata, "chat": turns})

        return processed_data