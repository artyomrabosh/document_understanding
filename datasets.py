from torch.utils.data import Dataset, DataLoader, Sampler
from transformers import RobertaTokenizerFast
import os




def pdf_to_tokens(path: str) -> list:
    pass


class DocBankNoImageDataset(Dataset):

    @staticmethod
    def get_vocab(labels: list[str]) -> tuple[dict]:

        labels_to_set = []
        for page in labels:
            labels_to_set += page
        labels = set(labels_to_set)
        label2id = {}
        id2label = {}

        for i, label in enumerate(labels):
            label2id[label] = i
            id2label[i] = label
        
        return label2id, id2label
    
    def __init__(self, data_dir: str, tokenizer: RobertaTokenizerFast=None):
        self.data = []
        self.data_dir = data_dir
        assert os.path.isdir(self.data_dir)

        for path in os.listdir(self.data_dir)[:100]:
            if os.path.isfile(os.path.join(self.data_dir, path)):
                self.data.append(os.path.join(self.data_dir, path))

        words = []
        bboxes = []
        labels = []
        
        for txt in self.data:
            with open(txt) as f:
                lines = f.readlines()
                data = [line.split('\t') for line in lines]
                line_words = [line[0] for line in data]
                line_bboxes = [line[1:5] for line in data]
                line_labels = [line[-1] for line in data]
                words.append(line_words)
                bboxes.append(line_bboxes)
                labels.append(line_labels)
        
        self.label2id, self.id2label = DocBankNoImageDataset.get_vocab(labels)
        
        if tokenizer is None:
            tokenizer = RobertaTokenizerFast.from_pretrained("FacebookAI/roberta-base")

        self.tokens = [[tokenizer.encode(word) for word in page] for page in words]
        self.bboxes = bboxes
        self.labels = [[self.label2id[label] for label in page] for page in labels]

    
        
    def __len__(self) -> int:
        return len(self.tokens)
    
    def __getitem__(self, idx: int):
        return self.tokens[idx], self.bboxes[idx], self.labels[idx]

def collate_fn_docbank(tokens, bboxes, labels):
    pass

class SpbuDataset(Dataset):

    def __init__(self, data_dir):
        pass

    def __len__(self):
        pass
    
    def __getitem__(self, idx):
        pass
