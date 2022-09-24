import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import gensim
import glob
import pickle
import numpy
import torch
from transformers import AutoTokenizer, AutoModel

class BertTextLabel:
    def __init__(self, dataset, if_random = False):
        assert dataset == "MiniImageNet" or dataset == "CUB" or dataset == "TieredImageNet" or dataset == "CIFAR100"
        if dataset == "MiniImageNet":
            label_word_path = "data/miniimagenet/MiniImageNet_cls.txt"
            self.label_dict = self.load_text_label(label_word_path)
        elif dataset == "TieredImageNet":
            label_word_path = "data/tiered_imagenet/words.txt"
            self.label_dict = self.load_text_label(label_word_path)
        else:
            self.label_dict = {}
        self.if_random = if_random
        
        self.tokenizer = AutoTokenizer.from_pretrained("bert_model")
        self.model = AutoModel.from_pretrained("bert_model", output_hidden_states=True).to("cuda:0")

        self.text_dict = {}

    #this is for mini-imagenet dataset
    def load_text_label(self, label_word_path):
        label_word_dict = {}
        with open(label_word_path) as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if "Train" in line or "Val"  in line or "Test" in line:
                    continue

                words = line.strip().split(" ")
                label_word_dict[words[0].strip()] = words[1].strip()
        
        return label_word_dict

    def embed_text(self, text):

        if text in self.text_dict.keys():
            return self.text_dict[text]

        #random for ablation study
        if self.if_random:
            self.text_dict[text] = numpy.random.uniform(-1, 1, 300)
            return self.text_dict[text]

        if self.label_dict:
            text = self.label_dict[text]

        lower_text = text.lower()
        if "_" in text:
            lower_text = lower_text.split("_")
            lower_text = " ".join(lower_text)
        if "," in text:
            lower_text = lower_text.split(",")
            lower_text = " ".join(lower_text)

        tokens = self.tokenizer(lower_text, return_tensors='pt').to("cuda:0")
        with torch.no_grad():
            out = self.model(**tokens)
            feature = out.hidden_states[-1].squeeze()

        self.text_dict[text] = torch.mean(feature, dim = 0).cpu()
        return self.text_dict[text]

if __name__ == "__main__":
    text = BertTextLabel("MiniImageNet")
    text_feature = text.embed_text("n04443257")
    print(text_feature.shape)
