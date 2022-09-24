import gensim
import glob
import pickle
import numpy
from gensim.scripts.glove2word2vec import glove2word2vec

weight_path = "glove.word2vec"

class GloveTextLabel:
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
        if "glove" in weight_path and not if_random:
            self.model = gensim.models.KeyedVectors.load_word2vec_format(weight_path, binary=False)
        elif not if_random:
            self.model = gensim.models.KeyedVectors.load_word2vec_format(weight_path, binary=True)

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

        text = text.lower()
        if "_" in text or "," in text:
            text_list = text.split("_")
            if len(text_list) == 0:
                text_list = text.split(",")
            feature = None
            for text in text_list:
                if feature is None:
                    if text not in self.model:
                        self.model[text] = numpy.random.uniform(-1, 1, 300)
                    feature = self.model[text]
                else:
                    if text not in self.model:
                        self.model[text] = numpy.random.uniform(-1, 1, 300)
                    feature = feature + self.model[text]
            #if text not in self.model:
            #    self.model[text] = numpy.random.uniform(-1, 1, 300)
            self.text_dict[text] = feature / len(text_list)
            return feature / len(text_list)

        if text not in self.model:
            self.model[text] = numpy.random.uniform(-1, 1, 300)
        self.text_dict[text] = self.model[text]
        return self.model[text]

if __name__ == "__main__":
    text = TextLabel()
    text_feature = text.embed_text("n04443257")
    print(text_feature.shape)
