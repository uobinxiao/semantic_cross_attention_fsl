import fasttext
import fasttext.util

label_word_path = "data/miniimagenet/MiniImageNet_cls.txt"
class FastTextLabel:
    def __init__(self, dataset):
        assert dataset == "MiniImageNet" or dataset == "CUB" or dataset == "TieredImageNet"
        if dataset == "MiniImageNet":
            self.label_dict = self.load_text_label()
        else:
            self.label_dict = {}

        fasttext.util.download_model("en", if_exists="ignore")
        self.model = fasttext.load_model("cc.en.300.bin")

        self.text_dict = {}

    #this is for mini-imagenet dataset
    def load_text_label(self):
        label_word_dict = {}
        with open(label_word_path) as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if "Train" in line or "Val"  in line or "Test" in line:
                    continue
                words = line.split(" ")
                label_word_dict[words[0]] = words[1]

        return label_word_dict

    def embed_text(self, text):
        text = text.strip()
        if text in self.text_dict.keys():
            return self.text_dict[text]
        if self.label_dict:
            text = self.label_dict[text]

        text_vec = None
        if len(text.split(" ")) == 1:
            text_vec = self.model.get_word_vector(text)
        else:
            text_vec = self.model.get_sentence_vector(text)

        return text_vec
