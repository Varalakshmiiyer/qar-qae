from collections import OrderedDict

class Vocabulary:
    def __init__(self, first_word='<unk>'):
        self.word_index = OrderedDict({})
        self.index_word = OrderedDict({})
        self.first_word = first_word
        self.add_word(first_word)

    def add_word(self,word):
        if word not in self.word_index:
            index = len(self.word_index)
            self.word_index[word] = index
            self.index_word[index] = word

    def __call__(self, word):
        if not word in self.word_index:
            return self.word_index[self.first_word]
        return self.word_index[word]

    def __len__(self):
        return len(self.word_index)