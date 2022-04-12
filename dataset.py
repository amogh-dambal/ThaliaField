import torch
import pandas as pd
from collections import Counter


class Dataset(torch.utils.data.Dataset):
    def __init__(self, args,):
        self.args = args
        self.sequenceLength = self.args.sequence_length
        self.words = self.loadWords()
        self.uniqueWords = self.getUniqueWords()

        self.idxToWord = {index: word for index, word in enumerate(self.uniqueWords)}
        self.wordToIdx = {word: index for index, word in enumerate(self.uniqueWords)}
        self.wordsIndices = [self.wordToIdx[w] for w in self.words]

    def loadWords(self):
        with open('data/lyrics.txt', 'r', encoding='utf-8-sig') as f:
            lines = f.readlines()
            words = [word for line in lines for word in line.split()]
        return words

    def getUniqueWords(self):
        wc = Counter(self.words)
        return sorted(wc, key=wc.get, reverse=True)

    def __len__(self):
        return len(self.wordsIndices) - self.sequenceLength

    def __getitem__(self, index):
        return (
            torch.tensor(self.wordsIndices[index : index + self.sequenceLength]),
            torch.tensor(
                self.wordsIndices[index + 1 : index + self.sequenceLength + 1]
            ),
        )
