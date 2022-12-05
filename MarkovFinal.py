import string
import numpy as np
import matplotlib as plt
from string import punctuation
import math

class markovChain:
    alphabet = string.ascii_lowercase
    def __init__(self, file_name = ""):
#         index for loopup certain word/letter
#         trans_matrix for p(z_n|z_n-1)
#         array of pairs of words/letters
#         prior probability
        self.index, self.trans_matrix, self.pairs, self.prior = 0, 0, 0, 0
        self.train_data = self.get_file(file_name)
        self.matrix_before = np.array([[0.001 for _ in range(26)] for _ in range(26)])
        self.train()

#   get the file as training set and return a list of unique word
    def get_file(self, file_name):
#       get the text from file and trimming special and numeric characters
        text = ''
        f = open(file_name, encoding='utf8')
        text +=f.read()
        text = text.replace('\n',' ')
        text = text.replace('\t',' ')
        text = text.replace('“', ' " ')
        text = text.replace('”', ' " ')
        for spaced in punctuation:
            text = text.replace(spaced, ' {0} '.format(spaced))
        for spaced in string.digits:
            text = text.replace(spaced, ' {0} '.format(spaced))
#       return training data
        return np.unique(np.array([word.lower() for word in text.split(" ") if word not in punctuation and word not in string.digits]))

    def train(self):
#       get the matrix
        for word in self.train_data:
            for i in range(1, len(word)):
                if word[i] not in self.alphabet or word[i-1] not in self.alphabet:
                    continue
                self.matrix_before[self.alphabet.index(word[i-1])][self.alphabet.index(word[i])] += 1
        self.matrix_before = np.log(self.matrix_before/self.matrix_before.sum(axis=1))
        return self.matrix_before

    def guess_word(self, hidden):
        while "_" in hidden:
            guess_loc = hidden.index('_')
            if guess_loc == 0:
                # print(self.alphabet[np.argmax(self.matrix_before, axis=0)[self.alphabet.index(hidden[1])]])
                hidden = self.alphabet[np.argmax(self.matrix_before, axis=0)[self.alphabet.index(hidden[1])]] + hidden[1:]
            elif guess_loc == len(hidden)-1:
                hidden = hidden[:-1] + self.alphabet[np.argmax(self.matrix_before[self.alphabet.index(hidden[guess_loc-1])])]
            elif hidden[guess_loc+1] != "_":
                hint_begin, hint_end = self.alphabet.index(hidden[guess_loc-1]), self.alphabet.index(hidden[guess_loc+1])
                guess = self.alphabet[np.argmax(np.array([self.matrix_before[hint_begin][i]+self.matrix_before[i][hint_end] for i in range(26)]))]
                hidden = hidden[:guess_loc] + guess + hidden[guess_loc+1:]
            elif guess_loc+2 == len(hidden):
                hint_begin = self.alphabet.index(hidden[guess_loc-1])
                matrix_temp = np.array([[self.matrix_before[hint_begin][i]+self.matrix_before[i][j] for i in range(26)] for j in range(26)])
                r, c = np.unravel_index(np.argmax(matrix_temp, axis=None), matrix_temp.shape)
                hidden = hidden[:guess_loc] + self.alphabet[r] + self.alphabet[c]
            else:
                hint_begin, hint_end = self.alphabet.index(hidden[guess_loc-1]), self.alphabet.index(hidden[guess_loc+2])
                matrix_temp = np.array([[self.matrix_before[hint_begin][i]+self.matrix_before[i][j]+self.matrix_before[j][hint_end] for i in range(26)] for j in range(26)])
                r, c = np.unravel_index(np.argmax(matrix_temp, axis=None), matrix_temp.shape)
                hidden = hidden[:guess_loc] + self.alphabet[r] + self.alphabet[c] + hidden[guess_loc+2:]
        return hidden
    
def miscalcRate(sentence, answer, missing):
    miscalc = 0
    if (len(sentence) != len(sentence)):
        print("2 strings need to have same length")
        return
    for i in range(len(sentence)):
        if (sentence[i] != answer[i]):
            miscalc+=1
    return miscalc/missing

markov = markovChain('./train_40k.csv')

answer1  = 'helloworld'
string1  = 'he_l_w_r_d'
predict1 = markov.guess_word(string1)
print("Answer:", answer1, "\tGuess:", string1, "\tPredict:", predict1, "\nMiscalculation Rate:", miscalcRate(predict1, answer1, 4), '\n')

answer2  = 'helloworld'
string2  = 'hel__w_r_d'
predict2 = markov.guess_word(string2)
print("Answer:", answer2, "\tGuess:", string2, "\tPredict:", predict2, "\nMiscalculation Rate:", miscalcRate(predict2, answer2, 4), '\n')

answer3  = 'thisisit'
string3  = 't_is__it'
predict3 = markov.guess_word(string3)
print("Answer:", answer3, "\tGuess:", string3, "\tPredict:", predict3, "\nMiscalculation Rate:", miscalcRate(predict3, answer3, 3), '\n')

answer4  = 'thisisoberlin'
string4  = 't__s_s_b_rlin'
predict4 = markov.guess_word(string4)
print("Answer:", answer4, "\tGuess:", string4, "\tPredict:", predict4, "\nMiscalculation Rate:", miscalcRate(predict4, answer4, 5), '\n')