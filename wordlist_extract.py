
import os
import random
import time
import pickle
import math
import string
import csv

from tqdm import tqdm
import numpy as np

# num matches of distinct words
name_wordlist = []
type_wordlist = []
food_wordlist = []
price_wordlist = []
area_wordlist = []
rating_wordlist = []
friendly_wordlist = []
# with open(os.path.join(tw_dir, category + '.txt'), 'r') as rf:
with open(os.path.join('./datasets/e2e_data/src1_train.txt'), 'r') as rf:
    for lines in rf:
        line = lines.lower().split('|')
        for words in line:
            if words == '':
                break
            word = words.lower().split(':')
            # print(word)
            if 'name' in word[0]:
                word_temp = word[1]
                # print('The first letter: {' + word_temp + '} !:')
                word_temp = list(word_temp)
                if word_temp[0] == ' ':
                    del(word_temp[0])
                if word_temp[-1] == ' ':
                    del(word_temp[-1])

                word_temp = "".join(word_temp)
                # print('The first letter: {' + word_temp + '} !:')
                name_wordlist.append(word_temp)
            elif 'type' in word[0]:
                word_temp = word[1]
                # print('The first letter: {' + word_temp + '} !:')
                word_temp = list(word_temp)
                if word_temp[0] == ' ':
                    del(word_temp[0])
                if word_temp[-1] == ' ':
                    del(word_temp[-1])

                word_temp = "".join(word_temp)
                # print('The first letter: {' + word_temp + '} !:')
                type_wordlist.append(word_temp)
            elif 'food' in word[0]:
                word_temp = word[1]
                # print('The first letter: {' + word_temp + '} !:')
                word_temp = list(word_temp)
                if word_temp[0] == ' ':
                    del(word_temp[0])
                if word_temp[-1] == ' ':
                    del(word_temp[-1])

                word_temp = "".join(word_temp)
                # print('The first letter: {' + word_temp + '} !:')
                food_wordlist.append(word_temp)
            elif 'price' in word[0]:
                word_temp = word[1]
                # print('The first letter: {' + word_temp + '} !:')
                word_temp = list(word_temp)
                if word_temp[0] == ' ':
                    del(word_temp[0])
                if word_temp[-1] == ' ':
                    del(word_temp[-1])

                word_temp = "".join(word_temp)
                # print('The first letter: {' + word_temp + '} !:')
                price_wordlist.append(word_temp)
            elif 'area' in word[0]:
                word_temp = word[1]
                # print('The first letter: {' + word_temp + '} !:')
                word_temp = list(word_temp)
                if word_temp[0] == ' ':
                    del(word_temp[0])
                if word_temp[-1] == ' ':
                    del(word_temp[-1])

                word_temp = "".join(word_temp)
                # print('The first letter: {' + word_temp + '} !:')
                area_wordlist.append(word_temp)
            elif 'customer rating' in word[0]:
                word_temp = word[1]
                # print('The first letter: {' + word_temp + '} !:')
                word_temp = list(word_temp)
                if word_temp[0] == ' ':
                    del(word_temp[0])
                if word_temp[-1] == ' ':
                    del(word_temp[-1])

                word_temp = "".join(word_temp)
                # print('The first letter: {' + word_temp + '} !:')
                rating_wordlist.append(word_temp)
            elif 'family friendly' in word[0]:
                word_temp = word[1]
                # print('The first letter: {' + word_temp + '} !:')
                word_temp = list(word_temp)
                if word_temp[0] == ' ':
                    del(word_temp[0])
                if word_temp[-1] == ' ':
                    del(word_temp[-1])

                word_temp = "".join(word_temp)
                # print('The first letter: {' + word_temp + '} !:')
                friendly_wordlist.append(word_temp)
            else:
                print('Else!')


name_wordlist = list(dict.fromkeys(name_wordlist))
print(name_wordlist)

type_wordlist = list(dict.fromkeys(type_wordlist))
print(type_wordlist)

food_wordlist = list(dict.fromkeys(food_wordlist))
print(food_wordlist)

price_wordlist = list(dict.fromkeys(price_wordlist))
print(price_wordlist)

area_wordlist = list(dict.fromkeys(area_wordlist))
print(area_wordlist)

rating_wordlist = list(dict.fromkeys(rating_wordlist))
print(rating_wordlist)

friendly_wordlist = list(dict.fromkeys(friendly_wordlist))
print(friendly_wordlist)



