import hashlib
import re
import sys
import tarfile
from collections import Counter, defaultdict
from pathlib import Path
import json
import matplotlib.pyplot as plt
import requests

import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud
# nltk.download('all')


# Read train, val, and test sets into string objects
train_data = Path('wikitext-103/wiki.train.tokens').read_text()
valid_data = Path('wikitext-103/wiki.valid.tokens').read_text()
test_data = Path('wikitext-103/wiki.test.tokens').read_text()

# train_data = Path('wikitext-2/wiki.train.tokens').read_text()
# valid_data = Path('wikitext-2/wiki.valid.tokens').read_text()
# test_data = Path('wikitext-2/wiki.test.tokens').read_text()



# Store regular expression pattern to search for wikipedia article headings
heading_pattern = '( \n \n = [^=]*[^=] = \n \n )'

# Split out train headings and articles
train_split = re.split(heading_pattern, train_data)
train_headings = [x[7:-7] for x in train_split[1::2]]
train_articles = [x for x in train_split[2::2]]

# Split out validation headings and articles
valid_split = re.split(heading_pattern, valid_data)
valid_headings = [x[7:-7] for x in valid_split[1::2]]
valid_articles = [x for x in valid_split[2::2]]

# Split out test headings and articles
test_split = re.split(heading_pattern, test_data)
test_headings = [x[7:-7] for x in test_split[1::2]]
test_articles = [x for x in test_split[2::2]]


# Number of Wikipedia articles in our training data
len(train_headings)


# Example article
print('Heading: ', train_headings[110])
print('Article sample: ', train_articles[110][:118])

TARGET_DIR = '/home/kangchen/Diffusion-LM-main/datasets/wikitext-103/'

fout = open(TARGET_DIR + "train" + ".json", 'w')
for line in train_articles:
    # json.dump(line, o)
    # data_temp.append(line.strip())
    print(json.dumps(line), file=fout)
fout.close()
print('finish writing', TARGET_DIR + "train" + ".json")


fout = open(TARGET_DIR + "test" + ".json", 'w')
for line in test_articles:
    # json.dump(line, o)
    # data_temp.append(line.strip())
    print(json.dumps(line), file=fout)
fout.close()
print('finish writing', TARGET_DIR + "test" + ".json")


fout = open(TARGET_DIR + "valid" + ".json", 'w')
for line in valid_articles:
    # json.dump(line, o)
    # data_temp.append(line.strip())
    print(json.dumps(line), file=fout)
fout.close()
print('finish writing', TARGET_DIR + "valid" + ".json")