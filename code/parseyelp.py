#!/usr/bin/env python
## python map for word counts

# Import Modules
import sys
import re
import json
# import codecs
# from collections import Counter # not in py<2.7

# all non alphanumeric
contractions = re.compile(r"'|-")
symbols = re.compile(r'(\W+)', re.U)
numeric = re.compile(r'(?<=\s)(\d+|\w\d+|\d+\w)(?=\s)', re.I)
swrd = re.compile(r'(?<=\s)(,|"|\(|\)|to|a|as|the|an|and|or|for|are|is)(?=\s)', re.I)
suffix = re.compile(r'(?<=\w)(s|ings*|ly|(?<=e)[sd]+)(?=\s)')
seps = re.compile(r'\s+')

# cleaner (order matters)
def clean(text): 
    text = u' ' +  text.lower() + u' '
    text = contractions.sub('', text)
    text = symbols.sub(r' \1 ', text)
    text = numeric.sub('00', text)
    text = swrd.sub(' ', text)
    text = suffix.sub('', text)
    text = seps.sub(' ', text)
    return text


fout = [ open("data/yelptrain%dstar.txt" % y, 'w') for y in range(1,6) ]
fin = open("data/yelp_training_set/yelp_training_set_review.json", 'r')
i = 0

for line in fin:
    d = json.loads(line)
    i += 1
    try:
        txt = clean(d['text'])
        fout[d['stars']-1].write(txt+'\n')
        print(i, end=" ")

    except:
        e = sys.exc_info()[0]
        sys.stderr.write("review reader error: %s\n"%str(e))

fin.close()
for f in fout:
    f.close()


fout = [ open("data/yelptest%dstar.txt" % y, 'w') for y in range(1,6) ]
fin = open("data/yelp_test_set/yelp_test_set_review.json", 'r')
i = 0

for line in fin:
    d = json.loads(line)
    i += 1
    try:
        txt = clean(d['text'])
        fout[d['stars']-1].write(txt+'\n')
        print(i, end=" ")

    except:
        e = sys.exc_info()[0]
        sys.stderr.write("review reader error: %s\n"%str(e))

fin.close()
for f in fout:
    f.close()
