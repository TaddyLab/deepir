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
# pure numeric
numeric = re.compile(r'(?<=\s)(\d+|\w\d+|\d+\w)(?=\s)', re.I|re.U)
# stop words
swrd = re.compile(r'(?<=\s)(,|"|to|a|the|an|and|or|in|at|with|for|are|is|the|if|of)(?=\s)', re.I|re.U)
# separators (any whitespace)
seps = re.compile(r'\s+')
# exclamation marks are a special way to end your sentence
exclaim = re.compile(r'(\!+)')

# cleaner (order matters)
def clean(text): 
    text = u' ' +  text.lower() + u' '
    text = contractions.sub('', text)
    text = symbols.sub(r' \1 ', text)
    text = numeric.sub(' ', text)
    text = swrd.sub(' ', text)
    text = seps.sub(' ', text)
    #text = exclaim.sub(r'\1 . ', text)
    return text


fout = [ open("data/yelp%dstar.txt" % y, 'w') for y in range(1,6) ]
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
