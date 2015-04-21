import sys
import numpy as np
from copy import deepcopy
from gensim.models import Word2Vec

fin = open("data/yelptrain1star.txt")
firstbadreview = fin.readline()
print(firstbadreview)

import re
alteos = re.compile(r'( [!\?] )')

## define a couple of generators to produce reviews/sentences
def YelpReviews( stars = [1,2,3,4,5], prefix="train" ):
    for nstar in stars:
        for line in open("data/yelp%s%dstar.txt"%(prefix,nstar)):
            line = alteos.sub(r' \1 . ', line).rstrip("( \. )*\n")
            yield [s.split() for s in line.split(" . ")]

reviews = {}
reviews['neg'] = list(YelpReviews([1,2]))
reviews['pos'] = list(YelpReviews([5]))
nbad = len(reviews['neg'])
ngood = len(reviews['pos'])
reviews['neg'][0][:4]

jointmodel = Word2Vec(workers=4)
allsentences = [s for r in reviews['neg']+reviews['pos'] for s in r]
np.random.shuffle(allsentences)
jointmodel.build_vocab(allsentences)  
models = {}
models['neg'] = deepcopy(jointmodel)
models['pos'] = deepcopy(jointmodel)

sentneg = [s for r in reviews['neg'] for s in r]
sentpos = [s for r in reviews['pos'] for s in r]

def trainx(mod, sent, T=20):
    mod.min_alpha = mod.alpha
    for epoch in range(T):
        print(epoch, end=" ")
        np.random.shuffle(sent)
        mod.train(sent)
        mod.alpha *= 0.9  
        mod.min_alpha = mod.alpha  # fix the learning rate, no decay
    print(".")

trainx( models['neg'], sentneg)
trainx( models['pos'], sentpos)

def nearby(word):
    print(word)
    print( "POS:", end=" ")
    for (w,v) in models["pos"].most_similar([word]):
        print(w, end=" ")
    print( "\nNEG:", end=" ")
    for (w,v) in models["neg"].most_similar([word]):
        print(w, end=" ")
    print("\n")

nearby("food")
nearby("service")
nearby("value")
nearby("atmosphere")

prior = ngood/(nbad+ngood)
prior

def getscore(rev):
    sentences =  [(i,s) for i,r in enumerate(rev) for s in r]
    eta = np.column_stack( 
                    ( models['neg'].score([s for i,s in sentences]),
                      models['pos'].score([s for i,s in sentences]) ) )
    probs = np.exp( eta - eta.max(axis=1)[:,np.newaxis] )
    #probs[:,0] *= (1-prior)
    #probs[:,1] *= prior
    probs = probs/probs.sum(axis=1)[:,np.newaxis]
    agg = np.column_stack( 
                    ( np.bincount([i for i,s in sentences], probs[:,0]),
                      np.bincount([i for i,s in sentences], probs[:,1]) ) )
    probpos = agg[:,1]/np.bincount([i for i,s in sentences])
    return(probpos)

testrev = {}
testrev['neg'] = list(YelpReviews([1,2], "test"))
testrev['pos'] = list(YelpReviews([5], "test"))

scores = {}
scores['neg'] = getscore(testrev['neg'])
scores['pos'] = getscore(testrev['pos'])

import matplotlib.pyplot as plt
%matplotlib inline

fig = plt.figure(figsize=(12,4))

fig.add_subplot(1,2,1)
plt.hist(scores['neg'],color="red", alpha=.5, normed=1)
plt.title("neg")
plt.xlabel("prob positive")
plt.ylabel("density")
fig.add_subplot(1,2,2)
plt.hist(scores['pos'],color="green", alpha=.5, normed=1)
plt.title("pos")
plt.xlabel("prob positive")
plt.ylabel("density")

yhat = {'pos': scores['pos']>0.5, 'neg': scores['neg']>0.5}
for sntmnt in yhat:
    print( "mean %s: %.3f" % (sntmnt, yhat[sntmnt].mean()))
print( "MCR: %.3f" % (1-(yhat['pos'].mean() + (1-yhat['neg'].mean()))/2) )


