import sys
import numpy as np
from copy import deepcopy
from gensim.models import Word2Vec

import re
alteos = re.compile(r'( [!\?] )')

## define a couple of generators to produce reviews/sentences
def YelpReviews( stars = [1,2,3,4,5], prefix="train" ):
    for nstar in stars:
        for line in open("data/yelp%s%dstar.txt"%(prefix,nstar)):
            line = alteos.sub(r' \1 . ', line).rstrip("( \. )*\n")
            yield [s.split() for s in line.split(" . ")]

reviews = { s: list(YelpReviews([s])) for s in range(1,6) }


jointmodel = Word2Vec(workers=4)
allsentences = [s for k in reviews for r in reviews[k] for s in r]
np.random.shuffle(allsentences)
jointmodel.build_vocab(allsentences)  

model = { s: deepcopy(jointmodel) for s in range(1,6) }
def trainx(s, T=10):
    sent = [l for r in reviews[s] for l in r]
    model[s].min_alpha = model[s].alpha
    for epoch in range(T):
        print(epoch, end=" ")
        np.random.shuffle(sent)
        model[s].train(sent)
        model[s].alpha *= 0.9  
        model[s].min_alpha = model[s].alpha  # fix the learning rate, no decay
    print(".")

for s in range(1,6):
    print(s)
    %time trainx( s )

def nearby(word, s):
    print(word)
    print( "%d:"%s, end=" ")
    for (w,v) in model[s].most_similar([word]):
        print(w, end=" ")
    print("\n")

nearby("food", 1)
nearby("food", 5)

nearby("service", 1)
nearby("service", 5)

nearby("value", 1)
nearby("value", 5)

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


