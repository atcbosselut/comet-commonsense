'''################################################################################

Adapted from Kevin Gimpel's demo available here:

https://ttic.uchicago.edu/~kgimpel/commonsense.html

################################################################################'''

import pickle
import numpy as np
import sys
import math

def getVec(We, words, t):
    t = t.strip()
    array = t.split('_')
    if array[0] in words:
        vec = We[words[array[0]], :]
    else:
        vec = We[words['UUUNKKK'], :]
        print 'can not find corresponding vector total:', array[0].lower()
    for i in range(len(array)-1):
        if array[i+1] in words:
            vec = vec + We[words[array[i+1]], :]
        else:
            print 'can not find corresponding vector some:', array[i+1].lower()
            vec = vec + We[words['UUUNKKK'], :]
    vec = vec/len(array)
    return vec


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def score(term1, term2, words, We, rel, Rel, Weight, Offset, evaType):
    v1 = getVec(We, words, term1)
    v2 = getVec(We, words, term2)
    result = {}

    for k, v in rel.items():
        v_r = Rel[rel[k], :]
        gv1 = np.tanh(np.dot(v1, Weight)+Offset)
        gv2 = np.tanh(np.dot(v2, Weight)+Offset)

        temp1 = np.dot(gv1,  v_r)
        score = np.inner(temp1, gv2)
        result[k] = (sigmoid(score))

    if(evaType.lower() == 'max'):
        result = sorted(result.items(),  key=lambda x: x[1],  reverse = True)
        for k, v in result[:1]:
            print k,  'score:',  v
        return result[:1]
    if(evaType.lower() == 'topfive'):
        result = sorted(result.items(),  key=lambda x: x[1],  reverse = True)
        for k, v in result[:5]:
            print k,  'score:',  v
        return result[:5]
    if(evaType.lower() == 'sum'):
        result = sorted(result.items(),  key=lambda x: x[1],  reverse = True)
        total = 0
        for i in result:
            total = total + i[1]
        print 'total score is:', total
        return total
    if(evaType.lower() == 'all'):
        result = sorted(result.items(),  key=lambda x: x[1],  reverse = True)
        for k, v in result[:]:
            print k,  'score:',  v
        return result
    else:
        tar_rel = evaType.lower()
        if result.get(tar_rel) == None:
            print 'illegal relation,  please re-enter a valid relation'
            return 'None'
        else:
            return result.get(tar_rel)

def run(gens_file, theshold=None, flip_r_e1=False):
    model = pickle.load(open("ckbc-demo/Bilinear_cetrainSize300frac1.0dSize200relSize150acti0.001.1e-05.800.RAND.tanh.txt19.pickle",  "r"))

    Rel = model['rel']
    We = model['embeddings']
    Weight = model['weight']
    Offset = model['bias']
    words = model['words_name']
    rel = model['rel_name']

    results = []

    if type(gens_file) == list:
        gens = []
        for file_name in gens_file:
            gens += open(file_name, "r").read().split("\n")
    else:
        gens = open(gens_file, "r").read().split("\n")

    formatted_gens = [tuple(i.split("\t")[:4]) for i in gens if i]

    for i, gen in enumerate(formatted_gens):
        if gen == ('s', 'r', 'o', 'minED'):
            continue
        if flip_r_e1:
            relation = "_".join(gen[1].split(" "))
            subject_ = "_".join(gen[0].split(" "))
        else:
            relation = "_".join(gen[0].split(" "))
            subject_ = "_".join(gen[1].split(" "))
        object_ = "_".join(gen[2].split(" "))
        result = score(subject_, object_, words, We, rel, Rel, Weight, Offset, relation)

        results.append((gen, result))

    return results


