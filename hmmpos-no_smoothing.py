import pickle, argparse, os, sys
from sklearn.metrics import accuracy_score
import random

def train(training_file):
    
    assert os.path.isfile(training_file), 'Training file does not exist'

    # Your code starts here

    from collections import Counter, defaultdict
    
    # Make vocabulary and change rare words to 'UNKA'
    # (occurs less than 3 times)
    # Make vocabulary and tag set.
    voca = Counter()
    alltags = set()
    with open(training_file, mode='r') as infile:
        for sentag in infile.read().splitlines():
            voca.update([word.lower() for word in sentag.split()[::2]])
            alltags.update(sentag.split()[1::2])
    rare = [word for word, count in voca.items() if count < 3]
    rare = set(rare)
    
    voca['UNKA'] = 0
    for key, val in voca.copy().items():
        if key in rare:
            voca['UNKA'] += val
            del voca[key]
        
    # Make a list of training data while changing rare words to 'UNKA'
    training_list = []
    with open(training_file, mode='r') as infile:
        for sentag in infile.read().splitlines():
            sen = [word.lower() for word in sentag.split()[::2]]
            sen = ['UNKA' if word in rare else word for word in sen]
            tags = [tag for tag in sentag.split()[1::2]]
            sentag = [None] * (len(sen) + len(tags))
            sentag[::2] = sen
            sentag[1::2] = tags
            training_list.append(sentag)
    
    # Count to calculate each prob.
    EOS = 'EOS'
    tran_cnt = defaultdict(int)
    emis_cnt = defaultdict(lambda: defaultdict(int))
    pi_cnt = defaultdict(int)
    for sentag in training_list:
    
        # Extract words and tags separately from current sentence
        words = sentag[::2]
        tags = sentag[1::2]
        
        # Count for transition prob.
        for tagBigram in [(t0, t1) for t0, t1 in zip(tags, tags[1:])]:
            tran_cnt[tagBigram] += 1
        tran_cnt[(tags[-1], EOS)] += 1
        
        # Count for emission prob.
        for ii in range(len(words)):
            emis_cnt[tags[ii]][words[ii]] += 1
        
        # Count for initial prob.
        pi_cnt[tags[0]] += 1
    
    # Make countings to log prob.
    from math import log, inf
    
    # Convert transition counts to transition prob.
    deno = defaultdict(int)
    for (t0, t1), count in tran_cnt.items():
        deno[t0] += count
    tran = {tag:log(count/deno[tag[0]])
                for tag, count in tran_cnt.items()}
                
    # Convert emission counts to emission prob.
    deno = {tag:sum(word.values()) for tag, word in emis_cnt.items()}
    emis = {tag:
                {word:log(count/deno[tag])
                    for word, count in words.items()}
                for tag, words in emis_cnt.items()}
    
    # Convert initial count to initial prob.
    deno = sum(pi_cnt.values())
    pi = {tag:log(count/deno) for tag, count in pi_cnt.items()}
    
    # Assign -inf to any cases which did not occur
    # in transition prob. and emission prob.
    for t0 in alltags:
        for t1 in alltags:
            try:
                tran[(t0,t1)]
            except:
                tran[(t0,t1)] = -inf

    for tag in alltags:
        for word in voca.keys():
            try:
                emis[tag][word]
            except:
                emis[tag][word] = -inf
    
    model = {'tran':tran, 'emis':emis, 'pi':pi, 'alltags':alltags}

    # Your code ends here

    return model

def test(model_file, data_file, label_file):
    assert os.path.isfile(model_file), 'Model file does not exist'
    assert os.path.isfile(data_file), 'Data file does not exist'
    assert os.path.isfile(label_file), 'Label file does not exist'

    # Your code starts here
    from math import inf

    model = pickle.load(open(model_file,'rb'))
    
    # Load test data and save it in a list.
    # Each sentence is a row of the list.
    data = []
    with open(data_file, mode='r') as infile:
        for sen in infile.read().splitlines():
            sen = [word.lower() if word != 'UNKA' else 'UNKA' for word in sen.split()]
            data.append(sen)
    
    # Define Viterbi algorithm
    def Viterbi(obs, states, tran, emis, pi):
        dp = [{} for ii in range(len(obs))]
        path = {}

        # Base case
        for ss in pi.keys():
            dp[0][ss] = pi[ss] + emis[ss][obs[0]]
            path[ss] = [ss]
        for ss in states:
            try:
                dp[0][ss]
            except:
                dp[0][ss] = -inf
        
        # All future cases
        for tt in range(1, len(obs)):
            new_path = {}
            
            for s1 in states:
                (prob, state) = \
                max(
                (dp[tt-1][s0] + tran[(s0,s1)] + emis[s1][obs[tt]], s0)
                for s0 in states)
                new_path[s1] = path[state] + [s1]
                dp[tt][s1] = prob
                
            path = new_path
        
        (prob, state) = max((dp[len(obs) - 1][s1], s1) for s1 in states)
        return (prob, path[state])
    
    prediction = []
    for sen in data:
        _, path = Viterbi(sen, model['alltags'],
        model['tran'], model['emis'], model['pi'])
        prediction.extend(path)

    ground_truth = []
    with open(label_file, mode='r', errors='ignore') as infile:
        for sentag in infile.read().splitlines():
            tags = sentag.split()[1::2]
            ground_truth.extend(tags)

    # Your code ends here

    print(f'The accuracy of the model is {100*accuracy_score(prediction, ground_truth):6.2f}%')

def main(params):
    if params.train:
        model = train(params.training_file)
        pickle.dump(model, open(params.model_file,'wb'))
    else:
        test(params.model_file, params.data_file, params.label_file)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HMM POS Tagger")
    parser.add_argument("--train", action='store_const', const=True, default=False)
    parser.add_argument('--model_file', type=str, default='model.pkl')
    parser.add_argument('--training_file', type=str, default='')
    parser.add_argument('--data_file', type=str, default='')
    parser.add_argument('--label_file', type=str, default='')
    
    main(parser.parse_args())
