import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor

nucleotides = ['A', 'C', 'G', 'T']
dinucleotides = [i + j for i in nucleotides for j in nucleotides]
trinucleotides = [i + j + k for i in nucleotides for j in nucleotides for k in nucleotides]
filename = 'c12a_predictor.sav'

def gc_content(seq):
    d = len([s for s in seq if s in 'CcGc']) / len(seq) * 100
    return round(d, 2)

def tokenize_sequence_global(seq, name):
    tokens = dinucleotides
    data = {n: sum(1 for i in range(len(seq)) if seq.startswith(n, i))/(len(seq)-len(n)+1) for n in tokens}
    s = pd.Series(data)
    s = s.reindex(tokens, axis=1)
    s.index = ['{}_{}'.format(name, x) for x in s.index]
    return s

def tokenize_sequence_local(seq, name):
    tokens = nucleotides + dinucleotides + trinucleotides
    data = {f'loc{len(n)}_{i}_{n}': 1  if seq.startswith(n, i) else 0 for i in range(len(seq)) for n in tokens}
    s = pd.Series(data)
    s.index = ['{}_{}'.format(name, x) for x in s.index]
    return s

def process_sequence(df):
    leftside_global_df = df.apply(lambda _: tokenize_sequence_global(_[:4], 'left'))
    rightside_global_df = df.apply(lambda _: tokenize_sequence_global(_[31:], 'right'))
    protospacer_global_df = df.apply(lambda _: tokenize_sequence_global(_[8:31], 'protospacer'))
    protospacer_global_df5 = pd.concat([df.apply(lambda _: tokenize_sequence_global(_[i:i+5], f'protospacer{i}')) for i in range(0,30)], axis=1)
    protospacer_global_df8 = pd.concat([df.apply(lambda _: tokenize_sequence_global(_[i:i+8], f'protospacer{i}')) for i in range(0,27)], axis=1)
    protospacer_global_df = pd.concat([protospacer_global_df, protospacer_global_df5, protospacer_global_df8], axis=1)
    leftside_local_df = df.apply(lambda _: tokenize_sequence_local(_[:4], 'left'))
    rightside_local_df = df.apply(lambda _: tokenize_sequence_local(_[31:], 'right'))
    protospacer_local_df = df.apply(lambda _: tokenize_sequence_local(_[8:31], 'protospacer'))
    protospacer_gc = df.apply(lambda _: gc_content(_[8:31])).rename('gc')
    rolling_gc = pd.concat([df.apply(lambda _: gc_content(_[i:i+7])).rename(f'gc{i}') for i in range(0,28)], axis=1)
    global_features_df = pd.concat([protospacer_global_df, leftside_global_df, rightside_global_df], axis=1)
    local_features_df = pd.concat([protospacer_local_df, leftside_local_df, rightside_local_df], axis=1)
    return pd.concat([global_features_df, local_features_df, rolling_gc, protospacer_gc], axis=1)


## Load Random Forest model from disk
predictor = joblib.load(filename)


## Predict the efficiency for each sequence in the file
for line in open('sequences.txt', 'r'):
    sequence = line.strip()
    prediction = predictor.predict(process_sequence(pd.Series([sequence])))
    print(sequence, prediction[0])

