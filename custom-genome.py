import pandas as pd

output_file = 'sd2-cpf1-mono.fa'
input_file = 'data/mini/sd2-cpf1-mono.xlsx'
targets = pd.read_excel(input_file)
with open(output_file, 'w') as f:
    for i in targets.index:
        row = targets.iloc[i]
        t = row['Nucleotide sequence'].replace(' ', '')
        f.write(f'>{row["Gene"]},{row["Chromosome"]}\n')
        f.write(f'{t}\n')