import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw, AllChem, DataStructs
from collections import Counter, defaultdict
import os
from sklearn import cluster

df = pd.read_csv(r"Hits_wSMILES.csv") #File to be analyzed with Molecule ID and its SMILES
smiles_list = df['SMILES'].to_list()
mols =[]
for i, smiles in enumerate(smiles_list):
     mol = Chem.MolFromSmiles(smiles)
     mols.append(mol)
 
n=len(mols);
fps = [AllChem.GetMorganFingerprintAsBitVect(s, 2, nBits=2048) for s in mols] 
sim_matrix = np.zeros((n, n))
for i in range(n):
    for j in range(i,n):
        sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
        sim_matrix[i, j] = sim
        sim_matrix[j, i] = sim
 
distmat = [1-sim_matrix[i,j] for i in range(n) for j in range(n)]

distmat = np.array(distmat).reshape(n,n)

css = cluster.AgglomerativeClustering(distance_threshold=0.32, linkage = 'complete',n_clusters=None, metric='precomputed').fit(distmat)

print(css.labels_)
print(css.n_clusters_)
print(Counter(css.labels_.tolist()))

clusters = Counter(css.labels_.tolist())
clustsizes = [val for key,val in clusters.items()]
print(Counter(clustsizes))
