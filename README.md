This is the responsibility for ``Scalable Graph Classification via Random Walk Fingerprints''.


**We implemented RWF based on A-DOGE, by using the same data inputs and the same svm classifier.**

  _Attributed Density of States based Graph Embedding
  https://github.com/sawlani/A-DOGE_


Some big files cannot be uploaded to GitHub.
The following shared zip file contains all the datasets in the required format. 
https://drive.google.com/file/d/1XxOIf04xpOoGaLIu0QOmvjqvx6P8678o/view?usp=sharing


### You can produce all the results by running:
```
python RWF_Degree.py                --k=2 --dataset=[]
python RWF_Degree.py                --k=3 --dataset=[]
python RWF_Degree_onehot_feature.py --k=2 --dataset=[]
python RWF_Degree_onehot_feature.py --k=3 --dataset=[]
python RWF_Degree_degree_feature.py --k=2 --dataset=[]
python RWF_Degree_degree_feature.py --k=3 --dataset=[]

python RWF_CP.py                    --dataset=[]
python RWF_CP_degree_feature.py     --dataset=[]
python RWF_CP_onehot_feature.py     --dataset=[]

python RWF_Random.py                --k=2 --dataset=[]
python RWF_Random.py                --k=3 --dataset=[]

python RWF_Without.py               --dataset=[]
```


e.g., 
```
python RWF_Degree.py                --k=2 --dataset=MUTAG
```

### required packages:
```
networkx,
scipy,
pytorch,
sklearn,
numpy,
pandas,
cpnet (for running km-config)
```