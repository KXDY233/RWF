## Random Walk Fingerprints for Graph Classification


A backup of the whole project (code and data) is available at: https://drive.google.com/file/d/1rZUMvIH9bsl2tnbqJGA255Ic_WGJzEkV/view?usp=sharing


This is the responsibility of **Scalable Graph Classification via Random Walk Fingerprints**. We implemented RWF based on A-DOGE, by using the same data inputs and the same svm classifier.

  Attributed Density of States based Graph Embedding
  https://github.com/sawlani/A-DOGE

### Required Packages:
```
networkx,
scipy,
pytorch,
sklearn,
numpy,
pandas,
cpnet (for running km-config)
```

### Datasets

We follow A-DOGE to preprocess the data.

Ten processed graph datasets utilized in our paper are provided in data/processed/ folder.

Two big datasets (> 25 MB) cannot be uploaded to GitHub.

The following shared zip file contains all the processed datasets in the required format. 

https://drive.google.com/file/d/1XxOIf04xpOoGaLIu0QOmvjqvx6P8678o/view?usp=sharing



### Reproducing the Results

<img width="1007" alt="image" src="https://github.com/KXDY233/RWF/assets/85337987/ca1eabe7-fec2-4b24-9689-e8ac27018b03">
<img width="504" alt="image" src="https://github.com/KXDY233/RWF/assets/85337987/054dffdd-26e4-43dd-902e-2dc9f723b781">

You can produce all the results by running:
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


