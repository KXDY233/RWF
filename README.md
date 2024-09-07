# 🌟 Scalable Graph Classification via Random Walk Fingerprints 🌟

> **Accepted at IEEE ICDM 2024**

This repository contains the implementation of **Random Walk Fingerprints (RWF)** for scalable graph classification, built upon the A-DOGE framework. Our method leverages the same data inputs and SVM classifier as A-DOGE.

👉 For more information on A-DOGE, visit: [Attributed Density of States based Graph Embedding (A-DOGE)](https://github.com/sawlani/A-DOGE)

## 🚀 Required Packages

To run the code, ensure the following packages are installed:

```
networkx,
scipy,
pytorch,
sklearn,
numpy,
pandas,
cpnet (for running km-config)
```


## 📂 Datasets

The data preprocessing follows the same steps as A-DOGE. The ten processed datasets used in this paper are available in the `data/processed/` folder.

**Note**: Two larger datasets (over 25MB) could not be uploaded to GitHub. You can download all the datasets, including the larger ones, from this [Google Drive link](https://drive.google.com/file/d/1XxOIf04xpOoGaLIu0QOmvjqvx6P8678o/view?usp=sharing).

## 🧪 Reproducing the Results

|               | MUTAG     | PTC_MM    | PTC_FR    | PROTEINS  | DD        | IMDB-M    | RED-B     | RED-5K    |
|---------------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|
| NetLSD        | 87.2(7.5) | 65.1(5.5) | 65.5(0.5) | 74.0(3.5) | 75.6(3.3) | 47.8(3.4) | 85.5(1.9) | 46.0(2.1) |
| A-DOGE        | 85.2(8.4) | 66.8(6.8) | 67.6(3.0) | 74.1(3.9) | 75.9(3.2) | 46.1(3.5) | **91.4(1.8)** | 55.1(2.2) |
| FGSD          | 87.4(6.8) | 65.0(5.0) | 65.5(0.5) | 71.8(3.7) | 75.7(3.2) | 49.2(3.8) | 82.2(2.6) | 45.1(2.0) |
| NetSimile     | 84.7(7.4) | 63.3(5.6) | 66.9(3.1) | 71.2(3.1) | 73.7(4.0) | 47.8(3.8) | 90.6(1.7) | 54.0(2.1) |
| NH            | 87.4(6.7) | 64.7(5.1) | 65.5(0.5) | 75.5(4.0) | 76.6(3.1) | **51.5(3.2)** | 82.1(2.8) | 49.7(1.8) |
| WL            | 84.9(8.1) | **67.5(6.2)** | 67.9(2.9) | 76.1(3.8) | 79.1(2.9) | 51.4(3.2) | 71.7(2.9) | 50.4(2.0) |
| RWK           | 84.4(7.9) | 63.4(5.3) | 65.5(0.5) | 65.0(2.2) | 58.7(0.3) | 51.7(3.9) | 67.6(1.0) | 50.0(2.2) |
| DOSGK         | 81.5(8.7) | 62.1(4.4) | 65.5(0.5) | 72.6(3.5) | 74.4(3.6) | 48.0(3.3) | 85.8(2.5) | 50.0(2.2) |
| GCN           | 83.5(7.3) | 61.7(11.0) | 65.5(8.6) | 67.8(8.5) | 78.0(1.8) | 45.6(3.4)★ | 87.8(2.5) | 49.2(1.2)★ |
| GraphSAGE     | 81.9(6.3) | 63.5(13.3) | 66.4(10.2) | 70.0(6.7) | 72.9(2.0) | 47.6(3.5)★ | 84.3(1.9)★ | 50.0(1.3)★ |
| GIN           | 85.3(9.1) | 64.1(10.5) | 66.4(9.8) | 73.1(4.8) | 75.3(2.9) | 48.5(3.3)★ | **89.9(1.9)** | 56.1(1.7)★ |
| RWF-CP        | 85.3(7.1) | 66.8(5.5) | 65.6(5.1) | 75.1(3.8) | 76.7(3.2) | 48.5(3.2) | 88.4(2.2) | 53.4(2.4) |
| RWF-CP-feature| 86.6(7.0) | 67.2(5.8) | 67.1(2.4) | 75.5(3.8) | **79.4(3.3)** | 48.8(3.5) | 89.0(2.1) | 55.0(2.5) |
| RWF-D         | 86.8(7.3) | 64.5(4.6) | 66.9(2.7) | 74.9(3.6) | 76.9(3.2) | 49.4(3.3) | 90.2(2.1) | 54.6(2.2) |
| RWF-D-feature | **87.7(6.8)** | 66.2(5.7) | **68.0(2.5)** | **77.1(3.7)** | **79.1(3.5)** | 49.6(3.7) | **90.7(2.0)** | **55.6(2.1)** |

|                 | DEE    | RED-12K | GIT   | TWI   |
|-----------------|--------|---------|-------|-------|
| NetLSD          | 56.1   | 38.5    | 64.6  | 68.6  |
| A-DOGE          | 55.6   | 47.8    | 67.0  | 69.5  |
| FGSD            | 56.2   | 33.2    | 60.7  | 64.2  |
| NetSimile       | 55.6   | 47.3    | 68.8  | 70.5  |
| NH              | 56.0   | ‡       | 60.2  | ‡     |
| WL              | 55.6   | 39.5    | 62.6  | ‡     |
| RWK             | ‡      | ‡       | ‡     | ‡     |
| DOSGK           | 55.7   | ‡       | 66.3  | ‡     |
| GCN             | 56.8   | 45.9    | 61.3  | 69.0  |
| GraphSAGE       | 56.8   | 42.2    | 53.7  | 61.4  |
| GIN             | 56.9   | 47.3    | 61.4  | 68.7  |
| RWF-CP          | 57.1   | 43.6    | 66.7  | 69.6  |
| RWF-CP-feature  | 56.6   | 46.2    | 70.4  | 69.1  |
| RWF-D           | 56.5   | 44.8    | 68.1  | 71.3  |
| RWF-D-feature   | 55.9   | 48.1    | 70.4  | 71.4  |


## ⚙️ Running the Code

To reproduce the results from our experiments, use the following commands:

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
