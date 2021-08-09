# SL-GAD

## Dependencies
+ python==3.6.1
+ dgl==0.4.1
+ matplotlib==3.3.4
+ networkx==2.5
+ numpy==1.19.2
+ pyparsing==2.4.7
+ scikit-learn==0.24.1
+ scipy==1.5.2
+ sklearn==0.24.1
+ torch==1.8.1
+ tqdm==4.59.0

To install all dependencies:
```
pip install -r requirements.txt
```

## Usage
Take BlogCatalog dataset for example, you can train and evaluate SL-GAD by executing:
```
python run.py --device cuda:0 --expid 1 --dataset BlogCatalog --runs 1 --auc_test_rounds 256 --alpha 1.0 --beta 0.6
```
For the specification on other datasets, please refer to our paper.