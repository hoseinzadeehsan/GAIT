# GAIT: Graph Neural Network Approach to Semantic Type Detection in Tables

This repository contains source code of the paper: [Graph Neural Network Approach to Semantic Type Detection in Tables](https://arxiv.org/abs/2405.00123), published at Pacific-Asia Conference on Knowledge Discovery and Data Mining 2024.

## Structure of the repository
At the root folder of the project, you will see:

```text
├── data  #training, and testing data.
├── model
|  └──  configs.py  #stores address of current directory
|  └──  dataset.py  #handles loading dataset
|  └──  GAIT_GAT.py  # Applies GAT to the output of single-column prediction module
|  └──  GAIT_GCN.py  # Applies GCN to the output of single-column prediction module
|  └──  GAIT_GGNN.py  # Applies GGNN to the output of single-column prediction module
|  └──  gnn.py  # handles defining different GNNs
├── results  #classification report
├── saved_models_GAT  #GAT models are saved here
├── saved_models_GCN  #GCN models are saved here
├── saved_models_GGNN  #GGNN models are saved here
├── README.md
├── requirements.txt  #required Python library
├── ...
```

## Environment setup
The following instruction were tested on Python 3.10, NVIDIA Tesla V100, and also NVIDIA GeForce RTX 4090.
We recommend using a python virtual environment:
```
mkdir venv
virtualenv --python=python3 venv

#fill in $BASEPATH with the repository address
export BASEPATH=[path to the repo]
source venv/bin/activate
```
After activating the virtual environment, install required library packages:
```
cd $BASEPATH
pip3 install -r requirements.txt
```
Environment setup is done. Next time just simply run the following code to activate the virtual environment:
```
source venv/bin/activate
```
## Dataset
Input to the GAIT is the logit outputted by RECA. You can either download the  [here](https://drive.google.com/file/d/10jvl2nP2XvN6LalkG4bxbi_pnD8WVTy8/view?usp=sharing) and put the files in the data directory or use code of RECA to generate them.
Input should be a pickle file that can be opened with the following code:
```
with open(path to data, "rb") as f:
    data = pickle.load(f)
```
Also, the the format of input should be a numpy array with dimension of "Number of tables" by "maximum number of columns in a table". An example of input data containing only a table is as following where
"features" are the logits outputted by the single module prediction (RECA or any other model) which acts as the initial features of nodes in a GNN.
"labels": are labels of the columns inside the table.
"mask": includes value 1 for valid columns and 0 for invalid dataset_big_2columns
"table_id": Id of the table
```
[{'features': array([[-2.52733374, ... , -3.81204939],
        [ 2.22598338, ... , -3.21132755],
        [ 0, ..., 0],
        [ 0, ..., 0],
        [ 0, ..., 0]]), 'labels': array([ 5, 14, -1, -1, -1, -1]), 'masks': array([1, 1, 0, 0, 0, 0]), 'table_id': '0_1438042987171.38;warc;CC-MAIN-20150728002307-00339-ip-10-236-191-2.ec2.internal.json.gz_993-Vanderbilt University | Co_D2SH5A2V3AI2PNTZ'}]
]
```
## Single-column prediction
We use RECA as the single-column module of GAIT. Source code of RECA can be found [here](https://github.com/ysunbp/RECA-paper). Any other model capable of generating logit vector for each column can be used, too. To convert output of RECA for Semtab and Webtables dataset to the format compatible with the GNN we added `/model/multi_phase_classification.py` and `/model/multi_phase_classification.py` to the semtab/experiment and  of RECA's source code.

## Training
In order to train GAIT we use the following command with their corresponding parameters(Use the code for more information about parameters and select their values according to your wish):
```
cd $BASEPATH/model
python GAIT_GAT.py --data-name dataset_semtab_4 --classes 275 --epochs 100 --num-heads 1 --num-out-heads 1  --num-layers 1 --mode train
python GAIT_GCN.py --data-name dataset_semtab_4 --classes 275 --epochs 100  --num-layers 1 --mode train
python GAIT_GGNN.py --data-name dataset_semtab_4 --classes 275 --epochs 100  --num-layers 1 --mode train
```
## Evaluation using an existing model
To evaluate GAIT using one of the existing model, we should use `--mode eval`
