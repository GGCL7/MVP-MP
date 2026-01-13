# Welcome to DeepCPP: Integrating Multi-View Residue Graph and Protein Language Model for Cell-Penetrating Peptide Prediction via Global–Local Graph Aggregation and Cross-Attentive Fusion
Cell-penetrating peptides (CPPs) enable intracellular delivery, but large-scale experimental discovery remains costly and slow. Existing computational predictors often rely on handcrafted sequence descriptors or protein language-model embeddings, leaving gaps in biophysical grounding and interpretability. In this study, we present DeepCPP, a dual-branch framework that integrates a biophysically informed, multi-view residue graph with ESM-2 sequence embeddings. Specifically, the graph branch encodes sequence neighborhood and physicochemical similarity via global–local aggregation with top-k subgraph focusing and gated broadcast, while the ESM branch provides context-aware representations. The two views are aligned and fused by cross-attention with an HSIC-based decorrelation term, and a Kolmogorov–Arnold network head enhances nonlinear separability. We also curate a new benchmark from CPPsite3 and adopt cluster-controlled splits to reduce leakage and enable credible generalization. Experimental results indicate that DeepCPP outperforms state-of-the-art CPP and peptide-function prediction methods. Interpretability analyses highlight charge clustering, oriented amphipathicity, and termini patterns, offering actionable guidance for rational design. Overall, DeepCPP provides an accurate, interpretable, and scalable pre-screening tool for CPP discovery.

![The workflow of this study](https://github.com/GGCL7/DeepCPP/blob/main/workflow.png)


## 🔧 Installation instructions

1. **Clone the repository**
```bash
git clone https://github.com/GGCL7/DeepCPP.git
cd DeepCPP
```
2. **Set up the Python environment**
```bash
conda create -n deepcpp python=3.10
conda activate deepcpp
pip install -r requirements.txt
```
## Model Training

Train the model from scratch:

```bash
python train.py \
  --train_fasta "Data/train.txt" \
  --test_fasta  "Data/test.txt"  \
  --esm_dir "ESM_pre_model" \
  --esm_batch_size 32 --esm_use_fp16 \
  --batch_size 128 --lr 5e-4 --wd 1e-4 --max_epochs 100 \
  --use_weighted_ce --pos_alpha 1.8 \
  --gnn_node_in 48 --gnn_edge_in 3 --use_kan \
  --pH 7.4 --edge_mode hybrid --window 3 --knn_k 8 \
  --save_path "best_model.pth" --seed 2025

```
The training script will automatically save the model with the best validation **MCC** to `best_model.pth`.

## Model Evaluation

Evaluate the trained model:

```bash
python evaluation.py
```
The script reports the following metrics:

* Accuracy
* Sensitivity
* Specificity
* Matthews Correlation Coefficient (MCC)
* Area Under the Curve (AUC)


## 🛠️ Using DeepCPP for Cell-penetrating peptides prediction

```bash
python predict.py \    
    --test_fasta "Data/test.txt" \
    --model_path "best_model.pth" \
    --out_csv "predictions.csv"
```
## Output example:

```bash
id   : pos1
prob  : 0.930378
pred_label   : CPP

```
