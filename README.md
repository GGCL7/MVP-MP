# Welcome to MVP-MP: a Multi-view Graph Learning Framework Augmented by Pathway-aware Representation Losses for Multi-label Metabolic Pathway Prediction
Accurate inference of metabolic pathway categories for unannotated small molecules is a key step for pathway-centric metabolomics interpretation and is increasingly relevant to drug discovery and lead optimization. 
In this study, we propose MVP-MP, a multi-view graph learning framework augmented by pathway-aware representation losses for multi-label metabolic pathway prediction. MVP-MP integrates a molecular graph branch and a fingerprint branch through a Bi-Gate fusion module for adaptive cross-view integration. In the graph branch, MVPool scores node importance from feature, structure, and diffusion perspectives, selects a top-k induced subgraph, and performs subgraph-to-global refinement, enabling explicit substructure focusing and propagating substructure evidence into global representations. At the objective level, we train the model end to end with a multi-label classification loss together with two pathway-aware regularizers. A pathway-aware contrastive loss exploits label overlap to structure compound relations in the embedding space, while a prototype-aware loss forms pathway prototypes within each mini-batch to promote pathway-discriminative, pathway-centric embeddings. In performance evaluation, MVP-MP consistently outperforms existing state-of-the-art methods on the independent test set. Moreover, prototype-centered latent analysis and MVPool-driven substructure interpretation provide pathway-centric interpretability by linking predictions to enriched substructures and reusable pathway-associated chemotypes.

![The workflow of this study](https://github.com/GGCL7/MVP-MP/blob/main/workflow.png)


## 🔧 Installation instructions

1. **Clone the repository**
```bash
git clone https://github.com/GGCL7/MVP-MP.git
cd MVP-MP
```
2. **Set up the Python environment**
```bash
conda create -n mvpmp python=3.10
conda activate mvpmp
pip install -r requirements.txt
```
## Model Training

Train the model from scratch:

```bash
  python main.py \
    --csv_path Data/kegg_dataset.csv \
    --index_path Data/data_index.txt \
    --num_labels 11 \
    --batch_size 64 \
    --epochs 200 \
    --lr 1e-3 \
    --seed 42 \
    --lambda_contrast 0.7 \
    --lambda_proto 0.4 \
    --select_metric f1 \
    --ckpt_path best_model.pth

```
The training script will automatically save the model with the best validation **F1** to `best_model.pth`.

## Model Evaluation

Evaluate the trained model:

```bash
python evaluation.py
```
The script reports the following metrics:

* Accuracy
* Precision
* Recall
* F1 score

