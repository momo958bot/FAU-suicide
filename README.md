# FAU-suicide

# FAU-suicide

Run on the AutoDL platform with an RTX 4090, using PyTorch 2.7.0 / Python 3.12 (Ubuntu 22.04) / CUDA 12.8.

## Environment Setup

```bash
# 1. Create a new environment named rsd_env, specifying Python version 3.10 (for the best compatibility)
conda create -n rsd_env python=3.10 -y

# 2. Activate the environment
conda init bash
source ~/.bashrc
conda activate rsd_env

# 3. Install core packages for data processing, machine learning, visualization, and Jupyter
pip install jupyter ipykernel pandas numpy scikit-learn xgboost matplotlib seaborn nltk

# 4. Pre-install deep learning related packages (to prepare for subsequent BERT tasks)
pip install torch torchvision torchaudio transformers evaluate

# 5. Add this Conda environment to the Jupyter kernel to enable switching on the web interface
python -m ipykernel install --user --name rsd_env --display-name "Python (RSD_ENV)"

# 6. Training
python train_DeBERTa.py

# Alternatively, run in the background
nohup bash -c "python train_DeBERTa.py; sleep 180; shutdown" > train_output.log 2>&1 &

# View the logs
tail -f train_output.log

# 7. Install the GCN environment
pip install torch_geometric networkx
