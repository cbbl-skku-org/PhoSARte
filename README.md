# PhoSARte

## Installation

We recommend using Conda to manage the environment for PhoSARte.

```bash
conda create -n phosarte python=3.9.19
conda activate phosarte

# Install PyTorch
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.7 -c pytorch -c nvidia

# Install Python dependencies
python -m pip install transformers==4.39.1 --no-cache-dir
python -m pip install numpy==1.26.4 pandas==2.2.1 scikit-learn==1.6.1 --no-cache-dir
python -m pip install sentencepiece==0.2.0 --no-cache-dir
python -m pip install termcolor==2.4.0 --no-cache-dir
```

### Pre-trained Protein Language Models (PLMs) Setup
You need to download the ProtTrans language models from HuggingFace and place them in a `ProtTrans_models` directory at the project root. 

Download from the following links:
- [prot_t5_xxl_uniref50](https://huggingface.co/Rostlab/prot_t5_xxl_uniref50)
- [prot_t5_xl_bfd](https://huggingface.co/Rostlab/prot_t5_xl_bfd)

Ensure the snapshot directories match the paths expected by the model:
```text
ProtTrans_models/models--Rostlab--prot_t5_xl_bfd/snapshots/7ae1d5c1d148d6c65c7e294cc72807e5b454fdb7
ProtTrans_models/models--Rostlab--prot_t5_xxl_uniref50/snapshots/31a40d7b55caf68d7a8a8dfd913b779b99dc09a9
```

## Input Format

The prediction script requires a CSV input file (without a header row). Each line must contain a peptide sequence (with a maximum length of 33 amino acids) and its corresponding label, separated by a comma. 

Labels:
- `1`: Positive
- `0`: Negative

**Example:**
```csv
EASLNKSKSATTTPSGSPRTSQQNVYNPSEGST,1
SNDSRSSLIRKRSTRRSVRGSQAQDRKLSTKEA,1
...
STATQHQTVVGDAVAETQHVLSKEDFLKLMLPD,0
DSNLKPEEVVHKEKRRTKSLLEEKLVLKSKSKT,0
```

## Running the Prediction

Use the `predict.py` script to run the end-to-end prediction pipeline, which extracts embeddings dynamically and predicts probabilities.

```bash
python predict.py --input <path_to_input_csv> [options]
```

**Arguments:**
- `--input` (Required): Path to the input CSV file.
- `--model_type`: Which PhoSARte model to use. Choices: `A549`, `VeroE6`, `Generic` (Default: `Generic`).
- `--model_dir`: Directory containing PhoSARte `.pt` model files (Default: `final_models`).
- `--device`: Device for computation, e.g., `cuda:0` or `cpu` (Default: automatically detects CUDA, otherwise CPU).
- `--batch_size`: Batch size for inference (Default: `128`).
- `--output`: Directory to save prediction results (Default: `results`).

**Example Usage:**
```bash
python predict.py --input data/Combined/Combined_Test.csv --model_type Generic --batch_size 128
# Results will be saved to results/PhoSARte_Generic_Results.csv
```