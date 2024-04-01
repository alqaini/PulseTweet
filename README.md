# PyTorch Financial News Sentiment Analysis with SageMaker

This project demonstrates how to train and evaluate a sentiment analysis model using PyTorch on AWS SageMaker. The model is trained to classify the sentiment of financial news articles as positive or negative. It leverages the Hugging Face `transformers` library for training and inference.

## Project Structure

- `train.py`: Script for training the sentiment analysis model using PyTorch and Hugging Face `transformers`.
- `testing.py`: Script for evaluating the trained model on a test dataset.
- `notebook.ipynb`: Jupyter notebook for data preparation, cleaning, initiating training, and testing jobs.

## Setup

### Requirements

- AWS account with access to SageMaker, S3, and IAM.
- SageMaker Notebook Instance or compatible local environment with Jupyter.
- Python 3.x and necessary libraries: `datasets`, `transformers`, `pandas`, `torch`.

### Instructions

1. **Prepare the Environment**: If using SageMaker, create a Notebook Instance and open `notebook.ipynb`. If locally, ensure your environment meets the requirements listed above.

2. **Data Preparation**: Run the cells in `notebook.ipynb` to download and preprocess the data from the Huggingface `datasets`.

3. **Training**: Run the training job by executing the cell in `notebook.ipynb` that configures and starts a SageMaker training job with `train.py`.

4. **Model Evaluation**: After training, evaluate your model on the test set by running the testing job configured in `notebook.ipynb`, which utilizes `testing.py`.

## Configuration

- Model and tokenizer are based on `DistilBert`.
- Data is sourced from the Hugging Face `datasets` library, specifically the `zeroshot/twitter-financial-news-topic`.

## Running the Training Job

Configure the SageMaker PyTorch estimator in `notebook.ipynb` and specify `train.py` as the entry point. Example:

```python
from sagemaker.pytorch import PyTorch

pytorch_estimator = PyTorch(
    entry_point='train.py',
    role=sagemaker.get_execution_role(),
    framework_version='1.7.1',
    py_version='py3',
    instance_count=1,
    instance_type='ml.m5.xlarge',
    hyperparameters={'epochs': 1},
    use_spot_instances=True,
    max_run=3600,
    max_wait=7200
)

pytorch_estimator.fit()
