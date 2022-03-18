
from datasets import load_dataset

train_dataset = load_dataset("ccdv/cnn_dailymail", "3.0.0", split="validation[:5%]")
