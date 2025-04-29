import gc
import os

import torch

from dataset import build_dataset, preprocess_data
from model import initialize_model
from tokenizer import initialize_tokenizer
from trainer import build_trainer
from utils import not_change_test_dataset, set_random_seeds


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def main():

    set_random_seeds()


    model = initialize_model()


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n===== 设备信息 =====")
    print(f"可用GPU: {torch.cuda.is_available()}")
    print(f"当前设备: {device}")
    print(f"GPU名称: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else '无'}")


    model.to(device)

    tokenizer = initialize_tokenizer()

    raw_datasets = build_dataset()

    assert not_change_test_dataset(raw_datasets), "You should not change the test dataset"


    tokenized_datasets = preprocess_data(raw_datasets, tokenizer)


    trainer = build_trainer(
        model=model,
        tokenizer=tokenizer,
        tokenized_datasets=tokenized_datasets,
    )
    trainer.train()

    gc.collect()
    torch.cuda.empty_cache()
    torch.backends.cuda.cufft_plan_cache.clear()


    test_metrics = trainer.evaluate(
        eval_dataset=tokenized_datasets["test"],
        metric_key_prefix="test",
    )
    print("Test Metrics:", test_metrics)


if __name__ == "__main__":
    main()
