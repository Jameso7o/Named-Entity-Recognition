from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict, load_dataset


def align_labels_with_tokens(
    tag_list: list[int],
    token_indices: list[int | None],
) -> list[int]:
    synced_tags = []
    last_token_idx = None
    
    for curr_token_idx in token_indices:
        if curr_token_idx is None:
            synced_tags.append(-100)
        elif curr_token_idx != last_token_idx:
            synced_tags.append(tag_list[curr_token_idx])
        else:

            if tag_list[curr_token_idx] % 2 == 1:
                synced_tags.append(tag_list[curr_token_idx] + 1)
            else:
                synced_tags.append(tag_list[curr_token_idx])
        last_token_idx = curr_token_idx
    
    return synced_tags


def tokenize_and_align_labels(sample_data: dict, token_processor) -> dict:
    processed_tokens = token_processor(
        sample_data["tokens"],
        truncation=True,
        is_split_into_words=True,
    )
    
    # 获取所有标签并同步
    tag_sequences = sample_data["ner_tags"]
    synced_tag_sequences = [
        align_labels_with_tokens(
            tag_list=tags,
            token_indices=processed_tokens.word_ids(idx),
        )
        for idx, tags in enumerate(tag_sequences)
    ]
    
    processed_tokens["labels"] = synced_tag_sequences
    return processed_tokens


def build_dataset() -> DatasetDict | Dataset | IterableDatasetDict | IterableDataset:

    data_collection = load_dataset('tomaarsen/MultiCoNER', 'multi')
    

    test_split = load_dataset('tomaarsen/MultiCoNER', 'multi', split="test")
    

    if "validation" not in data_collection:

        train_validation_split = data_collection["train"].train_test_split(
            test_size=0.1,
            seed=42
        )
        data_collection = DatasetDict({
            "train": train_validation_split["train"],
            "validation": train_validation_split["test"],
            "test": test_split
        })
    else:
        data_collection["test"] = test_split
    
    return data_collection


def preprocess_data(data_collection: DatasetDict, token_processor) -> DatasetDict:

    processed_data = data_collection.map(
        function=lambda samples: tokenize_and_align_labels(
            sample_data=samples,
            token_processor=token_processor,
        ),
        batched=True,
        remove_columns=data_collection["train"].column_names,
    )
    
    return processed_data
