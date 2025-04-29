from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast




def initialize_tokenizer() -> PreTrainedTokenizer | PreTrainedTokenizerFast:

    return AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path="Babelscape/wikineural-multilingual-ner",
        use_fast=True  # 启用快速Tokenizer提升性能
    )
