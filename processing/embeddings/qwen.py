from typing import List
import numpy as np
from torch import Tensor
import torch


## Based on Qwen's tokenizer
def chunk_by_characters(
    input_text: str, tokenizer: callable, chunk_char: List[str] = ["\n", "."]
):
    """
    Split the input text into sentences using the tokenizer
    :param input_text: The text snippet to split into sentences
    :param tokenizer: The tokenizer to use
    :return: A tuple containing the list of text chunks and their corresponding token spans
    """
    inputs = tokenizer(input_text, return_tensors="pt", return_offsets_mapping=True)
    char_tokens = [
        tokenizer.tokenize(char)[0] for char in chunk_char + ["<|endoftext|>"]
    ]
    token_offsets = inputs["offset_mapping"][0]
    token_ids = inputs["input_ids"][0]
    chunk_positions = [
        (i, end.item())
        for i, (token_id, (start, end)) in enumerate(zip(token_ids, token_offsets))
        if any(
            (token in tokenizer.convert_ids_to_tokens(token_id.item()))
            for token in char_tokens
        )
    ]
    chunks = [
        input_text[x[1] : y[1]]
        for x, y in zip([(1, 0)] + chunk_positions[:-1], chunk_positions)
    ]
    span_annotations = [
        (x[0], y[0]) for (x, y) in zip([(1, 0)] + chunk_positions[:-1], chunk_positions)
    ]
    return chunks, span_annotations


def get_query_embedding(tokenizer, model, query):
    inputs = tokenizer(query, return_tensors="pt")
    model_output = model(**inputs)
    return np.squeeze(
        last_token_pool(model_output.last_hidden_state, inputs["attention_mask"])
        .detach()
        .numpy()
    )


def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[
            torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths
        ]


if __name__ == "__main__":
    from transformers import AutoTokenizer, AutoModel

    example_text = """Hello world. How are you?
    What the sigma.

    YOUR NOT SIGMA
    """

    qwen_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-0.6B")
    qwen_model = AutoModel.from_pretrained("Qwen/Qwen3-Embedding-0.6B")
    chunks, span_annotations = chunk_by_characters(example_text, qwen_tokenizer)
    print("chunks: ", chunks)
    print("span_annotations: ", span_annotations)
    print(
        "query_embedding: ",
        get_query_embedding(qwen_tokenizer, qwen_model, "sigma"),
    )

    def cos_sim(x, y):
        return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

    print(
        "cos_sim: ",
        cos_sim(
            get_query_embedding(qwen_tokenizer, qwen_model, "sigma"),
            get_query_embedding(qwen_tokenizer, qwen_model, "sligmas"),
        ),
    )
