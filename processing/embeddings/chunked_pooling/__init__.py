def chunk_by_characters(input_text: str, tokenizer: callable, chunk_char: str = "."):
    """
    Split the input text into sentences using the tokenizer
    :param input_text: The text snippet to split into sentences
    :param tokenizer: The tokenizer to use
    :return: A tuple containing the list of text chunks and their corresponding token spans
    """
    inputs = tokenizer(input_text, return_tensors="pt", return_offsets_mapping=True)
    char_id = tokenizer.convert_tokens_to_ids(chunk_char)
    sep_id = tokenizer.convert_tokens_to_ids("[SEP]")
    token_offsets = inputs["offset_mapping"][0]
    token_ids = inputs["input_ids"][0]
    chunk_positions = [
        (i, int(start + 1))
        for i, (token_id, (start, end)) in enumerate(zip(token_ids, token_offsets))
        if token_id == char_id
        and (
            token_offsets[i + 1][0] - token_offsets[i][1] > 0
            or token_ids[i + 1] == sep_id
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


def chunk_by_newline(input_text: str, tokenizer: callable):
    return chunk_by_characters(input_text, tokenizer, chunk_char="\\n")


def chunk_by_sentences(input_text: str, tokenizer: callable):
    return chunk_by_characters(input_text, tokenizer, chunk_char=".")


def chunked_pooling(
    model_output: "BatchEncoding", span_annotation: list, max_length=None
):
    token_embeddings = model_output[0]
    outputs = []
    for embeddings, annotations in zip(token_embeddings, span_annotation):
        if (
            max_length is not None
        ):  # remove annotations which go bejond the max-length of the model
            annotations = [
                (start, min(end, max_length - 1))
                for (start, end) in annotations
                if start < (max_length - 1)
            ]
        pooled_embeddings = [
            embeddings[start:end].sum(dim=0) / (end - start)
            for start, end in annotations
            if (end - start) >= 1
        ]
        pooled_embeddings = [
            embedding.float().detach().cpu().numpy() for embedding in pooled_embeddings
        ]
        outputs.append(pooled_embeddings)

    return outputs


if __name__ == "__main__":
    from transformers import AutoTokenizer

    example_text = """Hello world. How are you. \n
    What the sigma.

    YOUR NOT SIGMA
    """

    tokenizer = AutoTokenizer.from_pretrained(
        "jinaai/jina-embeddings-v3", trust_remote_code=True
    )
    chunks, span_annotations = chunk_by_characters(example_text, tokenizer)
    print("chunks: ", chunks)
    print("span_annotations: ", span_annotations)
