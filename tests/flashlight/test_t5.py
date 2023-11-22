import transformers

import flashlight


def test_t5_small():
    tokenizer = transformers.T5Tokenizer.from_pretrained("google/flan-t5-small")
    model = transformers.T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")

    input_text = "translate English to German: How old are you?"
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    decoder_input_ids = input_ids  # TODO: provide decoder_inputs_ids separately

    with flashlight.tracer.trace(run_torch=True):
        flashlight_output = model(input_ids, decoder_input_ids=decoder_input_ids).logits

    assert tuple(flashlight_output.shape) == flashlight_output.lazy_tensor.shape
    assert len(flashlight_output.graph) == 1462
