import datasets
import torch
import transformers

import flashlight


def test_whisper():
    model = transformers.WhisperModel.from_pretrained("openai/whisper-tiny.en")
    model.eval()
    feature_extractor = transformers.AutoFeatureExtractor.from_pretrained("openai/whisper-tiny")
    ds = datasets.load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    inputs = feature_extractor(ds[0]["audio"]["array"], return_tensors="pt")
    input_features = inputs.input_features
    decoder_input_ids = torch.tensor([[1, 1]]) * model.config.decoder_start_token_id

    with flashlight.tracer.trace(run_torch=True):
        cached_forward = transformers.activations.GELUActivation.forward
        transformers.activations.GELUActivation.forward = lambda self, tensor: torch.nn.functional.gelu(tensor)

        flashlight_output = model(input_features=input_features, decoder_input_ids=decoder_input_ids).last_hidden_state

        transformers.activations.GELUActivation.forward = cached_forward

    assert tuple(flashlight_output.shape) == flashlight_output.lazy_tensor.shape
    assert len(flashlight_output.graph) == 863
