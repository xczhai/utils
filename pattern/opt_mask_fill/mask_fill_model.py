import torch

from openvino.tools.mo import convert_model
from openvino.runtime import serialize
import pdb


class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, attention_scores, attention_mask):
        # Apply masking to attention scores using torch.masked_fill
        out = torch.masked_fill(attention_scores, attention_mask, torch.finfo(attention_scores.dtype).min)
        return out 

def export_to_onnx(model, output_file_path, dummy_input_scores, dummy_input_mask):
    print("== PYTORCH --> ONNX --> OPENVINO")
    # Export the model to ONNX
    with torch.no_grad():
        torch.onnx.export(
            model,
            (dummy_input_scores, dummy_input_mask),
            output_file_path,
            input_names=["input_scores", "input_mask"],
            output_names=["output"],
            verbose=True,
        )
    ov_model = convert_model(output_file_path)
    serialize(ov_model, "onnx/ov_model.xml", "onnx/ov_model.bin")

def export_to_pt(model, output_file_path, dummy_input_scores, dummy_input_mask):
    print("== PYTORCH --> OPENVINO")
    ov_model = convert_model(model, example_input=(dummy_input_scores, dummy_input_mask))
    serialize(ov_model, "{}/ov_model.xml".format(output_file_path), "{}/ov_model.bin".format(output_file_path))

if __name__ == "__main__":
    batch_size = 4
    num_heads = 8
    sequence_length = 10

    # Create an instance of the AttentionMasking class
    model = SimpleModel()
    model.eval()

    # Define the attention scores and attention mask
    attention_scores = torch.rand((batch_size, num_heads, sequence_length, sequence_length))
    attention_mask = torch.randint(2, size=(batch_size, 1, sequence_length, sequence_length), dtype=torch.bool)

    # Get the masked attention weights
    outputs = model(attention_scores, attention_mask)
    # attn_weights = attention_masking.mask_attention_scores(attention_scores, attention_mask)

    # Print the masked attention weights
    print("Masked Attention Weights:")
    print(outputs)
    export_to_onnx(model, "simple_masked_fill.onnx", attention_scores, attention_mask)
    # traced = torch.jit.trace(model, example_inputs=(attention_scores, attention_mask))
    # pdb.set_trace()
    # ov_model = convert_model(model)
    export_to_pt(model, "pt", attention_scores, attention_mask)
    # ov_model = convert_model(model, example_input=(attention_scores, attention_mask))
    # serialize(ov_model, "pt/ov_model.xml", "pt/ov_model.bin")
