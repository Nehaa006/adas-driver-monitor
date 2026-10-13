import torch
from torchvision import models
import torch.nn as nn

def export_eye_to_onnx():
    pth_path = "models/eye_classifier.pth"
    onnx_path = "models/eye_classifier.onnx"

    # Re-create MobileNetV2 architecture
    model = models.mobilenet_v2(pretrained=False)
    model.classifier[1] = nn.Linear(model.last_channel, 2)
    model.load_state_dict(torch.load(pth_path, map_location="cpu"))
    model.eval()

    # Dummy tensor matching input dimensions (Batch, Channels, Height, Width)
    dummy_input = torch.randn(1, 3, 64, 64)

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
    )
    print(f"Eye classification model successfully exported to: {onnx_path}")

if __name__ == "__main__":
    export_eye_to_onnx()