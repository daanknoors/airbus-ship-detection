import torch
from pathlib import Path
from airbus_ship_detection import models
from airbus_ship_detection import configs
from airbus_ship_detection import trainer


def load_model(model_name, run_id):
    """
    Load a trained model from the specified path.
    """
    model_path =configs.DIR_MODELS / f"{model_name}_{run_id}.pt"
    model = models.MODELS[model_name]
    state = torch.load(str(model_path))
    state = {key.replace('module.', ''): value for key, value in state['model'].items()}
    model.load_state_dict(state)

    device = trainer.get_torch_device()
    model = model.to(device)
    model.eval()
    return model

def predict_mask(model, input_tensor):
    """
    Predict the ship mask for a given input tensor using the provided model.
    """
    if not isinstance(input_tensor, torch.Tensor):
        input_tensor = torch.from_numpy(input_tensor).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    
    with torch.no_grad():
        output = model(input_tensor)
        output = torch.sigmoid(output).squeeze().cpu().numpy()
        mask = (output > 0.5).astype('uint8') * 255  # Convert to binary mask
    return mask