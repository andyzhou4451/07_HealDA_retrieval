import torch
import os

def load_forecast_ckpt(ckpt_dir, model_name, forecast_model):    
    checkpoint = torch.load(
        os.path.join(
            ckpt_dir, 
            model_name,
            "runs",
            "checkpoints",
            "best.ckpt"
        ), 
        map_location="cpu"
    )    
    
    forecast_model.load_state_dict(checkpoint["model_state_dict"], strict=False)

    return forecast_model

def load_obsop_ckpt(ckpt_dir, model_name, obsop_model):    
    checkpoint = torch.load(
        os.path.join(
            ckpt_dir, 
            model_name,
            "runs",
            "checkpoints",
            "best.ckpt"
        ), 
        map_location="cpu"
    )    
    
    obsop_model.load_state_dict(checkpoint["model_state_dict"], strict=False)

    return obsop_model