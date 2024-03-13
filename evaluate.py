from engine import evaluate
from dataloader.dataloader import fetch_dataloader 
from model.vit import vit 

from utils.misc import load_checkpoint




if __name__ == '__main__':
    test_dataloader = fetch_dataloader(['test'])['test']
    model = vit()
    model_dir = ""
    
    load_checkpoint(model, model_dir)
    
    test_metrics = evaluate(
        model,
        loss_fn=,
        metrics=,
        params=
    )