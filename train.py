import torch

import data
import llm
import model
import spsa

#################################
## CONSTANTS
#################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONFIG_BASIC_RUN = {
        "llm": "albert",
        "dataset": "imdb",
}


#################################
## Class definitions
#################################

  
#################################
## Private Functions
#################################

#################################
## Public Functions
#################################

def train(model, dataloader, loss_fn):
    total_loss = 0.0
    for batch in dataloader:
        loss = model.step(batch["input_ids"].to(device), batch["labels"].to(device))
        total_loss += loss.item()
        print(loss)
    
    return total_loss / len(dataloader)

#################################
## Test function
#################################

if __name__ == "__main__":

    # Example usage
    llm_model = llm.get_model("albert")
    loss_fn = torch.nn.CrossEntropyLoss()
    layers = [torch.nn.Linear(768, 256), torch.nn.Linear(256, 10)]

    model = model.Model(llm_model["model"], loss_fn, layers)
 

    dataset = data.get_dataset("imdb", llm_model["tokenizer"])

    model= spsa.SPSA(model, lr=0.01, delta=0.01, noise_factor=0.1, loss_fn=loss_fn)
    
    train(model, dataset, loss_fn)
