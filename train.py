import torch
import model
import data
import llm


#################################
## CONSTANTS
#################################

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

def train(model, dataloader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        inputs = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    
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
    print(model)
 

    dataset = data.get_dataset("imdb")

