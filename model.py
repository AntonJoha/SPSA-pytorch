import torch
import llm

#################################
## CONSTANTS
#################################


#################################
## Class definitions
#################################


class Model(torch.nn.Module):
    def __init__(self, llm, loss_fn, layers, activation_fn=torch.nn.ReLU):
        super().__init__()
        self.llm = llm
        self.loss_fn = loss_fn

        self.layers = torch.nn.ModuleList()
        for layer in layers:
            self.layers.append(layer)
            self.layers.append(activation_fn())

    def forward(self, x):
        x = self.llm(x)
        for layer in self.layers:
            x = layer(x)
        return x

   
#################################
## Private Functions
#################################

#################################
## Public Functions
#################################

#################################
## Test function
#################################

if __name__ == "__main__":

    # Example usage
    llm_model = llm.get_model("albert")
    loss_fn = torch.nn.CrossEntropyLoss()
    layers = [torch.nn.Linear(768, 256), torch.nn.Linear(256, 10)]

    model = Model(llm_model["model"], loss_fn, layers)
    print(model)
 
