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
        
        self.grad_index = []
        self._llm_no_grad()

        self.layers = torch.nn.ModuleList()
        for layer in layers:
            self.layers.append(layer)
            self.layers.append(activation_fn())

    def forward(self, x):
        x = self.llm(x,output_hidden_states=True).hidden_states[-1][:, 0, :]
        for layer in self.layers:
            x = layer(x)
        return x

    def _llm_no_grad(self):
        for i, p in enumerate(self.llm.parameters()):
            if p.requires_grad:
                p.requires_grad = False
                self.grad_index.append(i)
        print(self.grad_index)

   
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
 
