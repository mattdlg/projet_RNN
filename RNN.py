import torch

class RNNcell(torch.nn.Module):
    """
    Recurrent Neural Network combination cell
    ------
    Parameters:
        emb_size: int
        Dimension of the word embedding
        hidden_size: int
        Dimension of the hidden state
        num_layers: int
        Number of recurrent layers. Default is 1
    """
    def __init__(self, emb_size, hidden_size, num_layers=1):
        super(RNNcell, self).__init__()
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.tanh = torch.nn.Tanh()

        # layers of the RNN
        self.i2h = torch.nn.Linear(self.emb_size,self.hidden_size)
        self.h2h = torch.nn.Linear(self.hidden_size,self.hidden_size)
        torch.nn.init.uniform_(self.i2h.weight,-0.001,0.001)
        torch.nn.init.uniform_(self.h2h.weight,-0.001,0.001)

    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_size) # initial hidden state

    def forward(self, x, hx=None, batch_first=False, recurrence=True):
        if batch_first:
            x = x.transpose(0, 1)
        seq_len, batch_size, _ = x.size()

        if not recurrence:
            seq_len = 1 # if we don't want to use the recurrence, we only feed the RNN one word

        if hx is None:
            hx = self.init_hidden(batch_size)
        h_t_minus_1 = hx.clone()
        h_t = hx.clone()
        output = []
        for t in range(seq_len): # we feed the RNN one word at a time
            for layer in range(self.num_layers):
                input_t = x[t] if layer == 0 else h_t[layer - 1]
                h_t[layer] = self.tanh(
                    self.i2h(input_t) + self.h2h(h_t_minus_1[layer])
                )
            output.append(h_t[-1].clone())
            h_t_minus_1 = h_t.clone()
        output = torch.stack(output)
        if batch_first:
            output = output.transpose(0, 1)

        return output, h_t

class RNN(torch.nn.Module):
    """
    Recurrent Neural Network architecture
    ------
    Parameters:
        input_size: int
        Dimension of the input (size of the one-hot encoding of each word)
        hidden_size: int
        Dimension of the hidden state
        emb_size: int
        Dimension of the word embedding
        output_size: int
        Dimension of the output (size of the one-hot encoding of each emotion class)
        num_layers: int
        Number of recurrent layers. Default is 1
    """
    def __init__(self, input_size, hidden_size, emb_size, output_size, num_layers=1):
        super(RNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.emb_size = emb_size
        self.output_size = output_size
        self.num_layers = num_layers

        ### Write here the activation function and layers
        # word embedding
        self.i2e = torch.nn.Linear(self.input_size, self.emb_size)
        torch.nn.init.uniform_(self.i2e.weight,-0.001,0.001)

        self.softmax = torch.nn.LogSoftmax(dim=1)

        # rnn cell
        self.rnn = RNNcell(self.emb_size, self.hidden_size, self.num_layers)

        # output layer
        self.h2o = torch.nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x, recurrence=True):
        """
        forward pass of the RNN
        ------
        Parameters:
            x: tensor
            input tensor of shape (batch_size, sentence_length, input_size)
            recurrence: bool
            if True, the RNN uses its recurrent connections, otherwise it doesn't (use a single word to predict the emotion)
        """
        x = self.i2e(x) # word embedding

        output, hidden = self.rnn(x, batch_first=True, recurrence=recurrence)
        output = self.h2o(output[:,-1,:]) # we only keep the output of the last time step
        output = self.softmax(output)
        return output


    
def train(dataloader, model, loss_fn, optimizer, device, reccurence=True, verbose=True):
    """
    Train the model for one epoch.
    ------
    Parameters:
        dataloader: DataLoader
        data loader for training
        model: torch.nn.Module
        neural network model
        loss_fn: torch.nn.Module
        loss function
        optimizer: torch.optim.Optimizer
        optimizer for the model
        device: torch.device
        device to use (cpu or cuda or xpu)
        recurrence: bool
        whether to use the recurrent connections of the RNN (default: True)
        verbose: bool
        whether to print the loss during training (default: True)
    ------
    Returns:
        avg_loss: float
        average loss over the epoch
    """
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.train()
    total_loss = 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X, recurrence=reccurence)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item() / len(X) # accumulate loss for averaging later
        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            if verbose:
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    
    avg_loss = total_loss / num_batches # average loss per epoch
    return avg_loss


def test(dataloader, model, loss_fn, device, recurrence=True, verbose=True):
    """
    Test the model on the test dataset.
    ------
    Parameters:
        dataloader: DataLoader
        data loader for testing
        model: torch.nn.Module
        neural network model
        loss_fn: torch.nn.Module
        loss function  
        device: torch.device
        device to use (cpu or cuda or xpu)
        recurrence: bool
        whether to use the recurrent connections of the RNN (default: True)
        verbose: bool
        whether to print the accuracy and loss during testing (default: True)
    ------
    Returns:
        correct: float
        accuracy of the model on the test dataset
        test_loss: float
        average loss of the model on the test dataset
    """
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X, recurrence=recurrence)
            test_loss += loss_fn(pred, y).item()
            correct += (torch.argmax(y,1) == torch.argmax(pred,1)).sum().item()
    test_loss /= num_batches
    correct /= size
    if verbose:
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return correct, test_loss

def early_stopping(best_score, score, threshold=0.001, patience=5, counter=0):
    """
    Early stopping to prevent overfitting.
    ------
    Parameters:
        best_score: float
        best score (accuracy here) obtained so far
        score: float
        current score
        threshold: float
        minimum improvement to consider (default: 0.001 = 0.1%)
        patience: int
        number of epochs to wait for improvement (default: 5)
        counter: int
        number of epochs without improvement
    ------
    Returns:
        stop: bool
        whether to stop training or not
    """
    if score < best_score + threshold:
        counter += 1
        if counter >= patience:
            print("Early stopping")
            return True, best_score, counter
    else:
        best_score = score
        counter = 0
    return False, best_score, counter