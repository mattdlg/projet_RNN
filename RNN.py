import torch
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

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

    def init_hidden(self, batch_size, device=None):
        if device is None:
            device = next(self.parameters()).device
        # shape: (num_layers, batch_size, hidden_size)
        return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device) # initial hidden state

    def forward(self, x, hx=None, batch_first=False, recurrence=True):
        if batch_first:
            x = x.transpose(0, 1)
        seq_len, batch_size, _ = x.size()

        if not recurrence:
            seq_len = 1 # if we don't want to use the recurrence, we only feed the RNN one word

        if hx is None:
            hx = self.init_hidden(batch_size, device=x.device)
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
    def __init__(self, input_size, hidden_size, emb_size, output_size, num_layers=1, device=None):
        super(RNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.emb_size = emb_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.device = device if device is not None else torch.device("cpu")

        ### activation function and layers
        # word embedding
        self.i2e = torch.nn.Linear(self.input_size, self.emb_size)

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
        """if not recurrence:
            # we only used a single word, the tensor is in 2D instead of 3D
            output = output.unsqueeze(1)"""
        output = self.h2o(hidden[0]) # we only keep the output of the last time step
        if output.dim() == 1:
            output = output.unsqueeze(0)
        # output = self.softmax(output)
        return output

# -------------------------
# 4) RNN Elman (mot par mot)
# -------------------------
class ElmanRNN(torch.nn.Module):
    """
    RNN Elman simple :
    - input must be of size (batch, input_size) = (batch, vocab_size) : one-hot per mot
    - i2h: Linear(input_size + hidden_size, hidden_size)
    - i2o: Linear(input_size + hidden_size, output_size)
    We will implement a helper forward_sequence that consumes (batch, seq_len) ids,
    converts to one-hot per time step (on the fly) and iterates timestep-wise.
    """
    def __init__(self, input_size: int, hidden_size: int, emb_size: int, output_size: int, num_layers: int=1, device=None):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.emb_size = emb_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.device = device if device is not None else torch.device("cpu")

        self.i2e = torch.nn.Linear(self.input_size, self.emb_size)
        self.i2h = torch.nn.Linear(self.emb_size + self.hidden_size, self.hidden_size)
        self.i2o = torch.nn.Linear(self.emb_size + self.hidden_size, self.output_size)

        self.activation = torch.tanh

    def init_hidden(self, batch_size: int, device=None):
        # allow overriding device to match the input tensors and avoid device mismatches
        if device is None:
            device = self.device
        return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)

    def step(self, embedded_input: torch.Tensor, hidden_prev: torch.Tensor):
        """
        embedded_input: (batch, emb_size)
        hidden_prev: (batch, hidden_size)
        returns: logits (batch, output_size), hidden_new (batch, hidden_size)
        """
        combined = torch.cat([embedded_input, hidden_prev], dim=1)
        hidden_new = self.activation(self.i2h(combined))
        logits = self.i2o(combined)  # raw logits
        return logits, hidden_new

    def forward_sequence(self, onehot_ids_batch: torch.LongTensor, lengths: torch.LongTensor, batch_first: bool=True, recurrence=True):
        """
        onehot_ids_batch: (batch, seq_len, vocab_size) LongTensor (one-hot encoded token ids)
        lengths: (batch,) actual lengths
        Returns:
          logits_last: (batch, output_size) logits computed at last real token for each sequence
          hidden_last: (batch, hidden_size)
        Procedure: iterate timestep t = 0 .. seq_len-1,
          at each step build one-hot vector for the batch for that timestep,
          call step(...), store logits; finally select logits at last real token per sequence.
        """
        if batch_first:
            onehot_ids_batch = onehot_ids_batch.transpose(0, 1)  # (seq_len, batch, vocab_size)
        seq_len, batch_size, _ = onehot_ids_batch.size()

        # ensure we create hidden on the same device as the input
        input_device = onehot_ids_batch.device

        if not recurrence:
            # if we don't want to use the recurrence, we only feed the RNN one word
            seq_len = 1

        hidden = self.init_hidden(batch_size, device=input_device)
        h_t_minus_1 = hidden.clone()
        h_t = hidden.clone()
        all_logits = []

        for t in range(seq_len):
            for layer in range(self.num_layers):
                input_t = self.i2e(onehot_ids_batch[t]) if layer == 0 else h_t[layer - 1]
                logits_t, h_t[layer] = self.step(input_t, h_t_minus_1[layer])
                
            all_logits.append(logits_t.unsqueeze(0))  # (1, batch, out)
            h_t_minus_1 = h_t.clone()

        all_logits = torch.cat(all_logits, dim=0)  # (seq_len, batch, out)
        
        if batch_first:
            all_logits = all_logits.transpose(0, 1)  # (batch, seq_len, out)

        # If recurrence is disabled, all sequences are treated as length 1 (we used only the first token)
        if not recurrence:
            lengths = torch.ones(batch_size, dtype=lengths.dtype, device=input_device)

        # For each sample in batch pick the logits at index lengths[i]-1 (last real token)
        last_indices = (lengths - 1).clamp(min=0).unsqueeze(1).unsqueeze(2).expand(batch_size, 1, self.output_size).to(input_device)
        logits_last = all_logits.gather(dim=1, index=last_indices).squeeze(1)  # (batch, out)
        
        if self.num_layers == 1:
            return logits_last, h_t.squeeze(0)
        return logits_last, h_t[-1]

def compute_class_weights(labels_list):
    """
    Compute class weights to handle class imbalance.
    ------
    Parameters:
        labels_list: list
        list of one-hot encoded labels (tensors) of shape (n_emotion_classes,)
    ------
    Returns:
        class_weights: torch.Tensor
        tensor of shape (n_emotion_classes,) containing the class weights
    """
    num_classes = labels_list.size(1)
    class_counts = labels_list.sum(dim=0)
    total = class_counts.sum()
    class_weights = total / (num_classes * class_counts) 
    class_weights[torch.isnan(class_weights)] = 0.0  # replace NaN with 0
    return class_weights.float()


def compute_sentence_lengths(seq, vocab):
    """
    Compute the length of sentences in a batch (i.e. number of non-pad tokens)
    ------
    Parameters:
        batch: list
        list of tuples (one_hot_encoded_sentence, one_hot_encoded_emotion)
        where one_hot_encoded_sentence is a tensor of shape 
        (sentence_length, n_tokens) and one_hot_encoded_emotion is a tensor of shape
        (n_emotion_classes,)
    ------
    Returns:
        lengths: torch.LongTensor
        tensor of shape (batch_size,) containing the lengths of each sentence in the batch
    """
    length = None
    for i in range(len(seq)):
        if seq[i, vocab["<pad>"]] == 1: # if the element at the index of <pad> is 1, this word is a pad and sentence ends here
            length = i # the length of the sentence is i
            break
    return length if length else len(seq) # if there is no pad, the length of the sentence is the full length

def collate_fn(batch,vocab):
    seqs, labels = zip(*batch)
    lengths = torch.tensor([compute_sentence_lengths(seq, vocab) for seq in seqs], dtype=torch.long)
    seqs = torch.stack(seqs)
    labels = torch.stack(labels)
    return seqs, labels, lengths


def train_epoch(model, loader, optimizer, criterion, device, recurrence=True):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    for batch in loader:
        ids, labels, lengths = batch
        ids = ids.to(device)
        labels = labels.to(device)
        lengths = lengths.to(device)

        optimizer.zero_grad()

        # If labels are one-hot, convert to class indices for loss/metrics
        if labels.dim() > 1:
            labels_idx = labels.argmax(dim=1)
        else:
            labels_idx = labels

        # Compute prediction error
        logits, _ = model.forward_sequence(ids, lengths, recurrence=recurrence)
        loss = criterion(logits, labels_idx)

        # Backpropagation
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * ids.size(0)
        preds = logits.argmax(dim=1).detach().cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels_idx.detach().cpu().numpy().tolist())

    avg_loss = running_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, acc


def eval_epoch(model, loader, criterion, device, recurrence=True):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in loader:
            ids, labels, lengths = batch
            ids = ids.to(device)
            labels = labels.to(device)
            lengths = lengths.to(device)
            
            # If labels are one-hot, convert to class indices for loss/metrics
            if labels.dim() > 1:
                labels_idx = labels.argmax(dim=1)
            else:
                labels_idx = labels

            # Compute prediction error
            logits, _ = model.forward_sequence(ids, lengths, recurrence=recurrence)
            loss = criterion(logits, labels_idx)

            running_loss += loss.item() * ids.size(0)
            preds = logits.argmax(dim=1).detach().cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels_idx.detach().cpu().numpy().tolist())

    avg_loss = running_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, acc, all_labels, all_preds


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

def plot_learning(history):
    epochs = list(range(1, len(history["train_loss"]) + 1))
    plt.figure()
    plt.plot(epochs, history["train_loss"], marker='o', label="train loss")
    plt.plot(epochs, history["val_loss"], marker='o', label="val loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.title("Loss")
    plt.show()

    plt.figure()
    plt.plot(epochs, history["train_acc"], marker='o', label="train acc")
    plt.plot(epochs, history["val_acc"], marker='o', label="val acc")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend(); plt.title("Accuracy")
    plt.show()

def plot_confusion(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,5))
    plt.imshow(cm, interpolation='nearest')
    plt.colorbar()
    plt.xticks(range(len(labels)), labels, rotation=45)
    plt.yticks(range(len(labels)), labels)
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Confusion Matrix")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i,j], ha='center', va='center', color='white' if cm[i,j] > cm.max()/2 else 'black')
    plt.tight_layout()
    plt.show()

def visualize_embeddings_pca_tsne(emb_matrix, itos: dict, top_n: int = 200):
    """
    Visualize word embeddings using PCA and t-SNE.
    ------
    Parameters:
        emb_matrix: torch.tensor
        embedding matrix containing the word embeddings
        itos: dict
        dictionary mapping indices to words
        top_n: int
        number of top words to visualize (default: 200)
    """
    n = min(top_n, emb_matrix.shape[0])
    X = emb_matrix[:n]
    pca = PCA(n_components=2)
    Xp = pca.fit_transform(X)
    plt.figure(figsize=(8,8))
    plt.scatter(Xp[:,0], Xp[:,1])
    for i in range(n):
        plt.annotate(itos[i], (Xp[i,0], Xp[i,1]), fontsize=7)
    plt.title("Embeddings PCA")
    plt.show()

    tsne = TSNE(n_components=2, init='pca', learning_rate='auto', perplexity=30)
    Xt = tsne.fit_transform(X)
    plt.figure(figsize=(8,8))
    plt.scatter(Xt[:,0], Xt[:,1])
    for i in range(n):
        plt.annotate(itos[i], (Xt[i,0], Xt[i,1]), fontsize=7)
    plt.title("Embeddings t-SNE")
    plt.show()
