# Code for the processing of the text dataset and the creation of a torch Dataset
import torch

def load_file(filename):
    """
    Open a txt file and separate the content into text and emotion.
    ------
    Parameters:
        filename: str
        Path to the txt file

    ------
    Returns:
        list_texts: list
        list of the sentences
        list_emotions: list
        list of the emotion associated to each sentence
    """
    list_texts = []
    list_emotions = []
    with open(filename, mode='r', encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            text, emotion = line.split(sep=";") # each line is composed by a sentence and an associated emotion
            if text == "" or emotion == "":
                continue # skip empty lines
            list_texts.append(text.strip()) # remove the spaces at the beginning and at the end of the sentence
            list_emotions.append(emotion.strip()) # remove the \n at the end

    return list_texts, list_emotions


def tokenizer(text, threshold=None):
    """
    Convert a string corresponding to a sentence into a list of tokens, 
    which are the words of the sentence.
    ------
    Parameters:
        text: str
        A given sentence
        threshold: int
        maximal number of word in a sentence (to reduce computational cost).
        Default is None

    ------
    Returns:
        tokenized_txt: list
        list of the tokens of the sentence (words)
    """
    tokenized_txt = text.split(sep=" ") # the text is tokenized word by word.
    if threshold is not None:
        n_token = len(tokenized_txt)
        if n_token > threshold:
            # return tokenized_txt[:threshold] # trimming 
            return [] # bizarre de rogner ? il vaut mieux les enlever non ?
        else:
            n_pad_to_add = threshold - n_token
            tokenized_txt += ["<pad>"] * n_pad_to_add

    return tokenized_txt

def yield_tokens(full_text, threshold=None):
    """
    Convert a list of sentences into a tokenized version, 
    which is a list containings all the tokens in order of 
    apparition in the sentences
    ------
    Parameters:
        text: list
        list of sentences
        threshold: int
        maximal number of word in a sentence (to reduce computational cost).
        Default is None

    ------
    Returns:
        tokenized_txt: list
        list of the tokens of the sentences (words)
    """
    tokenized_full_text = []
    for text in full_text:
        tokenized_full_text += tokenizer(text, threshold=threshold)

    return tokenized_full_text


def build_vocab_from_iterator(list_tokens, specials=None):
    """
    Attribute a unique positive integer to each token such that there 
    are no duplicate (e.g. 'i' is not assigned to two different integer and 
    one integer cannot represent two words).
    ------
    Parameters:
        list_tokens: list
        list of all tokens present in the dataset
        specials: list
        list of special tokens not present in the dataset but necessary for 
        subsequent computations. Default is None. Useful specials are:
        - <pad> : token added to each sentence having a size inferior to 
        an arbitrary threshold.
        - <unk> : token given to every word in the dataset that appears 
        less than an arbitrary threshold.

    ------
    Returns:
        vocab: dict
        dictionnary associating an integer to each token. 
    """
    vocab = {}
    counter = 0
    if specials is not None:
        for special_token in set(specials):
            vocab[special_token] = counter
            counter += 1

    for token in list_tokens: # ici il n'y a pas de majuscule mais doit on les prendre en compte s'il y en a ?
        if token not in vocab.keys():
            vocab[token] = counter
            counter += 1

    return vocab

def yield_tokens_with_unknown(tokenized_txt, threshold):
    dico_words = {}
    for i in range(len(tokenized_txt)):
        word = tokenized_txt[i]
        dico_words[word] = dico_words.get(word, {"count": 0, "pos": []}) 
        dico_words[word]["count"] += 1
        dico_words[word]["pos"].append(i)

    new_tokenized_txt = tokenized_txt.copy()
    for word, info in dico_words.items():
        if info["count"] <= threshold:
            for k in info["pos"]:
                new_tokenized_txt[k] = "<unk>"

    return new_tokenized_txt
    

class EmotionsDataset(torch.utils.data.Dataset):
    """
    Class to convert a list of sentences into a one-hot encoded 
    representation used to create a torch Dataset.
    Each sentence is represented by a tensor of shape (sentence_length, n_tokens)
    where sentence_length is the number of words in the sentence (after padding)
    ------
    Parameters:
        tokenized_text: list
        list of tokenized sentences padded and with <unk> if necessary
        vocab: dict
        dictionnary associating an integer to each token. 
        n_tokens: int
        number of different tokens in the vocabulary
        nb_sentences: int
        number of sentences in the dataset
        tokenized_emotion: list
        list of tokenized emotions
        emotion_classes: list
        dictionnary associating an integer to each emotion class
    """
    def __init__(self, tokenized_text, vocab, nb_sentences, tokenized_emotion, emotion_classes):
        self.tokenized_text = tokenized_text # list of tokenized sentences padded and with <unk> if necessary
        self.vocab = vocab
        self.n_tokens = len(vocab)
        self.nb_sentences = nb_sentences

        self.tokenized_emotion = tokenized_emotion
        self.emotion_classes = emotion_classes
        
        self.one_hot_encoded_texts = self.one_hot_encoding()
        self.one_hot_encoded_emotions = self.one_hot_encoding_emotion()

    def one_hot_encoding(self):
        indices = [self.vocab.get(token, self.vocab["<unk>"]) for token in self.tokenized_text] # convert each token into its index in the vocab, if the token is not in the vocab, it is considered as <unk> 
        # (for exemple when we encounter a new work in the validation/test set that was not present in the training set)
        tensor_indices = torch.tensor(indices)
        one_hot_encoded = torch.nn.functional.one_hot(tensor_indices, num_classes=self.n_tokens).float()
        one_hot_encoded = one_hot_encoded.view(self.nb_sentences, -1, self.n_tokens)
        return one_hot_encoded # return a tensor of shape (n_sentences, sentence_length, n_tokens)
    
    def one_hot_encoding_emotion(self):
        indices = [self.emotion_classes.get(token) for token in self.tokenized_emotion] # convert each emotion into its index in the emotion_classes
        tensor_indices = torch.tensor(indices)
        one_hot_encoded = torch.nn.functional.one_hot(tensor_indices, num_classes=len(self.emotion_classes)).float()
        return one_hot_encoded # return a tensor of shape (n_sentences, n_emotion_classes)

    def __getitem__(self, index):
        """
        should return the one-hot encoded version of the sentence at the given index
        ------
        Parameters:
            index: int
            index of the sentence in the dataset
        ------
        Returns:
            one_hot_encoded_texts[index]: tensor
            one-hot encoded version of the sentence at the given index
            one_hot_encoded_emotions[index]: tensor
            one-hot encoded version of the emotion at the given index
        """
        return self.one_hot_encoded_texts[index], self.one_hot_encoded_emotions[index]
    
    def __len__(self):
        """
        should return the number of sentences in the dataset
        ------
        Returns:
            nb_sentences: int
            number of sentences in the dataset
        """
        return self.nb_sentences
    
    
