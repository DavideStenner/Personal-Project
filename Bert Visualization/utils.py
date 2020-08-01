import transformers
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from tqdm.notebook import tqdm
import tensorflow as tf
from torch.autograd import grad
import torch.nn as nn
import nlp
import pandas as pd
from sklearn.model_selection import train_test_split
import gc
import matplotlib.pyplot as plt
import re
from IPython.core.display import display
import IPython
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import seaborn as sns
import numpy as np


MAX_LEN = 150

def load_dataset_sentiment():
    """Load sentiment140 dataset with nlp package

    @return: test: (pd.DataFrame) dataset with text of the tweet and sentiment as column
    """
    #We will use only train dataset because it is very large.
    sentiment140 = nlp.load_dataset('sentiment140', split = 'train')

    #Import sentiment column to define the split
    strat_supp = pd.DataFrame(sentiment140['sentiment']).values

    #With two different split we will take the needed amount of data
    _, supp, _strat, strat = train_test_split(pd.DataFrame(sentiment140).loc[:, ['text', 'sentiment']], strat_supp, test_size = .625, random_state = 3, shuffle = True, stratify = strat_supp)
    train, test = train_test_split(supp, test_size = .2, random_state = 3, shuffle = True, stratify = strat)

    #clean test dataset by removing error and url and correct sentiment format by changing 4 --> 1
    text_test = [text_preprocessing(x) for x in test['text'].copy()]
    sentiment_test = test['sentiment'].replace({4:1}).values
    
    test = pd.DataFrame({'text': text_test, 'sentiment': sentiment_test})
    
    #Remove blank tweet
    test = test.loc[[len(x)>0 for x in test.text]].reset_index(drop = True)
    
    return test

def call_html():
  display(IPython.core.display.HTML('''
        <script src="/static/components/requirejs/require.js"></script>
        <script>
          requirejs.config({
            paths: {
              base: '/static/base',
              "d3": "https://cdnjs.cloudflare.com/ajax/libs/d3/3.5.8/d3.min",
              jquery: '//ajax.googleapis.com/ajax/libs/jquery/2.0.0/jquery.min',
            },
          });
        </script>
        '''))
        
def get_embedding_matrix(model):
    """Calculate 3 embedding matrix for bert model
    @param model: (BertForSequenceClassification) bert model to extract embedding matrix
    
    @return word_emb: (torch.tensor) word embedding
    @return pos_emb: (torch.tensor) position embedding
    @return sent_emb: (torch.tensor) position of sentence embedding
    """
    word_emb, pos_emb, sent_emb = model.bert.embeddings.word_embeddings.weight, model.bert.embeddings.position_embeddings.weight, model.bert.embeddings.token_type_embeddings.weight
    return word_emb, pos_emb, sent_emb

def text_preprocessing(text):
    """
    - Remove entity mentions (eg. '@united')
    - Correct errors (eg. '&amp;' to '&')
    - Remove url (eg. 'https://...', 'http://...', 'www...' to '')
    @param    text (str): a string to be processed.
    @return   text (Str): the processed string.
    """


    # Remove '@name'
    text = re.sub(r'(@.*?)[\s]', ' ', text)

    # Replace '&amp;' with '&'
    text = re.sub(r'&amp;', '&', text)

    # Remove trailing whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    #remove url with http
    text = re.sub(r'https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)

    #remove url with www
    text = re.sub(r'www\..*[\r\n]*', '', text, flags=re.MULTILINE)

    return text


def embedding_pipeline(sentence, tokenizer, model):
    """Calculate input_embedding which is differantiable

    @param sentence: (str) sentence respect which calculate embedding matrix
    @param tokenizer: (BertTokenizer)  tokenizer used for tokenization
    @param model: (BertForSequenceClassification) model used for classification

    @return inputs_embeds: (torch.tensor) embedding matrix (in which word in one hot form are differentiable) used by model
    @return attention_mask: (torch.tensor) tensor of attention mask to pass to model during inference
    @return token_ids_tensor_one_hot: (torch.tensor) tensor in one hot form to calculate gradient relative to the token and not embedding
    @return token_words: (list) list of word
    @return token_types: (list) list of token type ids
    """
    
    #Tokenize the sentence
    encoded_tokens =  tokenizer.encode_plus(
    sentence, 
    add_special_tokens=True, return_token_type_ids=True,
    return_tensors="pt",
    max_length=MAX_LEN,             # Max length to truncate/pad
    truncation = True              # Truncate sentence to MAX_LEN
    )

    #Get token index and sentence index from input sentence
    token_ids = list(encoded_tokens["input_ids"].numpy()[0])
    token_type_ids = list(encoded_tokens["token_type_ids"].numpy()[0])

    #Create position array
    position_ids = torch.arange(len(token_ids), dtype=torch.long)
    
    #Create token, position and sentence embedding
    embedding_matrix, position_matrix, sentence_matrix = get_embedding_matrix(model)

    vocab_size = embedding_matrix.size()[0]
    hidden_dim = embedding_matrix.size()[1]

    # convert token ids to one hot. We can't differentiate wrt to int token ids hence the need for one hot representation
    token_ids_tensor = torch.tensor(token_ids, dtype = torch.int32)

    #Crete a one hot version of tokenized sentence which is differantiable
    ohe_array = tf.one_hot(token_ids_tensor, vocab_size).numpy()
    token_ids_tensor_one_hot = torch.tensor(ohe_array, requires_grad=True).cuda()

    #multiply one hot vector with name embedding and get positiona and sentence embedding
    #we will explain depending on the token but not the position and sentence.
    inputs_word_embeds = torch.matmul(token_ids_tensor_one_hot, embedding_matrix)
    
    
    # inputs_position_embeds = position_matrix[position_ids]
    # input_sentence_embeds = sentence_matrix[token_type_ids]

    #Get final embedding --> choose word or word + token + sentence
    summed = inputs_word_embeds #+ inputs_position_embeds + input_sentence_embeds

    inputs_embeds = torch.reshape(summed, (1, -1, hidden_dim))

    token_words = tokenizer.convert_ids_to_tokens(token_ids) 
    token_types = list(encoded_tokens["token_type_ids"].numpy()[0])
    attention_mask = encoded_tokens["attention_mask"].cuda()

    return inputs_embeds, attention_mask, token_ids_tensor_one_hot, token_words, token_types


def gradient_pipeline(model, tokenizer, inputs_embeds, attention_mask, label, token_ids_tensor_one_hot, loss_fn = nn.CrossEntropyLoss()):
    """Calculate gradient relative to token

    @param model: (BertForSequenceClassification) model used for classification
    @param tokenizer: (BertTokenizer) tokenizer used
    @param inputs_embeds: (torch.tensor) embedding matrix (in which word in one hot form are differentiable) used by model
    @param encoded_tokens: (Dictionary) dictionary of the different component after tokenization
    @return label: (torch.tensor) real label
    @return token_ids_tensor_one_hot: (torch.tensor) tensor in one hot form to calculate gradient relative to the token and not embedding
    @param loss: (torch.nn.modules.loss) loss used to calculate gradient

    @return gradients: (list) list of different gradients of token
    """
    
    logits = model(**{"inputs_embeds": inputs_embeds, "attention_mask": attention_mask})[0]

    # Compute loss and accumulate the loss values
    loss = loss_fn(logits, label)

    # Calculate gradients relative to every token
    model.zero_grad()
    d_loss_dx = grad(outputs = loss, inputs = token_ids_tensor_one_hot)[0]

    # Calculate Saliency
    gradient_non_normalized = torch.norm(d_loss_dx, dim = 1)

    gradient_tensor = (
        gradient_non_normalized /
        gradient_non_normalized.max()
    )
    gradients = gradient_tensor.cpu().numpy().tolist()

    return gradients


def plot_gradients(tokens, token_types, gradients, title): 
    """ Plot  explanations

    @param token_ids: (list) token index 
    @param encoded_tokens (Dictionary) dictionary with different information about tokens
    @param tokenizer (BertTokenizer) Bert tokenizer
    @param gradients (np.array) vector of different gradients taken from model
    @param title: (str)  title of the plot 
    """

    plt.figure(figsize=(21,3)) 
    xvals = [ x + str(i) for i,x in enumerate(tokens)]
    colors =  [ (0,0,1, c) for c,t in zip(gradients, token_types) ]
    edgecolors = [ "black" if t==0 else (0,0,1, c)  for c,t in zip(gradients, token_types) ]
    plt.tick_params(axis='both', which='minor', labelsize=29)
    p = plt.bar(xvals, gradients, color=colors, linewidth=1, edgecolor=edgecolors)
    plt.title(title) 
    p = plt.xticks(ticks=[i for i in range(len(tokens))], labels=tokens, fontsize=12,rotation=90) 

def clean_tokens(gradients, tokens, token_types):
  """
  Clean the tokens and gradients gradients
  Remove "[CLS]","[CLR]", "[SEP]" tokens
  Reduce (mean) gradients values for tokens that are split ##

  @param gradients: (np.array) vector of gradients
  @param token_ids: (list) token index
  @param encoded_tokens: ()
  """

  token_holder = []
  token_type_holder = []
  gradient_holder = [] 
  i = 0
  while i < len(tokens):
    if (tokens[i] not in ["[CLS]","[CLR]", "[SEP]"]):
      token = tokens[i]
      conn = gradients[i] 
      token_type = token_types[i]
      if i < len(tokens)-1 :
        if tokens[i+1][0:2] == "##":
          token = tokens[i]
          conn = gradients[i]  
          j = 1
          while i < len(tokens)-1 and tokens[i+1][0:2] == "##":
            i +=1 
            token += tokens[i][2:]
            conn += gradients[i]   
            j+=1
          conn = conn /j 
      token_holder.append(token)
      token_type_holder.append(token_type)
      gradient_holder.append(conn)
    i +=1
  return  gradient_holder,token_holder, token_type_holder
        
        
def attention_pipeline(model, tokenizer, sentence):
    """Calculate attention for each token

    @param model: (BertForSequenceClassification) model used for classification
    @param tokenizer: (BertTokenizer) tokenizer used during tokenization
    @param sentence: (str) sentence from which extracts attention

    @return attention: (np.array) matrix of attention of the different layer
    @return token: ()
    """
    inputs = tokenizer.encode_plus(
                text=sentence,  # Preprocess sentence
                add_special_tokens=True,        # Add `[CLS]` and `[SEP]`
                max_length=MAX_LEN,             # Max length to truncate/pad
                truncation = True,              # Truncate sentence to MAX_LEN
                return_tensors='pt',           # Return PyTorch tensor
                return_attention_mask=True      # Return attention mask
                )
    
    attention_mask = inputs['attention_mask'].cuda()
    input_ids = inputs['input_ids'].cuda()

    attention = model(input_ids, attention_mask=attention_mask)[-1]

    input_id_list = input_ids[0].tolist() # Batch index 0
    tokens = tokenizer.convert_ids_to_tokens(input_id_list)

    return attention, tokens



def preprocessing_for_bert(data, tokenizer):
    """Perform required preprocessing steps for pretrained BERT.
    @param    data (np.array): Array of texts to be processed.
    @param    tokenizer(transformers.tokenization_bert): Tokenizer used to tokenize the sentence

    @return   input_ids (torch.Tensor): Tensor of token ids to be fed to a model.
    @return   attention_masks (torch.Tensor): Tensor of indices specifying which
                  tokens should be attended to by the model.
    """
    # Create empty lists to store outputs
    input_ids = []
    attention_masks = []

    # For every sentence...
    for sent in tqdm(data):
        # `encode_plus` will:
        #    (1) Tokenize the sentence
        #    (2) Add the `[CLS]` and `[SEP]` token to the start and end
        #    (3) Truncate/Pad sentence to max length
        #    (4) Map tokens to their IDs
        #    (5) Create attention mask
        #    (6) Return a dictionary of outputs

        #encoding sentence
        encoded_sent = tokenizer.encode_plus(
            text=sent,  # Preprocess sentence
            add_special_tokens=True,        # Add `[CLS]` and `[SEP]`
            max_length=MAX_LEN,             # Max length to truncate/pad
            truncation = True,              # Truncate sentence to MAX_LEN
            pad_to_max_length=True,         # Pad sentence to max length
            #return_tensors='pt',           # Return PyTorch tensor
            return_attention_mask=True      # Return attention mask
            )
        
        # Add the outputs to the lists
        input_ids.append(encoded_sent.get('input_ids'))
        attention_masks.append(encoded_sent.get('attention_mask'))

    # Convert lists to tensors
    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)

    return input_ids, attention_masks



def layer_hidden_extractor(input_ids, attention_mask, model, layer, batch_size, device):
    """Extract Hidden state from selected layer

    @param input_ids: (list) list of input index to pass to model
    @param attention_mask: (list) list of attention mask to pass to model
    @param model: (BertForSequenceClassification) model used for classification
    @param layer: (int) layer to extract
    @param batch_size: (int) dimension of batch size
    @param device: (device) device (cuda) used during inference
    
    @return hidden_stat (torch.tensor) matrix of hidden_stat as (sample, MAX_LEN, hidden_size)
    """

    val_data = TensorDataset(input_ids, attention_mask)
    val_sampler = SequentialSampler(val_data)
    val_dataloader = DataLoader(val_data, sampler = val_sampler, batch_size = batch_size)

    for i, batch in enumerate(val_dataloader):
        # Load batch to GPU
        b_input_ids, b_attn_mask, = tuple(t.to(device) for t in batch)

        # Calculate hidden_state
        with torch.no_grad():
            forward_ = model(b_input_ids, b_attn_mask)

        #Create torch tensor of hidden_state by concatenating batch 
        if i == 0:

          hidden_state = list(forward_[1])[layer]
        else:
          
          new_hidden = list(forward_[1])[layer]
          hidden_state = torch.cat((hidden_state, new_hidden))

    return hidden_state

def outlier_filter(tsne_embedding, sentiment):
    """Filter Outlier from embedding for clear plot

    @param tsne_embedding: (np.array) matrix of tsne embedding
    @param sentiment: (np.array) vector of label

    @return tsne_embedding_filtered: (np.array) matrix of tsne embedding filtered from outlier
    @return sentiment_filtered: (np.array) vector of label filtered from outlier

    """
    col_0_max, col_0_min = np.quantile(tsne_embedding[:, 0], .95), np.quantile(tsne_embedding[:, 0], .05)
    col_1_max, col_1_min = np.quantile(tsne_embedding[:, 1], .95), np.quantile(tsne_embedding[:, 1], .05)

    mask_0 = (tsne_embedding[:, 0] >= col_0_min) & (tsne_embedding[:, 0] <= col_0_max)
    mask_1 = (tsne_embedding[:, 1] >= col_1_min) & (tsne_embedding[:, 1] <= col_1_max)
    mask = mask_0 & mask_1

    tsne_embedding_filtered, sentiment_filtered = tsne_embedding[mask], sentiment[mask]
    return tsne_embedding_filtered, sentiment_filtered

def print_hidden(tsne_list):
    """Print Hiddent state

    @param tsne_list (list of (coordinate tsne, label)) list of tsne-coordinate (of hidden state) and label for each layer 
    """

    num_layer = len(tsne_list)

    num_rows = int(np.ceil(num_layer/4))
    fig, axes = plt.subplots(nrows = num_rows, ncols = 4, figsize = (20, 20))
    iteration = 0

    for row in axes:
        for col in row:

            if iteration >= num_layer:
                fig.suptitle('Visualization of hidden state') 

                plt.show()
                return 

            coord, hue = tsne_list[iteration]
            element = sns.scatterplot(x = coord[:, 0], y = coord[:, 1], hue = hue, ax = col)
            
            element.set_title("Hidden State of Layer {} of BERT".format(iteration), fontsize=10, fontweight='bold')

            iteration += 1
