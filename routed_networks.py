import os, sys, inspect
PATH_CUR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(PATH_CUR)
sys.path.append('../')
basedir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(0, basedir)
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.optim as optim
import time
from data_utils import load_task, get_tokenized_text
import argparse
import pandas as pd

verbose = False
random_seed = 2
embedding_dim = 100
n_memories = 20
gradient_clip_value = 40
batch_size = 32
tie_keys = True
learn_keys = True
max_stuck_epochs = 6
teach = False
min_improvement = 0.0
n_tries = 10
try_n = 0
optimizer_name = 'sgd'
entnet_threshold = [0] * 20
# entnet_threshold = [0, 0.1, 4.1, 0, 0.3, 0.2, 0, 0.5, 0.1, 0.6, 0.3, 0, 1.3, 0, 0, 0.2, 0.5, 0.3, 2.3, 0]
data_dir = "./../babi_data/tasks_1-20_v1-2/en-valid-10k"
STATE_PATH = './trained_models/task_{}_try_{}.pth'
OPTIM_PATH = './trained_models/task_{}_try_{}.pth'
cuda = True
train_str = ""
test_str = ""


def print_start_train_message(name):
    key_state_text = "tied to vocab" if tie_keys else "NOT tied to vocab"
    key_learned_text = "learned" if learn_keys else "NOT learned"
    cuda_text = "gpu" if cuda else "cpu"
    verbose_text = "verbose" if verbose else "non-verbose"
    teaching_text = "enabled" if teach else "disabled"
    print("start learning task {}\n".format(name) +
          "learning on {}\n".format(cuda_text) +
          "random seed is {}\n".format(random_seed) +
          "embedding dimension is {}\n".format(embedding_dim) +
          "number of memories is {}\n".format(n_memories) +
          "gradient clip value is {}\n".format(gradient_clip_value) +
          "maximum stuck epochs is {}\n".format(max_stuck_epochs) +
          "teaching is {}\n".format(teaching_text) +
          "minimal improvement is {}\n".format(min_improvement) +
          "batch size is {}\n".format(batch_size) +
          "keys are {}\n".format(key_state_text) +
          "keys are {}\n".format(key_learned_text) +
          "{} mode\n".format(verbose_text))


def print_start_test_message(name):
    print("testing task {}\n".format(name))
    if verbose:
        print("random seed is {}\n".format(random_seed) +
              "embedding dimension is {}\n".format(embedding_dim) +
              "number of memories is {}\n".format(n_memories))


def get_vocab(tasks):
    vocab = set()

    text = list()
    for task in tasks:
        text += get_tokenized_text(data_dir, task)

    for word in text:
        if word.isalnum():
            vocab.add(word)

    vocab = list(vocab)
    vocab.sort()

    return vocab, len(vocab) + 1


def get_len_max_story(data):
    len_max_story = 0
    for story, query, answer in data:
        if len(story) > len_max_story:
            len_max_story = len(story)
    return len_max_story
    # return np.max([len(tuple[0]) for tuple in batch])


def get_len_max_sentence(data):
    len_max_sentence = 0
    for story, query, answer in data:
        for sentence in story:
            if len(sentence) > len_max_sentence:
                len_max_sentence = len(sentence)
        if len(query) > len_max_sentence:
            len_max_sentence = len(query)
    return len_max_sentence


def annotate(sentence):
    raise NotImplementedError


def vectorize_router_data(data, len_max_sentence):
    sentences = torch.zeros((len(data), len_max_sentence), dtype=torch.long, requires_grad=False)
    sentence_annotations = torch.zeros((len(data)), dtype=torch.long, requires_grad=False)

    for i, sentence in enumerate(data):
        word_padding_size = max(0, len_max_sentence - len(sentence))
        sentences[i] = torch.tensor(sentence + [0] * word_padding_size)
        sentence_annotations[i] = annotate(sentence)

    return sentences, sentence_annotations


def vectorize_data(data, len_max_sentence, len_max_story):
    len_masked_sentence = len_max_sentence * 3

    vec_stories = torch.zeros((len(data), len_max_story, len_masked_sentence), dtype=torch.long, requires_grad=False)
    vec_queries = torch.zeros((len(data), len_max_sentence), dtype=torch.long, requires_grad=False)
    vec_answers = torch.zeros((len(data)), requires_grad=False, dtype=torch.long)

    i = 0
    for story, query, answer in data:
        vec_curr_story = torch.zeros((len_max_story, len_masked_sentence), requires_grad=False)
        for j, sentence in enumerate(story):
            word_padding_size = max(0, len_masked_sentence - len(sentence))
            vec_curr_story[j] = torch.tensor(sentence + [0] * word_padding_size)

        sentence_padding_size = max(0, len_max_story - len(story))
        for j in range(1, sentence_padding_size + 1):
            vec_curr_story[-j] = torch.tensor([0] * len_masked_sentence)

        vec_stories[i] = vec_curr_story

        word_padding_size = max(0, len_max_sentence - len(query))
        vec_curr_query = torch.tensor(query + [0] * word_padding_size)
        vec_queries[i] = vec_curr_query

        vec_answers[i] = torch.tensor(answer)

        i += 1

    return vec_stories, vec_queries, vec_answers


def mask_sentence(sentence, token_to_idx, len_max_sentence):
    moves = ["journeyed", "moved", "travelled", "went", "went back"]
    grabs = ["grabbed", "got", "took", "picked up"]
    drops = ["discarded", "dropped", "left", "put down"]

    mask = [0] * (3 * len_max_sentence)

    word_padding_size = max(0, len_max_sentence - len(sentence))
    masked_sentence = [token_to_idx[w] for w in sentence] + [0] * word_padding_size

    for move in moves:
        if move in sentence:
            mask[0:len_max_sentence] = masked_sentence
            return mask
    for grab in grabs:
        if grab in sentence:
            mask[len_max_sentence: 2 * len_max_sentence] = masked_sentence
            return mask
    for drop in drops:
        if drop in sentence:
            mask[2 * len_max_sentence:] = masked_sentence
            return mask
    raise ValueError("No valid mask")


def indexize_data(data, token_to_idx, len_max_sentence):
    indexed_data = []

    for story, query, answer in data:
        indexed_story = []
        for sentence in story:
            masked_sentence = mask_sentence(sentence, token_to_idx, len_max_sentence)
            indexed_story.append(masked_sentence)


        word_padding_size = max(0, len_max_sentence - len(query))
        indexed_query = [token_to_idx[w] for w in query] + [0] * word_padding_size

        indexed_answer = token_to_idx[answer[0]]

        indexed_data.append((indexed_story, indexed_query, indexed_answer))

    indexed_data.sort(key=lambda tuple: len(tuple[0]))
    return indexed_data


def get_router_data(data, portion):
    routed_data_length = int(len(data) * portion)
    return np.random.permutation(np.array(data, dtype=object))[:routed_data_length]


def router_batch_generator(data, len_max_sentence, batch_size):
    sentences = [sentence for triple in data for story in triple for sentence in story]
    len_data = len(sentences)
    perm_data = np.random.permutation(np.array(sentences, dtype=object))

    pos = 0
    while pos < len_data:
        if pos < len_data - batch_size:
            batch = perm_data[pos:pos + batch_size]
            vec_batch = vectorize_router_data(batch, len_max_sentence)
            yield vec_batch
            pos = pos + batch_size
        else:
            batch = perm_data[pos:]
            vec_batch = vectorize_router_data(batch, len_max_sentence)
            yield vec_batch
            return


def network_batch_generator(data, len_max_sentence, batch_size, permute='full'):
    len_data = len(data)

    # only permutates the batchs, so is data is sorted by story length, batch will have sort of the same length.
    # this can actually save a lot of compute - nice job aviad!
    if permute == 'half':
        last_batch_pos = int(np.ceil(len_data / batch_size))
        perm = np.random.permutation(last_batch_pos)

        for pos in perm:
            if pos != last_batch_pos - 1:
                batch = data[pos * batch_size:(pos + 1) * batch_size]
                len_max_story = len(batch[-1][0])
                vec_batch = vectorize_data(batch, len_max_sentence, len_max_story)
                yield vec_batch
            else:
                batch = data[pos * batch_size:]
                len_max_story = len(batch[-1][0])
                vec_batch = vectorize_data(batch, len_max_sentence, len_max_story)
                yield vec_batch
        return

    perm_data = data # permute argument if 'no', or anything that's not 'half' or 'full'
    if permute == 'full':
        perm_data = np.random.permutation(np.array(data, dtype=object))

    pos = 0
    while pos < len_data:
        if pos < len_data - batch_size:
            batch = perm_data[pos:pos + batch_size]
            len_max_story = get_len_max_story(batch)
            vec_batch = vectorize_data(batch, len_max_sentence, len_max_story)
            yield vec_batch
            pos = pos + batch_size
        else:
            batch = perm_data[pos:]
            len_max_story = get_len_max_story(batch)
            vec_batch = vectorize_data(batch, len_max_sentence, len_max_story)
            yield vec_batch
            return


def init_embedding_matrix(vocab_size, device):
    embeddings_matrix = nn.Embedding(vocab_size, embedding_dim, 0).to(device)

    init_mean = torch.zeros((vocab_size, embedding_dim), device=device)
    init_standard_deviation = torch.cat((torch.full((1, embedding_dim), 0.0, device=device), torch.full((vocab_size - 1, embedding_dim), 0.1, device=device)))

    embeddings_matrix.weight = nn.Parameter(torch.normal(init_mean, init_standard_deviation), requires_grad=True).to(device)

    return embeddings_matrix


def get_key_tensors(n_routes, device):
    """
    returns a list of key tensors with length n_memories
    list may be randomly initialized (current version) or tied to specific entities
    """
    mean = torch.zeros((n_routes, embedding_dim), device=device)
    standard_deviation = torch.full((n_routes, embedding_dim), 0.1, device=device)
    keys = torch.normal(mean, standard_deviation)

    return nn.Parameter(keys, requires_grad=True).to(device)


def get_initial_state(device):
    """
    returns a list of key tensors with length n_memories
    list may be randomly initialized (current version) or tied to specific entities
    """
    mean = torch.zeros(embedding_dim, device=device)
    standard_deviation = torch.full([embedding_dim], 0.1, device=device)
    state = torch.normal(mean, standard_deviation)

    return nn.Parameter(state, requires_grad=True).to(device)


def get_matrix_weights(out_scale, in_scale, device):
    """
    :return: initial weights for any og the matrices U, V, W
     weights may be randomly initialized (current version) or initialized to zeros or the identity matrix
    """
    init_mean = torch.zeros((embedding_dim * out_scale, embedding_dim * in_scale), device=device)
    init_standard_deviation = torch.full((embedding_dim * out_scale, embedding_dim * in_scale), 0.1, device=device)

    return nn.Parameter(torch.normal(init_mean, init_standard_deviation), requires_grad=True).to(device)


def get_r_matrix_weights(vocab_size, device):
    """
    :return: initial weights for any og the matrices U, V, W
     weights may be randomly initialized (current version) or initialized to zeros or the identity matrix
    """
    init_mean = torch.zeros((vocab_size, embedding_dim), device=device)
    init_standard_deviation = torch.full((vocab_size, embedding_dim), 0.1, device=device)

    return nn.Parameter(torch.normal(init_mean, init_standard_deviation), requires_grad=True).to(device)


def get_non_linearity():
    """
    :return: the non-linearity function to be used in the model.
    this may be a parametric ReLU (current version) or (despite its name) the identity
    """
    # return nn.PReLU(num_parameters=embedding_dim, init=1)
    return nn.PReLU(init=1)


class Router(nn.Module):
    def __init__(self, vocab_size, len_max_sentence, n_routes, device):
        super().__init__()
        self.proxy = nn.Parameter(torch.zeros(1))

    def forward(self, batch):
        pass


class RoutedNetwork(nn.Module):
    #     def __init__(self, vocab_size, keys, len_max_sentence, embeddings_matrix, device):
    def __init__(self, vocab_size, len_max_sentence, n_routes, device):
        super().__init__()
        self.router = Router(vocab_size, len_max_sentence, n_routes, device)

        self.len_max_sentence = len_max_sentence
        self.device = device
        self.n_routes = n_routes

        #embedding
        self.embedding_matrix = init_embedding_matrix(vocab_size, device)

        # Encoder
        self.input_encoder_multiplier = nn.Parameter(torch.ones((len_max_sentence * n_routes, embedding_dim), device=device), requires_grad=True).to(device)
        self.query_encoder_multiplier = nn.Parameter(torch.ones((len_max_sentence, embedding_dim), device=device), requires_grad=True).to(device)
        # self.query_encoder_multiplier = self.input_encoder_multiplier

        # Memory
        self.keys = get_key_tensors(n_routes, device)
        self.initial_state = get_initial_state(device)
        self.state = None

        self.U = nn.Linear(embedding_dim * n_routes, embedding_dim, bias=False).to(device)
        self.V = nn.Linear(embedding_dim * n_routes, embedding_dim * n_routes, bias=False).to(device)
        self.W = nn.Linear(embedding_dim * n_routes, embedding_dim * n_routes, bias=False).to(device)

        self.U.weight = get_matrix_weights(n_routes, 1, device)
        self.V.weight = get_matrix_weights(n_routes, n_routes, device)
        self.W.weight = get_matrix_weights(n_routes, n_routes, device)

        self.in_non_linearity = get_non_linearity().to(device)
        self.query_non_linearity = get_non_linearity().to(device)
        # self.query_non_linearity = self.in_non_linearity

        # Decoder
        self.R = nn.Linear(embedding_dim, vocab_size, bias=False).to(device)
        self.H = nn.Linear(embedding_dim, embedding_dim, bias=False).to(device)
        self.R.weight = get_r_matrix_weights(vocab_size, device)
        self.H.weight = get_matrix_weights(1, 1, device)

    # def create_route_mask(self, route_logits):
    #     pass
    #
    # def sentences_from_stories(self, stories):
    #     pass

    def init_new_state(self, batch_size, device):
        # self.state = self.initial_state.clone().detach().to(device).repeat(batch_size, 1, 1)
        self.state = (self.initial_state * 1).repeat(batch_size, 1, 1)

    def forward(self, batch_stories, batch_queries):

        batch = self.embedding_matrix(batch_stories)

        # re-initialize memories to key-values
        self.init_new_state(len(batch), self.device)

        # Encoder
        a, b, c, d = batch.shape
        batch = (batch * self.input_encoder_multiplier).reshape(a, b, self.n_routes, c // self.n_routes, d)
        batch = batch.sum(dim=3)

        # Memory
        for sentence_idx in range(batch.shape[1]):
            sentence = batch[:, sentence_idx]
            # sentence_memory_repeat = sentence.repeat(1, n_memories).view(len(batch), n_memories, -1)

            # state_gate = (sentence * self.state).sum(dim=2)
            # key_gate = (sentence * self.keys).sum(dim=2)
            state_gate = sentence * self.state
            key_gate = sentence * self.keys
            gate = torch.sigmoid(state_gate + key_gate)

            original = sentence.shape
            update_candidate = self.in_non_linearity(self.U(self.state).reshape(original) + self.V(self.keys.reshape(-1)).reshape(self.keys.shape) + self.W(sentence.reshape(a, -1)).reshape(original))
            # the null sentence mask make sure the padding sentences (that are not part of the original story, but are fake "null" sentences) doesn't effect the memories of th network
            null_sentence_mask = gate.clone().detach()
            null_sentence_mask[null_sentence_mask == 0.5] = 0
            null_sentence_mask[null_sentence_mask != 0] = 1
            # self.memories = self.memories + (update_candidate.permute(2, 0, 1) * gate).permute(1, 2, 0)
            self.state = self.state + (update_candidate * (gate * null_sentence_mask)).sum(dim=1, keepdim=True)
            self.state = self.state / torch.norm(self.state, dim=1, keepdim=True)

        # Decoder
        batch = self.embedding_matrix(batch_queries)
        batch = batch * self.query_encoder_multiplier
        batch = batch.sum(dim=1, keepdim=True)
        results = self.R(self.query_non_linearity(batch + self.H(self.state))).reshape(batch.shape[0], -1)
        return results, None


def train_router(router_vec_train, len_max_sentence, device, routed_network, criterion, router_optimizer):
    router_wrong = 0
    router_epoch_loss = 0.0
    router_correct_epoch = 0
    while True:
        router_epoch = 0
        for i, batch in enumerate(router_batch_generator(router_vec_train, len_max_sentence, batch_size)):
            batch_sentences, batch_annotations = batch

            batch_sentences, batch_annotations = \
                batch_sentences.clone().detach().to(device), batch_annotations.clone().detach().to(device)

            output = routed_network.router(batch_sentences)
            loss = criterion(output, batch_annotations)
            loss.backward()

            router_epoch_loss += loss.item()

            nn.utils.clip_grad_value_(routed_network.router.parameters(), gradient_clip_value)
            router_optimizer.step()
            # zero the parameter gradients
            router_optimizer.zero_grad()

            pred_idx = np.argmax(output.cpu().detach().numpy(), axis=1)
            for j in range(len(output)):
                if pred_idx[j] == batch_annotations[j].item():
                    router_correct_epoch += 1

        if verbose:
            # print statistics
            print('[%d] loss: %.3f' % (router_epoch + 1, router_epoch_loss))
            print('[%d] correct: %d out of %d' % (router_epoch + 1, router_correct_epoch, len(router_vec_train)))

        if router_correct_epoch == len(router_vec_train) or router_epoch == 200:
            break


def train_network(vec_train, vec_test, len_max_sentence, permute_data, epoch, task_id, train_acc_history,
                  train_loss_history, test_acc_history, full_test_loss_history, net_history, optim_history, loss_history,
                  test_loss_history, correct_history, test_correct_history, learning_rate,
                  device, routed_network, network_optimizer, criterion, router_optimizer):
    epoch_loss = 0.0
    running_loss = 0.0
    correct_batch = 0
    correct_epoch = 0
    router_correct_epoch = 0
    router_wrong = 0
    start_time = time.time()
    if verbose:
        print("len vec train is: {}".format(len(vec_train)))
        print("len vec test is: {}".format(len(vec_test)))
    for i, batch in enumerate(network_batch_generator(vec_train, len_max_sentence, batch_size, permute_data)):
        batch_stories, batch_queries, batch_answers = batch

        batch_stories, batch_queries, batch_answers = \
            batch_stories.clone().detach().to(device), batch_queries.clone().detach().to(device), \
            batch_answers.clone().detach().to(device)

        network_output, router_output = routed_network(batch_stories, batch_queries)
        network_loss = criterion(network_output, batch_answers)
        # router_loss = criterion(router_output, story_annotations)
        network_loss.backward()
        # router_loss.backwards()

        running_loss += network_loss.item()
        epoch_loss += network_loss.item()

        nn.utils.clip_grad_value_(routed_network.parameters(), gradient_clip_value)
        network_optimizer.step()
        network_optimizer.zero_grad()
        # router_optimizer.step()
        # router_optimizer.zero_grad()

        pred_idx = np.argmax(network_output.cpu().detach().numpy(), axis=1)
        for j in range(len(network_output)):
            if pred_idx[j] == batch_answers[j].item():
                correct_batch += 1
                correct_epoch += 1
                router_correct_epoch += 1
            else:
                router_wrong += 1

        if verbose:
            if i % 50 == 49:
                # print statistics
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 50))
                running_loss = 0.0

                print('[%d, %5d] correct: %d out of %d' % (epoch + 1, i + 1, correct_batch, 50 * batch_size))
                correct_batch = 0

    # very loose approximation for the average loss over the epoch
    epoch_loss = epoch_loss / (len(vec_train) / batch_size)
    # print epoch time
    end_time = time.time()
    if verbose:
        print("###################################################################################################")
        print(end_time - start_time)
        print('epoch loss: %.3f' % epoch_loss)
        print("###################################################################################################")

    test_loss, test_correct = eval(task_id, device, routed_network, vec_test, len_max_sentence)
    test_fail_rate = 100 - (float(test_correct) / len(vec_test)) * 100

    # this segment for graphs only, no use for logic
    train_acc_history.append(float(correct_epoch) / len(vec_train))
    train_loss_history.append(epoch_loss)
    test_acc_history.append(float(test_correct) / len(vec_test))
    full_test_loss_history.append(test_loss)
    plt.ylim(0, 1.1)
    plt.plot(range(1, len(train_acc_history) + 1), train_acc_history, "C0", range(1, len(test_acc_history) + 1),
             test_acc_history, "C1")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Train and test accuracy over epochs")
    plt.legend(["train", "test"])
    plt.savefig(
        os.path.join(basedir, "results/train_graphs/train{}_test{}_try_{}.jpeg".format(train_str, test_str, try_n)))
    # //

    net_history.append(routed_network.state_dict())
    optim_history.append(router_optimizer.state_dict())
    loss_history.append(epoch_loss)
    test_loss_history.append(test_loss)
    correct_history.append(correct_epoch)
    test_correct_history.append(test_correct)


def train(tasks, vocab_tasks, device, mix=False, task_id=None):
    train, test = list(), list()
    if mix:
        for task in tasks:
            task_train, task_test = load_task(data_dir, task)
            train, test = train + task_train, test + task_test
    else:
        task = tasks[0]
        train, test = load_task(data_dir, task)

    vocab, vocab_size = get_vocab(vocab_tasks)

    data = train + test
    len_max_sentence = get_len_max_sentence(data)

    print_start_train_message(task_id)

    token_to_idx = {token: i + 1 for i, token in enumerate(vocab)}

    vec_train = indexize_data(train, token_to_idx, len_max_sentence)
    vec_test = indexize_data(test, token_to_idx, len_max_sentence)
    # router_vec_train = get_router_data(vec_train)

    # entnet.load_state_dict(torch.load(STATE_PATH.format(task, 0)))
    ########################################################################
    routed_network = RoutedNetwork(vocab_size, len_max_sentence, 3, device)
    routed_network.to(device)
    routed_network = routed_network.float()

    ##### Define Loss and Optimizer #####
    criterion = nn.CrossEntropyLoss().to(device)
    learning_rate = 0.01
    network_optimizer = optim.SGD(routed_network.parameters(), lr=learning_rate)
    router_optimizer = optim.SGD(routed_network.router.parameters(), lr=learning_rate)
    if optimizer_name == 'adam':
        network_optimizer = optim.Adam(routed_network.parameters(), lr=learning_rate)
        # router_optimizer = optim.Adam(routed_network.router.parameters(), lr=learning_rate)
    # optimizer.load_state_dict(torch.load(OPTIM_PATH.format(task, 0)))

    ##### Train Model #####
    epoch = 0
    permute_data = 'full'

    loss_history, test_loss_history = [], []
    correct_history, test_correct_history = [], []
    net_history, optim_history = [], []

    train_acc_history, train_loss_history = [], []
    test_acc_history, full_test_loss_history = [], []

    while True:

        # train_router(router_vec_train, len_max_sentence, device, routed_network, criterion, router_optimizer)

        train_network(vec_train, vec_test, len_max_sentence, permute_data, epoch, task_id, train_acc_history,
                      train_loss_history, test_acc_history, full_test_loss_history, net_history, optim_history,
                      loss_history, test_loss_history, correct_history, test_correct_history, learning_rate,
                      device, routed_network, network_optimizer, criterion, router_optimizer)
        epoch += 1

        if epoch == 200:
            best_idx = np.argmin(test_loss_history)
            best_model = net_history[best_idx]
            best_optim = optim_history[best_idx]
            model_score = loss_history[best_idx]
            model_test_score = test_loss_history[best_idx]
            model_correct_score = correct_history[best_idx]
            model_test_correct_score = test_correct_history[best_idx]
            break

            # adjust learning rate every 25 epochs until 200 epochs
        if epoch < 200 and epoch % 25 == 24:
            learning_rate = learning_rate / 2
            network_optimizer = optim.SGD(routed_network.parameters(), lr=learning_rate)
            router_optimizer = optim.SGD(routed_network.router.parameters(), lr=learning_rate)
            if optimizer_name == 'adam':
                network_optimizer = optim.Adam(routed_network.parameters(), lr=learning_rate)
                router_optimizer = optim.Adam(routed_network.router.parameters(), lr=learning_rate)


    torch.save(best_model, STATE_PATH.format(task_id, try_n))
    torch.save(best_optim, OPTIM_PATH.format(task_id, try_n))

    print("Finished Training task {}\n".format(task_id) +
          "try {} was best\n".format(best_idx) +
          "loss is: {}\n".format(model_score) +
          "correct: {} out of {}\n".format(model_correct_score, len(vec_train)) +
          "test loss is: {}\n".format(model_test_score) +
          "test correct: {} out of {}\n".format(model_test_correct_score, len(vec_test)))


def eval(task_is, device, routed_network, vec_test, len_max_sentence):
    ##### Define Loss and Optimizer #####
    criterion = nn.CrossEntropyLoss().to(device)

    with torch.no_grad():
        running_loss = 0
        correct = 0
        for i, batch in enumerate(network_batch_generator(vec_test, len_max_sentence, batch_size, permute='no')):
            batch_stories, batch_queries, batch_answers = batch

            batch_stories, batch_queries, batch_answers = \
                batch_stories.clone().detach().to(device), batch_queries.clone().detach().to(device), \
                batch_answers.clone().detach().to(device)

            network_output, router_output = routed_network(batch_stories, batch_queries)
            network_loss = criterion(network_output, batch_answers)
            # router_loss = criterion(router_output, story_annotations)

            running_loss += network_loss.item()

            pred_idx = np.argmax(network_output.cpu().detach().numpy(), axis=1)
            for j in range(len(network_output)):
                if pred_idx[j] == batch_answers[j].item():
                    correct += 1

        # very loose approximation for the average loss over the epoch
        if verbose:
            print("Finished Testing task {}\n".format(task_is) +
                  "loss is: {}\n".format(running_loss / (len(vec_test) / batch_size)) +
                  "correct: {} out of {}\n".format(correct, len(vec_test)))

        return running_loss / (len(vec_test) / batch_size), correct


def test(tasks, vocab_tasks, device, mix=False, task_id=None):
    train, test = list(), list()
    if mix:
        for task in tasks:
            task_train, task_test = load_task(data_dir, task, valid=False)
            train, test = train + task_train, test + task_test
    else:
        task = tasks[0]
        train, test = load_task(data_dir, task, valid=False)

    vocab, vocab_size = get_vocab(vocab_tasks)

    print_start_test_message(task_id)

    data = train + test
    len_max_sentence = get_len_max_sentence(data)
    token_to_idx = {token: i + 1 for i, token in enumerate(vocab)}
    vec_test = indexize_data(test, token_to_idx, len_max_sentence)

    routed_network = RoutedNetwork(vocab_size, len_max_sentence, 3, device)
    routed_network.to(device)
    routed_network = routed_network.float()
    routed_network.load_state_dict(torch.load(STATE_PATH.format(task_id, try_n)))

    loss, correct = eval(task_id, device, routed_network, vec_test, len_max_sentence)

    best_results = {"train tasks": [train_str], "test_tasks": [test_str], "try": [try_n], "accuracy": [float(correct) / len(vec_test)], "loss": [loss]}
    df = pd.DataFrame(best_results)
    df.to_csv(os.path.join(basedir, "results/csv_doc/train{}_test{}_try_{}.csv".format(train_str, test_str, try_n)))

    if not verbose:
        print("Finished Testing task {}\n".format(task_id) +
              "loss is: {}\n".format(loss) +
              "correct: {} out of {}\n".format(correct, len(vec_test)))

def main():

    parser = argparse.ArgumentParser(description='entnet')

    parser.add_argument(
        "--verbose",
        help="increases the verbosity of the output",
        action="store_true"
    )
    parser.add_argument(
        '--train_tasks',
        type=int,
        nargs='+',
        default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        help='the tasks to learn'
    )
    parser.add_argument(
        '--test_tasks',
        type=int,
        nargs='+',
        default=[],
        help='the tasks to test on'
    )
    parser.add_argument(
        "--embedding_dim",
        help="the dimension of the emmbeding space for the vocabulary",
        type=int,
        default=100
    )
    parser.add_argument(
        "--batch_size",
        help="the size of the mini-batches in the learning proccess",
        type=int,
        default=32
    )
    parser.add_argument(
        "--gradient_clip_value",
        help="the value at which to clip the gradients",
        type=int,
        default=40
    )
    parser.add_argument(
        "--optimizer",
        help="the name of the optimizer to use",
        type=str,
        default="sgd"
    )
    parser.add_argument(
        "--state_path",
        help="the path to load from or save to the routed network",
        type=str,
        default="./trained_models/task_{}.pth"
    )
    parser.add_argument(
        "--optim_path",
        help="the path to load from or save to the optimizer of the routed network",
        type=str,
        default="./trained_models/optim_{}.pth"
    )
    parser.add_argument(
        "--cpu",
        help="trains the net on CPU",
        action="store_true"
    )
    parser.add_argument(
        "--train",
        help="trains the net",
        action="store_true"
    )
    parser.add_argument(
        "--test",
        help="tests the net",
        action="store_true"
    )
    parser.add_argument(
        "--load_net",
        help="used saved trained nets to test or complete training",
        action="store_true"
    )
    parser.add_argument(
        "--try_n",
        help="index of current try",
        type=int,
        default=1
    )

    args = parser.parse_args()

    global verbose, embedding_dim, batch_size, gradient_clip_value, optimizer, STATE_PATH, OPTIM_PATH, try_n, cuda, train_str, test_str

    verbose = args.verbose
    embedding_dim = args.embedding_dim
    batch_size = args.batch_size
    gradient_clip_value = args.gradient_clip_value
    optimizer = args.optimizer
    train_tasks = args.train_tasks
    test_tasks = args.test_tasks
    STATE_PATH = args.state_path
    OPTIM_PATH = args.optim_path
    try_n = args.try_n

    task_ids = train_tasks
    test_ids = test_tasks
    if not test_ids:
        test_ids = train_tasks
    train_str = "".join([f"_{task}" for task in task_ids])
    test_str = "".join([f"_{task}" for task in test_ids])

    random_seed = "not set"
    # torch.manual_seed(args.random_seed)
    # np.random.seed(args.random_seed)
    # seed(args.random_seed)

    if torch.cuda.is_available() and not args.cpu:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
        cuda = False

    if not test_tasks:
        for task in train_tasks:
            if args.train:
                train([task], [task], device, mix=False, task_id=train_str + test_str + "_try_" + str(try_n))
            if args.test:
                test([task], [task], device, mix=False, task_id=train_str + test_str + "_try_" + str(try_n))
    else:
        vocab_tasks = list(set(train_tasks).union(set(test_tasks)))
        if args.train:
            train(train_tasks, vocab_tasks, device, mix=True, task_id=train_str + test_str + "_try_" + str(try_n))
        if args.test:
            test(test_tasks, vocab_tasks, device, mix=True, task_id=train_str + test_str + "_try_" + str(try_n))


if __name__ == "__main__":
    main()