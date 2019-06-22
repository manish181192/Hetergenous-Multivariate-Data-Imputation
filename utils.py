import numpy as np

def softmax(x, axis=None):
    x = x - x.max(axis=axis, keepdims=True)
    y = np.exp(x)
    return y / y.sum(axis=axis, keepdims=True)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def get_generated_sentence(nlp_logits):
    """

    :param nlp_logits: list of size T with each element
    [B x vocab_size]
    :return:
    """
    samples = []
    for time_step, logits in enumerate(nlp_logits):
        vocab_size = logits.shape[1]
        probs = softmax(logits, axis=-1)
        samples_t = np.array([np.random.choice(np.arange(vocab_size), p=row) for row in probs])
        samples.append(samples_t)

    np_samples = np.array(samples).T
    return np_samples