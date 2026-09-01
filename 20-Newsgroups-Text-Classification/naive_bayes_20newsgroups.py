import csv
from collections import defaultdict, Counter
import math
import numpy as np

# File paths for all required data
TRAIN_LABEL_PATH = '20newsgroups/20newsgroups/train_label.csv'
TRAIN_DATA_PATH = '20newsgroups/20newsgroups/train_data.csv'
TEST_LABEL_PATH = '20newsgroups/20newsgroups/test_label.csv'
TEST_DATA_PATH = '20newsgroups/20newsgroups/test_data.csv'
VOCAB_PATH = '20newsgroups/20newsgroups/vocabulary.txt'
MAP_PATH = '20newsgroups/20newsgroups/map.csv'

# Load vocabulary (word index starts from 1)
def load_vocabulary(path):
    """
    Loads the vocabulary from a text file.
    Each line contains a word; line number is the word index (starting from 1).
    Returns a dict: word_idx -> word
    """
    vocab = {}
    with open(path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f, 1):
            vocab[idx] = line.strip()
    return vocab

# Load label map
def load_label_map(path):
    """
    Loads the mapping from label id to label name from a CSV file.
    Returns a dict: label_id -> label_name
    """
    label_map = {}
    with open(path, 'r', encoding='utf-8') as f:
        for row in csv.reader(f):
            label_map[int(row[0])] = row[1]
    return label_map

# Load labels (one label per line)
def load_labels(path):
    """
    Loads document labels from a file.
    Each line contains a single integer label.
    Returns a list of labels (1-based, index = doc_idx-1)
    """
    labels = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            labels.append(int(line.strip()))
    return labels

# Load data (docIdx, wordIdx, count)
def load_data(path):
    """
    Loads document-word count data from a CSV file.
    Each line: docIdx, wordIdx, count
    Returns a dict: docIdx -> list of (wordIdx, count)
    """
    data = defaultdict(list)
    with open(path, 'r', encoding='utf-8') as f:
        for row in csv.reader(f):
            doc_idx, word_idx, count = map(int, row)
            data[doc_idx].append((word_idx, count))
    return data

def data_to_matrix(data, num_docs, vocab_size):
    """
    Converts sparse document-word data to a dense numpy matrix.
    Rows: documents, Columns: words (shape: num_docs x vocab_size)
    """
    mat = np.zeros((num_docs, vocab_size), dtype=np.float32)
    for doc_idx, word_counts in data.items():
        for word_idx, count in word_counts:
            mat[doc_idx-1, word_idx-1] = count
    return mat

class NaiveBayesClassifier:
    """
    Multinomial Naive Bayes classifier for text classification.
    Implements training, prediction, and evaluation using numpy for efficiency.
    """
    def __init__(self, num_classes, vocab_size):
        # Number of classes and vocabulary size
        self.num_classes = num_classes
        self.vocab_size = vocab_size
        # Model parameters
        self.class_priors = None  # P(class)
        self.word_likelihoods = None  # P(word|class)
        self.class_word_counts = None  # total word count per class
        self.class_total_docs = None  # doc count per class
        # Precomputed log-probabilities for fast prediction
        self.log_class_priors = None
        self.log_word_likelihoods = None

    def train(self, labels, data):
        """
        Trains the Naive Bayes model using training labels and document-word data.
        Calculates class priors and word likelihoods with Laplace smoothing.
        Precomputes log-probabilities for efficient batch prediction.
        """
        # Count documents per class
        class_doc_counts = Counter(labels)
        total_docs = len(labels)
        self.class_priors = {c: class_doc_counts[c] / total_docs for c in class_doc_counts}
        # Count word occurrences per class
        word_counts = {c: Counter() for c in class_doc_counts}
        class_word_totals = {c: 0 for c in class_doc_counts}
        for doc_idx, doc_label in enumerate(labels, 1):
            for word_idx, count in data[doc_idx]:
                word_counts[doc_label][word_idx] += count
                class_word_totals[doc_label] += count
        # Calculate likelihoods with Laplace smoothing
        self.word_likelihoods = {}
        for c in class_doc_counts:
            self.word_likelihoods[c] = {}
            for w in range(1, self.vocab_size + 1):
                # Laplace smoothing: add 1 to numerator, V to denominator
                num = word_counts[c][w] + 1
                denom = class_word_totals[c] + self.vocab_size
                self.word_likelihoods[c][w] = num / denom
        self.class_word_counts = class_word_totals
        self.class_total_docs = class_doc_counts
        # Precompute log-probabilities for numpy batch prediction
        self.log_class_priors = np.log(np.array([self.class_priors[c] for c in range(1, self.num_classes+1)]))
        self.log_word_likelihoods = np.zeros((self.num_classes, self.vocab_size), dtype=np.float32)
        for c in range(1, self.num_classes+1):
            for w in range(1, self.vocab_size+1):
                self.log_word_likelihoods[c-1, w-1] = math.log(self.word_likelihoods[c][w])

    def print_class_priors(self):
        """
        Prints the prior probability for each class.
        """
        print('Class priors:')
        for c in sorted(self.class_priors):
            print(f'P(Omega = {c}) = {self.class_priors[c]:.4f}')

    def predict_matrix(self, X):
        """
        Predicts class labels for all documents in X (dense matrix).
        Uses log-probabilities for numerical stability.
        Returns: numpy array of predicted class labels (1-based)
        """
        # Compute log-probabilities for each class for all documents
        # log_prob = log_prior + sum_i x_i * log P(w_i|c)
        log_probs = X @ self.log_word_likelihoods.T  # (num_docs, num_classes)
        log_probs += self.log_class_priors  # broadcast priors
        preds = np.argmax(log_probs, axis=1) + 1  # class labels are 1-based
        return preds

    def evaluate_matrix(self, labels, X):
        """
        Evaluates predictions: overall accuracy, class accuracy, and confusion matrix.
        labels: list of true labels (1-based)
        X: document-word matrix
        Returns: overall accuracy, class accuracy dict, confusion matrix (numpy array)
        """
        preds = self.predict_matrix(X)
        labels = np.array(labels)
        total = len(labels)
        correct = np.sum(preds == labels)
        overall_acc = correct / total
        class_acc = {}
        confusion = np.zeros((self.num_classes, self.num_classes), dtype=int)
        # Calculate class accuracy
        for c in range(1, self.num_classes+1):
            idxs = (labels == c)
            if np.sum(idxs) > 0:
                class_acc[c] = np.sum(preds[idxs] == c) / np.sum(idxs)
            else:
                class_acc[c] = 0.0
        # Build confusion matrix
        for t, p in zip(labels, preds):
            confusion[t-1, p-1] += 1
        return overall_acc, class_acc, confusion

    def print_confusion_matrix(self, confusion, label_map):
        """
        Prints the confusion matrix with class names.
        """
        classes = sorted(label_map.keys())
        print("Confusion Matrix:")
        header = "     " + " ".join(f"{label_map[c][:6]:>6}" for c in classes)
        print(header)
        for i, c in enumerate(classes):
            row = f"{label_map[c][:6]:>6} "
            for j in range(len(classes)):
                row += f"{confusion[i, j]:6d} "
            print(row)

if __name__ == '__main__':
    # Load all data files
    vocab = load_vocabulary(VOCAB_PATH)
    label_map = load_label_map(MAP_PATH)
    train_labels = load_labels(TRAIN_LABEL_PATH)
    test_labels = load_labels(TEST_LABEL_PATH)
    train_data = load_data(TRAIN_DATA_PATH)
    test_data = load_data(TEST_DATA_PATH)

    print(f'Loaded {len(vocab)} words in vocabulary.')
    print(f'Loaded {len(label_map)} label mappings.')
    print(f'Loaded {len(train_labels)} training labels, {len(test_labels)} test labels.')
    print(f'Loaded {len(train_data)} training documents, {len(test_data)} test documents.')

    # Initialize and train the Naive Bayes classifier
    nb = NaiveBayesClassifier(num_classes=len(label_map), vocab_size=len(vocab))
    nb.train(train_labels, train_data)
    nb.print_class_priors()

    # Convert training data to dense matrix for fast evaluation
    print("\nConverting training data to matrix...")
    X_train = data_to_matrix(train_data, len(train_labels), len(vocab))
    print("Evaluating on training data...")
    train_acc, train_class_acc, train_conf = nb.evaluate_matrix(train_labels, X_train)
    print(f"Overall accuracy (train): {train_acc:.4f}")
    print("Class accuracy (train):")
    for c in sorted(train_class_acc):
        print(f"  {label_map[c]:<25}: {train_class_acc[c]:.4f}")
    nb.print_confusion_matrix(train_conf, label_map)

    # Convert test data to dense matrix for fast evaluation
    print("\nConverting test data to matrix...")
    X_test = data_to_matrix(test_data, len(test_labels), len(vocab))
    print("Evaluating on test data...")
    test_acc, test_class_acc, test_conf = nb.evaluate_matrix(test_labels, X_test)
    print(f"Overall accuracy (test): {test_acc:.4f}")
    print("Class accuracy (test):")
    for c in sorted(test_class_acc):
        print(f"  {label_map[c]:<25}: {test_class_acc[c]:.4f}")
    nb.print_confusion_matrix(test_conf, label_map)
