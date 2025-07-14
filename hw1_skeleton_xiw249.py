import os
import subprocess
import csv
import re
import random
import numpy as np
import scipy


def read_in_shakespeare():
    """Reads in the Shakespeare dataset and processes it into a list of tuples.
       Also reads in the vocab and play name lists from files.

    Each tuple consists of
    tuple[0]: The name of the play
    tuple[1] A line from the play as a list of tokenized words.

    Returns:
      tuples: A list of tuples in the above format.
      document_names: A list of the plays present in the corpus.
      vocab: A list of all tokens in the vocabulary.
    """

    tuples = []

    with open("shakespeare_plays.csv") as f:
        csv_reader = csv.reader(f, delimiter=";")
        for row in csv_reader:
            play_name = row[1]
            line = row[5]
            line_tokens = re.sub(r"[^a-zA-Z0-9\s]", " ", line).split()
            line_tokens = [token.lower() for token in line_tokens]

            tuples.append((play_name, line_tokens))

    with open("vocab.txt") as f:
        vocab = [line.strip() for line in f]

    with open("play_names.txt") as f:
        document_names = [line.strip() for line in f]

    return tuples, document_names, vocab


def get_row_vector(matrix, row_id):
    """A convenience function to get a particular row vector from a numpy matrix

    Inputs:
      matrix: a 2-dimensional numpy array
      row_id: an integer row_index for the desired row vector

    Returns:
      1-dimensional numpy array of the row vector
    """
    return matrix[row_id, :]


def get_column_vector(matrix, col_id):
    """A convenience function to get a particular column vector from a numpy matrix

    Inputs:
      matrix: a 2-dimensional numpy array
      col_id: an integer col_index for the desired row vector

    Returns:
      1-dimensional numpy array of the column vector
    """
    return matrix[:, col_id]


def create_term_document_matrix(line_tuples, document_names, vocab):
    """Returns a numpy array containing the term document matrix for the input lines.

    Inputs:
      line_tuples: A list of tuples, containing the name of the document and
      a tokenized line from that document.
      document_names: A list of the document names
      vocab: A list of the tokens in the vocabulary

    Let m = len(vocab) and n = len(document_names).

    Returns:
      td_matrix: A mxn numpy array where the number of rows is the number of words
          and each column corresponds to a document. A_ij contains the
          frequency with which word i occurs in document j.
    """
    word_to_index = {word: idx for idx, word in enumerate(vocab)}
    doc_to_index = {doc: idx for idx, doc in enumerate(document_names)}

    # Initialize the term-document matrix with zeros
    m, n = len(vocab), len(document_names)
    td_matrix = np.zeros((m, n))

    # Fill the term-document matrix
    for doc_name, tokens in line_tuples:
        doc_idx = doc_to_index[doc_name]
        for token in tokens:
            # Only process words that are in the vocabulary
            if token in word_to_index:
                word_idx = word_to_index[token]
                td_matrix[word_idx][doc_idx] += 1
    return td_matrix


def create_term_context_matrix(line_tuples, vocab, context_window_size=1):
    """Returns a numpy array containing the term context matrix for the input lines.

    Inputs:
      line_tuples: A list of tuples, containing the name of the document and
      a tokenized line from that document.
      vocab: A list of the tokens in the vocabulary

    # NOTE: THIS DOCSTRING WAS UPDATED ON JAN 24, 12:39 PM.

    Let n = len(vocab).

    Returns:
      tc_matrix: A nxn numpy array where A_ij contains the frequency with which
          word j was found within context_window_size to the left or right of
          word i in any sentence in the tuples.
    """

    # Create vocabulary index: word -> index
    vocab_index = {word: idx for idx, word in enumerate(vocab)}

    # Initialize the term-context matrix with zeros
    tc_matrix = np.zeros((len(vocab), len(vocab)), dtype=int)

    # Process each line in the tuples
    for _, line in line_tuples:
        line_length = len(line)
        for target_idx, target_word in enumerate(line):
            # Check if target word is in vocab
            if target_word in vocab_index:
                target_word_index = vocab_index[target_word]

                # Calculate the start and end index for the context window
                start = max(0, target_idx - context_window_size)
                end = min(line_length, target_idx + context_window_size + 1)

                # Update matrix for each word in the window
                for context_idx in range(start, end):
                    if context_idx != target_idx:  # Exclude the target word itself
                        context_word = line[context_idx]
                        if context_word in vocab_index:
                            context_word_index = vocab_index[context_word]
                            tc_matrix[target_word_index, context_word_index] += 1

    return tc_matrix




def create_tf_idf_matrix(term_document_matrix):
    """Given the term document matrix, output a tf-idf weighted version.

    See section 6.5 in the textbook.

    Hint: Use numpy matrix and vector operations to speed up implementation.

    Input:
      term_document_matrix: Numpy array where each column represents a document
      and each row, the frequency of a word in that document.

    Returns:
      A numpy array with the same dimension as term_document_matrix, where
      A_ij is weighted by the inverse document frequency of document h.
    """

    # Calculate IDF
    N = term_document_matrix.shape[1]
    DF = np.count_nonzero(term_document_matrix, axis=1)
    IDF = np.log((1 + N) / (1 + DF)) + 1

    # Calculate TF-IDF
    TF = term_document_matrix
    TF_IDF = TF * IDF[:, None]  # Broadcasting the IDF vector to multiply it with every column of TF

    return TF_IDF


def create_ppmi_matrix(term_context_matrix):
    """Given the term context matrix, output a PPMI weighted version.

    See section 6.6 in the textbook.

    Hint: Use numpy matrix and vector operations to speed up implementation.

    Input:
      term_context_matrix: Numpy array where each column represents a context word
      and each row, the frequency of a word that occurs with that context word.

    Returns:
      A numpy array with the same dimension as term_context_matrix, where
      A_ij is weighted by PPMI.
    """

    # Calculate probabilities
    total_occurrences = np.sum(term_context_matrix)
    word_probabilities = np.sum(term_context_matrix, axis=1, keepdims=True) / total_occurrences
    joint_probabilities = term_context_matrix / total_occurrences

    # Calculate the denominator for PMI, adding a small number to avoid division by zero
    denom = word_probabilities.dot(word_probabilities.T) + 1e-10

    # Calculate PMI, avoiding log(0) by adding a small number to joint probabilities
    pmi = np.log((joint_probabilities + 1e-10) / denom)

    # Compute PPMI by taking the positive part of PMI
    ppmi = np.maximum(pmi, 0)

    # Ensure no NaN or Inf values remain
    ppmi[np.isnan(ppmi)] = 0
    ppmi[np.isinf(ppmi)] = 0

    return ppmi


def compute_cosine_similarity(vector1, vector2):
    """Computes the cosine similarity of the two input vectors.

    Inputs:
      vector1: A nx1 numpy array
      vector2: A nx1 numpy array

    Returns:
      A scalar similarity value.
    """
    # Check for 0 vectors
    if not np.any(vector1) or not np.any(vector2):
        sim = 0

    else:
        sim = 1 - scipy.spatial.distance.cosine(vector1, vector2)

    return sim


def rank_words(target_word_index, matrix):
    """Ranks the similarity of all of the words to the target word using compute_cosine_similarity.

    Inputs:
      target_word_index: The index of the word we want to compare all others against.
      matrix: Numpy matrix where the ith row represents a vector embedding of the ith word.

    Returns:
      A length-n list of integer word indices, ordered by decreasing similarity to the
      target word indexed by word_index
      A length-n list of similarity scores, ordered by decreasing similarity to the
      target word indexed by word_index
    """
    target_vector = matrix[target_word_index]
    similarities = []

    for index, vector in enumerate(matrix):
        if index != target_word_index:  # Avoid comparing the word with itself
            similarity = compute_cosine_similarity(target_vector, vector)
            similarities.append((index, similarity))

    # Sort the words by their similarity score in descending order
    similarities.sort(key=lambda x: x[1], reverse=True)

    # Extract the top 10 most similar words and their scores
    top_indices = [index for index, _ in similarities[:10]]
    top_scores = [score for _, score in similarities[:10]]

    return top_indices, top_scores


if __name__ == "__main__":
    tuples, document_names, vocab = read_in_shakespeare()

    print("Computing term document matrix...")
    td_matrix = create_term_document_matrix(tuples, document_names, vocab)
    #test
    #print(td_matrix)

    print("Computing tf-idf matrix...")
    tf_idf_matrix = create_tf_idf_matrix(td_matrix)


    print("Computing term context matrix...")
    tc_matrix = create_term_context_matrix(tuples, vocab, context_window_size=4)
    #print(tc_matrix)

    print("Computing PPMI matrix...")
    ppmi_matrix = create_ppmi_matrix(tc_matrix)

    # random_idx = random.randint(0, len(document_names) - 1)

    word = "attain"
    vocab_to_index = dict(zip(vocab, range(0, len(vocab))))

    print(
        '\nThe 10 most similar words to "%s" using cosine-similarity on term-document frequency matrix are:'
        % (word)
    )
    ranks, scores = rank_words(vocab_to_index[word], td_matrix)
    for idx in range(0,10):
        word_id = ranks[idx]
        print("%d: %s; %s" %(idx+1, vocab[word_id], scores[idx]))

    print(
        '\nThe 10 most similar words to "%s" using cosine-similarity on term-context frequency matrix are:'
        % (word)
    )
    ranks, scores = rank_words(vocab_to_index[word], tc_matrix)
    for idx in range(0,10):
        word_id = ranks[idx]
        print("%d: %s; %s" %(idx+1, vocab[word_id], scores[idx]))


    print(
        '\nThe 10 most similar words to "%s" using cosine-similarity on tf-idf matrix are:'
        % (word)
    )
    ranks, scores = rank_words(vocab_to_index[word], tf_idf_matrix)
    for idx in range(0,10):
        word_id = ranks[idx]
        print("%d: %s; %s" %(idx+1, vocab[word_id], scores[idx]))

    print(
        '\nThe 10 most similar words to "%s" using cosine-similarity on PPMI matrix are:'
        % (word)
    )
    ranks, scores = rank_words(vocab_to_index[word], ppmi_matrix)
    for idx in range(0,10):
        word_id = ranks[idx]
        print("%d: %s; %s" %(idx+1, vocab[word_id], scores[idx]))


