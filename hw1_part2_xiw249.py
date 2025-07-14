import os
import subprocess
import csv
import re
import random
import numpy as np
import scipy


def read_in_snli():
    """Reads in the SNLI dataset and processes it into a list of tuples.

    Each tuple consists of
    tuple[0]: The sentence ID or an index
    tuple[1]: A list of tokenized words from the sentence.

    Returns:
      sentences: A list of tuples in the above format.
      vocab: A set of all unique tokens in the vocabulary.
    """

    sentences = []

    with open("snli.csv") as f:
        csv_reader = csv.DictReader(f)
        for row in csv_reader:
            sentence_id = row['sentenceID']
            sentence_text = row['sentence']
            sentence_tokens = re.sub(r"[^a-zA-Z0-9\s]", " ", sentence_text).split()
            sentence_tokens = [token.lower() for token in sentence_tokens]

            sentences.append((sentence_id, sentence_tokens))


    with open("identity_labels.txt") as f:
        vocab = [line.strip() for line in f]

    return sentences, vocab


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


def create_term_document_matrix(sentences, vocab):
    """Returns a numpy array containing the term document matrix for the SNLI sentences.

    Inputs:
      sentences: A list of tuples, where each tuple contains a sentence ID and
                 a list of tokenized words from that sentence.
      vocab: A list of the tokens in the vocabulary (identity labels in this case).

    Returns:
      td_matrix: A mxn numpy array where m = len(vocab) and n = len(sentences).
                 A_ij contains the frequency with which word i occurs in sentence j.
    """
    # Mapping from word to index for faster access
    word_to_index = {word: idx for idx, word in enumerate(vocab)}
    sentence_ids = [sentence[0] for sentence in sentences]  # Extracting sentence IDs
    doc_to_index = {doc_id: idx for idx, doc_id in enumerate(sentence_ids)}

    m, n = len(vocab), len(sentences)
    td_matrix = np.zeros((m, n), dtype=int)

    for idx, (sentence_id, tokens) in enumerate(sentences):
        for token in tokens:
            if token in word_to_index:  # Only consider tokens that are in the identity labels
                word_idx = word_to_index[token]
                td_matrix[word_idx, idx] += 1  # Increment the frequency count

    return td_matrix


def create_term_context_matrix(sentences, vocab, context_window_size=1):
    """Returns a numpy array containing the term context matrix for the SNLI sentences.

    Inputs:
      sentences: A list of tuples, where each tuple contains a sentence ID and
                 a list of tokenized words from that sentence.
      vocab: A list of the tokens in the vocabulary.
      context_window_size: The size of the context window to consider to the left and right of each word.

    Returns:
      tc_matrix: A nxn numpy array where A_ij contains the frequency with which
          word j was found within context_window_size to the left or right of
          word i in any sentence.
    """

    # Create vocabulary index: word -> index
    vocab_index = {word: idx for idx, word in enumerate(vocab)}

    # Initialize the term-context matrix with zeros
    n = len(vocab)
    tc_matrix = np.zeros((n, n), dtype=int)

    # Process each sentence in the tuples
    for _, tokens in sentences:
        sentence_length = len(tokens)
        for target_idx, target_word in enumerate(tokens):
            if target_word in vocab_index:
                target_word_index = vocab_index[target_word]

                # Calculate the start and end index for the context window
                start = max(0, target_idx - context_window_size)
                end = min(sentence_length, target_idx + context_window_size + 1)

                # Update matrix for each word in the window
                for context_idx in range(start, end):
                    if context_idx != target_idx:  # Exclude the target word itself
                        context_word = tokens[context_idx]
                        if context_word in vocab_index:
                            context_word_index = vocab_index[context_word]
                            tc_matrix[target_word_index, context_word_index] += 1

    return tc_matrix







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
    sentences, vocab = read_in_snli()

    print("Computing term document matrix...")
    td_matrix = create_term_document_matrix(sentences, vocab)
    #test
    #print(td_matrix)




    print("Computing term context matrix...")
    tc_matrix = create_term_context_matrix(sentences, vocab, context_window_size=4)
    # test
    #print(tc_matrix)

    print("Computing PPMI matrix...")
    ppmi_matrix = create_ppmi_matrix(tc_matrix)

    # random_idx = random.randint(0, len(document_names) - 1)

    word = "american"
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
        '\nThe 10 most similar words to "%s" using cosine-similarity on PPMI matrix are:'
        % (word)
    )
    ranks, scores = rank_words(vocab_to_index[word], ppmi_matrix)
    for idx in range(0,10):
        word_id = ranks[idx]
        print("%d: %s; %s" %(idx+1, vocab[word_id], scores[idx]))


