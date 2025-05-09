import numpy as np

def viterbi(words, tags, transition_prob, emission_prob, initial_prob):

    num_words = len(words)
    num_tags = len(tags)

    # Initialize the Viterbi table and backpointer table
    viterbi_table = np.zeros((num_tags, num_words))
    backpointer = np.zeros((num_tags, num_words), dtype=int) #Fixed indentation here

    # Step 1: Initialization
    for i, tag in enumerate(tags):
        viterbi_table[i, 0] = initial_prob.get(tag, 0) * emission_prob.get((words[0], tag), 0)
        backpointer[i, 0] = -1 # No previous tag for the first word

    # Step 2: Recursion
    for t in range(1, num_words): # For each word in the sentence
        for s, tag in enumerate(tags): # For each possible tag
            max_prob = -1
            best_tag = -1
            for s_prev, prev_tag in enumerate(tags): # For each previous tag
                prob = viterbi_table[s_prev, t - 1] * transition_prob.get((prev_tag, tag), 0) * emission_prob.get((words[t], tag), 0)
                if prob > max_prob:
                    max_prob = prob
                    best_tag = s_prev
            viterbi_table[s, t] = max_prob
            backpointer[s, t] = best_tag

    # Step 3: Termination
    best_last_tag = np.argmax(viterbi_table[:, -1])
    best_path = [best_last_tag]

    # Step 4: Backtracking
    for t in range(num_words - 1, 0, -1):
        best_last_tag = backpointer[best_last_tag, t]
        best_path.insert(0, best_last_tag)

    # Convert tag indices to tag names
    best_path_tags = [tags[idx] for idx in best_path]
    return best_path_tags


# Example usage
if __name__ == "__main__":
    # Define the sentence and possible tags
    words = ["The", "cat", "sat"]
    tags = ["DT", "NN", "VB"]

    # Define probabilities (these would typically come from a trained model)
    transition_prob = {
        ("DT", "NN"): 0.8,
        ("NN", "VB"): 0.6,
        ("VB", "NN"): 0.1,
        ("DT", "VB"): 0.1,
        ("NN", "NN"): 0.2,
        ("VB", "VB"): 0.1,
    }

    emission_prob = {
        ("The", "DT"): 0.9,
        ("cat", "NN"): 0.8,
        ("sat", "VB"): 0.7,
        ("The", "NN"): 0.1,
        ("cat", "VB"): 0.1,
        ("sat", "NN"): 0.1,
    }

    initial_prob = {
        "DT": 0.6,
        "NN": 0.3,
        "VB": 0.1,
    }

    # Run Viterbi algorithm
    best_tags = viterbi(words, tags, transition_prob, emission_prob, initial_prob)
    print("Most likely POS tags:", best_tags)
