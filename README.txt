
---

 Project Title: Homework 1: Vector space word similarity

 Description

lorder function , def create_term_document_matrix(sentences, vocab):, def create_term_context_matrix(sentences, vocab, context_window_size=1): are modify from the part 1, other functions keeps same for part 1.
 How to Run
Ensure Python and all required packages are installed.
Place the Shakespeare dataset (shakespeare_plays.csv), vocab.txt, and play_names.txt, SNLI dataset (`snli.csv`) and the `identity_labels.txt`in the root directory.
Run the script using Python from the terminal for part 1:
python hw1_skeleton_xiw249.py
Run the script using Python from the terminal for part 2:
python hw1_part2_xiw249.py
    ```
for more detail please see the comment inside the py files
 Computing Environment

- Programming Language: Python 3.8.5
- Packages Required:
import os
import subprocess
import csv
import re
import random
import numpy as np
import scipy

 Additional Resources

- SNLI Corpus: [The Stanford Natural Language Inference (SNLI) Corpus](https://nlp.stanford.edu/projects/snli/)
- Python Documentation: [Python 3 Documentation](https://docs.python.org/3/)
- NumPy Documentation: [NumPy v1.19 Manual](https://numpy.org/doc/1.19/)

 Collaborations

No direct collaborations with individuals. Discussions and clarifications were made using OpenAI's ChatGPT to understand the requirements and for code examples.

 Generative AI Usage

OpenAI's ChatGPT was used to:
Generating a template and guidance on implementing TF-IDF and PPMI matrices in Python.
Suggestions for debugging common issues encountered during development.

Provide code explanations for processing the SNLI corpus.
Assist in drafting this README file.


 Unresolved Issues

No unresolved issues at the time of writing. For any errors or problems encountered during setup or execution, please ensure all package dependencies are correctly installed and Python version requirements are met.

 License

Include any license information here.

---

Ensure to update sections like Project Title, Description, and License as per your project specifics. This README template provides a general structure, including how to run the code, computing environment details, additional resources, collaboration mentions, generative AI tool usage, and a section for any unresolved issues or problems.