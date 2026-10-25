# Automata Learning Lab

Interactive Streamlit project for learning approximate DFAs from labeled string examples, visualizing automata, tracing inputs, and comparing learned automata with scikit-learn classifiers.

## Install

Python 3.10+ is recommended. Install the system Graphviz executable as well as the Python packages:

- Ubuntu/Debian: `sudo apt install graphviz`
- macOS: `brew install graphviz`
- Windows: download and install from [graphviz.org](https://graphviz.org/download/)

Then, from this directory:

```bash
pip install -r requirements.txt
streamlit run app.py
```

The application opens at <http://localhost:8501>.

## Demo flow

1. Choose a regular language and generate labeled examples in the sidebar.
2. Inspect the dataset and ground-truth DFA.
3. Run learning to see the positive-example prefix-tree acceptor, state-merged DFA, and reference DFA.
4. Test custom or batch strings and inspect the execution trace.
5. Compare the DFA with decision-tree, logistic-regression, and random-forest models.
6. Run sample-size and noise experiments.

The learner is a deterministic simplified PTA/state-merging algorithm constrained by observed negative examples; it is not a guarantee of recovering the unique target DFA from finite data. Approximate equivalence is measured over a reproducible random sample.
