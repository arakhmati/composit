# Composit = Compose it + Composite
Composit composes computational graphs using numpy-like API and then it breaks down composite tensors into tiles

## Tests
To get familiar with the code, read tests in this order:
1. tests/unit_tests/numpy/test_functions.py
2. tests/unit_tests/numpy/test_evaluate.py
3. tests/unit_tests/numpy/test_chain_rule.py
4. tests/unit_tests/numpy/test_train.py
5. tests/unit_tests/numpy/test_module.py
6. tests/integration_tests/test_bert.py
7. tests/unit_tests/tilelab/test_tilelab.py
8. tests/unit_tests/backends/x86/test_matmul.py

Each test builds up on the concepts from the previous tests

(TODO: add documentation for all of the concepts)