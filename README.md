# `composit`
ML framework for composing neural networks and mapping them to heterogeneous backends with as minimum levels of abstraction as possible

![tests](https://github.com/arakhmati/composit/actions/workflows/python-app.yml/badge.svg)

# Installation Instructions
```bash
./install.sh
```

# Why `composit`?
There are plenty of ML frameworks out there, so what's the point of this one?
The best way to answer this question is to let the reader try to add a new hardware-specific feature to an existing framework.
That could be something as small as adding a new operation and as big as implementing a backend for a whole new accelerator.
That's usually when most of the frameworks out there stop being user-friendly and start becoming very cumbersome to deal with.

The aim of `composit` is to make it easier to add new hardware-specific features. And, in general, to encourage hand-made approach
for writing neural networks. That means giving the user an ability to map a given neural network architecture to the desired hardware
and allowing the user to manually optimize during any step of the process.

`composit` is written in a way that allows it to be easily extended. Adding a new operation or a new backend means adding on top of `composit`, not to it.

`composit` is written using pure functions which helps it achieve the stated goals.

# Structure of `composit`

## `composit`
### `numpy`
The main building block of `composit` is its `numpy` submodule.

This submodule allows writing computational graphs using `numpy` APIs.
It's important to note that it's not `numpy`-like, but instead it's exactly like `numpy` (with, so far, only the notable exception of `__setitem__` method of `np.ndarray`)

##### Simplest example
Here is one of the simplest examples of specifying computations using `composit`'s `numpy`:
```python
import composit.numpy as cnp
from composit.types import LazyTensor

input: LazyTensor = cnp.ones((2, 3), dtype="float32")
output: LazyTensor = cnp.exp(input)
```
The code above looks exactly the same as the one that would be written using `numpy` itself if `cnp` was replaced with `np`.
The difference is that no computations have been actually performed.
What happened instead was:
1. `np.ones` created a constant array of ones and stored it in a node
2. `np.exp` created an exponent instruction that will be used to evaluate the `output`, stored it in another node and added an edge from the node of `input` to the new node

And that is possible because every `LazyTensor` stores the computational graph, needed to evaluate it, as `graph` attribute.
But obviously having the `graph` is not always enough to know what node is the output one, so it also has a `node` attribute to specify which node corresponds to the given `LazyTensor`.
And finally, some instructions like `np.split` can have multiple outputs themselves, so `LazyTensor` has an `output_index` attribute
to handle that case.

As part of `numpy` interface, `LazyTensor` has `shape`, `dtype` and `rank` property methods.

Finally, it also has a `name` property method.

An obvious question, is "Where are these properties stored?". And the answer is that they are stored inside of the `graph`.
This will be explained in more detail later, but for now, all that is needed be to known is that 
the `graph` is a persistent version of `networkx.MultiDiGraph`. And `networkx.MultiDiGraph` allows storing the attributes on both nodes and edges.


Finally, let's evaluate the graph:
```python
np_output: np.ndarray = cnp.evaluate(output)
```

All that was needed was just a call to `cnp.evaluate`. 
Internally, it traversed the graph, skipped over the input and ran the exponent instruction using `numpy` as the backend.

A key thing to get out of this example is that `cnp.evaluate` operates on top of the tensors.
So, theoretically (and practically, as will be shown later), it can be re-written using any other backend.

##### Multi-input and multi-output example
Let's dive into another example. This time with multiple inputs and outputs.
```python
import composit as cnp
from composit.types import LazyTensor

input_a: LazyTensor = cnp.ones((5, 10), dtype="float32")
input_b: LazyTensor = cnp.random.random((10,))
input_c: LazyTensor = cnp.random.random((5, 1))
exp: LazyTensor = cnp.exp(input_a + input_b)
tanh: LazyTensor = cnp.tanh(input_a + input_c)
```

`composit` is built with visualization in mind and so the computational graph 
can be easily visualized. All that needs to be done is to pass in the output `LazyTensor`s into `visualize` function:
```python
from composit.types import visualize
visualize(exp, tanh, render=True)
```
![Example 0](docs/images/composit_numpy_0.svg)

As mentioned earlier, a tensor in the computational graph is only aware of the operations leading towards it and the visualization obivously reflects that:
```python
from composit.types import visualize
visualize(exp, render=True)
```
![Example 1](docs/images/composit_numpy_1.svg)

And so does, evaluation:
```python
# Evaluate `input_c` by simply returning the data stored in `input_c`'s node
assert len(input_c.graph) == 1
np_input_c: np.ndarray = cnp.evaluate(input_c)

# Evaluate `exp` (`exp`'s graph only has nodes need to evaluate it)
assert len(exp.graph) == 4
np_exp: np.ndarray = cnp.evaluate(exp)

# Evaluate `exp` and `tanh` (re-use common subgraph)
np_exp, np_tanh = cnp.evaluate(exp, tanh)
```

There is a lot more to explain about the internals of `composit.numpy` but it's more exciting to show what can be built on top of its APIs.

### `nn` (neural networks submodule)
WIP