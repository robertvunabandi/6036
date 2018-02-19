# README

Various Machine Learning related algorithms from MIT 6.036 Intro to ML class. All code snippets are written in `Python3` syntax assuming [`numpy`](http://www.numpy.org/) is installed.

# NOTES

Algorithms are listed in order of how they were introduced throughout the course. 

Remember that every vector in 6.036 are considered to be column vector. For that reason, to perform a [dot product](https://en.wikipedia.org/wiki/Dot_product) on two vectors, one has to _transpose_ one of them. I.e., if `X` and `Y` are `d`-dimensional vectors, their dot product will be written `transpose(X) * Y` or `transpose(Y) * X`.

For code skeleton, the name and type are denoted with `name:Type`. `name` is a string, and `Type` is the type, which can be any of the following:
- `Integer`: an natural number.
- `Float`: a real number.
- `Boolean`: a python boolean. This can be the result of a conditional check (for example, `2 < 3` is `False`).
- `List`: a python list. It's content is represented within `<...>`. The number of items inside can be anything.
- `Tuple`: a python tuple. Its content is represented within `(...)`. The number of items is however many comma separated values there are.

For some of the terminology used here, see [definitions](#definitions) section. Words taken from the definitions section will be _italicized_. 

# TABLE OF CONTENTS

- [`ALGORITHMS`](#algorithms)
	- [`Perceptron`](#perceptron)
- [`DEFINITIONS`](#definitions)

# ALGORITHMS

## `Perceptron`

[Perceptron](https://en.wikipedia.org/wiki/Perceptron) is a _CLA_ efficient in producing _linear classifiers_. Extra links: [1](http://www.ciml.info/dl/v0_8/ciml-v0_8-ch03.pdf), [2](https://en.wikipedia.org/wiki/Perceptron). The details for the method parameters are:
```
both arguments are 2-dimensional numpy arrays.
data_set.shape = (d, n)
labels.shape = (1, n)
```

[:small_red_triangle: Back to Table of Contents](#table-of-contents)

#### Skeleton: `(Vanilla) Perceptron(data_set, labels)`

```python
d, n = data_set.shape
theta, theta_0 = np.zeros((d, 1)), np.zeros((1,1))
for trial in range(T):
    for index in range(n):
        data_point, label = data_set[:, index:index+1], labels[:,index:index+1]
        activation = label * np.sign(np.dot(np.transpose(theta), data_point) + theta_0)
        if activation <= 0.0:
            theta = theta + (data_point * label)
            theta_0 = theta_0 + label
return theta, theta_0
```

**Notes**: 
- If there was no update during one of the outer loop iteration, one does not need to go through it again. So, there should be a check for that and stop in case no update happened. 
- To prevent various negative effects from happening, it's beneficial to shuffle the _data set_ before each outer loop iteration. That will often lead to a faster convergence. 

#### Variant: `Pocket Perceptron(data_set, labels)`

The pocket perceptron returns the best `theta` and `theta_0` from the algorithm ran above. The "best" parameters are the ones that yield the least failures. We use `failure_count` to identify the best _classifier_ parameters `theta` and `theta_0`. Initially, set the `failure_count` to be a really high number Here, we used `sys.maxsize`.

```python
d, n = data_set.shape
theta, theta_0, failure_count = np.zeros((d, 1)), np.zeros((1,1)), sys.maxsize
best = (theta, theta_0, failure_count)
for trial in range(T):
    failure_count = 0
    for index in range(n):
        data_point, label = data_set[:, index:index+1], labels[:,index:index+1]
        activation = label * np.sign(np.dot(np.transpose(theta), data_point) + theta_0)
        if activation <= 0.0:
            failure_count += 1
            theta = theta + (data_point * label)
            theta_0 = theta_0 + label
    if failure_count < best[2]:
        best = (theta, theta_0, failure_count)
return best[0], best[1]
```

#### Variant: `Voted Perceptron(data_set, labels)`

The voted perceptron returns the best `theta` and `theta_0` from the algorithm ran above. It judges "best" by how long the classifier survives before needing to change. We use `survival_count` to identify the best _classifier_ parameters `theta` and `theta_0`. Initially, set the `survival_count` to 0 to start with, meaning it has not survived anything.

```python
d, n = data_set.shape
theta, theta_0, survival_count = np.zeros((d, 1)), np.zeros((1,1)), 0
best = (theta, theta_0, survival_count)
for trial in range(T):
    for index in range(n):
        data_point, label = data_set[:, index:index+1], labels[:,index:index+1]
        activation = label * np.sign(np.dot(np.transpose(theta), data_point) + theta_0)
        if activation <= 0.0:
            theta = theta + (data_point * label)
            theta_0 = theta_0 + label
            if survival_count > best[2]:
                best = (theta, theta_0, survival_count)
            survival_count = 0
        else:
            survival_count += 1
return best[0], best[1]
```

#### Variant: `Average Perceptron(data_set, labels)`

The average perceptron returns the average of all the `theta` and `theta_0` from the algorithm ran above.

```python
d, n = data_set.shape
theta, theta_0, coeff = np.zeros((d, 1)), np.zeros((1,1)), 1
thetas, theta_0s = np.zeros((d, 1)), np.zeros((1,1))
for trial in range(T):
    for index in range(n):
        data_point, label = data_set[:, index:index+1], labels[:,index:index+1]
        activation = label * np.sign(np.dot(np.transpose(theta), data_point) + theta_0)
        if activation <= 0.0:
            theta = theta + (data_point * label)
            theta_0 = theta_0 + label
            thetas = thetas + (coeff * data_point * label)
            theta_0s = theta_0s + (coeff * label)
        coeff = coeff - (1/(n*T))
return thetas, theta_0s
```

**Notes**: 
- One may need to start recording values to average only after a setup time when `theta` and `theta_0` are more or less semi-efficient. 

[:small_red_triangle: Back to Table of Contents](#table-of-contents)

# DEFINITIONS

- Classifier: A function `h: R^d -> {+1,-1}` that maps a `d`-dimensional _data point_ to either `1` or `-1`. 
	- Linear Classifier: A _classifier_ for which the mapping rule is defined as `h(x; theta, theta_0) = transpose(x) * theta + theta_0` where `x, theta` are `d`-dimensional and `theta_0` is one-dimensional. 
- Classifier Learning Algorithm (CLA): An algorithm that takes in a _data set_ and outputs a _classifier_ that is good at classifying data taken from that _data set_. I.e., a CLA often outputs a _classifier_ that minimizes the _testing error_ amongst a set of possible _classifiers_. [More on Leaning Algorithms](https://www.igi-global.com/dictionary/learning-algorithm/16821).
- Data point: A piece of information taken from a _data set_.
- Data set: Data collected / aggregated with the purpose of running "machine learning" on it. One of such purposes would be to classify each data point in the data set using a _classifier_. 
- Feature map: A function `phi: X -> R^d` that maps a piece of information (item to be studied) `X` to a `d`-dimensional _data point_, thus extracting the "features" of `X`. 
- `onefxn`: This is a function `f: Boolean -> {1, 0}` that outputs 1 if the boolean given is true and outputs zero if the boolean given is false. This is called [Kronecker's delta function](https://en.wikipedia.org/wiki/Kronecker_delta).
- Testing error: Given a _data set_ `S_n` containing pairs `(x_i, y_i)` for `i = 1,...,n` and assuming we have `k` extra data set not used for _training error_, the testing error is a function defined as `epsilon(h) = 1/k * sum(i = k to n + k, onefxn(h(x_i) == y_i))`. See definition of `onefxn` for more info.
- Training error: Given a _data set_ `S_n` containing pairs `(x_i, y_i)` for `i = 1,...,n`, the testing error is a function defined as `epsilon_n(h) = 1/n * sum(i = 1 to n, onefxn(h(x_i) == y_i))`. See definition of `onefxn` for more info.
- `transpose`: see [Wikipedia](https://en.wikipedia.org/wiki/Transpose) definition.

[:small_red_triangle: Back to Table of Contents](#table-of-contents)