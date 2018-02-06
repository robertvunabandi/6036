# README

Various Machine Learning related algorithms from MIT 6.036 Intro to ML class. All code snippets are written in `Python 3` syntax.

# ALGORITHMS

## Notes

Algorithms are listed in order of how they were introduced throughout the course. 

Remember that every vector in 6.036 are considered to be column vector. For that reason, to perform a [dot product](https://en.wikipedia.org/wiki/Dot_product) on two vectors, one has to _transpose_ one of them. I.e., if `X` and `Y` are `d`-dimensional vectors, their dot product will be written `transpose(X) * Y` or `transpose(Y) * X`.

For code skeleton, the name and type are denoted with `name:Type`. `name` is a string, and `Type` is the type, which can be any of the following:
- `Integer`: an natural number.
- `Float`: a real number.
- `Boolean`: a python boolean. This can be the result of a conditional check (for example, `2 < 3` is `False`).
- `List`: a python list. It's content is represented within `<...>`. The number of items inside can be anything.
- `Tuple`: a python tuple. Its content is represented within `(...)`. The number of items is however many comma separated values there are.

For some terminology used in this glossary, see [definitions](#definitions) section. Words taken from the definitions section will be _italicized_. 

## Table of Contents

- [`Perceptron`](#perceptron)

## Algorithm Skeletons

### `Perceptron`

[Perceptron](https://en.wikipedia.org/wiki/Perceptron) is a _CLA_ efficient in producing _linear classifiers_.

#### Skeleton: `Perceptron`

Assume we have `d:Integer`, the dimension of each _data point_; `Sn:List<X:Tuple>`, the _data set_ containing each data point (which is denoted by `X:Tuple(List<Float>,Integer)`, a tuple containing an list of numbers (the features of X) and its nature (what the classifier should output)); and `T:Integer` the number of times we run through the algorithm.
```python
# arbitrarily set theta and theta_0
theta, theta_0 = [0 for i in range(d)], 0
for i in range(T):
	for i in range(d):
		X, Y = Sn[i][0], Sn[i][1]
		condition = Y * (transpose(theta) * X + theta_0) <= 0
		if condition:
			theta = theta + Y * X
			theta_0 = theta_0 + Y
return theta, theta_0
```

A note here is that if there was no update, one does not need to go through the outer loop again. So, there should be a check for that and stop in case no update happened. 
Extra links: [1](http://www.ciml.info/dl/v0_8/ciml-v0_8-ch03.pdf).
#### Variant: `Pocket Perceptron`

The pocket perceptron returns the best `theta` and `theta_0` from the algorithm ran above. How it judges "best" is by the number of failures that the perceptron had while running. We use `failure_count` to identify the best _classifier_ parameters `theta` and `theta_0`. Initially, set the `failure_count` to be a really high number Here, we used `sys.maxsize`, which means we imported `sys` with `import sys`.

```python
# arbitrarily set theta and theta_0
theta, theta_0, failure_count = [0 for i in range(d)], 0, sys.maxsize
best = (theta, theta_0, failure_count)
for i in range(T):
	failure_count = 0
	for i in range(d):
		X, Y = Sn[i][0], Sn[i][1]
		condition = Y * (transpose(theta) * X + theta_0) <= 0
		if condition:
			failure_count += 1
			theta = theta + Y * X
			theta_0 = theta_0 + Y
	if failure_count < best[2]:
		best = (theta, theta_0, failure_count)
return best[0], best[1]
```

#### Variant: `Voted Perceptron`

The voted perceptron returns the best `theta` and `theta_0` from the algorithm ran above. How it judges "best" is by how long the classifier survives before needing to change. We use `survival_count` to identify the best _classifier_ parameters `theta` and `theta_0`. Initially, set the `survival_count` to 0 to start with, meaning it has not survived anything.

```python
# arbitrarily set theta and theta_0
theta, theta_0, survival_count = [0 for i in range(d)], 0, 0
best = (theta, theta_0, survival_count)
for i in range(T):
	survival_count = 0
	has_failed = False
	for i in range(d):
		X, Y = Sn[i][0], Sn[i][1]
		condition = Y * (transpose(theta) * X + theta_0) <= 0
		if condition:
			failure_count += 1
			theta = theta + Y * X
			theta_0 = theta_0 + Y
			has_failed = True
		elif not has_failed:
			survival_count += 1
	if survival_count > best[2]:
		best = (theta, theta_0, survival_count)
return best[0], best[1]
```

#### Variant: `Average Perceptron`

The average perceptron returns the average of all the `theta` and `theta_0` from the algorithm ran above. 

**TODO: Add snippet for Average Perceptron. _I am actually not sure how this one works._**

[Back to Table of Contents](#table-of-contents)

# DEFINITIONS

- Classifier: A function `h: R^d -> {+1,-1}` that maps a `d`-dimensional _data point_ to either `1` or `-1`. 
	- Linear Classifier: A _classifier_ for which the mapping rule is defined as `h(x; theta, theta_0) = transpose(x) * theta + theta_0` where `x, theta` are `d`-dimensional and `theta_0` is one-dimensional. 
- Classifier Learning Algorithm (CLA): An algorithm that takes in a _data set_ and outputs a _classifier_ that is good at classifying data taken from that _data set_. I.e., a CLA often outputs a _classifier_ that minimizes the _testing error_ amongst a set of possible _classifiers_. [More on Leaning Algorithms](https://www.igi-global.com/dictionary/learning-algorithm/16821).
- Data point: A piece of information taken from a _data set_.
- Data set: Data collected / aggregated with the purpose of running "machine learning" on it. One of such purposes would be to classify each data point in the data set using a _classifier_. 
- Feature map: A function `phi: X -> R^d` that maps a piece of information (item to be studied) `X` to a `d`-dimensional _data point_, thus extracting the "features" of `X`. 
- `onefxn`: This is a function `f: Boolean -> {1, 0}` that outputs 1 if the boolean given is true and outputs zero if the boolean given is false.
- Testing error: Given a _data set_ `S_n` containing pairs `(x_i, y_i)` for `i = 1,...,n` and assuming we have `k` extra data set not used for _training error_, the testing error is a function defined as `epsilon(h) = 1/k * sum(i = k to n + k, onefxn(h(x_i) == y_i))`. See definition of `onefxn` for more info.
- Training error: Given a _data set_ `S_n` containing pairs `(x_i, y_i)` for `i = 1,...,n`, the testing error is a function defined as `epsilon_n(h) = 1/n * sum(i = 1 to n, onefxn(h(x_i) == y_i))`. See definition of `onefxn` for more info.
- `transpose`: see [Wikipedia](https://en.wikipedia.org/wiki/Transpose) definition.

