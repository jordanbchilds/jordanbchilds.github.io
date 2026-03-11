---
layout: page
title: "A simple discrete random sampler in C++"
author: "Jordan Childs"

mathjax: true

tags: ["C++", "Random sampling"]

permlink: /blogs/discreterandomsampler
---

```cpp
#include <vector>
#include <stdexcept>

// un-normalised weights
template<typename T>
std::size_t rdiscrete (const std::vector<T>& weights) {
    T norm = 0.0;
    for (const T& w : weights) {
        if (w<0.0) throw std::domain_error("Weights must be non-negative.");
        norm += w;
    }
    
    double u = (double)rand() / (double)RAND_MAX * (double)norm;
    std::size_t iter = 0;
    for (const T& w : weights) {
        u -= w;
        if (u < 0.0) return iter;
        iter++;
    }
    return iter;
}

// (un-normalised) weights with a known normalising constant
template <typename T>
std::size_t rdiscrete (const std::vector<T>& weights, const T& norm) {
    if (norm<=0) std::domain_error("norm must be greater than zero.");
    
    double u = (double)rand() / (double)RAND_MAX * (double)norm;
    std::size_t iter = 0;
    for (const T& w : weights) {
        if (w<0.0) std::domain_error("weights must be non-negative.")
        u -= w;
        if (u < 0.0) return iter;
        iter++;
    }
    return iter;
}

```

Here we will briefly discuss how to sample a discrete random variable in theory and practice, before constructing our own algorithm and comparing its performance to the `std::discrete_distribution` in the C++ standard library. Many hand-coded discrete samplers have been proposed, but often these are unnecessarily clunky and, consequently, slow. The sampler proposed here is a function, designed to take an `std::vector` input representing the weights of the possible outcomes and return an `std::size_t` value indicating the randomly sampled vector index. Two versions of the function are given above, one where the sum of the weights (normalising constant) is known and one where it is not. If the normalising constant is not known, then it is calculated and the weights are checked to be non-negative, and if the it is given, it is assumed that this has already been checked.

# Brief theory

A discrete random variable (DRV) is one that can take a finite set of values, each with a defined probability of occurring. We assume that we are sampling a number between 0 and n-1, following vector indexes notation. A DRV can be sampled by starting with a random draw from a uniform(0,1) distribution and back calculating the its value on the cumulative mass function. For a random variable, *X*, which can take values 0,...,n-1, let the probabilities and cumulative probabilities be denoted as follows

$$ \text{Pr} \left( X=i \right) = p_i, $$
$$ \text{Pr} \left( X \leq i \right) = \sum_{j=0}^{i}p_i = c_i. $$

A function to sample a DRV should execute the following two steps. 

1. Sample: $$ U \sim \text{Uniform} \left(0, 1\right) $$
2. Find *i* such that $$ c_{i-1} \leq U \le c_i $$

Where $c_{-1}=0.0$. Hard coding these two steps directly is not particularly hard but the computational expense can be reduced by noting two things. Firstly, the cumulative mass function is usually not known and its calculation incurs additional cost. Secondly, the double comparison in step 2 is unnecessary. The second point can be dealt with by comparing the uniform random draw to the CMF sequentially. Knowing it is greater than the previous CMF value, $c_{i-1}$, it is then only necessary to check that it is smaller than $c_i$ before moving onto $c_{i+1}$. However, this still leaves the problem of calculating the CMF. 

We can re-write the comparison in terms of the probabilities as such
$$ U \le \sum_{j=0}^{i}p_i. $$
And then rearrange to allow $p_i$ to be on its own, 
$$ U - \sum_{j=0}^{i} p_i \le 0.0. $$

Checking the expression sequentially means with each iteration we only have one subtraction and one comparison. A true result allows the search to stop and no more computations are required. For the case where the un-normalised weights are known and not the probabilities themselves, this can be re-written. Let $p_i = w_i / K$, where $w_i$ is the un-normalised weight and $K$ is the normalising constant. 

$$ U - \frac{1}{K}\sum_{j=0}^{i} w_i \le 0.0, $$
$$ U*K -\sum_{j=0}^{i} w_i \le 0.0.$$

And so, comparing the probabilities against a uniform draw between 0 and 1, is equivalent to comparing the weights between 0 and $K$.

# Generating samples

## Standard library

C++ comes with the standard library (STL), which includes a way to sample a discrete random value, the snippet below shows how this could be done. Note that the `discrete_distrsibution` does not require the probabilities to be normalised, as such the weights do not sum to 1.0.  

```cpp
include <vector>
include <random>

int main () {
    std::default_random_engine generator;
    std::vector<int> weights({1,2,3,1,1,2});
    std::discrete_distribution<int> distribution(weights.begin(), weights.end());
    
    int randomInt = distribution(generator);
}
```

## Proposed function

In the snippet below we sample a DRV using the previously defined function. Although used here the STL header `vector` does not have to be used and a C style array could be used instead. A uniform random number is drawn using the `rand()` function, which randomly draws integers between 0 and the largest `unsigned int` value and is then scaled to be a `double` between 0.0 and 1.0. 

```cpp
/*  
    include or define above functions
*/

int main () {
    std::vector<int> weights({1,2,3,1,1,2});
    std::size_t randomInt = rdiscrete(weights);
}
```

Each weight and the normalising constant is checked to ensure that they positive or non-negative, as appropriate. The function will return `n`, one larger than the largest possible value of the DRV, as an indication that something is wrong. This may occur if the `norm` is greater than the sum of the weights although due to the random nature of the function, it is not guaranteed.

If it is guaranteed that each weight is non-negative, then the check can be removed. However it's additional expense is low and it will immediately flag if something is wrong.

# Comparison

This is a simple problem and so the computational expense (measured in time) is small, nevertheless it is always good to know your code is as fast as it can be. We consider two cases; sampling repeatedly from a single distribution and repeatedly sampling from different distributions. Both cases often occur in practice. 

## Sampling from one distribution

Assuming the STL method calculates the normalising constant once, when the object is defined, we will compare it with the case where the normalising constant is known and passed to the function, as this is more alike.  

We first consider the case of repeated re-sampling from the same distribution. For a varying number of possible outcomes, the figure below shows the results of timing the two methods. It is clear that for a small number of outcomes (10 or less) that there is little difference between the two methods. A larger number of possible outcomes slows down the simpler function drastically whereas the much more optimises STL version appears to be unaffected by the increase. 


## Sampling different distributions

The second case we are interested in is sampling from different distributions. For the STL method this requires re-defining the `std::discrete_distribution` object for each new sample. Following the same assumption as before, calculating the normalising constant is done once upon object definition, for this case will assume the normalising constant is not known and use the appropriate version of our function. For our simple function it requires passing a new vector of weights to the function, for each new sample. Note the different scale of the y-axis compared to the previous plot. It appears the STL object becomes much slower when needing to sample from a different distribution. 

# Final remarks

Depending on yhe use case, whether you're repeatedly drawing from the same distribution or not, you may want to chose either option. However the function(s) provided here are simple and do not require the use of a random number generator object, such as `std::default_random_engine` and instead depends upon the `rand` function. The functions provided here also have the benefit of not depending on any STL package (with modifications to use C style arrays and removing the `std::domain_error`'s) which may not always be available or desired. 
