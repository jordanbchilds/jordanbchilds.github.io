---
title: "Introduction to Bayesian Hierarchical Models"

mathjax: true
tags: ["Statistical Modelling", "Bayesian Methods"]
---

Hierarchical models are synonymous with Bayesian methods, which is better equiped to be able to infer the parameters of large and complex models when compared to frequentist methods. They are popular as they allow simple building blocks to be combined to form a large and complex model. The ability of hierarchical models to reflect complex systems means they have been applied to a variety of modelling situations, within all aspects of science.

In this blog the concept of hierarchical models are introduced, along with an example to demonstrate where one might be applicable, and the inference of their parameters is discussed within the Bayesian paradigm. As such, it is assumed the reader is familiar with Bayesian methods. If not, the blogs [here](https://jordanbchilds.github.io/posts/bayesian_introduction/) introduces Bayesian methods and [here](https://jordanbchilds.github.io/posts/mcmc_introduction/) continues the discussion on parameter inference.

# Bayesian hierarchical model

A Bayesian hierarchical model imposes dependencies between unknown parameters. In practice, prior parameters are considered unknown and to be inferred, reflecting uncertainty in the prior specification itself. To this end, hyper-priors, defined by hyper-parameters, are chosen to summarise beliefs on the prior parameters. The dependent structure enables the construction of complex models from simple components.

Hierarchical models are often used when the dataset can be split into non-overlapping groups, which may impact the response variable. The hierarchy allows each group to be modelled with a different parameter while learning the population-level parameters in the hyper-parameters. The top level of the hierarchy describes beliefs about the population-level parameters, which concern the whole dataset. After that, each level will describe a smaller subset of the data. 

One benefit of using a Bayesian hierarchical model is that information is shared between groups through the learning of top-level parameters, referred to as borrowing strength between groups. Whole-population parameters beliefs are passed on to sub-population parameters, minimising over-fitting risks for groups with few data points.

Mixture models, see [here](https://jordanbchilds.github.io/posts/mixture_models/) and hierarchical models are often combined, as they both concern data sub-populations. When this is the case, inference can proceed as usual, but care must be taken when calculating full conditional distirbutions (FCDs) due to the model's more complex structure.

# Bayesian hierarchical model example

Suppose we wish to model the proportion of students achieving a pass in five GCSEs (or more) within schools across the UK. It is reasonable to assume that the pass rate will vary between schools. As such, aggregating the UK-wide data would not be informative of the whole picture. Alternatively, each school could be modelled independently, but this would ignore the information within the dataset from other schools. The hierarchical model addresses both issues, allowing each school to have a unique pass rate while borrowing strength across schools.  

## The data

Suppose the dataset contains information for $N$ uniquely identified schools across the UK. Let $n_i$ be the number of students in the $i$-th school, and $Y_i$ be the number of students achieving a pass grade in five (or more) GCSEs in that school. Table 1 shows a snippet of what such a dataset may look like.

| School | $n$  |  $Y$ |
| :----- | ---- | ---- |
|1 | 120 | 95 |
|2 | 99 | 65 |
|3 | 108 | 71 |

Table 1: **Example subset of data concerning GCSE results of UK schools**.

## The model

Assume $Y_i$ follows a binomial distribution, with probability of success $\theta_i$, and the number of children in each school is known, then
$$
    Y_i | n_i, \theta_i \sim \text{Bin}\left(n_i, \theta_i\right).
$$

The parameters $\theta_1, \theta_2, \dots, \theta_N$ are unknown and to be inferred. The pass rates are proportions, so it is reasonable to model them using a Beta distribution, 
$$
    \theta_i | \alpha, \beta \sim \text{Beta}\left(\alpha, \beta \right)
$$

The hierarchy is implemented by considering the prior parameters $\alpha$ and $\beta$ unknown and, therefore, to be inferred. Both are continuous and positive; independent gamma distributions are one choice for the hyper-priors.
$$
\begin{split}
    \alpha &\sim \text{Ga}\left(a_1, b_1 \right) \\
    \beta &\sim \text{Ga}\left(a_2, b_2 \right)
\end{split}
$$

The hyper-parameters, $\{a_1, b_1, a_2, b_2\}$, are known and chosen by us to summarise prior beliefs. By inferring $\alpha$ and $\beta$, the distribution of a nationwide success rate can be inferred from the whole dataset. Which, in turn, affects the distribution of school-level success rates. The relationship can be seen by calculating the parameter FCDs up to proportionality.

## Parameter inference

First, consider the FCDs of the prior parameter $\alpha$,
$$
\begin{aligned}
    p(\alpha | \beta, \boldsymbol{\theta}, \boldsymbol{y}) &\propto p\left(\alpha, \beta, \boldsymbol{\theta}, \boldsymbol{y}\right), \\\\
    &\propto p(\boldsymbol{y} | \boldsymbol{\theta}) p (\boldsymbol{\theta} | \alpha, \beta) p(\alpha) p(\beta), \\\\
    &\propto p(\boldsymbol{\theta} | \alpha, \beta) p(\alpha), \\\\
    &\propto \prod_{i=1}^n \left\{ {\theta_i}^{\alpha-1} \left( 1 - \theta_i\right)^{\beta-1}\right\} \times \alpha^{a_1 -1} e^{-b_1 \alpha}, \\\\
     &\propto \alpha^{a_1 -1} e^{-b_1 \alpha} \prod_{i=1}^n{\theta_i}^{\alpha-1}. \\
\end{aligned}
$$
Unfortunately, this is not the form of a known probability distribution, and therefor the analysis is not semi-conjugate. Despite this, inference can still proceed by implementating MCMC methods, see [here](https://jordanbchilds.github.io/posts/mcmc_introduction/). Nevertheless, it can be seen that given $a_1, b_1$ and $\boldsymbol{\theta}$, $p(\alpha | \beta, \boldsymbol{\theta}, \boldsymbol{y})$ is independent of $\beta$ and $\boldsymbol{y}$. Similarly, calculating the FCD of $\beta$ shows that given $a_2, b_2$ and $\boldsymbol{\theta}$, it is independent of the observations $\boldsymbol{y}$ and $\alpha$. 
$$
    p(\beta | \alpha, \boldsymbol{\theta}, \boldsymbol{y}) 
     \propto \alpha^{a_\beta -1} e^{-b_\beta \beta} \prod_{i=1}^n\left( 1 - \theta_i\right)^{\beta-1}
$$

The final parameters to consider are the individual school success rates
$$
\begin{aligned}
    p(\theta_i | \alpha, \beta, \boldsymbol{\theta}_{-i}, \boldsymbol{y}) &\propto p\left(\alpha, \beta, \boldsymbol{\theta}, \boldsymbol{y}\right), \\\\
    &\propto p(\boldsymbol{y} | \boldsymbol{\theta}) p (\boldsymbol{\theta} | \alpha, \beta), \\\\
    &\propto \prod_{i=1}^n \left\{ {\theta_i}^{y_i} \left( 1 - \theta_i\right)^{n_i - y_i}\right\} \times \prod_{i=1}^n\left\{ {\theta_i}^{\alpha-1} \left(1 - \theta_i)^{\beta -1}\right)\right\}, \\\\
     &\propto {\theta_i}^{y_i + \alpha - 1} \left(1 - \theta_i\right)^{n_i - y_i + \beta - 1}.
\end{aligned}
$$
This is the form of a Beta distribution with parameters $y_i+\alpha$ and $n_i - y_i + \beta$, and, therefore, 
$$
    \theta_i | \alpha, \beta, y_i \sim \text{Beta} \left(y_i+\alpha, n_i - y_i + \beta \right).
$$
The FCD of the individual school success rates depends on the observed data of only that school, $y_i$, and the population-level parameters, $\alpha$ and $\beta$. 

By learning high-level parameters, $\alpha$ and $\beta$, information about the global population is passed to lower-level parameters, $\theta_i$, which are specific to a sub-population. Therefore, information is transferred to sub-population-specific parameters from the whole dataset, borrowing strength across groups. When there is little data in sub-populations, sharing information across groups is particularly useful; alternatively, treating each sub-population independently may lead to over-fitting in either a Bayesian or frequentist setting. 

Hierarchical models can become more complex when additional layers to the hierarchy are added. For example, suppose the schools are grouped into geographical regions. Another layer could be added which learns region-specific success rates. 

# Final remarks

Hierarchical models are broad class of statistical model which are able to better match the realities of the data which they are modelling. The Bayesian paradigm, and it's consitent treatment unkowns is well equiped to handle the structured nature and hierarchical models, where likelihood functions may be intractable. 

In this blog a small modelling example is given which hopefully gives an indication of the types of models that could used but also an understanding of why you may use a Bayesian hierarchical at all. 

