---
title: "Introduction to Bayesian Statistics"

mathjax: true
tags: ["Statistics", "Bayesian Methods"]
---

Unlike frequentist statisticians, Bayesian statisticians believe that there is no 'true' value of the parameters in a statistical model. Instead, Bayesian methods summarise parameter beliefs after observing data (*a posteriori*) by a probability distribution, giving more weight to more likely values. Parameter beliefs before observing any data (*a priori*) are called prior beliefs, and are similarly summarised by a probability distribution. Prior beliefs can be as vague or informed as required to reflect the beliefs of relevant experts. A lack of prior can be reflected by vague prior beliefs, with a high variance. In contrast, a large amount of prior knowledge from previous work may result in well-informed priors with high precision. Parameter beliefs are updated by combining prior beliefs and new evidence presented from a dataset. Bayesian methodology is fully probabilistic and considers model parameters, hidden states, as well as missing and observed data in the same vein. The coherent treatment of model parameters, data, and hidden states has allowed Bayesian methods to be used in a wide range of inference problems. 

Bayesian methods have gained popularity in all aspects of research. Historically the limiting factor to conducting Bayesian analysis was its computational expense. However, the rise in computing power has changed this. Applications range across disciplines including; biological sciences and mathematical finance. 

# Bayes' theorem

Bayes' theorem is the foundation of Bayesian statistics and provides the formula to find posterior beliefs. It follows immediately from the law of conditional probability, see [here](https://en.wikipedia.org/wiki/Conditional_probability).

$$
    \text{Pr}(A|B) = \frac{ \text{Pr}(B|A)\,\text{Pr}\left(A\right) } {\text{Pr}\left(B\right)}.
$$

The events $A$ and $B$ are interpreted as model parameters and observed data, respectively. Let $\boldsymbol{x} = \left(x_1, x_2, \dots, x_n\right)$ be a set of independent observations and $f_X(x|\theta)$ be a statistical model that is believed to explain the data. The model, $f_X(x|\theta)$, is dependent on a set of parameters, $\theta$, which are unknown and to be inferred. Let $\Theta$ denote the support of the model parameters, $\theta$. For example, in a simple linear model, $\theta=(m, c)^\mathrm{T}$ indicating the slope and intercept of the linear equation, and its support is the 2-dimensional real-plane, $\mathbb{R}^2$. The data-likelihood, $f(\boldsymbol{x}|\theta)$, is the probability of observing $\boldsymbol{x}$ given a set of parameter values, $\theta$, and is the product of the model density evaluated at each observation given $\theta$,
$$
    f\left(\boldsymbol{x}|\theta\right) = \prod_{i=1}^n f_X(x_i | \theta). 
$$

The probability density summarising the prior beliefs is often denoted $p(\theta)$. Substituting these elements into law of conditional probability and using the law of total probability to write the denominator with known entities, we get Bayes' theorem; 

$$
\begin{split}
    p \left(\theta | \boldsymbol{x} \right) &= \frac {p(\theta) f\left(\boldsymbol{x}|\theta\right)} { \int_\Theta f(\boldsymbol{x} | \theta) p(\theta) d\theta}.
\end{split}
$$

In Bayesian inference the observed data, $\boldsymbol{x}$, are considered fixed and known. Therefore, the likelihood function is viewed as a function of $\theta$ dependent $\boldsymbol{x}$. The equation is often rewritten using the parameter-likelihood function; its definition is identical to the data-likelihood function, but the data are considered known and, therefore, the given entity. Let $L(\theta|\boldsymbol{x})$ be the parameter-likelihood function and 

$$
    L\left(\theta | \boldsymbol{x}\right) = f(\boldsymbol{x}|\theta) =  \prod_{i=1}^n f_X(x_i | \theta). 
$$

Bayes' theorem can then be rewritten using the parameter-likelihood
$$
    p \left(\theta | \boldsymbol{x} \right) = \frac {p(\theta) L\left(\theta | \boldsymbol{x}\right)} { \int_\Theta L(\theta | \boldsymbol{x} ) p(\theta) d\theta}.
$$

The denominator, $\int_\Theta f(\theta | \boldsymbol{x} ) p(\theta) d\theta$, is referred to as the marginal likelihood and is a constant for a given dataset, $\boldsymbol{x}$. The integrand in the denominator is equal to the numerator of Bayes' theorem, and the integral therefore acts as a normalising constant ensuring that the posterior density integrates to 1.0. As it is a constant, Bayes' theorem is often simplified to express the posterior up to proportionality,
$$
    p \left(\theta | \boldsymbol{x} \right) \propto p(\theta) L\left(\theta | \boldsymbol{x}\right).
$$

The posterior distribution, up to proportionality, is therefore the prior density function multiplied by the likelihood function, the result is a function of $\theta$. How this formula is used to find a posterior distribution is discussed next. 

# Conjugate analysis

In simple cases, the posterior distribution can be found analytically when the multiplication, $p(\theta) \times L(\theta|\boldsymbol{x})$, evaluates to be a known probability density up to proportionality. This is called conjugate analysis, an example is shown below. Specific models and prior beliefs are known to be conjugate. Although restrictive, a variety of models and prior beliefs can be formed this way. Conjugacy is helpful because there is (almost) no computational cost to calculating the posterior distribution, only a human cost of doing the maths. However, conjugate models tend to be simple and so are not applicable in many situations.  

## Conjugate analysis example

Suppose $\boldsymbol{x}$ is a vector of $n$ independent observations, and we wish to model each observation as coming from a normal distribution. Further, assume the data's population mean, $\mu$, is known but the model precision, $\tau$, must be inferred. For $i=1,\dots,n$, the model is 
$$
    x_i|\mu, \tau \sim \text{N}\left(\mu, \tau^{-1}\right).
$$
The likelihood function is the product of the model densities evaluated at each observation. The model here is a normal density, and so the likelihood is the product of $n$ normal densities, with mean $\mu$ and precision $\tau$, evaluated at $x_i$ for $i=1,2,\dots,n$. Continuing the notation from before, that is
$$
    L(\tau|\boldsymbol{x}) = \prod_{i=1}^n \sqrt{\frac{\tau}{2\pi}} \exp \left[-\frac{\tau}{2} (x_i - \mu)^2\right].
$$

After consulting the literature and experts in the field, it is agreed that prior beliefs for the precision, $\tau$, are summarised by a gamma distribution with shape and rate $a$ and $b$,
$$
    \tau \sim \text{Ga}\left(a, b\right).
$$
The prior density function is, therefore, the gamma density function with shape and rate: $a$ and $b$, 
$$
    p(\tau) = \frac{b^a}{\Gamma(a)} \tau^{a-1}\exp\left(-b\tau\right).
$$

The posterior distribution $p(\tau | \boldsymbol{x})$ can then be found up to proportionality by multiplying the prior density and likelihood functions.
$$
\begin{aligned}
    p(\tau | \boldsymbol{x}) &\propto \frac{b^a}{\Gamma(a)} \tau^{a-1}\exp\left(-b\tau\right) \times \prod_{i=1}^n \sqrt{\frac{\tau}{2\pi}} \exp \left[-\frac{1}{2\tau} (x_i - \mu)^2\right] \\ \\
    &\propto \tau^{a + n/2 -1} \exp\left\{ -\left[b + \frac{1}{2}\sum_{i=1}^n(x_i - \mu)^2\right]\tau\right\}
\end{aligned}
$$

The form of $p(\tau | \boldsymbol{x})$, up to proportionality, has the same form as a gamma distribution and, therefore, $\tau$ must follow a gamma distribution *a posteriori*. Hence, the posterior belief for $\tau$ is summarised by a gamma distribution with shape $a + n/2$ and rate $\left[b + \frac{1}{2}\sum_{i=1}^n(x_i - \mu)^2\right]$, and the posterior distribution can be written as
$$
    \tau|\boldsymbol{x} \sim \text{Ga} \left(a + n/2 -1, b + \frac{1}{2}\sum_{i=1}^n(x_i - \mu)^2 \right).
$$
In this example, the posterior distribution can be found analytically and is a known distribution, and, therefore, the model is conjugate. 

# Non-conjugate analysis

Unfortunately, analysis is not always conjugate and the posterior distribution cannot be found in closed form. When this is the case it is still possible to conduct Bayesian analysis by repeatedly sampling from the posterior distribution. The posterior sample can be inspected to insight into posterior expectations and variances. How the sample is generated is discussed [here](https://jordanbchilds.github.io/posts/mcmc_introduction/).
