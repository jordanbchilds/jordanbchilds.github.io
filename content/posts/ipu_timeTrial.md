---
bibliography:
- bib.bib
title: "Hardware Accelerated Stochastic Simulation Using an Intelligence Processing Unit"
date: 2023-02-24T15:00:00Z
author: "Jordan Childs"

mathjax: true

tags: ["IPU", "Stochastic Modelling", "Gillespie Algorithm"]
---

# Introduction

Graphcore, a company based in Bristol, UK, have developed a new kind of
processor with the aim of being able to massively parallelise
computation. The chip is called an Intelligence Processing Unit (IPU)
and is now on its second generation, the Colossus MK2 IPU processor -
the GC200. Originally designed to dramatically increase performance in
machine learning applications, the IPUs architecture can also be used to
provide big performance increases in many other areas. Here, we will
show that the IPUs ability for massive parallelisation can be used to
decrease the computational time for repeated simulation of dynamic,
stochastic models. This family of models are used in a wide variety of
applications including biology, where they can be used to perform *in
silico* experiments, and finance, where they can be used to forecast
asset prices.

Simulating from such stochastic kinetic models can be achieved
relatively easily, An introduction to stochastic chemical kinetics and
simulations methods can be found [here](https://pubmed.ncbi.nlm.nih.gov/23715985/). A harder task
is to learn the parameters values governing the model. The stochasticity
of the models mean they are also intractable and so traditional
statistical inference cannot be take place. Instead, we must use
likelihood-free inference schemes, unfortunately these are usually
computationally expensive requiring hundreds of thousands or millions of
simulations to make meaningful inference on the model parameters.
Simulations themselves are often hard to parallelise as the state of the
system at time $t+\Delta t$ is dependent of the state at time $t$.
However, we can take advantage of modern computer chips which have more
than one core, capable of executing multiple tasks independently. We
will show that the IPUs ability for massive parallelisation can be used
to decrease the computational time for the kind of large-scale
simulations often required for likelihood-free inference.

One GC200 contains 1,472 IPU-Tiles, one IPU-Tile contains one IPU-Core
which is capable of executing six independent threads in parallel, that
is one GC200 can execute 8,832 tasks in parallel. For example, if we
were to run a single simulation, which is known to take 30 seconds on an
IPU-Core, it would still take 30 seconds to run 8,832 simulations by
utilising every thread on the processor.

# Method

To show that the IPU can offer considerable time savings we will compare
the time taken to simulate a discrete state space and continuous time
stochastic kinetic model by the [Gillespie algorithm](https://pubs.acs.org/doi/pdf/10.1021/j100540a008).
The individual IPU-Cores are not very powerful when compared to a CPU
but by using all available IPU-Tiles we can utilise the full power of
the processor. As such, it would not be appropriate to only consider the
time taken for the GC200 processor to complete one simulation, as this
effectively using 1/8,832 of the the processors power. Instead we will
consider 'batches' of simulations. We can compare the time taken for the
GC200 to simulate 8,832 simulations with the time it takes a CPU to
execute the same number of simulations. The CPU we are going to use is
the Intel Xeon Gold 6246R Processor. This has an all-core-turbo
frequency of 4.0GHz and eight cores. It is of the same family, but a
newer model, of CPU used as a base line comparison by [Kulkarni *et al.*](https://arxiv.org/abs/2012.14332). Kulkarni *et al.* showed that using the IPU can reduce computation time for approximate Bayesian Inference, for an SIR model, they however used PyTorch to implement there model. Here we are wish to show that a significant speed up is still available when writing the code yourself, allowing for a wider range of more bespoke inference schemes and models. An introduction to executing C++ scripts on the IPU can be found [here][https://jordanbchilds.github.io/ipu_howto]. A single GC200 chip, however, cannot be purchased and they can only be
purchased within an IPU-POD, which are made with a range of four to 256
processors. For our timings we will use an IPU-POD4, which has four
GC200s. To show the scalability of the IPU we will also show the time
taken for the IPU-POD4 to execute 35,328 simulations, utilizing every
thread available within the IPU-POD4.

Lastly, we will compare the processors computing power relative to the
cost associated with them. There are several companies through which
time can be rented on an IPU-POD, here we have used Gcore and will use
this price to compare. We have also used Gcore to rent time on the Intel
Xeon as Gcore allows the rental of baremetal processors, which will give
more stable performance for comparison sake's.

# The Model

The stochastic kinetic model we will use to compare IPUs and CPUs will
simulate the clonal expansion of mitochondrial DNA in post-mitotic
cells. The biological reasoning and meaning behind the model is not of
particular relevance here, however the interested reader is referred to [this article](https://royalsocietypublishing.org/doi/10.1098/rsob.200061) for further details. Importantly, the system has two
species and four reactions. Either species can replicate, one molecule
becomes to two daughter molecules, or degrade, a molecule dies. The
replication and degradation reaction rates of the both species are
assumed to be equal i.e. there is no replicative or degradative
advantage. We also impose a population control mechanism. Such that the
total population, $P(t)$, at a time $t$ cannot vary far from a target
population, $P_0$. Mechanisms which stop the total population exploding
or diminishing, are often implemented as this behaviour is seen in many
observed datasets. The mechanism implemented here increases or decreases
the probability of the next reaction being a replication if the current
population below or above $P_0$ respectively. This is done equally to
both species by updating the reaction rates of the replication
reactions. Let $X_1$ and $X_2$ denote the two species, the
pseudo-chemical equations for the model are described in Equation 1.

| $$ \begin{split} X_1 &\rightarrow 2X_1 \\\\ X_1 &\rightarrow \phi \\\\ X_2 &\rightarrow 2X_2 \\\\ X_2 &\rightarrow \phi \end{split} $$ |
| :--: |
| Equation 1: Psuedo chemical equations describing the four reactions which can occur in the model |

We chose constant reaction rates for the two degradation reactions.
While the replication reaction rates change due the population control
mechanism, the initial reaction rates, and if $P(t) = P_0$, are equal to
the degradation reaction rates. The simulation is continued until $t$
reaches some desired time $T_{max}$.

The Gillespie algorithm is an exact simulation algorithm of a discrete
stochastic biochemical network, this means for the same model, with the
same parameters and initial conditions we can get different output. Each
unique (and random) path through the simulation will take a different
amount of computing time to complete. Therefore we will be comparing
distributions of time simulation times from multiple executions of the
algorithm.

# Results

As this is a stochastic process the time taken for the simulation to
reach our chosen $T_{max}$ is itself a random variable. We therefor run
the simulation a large number of times to gain an understanding of the
distribution of time taken. We omit the results of running a single
simulation but to demonstrate the relative power of a single IPU-Core
with the Intel Xeon Processor, the mean time of 1,000 simulation can be
seen in Table 1.



|               | Intel Xeon    | IPU-POD4   |
|:------------  | :-----------: | :--------: |
| Time (ms)     | 32.149        | 2822.955   |



Table 1: Mean computing time to execute a single Gillespie simulation.

It is clear that a one IPU-Core is much less powerful than a one core in
the Intel Xeon. However, the massive number of IPU-Cores within a single
processor combined with their ability to run up to six independent tasks
without loss of performance results in a powerful processor as seen in
Figures Mango and Raspberry. Which shows the distribution of 1,000
executions of simulation batches of size 8,832, for the system described
in Equation 1.

| ![cpu_batchSim](/ipu_timeTrial/cpu_batchSim.png 'CPU Batch Simulation') | ![ipu_batchSim](/ipu_timeTrial/ipu_batchSim.png 'CPU Batch Simulation') |
| :--: | :--: |
| Figure Mango: Simulation times for one batch (8,832 simulations) using the Intel Xeon | Figure Raspberry: Simulation times for one batch using the GC200 |


These simulations were done in parallel. Each IPU-Core executed six
simulations and so the GC200 executed all 8,832 simulations
independently. The time to complete these simulations is essentially the
time taken for the longest of the 8,832 simulations to finish as the
GC200 thread allocation and data streaming adds negligible time.
The parallelisation for Intel Xeon processor was achieved by a Bash
script executing $N_{core}$ (the number of available cores) C++ scripts
each sequentially executing $M$ Gillespie simulations, such that
$M\times N_{core}=8,832$. It was done this way as using the C++ standard
library `std::thread` and the parallelisation software OpenMP added
considerable computational overhead due to thread creation and task
allocation. Resulting in the parallel simulations taking much longer
than if they were executed on a single thread. Although not ideal the
essentially static scheduling used was not a massive disadvantage. The
population control mechanism ensures the simulations stay approximately
the size and so the time to execute does not have a large variance. This
confirmed by trials with `std::thread` and OpenMP, where both static and
dynamic scheduling schemes were used and the difference between them was
negligible.

We see in Figure 1 that when compared to a single simulation
the time taken for the Intel Xeon to execute the batches has increased
considerably. This is as we would expect when comparing single runs to a much
larger number. However the GC200 has only increased slightly, from
$\approx 2800$ms to $\approx2930$ms. This is due to the massive number
of independent processes that can be executed on GC200. This gives a
speed-up factor of approximately 14.5 when comparing the two chips, as
seen in Figure 3.

| ![cpu_batchSim](/ipu_timeTrial/cpuIPU_ratio.png 'CPU Batch Simulation') |
| :--: |
| Figure 3: The speed-up factor when comparing the time taken execute one simulation batch on the Intel Xeon vs. the GC200. |

# Discussion

We have demonstrated that when a large number of independent stochastic
processes the GC200 can offer a considerable time saving, when compared
to the Intel Xeon. This, however, does not take into account the cost of
the processors. Unfortunately, it is not possible to buy/rent a single
GC200 as they come in PODs with sizes ranging from four to 256
processors. Due to the design of the GC200 and the PODs it is extremely
easy to scale the simulations to use more processors, adding negligible
computational overhead. When executing 35,328 Gillespie simulations on a
IPU-POD4 the mean simulation time was 2937.766 ms, this is approximately
the same as when we were simulating 8,832 simulations on a single
GC200.

To compare the relative computing power per cost we will assume there is
seamless parallelisation between one CPU and four and that if we had
four times the processors we could execute four times the simulations in
the same amount of time. For easy comparison we will take the median of
the distributions as the expected simulation time and divide by the cost
of renting the appropriate processors from Gcore for that amount of
time. The results can be seen in Table 2.

|               | Intel Xeon    | IPU-POD4   |
|:------------  | :-----------: | :--------: |
| Cost per hour | 3€2.5032 \*   | €6.12      |
| median sim. time | 42.5455s \*\*   | 2.9428s.   |
|expected sim. cost | €0.024.   | €0.005.    |

Table 2: The expected cost to run one batch or 35,328 simulations using an
  Intel Xeon Processor and an IPU-POD4.

(\*) The cost per hour for a single Intel Xeon Gold 6246R Processor is
€0.5133.
(\*\*) This is the expected simulation time assuming that increasing
using more than one Intel Xeon processor incurs not computational
overhead.


Despite the assumptions made about the seamless scalability of the CPUs,
the IPUs still offer a significant reduction in cost for one simulation
batch by a factor of almost five. Reducing the computation time of large
batches of simulations can result in massive time savings when using
likelihood-free simulation based inference methods, such as approximate
Bayesian computation. Simulation often being the largest computational
cost of the notoriously expensive algorithm, which can require millions
of simulations of a system to be able to infer the model parameters.
However it is usually not required to do batch simulations of such large
numbers during inference, the IPU allows us to run each simulation with
different parameters with little effort. This can also be done on a CPU
however it may require some thought and consideration due to the bash
script method of parallelisation.

Despite the clear time, advantage of the IPUs there is a steep learning
curve to be able to use them even if the user is already familiar with
C++. From personal experience taking over week to go from nothing to
executing a Gillespie simulation. The CPU parallelisation method used
here was however not without its faults. The Bash script method of
parallelisation would require any desired output to be saved to a text
file and then aggregated when all simulations are done. This would also
add development time and a small amount of computational time as well.
The timings presented should not be taken as the end of the story. A
more experienced programmer would be able to implement more efficient
code which could better utilise either hardware. But should hopefully
give the reader an indication of the power of the GC200 and the IPU-PODs
they come in.
