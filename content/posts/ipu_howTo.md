---
title: "A Beginners Guide to Executing C++ Scripts on an IPU"
date: 2023-02-24T14:30:00Z

mathjax: true
tags: ["IPU", "Guide", "C++"]
---

# Introduction

In this article we will discuss how execute a large number of
independent programs in parallel, using a new computer chip capable of
executing thousands of independent calculations at once. This may be
particularly useful in areas of stochastic modelling and Monte Carlo
simulation, where repeated and independent simulations are required to
learn about uncertainty in model predictions. The chip in question is
called the Intelligence Processing Unit (IPU) and is developed by
Graphcore, a UK-based company headquartered in Bristol. The IPU is now
(as of February 2023) is on its second generation, the Colossus MK2 IPU
processor, also known as the GC200.

Although the IPU has many possible applications where its scalability
can reduce computational expense we will use the example of a population
dynamics simulation model: a discrete time, discrete state space
stochastic process - a pure death process. This is a simple model but
one that highlights some key aspects of running programs on the IPU and
is an area of modelling that can benefit greatly from the IPU's design.
This model assumes that within a given time period there is a
probability, $p$, that one individual in the population dies, otherwise
the population remains unchanged. No individuals can be born or migrate
into the population.

Before moving on, it is import to note that in this article some
familiarity with C++ is assumed, as this is the only language which IPU
applications can be written in. However, it is by no means necessary to
be an expert in C++ and the most complex topic used here is pointer
arithmetic. The article is focused on being able to execute programs on
the IPU, as such an understanding of the death process is not necessary
but may be advantageous to have a fuller understanding of the code
presented however a brief description of model is given.

# A Brief Introduction to the IPU

The GC200 is made of 1,472 independent IPU-Tiles each containing one
IPU-Core , possessing 900MB of in-processor memory and capable of
executing six independent tasks at once. That is, one GC200 chip can run
8,832 tasks independently without loss of performance. A single chip can
be very powerful but the chips were developed with a focus on
scalability and offer near seamless performance when used together. This
is thanks to the ultra-low latency IPU-Fabric connecting all processors,
which is responsible for the transfer of data between IPU-Cores .
The processors are available to rent or buy within PODs, starting with
four GC200 and increase to have up to 256. The PODs are also designed
for scalability and the same ultra-low latency IPU-Fabric which connects
chips within the POD is also used to connect to other PODs. One POD
consists of both the GC200 processors and one CPU, referred to as the
host. The host defines a sequence of instructions to be executed on the
GC200 processors, the instructions are written using the Poplar C++
library which was explicitly developed by Graphcore for IPU
applications.

When executing a task on an IPU it is best to think of this as a
directed graph. Where variables are connected to vertices via edges, the
vertices being compute tasks to be executed in parallel and variables
being the data used within those tasks. Each vertex is executed on a
single thread on an IPU-Core . To gain the best performance from an IPU
we wish to have as many compute tasks to execute at once in parallel. A
set of compute tasks is referred to as a compute set, both throughout
this article and within the poplar programming model. When a compute set
is executed the program can not move onto the next task until all tasks
in the compute set are completed. As such it best to have tasks of a
similar duration otherwise the IPU-Tiles are stood idle awaiting a
single task to finish.

The IPUs already provide a lot of functionality within artificial
intelligence and machine learning, and can be used by popular ML
packages such as TensorFlow and PyTorch. These, however, will not
discussed here and instead we will focus on writing and executing
bespoke C++ programs for the IPU. Running applications on the IPU
requires two parts, both need to be written in C++. The first part is
the codelet, this is the compute task associated with a vertex. We will
show how to execute one codelet repeatedly, however it is not a
difficult extension to use different codelets for different vertices.
The second part is the `main.cpp` script which is read and executed by
the host, this is where the graph is defined which maps variables to
vertices. Graph creation and execution is done using the Poplar library
and is done entirely by the host, while the codelet is written in a
slightly restricted version of C++.

# The Codelet

## C++ Restrictions on the codelet

An IPU-Core can compile and read code written in C++, however it does
not provide the full functionality of the language. The major difference
is that the IPU-Core has no Heap, this means that dynamic memory
allocation is not possible. Although this may seem restricting to
experienced C++ programmers, if you are new to C++ it makes the learning
curve a less steep one.

There are some other important limitations that need to considered when
writing C++ code for an IPU-Core . The libraries available are a small
subset of those available on a CPU. However, Graphcore have created
several libraries which can be used as well. These provide functionality
such as random number generation and linear algebra as well as much
more, full documentation for using libraries can be found in the [Poplar
documentation](https://docs.graphcore.ai/projects/poplar-api/en/latest/poplibs_api.html ).
It is often the case that problems can be overcome by 'creative'
programming. For example, there are a limited number of distributions
which can be sampled from on an IPU-Core (normal and uniform). The host,
however, has full access to all C++ libraries and so samples can be made
from any desired distribution using a C++ library and then passed to the
IPU-Core.

Another important consideration when writing the codelet is that the
compiler must know the size of any array defined at compile time. This
may require writing functions to be less general by explicitly stating a
dimension that an array is known to be, rather than inferring this from
another variable. If the size of an array is not known in advance,
defining the array to be 'large enough' such that it is practically
impossible that it would be required to be any larger and dealing with
this in a post-hoc manner is an easy solution, albeit expensive in terms
of memory requirement.

## `codelet.cpp`

The codelet is the script that is executed on a single thread on an
IPU-Core . It needs access to all relevant functions, objects,
structures etc used within its execution. Variables can be passed to the vertex
from the host so that each thread does not have to run with the same parameters,
this will be discussed in the `main.cpp` chapter. For now, we will assume we
wish to run every thread with the same parameters and so they can be
defined within the codelet.

Here, we are going to use the example of a simple death process. This is
only to illustrate the methods required for executing repeated
computations on an IPU. The model simulates the scenario where for a
time step of size $\Delta t$ there is a probability, $p$, that one

member of a population will die otherwise the population remains
unchanged. To forward simulate this model we first simulate whether a
death occurs within $[0,\Delta)$ and then update the population
accordingly. Then simulate whether a death occurs within
$[d, 2\Delta)$ and update the population accordingly. This is
repeated until the population reached zero and thus no more reactions
can occur.

For the IPU to be able to run the codelet we must define a new class
which derived from the Poplar class `poplar::Vertex`, let's call the
class `myVertex`. Inside the definition of `myVertex` is where all
relevant objects related to the simulation are defined and nothing
should be defined outside of this. Importantly, `myVertex` must contain
a function called `compute` which returns a `true` value. This is akin
to the `main` function in standard C++ programming. A codelet example
can be seen in Listing 1, where a simple death process is simulated
from with an initial population of 100 and a probability of death within
$[t, t+\Delta t)$  is 0.1.

```{c++ title-one}
#include <ipu_builtins.h>

class myVertex : public poplar::Vertex
{
    public:

    float prob_death = 0.05;
    unsigned int initial_pop = 100;

    float rand_unif01()
    {   /*
        Returns a random number on the interval [0,1]
        of type float.
        */
        return float(__builtin_ipu_urand32()) / float(4294967295) ;
    }

    void sim_death_process(prob_death, init_pop, output_ptr)
    {   /*
        This function simulates a discrete time death process.

        output_ptr: is a pointer to the output array.
        prob_death: defines the probability of a death in the chosen time step.
        init_pop: is the initial population of the system.
        */

        unsigned int current_pop = init_pop ;
        *(output_ptr) = current_pop ;
        unsigned int t = 1;

        while( current_pop > 0 ) {
            if( rand_unif01() < prob_death )
            {
                current_pop-- ;
            }
            *(output_ptr+t) = current_pop ;
            t++ ;
        }
    }

    bool compute()
    {
        const unsigned int MAX_LENGTH = 5000 ; // large number
        unsigned int output[MAX_LENGTH] = {0} ;
        unsigned int* output_ptr = &output[0];

        sim_death_process( prob_death, initial_pop, output_ptr, MAX_LENGTH );

        return true;
    }
};
```
Listing 1: An codelet.cpp script.

The first thing defined within the `myVertex` class are the parameters
of the model; the probability of death within the time step and the
initial population. We then define one function to generate a random
number from a Uniform(0,1) distribution. This also shows an example of
using a function from a Poplar library, the inclusion of the library can
be seen at the top of the script using the `#include` macro. The
`popLibs` libraries do provide functionality to simulate from a uniform
distribution but here we define the uniform in this way as another
example of function definition. The function, `rand_unif01`, is
defined to draw a random `unsigned int` and standardise this by dividing
by the maximum value of an unsigned integer, outputting a `float` in the
range $[0,1]$. After this we use the function to simulate the model of
interest, `sim_death_process`.

Lastly we have the `compute` function which defines the output array and
the runs the simulation. Also defined here is the `MAX_LENGTH` variable,
as previously mentioned, the IPU must know the exact amount of memory to
allocate at compile time. The `MAX_LENGTH` variable defines the size of
the output array and is defined such that we would never expect our
simulation output to be larger than `MAX_LENGTH`. Its value can be
chosen by running a few simulations off the IPU and seeing what can be
expected from the output.

# main.cpp

Unlike the codelet, since the main script is executed on the host, which
is a CPU, it has the full functionality of C++, including dynamic memory
allocation and all libraries. It is in this script we build and execute
the graph: we define and map variables to vertices, attach codelets to
those vertices and retrieve output from the IPU.

## Creating a Graph

Firstly, a graph object must be created and attached to an IPU. Listing
2 shows some boiler plate code to create a
device and attach to `numberOfProcs` GC200 processor(s) to this device.
A device is poplar terminology for the IPU system the graph will be
executed on. First a device manager is created, this searches for IPU
devices which can be attached to. The final two lines attach the
`main.cpp` script to the device created and create an empty graph on
that device. An empty graph is one without any variables or vertices,
these will have to added. The code snippet presented here also has
several print statements which inform the user if there has been a
problem when trying to attach to an IPU device.

``` {#lst:attachIPU label="lst:attachIPU" caption="Code snippet to construct an empty and attach to an IPU."}
// Create the DeviceManager which is used to discover devices (IPUs)
auto manager = DeviceManager::createDeviceManager();
// Attempt to attach to a numberOfProcs IPU(s):
auto devices = manager.getDevices(poplar::TargetType::IPU, numberOfProcs);
std::cout << "Trying to attach to IPU\n";
auto it = std::find_if(devices.begin(), devices.end(), [](Device &device) {
    return device.attach();
});
if (it == devices.end()) {
    std::cerr << "Error attaching to device\n";
    return -1;
}
auto device = std::move(*it);

std::cout << "Attached to IPU " << device.getId() << std::endl;
// Target device so that any graph will be associated with device
Target target = device.getTarget();
// Create the Graph object
Graph graph(target);
```
Listing 2: Creating an IPU device in Poplar and defining an empty graph object attached to that device.

Before being able to execute or pass parameters to codelets, we need to
add the codelet to the graph using the `poplar::Graph::addCodelets`
function. Adding a codelet allows it to then be associated with a vertex
and later executed. This can be seen in Listing 3. Also seen in this snippet is the definition
of a `poplar::program::Sequence` object. This is a series of control
programs to be executed in order to do things such as map variables to
vertices and execute compute sets. These can be defined to be as complex
as needed for the task, here we will show an example of a simple one, executing
only one compute set. The last line of Listing 3 creates a compute set object
and labels this `"computeSet"`. In the next section we will see how to add vertices to
this compute set and to the graph itself.

``` {#lst:addCodelet label="lst:addCodelet" caption="Adding a codelet to a graph object as well as creating a \\texttt{Sequence} object."}
// Add codelets to the graph
graph.addCodelets("myCodelet.cpp");

// Create a control program that is a sequence of steps
poplar::program::Sequence prog;

// Define the compute set to add vertices to later
poplar::ComputeSet computeSet = graph.addComputeSet("computeSet");
```
Listing 3: Adding a codelet script to the graph, creating a control sequence object and defining an empty compute set.

## Mapping Vertices to Tiles

To be able to execute a compute set it is necessary to first map
vertices to specific tiles on which they will be executed. This is done
explicitly by connecting a vertex on the graph to a tile index. A single
GC200 processor has 1,472 tiles each with a unique index, in the range
\[0, 1,471\]. It should not matter which tile a vertex is mapped to as
the they are identical and can share information with ease. Suppose we
wish to map a vertex, `myVertex`, to all tiles on a GC200. Adding the
vertex to a compute set informs poplar what vertices can be executed in
parallel. The function also creates a vertex reference which is used to
map the vertex to a tile. Using the vertex reference, it is then mapped
to a specific tile using the `poplar::Graph::setTileMapping` function,
with the reference as its first parameter and the tile index as its
second. By doing this repeatedly a vertex can be mapped to every tile on
the chip, as as seen in Listing 4.

``` {#lst:mapTiles label="lst:mapTiles" caption="How to map a vertex to all tiles on a GC200 processor."}
const unsigned int numberOfTiles = 1472; // maximum is 1472

for( std::size_t i=0; i<numberOfTiles; ++i){
    VertexRef vtx = graph.addVertex(computeSet, "myVertex");
    graph.setTileMapping(vtx, i);
}
```
Listing 4: Mapping a vertex to evey tile on a GC200.

However, mapping a single vertex to every tile on a GC200 would only be
using one sixth of the processor computing power, as each tile can
execute six tasks independently. It would therefore be good to be able
to map multiple tasks to a single tile, so that they can be executed on
different threads on that tile. This can be easily done by mapping six
(the maximum number of threads) vertices to every tile, the IPU-Core
will automatically split the vertices between threads. The first two
lines of the `for loop` ensure that the tile index does not exceed the
number maximum, 1,472, and that each tile is mapped to `numberOfThreads`
times. The snippet maps a single vertex to every thread on the chip, it
is possible to execute more vertices than there are threads by mapping
more than six vertices to a single tile.

``` {#lst:mapThreads label="lst:mapThreads" caption="How to map a vertex to all threads on an GC200 processor."}
const unsigned int numberOfTiles = 1472; // maximum is 1472
const unsigend int numberOfThreads = 6; // maximum is 6

const unsigend int totalThreads = numberOfThreads * numberOfTiles ;

for( std::size_t i=0; i<totalThreads; ++i){
    int roundCount = i % int( numberOfTiles*threadsPerTile );
    int tileInt = std::floor( float(roundCount) / float(threadsPerTile) );

    VertexRef vtx = graph.addVertex(computeSet, "myVertex");
    graph.setTileMapping(vtx, tileInt);
}
```
Listing 5: Mapping one vertex to every thread on a every tile on one GC200.

In practise there will be more than one GC200 processor available, as
they come in IPU-PODs. We therefor wish to be able to map vertices to
all available processors. When using `numberOfProcs` processors each
tile still has a unique index with a range
$[0, 1471\times$`numberOfProcs`$]$. Listing 6 shows how to define the `tileInt` parameter
to get the desired output. Six vertices are still being mapped to every
tile but now all IPU-Tiles within an IPU-POD4 system are being used.

``` {#lst:mapThreads label="lst:mapThreads" caption="How to map a vertex to all available threads in an IPU-POD4."}
const unsigned int numberOfTiles = 1472; // maximum is 1472
const unsigend int numberOfThreads = 6; // maximum is 6
const unsigend int numberOfProcs = 4; // maximum depends on POD size

const unsigend int totalThreads = numberOfThreads * numberOfTiles * numberOfProcs ;

for( std::size_t i=0; i<totalThreads; ++i){
    int roundCount = i % int( numberOfTiles * threadsPerTile * numberOfProcs);
    int tileInt = std::floor( float(roundCount) / float(threadsPerTile) );

    VertexRef vtx = graph.addVertex(computeSet, "myVertex");
    graph.setTileMapping(vtx, tileInt);
}
```
Listing 6: Mapping one vertex to every thread on every tile across multiple GC200 chips.

## Passing Parameters

Being able to pass parameters to vertices would allow calculations to be
executed for different data or parameter values. Or, if one task is
dependent on the previous, it also allows calculation of variables in
one compute set and then be passed forward to the next compute set.
Passing parameters to a vertex is very similar to mapping a vertex to a
tile, except here data is being mapped to a tile and then connected to a
variable within a codelet.

When passing a variable to a codelet, the variable type inside the
vertex must reflect the fact it is going to mapped to the codelet. This
is done simply by declaring an object as type `poplar::Input<T>`.
Listing 7 shows an example of two variables to be
mapped, one an integer scalar and the other a vector of floating
points.

``` {#lst:inputType label="lst:inputType" caption="Declaring input type variables inside a codelet."}
class myVertex : public poplar::Vertex
{
    public:
    poplar::Input<int> myScalar ;
    poplar::Input<poplar::Vector<float>> myVector ;

    // ... the rest of the vertex object
}
```
Listing 7: Declaring an object is to be passed to a codelet.

### Scalars

A scalar is a single value (a single number). Passing scalars is
relatively easy however the number of parameters usually increases with
model complexity and so this is only practical for simple models. For
large models it is often more convenient to pass a vector of parameters,
which may also be preferred if the variable is naturally described as a
vector. The method shown here follows most of the tutorials on the
Graphcore website, these can be found
[here](https://github.com/graphcore/tutorials) for further information.
We present a general case where a different value can be passed to each
tile. First, we define a one dimensional array to store the values being
passed, this is done in standard C++. Once this is created we add a
`poplar::Tensor` to the graph, only a `poplar::Tensor` type can be used
when adding variables to a graph. The `poplar::Tensor` object is very
similar to the C++ `std::vector` object. The documentation can be found
[here](https://docs.graphcore.ai/projects/poplar-api/en/latest/poplar/graph/Tensor.html#_CPPv4NO6poplar6TensorixENSt6size_tE),
for all details on functions and operators available.

Listing 8 shows the creation of C++ array,
`myScalars`, which is then added to the graph as a `poplar::Tensor`
using the `poplar::Graph::addConstant` function. The function also
informs poplar of the type of the elements within the tensor as well as
its size and shape. Here, the tensor is declared with elements of the
poplar type `FLOAT` and it is a single dimension tensor of length
`totalThreads`.

``` {#lst:scalarMap def label="lst:scalarMap def" caption="Defining a one dimensional tensor to map individual elements to the codelets."}
float myScalars[totalThreads];
for( std::size_t i=0; i<totalThreads; ++i ){
    myScalars[i] = i*0.001 ; // or something useful
}

poplar::Tensor myScalars_tensor = graph.addConstant<float>(FLOAT, {totalThreads}, myScalars);
```
Listing 8: Creating a `Tensor` to be added to the graph as a constant.

Now, we need to define which element of the tensor is being mapped to
which tile. We do this using the `poplar::Grpah::setTileMapping`
function. The second element of the function is the tile index we wish
to map to. The last thing to do is inform poplar what variable inside
the codelet is being mapped to. The variable on the graph can 'connect'
to a parameter in the codelet via the `poplar::Graph::connect` function.
These steps can all seen in Listing 9, where the calculation of `tileInt` has been
omitted. Note the mapping of the vertices is included in this snippet as
it is necessary that the vertex is mapped to be able to connect a
variable to a parameter within it.

``` {#lst:scalarMap label="lst:scalarMap" caption="Creating a map of scalar to codelet."}
for( std::size_t i=0; i<totalThreads; ++i){
    // ... Calculate tileInt
    graph.setTileMapping(myScalars_tensor[i], tileInt);

    graph.addVertex(computeSet, "myVertex");
    graph.setTileMapping(vtx, tileInt);

    graph.connect(vtx["myScalar"], myScalars_tensor[i]);
}
```
Listing 9: Mapping a scalar to every thread to be used in a codelet.

This method is simple to implement but if the codelet requires vector or
matrix input then each element would have to be mapped separately and
the vector/matrix reconstructed within the codelet. This quickly becomes
tedious to write and not easy to read.

### Vectors (1-Dimensional Arrays)

Here, we discuss how to stream a vector to tiles on the IPU, this
creates and stores the vector on the host before it is passed to each
tile via a data stream. This is not the only way in which to pass a
vector to a vertex but it was useful and it also illustrates another
functionality of an IPU-POD. Another way in which tensors can be mapped
to a tile is by first storing them on the tiles themselves rather than
on the host, as part of the graph. A tensor can be stored on a single
tile or across many, if it is very large. This method will not be
covered here, although the method is similar to the ones that have been
covered. This short [video](https://www.youtube.com/watch?v=k_jR7DfN67c) tutorial
gives a brief example of how this, and graph creation and execution is done.

Here we will show how to stream a single vector to every vertex although
this method can be extended to pass different vectors to each vertex.
Similarly to before, we start by defining a vector to be streamed and
adding this to graph. However, when streaming we use the
`poplar::Graph::addVariable` function not `addConstant`. The three
parameters of the function used here, are the type of elements which
fill the `poplar::Tensor`, the size and shape of the tensor and finally
the label. Although we can define a multidimensional tensor and add this
to the graph, only a vector can be streamed to an IPU-Tile . This may
seem restrictive but it is not a difficult problem to overcome, if
desired a multidimensional array can be flattened into a single
dimensional tensor and then re-shaped into the multidimensional array on
the IPU-Tile. C++ stores multidimensional arrays in contiguous blocks,
as though they were flattened anyway.

``` {#lst:addVariable .c++ caption="Adding a variable the graph and creating the compute set." label="lst:addVariable" language="C++"}
const unsigned int myVector_size = 10;
float myVector[myVector_size];
for( std::size_t i=0; i<myVector_size; ++i )
    myVector[i] = i*0.0001; // or something useful

poplar::Tensor myVector_tensor = graph.addVariable(FLOAT, {myVector_size}, "myVector");
```
Listing 10: Defining a Tensor and added it to the graph as a variable.

After the definition, similar to before we need to create the mapping to
a specific tile and specific codelet vertex on that tile. This is done
exactly the same as before.

``` {#lst:vectorMap caption="Mapping a vector to the codelet." label="lst:vectorMap"}
// Map tensors to tiles
for( std::size_t i=0; i<totalThreads; ++i ){
    // ... Calculate tileInt
    graph.setTileMapping(myVector, tileInt);

    VertexRef vtx = graph.addVertex(computeSet, "myVertex");
    graph.setTileMapping(vtx, tileInt);

    graph.connect(vtx["myVector"], myVector_tensor);
}

auto myVector_stream = graph.addHostToDeviceFIFO("write_myVector", FLOAT, myVector_size);
```
Listing 11: Mapping a vector to a vertex through stream.

To be able to stream information from host to tile the stream object
must be created, using the `poplar::Graph::addHostToDeviceFIFO`
function. The first argument of this function is the stream label then
the type of the elements being streamed and the number of elements.
There is one last thing that must be done for the tensor to be streamed
from host to tile.

Before this is done, we first introduce the `poplar::Engine` object. The
engine represents the whole graph and execution sequence on a given
device. Its creation and eventual execution will be covered later but
for now assume we have a `poplar::Engine` object called `engine`. The
function `poplar::Engine::connectStream` is used to let the engine know
there is a stream that needs to be executed. Listing
12 shows an example of this. The first parameter is the stream label,
as defined in Listing 11 and the other two parameters are the memory
addresses of the first and last elements of the array.

``` {#lst:connectStream caption="Streaming the vector the tiles." label="lst:connectStream"}
// Attach the data stream to the engine so that the stream is executed
engine.connectStream("write_myVector", &myVector[0], &myVector[0]+myVector_size);
```
Listing 12: Adding the stream to be executed by the Poplar engine.

## Retrieving Output

Retrieving output from the codelet is very similar to the process of
streaming vectors to the codelet. First we have to inform the codelet
that we are expecting an output array to be streamed from the vertex, we
do this using the type `poplar::Output<T>`. Doing this allows us to
write the output directly into the `poplar::Output` object and we do not
have to define an output array within the `compute` function, like was
done in Listing 13.

``` {#lst:outputType label="lst:outputType" caption="Defining an output type in the codelet."}
class myVertex : public poplar::Vertex
{
    public:
    poplar::Output<vector<float>> out;
    // ...
}
```
Listing 13: Defining an object to be outputed from a vertex.

Now we must define an output array and map it to the tiles and vertex,
similarly to when streaming a vector to the vertex. Here, however, there
is a convenient shortcut to be able to stream output from tiles to host.
The function `poplar::Graph::createHostRead` creates and connects the
streaming object for us. Listing 14 shows how to implement this. The variable
`output_size` is the known size of the output vector, for the death
process example this would be `MAX_LENGTH`.

``` {#lst:streamOutput label="lst:streamOutput" caption="Retrieving codelet output via a stream."}
poplar::Tensor output = graph.addVariable(FLOAT, {totalThreads, output_size}, "output");

for( std::size_t i = 0; i < totalThreads; ++i ){
    // ... Calculate tileInt
    graph.setTileMapping(output[i], tileInt);

    VertexRef vtx = graph.addVertex(computeSet, "myVertex");
    graph.setTileMapping(vtx, tileInt);

    graph.connect(vtx["out"], output[i]);
 }
 graph.createHostRead("output-read", output);
```
Listing 14: Mapping an output vector from a tensor on the host to a vertex and creating the a stream object using `createHostRead`.

A small thing to note is that the output tensor is multidimensional but
only a single dimension is streamed from the codelet. As mentioned, we
cannot stream multidimensional tensors directly to or from the codelet,
but here is an example of how we can stream sections (slices) of a
tensor from the vertex as long as the slice itself only has one
dimension. This same method can be used when passing variables to a
vertex.

Once the graph has been executed (this will be discussed in the next
section) it is likely we wish to read the output of the of the
computations. To do this there is another helpful function,
`popolar::Engine::readTensor`, with the first parameter being the name
of the streaming object, defined in the `createHostRead` function and
the following two parameters being the first and last memory address of
where to save the output on the host. Listing 15 gives an example of this. The output does
not have to be read into a standard C++ array, as we are reading the
output to the host we have access to all C++ libraries so the output can
be read into other formats if desired e.g. `std::vector`.

``` {#lst:readOutput caption="How to read the output." label="lst:readOutput"}
float output_array[totalThreads * output_size] ;
engine.readTensor("output-read", &output_array[0] &output_array[0]+output_size)
```
Listing 15: Reading a tensor from the IPU onto the host.

## Graph Execution

Lastly, we need to execute the graph and compute set. However, before
this we must let poplar know what order to execute control programs, for
this we add to `prog`, the `poplar::program::Sequence` object defined in
Listing 3. For a simple graph, with no data streams
and one compute set. This is relatively easy and is done using the
function `poplar::program::add` with the parameter being the compute set
executed with `poplar::program::Execute` function.

``` {#lst:progAdd label="lst:progAdd" caption="Adding the compute set execution to the sequence of programs to execute."}
// Adding the execution of the compute set to the sequence of programs
prog.add(Execute(computeSet));
```
Listing 17: Adding an execute compute set command to the control sequence.

Hopefully, truly the last thing to do is to execute the entire graph in
the order defined by `prog`. This requires the use of the
`poplar::Engine` object. The engine is the combination of all we've done
so far; the device, the graph, sequence of control programs and data
streams. An `poplar::Engine` object is created by passing a
`poplar::Graph` object and a `poplar::Sequence` object to the `Engine`
constructor. The engine is then loaded onto a device and ran, see
Listing 18. For illustrative purpose in this snippet,
although commented out, a stream connection has been added to the
engine.

``` {#lst:engine label="lst:engine" caption="Executing the graph in the sequence defined by \\texttt{prog}."}
// Create the engine
Engine engine(graph, prog);
engine.load(device);
// Add the host stream to the execution pipe
// engine.connectStream( ... );
engine.run();
```
Listing 18: Running the engine - executing the control sequence and any anything else added to the engine.

#  Complete Example

To bring together what has been discussed in the previous sections we
will construct the code to stream two parameters to the death process
codelet example used earlier. First we must define the parameters in the
codelet which are to be passed to the vertex. For this, we change the
types of the `prob_death` and `initial_pop` variables. The codelet has
also been updated to write output directly into the `out` vector and
comments have been removed.

``` {#lst:something clever .c++ caption="Streaming parameter example, codelet definition." label="lst:something clever" language="C++"}
class myVertex : public poplar::Vertex
{
    public:
    poplar::Input<float> prob_death;
    poplar::Input<int> initial_pop;
    poplar::Output<poplar::Vector<int>> out;

    unsigned int MAX_LENGTH = 5000;

    float runif_01(){
        return float(__builtin_ipu_urand32()) / float(4294967295) ;
    }

    void death_process(float prob_death, int init_pop){
        int current_pop = init_pop ;
        out[0] = current_pop ;
        int t = 1;
        while( current_pop > 0 && t < MAX_LENGTH ) {
            if( runif_01() < prob_death ){
                current_pop-- ;
            }
            out[t] = current_pop ;
            t++ ;
        }
    }

    bool compute () {
        death_process(prob_death, init_pop);
        return true;
    }
}
```
Listing 19: Codelet example.

Then in the `main.cpp` file we can define the variables, create the
mappings and vertices to specific tiles and connect the variables to
those in the codelet. The mappings for all vertices and variables can be
done within a single for loop, as seen in Listing 9. This example maps a single vertex
to every thread on every tile on an IPU-POD4. The graph and control
sequence are then executed before the output is streamed back to the
host and the saved in a `std::vector` called `cpu_vector`.

``` {#lst:variableMap example label="lst:variableMap example" caption="Defining a one dimensional tensor to map individual elements to the codelets."}
int main () {
    unsigned int numberOfThreads = 6;
    unsigned int numberOfTiles = 1472;
    unsigned int numberOfProcs = 4;

    unsigned int totalThreads = numberOfThreads * numberOfTiles * numberOfProcs ;

    // Create the DeviceManager which is used to discover devices (IPUs)
    auto manager = DeviceManager::createDeviceManager();
    // Attempt to attach to a numberOfCores IPU(s):
    auto devices = manager.getDevices(poplar::TargetType::IPU, numberOfCores);
    std::cout << "Trying to attach to IPU\n";
    auto it = std::find_if(devices.begin(), devices.end(), [](Device &device) {
            return device.attach();
    });
    if (it == devices.end()) {
            std::cerr << "Error attaching to device\n";
            return -1;
    }
    auto device = std::move(*it);

    std::cout << "Attached to IPU " << device.getId() << std::endl;
    // target device so that any graph will be associated with device
    Target target = device.getTarget();
    // Create the Graph object
    Graph graph(target);
    // Add codelets to the graph
    graph.addCodelets("myCodelet_simple.cpp");

    // Create a control program that is a sequence of steps
    poplar::program::Sequence prog;

    const unsigned int MAX_LENGTH = 5000;

    float prob_death[totalThreads];
    unsigned int init_pop[totalThreads];

    for( int i=0; i<totalThreads; ++i ){
        prob_death[i] = 0.1 ;
        init_pop[i] = 100 ;
    }

    poplar::Tensor prob = graph.addConstant<float>(FLOAT, {totalThreads}, prob_death) ;
    poplar::Tensor init = graph.addConstant<float>(FLOAT, {totalThreads}, init_pop) ;
    poplar::Tensor output = graph.addVariable(FLOAT, {totalThreads, MAX_LENGTH}, "output") ;

    ComputeSet computeSet = graph.addComputeSet("computeSet");

    // Map tensors to tiles
    for(int i=0; i<totalThreads; ++i){
        int roundCount = i % int( numberOfTiles * threadsPerTile * numberOfProcs);
        int tileInt = std::floor( float(roundCount) / float(threadsPerTile) );

        graph.setTileMapping(prob[i], tileInt);
        graph.setTileMapping(init[i], tileInt);
        graph.setTileMapping(output[i], tileInt);

        // Add the codelet vertex to the graph
        VertexRef vtx = graph.addVertex(computeSet, "myVertex");
        // map the vertex to every tile
        graph.setTileMapping(vtx, tileInte);

        // Connect the parameter names in the codelet to the Tensors defined here
        graph.connect(vtx["prob_death"], prob[i]);
        graph.connect(vtx["init_pop"], init[i]);
        graph.connect(vtx["out"], output[i]);
    }

    graph.connectHostRead("output-read", output);

    prog.add(Execute(computeSet));
    prog.add(PrintTensor("output", output));

    // Create the engine
    Engine engine(graph, prog);
    engine.load(device);
    engine.run();

    // Read the output tensor onto the host as a std::vector object
    std::vector<int> cpu_vector( totalThreads * MAX_LENGTH ) ;
    engine.readTensor("output-read", cpu_vector.data(), cpu_vector.data()+cpu_vector.size());

    return 0;
}
```
Listing 20: Main script example.

# Conclusion

Although the development time to be able to execute tasks on an IPU can
be quite significant, it can offer significant computational savings
where repeated independent tasks need to be executed and hopefully this
introduction will help with some of the initial problems. Graphcore have
created many tutorials, which are available on their
[GitHub](https://github.com/graphcore), although many of these are
directed at using PyTorch and TensorFlow. Whereas the pipeline here
allows the user to be able to write their own bespoke codelets to
executed en masse.

The Poplar programming model and IPU offer far more functionality than
that is presented here and this is only a small introduction to their
use. For examples of IPU uses see Graphcore
[blogs](https://www.graphcore.ai/blog) and to see for the full
documentation of the Poplar libraries see
[here](https://docs.graphcore.ai/projects/poplar-user-guide/en/latest/index.html).
