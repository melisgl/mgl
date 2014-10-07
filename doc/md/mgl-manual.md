<a name='x-28MGL-3A-40MGL-MANUAL-20MGL-PAX-3ASECTION-29'></a>

# MGL Manual

## Table of Contents

- [1 mgl ASDF System Details][e0d7]
- [2 Overview][f995]
    - [2.1 Features][8665]
    - [2.2 Dependencies][6d2c]
    - [2.3 Tests][303a]
    - [2.4 Bundled Software][b96a]
- [3 Basic Concepts][516d]
- [4 Dataset][72e9]
    - [4.1 Sampler][af7d]
        - [4.1.1 Function Sampler][2100]
- [5 Resampling][8fc3]
    - [5.1 Partitions][9f93]
    - [5.2 Cross-validation][4293]
    - [5.3 Bagging][0675]
    - [5.4 CV Bagging][ca85]
    - [5.5 Miscellaneous Operations][7540]
- [6 Gradient Based Optimization][fe97]
    - [6.1 Extension API][2730]
        - [6.1.1 Implementing Optimizers][794a]
        - [6.1.2 Implementing Gradient Sources][984f]
        - [6.1.3 Implementing Gradient Sinks][f18a]
    - [6.2 Iterative Optimizer][f805]
    - [6.3 Gradient Descent][53a7]
        - [6.3.1 Batch GD Optimizer][df57]
        - [6.3.2 Segmented GD Optimizer][25a8]
        - [6.3.3 Per-weight Optimization][d275]
    - [6.4 Conjugate Gradient][8729]
- [7 Differentiable Function][1a5d]
- [8 Backprogation Neural Networks][74a7]
- [9 Boltzmann Machines][94c7]
- [10 Gaussian Processes][026c]

###### \[in package MGL\]
<a name='x-28-22mgl-22-20ASDF-2FSYSTEM-3ASYSTEM-29'></a>

## 1 mgl ASDF System Details

- Version: 0.0.8
- Description: MGL is a machine learning library for backpropagation
  neural networks, boltzmann machines, gaussian processes and more.
- Licence: MIT, see COPYING.
- Author: Gábor Melis
- Mailto: [mega@retes.hu](mailto:mega@retes.hu)
- Homepage: [http://quotenil.com](http://quotenil.com)

<a name='x-28MGL-3A-40MGL-OVERVIEW-20MGL-PAX-3ASECTION-29'></a>

## 2 Overview

MGL is a Common Lisp machine learning library by [Gábor
Melis](http://quotenil.com) with some parts originally contributed
by Ravenpack International. It implements:

- Backpropagation networks (BPN)

    - Dropout

    - Rectified linear units

    - Maxout

    - Max-channel

- Boltzmann Machines

- Restricted Boltzmann Machines (RBM)

- Deep Belief Networks (DBN)

- Semi Restricted Boltzmann Machines

- Boltzmann Machines

- Unrolling DBN to a BPN

- Contrastive Divergence (CD) learning

- Persistent Contrastive Divergence (PCD) learning

- Gradient descent optimization

- Nesterov momentum

- Conjugate gradient optimization

- Gaussian Processes

- Optimizing Gaussian Processes as BPNs


<a name='x-28MGL-3A-40MGL-FEATURES-20MGL-PAX-3ASECTION-29'></a>

### 2.1 Features

In general, the focus is on power and performance not on ease of use.
For example, it's possible to:

- control the order of presentation of training examples,

- vary learning rate depending on time, state of the optimizer,

- track all kinds of statistics during training,
etc.

Perhaps one day there will be a cookie cutter interface with
restricted functionality if a reasonable compromise is found between
power and utility.

<a name='x-28MGL-3A-40MGL-DEPENDENCIES-20MGL-PAX-3ASECTION-29'></a>

### 2.2 Dependencies

MGL used to rely on [LLA](https://github.com/tpapp/lla) to
interface to BLAS and LAPACK. That's mostly history by now, but
configuration of foreign libraries is still done via `LLA`. See the
README in `LLA` on how to set things up. Note that these days OpenBLAS
is easier to set up and just as fast as ATLAS.

[CL-CUDA](https://github.com/takagi/cl-cuda) is a dependency for
which the NVIDIA CUDA Toolkit needs to be installed, but MGL is
fully functional even if there is no cuda capable gpu installed. See
the `MGL-MAT:WITH-CUDA*` macro for how to use it.

<a name='x-28MGL-3A-40MGL-TESTS-20MGL-PAX-3ASECTION-29'></a>

### 2.3 Tests

Run the built in tests 
with:

    (ASDF:OOS 'ASDF:TEST-OP '#:MGL)

Note, that most of the tests are rather stochastic and can fail once
in a while.

<a name='x-28MGL-3A-40MGL-BUNDLED-SOFTWARE-20MGL-PAX-3ASECTION-29'></a>

### 2.4 Bundled Software

With [MGL-PAX](https://github.com/melisgl/mgl-pax) and
[MGL-MAT](https://github.com/melisgl/mgl-mat) libraries split off
there remains only a single library bundled with MGL which does
not depend on the rest of MGL:

- `MGL-GNUPLOT`, a plotting library.

There is also MGL-VISUALS which does depend on MGL.

<a name='x-28MGL-3A-40MGL-BASIC-CONCEPTS-20MGL-PAX-3ASECTION-29'></a>

## 3 Basic Concepts

MODEL, training set, test set, validation set,
sample/instance/example.

<a name='x-28MGL-DATASET-3A-40MGL-DATASET-20MGL-PAX-3ASECTION-29'></a>

## 4 Dataset

###### \[in package MGL-DATASET\]
Ultimately machine learning is about creating models of some
domain. The observations in the modelled domain are called
*instances*. Sets of instances are called *datasets*. Datasets are
used when fitting a model or when making predictions.

Implementationally speaking, an instance is typically represented by
a set of numbers which is called *feature vector*. A dataset is a
`SEQUENCE` of such instances or a [Sampler][af7d] object that produces
instances.

<a name='x-28MGL-DATASET-3A-40MGL-SAMPLER-20MGL-PAX-3ASECTION-29'></a>

### 4.1 Sampler

Some algorithms do not need random access to the entire dataset and
can work with a stream observations. Samplers are simple generators
providing two functions: [`SAMPLE`][6fc3] and [`FINISHEDP`][d503].

<a name='x-28MGL-DATASET-3ASAMPLE-20GENERIC-FUNCTION-29'></a>

- [generic-function] **SAMPLE** *SAMPLER*

    If not `SAMPLER` has not run out of data (see
    [`FINISHEDP`][d503]) [`SAMPLE`][6fc3] returns an object that represents a sample from
    the world to be experienced or, in other words, simply something the
    can be used as input for training or prediction.

<a name='x-28MGL-DATASET-3AFINISHEDP-20GENERIC-FUNCTION-29'></a>

- [generic-function] **FINISHEDP** *SAMPLER*

    See if `SAMPLER` has run out of examples.

<a name='x-28MGL-DATASET-3ALIST-SAMPLES-20FUNCTION-29'></a>

- [function] **LIST-SAMPLES** *SAMPLER MAX-SIZE*

    Return a list of samples of length at most `MAX-SIZE` or less if
    `SAMPLER` runs out.

<a name='x-28MGL-DATASET-3AMAKE-SEQUENCE-SAMPLER-20FUNCTION-29'></a>

- [function] **MAKE-SEQUENCE-SAMPLER** *SEQ*

    A simple sampler that returns elements of `SEQ` once, in order.

<a name='x-28MGL-DATASET-3A-2AINFINITELY-EMPTY-DATASET-2A-20VARIABLE-29'></a>

- [variable] **\*INFINITELY-EMPTY-DATASET\*** *#\<FUNCTION-SAMPLER "infintely empty" \>*

    This is the default dataset for [`MGL-OPT:MINIMIZE`][bca8]. It's an infinite
    stream of NILs.

<a name='x-28MGL-DATASET-3A-40MGL-SAMPLER-FUNCTION-SAMPLER-20MGL-PAX-3ASECTION-29'></a>

#### 4.1.1 Function Sampler

<a name='x-28MGL-DATASET-3AFUNCTION-SAMPLER-20CLASS-29'></a>

- [class] **FUNCTION-SAMPLER**

    A sampler with a function in its [`SAMPLER`][37bf] that
    produces a stream of samples which may or may not be finite
    depending on [`MAX-N-SAMPLES`][f56b]. [`FINISHEDP`][d503] returns `T` iff [`MAX-N-SAMPLES`][f56b] is
    non-nil, and it's not greater than the number of samples
    generated ([`N-SAMPLES`][fd45]).
    
        (list-samples (make-instance 'function-sampler
                                     :sampler (lambda ()
                                                (random 10))
                                     :max-n-samples 5)
                      10)
        => (3 5 2 3 3)


<a name='x-28MGL-DATASET-3ASAMPLER-20-28MGL-PAX-3AREADER-20MGL-DATASET-3AFUNCTION-SAMPLER-29-29'></a>

- [reader] **SAMPLER** *FUNCTION-SAMPLER* *(:SAMPLER)*

    A generator function of no arguments that returns
    the next sample.

<a name='x-28MGL-DATASET-3AMAX-N-SAMPLES-20-28MGL-PAX-3AACCESSOR-20MGL-DATASET-3AFUNCTION-SAMPLER-29-29'></a>

- [accessor] **MAX-N-SAMPLES** *FUNCTION-SAMPLER* *(:MAX-N-SAMPLES = NIL)*

<a name='x-28MGL-COMMON-3ANAME-20-28MGL-PAX-3AREADER-20MGL-DATASET-3AFUNCTION-SAMPLER-29-29'></a>

- [reader] **NAME** *FUNCTION-SAMPLER* *(:NAME = NIL)*

    An arbitrary object naming the sampler. Only used
    for printing the sampler object.

<a name='x-28MGL-DATASET-3AN-SAMPLES-20-28MGL-PAX-3AREADER-20MGL-DATASET-3AFUNCTION-SAMPLER-29-29'></a>

- [reader] **N-SAMPLES** *FUNCTION-SAMPLER* *(:N-SAMPLES = 0)*



<a name='x-28MGL-RESAMPLE-3A-40MGL-RESAMPLE-20MGL-PAX-3ASECTION-29'></a>

## 5 Resampling

###### \[in package MGL-RESAMPLE\]
The focus of this package is on resampling methods such as
cross-validation and bagging which can be used for model evaluation,
model selection, and also as a simple form of ensembling. Data
partitioning and sampling functions are also provided because they
tend to be used together with resampling.

<a name='x-28MGL-RESAMPLE-3A-40MGL-RESAMPLE-PARTITIONS-20MGL-PAX-3ASECTION-29'></a>

### 5.1 Partitions

The following functions partition a dataset (currently only
SEQUENCEs are supported) into a number of partitions. For each
element in the original dataset there is exactly one partition that
contains it.

<a name='x-28MGL-RESAMPLE-3AFRACTURE-20FUNCTION-29'></a>

- [function] **FRACTURE** *FRACTIONS SEQ &KEY WEIGHT*

    Partition `SEQ` into a number of subsequences. `FRACTIONS` is either a
    positive integer or a list of non-negative real numbers. `WEIGHT` is
    `NIL` or a function that returns a non-negative real number when
    called with an element from `SEQ`. If `FRACTIONS` is a positive integer
    then return a list of that many subsequences with equal sum of
    weights bar rounding errors, else partition `SEQ` into subsequences,
    where the sum of weights of subsequence I is proportional to element
    I of `FRACTIONS`. If `WEIGHT` is `NIL`, then it's element is assumed to
    have the same weight.
    
    To split into 5 sequences:
    
        (fracture 5 '(0 1 2 3 4 5 6 7 8 9))
        => ((0 1) (2 3) (4 5) (6 7) (8 9))
    
    To split into two sequences whose lengths are proportional to 2 and
    3:
    
        (fracture '(2 3) '(0 1 2 3 4 5 6 7 8 9))
        => ((0 1 2 3) (4 5 6 7 8 9))


<a name='x-28MGL-RESAMPLE-3ASTRATIFY-20FUNCTION-29'></a>

- [function] **STRATIFY** *SEQ &KEY (KEY #'IDENTITY) (TEST #'EQL)*

    Return the list of strata of `SEQ`. `SEQ` is sequence of elements for
    which the function `KEY` returns the class they belong to. Such
    classes are opaque objects compared for equality with `TEST`. A
    stratum is a sequence of elements with the same (under `TEST`) `KEY`.
    
        (stratify '(0 1 2 3 4 5 6 7 8 9) :key #'evenp)
        => ((0 2 4 6 8) (1 3 5 7 9))


<a name='x-28MGL-RESAMPLE-3AFRACTURE-STRATIFIED-20FUNCTION-29'></a>

- [function] **FRACTURE-STRATIFIED** *FRACTIONS SEQ &KEY (KEY #'IDENTITY) (TEST #'EQL) WEIGHT*

    Similar to [`FRACTURE`][2b76], but also makes sure that keys are evenly
    distributed among the partitions (see [`STRATIFY`][5a3f]). It can be useful
    for classification tasks to partition the data set while keeping the
    distribution of classes the same.
    
    Note that the sets returned are not in random order. In fact, they
    are sorted internally by `KEY`.
    
    For example, to make two splits with approximately the same number
    of even and odd numbers:
    
        (fracture-stratified 2 '(0 1 2 3 4 5 6 7 8 9) :key #'evenp)
        => ((0 2 1 3) (4 6 8 5 7 9))


<a name='x-28MGL-RESAMPLE-3A-40MGL-RESAMPLE-CROSS-VALIDATION-20MGL-PAX-3ASECTION-29'></a>

### 5.2 Cross-validation

<a name='x-28MGL-RESAMPLE-3ACROSS-VALIDATE-20FUNCTION-29'></a>

- [function] **CROSS-VALIDATE** *DATA FN &KEY (N-FOLDS 5) (FOLDS (ALEXANDRIA.0.DEV:IOTA N-FOLDS)) (SPLIT-FN #'SPLIT-FOLD/MOD) PASS-FOLD*

    Map `FN` over the `FOLDS` of `DATA` split with `SPLIT-FN` and collect the
    results in a list. The simplest demonstration is:
    
        (cross-validate '(0 1 2 3 4)
                        (lambda (test training)
                         (list test training))
                        :n-folds 5)
        => (((0) (1 2 3 4))
            ((1) (0 2 3 4))
            ((2) (0 1 3 4))
            ((3) (0 1 2 4))
            ((4) (0 1 2 3)))
    
    Of course, in practice one would typically train a model and return
    the trained model and/or its score on `TEST`. Also, sometimes one may
    want to do only some of the folds and remember which ones they were:
    
        (cross-validate '(0 1 2 3 4)
                        (lambda (fold test training)
                         (list :fold fold test training))
                        :folds '(2 3)
                        :pass-fold t)
        => ((:fold 2 (2) (0 1 3 4))
            (:fold 3 (3) (0 1 2 4)))
    
    Finally, the way the data is split can be customized. By default
    [`SPLIT-FOLD/MOD`][02de] is called with the arguments `DATA`, the fold (from
    among `FOLDS`) and `N-FOLDS`. [`SPLIT-FOLD/MOD`][02de] returns two values which
    are then passed on to `FN`. One can use [`SPLIT-FOLD/CONT`][9589] or
    [`SPLIT-STRATIFIED`][edd9] or any other function that works with these
    arguments. The only real constraint is that `FN` has to take as many
    arguments (plus the fold argument if `PASS-FOLD`) as `SPLIT-FN`
    returns.

<a name='x-28MGL-RESAMPLE-3ASPLIT-FOLD-2FMOD-20FUNCTION-29'></a>

- [function] **SPLIT-FOLD/MOD** *SEQ FOLD N-FOLDS*

    Partition `SEQ` into two sequences: one with elements of `SEQ` with
    indices whose remainder is `FOLD` when divided with `N-FOLDS`, and a
    second one with the rest. The second one is the larger set. The
    order of elements remains stable. This function is suitable as the
    `SPLIT-FN` argument of [`CROSS-VALIDATE`][8375].

<a name='x-28MGL-RESAMPLE-3ASPLIT-FOLD-2FCONT-20FUNCTION-29'></a>

- [function] **SPLIT-FOLD/CONT** *SEQ FOLD N-FOLDS*

    Imagine dividing `SEQ` into `N-FOLDS` subsequences of the same
    size (bar rounding). Return the subsequence of index `FOLD` as the
    first value and the all the other subsequences concatenated into one
    as the second value. The order of elements remains stable. This
    function is suitable as the `SPLIT-FN` argument of [`CROSS-VALIDATE`][8375].

<a name='x-28MGL-RESAMPLE-3ASPLIT-STRATIFIED-20FUNCTION-29'></a>

- [function] **SPLIT-STRATIFIED** *SEQ FOLD N-FOLDS &KEY (KEY #'IDENTITY) (TEST #'EQL) WEIGHT*

    Split `SEQ` into `N-FOLDS` partitions (as in [`FRACTURE-STRATIFIED`][e57e]).
    Return the partition of index `FOLD` as the first value, and the
    concatenation of the rest as the second value. This function is
    suitable as the `SPLIT-FN` argument of [`CROSS-VALIDATE`][8375] (mostly likely
    as a closure with `KEY`, `TEST`, `WEIGHT` bound).

<a name='x-28MGL-RESAMPLE-3A-40MGL-RESAMPLE-BAGGING-20MGL-PAX-3ASECTION-29'></a>

### 5.3 Bagging

<a name='x-28MGL-RESAMPLE-3ABAG-20FUNCTION-29'></a>

- [function] **BAG** *SEQ FN &KEY (RATIO 1) N WEIGHT (REPLACEMENT T) KEY (TEST #'EQL) (RANDOM-STATE \*RANDOM-STATE\*)*

    Sample from `SEQ` with [`SAMPLE-FROM`][76b8] (passing `RATIO`, `WEIGHT`,
    `REPLACEMENT`), or [`SAMPLE-STRATIFIED`][7ae7] if `KEY` is not `NIL`. Call `FN` with
    the sample. If `N` is `NIL` then keep repeating this until `FN` performs a
    non-local exit. Else `N` must be a non-negative integer, `N` iterations
    will be performed, the primary values returned by `FN` collected into
    a list and returned. See [`SAMPLE-FROM`][76b8] and [`SAMPLE-STRATIFIED`][7ae7] for
    examples.

<a name='x-28MGL-RESAMPLE-3ASAMPLE-FROM-20FUNCTION-29'></a>

- [function] **SAMPLE-FROM** *RATIO SEQ &KEY WEIGHT REPLACEMENT (RANDOM-STATE \*RANDOM-STATE\*)*

    Return a sequence constructed by sampling with or without
    `REPLACEMENT` from `SEQ`. The sum of weights in the result sequence will
    approximately be the sum of weights of `SEQ` times `RATIO`. If `WEIGHT` is
    `NIL` then elements are assumed to have equal weights, else `WEIGHT`
    should return a non-negative real number when called with an element
    of `SEQ`.
    
    To randomly select half of the elements:
    
        (sample-from 1/2 '(0 1 2 3 4 5))
        => (5 3 2)
    
    To randomly select some elements such that the sum of their weights
    constitute about half of the sum of weights across the whole
    sequence:
    
        (sample-from 1/2 '(0 1 2 3 4 5 6 7 8 9) :weight #'identity)
        => (9 4 1 6 8) ; sums to 28 that's near 45/2
    
    To sample with replacement (that is, allowing the element to be
    sampled multiple times):
    
        (sample-from 1 '(0 1 2 3 4 5) :replacement t)
        => (1 1 5 1 4 4)


<a name='x-28MGL-RESAMPLE-3ASAMPLE-STRATIFIED-20FUNCTION-29'></a>

- [function] **SAMPLE-STRATIFIED** *RATIO SEQ &KEY WEIGHT REPLACEMENT (KEY #'IDENTITY) (TEST #'EQL) (RANDOM-STATE \*RANDOM-STATE\*)*

    Like [`SAMPLE-FROM`][76b8] but makes sure that the weighted proportion of
    classes in the result is approximately the same as the proportion in
    `SEQ`. See [`STRATIFY`][5a3f] for the description of `KEY` and `TEST`.

<a name='x-28MGL-RESAMPLE-3A-40MGL-RESAMPLE-CV-BAGGING-20MGL-PAX-3ASECTION-29'></a>

### 5.4 CV Bagging

<a name='x-28MGL-RESAMPLE-3ABAG-CV-20FUNCTION-29'></a>

- [function] **BAG-CV** *DATA FN &KEY N (N-FOLDS 5) (FOLDS (ALEXANDRIA.0.DEV:IOTA N-FOLDS)) (SPLIT-FN #'SPLIT-FOLD/MOD) PASS-FOLD (RANDOM-STATE \*RANDOM-STATE\*)*

    Perform cross-validation on different shuffles of `DATA` `N` times and
    collect the results. Since [`CROSS-VALIDATE`][8375] collects the return values
    of `FN`, the return value of this function is a list of lists of `FN`
    results. If `N` is `NIL`, don't collect anything just keep doing
    repeated CVs until `FN` performs an non-local exit.
    
    The following example simply collects the test and training sets for
    2-fold CV repeated 3 times with shuffled data:
    
         (bag-cv '(0 1 2 3 4) #'list :n 3 :n-folds 2)
         => ((((2 3 4) (1 0))
              ((1 0) (2 3 4)))
             (((2 1 0) (4 3))
              ((4 3) (2 1 0)))
             (((1 0 3) (2 4))
              ((2 4) (1 0 3))))
    
    CV bagging is useful when a single CV is not producing stable
    results. As an ensemble method, CV bagging has the advantage over
    bagging that each example will occur the same number of times and
    after the first CV is complete there is a complete but less reliable
    estimate for each example which gets refined by further CVs.

<a name='x-28MGL-RESAMPLE-3A-40MGL-RESAMPLE-MISC-20MGL-PAX-3ASECTION-29'></a>

### 5.5 Miscellaneous Operations

<a name='x-28MGL-RESAMPLE-3ASPREAD-STRATA-20FUNCTION-29'></a>

- [function] **SPREAD-STRATA** *SEQ &KEY (KEY #'IDENTITY) (TEST #'EQL)*

    Return a sequence that's a reordering of `SEQ` such that elements
    belonging to different strata (under `KEY` and `TEST`, see [`STRATIFY`][5a3f]) are
    distributed evenly. The order of elements belonging to the same
    stratum is unchanged.
    
    For example, to make sure that even and odd numbers are distributed
    evenly:
    
        (spread-strata '(0 2 4 6 8 1 3 5 7 9) :key #'evenp)
        => (0 1 2 3 4 5 6 7 8 9)
    
    Same thing with unbalanced classes:
    
        (spread-strata (vector 0 2 3 5 6 1 4)
                       :key (lambda (x)
                              (if (member x '(1 4))
                                  t
                                  nil)))
        => #(0 1 2 3 4 5 6)


<a name='x-28MGL-RESAMPLE-3AZIP-EVENLY-20FUNCTION-29'></a>

- [function] **ZIP-EVENLY** *SEQS &KEY RESULT-TYPE*

    Make a single sequence out of the sequences in `SEQS` so that in the
    returned sequence indices of elements belonging to the same source
    sequence are spread evenly across the whole range. The result is a
    list is `RESULT-TYPE` is `LIST`, it's a vector if `RESULT-TYPE` is `VECTOR`.
    If `RESULT-TYPE` is `NIL`, then it's determined by the type of the first
    sequence in `SEQS`.
    
        (zip-evenly '((0 2 4) (1 3)))
        => (0 1 2 3 4)


<a name='x-28MGL-OPT-3A-40MGL-OPT-20MGL-PAX-3ASECTION-29'></a>

## 6 Gradient Based Optimization

###### \[in package MGL-OPT\]
We have a real valued, differentiable function F and the task is to
find the parameters that minimize its value. Optimization starts
from a single point in the parameter space of F, and this single
point is updated iteratively based on the gradient and value of F at
or around the current point.

Note that while the stated problem is that of global optimization,
for non-convex functions, most algorithms will tend to converge to a
local optimum.

Currently, there are two optimization algorithms:
[Gradient Descent][53a7] (with several variants) and [Conjugate Gradient][8729] both of
which are first order methods (they do not need second order
gradients) but more can be added with the [Extension API][2730].

<a name='x-28MGL-OPT-3AMINIMIZE-20FUNCTION-29'></a>

- [function] **MINIMIZE** *OPTIMIZER GRADIENT-SOURCE &KEY (WEIGHTS (LIST-SEGMENTS GRADIENT-SOURCE)) (DATASET \*INFINITELY-EMPTY-DATASET\*)*

    Minimize the value of the real valued function represented by
    `GRADIENT-SOURCE` by updating some of its parameters in `WEIGHTS` (a MAT
    or a sequence of MATs). Return `WEIGHTS`. `DATASET` (see
    MGL:@MGL-DATASETS) is a set of unoptimized parameters of the same
    function. For example, `WEIGHTS` may be the weights of a neural
    network while `DATASET` is the training set consisting of inputs
    suitable for MGL-TRAIN:SET-INPUT. The default `DATASET`,
    (*EMPTY-DATASET*) is suitable for when all parameters are optimized,
    so there is nothing left to come from the environment.
    
    Optimization terminates if `DATASET` is a sampler and it runs out or
    when some other condition met (see [`TERMINATION`][bec0], for example). If
    `DATASET` is a `SEQUENCE`, then it is reused over and over again.

<a name='x-28MGL-OPT-3A-2AACCUMULATING-INTERESTING-GRADIENTS-2A-20VARIABLE-29'></a>

- [variable] **\*ACCUMULATING-INTERESTING-GRADIENTS\*** *NIL*

<a name='x-28MGL-OPT-3A-40MGL-OPT-EXTENSION-API-20MGL-PAX-3ASECTION-29'></a>

### 6.1 Extension API

<a name='x-28MGL-OPT-3A-40MGL-OPT-OPTIMIZER-20MGL-PAX-3ASECTION-29'></a>

#### 6.1.1 Implementing Optimizers

<a name='x-28MGL-OPT-3AMINIMIZE-2A-20GENERIC-FUNCTION-29'></a>

- [generic-function] **MINIMIZE\*** *OPTIMIZER GRADIENT-SOURCE WEIGHTS DATASET*

<a name='x-28MGL-OPT-3AINITIALIZE-OPTIMIZER-2A-20GENERIC-FUNCTION-29'></a>

- [generic-function] **INITIALIZE-OPTIMIZER\*** *OPTIMIZER GRADIENT-SOURCE WEIGHTS DATASET*

    Called automatically before training starts, this
    function sets up `OPTIMIZER` to be suitable for optimizing
    `GRADIENT-SOURCE`. It typically creates appropriately sized
    accumulators for the gradients.

<a name='x-28MGL-OPT-3ATERMINATE-OPTIMIZATION-P-20FUNCTION-29'></a>

- [function] **TERMINATE-OPTIMIZATION-P** *N-INSTANCES TERMINATION*

    Utility function for subclasses of [`ITERATIVE-OPTIMIZER`][83bf]. It returns
    whether optimization is to be terminated based on `N-INSTANCES` and
    `TERMINATION` that are values of the respective accessors of
    [`ITERATIVE-OPTIMIZER`][83bf].

<a name='x-28MGL-OPT-3A-40MGL-OPT-GRADIENT-SOURCE-20MGL-PAX-3ASECTION-29'></a>

#### 6.1.2 Implementing Gradient Sources

<a name='x-28MGL-OPT-3AINITIALIZE-GRADIENT-SOURCE-2A-20GENERIC-FUNCTION-29'></a>

- [generic-function] **INITIALIZE-GRADIENT-SOURCE\*** *OPTIMIZER GRADIENT-SOURCE WEIGHTS DATASET*

    Called automatically before training starts, this
    function sets up `SINK` to be suitable for `SOURCE`. It typically
    creates accumulator arrays in the sink for the gradients.

<a name='x-28MGL-OPT-3AACCUMULATE-GRADIENTS-2A-20GENERIC-FUNCTION-29'></a>

- [generic-function] **ACCUMULATE-GRADIENTS\*** *SOURCE SINK BATCH MULTIPLIER VALUEP*

    Add `MULTIPLIER` times the sum of first-order
    gradients to accumulators of `SINK` (normally accessed with
    [`DO-GRADIENT-SINK`][643d]) and if `VALUEP`, return the sum of values of the
    function being optimized for a `BATCH` of instances. `SOURCE` is the
    object representing the function being optimized, `SINK` is gradient
    sink.

<a name='x-28MGL-OPT-3A-40MGL-OPT-GRADIENT-SINK-20MGL-PAX-3ASECTION-29'></a>

#### 6.1.3 Implementing Gradient Sinks

<a name='x-28MGL-OPT-3AMAP-GRADIENT-SINK-20GENERIC-FUNCTION-29'></a>

- [generic-function] **MAP-GRADIENT-SINK** *FN SINK*

    Call `FN` of lambda list (`SEGMENT` `ACCUMULATOR`) on
    each segment and their corresponding accumulator MAT in `SINK`.

<a name='x-28MGL-OPT-3ADO-GRADIENT-SINK-20MGL-PAX-3AMACRO-29'></a>

- [macro] **DO-GRADIENT-SINK** *((SEGMENT ACCUMULATOR) SINK) &BODY BODY*

<a name='x-28MGL-OPT-3ACALL-WITH-SINK-ACCUMULATOR-20GENERIC-FUNCTION-29'></a>

- [generic-function] **CALL-WITH-SINK-ACCUMULATOR** *FN SEGMENT SOURCE SINK*

<a name='x-28MGL-OPT-3AWITH-SINK-ACCUMULATOR-20MGL-PAX-3AMACRO-29'></a>

- [macro] **WITH-SINK-ACCUMULATOR** *(ACCUMULATOR (SEGMENT SOURCE SINK)) &BODY BODY*

<a name='x-28MGL-OPT-3AACCUMULATED-IN-SINK-P-20FUNCTION-29'></a>

- [function] **ACCUMULATED-IN-SINK-P** *SEGMENT SOURCE SINK*

<a name='x-28MGL-OPT-3A-40MGL-OPT-ITERATIVE-OPTIMIZER-20MGL-PAX-3ASECTION-29'></a>

### 6.2 Iterative Optimizer

<a name='x-28MGL-OPT-3AITERATIVE-OPTIMIZER-20CLASS-29'></a>

- [class] **ITERATIVE-OPTIMIZER**

    An abstract base class of [Gradient Descent][53a7] and
    [Conjugate Gradient][8729] based optimizers that iterate over instances until a
    termination condition is met.

<a name='x-28MGL-OPT-3AN-INSTANCES-20-28MGL-PAX-3AREADER-20MGL-OPT-3AITERATIVE-OPTIMIZER-29-29'></a>

- [reader] **N-INSTANCES** *ITERATIVE-OPTIMIZER* *(:N-INSTANCES = 0)*

    The number of instances this optimizer has seen so
    far. Incremented automatically during optimization.

<a name='x-28MGL-OPT-3ATERMINATION-20-28MGL-PAX-3AACCESSOR-20MGL-OPT-3AITERATIVE-OPTIMIZER-29-29'></a>

- [accessor] **TERMINATION** *ITERATIVE-OPTIMIZER* *(:TERMINATION = NIL)*

    If a number, it's the number of instances to train
    on in the sense of [`N-INSTANCES`][66a1]. If [`N-INSTANCES`][66a1] is equal or greater
    than this value optimization stops. If [`TERMINATION`][bec0] is `NIL`, then
    optimization will continue. If it is `T`, then optimization will
    stop. If it is a function of no arguments, then its return value
    is processed as if it was returned by [`TERMINATION`][bec0].

<a name='x-28MGL-OPT-3ASET-N-INSTANCES-20GENERIC-FUNCTION-29'></a>

- [generic-function] **SET-N-INSTANCES** *OPTIMIZER GRADIENT-SOURCE N-INSTANCES*

    Called whenever `N-INSTANCES` of `OPTIMIZER` is
    incremented. Hang an `:AFTER` method on this to print some
    statistics.

<a name='x-28MGL-GD-3A-40MGL-GD-20MGL-PAX-3ASECTION-29'></a>

### 6.3 Gradient Descent

###### \[in package MGL-GD\]
Gradient descent is a first-order optimization algorithm. Relying
completely on first derivatives, it does not even evaluate the
function to be minimized. Let's see how to minimize a numerical lisp
function with respect to some of its parameters.

```
;;; Create an object representing the sine function.
(defparameter *diff-fn-1*
  (make-instance 'mgl-diffun:diffun
                 :fn #'sin
                 ;; We are going to optimize its only parameter.
                 :weight-indices '(0)))

;;; Minimize SIN. Note that there is no dataset involved because all
;;; parameters are being optimized.
(minimize (make-instance 'batch-gd-optimizer :termination 1000)
          *diff-fn-1*
          :weights (make-mat 1))
;;; => A MAT with a single value of about -pi/2.

;;; Create a differentiable function for f(x,y)=(x-y)^2. X is a
;;; parameter whose values come from the DATASET argument passed to
;;; MINIMIZE. Y is a parameter to be optimized (a 'weight').
(defparameter *diff-fn-2*
  (make-instance 'mgl-diffun:diffun
                 :fn (lambda (x y)
                       (expt (- x y) 2))
                 :parameter-indices '(0)
                 :weight-indices '(1)))

;;; Find the Y that minimizes the distance from the instances
;;; generated by the sampler.
(minimize (make-instance 'batch-gd-optimizer :batch-size 10)
          *diff-fn-2*
          :weights (make-mat 1)
          :dataset (make-instance 'function-sampler
                                  :sampler (lambda ()
                                             (list (+ 10
                                                      (gaussian-random-1))))
                                  :max-n-samples 1000))
;;; => A MAT with a single value of about 10, the expected value of
;;; the instances in the dataset.

;;; The dataset can be a SEQUENCE in which case we'd better set
;;; TERMINATION else optimization would never finish.
(minimize (make-instance 'batch-gd-optimizer
                         :termination 1000)
          *diff-fn-2*
          :weights (make-mat 1)
          :dataset '((0) (1) (2) (3) (4) (5)))
;;; => A MAT with a single value of about 2.5.
```

We are going to see a number of accessors for optimizer paramaters.
In general, it's allowed to `SETF` real slot accessors (as opposed to
readers and writers) at any time during optimization and so is
defining a method on an optimizer subclass that computes the value
in any way. For example, to decay the learning rate on a per
mini-batch basis:

    (defmethod learning-rate ((optimizer my-batch-gd-optimizer))
      (* (slot-value optimizer 'learning-rate)
         (expt 0.998
               (/ (n-instances optimizer) 60000))))


<a name='x-28MGL-GD-3A-40MGL-GD-BATCH-GD-OPTIMIZER-20MGL-PAX-3ASECTION-29'></a>

#### 6.3.1 Batch GD Optimizer

<a name='x-28MGL-GD-3ABATCH-GD-OPTIMIZER-20CLASS-29'></a>

- [class] **BATCH-GD-OPTIMIZER** *GD-OPTIMIZER*

    Updates all weights simultaneously after chewing
    through `BATCH-SIZE`([`0`][dc9d] [`1`][f94f]) inputs. [`PER-WEIGHT-BATCH-GD-OPTIMIZER`][1fa8] may be a
    better choice when some weights can go unused for instance due to
    missing input values.
    
    Assuming that `ACCUMULATOR` has the sum of gradients for a mini-batch,
    the weight update looks like this:
    
        delta_w' += momentum * delta_w +
                    accumulator / batch_size + l2 * w + l1 * sign(w)
        
        w' -= learning_rate * delta_w'
    
    which is the same as the more traditional formulation:
    
        delta_w' += momentum * delta_w +
                    learning_rate * (df/dw / batch_size + l2 * w + l1 * sign(w))
        
        w' -= delta_w'
    
    but the former works better when batch size, momentum or learning
    rate change during the course of optimization. The above is with
    normal momentum, Nesterov's momentum (see [`MOMENTUM-TYPE`][e0c8]) momentum is
    also available.

<a name='x-28MGL-OPT-3AN-INSTANCES-20-28MGL-PAX-3AREADER-20MGL-OPT-3AITERATIVE-OPTIMIZER-29-29'></a>

- [reader] **N-INSTANCES** *ITERATIVE-OPTIMIZER* *(:N-INSTANCES = 0)*

    The number of instances this optimizer has seen so
    far. Incremented automatically during optimization.

<a name='x-28MGL-OPT-3ATERMINATION-20-28MGL-PAX-3AACCESSOR-20MGL-OPT-3AITERATIVE-OPTIMIZER-29-29'></a>

- [accessor] **TERMINATION** *ITERATIVE-OPTIMIZER* *(:TERMINATION = NIL)*

    If a number, it's the number of instances to train
    on in the sense of [`N-INSTANCES`][66a1]. If [`N-INSTANCES`][66a1] is equal or greater
    than this value optimization stops. If [`TERMINATION`][bec0] is `NIL`, then
    optimization will continue. If it is `T`, then optimization will
    stop. If it is a function of no arguments, then its return value
    is processed as if it was returned by [`TERMINATION`][bec0].

<a name='x-28MGL-COMMON-3ABATCH-SIZE-20-28MGL-PAX-3AACCESSOR-20MGL-GD-3A-3AGD-OPTIMIZER-29-29'></a>

- [accessor] **BATCH-SIZE** *GD-OPTIMIZER* *(:BATCH-SIZE = 1)*

    After having gone through `BATCH-SIZE`([`0`][dc9d] [`1`][f94f]) number of
    inputs, weights are updated. With `BATCH-SIZE`([`0`][dc9d] [`1`][f94f]) 1, one gets
    Stochastics Gradient Descent. With `BATCH-SIZE`([`0`][dc9d] [`1`][f94f]) equal to the number
    of instances in the dataset, one gets standard, 'batch' gradient
    descent. With `BATCH-SIZE`([`0`][dc9d] [`1`][f94f]) between these two extremes, one gets the
    most practical 'mini-batch' compromise.

<a name='x-28MGL-GD-3ALEARNING-RATE-20-28MGL-PAX-3AACCESSOR-20MGL-GD-3A-3AGD-OPTIMIZER-29-29'></a>

- [accessor] **LEARNING-RATE** *GD-OPTIMIZER* *(:LEARNING-RATE = 0.10000000149011612d0)*

    This is the step size along the gradient. Decrease
    it if optimization diverges, increase it if it doesn't make
    progress.

<a name='x-28MGL-GD-3AMOMENTUM-20-28MGL-PAX-3AACCESSOR-20MGL-GD-3A-3AGD-OPTIMIZER-29-29'></a>

- [accessor] **MOMENTUM** *GD-OPTIMIZER* *(:MOMENTUM = 0.0d0)*

    A value in the [0, 1) interval. [`MOMENTUM`][ed3d] times the
    previous weight change is added to the gradient. 0 means no
    momentum.

<a name='x-28MGL-GD-3AMOMENTUM-TYPE-20-28MGL-PAX-3AREADER-20MGL-GD-3A-3AGD-OPTIMIZER-29-29'></a>

- [reader] **MOMENTUM-TYPE** *GD-OPTIMIZER* *(:MOMENTUM-TYPE = :NORMAL)*

    One of `:NORMAL` and `:NESTEROV`. For pure
    optimization Nesterov's momentum may be better, but it also
    increases chances of overfitting.

<a name='x-28MGL-GD-3AWEIGHT-DECAY-20-28MGL-PAX-3AACCESSOR-20MGL-GD-3A-3AGD-OPTIMIZER-29-29'></a>

- [accessor] **WEIGHT-DECAY** *GD-OPTIMIZER* *(:WEIGHT-DECAY = 0.0d0)*

    An L2 penalty. It discourages large weights, much
    like a zero mean gaussian prior. [`WEIGHT-DECAY`][ce14] \* WEIGHT is added to
    the gradient to penalize large weights. It's as if the function
    whose minimum is sought had WEIGHT-DECAY\*sum\_i{0.5 \* WEIGHT\_i^2}
    added to it.

<a name='x-28MGL-GD-3AWEIGHT-PENALTY-20-28MGL-PAX-3AACCESSOR-20MGL-GD-3A-3AGD-OPTIMIZER-29-29'></a>

- [accessor] **WEIGHT-PENALTY** *GD-OPTIMIZER* *(:WEIGHT-PENALTY = 0.0d0)*

    An L1 penalty. It encourages sparsity.
    `SIGN`(WEIGHT) \* [`WEIGHT-PENALTY`][a7de] is added to the gradient pushing the
    weight towards negative infinity. It's as if the function whose
    minima is sought had WEIGHT-PENALTY\*sum\_i{abs(WEIGHT\_i)} added to
    it. Putting it on feature biases consitutes a sparsity constraint
    on the features.

<a name='x-28MGL-GD-3AAFTER-UPDATE-HOOK-20-28MGL-PAX-3AACCESSOR-20MGL-GD-3A-3AGD-OPTIMIZER-29-29'></a>

- [accessor] **AFTER-UPDATE-HOOK** *GD-OPTIMIZER* *(:AFTER-UPDATE-HOOK = NIL)*

    A list of functions with no arguments called after
    each weight update.

<a name='x-28MGL-GD-3ABEFORE-UPDATE-HOOK-20-28MGL-PAX-3AACCESSOR-20MGL-GD-3ABATCH-GD-OPTIMIZER-29-29'></a>

- [accessor] **BEFORE-UPDATE-HOOK** *BATCH-GD-OPTIMIZER* *(:BEFORE-UPDATE-HOOK = NIL)*

    A list of functions of no parameters. Each
    function is called just before a weight update takes place.
    Convenient to hang some additional gradient accumulating code
    on.

<a name='x-28MGL-GD-3A-40MGL-GD-SEGMENTED-GD-OPTIMIZER-20MGL-PAX-3ASECTION-29'></a>

#### 6.3.2 Segmented GD Optimizer

<a name='x-28MGL-GD-3ASEGMENTED-GD-OPTIMIZER-20CLASS-29'></a>

- [class] **SEGMENTED-GD-OPTIMIZER** *BASE-GD-OPTIMIZER*

    An optimizer that delegates training of segments to
    other optimizers. Useful to delegate training of different segments
    to different optimizers (capable of working with segmentables) or
    simply to not train all segments.

<a name='x-28MGL-OPT-3AN-INSTANCES-20-28MGL-PAX-3AREADER-20MGL-OPT-3AITERATIVE-OPTIMIZER-29-29'></a>

- [reader] **N-INSTANCES** *ITERATIVE-OPTIMIZER* *(:N-INSTANCES = 0)*

    The number of instances this optimizer has seen so
    far. Incremented automatically during optimization.

<a name='x-28MGL-OPT-3ATERMINATION-20-28MGL-PAX-3AACCESSOR-20MGL-OPT-3AITERATIVE-OPTIMIZER-29-29'></a>

- [accessor] **TERMINATION** *ITERATIVE-OPTIMIZER* *(:TERMINATION = NIL)*

    If a number, it's the number of instances to train
    on in the sense of [`N-INSTANCES`][66a1]. If [`N-INSTANCES`][66a1] is equal or greater
    than this value optimization stops. If [`TERMINATION`][bec0] is `NIL`, then
    optimization will continue. If it is `T`, then optimization will
    stop. If it is a function of no arguments, then its return value
    is processed as if it was returned by [`TERMINATION`][bec0].

<a name='x-28MGL-GD-3ASEGMENTER-20-28MGL-PAX-3AREADER-20MGL-GD-3ASEGMENTED-GD-OPTIMIZER-29-29'></a>

- [reader] **SEGMENTER** *SEGMENTED-GD-OPTIMIZER* *(:SEGMENTER)*

    When this optimizer is initialized it loops over
    the segment of the learner with `MAP-SEGMENTS`. [`SEGMENTER`][b6ac] is a
    function that is called with each segment and returns an optimizer
    or `NIL`. Several segments may be mapped to the same optimizer.
    After the segment->optimizer mappings are collected, each
    optimizer is initialized by INITIALIZE-OPTIMIZER with the list of
    segments mapped to it.

<a name='x-28MGL-CORE-3ASEGMENTS-20-28MGL-PAX-3AREADER-20MGL-GD-3ASEGMENTED-GD-OPTIMIZER-29-29'></a>

- [reader] **SEGMENTS** *SEGMENTED-GD-OPTIMIZER*

<a name='x-28MGL-GD-3A-40MGL-GD-PER-WEIGHT-OPTIMIZATION-20MGL-PAX-3ASECTION-29'></a>

#### 6.3.3 Per-weight Optimization

<a name='x-28MGL-GD-3ANORMALIZED-BATCH-GD-OPTIMIZER-20CLASS-29'></a>

- [class] **NORMALIZED-BATCH-GD-OPTIMIZER** *BATCH-GD-OPTIMIZER*

    Like [`BATCH-GD-OPTIMIZER`][9aa2] but keeps count of how many
    times each weight was used in the batch and divides the accumulated
    gradient by this count instead of dividing by `N-INSTANCES-IN-BATCH`.
    This only makes a difference if there are missing values in the
    learner that's being trained. The main feature that distuinguishes
    this class from [`PER-WEIGHT-BATCH-GD-OPTIMIZER`][1fa8] is that batches end at
    same time for all weights.

<a name='x-28MGL-GD-3APER-WEIGHT-BATCH-GD-OPTIMIZER-20CLASS-29'></a>

- [class] **PER-WEIGHT-BATCH-GD-OPTIMIZER** *GD-OPTIMIZER*

    This is much like [`BATCH-GD-OPTIMIZER`][9aa2] but it is more
    clever about when to update weights. Basically every weight has its
    own batch independent from the batches of others. It has desirable
    properties. One can for example put two neural networks together
    without adding any connections between them and the learning will
    produce results equivalent to the separated case. Also, adding
    inputs with only missing values does not change anything.

<a name='x-28MGL-CG-3A-40MGL-CG-20MGL-PAX-3ASECTION-29'></a>

### 6.4 Conjugate Gradient

###### \[in package MGL-CG\]
Conjugate gradient is a first-order optimization algorithm. It's
more advanced than gradient descent as it does line searches which
unfortunately also makes it unsuitable for non-deterministic
functions. Let's see how to minimize a numerical lisp function with
respect to some of its parameters.

```
;;; Create an object representing the sine function.
(defparameter *diff-fn-1*
  (make-instance 'mgl-diffun:diffun
                 :fn #'sin
                 ;; We are going to optimize its only parameter.
                 :weight-indices '(0)))

;;; Minimize SIN. Note that there is no dataset involved because all
;;; parameters are being optimized.
(minimize (make-instance 'cg-optimizer
                         :batch-size 1
                         :termination 1)
          *diff-fn-1*
          :weights (make-mat 1))
;;; => A MAT with a single value of about -pi/2.

;;; Create a differentiable function for f(x,y)=(x-y)^2. X is a
;;; parameter whose values come from the DATASET argument passed to
;;; MINIMIZE. Y is a parameter to be optimized (a 'weight').
(defparameter *diff-fn-2*
  (make-instance 'mgl-diffun:diffun
                 :fn (lambda (x y)
                       (expt (- x y) 2))
                 :parameter-indices '(0)
                 :weight-indices '(1)))

;;; Find the Y that minimizes the distance from the instances
;;; generated by the sampler.
(minimize (make-instance 'cg-optimizer :batch-size 10)
          *diff-fn-2*
          :weights (make-mat 1)
          :dataset (make-instance 'function-sampler
                                  :sampler (lambda ()
                                             (list (+ 10
                                                      (gaussian-random-1))))
                                  :max-n-samples 1000))
;;; => A MAT with a single value of about 10, the expected value of
;;; the instances in the dataset.

;;; The dataset can be a SEQUENCE in which case we'd better set
;;; TERMINATION else optimization would never finish. Note how a
;;; single epoch suffices.
(minimize (make-instance 'cg-optimizer :termination 6)
          *diff-fn-2*
          :weights (make-mat 1)
          :dataset '((0) (1) (2) (3) (4) (5)))
;;; => A MAT with a single value of about 2.5.
```


<a name='x-28MGL-CG-3ACG-20FUNCTION-29'></a>

- [function] **CG** *FN W &KEY (MAX-N-LINE-SEARCHES \*DEFAULT-MAX-N-LINE-SEARCHES\*) (MAX-N-EVALUATIONS-PER-LINE-SEARCH \*DEFAULT-MAX-N-EVALUATIONS-PER-LINE-SEARCH\*) (MAX-N-EVALUATIONS \*DEFAULT-MAX-N-EVALUATIONS\*) (SIG \*DEFAULT-SIG\*) (RHO \*DEFAULT-RHO\*) (INT \*DEFAULT-INT\*) (EXT \*DEFAULT-EXT\*) (RATIO \*DEFAULT-RATIO\*) SPARE-VECTORS*

    [`CG-OPTIMIZER`][864e] passes each batch of data to this function with its
    [`CG-ARGS`][7f6b] passed on.
    
    Minimize a differentiable multivariate function with conjugate
    gradient. The Polak-Ribiere flavour of conjugate gradients is used
    to compute search directions, and a line search using quadratic and
    cubic polynomial approximations and the Wolfe-Powell stopping
    criteria is used together with the slope ratio method for guessing
    initial step sizes. Additionally a bunch of checks are made to make
    sure that exploration is taking place and that extrapolation will
    not be unboundedly large.
    
    `FN` is a function of two parameters: `WEIGHTS` and `DERIVATIVES`. `WEIGHTS`
    is a `MAT` of the same size as `W` that is where the search start from.
    `DERIVATIVES` is also a `MAT` of that size and it is where `FN` shall
    place the partial derivatives. `FN` returns the value of the function
    that is being minimized.
    
    [`CG`][f9f7] performs a number of line searches and invokes `FN` at each step. A
    line search invokes `FN` at most `MAX-N-EVALUATIONS-PER-LINE-SEARCH`
    number of times and can succeed in improving the minimum by the
    sufficient margin or it can fail. Note, the even a failed line
    search may improve further and hence change the weights it's just
    that the improvement was deemed too small. [`CG`][f9f7] stops when either:
    
    - two line searches fail in a row
    
    - `MAX-N-LINE-SEARCHES` is reached
    
    - `MAX-N-EVALUATIONS` is reached
    
    [`CG`][f9f7] returns a `MAT` that contains the best weights, the minimum, the
    number of line searches performed, the number of succesful line
    searches and the number of evaluations.
    
    When using `MAX-N-EVALUATIONS` remember that there is an extra
    evaluation of `FN` before the first line search.
    
    `SPARE-VECTORS` is a list of preallocated MATs of the same size as `W`.
    Passing 6 of them covers the current need of the algorithm and it
    will not cons up vectors of size `W` at all.
    
    NOTE: If the function terminates within a few iterations, it could
    be an indication that the function values and derivatives are not
    consistent (ie, there may be a bug in the implementation of `FN`
    function).
    
    `SIG` and `RHO` are the constants controlling the Wolfe-Powell
    conditions. `SIG` is the maximum allowed absolute ratio between
    previous and new slopes (derivatives in the search direction), thus
    setting `SIG` to low (positive) values forces higher precision in the
    line-searches. `RHO` is the minimum allowed fraction of the
    expected (from the slope at the initial point in the linesearch).
    Constants must satisfy 0 < `RHO` < `SIG` < 1. Tuning of `SIG` (depending
    on the nature of the function to be optimized) may speed up the
    minimization; it is probably not worth playing much with `RHO`.

<a name='x-28MGL-CG-3A-2ADEFAULT-INT-2A-20VARIABLE-29'></a>

- [variable] **\*DEFAULT-INT\*** *0.1*

    Don't reevaluate within `INT` of the limit of the current bracket.

<a name='x-28MGL-CG-3A-2ADEFAULT-EXT-2A-20VARIABLE-29'></a>

- [variable] **\*DEFAULT-EXT\*** *3*

    Extrapolate maximum `EXT` times the current step-size.

<a name='x-28MGL-CG-3A-2ADEFAULT-SIG-2A-20VARIABLE-29'></a>

- [variable] **\*DEFAULT-SIG\*** *0.1*

    `SIG` and `RHO` are the constants controlling the Wolfe-Powell
    conditions. `SIG` is the maximum allowed absolute ratio between
    previous and new slopes (derivatives in the search direction), thus
    setting `SIG` to low (positive) values forces higher precision in the
    line-searches.

<a name='x-28MGL-CG-3A-2ADEFAULT-RHO-2A-20VARIABLE-29'></a>

- [variable] **\*DEFAULT-RHO\*** *0.05*

    `RHO` is the minimum allowed fraction of the expected (from the slope
    at the initial point in the linesearch). Constants must satisfy 0 <
    `RHO` < `SIG` < 1.

<a name='x-28MGL-CG-3A-2ADEFAULT-RATIO-2A-20VARIABLE-29'></a>

- [variable] **\*DEFAULT-RATIO\*** *10*

    Maximum allowed slope ratio.

<a name='x-28MGL-CG-3A-2ADEFAULT-MAX-N-LINE-SEARCHES-2A-20VARIABLE-29'></a>

- [variable] **\*DEFAULT-MAX-N-LINE-SEARCHES\*** *NIL*

<a name='x-28MGL-CG-3A-2ADEFAULT-MAX-N-EVALUATIONS-PER-LINE-SEARCH-2A-20VARIABLE-29'></a>

- [variable] **\*DEFAULT-MAX-N-EVALUATIONS-PER-LINE-SEARCH\*** *20*

<a name='x-28MGL-CG-3A-2ADEFAULT-MAX-N-EVALUATIONS-2A-20VARIABLE-29'></a>

- [variable] **\*DEFAULT-MAX-N-EVALUATIONS\*** *NIL*

<a name='x-28MGL-CG-3ACG-OPTIMIZER-20CLASS-29'></a>

- [class] **CG-OPTIMIZER** *ITERATIVE-OPTIMIZER*

    Updates all weights simultaneously after chewing
    through `BATCH-SIZE`([`0`][dc9d] [`1`][f94f]) inputs.

<a name='x-28MGL-OPT-3AN-INSTANCES-20-28MGL-PAX-3AREADER-20MGL-OPT-3AITERATIVE-OPTIMIZER-29-29'></a>

- [reader] **N-INSTANCES** *ITERATIVE-OPTIMIZER* *(:N-INSTANCES = 0)*

    The number of instances this optimizer has seen so
    far. Incremented automatically during optimization.

<a name='x-28MGL-OPT-3ATERMINATION-20-28MGL-PAX-3AACCESSOR-20MGL-OPT-3AITERATIVE-OPTIMIZER-29-29'></a>

- [accessor] **TERMINATION** *ITERATIVE-OPTIMIZER* *(:TERMINATION = NIL)*

    If a number, it's the number of instances to train
    on in the sense of [`N-INSTANCES`][66a1]. If [`N-INSTANCES`][66a1] is equal or greater
    than this value optimization stops. If [`TERMINATION`][bec0] is `NIL`, then
    optimization will continue. If it is `T`, then optimization will
    stop. If it is a function of no arguments, then its return value
    is processed as if it was returned by [`TERMINATION`][bec0].

<a name='x-28MGL-COMMON-3ABATCH-SIZE-20-28MGL-PAX-3AACCESSOR-20MGL-CG-3ACG-OPTIMIZER-29-29'></a>

- [accessor] **BATCH-SIZE** *CG-OPTIMIZER* *(:BATCH-SIZE)*

    After having gone through `BATCH-SIZE`([`0`][dc9d] [`1`][f94f]) number of
    instances, weights are updated. Normally, [`CG`][f9f7] operates on all
    available data, but it may be useful to introduce some noise into
    the optimization to reduce overfitting by using smaller batch
    sizes. If `BATCH-SIZE`([`0`][dc9d] [`1`][f94f]) is not set, it is initialized to the size of
    the dataset at the start of optimization.

<a name='x-28MGL-CG-3ACG-ARGS-20-28MGL-PAX-3AACCESSOR-20MGL-CG-3ACG-OPTIMIZER-29-29'></a>

- [accessor] **CG-ARGS** *CG-OPTIMIZER* *(:CG-ARGS = 'NIL)*

<a name='x-28MGL-CG-3ASEGMENT-FILTER-20-28MGL-PAX-3AREADER-20MGL-CG-3ACG-OPTIMIZER-29-29'></a>

- [reader] **SEGMENT-FILTER** *CG-OPTIMIZER* *(:SEGMENT-FILTER = (CONSTANTLY T))*

    A predicate function on segments that filters out
    uninteresting segments. Called from [`INITIALIZE-OPTIMIZER*`][4a97].

<a name='x-28MGL-CG-3ADECAYED-CG-OPTIMIZER-MIXIN-20CLASS-29'></a>

- [class] **DECAYED-CG-OPTIMIZER-MIXIN**

    Mix this before a [`CG`][f9f7] based optimizer to conveniently
    add decay on a per-segment basis.

<a name='x-28MGL-CG-3ASEGMENT-DECAY-FN-20-28MGL-PAX-3AACCESSOR-20MGL-CG-3ADECAYED-CG-OPTIMIZER-MIXIN-29-29'></a>

- [accessor] **SEGMENT-DECAY-FN** *DECAYED-CG-OPTIMIZER-MIXIN* *(:SEGMENT-DECAY-FN = NIL)*

    If not `NIL`, it's a designator for a function that
    returns the decay for a given segment. For convenience `NIL` is also
    treated as 0 decay.

<a name='x-28MGL-DIFFUN-3A-40MGL-DIFFUN-20MGL-PAX-3ASECTION-29'></a>

## 7 Differentiable Function

###### \[in package MGL-DIFFUN\]
<a name='x-28MGL-DIFFUN-3ADIFFUN-20CLASS-29'></a>

- [class] **DIFFUN**

    [`DIFFUN`][f4f4] dresses a lisp function (in its [`FN`][434c] slot) as
    a gradient source (see [Implementing Gradient Sources][984f]) which allows it to
    be used in [`MINIMIZE`][bca8]. See the examples in [Gradient Descent][53a7] and
    [Conjugate Gradient][8729].

<a name='x-28MGL-DIFFUN-3AFN-20-28MGL-PAX-3AREADER-20MGL-DIFFUN-3ADIFFUN-29-29'></a>

- [reader] **FN** *DIFFUN* *(:FN)*

    A lisp function. It may have any number of
    parameters.

<a name='x-28MGL-DIFFUN-3APARAMETER-INDICES-20-28MGL-PAX-3AREADER-20MGL-DIFFUN-3ADIFFUN-29-29'></a>

- [reader] **PARAMETER-INDICES** *DIFFUN* *(:PARAMETER-INDICES = NIL)*

    The list of indices of parameters that we don't
    optimize. Values for these will come from the DATASET argument of
    [`MINIMIZE`][bca8].

<a name='x-28MGL-DIFFUN-3AWEIGHT-INDICES-20-28MGL-PAX-3AREADER-20MGL-DIFFUN-3ADIFFUN-29-29'></a>

- [reader] **WEIGHT-INDICES** *DIFFUN* *(:WEIGHT-INDICES = NIL)*

    The list of indices of parameters to be optimized,
    the values of which will come from the `WEIGHTS` argument of
    [`MINIMIZE`][bca8].

<a name='x-28MGL-3A-40MGL-BP-20MGL-PAX-3ASECTION-29'></a>

## 8 Backprogation Neural Networks


<a name='x-28MGL-3A-40MGL-BM-20MGL-PAX-3ASECTION-29'></a>

## 9 Boltzmann Machines


<a name='x-28MGL-3A-40MGL-GP-20MGL-PAX-3ASECTION-29'></a>

## 10 Gaussian Processes


  [026c]: #x-28MGL-3A-40MGL-GP-20MGL-PAX-3ASECTION-29 "(MGL:@MGL-GP MGL-PAX:SECTION)"
  [02de]: #x-28MGL-RESAMPLE-3ASPLIT-FOLD-2FMOD-20FUNCTION-29 "(MGL-RESAMPLE:SPLIT-FOLD/MOD FUNCTION)"
  [0675]: #x-28MGL-RESAMPLE-3A-40MGL-RESAMPLE-BAGGING-20MGL-PAX-3ASECTION-29 "(MGL-RESAMPLE:@MGL-RESAMPLE-BAGGING MGL-PAX:SECTION)"
  [1a5d]: #x-28MGL-DIFFUN-3A-40MGL-DIFFUN-20MGL-PAX-3ASECTION-29 "(MGL-DIFFUN:@MGL-DIFFUN MGL-PAX:SECTION)"
  [1fa8]: #x-28MGL-GD-3APER-WEIGHT-BATCH-GD-OPTIMIZER-20CLASS-29 "(MGL-GD:PER-WEIGHT-BATCH-GD-OPTIMIZER CLASS)"
  [2100]: #x-28MGL-DATASET-3A-40MGL-SAMPLER-FUNCTION-SAMPLER-20MGL-PAX-3ASECTION-29 "(MGL-DATASET:@MGL-SAMPLER-FUNCTION-SAMPLER MGL-PAX:SECTION)"
  [25a8]: #x-28MGL-GD-3A-40MGL-GD-SEGMENTED-GD-OPTIMIZER-20MGL-PAX-3ASECTION-29 "(MGL-GD:@MGL-GD-SEGMENTED-GD-OPTIMIZER MGL-PAX:SECTION)"
  [2730]: #x-28MGL-OPT-3A-40MGL-OPT-EXTENSION-API-20MGL-PAX-3ASECTION-29 "(MGL-OPT:@MGL-OPT-EXTENSION-API MGL-PAX:SECTION)"
  [2b76]: #x-28MGL-RESAMPLE-3AFRACTURE-20FUNCTION-29 "(MGL-RESAMPLE:FRACTURE FUNCTION)"
  [303a]: #x-28MGL-3A-40MGL-TESTS-20MGL-PAX-3ASECTION-29 "(MGL:@MGL-TESTS MGL-PAX:SECTION)"
  [37bf]: #x-28MGL-DATASET-3ASAMPLER-20-28MGL-PAX-3AREADER-20MGL-DATASET-3AFUNCTION-SAMPLER-29-29 "(MGL-DATASET:SAMPLER (MGL-PAX:READER MGL-DATASET:FUNCTION-SAMPLER))"
  [4293]: #x-28MGL-RESAMPLE-3A-40MGL-RESAMPLE-CROSS-VALIDATION-20MGL-PAX-3ASECTION-29 "(MGL-RESAMPLE:@MGL-RESAMPLE-CROSS-VALIDATION MGL-PAX:SECTION)"
  [434c]: #x-28MGL-DIFFUN-3AFN-20-28MGL-PAX-3AREADER-20MGL-DIFFUN-3ADIFFUN-29-29 "(MGL-DIFFUN:FN (MGL-PAX:READER MGL-DIFFUN:DIFFUN))"
  [4a97]: #x-28MGL-OPT-3AINITIALIZE-OPTIMIZER-2A-20GENERIC-FUNCTION-29 "(MGL-OPT:INITIALIZE-OPTIMIZER* GENERIC-FUNCTION)"
  [516d]: #x-28MGL-3A-40MGL-BASIC-CONCEPTS-20MGL-PAX-3ASECTION-29 "(MGL:@MGL-BASIC-CONCEPTS MGL-PAX:SECTION)"
  [53a7]: #x-28MGL-GD-3A-40MGL-GD-20MGL-PAX-3ASECTION-29 "(MGL-GD:@MGL-GD MGL-PAX:SECTION)"
  [5a3f]: #x-28MGL-RESAMPLE-3ASTRATIFY-20FUNCTION-29 "(MGL-RESAMPLE:STRATIFY FUNCTION)"
  [643d]: #x-28MGL-OPT-3ADO-GRADIENT-SINK-20MGL-PAX-3AMACRO-29 "(MGL-OPT:DO-GRADIENT-SINK MGL-PAX:MACRO)"
  [66a1]: #x-28MGL-OPT-3AN-INSTANCES-20-28MGL-PAX-3AREADER-20MGL-OPT-3AITERATIVE-OPTIMIZER-29-29 "(MGL-OPT:N-INSTANCES (MGL-PAX:READER MGL-OPT:ITERATIVE-OPTIMIZER))"
  [6d2c]: #x-28MGL-3A-40MGL-DEPENDENCIES-20MGL-PAX-3ASECTION-29 "(MGL:@MGL-DEPENDENCIES MGL-PAX:SECTION)"
  [6fc3]: #x-28MGL-DATASET-3ASAMPLE-20GENERIC-FUNCTION-29 "(MGL-DATASET:SAMPLE GENERIC-FUNCTION)"
  [72e9]: #x-28MGL-DATASET-3A-40MGL-DATASET-20MGL-PAX-3ASECTION-29 "(MGL-DATASET:@MGL-DATASET MGL-PAX:SECTION)"
  [74a7]: #x-28MGL-3A-40MGL-BP-20MGL-PAX-3ASECTION-29 "(MGL:@MGL-BP MGL-PAX:SECTION)"
  [7540]: #x-28MGL-RESAMPLE-3A-40MGL-RESAMPLE-MISC-20MGL-PAX-3ASECTION-29 "(MGL-RESAMPLE:@MGL-RESAMPLE-MISC MGL-PAX:SECTION)"
  [76b8]: #x-28MGL-RESAMPLE-3ASAMPLE-FROM-20FUNCTION-29 "(MGL-RESAMPLE:SAMPLE-FROM FUNCTION)"
  [794a]: #x-28MGL-OPT-3A-40MGL-OPT-OPTIMIZER-20MGL-PAX-3ASECTION-29 "(MGL-OPT:@MGL-OPT-OPTIMIZER MGL-PAX:SECTION)"
  [7ae7]: #x-28MGL-RESAMPLE-3ASAMPLE-STRATIFIED-20FUNCTION-29 "(MGL-RESAMPLE:SAMPLE-STRATIFIED FUNCTION)"
  [7f6b]: #x-28MGL-CG-3ACG-ARGS-20-28MGL-PAX-3AACCESSOR-20MGL-CG-3ACG-OPTIMIZER-29-29 "(MGL-CG:CG-ARGS (MGL-PAX:ACCESSOR MGL-CG:CG-OPTIMIZER))"
  [8375]: #x-28MGL-RESAMPLE-3ACROSS-VALIDATE-20FUNCTION-29 "(MGL-RESAMPLE:CROSS-VALIDATE FUNCTION)"
  [83bf]: #x-28MGL-OPT-3AITERATIVE-OPTIMIZER-20CLASS-29 "(MGL-OPT:ITERATIVE-OPTIMIZER CLASS)"
  [864e]: #x-28MGL-CG-3ACG-OPTIMIZER-20CLASS-29 "(MGL-CG:CG-OPTIMIZER CLASS)"
  [8665]: #x-28MGL-3A-40MGL-FEATURES-20MGL-PAX-3ASECTION-29 "(MGL:@MGL-FEATURES MGL-PAX:SECTION)"
  [8729]: #x-28MGL-CG-3A-40MGL-CG-20MGL-PAX-3ASECTION-29 "(MGL-CG:@MGL-CG MGL-PAX:SECTION)"
  [8fc3]: #x-28MGL-RESAMPLE-3A-40MGL-RESAMPLE-20MGL-PAX-3ASECTION-29 "(MGL-RESAMPLE:@MGL-RESAMPLE MGL-PAX:SECTION)"
  [94c7]: #x-28MGL-3A-40MGL-BM-20MGL-PAX-3ASECTION-29 "(MGL:@MGL-BM MGL-PAX:SECTION)"
  [9589]: #x-28MGL-RESAMPLE-3ASPLIT-FOLD-2FCONT-20FUNCTION-29 "(MGL-RESAMPLE:SPLIT-FOLD/CONT FUNCTION)"
  [984f]: #x-28MGL-OPT-3A-40MGL-OPT-GRADIENT-SOURCE-20MGL-PAX-3ASECTION-29 "(MGL-OPT:@MGL-OPT-GRADIENT-SOURCE MGL-PAX:SECTION)"
  [9aa2]: #x-28MGL-GD-3ABATCH-GD-OPTIMIZER-20CLASS-29 "(MGL-GD:BATCH-GD-OPTIMIZER CLASS)"
  [9f93]: #x-28MGL-RESAMPLE-3A-40MGL-RESAMPLE-PARTITIONS-20MGL-PAX-3ASECTION-29 "(MGL-RESAMPLE:@MGL-RESAMPLE-PARTITIONS MGL-PAX:SECTION)"
  [a7de]: #x-28MGL-GD-3AWEIGHT-PENALTY-20-28MGL-PAX-3AACCESSOR-20MGL-GD-3A-3AGD-OPTIMIZER-29-29 "(MGL-GD:WEIGHT-PENALTY (MGL-PAX:ACCESSOR MGL-GD::GD-OPTIMIZER))"
  [af7d]: #x-28MGL-DATASET-3A-40MGL-SAMPLER-20MGL-PAX-3ASECTION-29 "(MGL-DATASET:@MGL-SAMPLER MGL-PAX:SECTION)"
  [b6ac]: #x-28MGL-GD-3ASEGMENTER-20-28MGL-PAX-3AREADER-20MGL-GD-3ASEGMENTED-GD-OPTIMIZER-29-29 "(MGL-GD:SEGMENTER (MGL-PAX:READER MGL-GD:SEGMENTED-GD-OPTIMIZER))"
  [b96a]: #x-28MGL-3A-40MGL-BUNDLED-SOFTWARE-20MGL-PAX-3ASECTION-29 "(MGL:@MGL-BUNDLED-SOFTWARE MGL-PAX:SECTION)"
  [bca8]: #x-28MGL-OPT-3AMINIMIZE-20FUNCTION-29 "(MGL-OPT:MINIMIZE FUNCTION)"
  [bec0]: #x-28MGL-OPT-3ATERMINATION-20-28MGL-PAX-3AACCESSOR-20MGL-OPT-3AITERATIVE-OPTIMIZER-29-29 "(MGL-OPT:TERMINATION (MGL-PAX:ACCESSOR MGL-OPT:ITERATIVE-OPTIMIZER))"
  [ca85]: #x-28MGL-RESAMPLE-3A-40MGL-RESAMPLE-CV-BAGGING-20MGL-PAX-3ASECTION-29 "(MGL-RESAMPLE:@MGL-RESAMPLE-CV-BAGGING MGL-PAX:SECTION)"
  [ce14]: #x-28MGL-GD-3AWEIGHT-DECAY-20-28MGL-PAX-3AACCESSOR-20MGL-GD-3A-3AGD-OPTIMIZER-29-29 "(MGL-GD:WEIGHT-DECAY (MGL-PAX:ACCESSOR MGL-GD::GD-OPTIMIZER))"
  [d275]: #x-28MGL-GD-3A-40MGL-GD-PER-WEIGHT-OPTIMIZATION-20MGL-PAX-3ASECTION-29 "(MGL-GD:@MGL-GD-PER-WEIGHT-OPTIMIZATION MGL-PAX:SECTION)"
  [d503]: #x-28MGL-DATASET-3AFINISHEDP-20GENERIC-FUNCTION-29 "(MGL-DATASET:FINISHEDP GENERIC-FUNCTION)"
  [dc9d]: #x-28MGL-COMMON-3ABATCH-SIZE-20-28MGL-PAX-3AACCESSOR-20MGL-CG-3ACG-OPTIMIZER-29-29 "(MGL-COMMON:BATCH-SIZE (MGL-PAX:ACCESSOR MGL-CG:CG-OPTIMIZER))"
  [df57]: #x-28MGL-GD-3A-40MGL-GD-BATCH-GD-OPTIMIZER-20MGL-PAX-3ASECTION-29 "(MGL-GD:@MGL-GD-BATCH-GD-OPTIMIZER MGL-PAX:SECTION)"
  [e0c8]: #x-28MGL-GD-3AMOMENTUM-TYPE-20-28MGL-PAX-3AREADER-20MGL-GD-3A-3AGD-OPTIMIZER-29-29 "(MGL-GD:MOMENTUM-TYPE (MGL-PAX:READER MGL-GD::GD-OPTIMIZER))"
  [e0d7]: #x-28-22mgl-22-20ASDF-2FSYSTEM-3ASYSTEM-29 "(\"mgl\" ASDF/SYSTEM:SYSTEM)"
  [e57e]: #x-28MGL-RESAMPLE-3AFRACTURE-STRATIFIED-20FUNCTION-29 "(MGL-RESAMPLE:FRACTURE-STRATIFIED FUNCTION)"
  [ed3d]: #x-28MGL-GD-3AMOMENTUM-20-28MGL-PAX-3AACCESSOR-20MGL-GD-3A-3AGD-OPTIMIZER-29-29 "(MGL-GD:MOMENTUM (MGL-PAX:ACCESSOR MGL-GD::GD-OPTIMIZER))"
  [edd9]: #x-28MGL-RESAMPLE-3ASPLIT-STRATIFIED-20FUNCTION-29 "(MGL-RESAMPLE:SPLIT-STRATIFIED FUNCTION)"
  [f18a]: #x-28MGL-OPT-3A-40MGL-OPT-GRADIENT-SINK-20MGL-PAX-3ASECTION-29 "(MGL-OPT:@MGL-OPT-GRADIENT-SINK MGL-PAX:SECTION)"
  [f4f4]: #x-28MGL-DIFFUN-3ADIFFUN-20CLASS-29 "(MGL-DIFFUN:DIFFUN CLASS)"
  [f56b]: #x-28MGL-DATASET-3AMAX-N-SAMPLES-20-28MGL-PAX-3AACCESSOR-20MGL-DATASET-3AFUNCTION-SAMPLER-29-29 "(MGL-DATASET:MAX-N-SAMPLES (MGL-PAX:ACCESSOR MGL-DATASET:FUNCTION-SAMPLER))"
  [f805]: #x-28MGL-OPT-3A-40MGL-OPT-ITERATIVE-OPTIMIZER-20MGL-PAX-3ASECTION-29 "(MGL-OPT:@MGL-OPT-ITERATIVE-OPTIMIZER MGL-PAX:SECTION)"
  [f94f]: #x-28MGL-COMMON-3ABATCH-SIZE-20-28MGL-PAX-3AACCESSOR-20MGL-GD-3A-3AGD-OPTIMIZER-29-29 "(MGL-COMMON:BATCH-SIZE (MGL-PAX:ACCESSOR MGL-GD::GD-OPTIMIZER))"
  [f995]: #x-28MGL-3A-40MGL-OVERVIEW-20MGL-PAX-3ASECTION-29 "(MGL:@MGL-OVERVIEW MGL-PAX:SECTION)"
  [f9f7]: #x-28MGL-CG-3ACG-20FUNCTION-29 "(MGL-CG:CG FUNCTION)"
  [fd45]: #x-28MGL-DATASET-3AN-SAMPLES-20-28MGL-PAX-3AREADER-20MGL-DATASET-3AFUNCTION-SAMPLER-29-29 "(MGL-DATASET:N-SAMPLES (MGL-PAX:READER MGL-DATASET:FUNCTION-SAMPLER))"
  [fe97]: #x-28MGL-OPT-3A-40MGL-OPT-20MGL-PAX-3ASECTION-29 "(MGL-OPT:@MGL-OPT MGL-PAX:SECTION)"

* * *
###### \[generated by [MGL-PAX](https://github.com/melisgl/mgl-pax)\]
