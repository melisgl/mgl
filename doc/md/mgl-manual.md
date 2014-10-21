<a name='x-28MGL-3A-40MGL-MANUAL-20MGL-PAX-3ASECTION-29'></a>

# MGL Manual

## Table of Contents

- [1 mgl ASDF System Details][e0d7]
- [2 Overview][f995]
    - [2.1 Dependencies][6d2c]
    - [2.2 Code Organization][45db]
    - [2.3 Glossary][0ab9]
- [3 Datasets][72e9]
    - [3.1 Samplers][af7d]
        - [3.1.1 Function Sampler][2100]
- [4 Resampling][8fc3]
    - [4.1 Partitions][9f93]
    - [4.2 Cross-validation][4293]
    - [4.3 Bagging][0675]
    - [4.4 CV Bagging][ca85]
    - [4.5 Miscellaneous Operations][7540]
- [5 Models][8b7f]
    - [5.1 Model Persistence][56ee]
    - [5.2 Batch Processing][0552]
    - [5.3 Executors][6e12]
        - [5.3.1 Parameterized Executor Cache][1426]
- [6 Monitoring][0924]
    - [6.1 Monitors][6e54]
    - [6.2 Measurers][2364]
    - [6.3 Counters][998f]
        - [6.3.1 Attributes][d011]
        - [6.3.2 Counter classes][8966]
- [7 Classification][c1b6]
    - [7.1 Classification Monitors][cc50]
    - [7.2 Classification Measurers][505e]
    - [7.3 Classification Counters][32b3]
        - [7.3.1 Confusion Matrices][1541]
- [8 Gradient Based Optimization][fe97]
    - [8.1 Iterative Optimizer][f805]
    - [8.2 Gradient Descent][53a7]
        - [8.2.1 Batch GD Optimizer][df57]
        - [8.2.2 Segmented GD Optimizer][25a8]
        - [8.2.3 Per-weight Optimization][d275]
    - [8.3 Conjugate Gradient][8729]
    - [8.4 Extension API][2730]
        - [8.4.1 Implementing Optimizers][794a]
        - [8.4.2 Implementing Gradient Sources][984f]
        - [8.4.3 Implementing Gradient Sinks][f18a]
- [9 Differentiable Functions][1a5d]
- [10 Backprogation Neural Networks][74a7]
- [11 Boltzmann Machines][94c7]
- [12 Gaussian Processes][026c]

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

- Gradient descent optimization

    - Nesterov momentum

- Conjugate gradient optimization

- Backpropagation networks (BPN)

    - Dropout

    - Rectified linear units

    - Maxout

    - Max-channel

- Boltzmann Machines

    - Restricted Boltzmann Machines (RBM)

    - Contrastive Divergence (CD) learning

    - Deep Belief Networks (DBN)

    - Semi Restricted Boltzmann Machines

    - Deep Boltzmann Machines

    - Persistent Contrastive Divergence (PCD) learning

    - Unrolling DBN or a DBM to a BPN

- Gaussian Processes

    - Optimizing Gaussian Processes as BPNs

In general, the focus is on power and performance not on ease of
use. Perhaps one day there will be a cookie cutter interface with
restricted functionality if a reasonable compromise is found between
power and utility.

<a name='x-28MGL-3A-40MGL-DEPENDENCIES-20MGL-PAX-3ASECTION-29'></a>

### 2.1 Dependencies

MGL used to rely on [LLA](https://github.com/tpapp/lla) to
interface to BLAS and LAPACK. That's mostly history by now, but
configuration of foreign libraries is still done via `LLA`. See the
README in `LLA` on how to set things up. Note that these days OpenBLAS
is easier to set up and just as fast as ATLAS.

[CL-CUDA](https://github.com/takagi/cl-cuda) is a dependency for
which the NVIDIA CUDA Toolkit needs to be installed, but MGL is
fully functional even if there is no cuda capable gpu installed. See
the `MGL-MAT:WITH-CUDA*` macro for how to use it.

<a name='x-28MGL-3A-40MGL-CODE-ORGANIZATION-20MGL-PAX-3ASECTION-29'></a>

### 2.2 Code Organization

MGL consists of several packages dedicated to different tasks.
For example, package `MGL-RESAMPLE` is about [Resampling][8fc3] and
`MGL-GD` is about [Gradient Descent][53a7] and so on. On one hand, having many
packages makes it easier to cleanly separate API and implementation
and also to explore into a specific task. At other times, they can
be a hassle, so the [`MGL`][e0d7] package itself reexports every external
symbol found in all the other packages that make up MGL.

One exception to this rule is the bundled, but independent
`MGL-GNUPLOT` library.

The built in tests can be run with:

    (ASDF:OOS 'ASDF:TEST-OP '#:MGL)

Note, that most of the tests are rather stochastic and can fail once
in a while.

<a name='x-28MGL-3A-40MGL-GLOSSARY-20MGL-PAX-3ASECTION-29'></a>

### 2.3 Glossary

Ultimately machine learning is about creating **models** of some
domain. The observations in the modelled domain are called
**instances** (also known as examples or samples). Sets of instances
are called **datasets**. Datasets are used when fitting a model or
when making **predictions**. Sometimes the word predictions is too
specific, and the results obtained from applying a model to some
instances are simply called **results**.

<a name='x-28MGL-DATASET-3A-40MGL-DATASET-20MGL-PAX-3ASECTION-29'></a>

## 3 Datasets

###### \[in package MGL-DATASET\]
An instance can often be any kind of object of the user's choice.
It is typically represented by a set of numbers which is called a
feature vector or by a structure holding the feature vector, the
label, etc. A dataset is a `SEQUENCE` of such instances or a
[Samplers][af7d] object that produces instances.

<a name='x-28MGL-DATASET-3A-40MGL-SAMPLER-20MGL-PAX-3ASECTION-29'></a>

### 3.1 Samplers

Some algorithms do not need random access to the entire dataset and
can work with a stream observations. Samplers are simple generators
providing two functions: [`SAMPLE`][6fc3] and [`FINISHEDP`][d503].

<a name='x-28MGL-DATASET-3ASAMPLE-20GENERIC-FUNCTION-29'></a>

- [generic-function] **SAMPLE** *SAMPLER*

    If `SAMPLER` has not run out of data (see [`FINISHEDP`][d503])
    [`SAMPLE`][6fc3] returns an object that represents a sample from the world to
    be experienced or, in other words, simply something the can be used
    as input for training or prediction. It is not allowed to call
    [`SAMPLE`][6fc3] if `SAMPLER` is [`FINISHEDP`][d503].

<a name='x-28MGL-DATASET-3AFINISHEDP-20GENERIC-FUNCTION-29'></a>

- [generic-function] **FINISHEDP** *SAMPLER*

    See if `SAMPLER` has run out of examples.

<a name='x-28MGL-DATASET-3ALIST-SAMPLES-20FUNCTION-29'></a>

- [function] **LIST-SAMPLES** *SAMPLER MAX-SIZE*

    Return a list of samples of length at most `MAX-SIZE` or less if
    `SAMPLER` runs out.

<a name='x-28MGL-DATASET-3AMAKE-SEQUENCE-SAMPLER-20FUNCTION-29'></a>

- [function] **MAKE-SEQUENCE-SAMPLER** *SEQ &KEY MAX-N-SAMPLES*

    Create a sampler that returns elements of `SEQ` in their original
    order. If `MAX-N-SAMPLES` is non-nil, then at most `MAX-N-SAMPLES` are
    sampled.

<a name='x-28MGL-DATASET-3AMAKE-RANDOM-SAMPLER-20FUNCTION-29'></a>

- [function] **MAKE-RANDOM-SAMPLER** *SEQ &KEY MAX-N-SAMPLES (REORDER #'MGL-RESAMPLE:SHUFFLE)*

    Create a sampler that returns elements of `SEQ` in random order. If
    `MAX-N-SAMPLES` is non-nil, then at most `MAX-N-SAMPLES` are sampled.
    The first pass over a shuffled copy of `SEQ`, and this copy is
    reshuffled whenever the sampler reaches the end of it. Shuffling is
    performed by calling the `REORDER` function.

<a name='x-28MGL-DATASET-3A-2AINFINITELY-EMPTY-DATASET-2A-20VARIABLE-29'></a>

- [variable] **\*INFINITELY-EMPTY-DATASET\*** *#\<FUNCTION-SAMPLER "infinitely empty" \>*

    This is the default dataset for [`MGL-OPT:MINIMIZE`][bca8]. It's an infinite
    stream of NILs.

<a name='x-28MGL-DATASET-3A-40MGL-SAMPLER-FUNCTION-SAMPLER-20MGL-PAX-3ASECTION-29'></a>

#### 3.1.1 Function Sampler

<a name='x-28MGL-DATASET-3AFUNCTION-SAMPLER-20CLASS-29'></a>

- [class] **FUNCTION-SAMPLER**

    A sampler with a function in its [`GENERATOR`][8521] that
    produces a stream of samples which may or may not be finite
    depending on [`MAX-N-SAMPLES`][f56b]. [`FINISHEDP`][d503] returns `T` iff [`MAX-N-SAMPLES`][f56b] is
    non-nil, and it's not greater than the number of samples
    generated ([`N-SAMPLES`][fd45]).
    
        (list-samples (make-instance 'function-sampler
                                     :generator (lambda ()
                                                  (random 10))
                                     :max-n-samples 5)
                      10)
        => (3 5 2 3 3)


<a name='x-28MGL-DATASET-3AGENERATOR-20-28MGL-PAX-3AREADER-20MGL-DATASET-3AFUNCTION-SAMPLER-29-29'></a>

- [reader] **GENERATOR** *FUNCTION-SAMPLER* *(:GENERATOR)*

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

## 4 Resampling

###### \[in package MGL-RESAMPLE\]
The focus of this package is on resampling methods such as
cross-validation and bagging which can be used for model evaluation,
model selection, and also as a simple form of ensembling. Data
partitioning and sampling functions are also provided because they
tend to be used together with resampling.

<a name='x-28MGL-RESAMPLE-3A-40MGL-RESAMPLE-PARTITIONS-20MGL-PAX-3ASECTION-29'></a>

### 4.1 Partitions

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
    
    ```cl-transcript
    (fracture 5 '(0 1 2 3 4 5 6 7 8 9))
    => ((0 1) (2 3) (4 5) (6 7) (8 9))
    
    ```
    
    To split into two sequences whose lengths are proportional to 2 and
    3:
    
    ```cl-transcript
    (fracture '(2 3) '(0 1 2 3 4 5 6 7 8 9))
    => ((0 1 2 3) (4 5 6 7 8 9))
    
    ```


<a name='x-28MGL-RESAMPLE-3ASTRATIFY-20FUNCTION-29'></a>

- [function] **STRATIFY** *SEQ &KEY (KEY #'IDENTITY) (TEST #'EQL)*

    Return the list of strata of `SEQ`. `SEQ` is a sequence of elements for
    which the function `KEY` returns the class they belong to. Such
    classes are opaque objects compared for equality with `TEST`. A
    stratum is a sequence of elements with the same (under `TEST`) `KEY`.
    
    ```cl-transcript
    (stratify '(0 1 2 3 4 5 6 7 8 9) :key #'evenp)
    => ((0 2 4 6 8) (1 3 5 7 9))
    
    ```


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
    
    ```cl-transcript
    (fracture-stratified 2 '(0 1 2 3 4 5 6 7 8 9) :key #'evenp)
    => ((0 2 1 3) (4 6 8 5 7 9))
    
    ```


<a name='x-28MGL-RESAMPLE-3A-40MGL-RESAMPLE-CROSS-VALIDATION-20MGL-PAX-3ASECTION-29'></a>

### 4.2 Cross-validation

<a name='x-28MGL-RESAMPLE-3ACROSS-VALIDATE-20FUNCTION-29'></a>

- [function] **CROSS-VALIDATE** *DATA FN &KEY (N-FOLDS 5) (FOLDS (ALEXANDRIA.0.DEV:IOTA N-FOLDS)) (SPLIT-FN #'SPLIT-FOLD/MOD) PASS-FOLD*

    Map `FN` over the `FOLDS` of `DATA` split with `SPLIT-FN` and collect the
    results in a list. The simplest demonstration is:
    
    ```cl-transcript
    (cross-validate '(0 1 2 3 4)
                    (lambda (test training)
                     (list test training))
                    :n-folds 5)
    => (((0) (1 2 3 4))
        ((1) (0 2 3 4))
        ((2) (0 1 3 4))
        ((3) (0 1 2 4))
        ((4) (0 1 2 3)))
    
    ```
    
    Of course, in practice one would typically train a model and return
    the trained model and/or its score on `TEST`. Also, sometimes one may
    want to do only some of the folds and remember which ones they were:
    
    ```cl-transcript
    (cross-validate '(0 1 2 3 4)
                    (lambda (fold test training)
                     (list :fold fold test training))
                    :folds '(2 3)
                    :pass-fold t)
    => ((:fold 2 (2) (0 1 3 4))
        (:fold 3 (3) (0 1 2 4)))
    
    ```
    
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

### 4.3 Bagging

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
    
    ```common-lisp
    (sample-from 1/2 '(0 1 2 3 4 5))
    => (5 3 2)
    ```
    
    To randomly select some elements such that the sum of their weights
    constitute about half of the sum of weights across the whole
    sequence:
    
    ```common-lisp
    (sample-from 1/2 '(0 1 2 3 4 5 6 7 8 9) :weight #'identity)
    => ;; sums to 28 that's near 45/2
       (9 4 1 6 8)
    ```
    
    To sample with replacement (that is, allowing the element to be
    sampled multiple times):
    
    ```common-lisp
    (sample-from 1 '(0 1 2 3 4 5) :replacement t)
    => (1 1 5 1 4 4)
    ```


<a name='x-28MGL-RESAMPLE-3ASAMPLE-STRATIFIED-20FUNCTION-29'></a>

- [function] **SAMPLE-STRATIFIED** *RATIO SEQ &KEY WEIGHT REPLACEMENT (KEY #'IDENTITY) (TEST #'EQL) (RANDOM-STATE \*RANDOM-STATE\*)*

    Like [`SAMPLE-FROM`][76b8] but makes sure that the weighted proportion of
    classes in the result is approximately the same as the proportion in
    `SEQ`. See [`STRATIFY`][5a3f] for the description of `KEY` and `TEST`.

<a name='x-28MGL-RESAMPLE-3A-40MGL-RESAMPLE-CV-BAGGING-20MGL-PAX-3ASECTION-29'></a>

### 4.4 CV Bagging

<a name='x-28MGL-RESAMPLE-3ABAG-CV-20FUNCTION-29'></a>

- [function] **BAG-CV** *DATA FN &KEY N (N-FOLDS 5) (FOLDS (ALEXANDRIA.0.DEV:IOTA N-FOLDS)) (SPLIT-FN #'SPLIT-FOLD/MOD) PASS-FOLD (RANDOM-STATE \*RANDOM-STATE\*)*

    Perform cross-validation on different shuffles of `DATA` `N` times and
    collect the results. Since [`CROSS-VALIDATE`][8375] collects the return values
    of `FN`, the return value of this function is a list of lists of `FN`
    results. If `N` is `NIL`, don't collect anything just keep doing
    repeated CVs until `FN` performs an non-local exit.
    
    The following example simply collects the test and training sets for
    2-fold CV repeated 3 times with shuffled data:
    
    `cl-transcript
     (bag-cv '(0 1 2 3 4) #'list :n 3 :n-folds 2)
     => ((((2 3 4) (1 0))
          ((1 0) (2 3 4)))
         (((2 1 0) (4 3))
          ((4 3) (2 1 0)))
         (((1 0 3) (2 4))
          ((2 4) (1 0 3))))
    `
    
    CV bagging is useful when a single CV is not producing stable
    results. As an ensemble method, CV bagging has the advantage over
    bagging that each example will occur the same number of times and
    after the first CV is complete there is a complete but less reliable
    estimate for each example which gets refined by further CVs.

<a name='x-28MGL-RESAMPLE-3A-40MGL-RESAMPLE-MISC-20MGL-PAX-3ASECTION-29'></a>

### 4.5 Miscellaneous Operations

<a name='x-28MGL-RESAMPLE-3ASPREAD-STRATA-20FUNCTION-29'></a>

- [function] **SPREAD-STRATA** *SEQ &KEY (KEY #'IDENTITY) (TEST #'EQL)*

    Return a sequence that's a reordering of `SEQ` such that elements
    belonging to different strata (under `KEY` and `TEST`, see [`STRATIFY`][5a3f]) are
    distributed evenly. The order of elements belonging to the same
    stratum is unchanged.
    
    For example, to make sure that even and odd numbers are distributed
    evenly:
    
    ```cl-transcript
    (spread-strata '(0 2 4 6 8 1 3 5 7 9) :key #'evenp)
    => (0 1 2 3 4 5 6 7 8 9)
    
    ```
    
    Same thing with unbalanced classes:
    
    ```cl-transcript
    (spread-strata (vector 0 2 3 5 6 1 4)
                   :key (lambda (x)
                          (if (member x '(1 4))
                              t
                              nil)))
    => #(0 1 2 3 4 5 6)
    
    ```


<a name='x-28MGL-RESAMPLE-3AZIP-EVENLY-20FUNCTION-29'></a>

- [function] **ZIP-EVENLY** *SEQS &KEY RESULT-TYPE*

    Make a single sequence out of the sequences in `SEQS` so that in the
    returned sequence indices of elements belonging to the same source
    sequence are spread evenly across the whole range. The result is a
    list is `RESULT-TYPE` is `LIST`, it's a vector if `RESULT-TYPE` is `VECTOR`.
    If `RESULT-TYPE` is `NIL`, then it's determined by the type of the first
    sequence in `SEQS`.
    
    ```cl-transcript
    (zip-evenly '((0 2 4) (1 3)))
    => (0 1 2 3 4)
    
    ```


<a name='x-28MGL-CORE-3A-40MGL-MODEL-20MGL-PAX-3ASECTION-29'></a>

## 5 Models

###### \[in package MGL-CORE\]
<a name='x-28MGL-CORE-3A-40MGL-MODEL-PERSISTENCE-20MGL-PAX-3ASECTION-29'></a>

### 5.1 Model Persistence

<a name='x-28MGL-CORE-3AREAD-WEIGHTS-20GENERIC-FUNCTION-29'></a>

- [generic-function] **READ-WEIGHTS** *MODEL STREAM*

    Read the weights of `MODEL` from the bivalent `STREAM`
    where weights mean the learnt parameters. There is currently no
    sanity checking of data which will most certainly change in the
    future together with the serialization format.

<a name='x-28MGL-CORE-3AWRITE-WEIGHTS-20GENERIC-FUNCTION-29'></a>

- [generic-function] **WRITE-WEIGHTS** *MODEL STREAM*

    Write weight of `MODEL` to the bivalent `STREAM`.

<a name='x-28MGL-CORE-3ALOAD-WEIGHTS-20FUNCTION-29'></a>

- [function] **LOAD-WEIGHTS** *FILENAME MODEL*

    Load weights of `MODEL` from `FILENAME`.

<a name='x-28MGL-CORE-3ASAVE-WEIGHTS-20FUNCTION-29'></a>

- [function] **SAVE-WEIGHTS** *FILENAME MODEL &KEY (IF-EXISTS :ERROR) (ENSURE T)*

    Save weights of `MODEL` to `FILENAME`. If `ENSURE`, then
    `ENSURE-DIRECTORIES-EXIST` is called on `FILENAME`. `IF-EXISTS` is passed
    on to `OPEN`.

<a name='x-28MGL-CORE-3A-40MGL-MODEL-STRIPE-20MGL-PAX-3ASECTION-29'></a>

### 5.2 Batch Processing

Processing instances one by one during training or prediction can
be slow. The models that support batch processing for greater
efficiency are said to be *striped*.

Typically, during or after creating a model, one sets [`MAX-N-STRIPES`][9598]
on it a positive integer. When a batch of instances is to be fed to
the model it is first broken into subbatches of length that's at
most [`MAX-N-STRIPES`][9598]. For each subbatch, [`SET-INPUT`][8795] (FIXDOC) is called
and a before method takes care of setting [`N-STRIPES`][dca7] to the actual
number of instances in the subbatch. When [`MAX-N-STRIPES`][9598] is set
internal data structures may be resized which is an expensive
operation. Setting [`N-STRIPES`][dca7] is a comparatively cheap operation,
often implemented as matrix reshaping.

Note that for models made of different parts (for example,
[MGL-BP:BPN][CLASS] consists of [MGL-BP:LUMP][]s) , setting these
values affects the constituent parts, but one should never change
the number stripes of the parts directly because that would lead to
an internal inconsistency in the model.

<a name='x-28MGL-CORE-3AMAX-N-STRIPES-20GENERIC-FUNCTION-29'></a>

- [generic-function] **MAX-N-STRIPES** *OBJECT*

    The number of stripes with which the `OBJECT` is
    capable of dealing simultaneously. 

<a name='x-28MGL-CORE-3ASET-MAX-N-STRIPES-20GENERIC-FUNCTION-29'></a>

- [generic-function] **SET-MAX-N-STRIPES** *MAX-N-STRIPES OBJECT*

    Allocate the necessary stuff to allow for
    `MAX-N-STRIPES` number of stripes to be worked with simultaneously in
    `OBJECT`. This is called when `MAX-N-STRIPES` is `SETF`'ed.

<a name='x-28MGL-CORE-3AN-STRIPES-20GENERIC-FUNCTION-29'></a>

- [generic-function] **N-STRIPES** *OBJECT*

    The number of stripes currently present in `OBJECT`.
    This is at most [`MAX-N-STRIPES`][9598].

<a name='x-28MGL-CORE-3ASET-N-STRIPES-20GENERIC-FUNCTION-29'></a>

- [generic-function] **SET-N-STRIPES** *N-STRIPES OBJECT*

    Set the number of stripes (out of [`MAX-N-STRIPES`][9598])
    that are in use in `OBJECT`. This is called when `N-STRIPES` is
    `SETF`'ed.

<a name='x-28MGL-CORE-3AWITH-STRIPES-20MGL-PAX-3AMACRO-29'></a>

- [macro] **WITH-STRIPES** *SPECS &BODY BODY*

    Bind start and optionally end indices belonging to stripes in
    striped objects.
    
        (WITH-STRIPES ((STRIPE1 OBJECT1 START1 END1)
                       (STRIPE2 OBJECT2 START2)
                       ...)
         ...)
    
    This is how one's supposed to find the index range corresponding to
    the Nth input in an input lump of a bpn:
    
         (with-stripes ((n input-lump start end))
           (loop for i upfrom start below end
                 do (setf (mref (nodes input-lump) i) 0d0)))
    
    Note how the input lump is striped, but the matrix into which we are
    indexing (`NODES`) is not known to [`WITH-STRIPES`][3ed1]. In fact, for lumps
    the same stripe indices work with `NODES` and `MGL-BP:DERIVATIVES`.

<a name='x-28MGL-CORE-3ASTRIPE-START-20GENERIC-FUNCTION-29'></a>

- [generic-function] **STRIPE-START** *STRIPE OBJECT*

    Return the start index of `STRIPE` in some array or
    matrix of `OBJECT`.

<a name='x-28MGL-CORE-3ASTRIPE-END-20GENERIC-FUNCTION-29'></a>

- [generic-function] **STRIPE-END** *STRIPE OBJECT*

    Return the end index (exclusive) of `STRIPE` in some
    array or matrix of `OBJECT`.

<a name='x-28MGL-CORE-3ASET-INPUT-20GENERIC-FUNCTION-29'></a>

- [generic-function] **SET-INPUT** *INSTANCES MODEL*

    Set `INSTANCES` as inputs in `MODEL`. SAMPLES is always
    a `SEQUENCE` of instances even for models not capable of batch
    operation. It sets [`N-STRIPES`][dca7] to (`LENGTH` `INSTANCES`) in a `:BEFORE`
    method.

<a name='x-28MGL-CORE-3AMAP-BATCHES-FOR-MODEL-20FUNCTION-29'></a>

- [function] **MAP-BATCHES-FOR-MODEL** *FN DATASET MODEL*

    Call `FN` with batches of instances from `DATASET` suitable for `MODEL`.
    The number of instances in a batch is [`MAX-N-STRIPES`][9598] of `MODEL` or less
    if there are no more instances left.

<a name='x-28MGL-CORE-3ADO-BATCHES-FOR-MODEL-20MGL-PAX-3AMACRO-29'></a>

- [macro] **DO-BATCHES-FOR-MODEL** *(BATCH (DATASET MODEL)) &BODY BODY*

    Convenience macro over [`MAP-BATCHES-FOR-MODEL`][fdf3].

<a name='x-28MGL-CORE-3A-40MGL-EXECUTORS-20MGL-PAX-3ASECTION-29'></a>

### 5.3 Executors

<a name='x-28MGL-CORE-3AMAP-OVER-EXECUTORS-20GENERIC-FUNCTION-29'></a>

- [generic-function] **MAP-OVER-EXECUTORS** *FN INSTANCES PROTOTYPE-EXECUTOR*

    Divide `INSTANCES` between executors that perform the
    same function as `PROTOTYPE-EXECUTOR` and call `FN` with the instances
    and the executor for which the instances are.
    
    Some objects conflate function and call: the forward pass of a
    [MGL-BP:BPN][class] computes output from inputs so it is like a
    function but it also doubles as a function call in the sense that
    the bpn (function) object changes state during the computation of
    the output. Hence not even the forward pass of a bpn is thread safe.
    There is also the restriction that all inputs must be of the same
    size.
    
    For example, if we have a function that builds bpn a for an input of
    a certain size, then we can create a factory that creates bpns for a
    particular call. The factory probably wants keep the weights the
    same though. In [Parameterized Executor Cache][1426],
    [`MAKE-EXECUTOR-WITH-PARAMETERS`][b73e] is this factory.
    
    Parallelization of execution is another possibility
    [`MAP-OVER-EXECUTORS`][c27a] allows, but there is no prebuilt solution for it,
    yet.
    
    The default implementation simply calls `FN` with `INSTANCES` and
    `PROTOTYPE-EXECUTOR`.

<a name='x-28MGL-CORE-3ADO-EXECUTORS-20MGL-PAX-3AMACRO-29'></a>

- [macro] **DO-EXECUTORS** *(INSTANCES OBJECT) &BODY BODY*

    Convenience macro on top of [`MAP-OVER-EXECUTORS`][c27a].

<a name='x-28MGL-CORE-3A-40MGL-PARAMETERIZED-EXECUTOR-CACHE-20MGL-PAX-3ASECTION-29'></a>

#### 5.3.1 Parameterized Executor Cache

<a name='x-28MGL-CORE-3APARAMETERIZED-EXECUTOR-CACHE-MIXIN-20CLASS-29'></a>

- [class] **PARAMETERIZED-EXECUTOR-CACHE-MIXIN**

    Mix this into a model, implement
    [`INSTANCE-TO-EXECUTOR-PARAMETERS`][b8b6] and [`MAKE-EXECUTOR-WITH-PARAMETERS`][b73e]
    and [`DO-EXECUTORS`][2cc2] will be to able build executors suitable for
    different instances. The canonical example is using a BPN to compute
    the means and convariances of a gaussian process. Since each
    instance is made of a variable number of observations, the size of
    the input is not constant, thus we have a bpn (an executor) for each
    input dimension (the parameters).

<a name='x-28MGL-CORE-3AMAKE-EXECUTOR-WITH-PARAMETERS-20GENERIC-FUNCTION-29'></a>

- [generic-function] **MAKE-EXECUTOR-WITH-PARAMETERS** *PARAMETERS CACHE*

    Create a new executor for `PARAMETERS`. `CACHE` is a
    [`PARAMETERIZED-EXECUTOR-CACHE-MIXIN`][d74b]. In the BPN gaussian process
    example, `PARAMETERS` would be a list of input dimensions.

<a name='x-28MGL-CORE-3AINSTANCE-TO-EXECUTOR-PARAMETERS-20GENERIC-FUNCTION-29'></a>

- [generic-function] **INSTANCE-TO-EXECUTOR-PARAMETERS** *INSTANCE CACHE*

    Return the parameters for an executor able to
    handle `INSTANCE`. Called by [`MAP-OVER-EXECUTORS`][c27a] on `CACHE` (that's a
    [`PARAMETERIZED-EXECUTOR-CACHE-MIXIN`][d74b]). The returned parameters are
    keys in an `EQUAL` parameters->executor hash table.

<a name='x-28MGL-CORE-3A-40MGL-MONITORING-20MGL-PAX-3ASECTION-29'></a>

## 6 Monitoring

###### \[in package MGL-CORE\]
When training or applying a model, one often wants to track various
statistics. For example, in the case of training a neural network
with cross-entropy loss, these statistics could be the average
cross-entropy loss itself, classification accuracy, or even the
entire confusion matrix and sparsity levels in hidden layers. Also,
there is the question of what to do with the measured values (log
and forget, add to some counter or a list).

So there may be several phases of operation when we want to keep an
eye on. Let's call these **events**. There can also be many fairly
independent things to do in response to an event. Let's call these
**monitors**. Some monitors are a composition of two operations: one
that extracts some measurements and another that aggregates those
measurements. Let's call these two **measurers** and **counters**,
respectively.

For example, consider training a backpropagation neural network. We
want to look at the state of of network just after the backward
pass. `BP-LEARNER` has a [MONITORS][(accessor bp-learner)] event hook
corresponding to the moment after backpropagating the gradients.
Suppose we are interested in how the training cost evolves:

    (push (make-instance 'monitor
                         :measurer (lambda (instances bpn)
                                     (declare (ignore instances))
                                     (mgl-bp:cost bpn))
                         :counter (make-instance 'basic-counter))
          (monitors learner))

During training, this monitor will track the cost of training
examples behind the scenes. If we want to print and reset this
monitor periodically we can put another monitor on
[`MGL-OPT:ITERATIVE-OPTIMIZER`][83bf]'s [`MGL-OPT:ON-N-INSTANCES-CHANGED`][9cdc]
accessor:

    (push (lambda (optimizer gradient-source n-instances)
            (declare (ignore optimizer))
            (when (zerop (mod n-instances 1000))
              (format t "n-instances: ~S~%" n-instances)
              (dolist (monitor (monitors gradient-source))
                (when (counter monitor)
                  (format t "~A~%" (counter monitor))
                  (reset-counter (counter monitor)))))
          (mgl-opt:on-n-instances-changed optimizer))

Note that the monitor we push can be anything as long as
[`APPLY-MONITOR`][f95f] is implemented on it with the appropriate signature.
Also note that the `ZEROP` + `MOD` logic is fragile, so you will likely
want to use [`MGL-OPT:MONITOR-OPTIMIZATION-PERIODICALLY`][918e] instead of
doing the above.

So that's the general idea. Concrete events are documented where
they are signalled. Often there are task specific utilities that
create a reasonable set of default monitors (see
[Classification Monitors][cc50]).

<a name='x-28MGL-CORE-3AAPPLY-MONITORS-20FUNCTION-29'></a>

- [function] **APPLY-MONITORS** *MONITORS &REST ARGUMENTS*

    Call [`APPLY-MONITOR`][f95f] on each monitor in `MONITORS` and `ARGUMENTS`. This
    qqqqis how an event is fired.

<a name='x-28MGL-CORE-3AAPPLY-MONITOR-20GENERIC-FUNCTION-29'></a>

- [generic-function] **APPLY-MONITOR** *MONITOR &REST ARGUMENTS*

    Apply `MONITOR` to `ARGUMENTS`. This sound fairly
    generic, because it is. `MONITOR` can be anything, even a simple
    function or symbol, in which case this is just `CL:APPLY`. See
    [Monitors][6e54] for more.

<a name='x-28MGL-CORE-3ACOUNTER-20GENERIC-FUNCTION-29'></a>

- [generic-function] **COUNTER** *MONITOR*

    Return an object representing the state of `MONITOR`
    or `NIL`, if it doesn't have any (say because it's a simple logging
    function). Most monitors have counters into which they accumulate
    results until they are printed and reset. See [Counters][998f] for
    more.

<a name='x-28MGL-CORE-3AMONITOR-MODEL-RESULTS-20FUNCTION-29'></a>

- [function] **MONITOR-MODEL-RESULTS** *FN DATASET MODEL MONITORS*

    Call `FN` with batches of instances from `DATASET` until it runs
    out (as in [`DO-BATCHES-FOR-MODEL`][39c1]). `FN` is supposed to apply `MODEL` to
    the batch and return some kind of result (for neural networks, the
    result is the model state itself). Apply `MONITORS` to each batch and
    the result returned by `FN` for that batch. Finally, return the list
    of counters of `MONITORS`.
    
    The purpose of this function is to collect various results and
    statistics (such as error measures) efficiently by applying the
    model only once, leaving extraction of quantities of interest from
    the model's results to `MONITORS`.
    
    See the model specific versions of this functions such as
    MONITOR-BPN-RESULTS.

<a name='x-28MGL-CORE-3AMONITORS-20GENERIC-FUNCTION-29'></a>

- [generic-function] **MONITORS** *OBJECT*

    Return monitors associated with `OBJECT`. See various
    methods such as [MONITORS][(accessor mgl-bp:bp-learner)] for more
    documentation.

<a name='x-28MGL-CORE-3A-40MGL-MONITOR-20MGL-PAX-3ASECTION-29'></a>

### 6.1 Monitors

<a name='x-28MGL-CORE-3AMONITOR-20CLASS-29'></a>

- [class] **MONITOR**

    A monitor that has another monitor called [`MEASURER`][3339]
    embedded in it. When this monitor is applied, it applies the
    measurer and passes the returned values to [`ADD-TO-COUNTER`][1f57] called on
    its [`COUNTER`][7471] slot. One may further specialize [`APPLY-MONITOR`][f95f] to change
    that.
    
    This class is useful when the same event monitor is applied
    repeatedly over a period and its results must be aggregated such as
    when training statistics are being tracked or when predictions are
    begin made. Note that the monitor must be compatible with the event
    it handles. That is, the embedded [`MEASURER`][3339] must be prepared to take
    the arguments that are documented to come with the event.

<a name='x-28MGL-CORE-3AMEASURER-20-28MGL-PAX-3AREADER-20MGL-CORE-3AMONITOR-29-29'></a>

- [reader] **MEASURER** *MONITOR* *(:MEASURER)*

    This must be a monitor itself which only means
    that [`APPLY-MONITOR`][f95f] is defined on it (but see [Monitoring][0924]). The
    returned values are aggregated by [`COUNTER`][4e21]. See
    [Measurers][2364] for a library of measurers.

<a name='x-28MGL-CORE-3ACOUNTER-20-28MGL-PAX-3AREADER-20MGL-CORE-3AMONITOR-29-29'></a>

- [reader] **COUNTER** *MONITOR* *(:COUNTER)*

    The [`COUNTER`][7471] of a monitor carries out the
    aggregation of results returned by [`MEASURER`][3339]. The See [Counters][998f]
    for a library of counters.

<a name='x-28MGL-CORE-3A-40MGL-MEASURER-20MGL-PAX-3ASECTION-29'></a>

### 6.2 Measurers

[`MEASURER`][3339] is a part of [`MONITOR`][a22b] objects, an embedded monitor that
computes a specific quantity (e.g. classification accuracy) from the
arguments of event it is applied to (e.g. the model results).
Measurers are often implemented by combining some kind of model
specific extractor with a generic measurer function.

All generic measurer functions return their results as multiple
values matching the arguments of [`ADD-TO-COUNTER`][1f57] for a counter of a
certain type (see [Counters][998f]) so as to make them easily used in a
[`MONITOR`][a22b]:

    (multiple-value-call #'add-to-counter <some-counter>
                         <call-to-some-measurer>)

The counter class compatible with the measurer this way is noted for
each function.

For a list of measurer functions see [Classification Measurers][505e].

<a name='x-28MGL-CORE-3A-40MGL-COUNTER-20MGL-PAX-3ASECTION-29'></a>

### 6.3 Counters

<a name='x-28MGL-CORE-3AADD-TO-COUNTER-20GENERIC-FUNCTION-29'></a>

- [generic-function] **ADD-TO-COUNTER** *COUNTER &REST ARGS*

    Add `ARGS` to `COUNTER` in some way. See specialized
    methods for type specific documentation.

<a name='x-28MGL-CORE-3ACOUNTER-VALUES-20GENERIC-FUNCTION-29'></a>

- [generic-function] **COUNTER-VALUES** *COUNTER*

    Return any number of values representing the state
    of `COUNTER`. See specialized methods for type specific
    documentation.

<a name='x-28MGL-CORE-3ARESET-COUNTER-20GENERIC-FUNCTION-29'></a>

- [generic-function] **RESET-COUNTER** *COUNTER*

    Restore state of `COUNTER` to what it was just after
    creation.

<a name='x-28MGL-CORE-3A-40MGL-ATTRIBUTES-20MGL-PAX-3ASECTION-29'></a>

#### 6.3.1 Attributes

<a name='x-28MGL-CORE-3AATTRIBUTED-20CLASS-29'></a>

- [class] **ATTRIBUTED**

    This is a utility class that all counters subclass.
    The [`ATTRIBUTES`][9112] plist can hold basically anything. Currently the
    attributes are only used when printing and they can be specified by
    the user. The monitor maker functions such as those in
    [Classification Monitors][cc50] also add attributes of their own to the
    counters they create.
    
    With the `:PREPEND-ATTRIBUTES` initarg when can easily add new
    attributes without clobbering the those in the `:INITFORM`, (`:TYPE`
    "rmse") in this case.
    
        (princ (make-instance 'rmse-counter
                              :prepend-attributes '(:event "pred."
                                                    :dataset "test")))
        ;; pred. test rmse: 0.000e+0 (0)
        => #<RMSE-COUNTER pred. test rmse: 0.000e+0 (0)>


<a name='x-28MGL-CORE-3AATTRIBUTES-20-28MGL-PAX-3AACCESSOR-20MGL-CORE-3AATTRIBUTED-29-29'></a>

- [accessor] **ATTRIBUTES** *ATTRIBUTED* *(:ATTRIBUTES = NIL)*

    A plist of attribute keys and values.

<a name='x-28MGL-COMMON-3ANAME-20-28METHOD-20NIL-20-28MGL-CORE-3AATTRIBUTED-29-29-29'></a>

- [method] **NAME** *(ATTRIBUTED ATTRIBUTED)*

    Return a string assembled from the values of the [`ATTRIBUTES`][9112] of
    `ATTRIBUTED`. If there are multiple entries with the same key, then
    they are printed near together.
    
    Values may be padded according to an enclosing
    [`WITH-PADDED-ATTRIBUTE-PRINTING`][12e8].

<a name='x-28MGL-CORE-3AWITH-PADDED-ATTRIBUTE-PRINTING-20MGL-PAX-3AMACRO-29'></a>

- [macro] **WITH-PADDED-ATTRIBUTE-PRINTING** *(ATTRIBUTEDS) &BODY BODY*

    Note the width of values for each attribute key which is the number
    of characters in the value's `PRINC-TO-STRING`'ed representation. In
    `BODY`, if attributes with they same key are printed they are forced
    to be at least this wide. This allows for nice, table-like output:
    
        (let ((attributeds
                (list (make-instance 'basic-counter
                                     :attributes '(:a 1 :b 23 :c 456))
                      (make-instance 'basic-counter
                                     :attributes '(:a 123 :b 45 :c 6)))))
          (with-padded-attribute-printing (attributeds)
            (map nil (lambda (attributed)
                       (format t "~A~%" attributed))
                 attributeds)))
        ;; 1   23 456: 0.000e+0 (0)
        ;; 123 45 6  : 0.000e+0 (0)


<a name='x-28MGL-CORE-3ALOG-PADDED-20FUNCTION-29'></a>

- [function] **LOG-PADDED** *ATTRIBUTEDS*

    Log (see `LOG-MSG`) `ATTRIBUTEDS` non-escaped (as in `PRINC` or ~A) with
    the output being as table-like as possible.

<a name='x-28MGL-CORE-3A-40MGL-COUNTER-CLASSES-20MGL-PAX-3ASECTION-29'></a>

#### 6.3.2 Counter classes

In addition to the really basic ones here, also see
[Classification Counters][32b3].

<a name='x-28MGL-CORE-3ABASIC-COUNTER-20CLASS-29'></a>

- [class] **BASIC-COUNTER** *ATTRIBUTED*

    A simple counter whose [`ADD-TO-COUNTER`][1f57] takes two
    additional parameters: an increment to the internal sums of called
    the `NUMERATOR` and `DENOMINATOR`. [`COUNTER-VALUES`][8a3b] returns two
    values:
    
    - `NUMERATOR` divided by `DENOMINATOR` (or 0 if `DENOMINATOR` is 0) and
    
    - `DENOMINATOR`
    
    Here is an example the compute the mean of 5 things received in two
    batches:
    
         (let ((counter (make-instance 'basic-counter)))
           (add-to-counter counter 6.5 3)
           (add-to-counter counter 3.5 2)
           counter)
         => #<BASIC-COUNTER 2.00000e+0 (5)>


<a name='x-28MGL-CORE-3ARMSE-COUNTER-20CLASS-29'></a>

- [class] **RMSE-COUNTER** *BASIC-COUNTER*

    A [`BASIC-COUNTER`][d3e3] with whose nominator accumulates
    the square of some statistics. It has the attribute `:TYPE` "rmse".
    [`COUNTER-VALUES`][8a3b] returns the square root of what [`BASIC-COUNTER`][d3e3]'s
    [`COUNTER-VALUES`][8a3b] would return.
    
        (let ((counter (make-instance 'rmse-counter)))
          (add-to-counter counter (+ (* 3 3) (* 4 4)) 2)
          counter)
        => #<RMSE-COUNTER rmse: 3.53553e+0 (2)>


<a name='x-28MGL-CORE-3A-40MGL-CLASSIFICATION-20MGL-PAX-3ASECTION-29'></a>

## 7 Classification

###### \[in package MGL-CORE\]
To be able to measure classification related quantities, we need to
define what the label of an instance is. Customization is possible
by implementing a method for a specific type of instance, but these
functions only ever appear as defaults that can be overridden.

<a name='x-28MGL-CORE-3ALABEL-INDEX-20GENERIC-FUNCTION-29'></a>

- [generic-function] **LABEL-INDEX** *INSTANCE*

    Return the label of `INSTANCE` as a non-negative
    integer.

<a name='x-28MGL-CORE-3ALABEL-INDEX-DISTRIBUTION-20GENERIC-FUNCTION-29'></a>

- [generic-function] **LABEL-INDEX-DISTRIBUTION** *INSTANCE*

    Return a one dimensional array of probabilities
    representing the distribution of labels. The probability of the
    label with [`LABEL-INDEX`][950d] `I` is element at index `I` of the returned
    arrray.

The following two functions are basically the same as the previous
two, but in batch mode: they return a sequence of label indices or
distributions. These are called on results produced by models.
Implement these for a model and the monitor maker functions below
will automatically work. See FIXDOC: for bpn and boltzmann.

<a name='x-28MGL-CORE-3ALABEL-INDICES-20GENERIC-FUNCTION-29'></a>

- [generic-function] **LABEL-INDICES** *RESULTS*

    Return a sequence of label indices for `RESULTS`
    produced by some model for a batch of instances. This is akin to
    [`LABEL-INDEX`][950d].

<a name='x-28MGL-CORE-3ALABEL-INDEX-DISTRIBUTIONS-20GENERIC-FUNCTION-29'></a>

- [generic-function] **LABEL-INDEX-DISTRIBUTIONS** *RESULT*

    Return a sequence of label index distributions for
    `RESULTS` produced by some model for a batch of instances. This is
    akin to [`LABEL-INDEX-DISTRIBUTION`][089c].

<a name='x-28MGL-CORE-3A-40MGL-CLASSIFICATION-MONITOR-20MGL-PAX-3ASECTION-29'></a>

### 7.1 Classification Monitors

The following functions return a list monitors. The monitors are
for events of signature (`INSTANCES` `MODEL`) such as those produced by
[`MONITOR-MODEL-RESULTS`][3ca8] and its various model specific variations.
They are model-agnostic functions, extensible to new classifier
types. 

<a name='x-28MGL-CORE-3AMAKE-CLASSIFICATION-ACCURACY-MONITORS-20FUNCTION-29'></a>

- [function] **MAKE-CLASSIFICATION-ACCURACY-MONITORS** *MODEL &KEY OPERATION-MODE ATTRIBUTES (LABEL-INDEX-FN #'LABEL-INDEX)*

    Return a list of [`MONITOR`][a22b] objects associated with
    [`CLASSIFICATION-ACCURACY-COUNTER`][f5e0]s. `LABEL-INDEX-FN` is a function
    like [`LABEL-INDEX`][950d]. See that function for more.
    
    Implemented in terms of [`MAKE-CLASSIFICATION-ACCURACY-MONITORS*`][3626].

<a name='x-28MGL-CORE-3AMAKE-CROSS-ENTROPY-MONITORS-20FUNCTION-29'></a>

- [function] **MAKE-CROSS-ENTROPY-MONITORS** *MODEL &KEY OPERATION-MODE ATTRIBUTES (LABEL-INDEX-DISTRIBUTION-FN #'LABEL-INDEX-DISTRIBUTION)*

    Return a list of [`MONITOR`][a22b] objects associated with
    [`CROSS-ENTROPY-COUNTER`][93e5]s. `LABEL-INDEX-DISTRIBUTION-FN` is a
    function like [`LABEL-INDEX-DISTRIBUTION`][089c]. See that function for more.
    
    Implemented in terms of [`MAKE-CROSS-ENTROPY-MONITORS*`][f1be].

<a name='x-28MGL-CORE-3AMAKE-LABEL-MONITORS-20FUNCTION-29'></a>

- [function] **MAKE-LABEL-MONITORS** *MODEL &KEY OPERATION-MODE ATTRIBUTES (LABEL-INDEX-FN #'LABEL-INDEX) (LABEL-INDEX-DISTRIBUTION-FN #'LABEL-INDEX-DISTRIBUTION)*

    Return classification accuracy and cross-entropy monitors. See
    [`MAKE-CLASSIFICATION-ACCURACY-MONITORS`][ec6a] and
    [`MAKE-CROSS-ENTROPY-MONITORS`][29a1] for a description of paramters.

The monitor makers above can be extended to support new classifier
types via the following generic functions.

<a name='x-28MGL-CORE-3AMAKE-CLASSIFICATION-ACCURACY-MONITORS-2A-20GENERIC-FUNCTION-29'></a>

- [generic-function] **MAKE-CLASSIFICATION-ACCURACY-MONITORS\*** *MODEL OPERATION-MODE LABEL-INDEX-FN ATTRIBUTES*

    Identical to [`MAKE-CLASSIFICATION-ACCURACY-MONITORS`][ec6a]
    bar the keywords arguments. Specialize this to add to support for
    new model types. The default implementation also allows for some
    extensibility: if [`LABEL-INDICES`][aac7] is defined on `MODEL`, then it will be
    used to extract label indices from model results.

<a name='x-28MGL-CORE-3AMAKE-CROSS-ENTROPY-MONITORS-2A-20GENERIC-FUNCTION-29'></a>

- [generic-function] **MAKE-CROSS-ENTROPY-MONITORS\*** *MODEL OPERATION-MODE LABEL-INDEX-DISTRIBUTION-FN ATTRIBUTES*

    Identical to [`MAKE-CROSS-ENTROPY-MONITORS`][29a1] bar the
    keywords arguments. Specialize this to add to support for new model
    types. The default implementation also allows for some
    extensibility: if [`LABEL-INDEX-DISTRIBUTIONS`][c246] is defined on `MODEL`,
    then it will be used to extract label distributions from model
    results.

<a name='x-28MGL-CORE-3A-40MGL-CLASSIFICATION-MEASURER-20MGL-PAX-3ASECTION-29'></a>

### 7.2 Classification Measurers

The functions here compare some known good solution (also known as
*ground truth* or *target*) to a prediction or approximation and
return some measure of their [dis][]similarity. They are model
independent, hence one has to extract the ground truths and
predictions first. Rarely used directly, they are mostly hidden
behind [Classification Monitors][cc50].

<a name='x-28MGL-CORE-3AMEASURE-CLASSIFICATION-ACCURACY-20FUNCTION-29'></a>

- [function] **MEASURE-CLASSIFICATION-ACCURACY** *TRUTHS PREDICTIONS &KEY (TEST #'EQL) TRUTH-KEY PREDICTION-KEY WEIGHT*

    Return the number of correct classifications and as the second
    value the number of instances (equal to length of `TRUTHS` in the
    non-weighted case). `TRUTHS` (keyed by `TRUTH-KEY`) is a sequence of
    opaque class labels compared with `TEST` to another sequence of
    classes labels in `PREDICTIONS` (keyed by `PREDICTION-KEY`). If `WEIGHT`
    is non-nil, then it is a function that returns the weight of an
    element of `TRUTHS`. Weighted cases add their weight to both
    counts (returned as the first and second values) instead of 1 as in
    the non-weighted case.
    
    Note how the returned values are suitable for `MULTIPLE-VALUE-CALL`
    with #'[`ADD-TO-COUNTER`][1f57] and a [`CLASSIFICATION-ACCURACY-COUNTER`][f5e0].

<a name='x-28MGL-CORE-3AMEASURE-CROSS-ENTROPY-20FUNCTION-29'></a>

- [function] **MEASURE-CROSS-ENTROPY** *TRUTHS PREDICTIONS &KEY TRUTH-KEY PREDICTION-KEY (MIN-PREDICTION-PR 1.d-15)*

    Return the sum of the cross-entropy between pairs of elements with
    the same index of `TRUTHS` and `PREDICTIONS`. `TRUTH-KEY` is a function
    that's when applied to an element of `TRUTHS` returns a sequence
    representing some kind of discrete target distribution (P in the
    definition below). `TRUTH-KEY` may be `NIL` which is equivalent to the
    `IDENTITY` function. `PREDICTION-KEY` is the same kind of key for
    `PREDICTIONS`, but the sequence it returns represents a distribution
    that approximates (Q below) the true one.
    
    Cross-entropy of the true and approximating distributions is defined
    as:
    
        cross-entropy(p,q) = - sum_i p(i) * log(q(i))
    
    of which this function returns the sum over the pairs of elements of
    `TRUTHS` and `PREDICTIONS` keyed by `TRUTH-KEY` and `PREDICTION-KEY`.
    
    Due to the logarithm, if q(i) is close to zero, we run into
    numerical problems. To prevent this, all q(i) that are less than
    `MIN-PREDICTION-PR` are treated as if they were `MIN-PREDICTION-PR`.
    
    The second value returned is the sum of p(i) over all `TRUTHS` and all
    `I`. This is normally equal to `(LENGTH TRUTHS)`, since elements of
    `TRUTHS` represent a probability distribution, but this is not
    enforced which allows relative importance of elements to be
    controlled.
    
    The third value returned is a plist that maps each index occurring
    in the distribution sequences to a list of two elements:
    
         sum_j p_j(i) * log(q_j(i))
    
    and
    
        sum_j p_j(i)
    
    where `J` indexes into `TRUTHS` and `PREDICTIONS`.
    
        (measure-cross-entropy '((0 1 0)) '((0.1 0.7 0.2)))
        => 0.35667497
           1
           (2 (0.0 0)
            1 (0.35667497 1)
            0 (0.0 0))
    
    Note how the returned values are suitable for `MULTIPLE-VALUE-CALL`
    with #'[`ADD-TO-COUNTER`][1f57] and a [`CROSS-ENTROPY-COUNTER`][93e5].

<a name='x-28MGL-CORE-3AMEASURE-ROC-AUC-20FUNCTION-29'></a>

- [function] **MEASURE-ROC-AUC** *PREDICTIONS PRED &KEY (KEY #'IDENTITY) WEIGHT*

    Return the area under the ROC curve for `PREDICTIONS` representing
    predictions for a binary classification problem. `PRED` is a predicate
    function for deciding whether a prediction belongs to the so called
    positive class. `KEY` returns a number for each element which is the
    predictor's idea of how much that element is likely to belong to the
    class, although it's not necessarily a probability.
    
    If `WEIGHT` is `NIL`, then all elements of `PREDICTIONS` count as 1
    towards the unnormalized sum within AUC. Else `WEIGHT` must be a
    function like `KEY`, but it should return the importance (a positive
    real number) of elements. If the weight of an prediction is 2 then
    it's as if there were another identical copy of that prediction in
    `PREDICTIONS`.
    
    The algorithm is based on algorithm 2 in the paper 'An introduction
    to ROC analysis' by Tom Fawcett.
    
    ROC AUC is equal to the probability of a randomly chosen positive
    having higher `KEY` (score) than a randomly chosen negative element.
    With equal scores in mind, a more precise version is: AUC is the
    expectation of the above probability over all possible sequences
    sorted by scores.

<a name='x-28MGL-CORE-3AMEASURE-CONFUSION-20FUNCTION-29'></a>

- [function] **MEASURE-CONFUSION** *TRUTHS PREDICTIONS &KEY (TEST #'EQL) TRUTH-KEY PREDICTION-KEY WEIGHT*

    Create a [`CONFUSION-MATRIX`][08c9] from `TRUTHS` and `PREDICTIONS`.
    `TRUTHS` (keyed by `TRUTH-KEY`) is a sequence of class labels compared
    with `TEST` to another sequence of class labels in `PREDICTIONS` (keyed
    by `PREDICTION-KEY`). If `WEIGHT` is non-nil, then it is a function that
    returns the weight of an element of `TRUTHS`. Weighted cases add their
    weight to both counts (returned as the first and second values).
    
    Note how the returned confusion matrix can be added to another with
    [`ADD-TO-COUNTER`][1f57].

<a name='x-28MGL-CORE-3A-40MGL-CLASSIFICATION-COUNTER-20MGL-PAX-3ASECTION-29'></a>

### 7.3 Classification Counters

<a name='x-28MGL-CORE-3ACLASSIFICATION-ACCURACY-COUNTER-20CLASS-29'></a>

- [class] **CLASSIFICATION-ACCURACY-COUNTER** *BASIC-COUNTER*

    A [`BASIC-COUNTER`][d3e3] with "acc." as its `:TYPE`
    attribute and a `PRINT-OBJECT` method that prints percentages.

<a name='x-28MGL-CORE-3ACROSS-ENTROPY-COUNTER-20CLASS-29'></a>

- [class] **CROSS-ENTROPY-COUNTER** *BASIC-COUNTER*

    A [`BASIC-COUNTER`][d3e3] with "xent" as its `:TYPE`
    attribute.

<a name='x-28MGL-CORE-3A-40MGL-CONFUSION-MATRIX-20MGL-PAX-3ASECTION-29'></a>

#### 7.3.1 Confusion Matrices

<a name='x-28MGL-CORE-3ACONFUSION-MATRIX-20CLASS-29'></a>

- [class] **CONFUSION-MATRIX**

    A confusion matrix keeps count of classification
    results. The correct class is called `target' and the output of the
    classifier is called`prediction'. Classes are compared with
    `EQUAL`.

<a name='x-28MGL-CORE-3AMAKE-CONFUSION-MATRIX-20FUNCTION-29'></a>

- [function] **MAKE-CONFUSION-MATRIX** *&KEY (TEST #'EQL)*

<a name='x-28MGL-CORE-3ASORT-CONFUSION-CLASSES-20GENERIC-FUNCTION-29'></a>

- [generic-function] **SORT-CONFUSION-CLASSES** *MATRIX CLASSES*

    Return a list of `CLASSES` sorted for presentation
    purposes.

<a name='x-28MGL-CORE-3ACONFUSION-CLASS-NAME-20GENERIC-FUNCTION-29'></a>

- [generic-function] **CONFUSION-CLASS-NAME** *MATRIX CLASS*

    Name of `CLASS` for presentation purposes.

<a name='x-28MGL-CORE-3ACONFUSION-COUNT-20GENERIC-FUNCTION-29'></a>

- [generic-function] **CONFUSION-COUNT** *MATRIX TARGET PREDICTION*

<a name='x-28MGL-CORE-3AMAP-CONFUSION-MATRIX-20GENERIC-FUNCTION-29'></a>

- [generic-function] **MAP-CONFUSION-MATRIX** *FN MATRIX*

    Call `FN` with `TARGET`, `PREDICTION`, `COUNT` paramaters
    for each cell in the confusion matrix. Cells with a zero count may
    be ommitted.

<a name='x-28MGL-CORE-3ACONFUSION-MATRIX-CLASSES-20GENERIC-FUNCTION-29'></a>

- [generic-function] **CONFUSION-MATRIX-CLASSES** *MATRIX*

    A list of all classes. The default is to collect
    classes from the counts. This can be overridden if, for instance,
    some classes are not present in the results.

<a name='x-28MGL-CORE-3ACONFUSION-MATRIX-ACCURACY-20FUNCTION-29'></a>

- [function] **CONFUSION-MATRIX-ACCURACY** *MATRIX &KEY FILTER*

    Return the overall accuracy of the results in `MATRIX`. It's computed
    as the number of correctly classified cases (hits) divided by the
    name of cases. Return the number of hits and the number of cases as
    the second and third value. If `FILTER` function is given, then call
    it with the target and the prediction of the cell. Disregard cell
    for which `FILTER` returns `NIL`.
    
    Precision and recall can be easily computed by giving the right
    filter, although those are provided in separate convenience
    functions.

<a name='x-28MGL-CORE-3ACONFUSION-MATRIX-PRECISION-20FUNCTION-29'></a>

- [function] **CONFUSION-MATRIX-PRECISION** *MATRIX PREDICTION*

    Return the accuracy over the cases when the classifier said
    `PREDICTION`.

<a name='x-28MGL-CORE-3ACONFUSION-MATRIX-RECALL-20FUNCTION-29'></a>

- [function] **CONFUSION-MATRIX-RECALL** *MATRIX TARGET*

    Return the accuracy over the cases when the correct class is
    `TARGET`.

<a name='x-28MGL-CORE-3AADD-CONFUSION-MATRIX-20FUNCTION-29'></a>

- [function] **ADD-CONFUSION-MATRIX** *MATRIX RESULT-MATRIX*

    Add `MATRIX` into `RESULT-MATRIX`.

<a name='x-28MGL-OPT-3A-40MGL-OPT-20MGL-PAX-3ASECTION-29'></a>

## 8 Gradient Based Optimization

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
    `GRADIENT-SOURCE` by updating some of its parameters in `WEIGHTS` (a `MAT`
    or a sequence of MATs). Return `WEIGHTS`. `DATASET` (see
    [Datasets][72e9]) is a set of unoptimized parameters of the same
    function. For example, `WEIGHTS` may be the weights of a neural
    network while `DATASET` is the training set consisting of inputs
    suitable for [`SET-INPUT`][8795]. The default
    `DATASET`, ([`*INFINITELY-EMPTY-DATASET*`][0966]) is suitable for when all
    parameters are optimized, so there is nothing left to come from the
    environment.
    
    Optimization terminates if `DATASET` is a sampler and it runs out or
    when some other condition met (see [`TERMINATION`][bec0], for example). If
    `DATASET` is a `SEQUENCE`, then it is reused over and over again.
    
    Examples for various optimizers are provided in [Gradient Descent][53a7] and
    [Conjugate Gradient][8729].

<a name='x-28MGL-OPT-3A-40MGL-OPT-ITERATIVE-OPTIMIZER-20MGL-PAX-3ASECTION-29'></a>

### 8.1 Iterative Optimizer

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

<a name='x-28MGL-OPT-3AON-OPTIMIZATION-STARTED-20-28MGL-PAX-3AACCESSOR-20MGL-OPT-3AITERATIVE-OPTIMIZER-29-29'></a>

- [accessor] **ON-OPTIMIZATION-STARTED** *ITERATIVE-OPTIMIZER* *(:ON-OPTIMIZATION-STARTED = NIL)*

    An event hook with parameters `(OPTIMIZER
    GRADIENT-SOURCE N-INSTANCES)`. Called after initializations are
    performed (INITIALIZE-OPTIMIZER*, INITIALIZE-GRADIENT-SOURCE*) but
    before optimization is started.

<a name='x-28MGL-OPT-3AON-OPTIMIZATION-FINISHED-20-28MGL-PAX-3AACCESSOR-20MGL-OPT-3AITERATIVE-OPTIMIZER-29-29'></a>

- [accessor] **ON-OPTIMIZATION-FINISHED** *ITERATIVE-OPTIMIZER* *(:ON-OPTIMIZATION-FINISHED = NIL)*

    An event hook with parameters `(OPTIMIZER
    GRADIENT-SOURCE N-INSTANCES)`. Called when optimization has
    finished.

<a name='x-28MGL-OPT-3AON-N-INSTANCES-CHANGED-20-28MGL-PAX-3AACCESSOR-20MGL-OPT-3AITERATIVE-OPTIMIZER-29-29'></a>

- [accessor] **ON-N-INSTANCES-CHANGED** *ITERATIVE-OPTIMIZER* *(:ON-N-INSTANCES-CHANGED = NIL)*

    An event hook with parameters `(OPTIMIZER
    GRADIENT-SOURCE N-INSTANCES)`. Called when optimization of a batch
    of instances is done and [`N-INSTANCES`][66a1] is incremented.

Now let's discuss a few handy utilities.

<a name='x-28MGL-OPT-3AMONITOR-OPTIMIZATION-PERIODICALLY-20FUNCTION-29'></a>

- [function] **MONITOR-OPTIMIZATION-PERIODICALLY** *OPTIMIZER PERIODIC-FNS*

    For each periodic function in the list of `PERIODIC-FNS`, add a
    monitor to `OPTIMIZER`'s [`ON-OPTIMIZATION-STARTED`][dae0],
    [`ON-OPTIMIZATION-FINISHED`][9c36] and [`ON-N-INSTANCES-CHANGED`][9cdc] hooks. The
    monitors are simple functions that just call each periodic function
    with the event parameters (`OPTIMIZER` `GRADIENT-SOURCE` [`N-INSTANCES`][66a1]).
    Return `OPTIMIZER`.
    
    To log and reset the monitors of the gradient source after every
    1000 instances seen by `OPTIMIZER`:
    
        (monitor-optimization-periodically optimizer
                                           '((:fn log-my-test-error
                                              :period 2000)
                                             (:fn reset-optimization-monitors
                                              :period 1000
                                              :last-eval 0)))
    
    Note how we don't pass it's allowed to just pass the initargs for a
    `PERIODIC-FN` instead of `PERIODIC-FN` itself. The `:LAST-EVAL` 0 bit
    prevents [`RESET-OPTIMIZATION-MONITORS`][326c] from being called at the start
    of the optimization when the monitors are empty anyway.

<a name='x-28MGL-OPT-3ARESET-OPTIMIZATION-MONITORS-20GENERIC-FUNCTION-29'></a>

- [generic-function] **RESET-OPTIMIZATION-MONITORS** *OPTIMIZER GRADIENT-SOURCE*

    Report the state of [`MONITORS`][b22b] of
    `OPTIMIZER` and `GRADIENT-SOURCE` and reset their counters. See
    [`MONITOR-OPTIMIZATION-PERIODICALLY`][918e] for an example of how this is
    used.

<a name='x-28MGL-OPT-3ARESET-OPTIMIZATION-MONITORS-20-28METHOD-20NIL-20-28MGL-OPT-3AITERATIVE-OPTIMIZER-20T-29-29-29'></a>

- [method] **RESET-OPTIMIZATION-MONITORS** *(OPTIMIZER ITERATIVE-OPTIMIZER) GRADIENT-SOURCE*

    Log the counters of the monitors and reset them.

<a name='x-28MGL-OPT-3AREPORT-OPTIMIZATION-PARAMETERS-20GENERIC-FUNCTION-29'></a>

- [generic-function] **REPORT-OPTIMIZATION-PARAMETERS** *OPTIMIZER GRADIENT-SOURCE*

    A utility that's often called at the start of
    optimization (from [`ON-OPTIMIZATION-STARTED`][dae0]). The default
    implementation logs the description of `GRADIENT-SOURCE` (as in
    `DESCRIBE`) and `OPTIMIZER` and calls `LOG-CUDA`.

<a name='x-28MGL-GD-3A-40MGL-GD-20MGL-PAX-3ASECTION-29'></a>

### 8.2 Gradient Descent

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
                                  :generator (lambda ()
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

#### 8.2.1 Batch GD Optimizer

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

#### 8.2.2 Segmented GD Optimizer

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
    the segment of the learner with [`MAP-SEGMENTS`][8202]. [`SEGMENTER`][b6ac] is a
    function that is called with each segment and returns an optimizer
    or `NIL`. Several segments may be mapped to the same optimizer.
    After the segment->optimizer mappings are collected, each
    optimizer is initialized by INITIALIZE-OPTIMIZER with the list of
    segments mapped to it.

<a name='x-28MGL-OPT-3ASEGMENTS-20-28MGL-PAX-3AREADER-20MGL-GD-3ASEGMENTED-GD-OPTIMIZER-29-29'></a>

- [reader] **SEGMENTS** *SEGMENTED-GD-OPTIMIZER*

<a name='x-28MGL-GD-3A-40MGL-GD-PER-WEIGHT-OPTIMIZATION-20MGL-PAX-3ASECTION-29'></a>

#### 8.2.3 Per-weight Optimization

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

### 8.3 Conjugate Gradient

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
                                  :generator (lambda ()
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

<a name='x-28MGL-CG-3AON-CG-BATCH-DONE-20-28MGL-PAX-3AACCESSOR-20MGL-CG-3ACG-OPTIMIZER-29-29'></a>

- [accessor] **ON-CG-BATCH-DONE** *CG-OPTIMIZER* *(:ON-CG-BATCH-DONE = NIL)*

    An event hook called when processing a conjugate
    gradient batch is done. The handlers on the hook are called with 8
    arguments:
    
        (optimizer gradient-source instances
         best-w best-f n-line-searches
         n-succesful-line-searches n-evaluations)
    
    The latter 5 of which are the return values of the [`CG`][f9f7] function.

<a name='x-28MGL-CG-3ALOG-CG-BATCH-DONE-20GENERIC-FUNCTION-29'></a>

- [generic-function] **LOG-CG-BATCH-DONE** *OPTIMIZER GRADIENT-SOURCE INSTANCES BEST-W BEST-F N-LINE-SEARCHES N-SUCCESFUL-LINE-SEARCHES N-EVALUATIONS*

    This is a function can be added to
    [`ON-CG-BATCH-DONE`][e6a3]. The default implementation simply logs the event
    arguments.

<a name='x-28MGL-CG-3ASEGMENT-FILTER-20-28MGL-PAX-3AREADER-20MGL-CG-3ACG-OPTIMIZER-29-29'></a>

- [reader] **SEGMENT-FILTER** *CG-OPTIMIZER* *(:SEGMENT-FILTER = (CONSTANTLY T))*

    A predicate function on segments that filters out
    uninteresting segments. Called from [`INITIALIZE-OPTIMIZER*`][4a97].

<a name='x-28MGL-OPT-3A-40MGL-OPT-EXTENSION-API-20MGL-PAX-3ASECTION-29'></a>

### 8.4 Extension API

<a name='x-28MGL-OPT-3A-40MGL-OPT-OPTIMIZER-20MGL-PAX-3ASECTION-29'></a>

#### 8.4.1 Implementing Optimizers

The following generic functions must be specialized for new
optimizer types.

<a name='x-28MGL-OPT-3AMINIMIZE-2A-20GENERIC-FUNCTION-29'></a>

- [generic-function] **MINIMIZE\*** *OPTIMIZER GRADIENT-SOURCE WEIGHTS DATASET*

    Called by [`MINIMIZE`][bca8] after [`INITIALIZE-OPTIMIZER*`][4a97] and
    [`INITIALIZE-GRADIENT-SOURCE*`][c54c], this generic function is the main
    extension point for writing optimizers.

<a name='x-28MGL-OPT-3AINITIALIZE-OPTIMIZER-2A-20GENERIC-FUNCTION-29'></a>

- [generic-function] **INITIALIZE-OPTIMIZER\*** *OPTIMIZER GRADIENT-SOURCE WEIGHTS DATASET*

    Called automatically before training starts, this
    function sets up `OPTIMIZER` to be suitable for optimizing
    `GRADIENT-SOURCE`. It typically creates appropriately sized
    accumulators for the gradients.

<a name='x-28MGL-OPT-3ASEGMENTS-20GENERIC-FUNCTION-29'></a>

- [generic-function] **SEGMENTS** *OPTIMIZER*

    Several weight matrices known as *segments* can be
    optimized by a single optimizer. This function returns them as a
    list.

The rest are just useful for utilities for implementing
optimizers.

<a name='x-28MGL-OPT-3ATERMINATE-OPTIMIZATION-P-20FUNCTION-29'></a>

- [function] **TERMINATE-OPTIMIZATION-P** *N-INSTANCES TERMINATION*

    Utility function for subclasses of [`ITERATIVE-OPTIMIZER`][83bf]. It returns
    whether optimization is to be terminated based on `N-INSTANCES` and
    `TERMINATION` that are values of the respective accessors of
    [`ITERATIVE-OPTIMIZER`][83bf].

<a name='x-28MGL-OPT-3ASET-N-INSTANCES-20FUNCTION-29'></a>

- [function] **SET-N-INSTANCES** *OPTIMIZER GRADIENT-SOURCE N-INSTANCES*

    Set [`N-INSTANCES`][66a1] of `OPTIMIZER` and
    fire [`ON-N-INSTANCES-CHANGED`][9cdc]. [`ITERATIVE-OPTIMIZER`][83bf] subclasses must
    call this to increment [`N-INSTANCES`][66a1].

<a name='x-28MGL-OPT-3ASEGMENT-SET-20CLASS-29'></a>

- [class] **SEGMENT-SET**

    This is a utility class for optimizers that have a
    list of [`SEGMENTS`][f1cd] and (the weights being optimized) is able to copy
    back and forth between those segments and a single `MAT` (the
    accumulator).

<a name='x-28MGL-OPT-3ASEGMENTS-20-28MGL-PAX-3AREADER-20MGL-OPT-3ASEGMENT-SET-29-29'></a>

- [reader] **SEGMENTS** *SEGMENT-SET* *(:SEGMENTS)*

    A list of weight matrices.

<a name='x-28MGL-COMMON-3ASIZE-20-28MGL-PAX-3AREADER-20MGL-OPT-3ASEGMENT-SET-29-29'></a>

- [reader] **SIZE** *SEGMENT-SET*

    The sum of the sizes of the weight matrices of
    [`SEGMENTS`][f1cd].

<a name='x-28MGL-OPT-3ADO-SEGMENT-SET-20MGL-PAX-3AMACRO-29'></a>

- [macro] **DO-SEGMENT-SET** *(SEGMENT &OPTIONAL START) SEGMENT-SET &BODY BODY*

    Iterate over [`SEGMENTS`][f1cd] in `SEGMENT-SET`. If `START` is specified, the it
    is bound to the start index of `SEGMENT` within `SEGMENT-SET`. The start
    index is the sum of the sizes of previous segments.

<a name='x-28MGL-OPT-3ASEGMENT-SET-3C-MAT-20FUNCTION-29'></a>

- [function] **SEGMENT-SET\<-MAT** *SEGMENT-SET MAT*

    Copy the values of `MAT` to the weight matrices of `SEGMENT-SET` as if
    they were concatenated into a single `MAT`.

<a name='x-28MGL-OPT-3ASEGMENT-SET--3EMAT-20FUNCTION-29'></a>

- [function] **SEGMENT-SET-\>MAT** *SEGMENT-SET MAT*

    Copy the values of `SEGMENT-SET` to `MAT` as if they were concatenated
    into a single `MAT`.

<a name='x-28MGL-OPT-3A-40MGL-OPT-GRADIENT-SOURCE-20MGL-PAX-3ASECTION-29'></a>

#### 8.4.2 Implementing Gradient Sources

Weights can be stored in a multitude of ways. Optimizers need to
update weights, so it is assumed that weights are stored in any
number of `MAT` objects called segments.

The generic functions in this section must all be specialized for
new gradient sources except where noted.

<a name='x-28MGL-OPT-3AMAP-SEGMENTS-20GENERIC-FUNCTION-29'></a>

- [generic-function] **MAP-SEGMENTS** *FN GRADIENT-SOURCE*

    Apply `FN` to each segment of `GRADIENT-SOURCE`.

<a name='x-28MGL-OPT-3AMAP-SEGMENT-RUNS-20GENERIC-FUNCTION-29'></a>

- [generic-function] **MAP-SEGMENT-RUNS** *FN SEGMENT*

    Call `FN` with start and end of intervals of
    consecutive indices that are not missing in `SEGMENT`. Called by
    optimizers that support partial updates. The default implementation
    assumes that all weights are present. This only needs to be
    specialized if one plans to use an optimizer that knows how to deal
    unused/missing weights such as [`MGL-GD:NORMALIZED-BATCH-GD-OPTIMIZER`][51ad]
    and `OPTIMIZER` [`MGL-GD:PER-WEIGHT-BATCH-GD-OPTIMIZER`][1fa8].

<a name='x-28MGL-OPT-3ASEGMENT-WEIGHTS-20GENERIC-FUNCTION-29'></a>

- [generic-function] **SEGMENT-WEIGHTS** *SEGMENT*

    Return the weight matrix of `SEGMENT`. A segment
    doesn't need to be a `MAT` object itself. For example, it may be a
    `MGL-BM:CHUNK` of a [MGL-BM:BM][CLASS] or a `MGL-BP:LUMP` of a
    [MGL-BP:BPN][CLASS] whose `NODES` slot holds the weights.

<a name='x-28MGL-OPT-3ASEGMENT-WEIGHTS-20-28METHOD-20NIL-20-28MGL-MAT-3AMAT-29-29-29'></a>

- [method] **SEGMENT-WEIGHTS** *(MAT MAT)*

    When the segment is really a `MAT`, then just return it.

<a name='x-28MGL-OPT-3ALIST-SEGMENTS-20FUNCTION-29'></a>

- [function] **LIST-SEGMENTS** *GRADIENT-SOURCE*

    A utility function that returns the list of segments from
    [`MAP-SEGMENTS`][8202] on `GRADIENT-SOURCE`.

<a name='x-28MGL-OPT-3AINITIALIZE-GRADIENT-SOURCE-2A-20GENERIC-FUNCTION-29'></a>

- [generic-function] **INITIALIZE-GRADIENT-SOURCE\*** *OPTIMIZER GRADIENT-SOURCE WEIGHTS DATASET*

    Called automatically before [`MINIMIZE*`][3a6e] is called,
    this function may be specialized if `GRADIENT-SOURCE` needs some kind
    of setup.

<a name='x-28MGL-OPT-3AINITIALIZE-GRADIENT-SOURCE-2A-20-28METHOD-20NIL-20-28T-20T-20T-20T-29-29-29'></a>

- [method] **INITIALIZE-GRADIENT-SOURCE\*** *OPTIMIZER GRADIENT-SOURCE WEIGHTS DATASET*

    The default method does nothing.

<a name='x-28MGL-OPT-3AACCUMULATE-GRADIENTS-2A-20GENERIC-FUNCTION-29'></a>

- [generic-function] **ACCUMULATE-GRADIENTS\*** *GRADIENT-SOURCE SINK BATCH MULTIPLIER VALUEP*

    Add `MULTIPLIER` times the sum of first-order
    gradients to accumulators of `SINK` (normally accessed with
    [`DO-GRADIENT-SINK`][643d]) and if `VALUEP`, return the sum of values of the
    function being optimized for a `BATCH` of instances. `GRADIENT-SOURCE`
    is the object representing the function being optimized, `SINK` is
    gradient sink.
    
    Note the number of instances in `BATCH` may be larger than what
    `GRADIENT-SOURCE` process in one go (in the sense of say,
    [`MAX-N-STRIPES`][9598]), so [`DO-BATCHES-FOR-MODEL`][39c1] or something like (`GROUP`
    `BATCH` [`MAX-N-STRIPES`][9598]) can be handy.

<a name='x-28MGL-OPT-3A-40MGL-OPT-GRADIENT-SINK-20MGL-PAX-3ASECTION-29'></a>

#### 8.4.3 Implementing Gradient Sinks

Optimizers call [`ACCUMULATE-GRADIENTS*`][4c7c] on gradient sources. One
parameter of [`ACCUMULATE-GRADIENTS*`][4c7c] is the `SINK`. A gradient sink
knows what accumulator matrix (if any) belongs to a segment. Sinks
are defined entirely by [`MAP-GRADIENT-SINK`][97ba].

<a name='x-28MGL-OPT-3AMAP-GRADIENT-SINK-20GENERIC-FUNCTION-29'></a>

- [generic-function] **MAP-GRADIENT-SINK** *FN SINK*

    Call `FN` of lambda list (`SEGMENT` `ACCUMULATOR`) on
    each segment and their corresponding accumulator `MAT` in `SINK`.

<a name='x-28MGL-OPT-3ADO-GRADIENT-SINK-20MGL-PAX-3AMACRO-29'></a>

- [macro] **DO-GRADIENT-SINK** *((SEGMENT ACCUMULATOR) SINK) &BODY BODY*

    A convenience macro on top of [`MAP-GRADIENT-SINK`][97ba].

<a name='x-28MGL-DIFFUN-3A-40MGL-DIFFUN-20MGL-PAX-3ASECTION-29'></a>

## 9 Differentiable Functions

###### \[in package MGL-DIFFUN\]
<a name='x-28MGL-DIFFUN-3ADIFFUN-20CLASS-29'></a>

- [class] **DIFFUN**

    [`DIFFUN`][f4f4] dresses a lisp function (in its [`FN`][b96b] slot) as
    a gradient source (see [Implementing Gradient Sources][984f]) which allows it to
    be used in [`MINIMIZE`][bca8]. See the examples in [Gradient Descent][53a7] and
    [Conjugate Gradient][8729].

<a name='x-28MGL-COMMON-3AFN-20-28MGL-PAX-3AREADER-20MGL-DIFFUN-3ADIFFUN-29-29'></a>

- [reader] **FN** *DIFFUN* *(:FN)*

    A real valued lisp function. It may have any
    number of parameters.

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

## 10 Backprogation Neural Networks


<a name='x-28MGL-3A-40MGL-BM-20MGL-PAX-3ASECTION-29'></a>

## 11 Boltzmann Machines


<a name='x-28MGL-3A-40MGL-GP-20MGL-PAX-3ASECTION-29'></a>

## 12 Gaussian Processes


  [026c]: #x-28MGL-3A-40MGL-GP-20MGL-PAX-3ASECTION-29 "(MGL:@MGL-GP MGL-PAX:SECTION)"
  [02de]: #x-28MGL-RESAMPLE-3ASPLIT-FOLD-2FMOD-20FUNCTION-29 "(MGL-RESAMPLE:SPLIT-FOLD/MOD FUNCTION)"
  [0552]: #x-28MGL-CORE-3A-40MGL-MODEL-STRIPE-20MGL-PAX-3ASECTION-29 "(MGL-CORE:@MGL-MODEL-STRIPE MGL-PAX:SECTION)"
  [0675]: #x-28MGL-RESAMPLE-3A-40MGL-RESAMPLE-BAGGING-20MGL-PAX-3ASECTION-29 "(MGL-RESAMPLE:@MGL-RESAMPLE-BAGGING MGL-PAX:SECTION)"
  [089c]: #x-28MGL-CORE-3ALABEL-INDEX-DISTRIBUTION-20GENERIC-FUNCTION-29 "(MGL-CORE:LABEL-INDEX-DISTRIBUTION GENERIC-FUNCTION)"
  [08c9]: #x-28MGL-CORE-3ACONFUSION-MATRIX-20CLASS-29 "(MGL-CORE:CONFUSION-MATRIX CLASS)"
  [0924]: #x-28MGL-CORE-3A-40MGL-MONITORING-20MGL-PAX-3ASECTION-29 "(MGL-CORE:@MGL-MONITORING MGL-PAX:SECTION)"
  [0966]: #x-28MGL-DATASET-3A-2AINFINITELY-EMPTY-DATASET-2A-20VARIABLE-29 "(MGL-DATASET:*INFINITELY-EMPTY-DATASET* VARIABLE)"
  [0ab9]: #x-28MGL-3A-40MGL-GLOSSARY-20MGL-PAX-3ASECTION-29 "(MGL:@MGL-GLOSSARY MGL-PAX:SECTION)"
  [12e8]: #x-28MGL-CORE-3AWITH-PADDED-ATTRIBUTE-PRINTING-20MGL-PAX-3AMACRO-29 "(MGL-CORE:WITH-PADDED-ATTRIBUTE-PRINTING MGL-PAX:MACRO)"
  [1426]: #x-28MGL-CORE-3A-40MGL-PARAMETERIZED-EXECUTOR-CACHE-20MGL-PAX-3ASECTION-29 "(MGL-CORE:@MGL-PARAMETERIZED-EXECUTOR-CACHE MGL-PAX:SECTION)"
  [1541]: #x-28MGL-CORE-3A-40MGL-CONFUSION-MATRIX-20MGL-PAX-3ASECTION-29 "(MGL-CORE:@MGL-CONFUSION-MATRIX MGL-PAX:SECTION)"
  [1a5d]: #x-28MGL-DIFFUN-3A-40MGL-DIFFUN-20MGL-PAX-3ASECTION-29 "(MGL-DIFFUN:@MGL-DIFFUN MGL-PAX:SECTION)"
  [1f57]: #x-28MGL-CORE-3AADD-TO-COUNTER-20GENERIC-FUNCTION-29 "(MGL-CORE:ADD-TO-COUNTER GENERIC-FUNCTION)"
  [1fa8]: #x-28MGL-GD-3APER-WEIGHT-BATCH-GD-OPTIMIZER-20CLASS-29 "(MGL-GD:PER-WEIGHT-BATCH-GD-OPTIMIZER CLASS)"
  [2100]: #x-28MGL-DATASET-3A-40MGL-SAMPLER-FUNCTION-SAMPLER-20MGL-PAX-3ASECTION-29 "(MGL-DATASET:@MGL-SAMPLER-FUNCTION-SAMPLER MGL-PAX:SECTION)"
  [2364]: #x-28MGL-CORE-3A-40MGL-MEASURER-20MGL-PAX-3ASECTION-29 "(MGL-CORE:@MGL-MEASURER MGL-PAX:SECTION)"
  [25a8]: #x-28MGL-GD-3A-40MGL-GD-SEGMENTED-GD-OPTIMIZER-20MGL-PAX-3ASECTION-29 "(MGL-GD:@MGL-GD-SEGMENTED-GD-OPTIMIZER MGL-PAX:SECTION)"
  [2730]: #x-28MGL-OPT-3A-40MGL-OPT-EXTENSION-API-20MGL-PAX-3ASECTION-29 "(MGL-OPT:@MGL-OPT-EXTENSION-API MGL-PAX:SECTION)"
  [29a1]: #x-28MGL-CORE-3AMAKE-CROSS-ENTROPY-MONITORS-20FUNCTION-29 "(MGL-CORE:MAKE-CROSS-ENTROPY-MONITORS FUNCTION)"
  [2b76]: #x-28MGL-RESAMPLE-3AFRACTURE-20FUNCTION-29 "(MGL-RESAMPLE:FRACTURE FUNCTION)"
  [2cc2]: #x-28MGL-CORE-3ADO-EXECUTORS-20MGL-PAX-3AMACRO-29 "(MGL-CORE:DO-EXECUTORS MGL-PAX:MACRO)"
  [326c]: #x-28MGL-OPT-3ARESET-OPTIMIZATION-MONITORS-20GENERIC-FUNCTION-29 "(MGL-OPT:RESET-OPTIMIZATION-MONITORS GENERIC-FUNCTION)"
  [32b3]: #x-28MGL-CORE-3A-40MGL-CLASSIFICATION-COUNTER-20MGL-PAX-3ASECTION-29 "(MGL-CORE:@MGL-CLASSIFICATION-COUNTER MGL-PAX:SECTION)"
  [3339]: #x-28MGL-CORE-3AMEASURER-20-28MGL-PAX-3AREADER-20MGL-CORE-3AMONITOR-29-29 "(MGL-CORE:MEASURER (MGL-PAX:READER MGL-CORE:MONITOR))"
  [3626]: #x-28MGL-CORE-3AMAKE-CLASSIFICATION-ACCURACY-MONITORS-2A-20GENERIC-FUNCTION-29 "(MGL-CORE:MAKE-CLASSIFICATION-ACCURACY-MONITORS* GENERIC-FUNCTION)"
  [39c1]: #x-28MGL-CORE-3ADO-BATCHES-FOR-MODEL-20MGL-PAX-3AMACRO-29 "(MGL-CORE:DO-BATCHES-FOR-MODEL MGL-PAX:MACRO)"
  [3a6e]: #x-28MGL-OPT-3AMINIMIZE-2A-20GENERIC-FUNCTION-29 "(MGL-OPT:MINIMIZE* GENERIC-FUNCTION)"
  [3ca8]: #x-28MGL-CORE-3AMONITOR-MODEL-RESULTS-20FUNCTION-29 "(MGL-CORE:MONITOR-MODEL-RESULTS FUNCTION)"
  [3ed1]: #x-28MGL-CORE-3AWITH-STRIPES-20MGL-PAX-3AMACRO-29 "(MGL-CORE:WITH-STRIPES MGL-PAX:MACRO)"
  [4293]: #x-28MGL-RESAMPLE-3A-40MGL-RESAMPLE-CROSS-VALIDATION-20MGL-PAX-3ASECTION-29 "(MGL-RESAMPLE:@MGL-RESAMPLE-CROSS-VALIDATION MGL-PAX:SECTION)"
  [45db]: #x-28MGL-3A-40MGL-CODE-ORGANIZATION-20MGL-PAX-3ASECTION-29 "(MGL:@MGL-CODE-ORGANIZATION MGL-PAX:SECTION)"
  [4a97]: #x-28MGL-OPT-3AINITIALIZE-OPTIMIZER-2A-20GENERIC-FUNCTION-29 "(MGL-OPT:INITIALIZE-OPTIMIZER* GENERIC-FUNCTION)"
  [4c7c]: #x-28MGL-OPT-3AACCUMULATE-GRADIENTS-2A-20GENERIC-FUNCTION-29 "(MGL-OPT:ACCUMULATE-GRADIENTS* GENERIC-FUNCTION)"
  [4e21]: #x-28MGL-CORE-3ACOUNTER-20-28MGL-PAX-3AREADER-20MGL-CORE-3AMONITOR-29-29 "(MGL-CORE:COUNTER (MGL-PAX:READER MGL-CORE:MONITOR))"
  [505e]: #x-28MGL-CORE-3A-40MGL-CLASSIFICATION-MEASURER-20MGL-PAX-3ASECTION-29 "(MGL-CORE:@MGL-CLASSIFICATION-MEASURER MGL-PAX:SECTION)"
  [51ad]: #x-28MGL-GD-3ANORMALIZED-BATCH-GD-OPTIMIZER-20CLASS-29 "(MGL-GD:NORMALIZED-BATCH-GD-OPTIMIZER CLASS)"
  [53a7]: #x-28MGL-GD-3A-40MGL-GD-20MGL-PAX-3ASECTION-29 "(MGL-GD:@MGL-GD MGL-PAX:SECTION)"
  [56ee]: #x-28MGL-CORE-3A-40MGL-MODEL-PERSISTENCE-20MGL-PAX-3ASECTION-29 "(MGL-CORE:@MGL-MODEL-PERSISTENCE MGL-PAX:SECTION)"
  [5a3f]: #x-28MGL-RESAMPLE-3ASTRATIFY-20FUNCTION-29 "(MGL-RESAMPLE:STRATIFY FUNCTION)"
  [643d]: #x-28MGL-OPT-3ADO-GRADIENT-SINK-20MGL-PAX-3AMACRO-29 "(MGL-OPT:DO-GRADIENT-SINK MGL-PAX:MACRO)"
  [66a1]: #x-28MGL-OPT-3AN-INSTANCES-20-28MGL-PAX-3AREADER-20MGL-OPT-3AITERATIVE-OPTIMIZER-29-29 "(MGL-OPT:N-INSTANCES (MGL-PAX:READER MGL-OPT:ITERATIVE-OPTIMIZER))"
  [6d2c]: #x-28MGL-3A-40MGL-DEPENDENCIES-20MGL-PAX-3ASECTION-29 "(MGL:@MGL-DEPENDENCIES MGL-PAX:SECTION)"
  [6e12]: #x-28MGL-CORE-3A-40MGL-EXECUTORS-20MGL-PAX-3ASECTION-29 "(MGL-CORE:@MGL-EXECUTORS MGL-PAX:SECTION)"
  [6e54]: #x-28MGL-CORE-3A-40MGL-MONITOR-20MGL-PAX-3ASECTION-29 "(MGL-CORE:@MGL-MONITOR MGL-PAX:SECTION)"
  [6fc3]: #x-28MGL-DATASET-3ASAMPLE-20GENERIC-FUNCTION-29 "(MGL-DATASET:SAMPLE GENERIC-FUNCTION)"
  [72e9]: #x-28MGL-DATASET-3A-40MGL-DATASET-20MGL-PAX-3ASECTION-29 "(MGL-DATASET:@MGL-DATASET MGL-PAX:SECTION)"
  [7471]: #x-28MGL-CORE-3ACOUNTER-20GENERIC-FUNCTION-29 "(MGL-CORE:COUNTER GENERIC-FUNCTION)"
  [74a7]: #x-28MGL-3A-40MGL-BP-20MGL-PAX-3ASECTION-29 "(MGL:@MGL-BP MGL-PAX:SECTION)"
  [7540]: #x-28MGL-RESAMPLE-3A-40MGL-RESAMPLE-MISC-20MGL-PAX-3ASECTION-29 "(MGL-RESAMPLE:@MGL-RESAMPLE-MISC MGL-PAX:SECTION)"
  [76b8]: #x-28MGL-RESAMPLE-3ASAMPLE-FROM-20FUNCTION-29 "(MGL-RESAMPLE:SAMPLE-FROM FUNCTION)"
  [794a]: #x-28MGL-OPT-3A-40MGL-OPT-OPTIMIZER-20MGL-PAX-3ASECTION-29 "(MGL-OPT:@MGL-OPT-OPTIMIZER MGL-PAX:SECTION)"
  [7ae7]: #x-28MGL-RESAMPLE-3ASAMPLE-STRATIFIED-20FUNCTION-29 "(MGL-RESAMPLE:SAMPLE-STRATIFIED FUNCTION)"
  [7f6b]: #x-28MGL-CG-3ACG-ARGS-20-28MGL-PAX-3AACCESSOR-20MGL-CG-3ACG-OPTIMIZER-29-29 "(MGL-CG:CG-ARGS (MGL-PAX:ACCESSOR MGL-CG:CG-OPTIMIZER))"
  [8202]: #x-28MGL-OPT-3AMAP-SEGMENTS-20GENERIC-FUNCTION-29 "(MGL-OPT:MAP-SEGMENTS GENERIC-FUNCTION)"
  [8375]: #x-28MGL-RESAMPLE-3ACROSS-VALIDATE-20FUNCTION-29 "(MGL-RESAMPLE:CROSS-VALIDATE FUNCTION)"
  [83bf]: #x-28MGL-OPT-3AITERATIVE-OPTIMIZER-20CLASS-29 "(MGL-OPT:ITERATIVE-OPTIMIZER CLASS)"
  [8521]: #x-28MGL-DATASET-3AGENERATOR-20-28MGL-PAX-3AREADER-20MGL-DATASET-3AFUNCTION-SAMPLER-29-29 "(MGL-DATASET:GENERATOR (MGL-PAX:READER MGL-DATASET:FUNCTION-SAMPLER))"
  [864e]: #x-28MGL-CG-3ACG-OPTIMIZER-20CLASS-29 "(MGL-CG:CG-OPTIMIZER CLASS)"
  [8729]: #x-28MGL-CG-3A-40MGL-CG-20MGL-PAX-3ASECTION-29 "(MGL-CG:@MGL-CG MGL-PAX:SECTION)"
  [8795]: #x-28MGL-CORE-3ASET-INPUT-20GENERIC-FUNCTION-29 "(MGL-CORE:SET-INPUT GENERIC-FUNCTION)"
  [8966]: #x-28MGL-CORE-3A-40MGL-COUNTER-CLASSES-20MGL-PAX-3ASECTION-29 "(MGL-CORE:@MGL-COUNTER-CLASSES MGL-PAX:SECTION)"
  [8a3b]: #x-28MGL-CORE-3ACOUNTER-VALUES-20GENERIC-FUNCTION-29 "(MGL-CORE:COUNTER-VALUES GENERIC-FUNCTION)"
  [8b7f]: #x-28MGL-CORE-3A-40MGL-MODEL-20MGL-PAX-3ASECTION-29 "(MGL-CORE:@MGL-MODEL MGL-PAX:SECTION)"
  [8fc3]: #x-28MGL-RESAMPLE-3A-40MGL-RESAMPLE-20MGL-PAX-3ASECTION-29 "(MGL-RESAMPLE:@MGL-RESAMPLE MGL-PAX:SECTION)"
  [9112]: #x-28MGL-CORE-3AATTRIBUTES-20-28MGL-PAX-3AACCESSOR-20MGL-CORE-3AATTRIBUTED-29-29 "(MGL-CORE:ATTRIBUTES (MGL-PAX:ACCESSOR MGL-CORE:ATTRIBUTED))"
  [918e]: #x-28MGL-OPT-3AMONITOR-OPTIMIZATION-PERIODICALLY-20FUNCTION-29 "(MGL-OPT:MONITOR-OPTIMIZATION-PERIODICALLY FUNCTION)"
  [93e5]: #x-28MGL-CORE-3ACROSS-ENTROPY-COUNTER-20CLASS-29 "(MGL-CORE:CROSS-ENTROPY-COUNTER CLASS)"
  [94c7]: #x-28MGL-3A-40MGL-BM-20MGL-PAX-3ASECTION-29 "(MGL:@MGL-BM MGL-PAX:SECTION)"
  [950d]: #x-28MGL-CORE-3ALABEL-INDEX-20GENERIC-FUNCTION-29 "(MGL-CORE:LABEL-INDEX GENERIC-FUNCTION)"
  [9589]: #x-28MGL-RESAMPLE-3ASPLIT-FOLD-2FCONT-20FUNCTION-29 "(MGL-RESAMPLE:SPLIT-FOLD/CONT FUNCTION)"
  [9598]: #x-28MGL-CORE-3AMAX-N-STRIPES-20GENERIC-FUNCTION-29 "(MGL-CORE:MAX-N-STRIPES GENERIC-FUNCTION)"
  [97ba]: #x-28MGL-OPT-3AMAP-GRADIENT-SINK-20GENERIC-FUNCTION-29 "(MGL-OPT:MAP-GRADIENT-SINK GENERIC-FUNCTION)"
  [984f]: #x-28MGL-OPT-3A-40MGL-OPT-GRADIENT-SOURCE-20MGL-PAX-3ASECTION-29 "(MGL-OPT:@MGL-OPT-GRADIENT-SOURCE MGL-PAX:SECTION)"
  [998f]: #x-28MGL-CORE-3A-40MGL-COUNTER-20MGL-PAX-3ASECTION-29 "(MGL-CORE:@MGL-COUNTER MGL-PAX:SECTION)"
  [9aa2]: #x-28MGL-GD-3ABATCH-GD-OPTIMIZER-20CLASS-29 "(MGL-GD:BATCH-GD-OPTIMIZER CLASS)"
  [9c36]: #x-28MGL-OPT-3AON-OPTIMIZATION-FINISHED-20-28MGL-PAX-3AACCESSOR-20MGL-OPT-3AITERATIVE-OPTIMIZER-29-29 "(MGL-OPT:ON-OPTIMIZATION-FINISHED (MGL-PAX:ACCESSOR MGL-OPT:ITERATIVE-OPTIMIZER))"
  [9cdc]: #x-28MGL-OPT-3AON-N-INSTANCES-CHANGED-20-28MGL-PAX-3AACCESSOR-20MGL-OPT-3AITERATIVE-OPTIMIZER-29-29 "(MGL-OPT:ON-N-INSTANCES-CHANGED (MGL-PAX:ACCESSOR MGL-OPT:ITERATIVE-OPTIMIZER))"
  [9f93]: #x-28MGL-RESAMPLE-3A-40MGL-RESAMPLE-PARTITIONS-20MGL-PAX-3ASECTION-29 "(MGL-RESAMPLE:@MGL-RESAMPLE-PARTITIONS MGL-PAX:SECTION)"
  [a22b]: #x-28MGL-CORE-3AMONITOR-20CLASS-29 "(MGL-CORE:MONITOR CLASS)"
  [a7de]: #x-28MGL-GD-3AWEIGHT-PENALTY-20-28MGL-PAX-3AACCESSOR-20MGL-GD-3A-3AGD-OPTIMIZER-29-29 "(MGL-GD:WEIGHT-PENALTY (MGL-PAX:ACCESSOR MGL-GD::GD-OPTIMIZER))"
  [aac7]: #x-28MGL-CORE-3ALABEL-INDICES-20GENERIC-FUNCTION-29 "(MGL-CORE:LABEL-INDICES GENERIC-FUNCTION)"
  [af7d]: #x-28MGL-DATASET-3A-40MGL-SAMPLER-20MGL-PAX-3ASECTION-29 "(MGL-DATASET:@MGL-SAMPLER MGL-PAX:SECTION)"
  [b22b]: #x-28MGL-CORE-3AMONITORS-20GENERIC-FUNCTION-29 "(MGL-CORE:MONITORS GENERIC-FUNCTION)"
  [b6ac]: #x-28MGL-GD-3ASEGMENTER-20-28MGL-PAX-3AREADER-20MGL-GD-3ASEGMENTED-GD-OPTIMIZER-29-29 "(MGL-GD:SEGMENTER (MGL-PAX:READER MGL-GD:SEGMENTED-GD-OPTIMIZER))"
  [b73e]: #x-28MGL-CORE-3AMAKE-EXECUTOR-WITH-PARAMETERS-20GENERIC-FUNCTION-29 "(MGL-CORE:MAKE-EXECUTOR-WITH-PARAMETERS GENERIC-FUNCTION)"
  [b8b6]: #x-28MGL-CORE-3AINSTANCE-TO-EXECUTOR-PARAMETERS-20GENERIC-FUNCTION-29 "(MGL-CORE:INSTANCE-TO-EXECUTOR-PARAMETERS GENERIC-FUNCTION)"
  [b96b]: #x-28MGL-COMMON-3AFN-20-28MGL-PAX-3AREADER-20MGL-DIFFUN-3ADIFFUN-29-29 "(MGL-COMMON:FN (MGL-PAX:READER MGL-DIFFUN:DIFFUN))"
  [bca8]: #x-28MGL-OPT-3AMINIMIZE-20FUNCTION-29 "(MGL-OPT:MINIMIZE FUNCTION)"
  [bec0]: #x-28MGL-OPT-3ATERMINATION-20-28MGL-PAX-3AACCESSOR-20MGL-OPT-3AITERATIVE-OPTIMIZER-29-29 "(MGL-OPT:TERMINATION (MGL-PAX:ACCESSOR MGL-OPT:ITERATIVE-OPTIMIZER))"
  [c1b6]: #x-28MGL-CORE-3A-40MGL-CLASSIFICATION-20MGL-PAX-3ASECTION-29 "(MGL-CORE:@MGL-CLASSIFICATION MGL-PAX:SECTION)"
  [c246]: #x-28MGL-CORE-3ALABEL-INDEX-DISTRIBUTIONS-20GENERIC-FUNCTION-29 "(MGL-CORE:LABEL-INDEX-DISTRIBUTIONS GENERIC-FUNCTION)"
  [c27a]: #x-28MGL-CORE-3AMAP-OVER-EXECUTORS-20GENERIC-FUNCTION-29 "(MGL-CORE:MAP-OVER-EXECUTORS GENERIC-FUNCTION)"
  [c54c]: #x-28MGL-OPT-3AINITIALIZE-GRADIENT-SOURCE-2A-20GENERIC-FUNCTION-29 "(MGL-OPT:INITIALIZE-GRADIENT-SOURCE* GENERIC-FUNCTION)"
  [ca85]: #x-28MGL-RESAMPLE-3A-40MGL-RESAMPLE-CV-BAGGING-20MGL-PAX-3ASECTION-29 "(MGL-RESAMPLE:@MGL-RESAMPLE-CV-BAGGING MGL-PAX:SECTION)"
  [cc50]: #x-28MGL-CORE-3A-40MGL-CLASSIFICATION-MONITOR-20MGL-PAX-3ASECTION-29 "(MGL-CORE:@MGL-CLASSIFICATION-MONITOR MGL-PAX:SECTION)"
  [ce14]: #x-28MGL-GD-3AWEIGHT-DECAY-20-28MGL-PAX-3AACCESSOR-20MGL-GD-3A-3AGD-OPTIMIZER-29-29 "(MGL-GD:WEIGHT-DECAY (MGL-PAX:ACCESSOR MGL-GD::GD-OPTIMIZER))"
  [d011]: #x-28MGL-CORE-3A-40MGL-ATTRIBUTES-20MGL-PAX-3ASECTION-29 "(MGL-CORE:@MGL-ATTRIBUTES MGL-PAX:SECTION)"
  [d275]: #x-28MGL-GD-3A-40MGL-GD-PER-WEIGHT-OPTIMIZATION-20MGL-PAX-3ASECTION-29 "(MGL-GD:@MGL-GD-PER-WEIGHT-OPTIMIZATION MGL-PAX:SECTION)"
  [d3e3]: #x-28MGL-CORE-3ABASIC-COUNTER-20CLASS-29 "(MGL-CORE:BASIC-COUNTER CLASS)"
  [d503]: #x-28MGL-DATASET-3AFINISHEDP-20GENERIC-FUNCTION-29 "(MGL-DATASET:FINISHEDP GENERIC-FUNCTION)"
  [d74b]: #x-28MGL-CORE-3APARAMETERIZED-EXECUTOR-CACHE-MIXIN-20CLASS-29 "(MGL-CORE:PARAMETERIZED-EXECUTOR-CACHE-MIXIN CLASS)"
  [dae0]: #x-28MGL-OPT-3AON-OPTIMIZATION-STARTED-20-28MGL-PAX-3AACCESSOR-20MGL-OPT-3AITERATIVE-OPTIMIZER-29-29 "(MGL-OPT:ON-OPTIMIZATION-STARTED (MGL-PAX:ACCESSOR MGL-OPT:ITERATIVE-OPTIMIZER))"
  [dc9d]: #x-28MGL-COMMON-3ABATCH-SIZE-20-28MGL-PAX-3AACCESSOR-20MGL-CG-3ACG-OPTIMIZER-29-29 "(MGL-COMMON:BATCH-SIZE (MGL-PAX:ACCESSOR MGL-CG:CG-OPTIMIZER))"
  [dca7]: #x-28MGL-CORE-3AN-STRIPES-20GENERIC-FUNCTION-29 "(MGL-CORE:N-STRIPES GENERIC-FUNCTION)"
  [df57]: #x-28MGL-GD-3A-40MGL-GD-BATCH-GD-OPTIMIZER-20MGL-PAX-3ASECTION-29 "(MGL-GD:@MGL-GD-BATCH-GD-OPTIMIZER MGL-PAX:SECTION)"
  [e0c8]: #x-28MGL-GD-3AMOMENTUM-TYPE-20-28MGL-PAX-3AREADER-20MGL-GD-3A-3AGD-OPTIMIZER-29-29 "(MGL-GD:MOMENTUM-TYPE (MGL-PAX:READER MGL-GD::GD-OPTIMIZER))"
  [e0d7]: #x-28-22mgl-22-20ASDF-2FSYSTEM-3ASYSTEM-29 "(\"mgl\" ASDF/SYSTEM:SYSTEM)"
  [e57e]: #x-28MGL-RESAMPLE-3AFRACTURE-STRATIFIED-20FUNCTION-29 "(MGL-RESAMPLE:FRACTURE-STRATIFIED FUNCTION)"
  [e6a3]: #x-28MGL-CG-3AON-CG-BATCH-DONE-20-28MGL-PAX-3AACCESSOR-20MGL-CG-3ACG-OPTIMIZER-29-29 "(MGL-CG:ON-CG-BATCH-DONE (MGL-PAX:ACCESSOR MGL-CG:CG-OPTIMIZER))"
  [ec6a]: #x-28MGL-CORE-3AMAKE-CLASSIFICATION-ACCURACY-MONITORS-20FUNCTION-29 "(MGL-CORE:MAKE-CLASSIFICATION-ACCURACY-MONITORS FUNCTION)"
  [ed3d]: #x-28MGL-GD-3AMOMENTUM-20-28MGL-PAX-3AACCESSOR-20MGL-GD-3A-3AGD-OPTIMIZER-29-29 "(MGL-GD:MOMENTUM (MGL-PAX:ACCESSOR MGL-GD::GD-OPTIMIZER))"
  [edd9]: #x-28MGL-RESAMPLE-3ASPLIT-STRATIFIED-20FUNCTION-29 "(MGL-RESAMPLE:SPLIT-STRATIFIED FUNCTION)"
  [f18a]: #x-28MGL-OPT-3A-40MGL-OPT-GRADIENT-SINK-20MGL-PAX-3ASECTION-29 "(MGL-OPT:@MGL-OPT-GRADIENT-SINK MGL-PAX:SECTION)"
  [f1be]: #x-28MGL-CORE-3AMAKE-CROSS-ENTROPY-MONITORS-2A-20GENERIC-FUNCTION-29 "(MGL-CORE:MAKE-CROSS-ENTROPY-MONITORS* GENERIC-FUNCTION)"
  [f1cd]: #x-28MGL-OPT-3ASEGMENTS-20GENERIC-FUNCTION-29 "(MGL-OPT:SEGMENTS GENERIC-FUNCTION)"
  [f4f4]: #x-28MGL-DIFFUN-3ADIFFUN-20CLASS-29 "(MGL-DIFFUN:DIFFUN CLASS)"
  [f56b]: #x-28MGL-DATASET-3AMAX-N-SAMPLES-20-28MGL-PAX-3AACCESSOR-20MGL-DATASET-3AFUNCTION-SAMPLER-29-29 "(MGL-DATASET:MAX-N-SAMPLES (MGL-PAX:ACCESSOR MGL-DATASET:FUNCTION-SAMPLER))"
  [f5e0]: #x-28MGL-CORE-3ACLASSIFICATION-ACCURACY-COUNTER-20CLASS-29 "(MGL-CORE:CLASSIFICATION-ACCURACY-COUNTER CLASS)"
  [f805]: #x-28MGL-OPT-3A-40MGL-OPT-ITERATIVE-OPTIMIZER-20MGL-PAX-3ASECTION-29 "(MGL-OPT:@MGL-OPT-ITERATIVE-OPTIMIZER MGL-PAX:SECTION)"
  [f94f]: #x-28MGL-COMMON-3ABATCH-SIZE-20-28MGL-PAX-3AACCESSOR-20MGL-GD-3A-3AGD-OPTIMIZER-29-29 "(MGL-COMMON:BATCH-SIZE (MGL-PAX:ACCESSOR MGL-GD::GD-OPTIMIZER))"
  [f95f]: #x-28MGL-CORE-3AAPPLY-MONITOR-20GENERIC-FUNCTION-29 "(MGL-CORE:APPLY-MONITOR GENERIC-FUNCTION)"
  [f995]: #x-28MGL-3A-40MGL-OVERVIEW-20MGL-PAX-3ASECTION-29 "(MGL:@MGL-OVERVIEW MGL-PAX:SECTION)"
  [f9f7]: #x-28MGL-CG-3ACG-20FUNCTION-29 "(MGL-CG:CG FUNCTION)"
  [fd45]: #x-28MGL-DATASET-3AN-SAMPLES-20-28MGL-PAX-3AREADER-20MGL-DATASET-3AFUNCTION-SAMPLER-29-29 "(MGL-DATASET:N-SAMPLES (MGL-PAX:READER MGL-DATASET:FUNCTION-SAMPLER))"
  [fdf3]: #x-28MGL-CORE-3AMAP-BATCHES-FOR-MODEL-20FUNCTION-29 "(MGL-CORE:MAP-BATCHES-FOR-MODEL FUNCTION)"
  [fe97]: #x-28MGL-OPT-3A-40MGL-OPT-20MGL-PAX-3ASECTION-29 "(MGL-OPT:@MGL-OPT MGL-PAX:SECTION)"

* * *
###### \[generated by [MGL-PAX](https://github.com/melisgl/mgl-pax)\]
