<a id="x-28MGL-3A-40MGL-MANUAL-20MGL-PAX-3ASECTION-29"></a>
# MGL Manual

## Table of Contents

- [1 `MGL` ASDF System][dbc7]
- [2 Introduction][f7aa]
    - [2.1 Overview][9192]
    - [2.2 Links][00ee]
    - [2.3 Dependencies][e7ea]
    - [2.4 Code Organization][443c]
    - [2.5 Glossary][4a8e]
- [3 Datasets][109e]
    - [3.1 Samplers][7bc3]
        - [3.1.1 Function Sampler][be8d]
- [4 Resampling][a39b]
    - [4.1 Partitions][f790]
    - [4.2 Cross-validation][f17b]
    - [4.3 Bagging][b647]
    - [4.4 CV Bagging][3f9f]
    - [4.5 Miscellaneous Operations][59c2]
- [5 Core][f257]
    - [5.1 Persistence][29a1]
    - [5.2 Batch Processing][ff82]
    - [5.3 Executors][4476]
        - [5.3.1 Parameterized Executor Cache][ada2]
- [6 Monitoring][e668]
    - [6.1 Monitors][c701]
    - [6.2 Measurers][cd3b]
    - [6.3 Counters][be95]
        - [6.3.1 Attributes][6da5]
        - [6.3.2 Counter classes][7ee3]
- [7 Classification][60e3]
    - [7.1 Classification Monitors][c573]
    - [7.2 Classification Measurers][0ba7]
    - [7.3 Classification Counters][6598]
        - [7.3.1 Confusion Matrices][07c7]
- [8 Features][c8db]
    - [8.1 Feature Selection][1b5e]
    - [8.2 Feature Encoding][24aa]
- [9 Gradient Based Optimization][c74a]
    - [9.1 Iterative Optimizer][779d]
    - [9.2 Cost Function][e746]
    - [9.3 Gradient Descent][10e7]
        - [9.3.1 Batch Based Optimizers][2c39]
        - [9.3.2 Segmented GD Optimizer][989a]
        - [9.3.3 Per-weight Optimization][a884]
        - [9.3.4 Utilities][c40e]
    - [9.4 Conjugate Gradient][83e6]
    - [9.5 Extension API][6a6f]
        - [9.5.1 Implementing Optimizers][5748]
        - [9.5.2 Implementing Gradient Sources][c58b]
        - [9.5.3 Implementing Gradient Sinks][a210]
- [10 Differentiable Functions][2981]
- [11 Backpropagation Neural Networks][8788]
    - [11.1 Backprop Overview][56b2]
    - [11.2 Clump API][7a28]
    - [11.3 `BPN`s][d1e0]
        - [11.3.1 Training][0d82]
        - [11.3.2 Monitoring][4f0e]
        - [11.3.3 Feed-Forward Nets][1355]
        - [11.3.4 Recurrent Neural Nets][871e]
    - [11.4 Lumps][9641]
        - [11.4.1 Lump Base Class][3045]
        - [11.4.2 Inputs][207b]
        - [11.4.3 Weight Lump][6872]
        - [11.4.4 Activations][9105]
        - [11.4.5 Activation Functions][5d86]
        - [11.4.6 Losses][93a7]
        - [11.4.7 Stochasticity][aa2e]
        - [11.4.8 Arithmetic][2fe9]
        - [11.4.9 Operations for `RNN`s][51f7]
    - [11.5 Utilities][91f3]
- [12 Boltzmann Machines][332e]
- [13 Gaussian Processes][60b3]
- [14 Natural Language Processing][0d6a]
    - [14.1 Bag of Words][0784]

###### \[in package MGL\]
<a id="x-28-22mgl-22-20ASDF-2FSYSTEM-3ASYSTEM-29"></a>
## 1 `MGL` ASDF System

- Version: 0.1.0
- Description: [`MGL`][dbc7] is a machine learning library for backpropagation
  neural networks, boltzmann machines, gaussian processes and more.
- Licence: MIT, see COPYING.
- Author: Gábor Melis <mega@retes.hu>
- Mailto: [mega@retes.hu](mailto:mega@retes.hu)
- Homepage: [http://melisgl.github.io/mgl](http://melisgl.github.io/mgl)
- Bug tracker: [https://github.com/melisgl/mgl/issues](https://github.com/melisgl/mgl/issues)
- Source control: [GIT](https://github.com/melisgl/mgl.git)

<a id="x-28MGL-3A-40MGL-INTRODUCTION-20MGL-PAX-3ASECTION-29"></a>
## 2 Introduction

<a id="x-28MGL-3A-40MGL-OVERVIEW-20MGL-PAX-3ASECTION-29"></a>
### 2.1 Overview

MGL is a Common Lisp machine learning library by [Gábor
Melis](http://quotenil.com) with some parts originally contributed
by Ravenpack International. It mainly concentrates on various forms
of neural networks (boltzmann machines, feed-forward and recurrent
backprop nets). Most of MGL is built on top of `MGL-MAT` so it has
BLAS and CUDA support.

In general, the focus is on power and performance not on ease of
use. Perhaps one day there will be a cookie cutter interface with
restricted functionality if a reasonable compromise is found between
power and utility.

<a id="x-28MGL-3A-40MGL-LINKS-20MGL-PAX-3ASECTION-29"></a>
### 2.2 Links

Here is the [official repository](https://github.com/melisgl/mgl)
and the [HTML
documentation](http://melisgl.github.io/mgl-pax-world/mgl-manual.html)
for the latest version.

<a id="x-28MGL-3A-40MGL-DEPENDENCIES-20MGL-PAX-3ASECTION-29"></a>
### 2.3 Dependencies

MGL used to rely on [LLA](https://github.com/tpapp/lla) to
interface to BLAS and LAPACK. That's mostly history by now, but
configuration of foreign libraries is still done via `LLA`. See the
README in `LLA` on how to set things up. Note that these days OpenBLAS
is easier to set up and just as fast as ATLAS.

[CL-CUDA](https://github.com/takagi/cl-cuda) and
[MGL-MAT](https://github.com/melisgl/mgl) are the two main
dependencies and also the ones not yet in quicklisp, so just drop
them into `quicklisp/local-projects/`. If there is no suitable GPU
on the system or the CUDA SDK is not installed, MGL will simply
fall back on using BLAS and Lisp code. Wrapping code in
`MGL-MAT:WITH-CUDA*` is basically all that's needed to run on the GPU,
and with `MGL-MAT:CUDA-AVAILABLE-P` one can check whether the GPU is
really being used.

<a id="x-28MGL-3A-40MGL-CODE-ORGANIZATION-20MGL-PAX-3ASECTION-29"></a>
### 2.4 Code Organization

MGL consists of several packages dedicated to different tasks.
For example, package `MGL-RESAMPLE` is about [Resampling][a39b] and
`MGL-GD` is about [Gradient Descent][10e7] and so on. On one hand, having many
packages makes it easier to cleanly separate API and implementation
and also to explore into a specific task. At other times, they can
be a hassle, so the [`MGL`][dbc7] package itself reexports every external
symbol found in all the other packages that make up MGL and
MGL-MAT (see `MGL-MAT:@MAT-MANUAL`) on which it heavily relies.

One exception to this rule is the bundled, but independent
`MGL-GNUPLOT` library.

The built in tests can be run with:

    (ASDF:OOS 'ASDF:TEST-OP '#:MGL)

Note, that most of the tests are rather stochastic and can fail once
in a while.

<a id="x-28MGL-3A-40MGL-GLOSSARY-20MGL-PAX-3ASECTION-29"></a>
### 2.5 Glossary

Ultimately machine learning is about creating **models** of some
domain. The observations in the modelled domain are called
**instances** (also known as examples or samples). Sets of instances
are called **datasets**. Datasets are used when fitting a model or
when making **predictions**. Sometimes the word predictions is too
specific, and the results obtained from applying a model to some
instances are simply called **results**.

<a id="x-28MGL-DATASET-3A-40MGL-DATASET-20MGL-PAX-3ASECTION-29"></a>
## 3 Datasets

###### \[in package MGL-DATASET\]
An instance can often be any kind of object of the user's choice.
It is typically represented by a set of numbers which is called a
feature vector or by a structure holding the feature vector, the
label, etc. A dataset is a [`SEQUENCE`][b9c1] of such instances or a
[Samplers][7bc3] object that produces instances.

<a id="x-28MGL-DATASET-3AMAP-DATASET-20FUNCTION-29"></a>
- [function] **MAP-DATASET** *FN DATASET*

    Call `FN` with each instance in `DATASET`. This is basically equivalent
    to iterating over the elements of a sequence or a sampler (see
    [Samplers][7bc3]).

<a id="x-28MGL-DATASET-3AMAP-DATASETS-20FUNCTION-29"></a>
- [function] **MAP-DATASETS** *FN DATASETS &KEY (IMPUTE NIL IMPUTEP)*

    Call `FN` with a list of instances, one from each dataset in
    `DATASETS`. Return nothing. If `IMPUTE` is specified then iterate until
    the largest dataset is consumed imputing `IMPUTE` for missing values.
    If `IMPUTE` is not specified then iterate until the smallest dataset
    runs out.
    
    ```common-lisp
    (map-datasets #'prin1 '((0 1 2) (:a :b)))
    .. (0 :A)(1 :B)
    
    (map-datasets #'prin1 '((0 1 2) (:a :b)) :impute nil)
    .. (0 :A)(1 :B)(2 NIL)
    ```
    
    It is of course allowed to mix sequences with samplers:
    
    ```common-lisp
    (map-datasets #'prin1
                  (list '(0 1 2)
                        (make-sequence-sampler '(:a :b) :max-n-samples 2)))
    .. (0 :A)(1 :B)
    ```


<a id="x-28MGL-DATASET-3A-40MGL-SAMPLER-20MGL-PAX-3ASECTION-29"></a>
### 3.1 Samplers

Some algorithms do not need random access to the entire dataset and
can work with a stream observations. Samplers are simple generators
providing two functions: [`SAMPLE`][f956] and [`FINISHEDP`][401f].

<a id="x-28MGL-DATASET-3ASAMPLE-20GENERIC-FUNCTION-29"></a>
- [generic-function] **SAMPLE** *SAMPLER*

    If `SAMPLER` has not run out of data (see [`FINISHEDP`][401f])
    `SAMPLE` returns an object that represents a sample from the world to
    be experienced or, in other words, simply something the can be used
    as input for training or prediction. It is not allowed to call
    `SAMPLE` if `SAMPLER` is `FINISHEDP`.

<a id="x-28MGL-DATASET-3AFINISHEDP-20GENERIC-FUNCTION-29"></a>
- [generic-function] **FINISHEDP** *SAMPLER*

    See if `SAMPLER` has run out of examples.

<a id="x-28MGL-DATASET-3ALIST-SAMPLES-20FUNCTION-29"></a>
- [function] **LIST-SAMPLES** *SAMPLER MAX-SIZE*

    Return a list of samples of length at most `MAX-SIZE` or less if
    `SAMPLER` runs out.

<a id="x-28MGL-DATASET-3AMAKE-SEQUENCE-SAMPLER-20FUNCTION-29"></a>
- [function] **MAKE-SEQUENCE-SAMPLER** *SEQ &KEY MAX-N-SAMPLES*

    Create a sampler that returns elements of `SEQ` in their original
    order. If `MAX-N-SAMPLES` is non-nil, then at most `MAX-N-SAMPLES` are
    sampled.

<a id="x-28MGL-DATASET-3AMAKE-RANDOM-SAMPLER-20FUNCTION-29"></a>
- [function] **MAKE-RANDOM-SAMPLER** *SEQ &KEY MAX-N-SAMPLES (REORDER #'MGL-RESAMPLE:SHUFFLE)*

    Create a sampler that returns elements of `SEQ` in random order. If
    `MAX-N-SAMPLES` is non-nil, then at most `MAX-N-SAMPLES` are sampled.
    The first pass over a shuffled copy of `SEQ`, and this copy is
    reshuffled whenever the sampler reaches the end of it. Shuffling is
    performed by calling the `REORDER` function.

<a id="x-28MGL-DATASET-3A-2AINFINITELY-EMPTY-DATASET-2A-20VARIABLE-29"></a>
- [variable] **\*INFINITELY-EMPTY-DATASET\*** *#\<FUNCTION-SAMPLER "infinitely empty" \>*

    This is the default dataset for [`MGL-OPT:MINIMIZE`][46a4]. It's an infinite
    stream of `NIL`s.

<a id="x-28MGL-DATASET-3A-40MGL-SAMPLER-FUNCTION-SAMPLER-20MGL-PAX-3ASECTION-29"></a>
#### 3.1.1 Function Sampler

<a id="x-28MGL-DATASET-3AFUNCTION-SAMPLER-20CLASS-29"></a>
- [class] **FUNCTION-SAMPLER**

    A sampler with a function in its [`GENERATOR`][08ac] that
    produces a stream of samples which may or may not be finite
    depending on [`MAX-N-SAMPLES`][1cab]. [`FINISHEDP`][401f] returns `T` iff `MAX-N-SAMPLES` is
    non-nil, and it's not greater than the number of samples
    generated ([`N-SAMPLES`][bdf9]).
    
        (list-samples (make-instance 'function-sampler
                                     :generator (lambda ()
                                                  (random 10))
                                     :max-n-samples 5)
                      10)
        => (3 5 2 3 3)


<a id="x-28MGL-DATASET-3AGENERATOR-20-28MGL-PAX-3AREADER-20MGL-DATASET-3AFUNCTION-SAMPLER-29-29"></a>
- [reader] **GENERATOR** *FUNCTION-SAMPLER (:GENERATOR)*

    A generator function of no arguments that returns
    the next sample.

<a id="x-28MGL-DATASET-3AMAX-N-SAMPLES-20-28MGL-PAX-3AACCESSOR-20MGL-DATASET-3AFUNCTION-SAMPLER-29-29"></a>
- [accessor] **MAX-N-SAMPLES** *FUNCTION-SAMPLER (:MAX-N-SAMPLES = NIL)*

<a id="x-28MGL-COMMON-3ANAME-20-28MGL-PAX-3AREADER-20MGL-DATASET-3AFUNCTION-SAMPLER-29-29"></a>
- [reader] **NAME** *FUNCTION-SAMPLER (:NAME = NIL)*

    An arbitrary object naming the sampler. Only used
    for printing the sampler object.

<a id="x-28MGL-DATASET-3AN-SAMPLES-20-28MGL-PAX-3AREADER-20MGL-DATASET-3AFUNCTION-SAMPLER-29-29"></a>
- [reader] **N-SAMPLES** *FUNCTION-SAMPLER (:N-SAMPLES = 0)*



<a id="x-28MGL-RESAMPLE-3A-40MGL-RESAMPLE-20MGL-PAX-3ASECTION-29"></a>
## 4 Resampling

###### \[in package MGL-RESAMPLE\]
The focus of this package is on resampling methods such as
cross-validation and bagging which can be used for model evaluation,
model selection, and also as a simple form of ensembling. Data
partitioning and sampling functions are also provided because they
tend to be used together with resampling.

<a id="x-28MGL-RESAMPLE-3A-40MGL-RESAMPLE-PARTITIONS-20MGL-PAX-3ASECTION-29"></a>
### 4.1 Partitions

The following functions partition a dataset (currently only
[`SEQUENCE`][b9c1]s are supported) into a number of partitions. For each
element in the original dataset there is exactly one partition that
contains it.

<a id="x-28MGL-RESAMPLE-3AFRACTURE-20FUNCTION-29"></a>
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
    
    ```common-lisp
    (fracture 5 '(0 1 2 3 4 5 6 7 8 9))
    => ((0 1) (2 3) (4 5) (6 7) (8 9))
    ```
    
    To split into two sequences whose lengths are proportional to 2 and
    3:
    
    ```common-lisp
    (fracture '(2 3) '(0 1 2 3 4 5 6 7 8 9))
    => ((0 1 2 3) (4 5 6 7 8 9))
    ```


<a id="x-28MGL-RESAMPLE-3ASTRATIFY-20FUNCTION-29"></a>
- [function] **STRATIFY** *SEQ &KEY (KEY #'IDENTITY) (TEST #'EQL)*

    Return the list of strata of `SEQ`. `SEQ` is a sequence of elements for
    which the function `KEY` returns the class they belong to. Such
    classes are opaque objects compared for equality with `TEST`. A
    stratum is a sequence of elements with the same (under `TEST`) `KEY`.
    
    ```common-lisp
    (stratify '(0 1 2 3 4 5 6 7 8 9) :key #'evenp)
    => ((0 2 4 6 8) (1 3 5 7 9))
    ```


<a id="x-28MGL-RESAMPLE-3AFRACTURE-STRATIFIED-20FUNCTION-29"></a>
- [function] **FRACTURE-STRATIFIED** *FRACTIONS SEQ &KEY (KEY #'IDENTITY) (TEST #'EQL) WEIGHT*

    Similar to [`FRACTURE`][6f82], but also makes sure that keys are evenly
    distributed among the partitions (see [`STRATIFY`][ba91]). It can be useful
    for classification tasks to partition the data set while keeping the
    distribution of classes the same.
    
    Note that the sets returned are not in random order. In fact, they
    are sorted internally by `KEY`.
    
    For example, to make two splits with approximately the same number
    of even and odd numbers:
    
    ```common-lisp
    (fracture-stratified 2 '(0 1 2 3 4 5 6 7 8 9) :key #'evenp)
    => ((0 2 1 3) (4 6 8 5 7 9))
    ```


<a id="x-28MGL-RESAMPLE-3A-40MGL-RESAMPLE-CROSS-VALIDATION-20MGL-PAX-3ASECTION-29"></a>
### 4.2 Cross-validation

<a id="x-28MGL-RESAMPLE-3ACROSS-VALIDATE-20FUNCTION-29"></a>
- [function] **CROSS-VALIDATE** *DATA FN &KEY (N-FOLDS 5) (FOLDS (ALEXANDRIA:IOTA N-FOLDS)) (SPLIT-FN #'SPLIT-FOLD/MOD) PASS-FOLD*

    Map `FN` over the `FOLDS` of `DATA` split with `SPLIT-FN` and collect the
    results in a list. The simplest demonstration is:
    
    ```common-lisp
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
    
    ```common-lisp
    (cross-validate '(0 1 2 3 4)
                    (lambda (fold test training)
                     (list :fold fold test training))
                    :folds '(2 3)
                    :pass-fold t)
    => ((:fold 2 (2) (0 1 3 4))
        (:fold 3 (3) (0 1 2 4)))
    ```
    
    Finally, the way the data is split can be customized. By default
    [`SPLIT-FOLD/MOD`][5ded] is called with the arguments `DATA`, the fold (from
    among `FOLDS`) and `N-FOLDS`. `SPLIT-FOLD/MOD` returns two values which
    are then passed on to `FN`. One can use [`SPLIT-FOLD/CONT`][5293] or
    [`SPLIT-STRATIFIED`][8cb8] or any other function that works with these
    arguments. The only real constraint is that `FN` has to take as many
    arguments (plus the fold argument if `PASS-FOLD`) as `SPLIT-FN`
    returns.

<a id="x-28MGL-RESAMPLE-3ASPLIT-FOLD-2FMOD-20FUNCTION-29"></a>
- [function] **SPLIT-FOLD/MOD** *SEQ FOLD N-FOLDS*

    Partition `SEQ` into two sequences: one with elements of `SEQ` with
    indices whose remainder is `FOLD` when divided with `N-FOLDS`, and a
    second one with the rest. The second one is the larger set. The
    order of elements remains stable. This function is suitable as the
    `SPLIT-FN` argument of [`CROSS-VALIDATE`][9524].

<a id="x-28MGL-RESAMPLE-3ASPLIT-FOLD-2FCONT-20FUNCTION-29"></a>
- [function] **SPLIT-FOLD/CONT** *SEQ FOLD N-FOLDS*

    Imagine dividing `SEQ` into `N-FOLDS` subsequences of the same
    size (bar rounding). Return the subsequence of index `FOLD` as the
    first value and the all the other subsequences concatenated into one
    as the second value. The order of elements remains stable. This
    function is suitable as the `SPLIT-FN` argument of [`CROSS-VALIDATE`][9524].

<a id="x-28MGL-RESAMPLE-3ASPLIT-STRATIFIED-20FUNCTION-29"></a>
- [function] **SPLIT-STRATIFIED** *SEQ FOLD N-FOLDS &KEY (KEY #'IDENTITY) (TEST #'EQL) WEIGHT*

    Split `SEQ` into `N-FOLDS` partitions (as in [`FRACTURE-STRATIFIED`][627a]).
    Return the partition of index `FOLD` as the first value, and the
    concatenation of the rest as the second value. This function is
    suitable as the `SPLIT-FN` argument of [`CROSS-VALIDATE`][9524] (mostly likely
    as a closure with `KEY`, `TEST`, `WEIGHT` bound).

<a id="x-28MGL-RESAMPLE-3A-40MGL-RESAMPLE-BAGGING-20MGL-PAX-3ASECTION-29"></a>
### 4.3 Bagging

<a id="x-28MGL-RESAMPLE-3ABAG-20FUNCTION-29"></a>
- [function] **BAG** *SEQ FN &KEY (RATIO 1) N WEIGHT (REPLACEMENT T) KEY (TEST #'EQL) (RANDOM-STATE \*RANDOM-STATE\*)*

    Sample from `SEQ` with [`SAMPLE-FROM`][86fd] (passing `RATIO`, `WEIGHT`,
    `REPLACEMENT`), or [`SAMPLE-STRATIFIED`][aee6] if `KEY` is not `NIL`. Call `FN` with
    the sample. If `N` is `NIL` then keep repeating this until `FN` performs a
    non-local exit. Else `N` must be a non-negative integer, `N` iterations
    will be performed, the primary values returned by `FN` collected into
    a list and returned. See `SAMPLE-FROM` and `SAMPLE-STRATIFIED` for
    examples.

<a id="x-28MGL-RESAMPLE-3ASAMPLE-FROM-20FUNCTION-29"></a>
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


<a id="x-28MGL-RESAMPLE-3ASAMPLE-STRATIFIED-20FUNCTION-29"></a>
- [function] **SAMPLE-STRATIFIED** *RATIO SEQ &KEY WEIGHT REPLACEMENT (KEY #'IDENTITY) (TEST #'EQL) (RANDOM-STATE \*RANDOM-STATE\*)*

    Like [`SAMPLE-FROM`][86fd] but makes sure that the weighted proportion of
    classes in the result is approximately the same as the proportion in
    `SEQ`. See [`STRATIFY`][ba91] for the description of `KEY` and `TEST`.

<a id="x-28MGL-RESAMPLE-3A-40MGL-RESAMPLE-CV-BAGGING-20MGL-PAX-3ASECTION-29"></a>
### 4.4 CV Bagging

<a id="x-28MGL-RESAMPLE-3ABAG-CV-20FUNCTION-29"></a>
- [function] **BAG-CV** *DATA FN &KEY N (N-FOLDS 5) (FOLDS (ALEXANDRIA:IOTA N-FOLDS)) (SPLIT-FN #'SPLIT-FOLD/MOD) PASS-FOLD (RANDOM-STATE \*RANDOM-STATE\*)*

    Perform cross-validation on different shuffles of `DATA` `N` times and
    collect the results. Since [`CROSS-VALIDATE`][9524] collects the return values
    of `FN`, the return value of this function is a list of lists of `FN`
    results. If `N` is `NIL`, don't collect anything just keep doing
    repeated CVs until `FN` performs a non-local exit.
    
    The following example simply collects the test and training sets for
    2-fold CV repeated 3 times with shuffled data:
    
    ```commonlisp
    ;;; This is non-deterministic.
    (bag-cv '(0 1 2 3 4) #'list :n 3 :n-folds 2)
    => ((((2 3 4) (1 0))
         ((1 0) (2 3 4)))
        (((2 1 0) (4 3))
         ((4 3) (2 1 0)))
        (((1 0 3) (2 4))
         ((2 4) (1 0 3))))
    ```
    
    CV bagging is useful when a single CV is not producing stable
    results. As an ensemble method, CV bagging has the advantage over
    bagging that each example will occur the same number of times and
    after the first CV is complete there is a complete but less reliable
    estimate for each example which gets refined by further CVs.

<a id="x-28MGL-RESAMPLE-3A-40MGL-RESAMPLE-MISC-20MGL-PAX-3ASECTION-29"></a>
### 4.5 Miscellaneous Operations

<a id="x-28MGL-RESAMPLE-3ASPREAD-STRATA-20FUNCTION-29"></a>
- [function] **SPREAD-STRATA** *SEQ &KEY (KEY #'IDENTITY) (TEST #'EQL)*

    Return a sequence that's a reordering of `SEQ` such that elements
    belonging to different strata (under `KEY` and `TEST`, see [`STRATIFY`][ba91]) are
    distributed evenly. The order of elements belonging to the same
    stratum is unchanged.
    
    For example, to make sure that even and odd numbers are distributed
    evenly:
    
    ```common-lisp
    (spread-strata '(0 2 4 6 8 1 3 5 7 9) :key #'evenp)
    => (0 1 2 3 4 5 6 7 8 9)
    ```
    
    Same thing with unbalanced classes:
    
    ```common-lisp
    (spread-strata (vector 0 2 3 5 6 1 4)
                   :key (lambda (x)
                          (if (member x '(1 4))
                              t
                              nil)))
    => #(0 1 2 3 4 5 6)
    ```


<a id="x-28MGL-RESAMPLE-3AZIP-EVENLY-20FUNCTION-29"></a>
- [function] **ZIP-EVENLY** *SEQS &KEY RESULT-TYPE*

    Make a single sequence out of the sequences in `SEQS` so that in the
    returned sequence indices of elements belonging to the same source
    sequence are spread evenly across the whole range. The result is a
    list is `RESULT-TYPE` is `LIST`([`0`][592c] [`1`][98f9]), it's a vector if `RESULT-TYPE` is `VECTOR`([`0`][5003] [`1`][e2a3]).
    If `RESULT-TYPE` is `NIL`, then it's determined by the type of the first
    sequence in `SEQS`.
    
    ```common-lisp
    (zip-evenly '((0 2 4) (1 3)))
    => (0 1 2 3 4)
    ```


<a id="x-28MGL-CORE-3A-40MGL-CORE-20MGL-PAX-3ASECTION-29"></a>
## 5 Core

###### \[in package MGL-CORE\]
<a id="x-28MGL-CORE-3A-40MGL-PERSISTENCE-20MGL-PAX-3ASECTION-29"></a>
### 5.1 Persistence

<a id="x-28MGL-CORE-3ALOAD-STATE-20FUNCTION-29"></a>
- [function] **LOAD-STATE** *FILENAME OBJECT*

    Load weights of `OBJECT` from `FILENAME`. Return `OBJECT`.

<a id="x-28MGL-CORE-3ASAVE-STATE-20FUNCTION-29"></a>
- [function] **SAVE-STATE** *FILENAME OBJECT &KEY (IF-EXISTS :ERROR) (ENSURE T)*

    Save weights of `OBJECT` to `FILENAME`. If `ENSURE`, then
    [`ENSURE-DIRECTORIES-EXIST`][e4b0] is called on `FILENAME`. `IF-EXISTS` is passed
    on to [`OPEN`][117a]. Return `OBJECT`.

<a id="x-28MGL-CORE-3AREAD-STATE-20FUNCTION-29"></a>
- [function] **READ-STATE** *OBJECT STREAM*

    Read the weights of `OBJECT` from the bivalent `STREAM` where weights
    mean the learnt parameters. There is currently no sanity checking of
    data which will most certainly change in the future together with
    the serialization format. Return `OBJECT`.

<a id="x-28MGL-CORE-3AWRITE-STATE-20FUNCTION-29"></a>
- [function] **WRITE-STATE** *OBJECT STREAM*

    Write weight of `OBJECT` to the bivalent `STREAM`. Return `OBJECT`.

<a id="x-28MGL-CORE-3AREAD-STATE-2A-20GENERIC-FUNCTION-29"></a>
- [generic-function] **READ-STATE\*** *OBJECT STREAM CONTEXT*

    This is the extension point for [`READ-STATE`][8148]. It is
    guaranteed that primary `READ-STATE*` methods will be called only once
    for each `OBJECT` (under [`EQ`][a1d4]). `CONTEXT` is an opaque object and must be
    passed on to any recursive `READ-STATE*` calls.

<a id="x-28MGL-CORE-3AWRITE-STATE-2A-20GENERIC-FUNCTION-29"></a>
- [generic-function] **WRITE-STATE\*** *OBJECT STREAM CONTEXT*

    This is the extension point for [`WRITE-STATE`][95fe]. It is
    guaranteed that primary `WRITE-STATE*` methods will be called only
    once for each `OBJECT` (under [`EQ`][a1d4]). `CONTEXT` is an opaque object and must
    be passed on to any recursive `WRITE-STATE*` calls.

<a id="x-28MGL-CORE-3A-40MGL-MODEL-STRIPE-20MGL-PAX-3ASECTION-29"></a>
### 5.2 Batch Processing

Processing instances one by one during training or prediction can
be slow. The models that support batch processing for greater
efficiency are said to be *striped*.

Typically, during or after creating a model, one sets [`MAX-N-STRIPES`][16c4]
on it a positive integer. When a batch of instances is to be fed to
the model it is first broken into subbatches of length that's at
most `MAX-N-STRIPES`. For each subbatch, [`SET-INPUT`][0c9e] (FIXDOC) is called
and a before method takes care of setting [`N-STRIPES`][8dd7] to the actual
number of instances in the subbatch. When `MAX-N-STRIPES` is set
internal data structures may be resized which is an expensive
operation. Setting `N-STRIPES` is a comparatively cheap operation,
often implemented as matrix reshaping.

Note that for models made of different parts (for example,
[`MGL-BP:BPN`][5187] consists of [`MGL-BP:LUMP`][c1ac]s) , setting these
values affects the constituent parts, but one should never change
the number stripes of the parts directly because that would lead to
an internal inconsistency in the model.

<a id="x-28MGL-CORE-3AMAX-N-STRIPES-20GENERIC-FUNCTION-29"></a>
- [generic-function] **MAX-N-STRIPES** *OBJECT*

    The number of stripes with which the `OBJECT` is
    capable of dealing simultaneously. 

<a id="x-28MGL-CORE-3ASET-MAX-N-STRIPES-20GENERIC-FUNCTION-29"></a>
- [generic-function] **SET-MAX-N-STRIPES** *MAX-N-STRIPES OBJECT*

    Allocate the necessary stuff to allow for
    `MAX-N-STRIPES` number of stripes to be worked with simultaneously in
    `OBJECT`. This is called when `MAX-N-STRIPES` is [`SETF`][17b7]'ed.

<a id="x-28MGL-CORE-3AN-STRIPES-20GENERIC-FUNCTION-29"></a>
- [generic-function] **N-STRIPES** *OBJECT*

    The number of stripes currently present in `OBJECT`.
    This is at most [`MAX-N-STRIPES`][16c4].

<a id="x-28MGL-CORE-3ASET-N-STRIPES-20GENERIC-FUNCTION-29"></a>
- [generic-function] **SET-N-STRIPES** *N-STRIPES OBJECT*

    Set the number of stripes (out of [`MAX-N-STRIPES`][16c4])
    that are in use in `OBJECT`. This is called when `N-STRIPES` is
    [`SETF`][17b7]'ed.

<a id="x-28MGL-CORE-3AWITH-STRIPES-20MGL-PAX-3AMACRO-29"></a>
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
    indexing ([`NODES`][cc1c]) is not known to `WITH-STRIPES`. In fact, for lumps
    the same stripe indices work with `NODES` and [`MGL-BP:DERIVATIVES`][a81b].

<a id="x-28MGL-CORE-3ASTRIPE-START-20GENERIC-FUNCTION-29"></a>
- [generic-function] **STRIPE-START** *STRIPE OBJECT*

    Return the start index of `STRIPE` in some array or
    matrix of `OBJECT`.

<a id="x-28MGL-CORE-3ASTRIPE-END-20GENERIC-FUNCTION-29"></a>
- [generic-function] **STRIPE-END** *STRIPE OBJECT*

    Return the end index (exclusive) of `STRIPE` in some
    array or matrix of `OBJECT`.

<a id="x-28MGL-CORE-3ASET-INPUT-20GENERIC-FUNCTION-29"></a>
- [generic-function] **SET-INPUT** *INSTANCES MODEL*

    Set `INSTANCES` as inputs in `MODEL`. `INSTANCES` is
    always a [`SEQUENCE`][b9c1] of instances even for models not capable of batch
    operation. It sets [`N-STRIPES`][8dd7] to ([`LENGTH`][f466] `INSTANCES`) in a `:BEFORE`
    method.

<a id="x-28MGL-CORE-3AMAP-BATCHES-FOR-MODEL-20FUNCTION-29"></a>
- [function] **MAP-BATCHES-FOR-MODEL** *FN DATASET MODEL*

    Call `FN` with batches of instances from `DATASET` suitable for `MODEL`.
    The number of instances in a batch is [`MAX-N-STRIPES`][16c4] of `MODEL` or less
    if there are no more instances left.

<a id="x-28MGL-CORE-3ADO-BATCHES-FOR-MODEL-20MGL-PAX-3AMACRO-29"></a>
- [macro] **DO-BATCHES-FOR-MODEL** *(BATCH (DATASET MODEL)) &BODY BODY*

    Convenience macro over [`MAP-BATCHES-FOR-MODEL`][5fdc].

<a id="x-28MGL-CORE-3A-40MGL-EXECUTORS-20MGL-PAX-3ASECTION-29"></a>
### 5.3 Executors

<a id="x-28MGL-CORE-3AMAP-OVER-EXECUTORS-20GENERIC-FUNCTION-29"></a>
- [generic-function] **MAP-OVER-EXECUTORS** *FN INSTANCES PROTOTYPE-EXECUTOR*

    Divide `INSTANCES` between executors that perform the
    same function as `PROTOTYPE-EXECUTOR` and call `FN` with the instances
    and the executor for which the instances are.
    
    Some objects conflate function and call: the forward pass of a
    [`MGL-BP:BPN`][5187] computes output from inputs so it is like a
    function but it also doubles as a function call in the sense that
    the bpn (function) object changes state during the computation of
    the output. Hence not even the forward pass of a bpn is thread safe.
    There is also the restriction that all inputs must be of the same
    size.
    
    For example, if we have a function that builds bpn a for an input of
    a certain size, then we can create a factory that creates bpns for a
    particular call. The factory probably wants to keep the weights the
    same though. In [Parameterized Executor Cache][ada2],
    [`MAKE-EXECUTOR-WITH-PARAMETERS`][331b] is this factory.
    
    Parallelization of execution is another possibility
    `MAP-OVER-EXECUTORS` allows, but there is no prebuilt solution for it,
    yet.
    
    The default implementation simply calls `FN` with `INSTANCES` and
    `PROTOTYPE-EXECUTOR`.

<a id="x-28MGL-CORE-3ADO-EXECUTORS-20MGL-PAX-3AMACRO-29"></a>
- [macro] **DO-EXECUTORS** *(INSTANCES OBJECT) &BODY BODY*

    Convenience macro on top of [`MAP-OVER-EXECUTORS`][b01b].

<a id="x-28MGL-CORE-3A-40MGL-PARAMETERIZED-EXECUTOR-CACHE-20MGL-PAX-3ASECTION-29"></a>
#### 5.3.1 Parameterized Executor Cache

<a id="x-28MGL-CORE-3APARAMETERIZED-EXECUTOR-CACHE-MIXIN-20CLASS-29"></a>
- [class] **PARAMETERIZED-EXECUTOR-CACHE-MIXIN**

    Mix this into a model, implement
    [`INSTANCE-TO-EXECUTOR-PARAMETERS`][0078] and [`MAKE-EXECUTOR-WITH-PARAMETERS`][331b]
    and [`DO-EXECUTORS`][f98e] will be to able build executors suitable for
    different instances. The canonical example is using a BPN to compute
    the means and convariances of a gaussian process. Since each
    instance is made of a variable number of observations, the size of
    the input is not constant, thus we have a bpn (an executor) for each
    input dimension (the parameters).

<a id="x-28MGL-CORE-3AMAKE-EXECUTOR-WITH-PARAMETERS-20GENERIC-FUNCTION-29"></a>
- [generic-function] **MAKE-EXECUTOR-WITH-PARAMETERS** *PARAMETERS CACHE*

    Create a new executor for `PARAMETERS`. `CACHE` is a
    [`PARAMETERIZED-EXECUTOR-CACHE-MIXIN`][d3b2]. In the BPN gaussian process
    example, `PARAMETERS` would be a list of input dimensions.

<a id="x-28MGL-CORE-3AINSTANCE-TO-EXECUTOR-PARAMETERS-20GENERIC-FUNCTION-29"></a>
- [generic-function] **INSTANCE-TO-EXECUTOR-PARAMETERS** *INSTANCE CACHE*

    Return the parameters for an executor able to
    handle `INSTANCE`. Called by [`MAP-OVER-EXECUTORS`][b01b] on `CACHE` (that's a
    [`PARAMETERIZED-EXECUTOR-CACHE-MIXIN`][d3b2]). The returned parameters are
    keys in an [`EQUAL`][96d0] parameters->executor hash table.

<a id="x-28MGL-CORE-3A-40MGL-MONITORING-20MGL-PAX-3ASECTION-29"></a>
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
pass. [`MGL-BP:BP-LEARNER`][00a0] has a [`MONITORS`][6202] event hook corresponding to the moment after
backpropagating the gradients. Suppose we are interested in how the
training cost evolves:

    (push (make-instance 'monitor
                         :measurer (lambda (instances bpn)
                                     (declare (ignore instances))
                                     (mgl-bp:cost bpn))
                         :counter (make-instance 'basic-counter))
          (monitors learner))

During training, this monitor will track the cost of training
examples behind the scenes. If we want to print and reset this
monitor periodically we can put another monitor on
[`MGL-OPT:ITERATIVE-OPTIMIZER`][8da0]'s [`MGL-OPT:ON-N-INSTANCES-CHANGED`][4f0b]
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
[`APPLY-MONITOR`][bbdf] is implemented on it with the appropriate signature.
Also note that the [`ZEROP`][afff] + `MOD`([`0`][6c7f] [`1`][9358]) logic is fragile, so you will likely
want to use [`MGL-OPT:MONITOR-OPTIMIZATION-PERIODICALLY`][4528] instead of
doing the above.

So that's the general idea. Concrete events are documented where
they are signalled. Often there are task specific utilities that
create a reasonable set of default monitors (see
[Classification Monitors][c573]).

<a id="x-28MGL-CORE-3AAPPLY-MONITORS-20FUNCTION-29"></a>
- [function] **APPLY-MONITORS** *MONITORS &REST ARGUMENTS*

    Call [`APPLY-MONITOR`][bbdf] on each monitor in `MONITORS` and `ARGUMENTS`. This
    is how an event is fired.

<a id="x-28MGL-CORE-3AAPPLY-MONITOR-20GENERIC-FUNCTION-29"></a>
- [generic-function] **APPLY-MONITOR** *MONITOR &REST ARGUMENTS*

    Apply `MONITOR` to `ARGUMENTS`. This sound fairly
    generic, because it is. `MONITOR` can be anything, even a simple
    function or symbol, in which case this is just [`CL:APPLY`][376e]. See
    [Monitors][c701] for more.

<a id="x-28MGL-CORE-3ACOUNTER-20GENERIC-FUNCTION-29"></a>
- [generic-function] **COUNTER** *MONITOR*

    Return an object representing the state of `MONITOR`
    or `NIL`, if it doesn't have any (say because it's a simple logging
    function). Most monitors have counters into which they accumulate
    results until they are printed and reset. See [Counters][be95] for
    more.

<a id="x-28MGL-CORE-3AMONITOR-MODEL-RESULTS-20FUNCTION-29"></a>
- [function] **MONITOR-MODEL-RESULTS** *FN DATASET MODEL MONITORS*

    Call `FN` with batches of instances from `DATASET` until it runs
    out (as in [`DO-BATCHES-FOR-MODEL`][faaa]). `FN` is supposed to apply `MODEL` to
    the batch and return some kind of result (for neural networks, the
    result is the model state itself). Apply `MONITORS` to each batch and
    the result returned by `FN` for that batch. Finally, return the list
    of counters of `MONITORS`.
    
    The purpose of this function is to collect various results and
    statistics (such as error measures) efficiently by applying the
    model only once, leaving extraction of quantities of interest from
    the model's results to `MONITORS`.
    
    See the model specific versions of this functions such as
    [`MGL-BP:MONITOR-BPN-RESULTS`][0933].

<a id="x-28MGL-CORE-3AMONITORS-20GENERIC-FUNCTION-29"></a>
- [generic-function] **MONITORS** *OBJECT*

    Return monitors associated with `OBJECT`. See various
    methods such as [`MONITORS`][6202] for more
    documentation.

<a id="x-28MGL-CORE-3A-40MGL-MONITOR-20MGL-PAX-3ASECTION-29"></a>
### 6.1 Monitors

<a id="x-28MGL-CORE-3AMONITOR-20CLASS-29"></a>
- [class] **MONITOR**

    A monitor that has another monitor called [`MEASURER`][eb05]
    embedded in it. When this monitor is applied, it applies the
    measurer and passes the returned values to [`ADD-TO-COUNTER`][62de] called on
    its [`COUNTER`][a077] slot. One may further specialize [`APPLY-MONITOR`][bbdf] to change
    that.
    
    This class is useful when the same event monitor is applied
    repeatedly over a period and its results must be aggregated such as
    when training statistics are being tracked or when predictions are
    begin made. Note that the monitor must be compatible with the event
    it handles. That is, the embedded `MEASURER` must be prepared to take
    the arguments that are documented to come with the event.

<a id="x-28MGL-CORE-3AMEASURER-20-28MGL-PAX-3AREADER-20MGL-CORE-3AMONITOR-29-29"></a>
- [reader] **MEASURER** *MONITOR (:MEASURER)*

    This must be a monitor itself which only means
    that [`APPLY-MONITOR`][bbdf] is defined on it (but see [Monitoring][e668]). The
    returned values are aggregated by [`COUNTER`][5752]. See
    [Measurers][cd3b] for a library of measurers.

<a id="x-28MGL-CORE-3ACOUNTER-20-28MGL-PAX-3AREADER-20MGL-CORE-3AMONITOR-29-29"></a>
- [reader] **COUNTER** *MONITOR (:COUNTER)*

    The `COUNTER` of a monitor carries out the
    aggregation of results returned by [`MEASURER`][eb05]. The See [Counters][be95]
    for a library of counters.

<a id="x-28MGL-CORE-3A-40MGL-MEASURER-20MGL-PAX-3ASECTION-29"></a>
### 6.2 Measurers

[`MEASURER`][eb05] is a part of [`MONITOR`][7068] objects, an embedded monitor that
computes a specific quantity (e.g. classification accuracy) from the
arguments of event it is applied to (e.g. the model results).
Measurers are often implemented by combining some kind of model
specific extractor with a generic measurer function.

All generic measurer functions return their results as multiple
values matching the arguments of [`ADD-TO-COUNTER`][62de] for a counter of a
certain type (see [Counters][be95]) so as to make them easily used in a
`MONITOR`:

    (multiple-value-call #'add-to-counter <some-counter>
                         <call-to-some-measurer>)

The counter class compatible with the measurer this way is noted for
each function.

For a list of measurer functions see [Classification Measurers][0ba7].

<a id="x-28MGL-CORE-3A-40MGL-COUNTER-20MGL-PAX-3ASECTION-29"></a>
### 6.3 Counters

<a id="x-28MGL-CORE-3AADD-TO-COUNTER-20GENERIC-FUNCTION-29"></a>
- [generic-function] **ADD-TO-COUNTER** *COUNTER &REST ARGS*

    Add `ARGS` to `COUNTER` in some way. See specialized
    methods for type specific documentation. The kind of arguments to be
    supported is the what the measurer functions (see [Measurers][cd3b])
    intended to be paired with the counter return as multiple values.

<a id="x-28MGL-CORE-3ACOUNTER-VALUES-20GENERIC-FUNCTION-29"></a>
- [generic-function] **COUNTER-VALUES** *COUNTER*

    Return any number of values representing the state
    of `COUNTER`. See specialized methods for type specific
    documentation.

<a id="x-28MGL-CORE-3ACOUNTER-RAW-VALUES-20GENERIC-FUNCTION-29"></a>
- [generic-function] **COUNTER-RAW-VALUES** *COUNTER*

    Return any number of values representing the state
    of `COUNTER` in such a way that passing the returned values as
    arguments [`ADD-TO-COUNTER`][62de] on a fresh instance of the same type
    recreates the original state.

<a id="x-28MGL-CORE-3ARESET-COUNTER-20GENERIC-FUNCTION-29"></a>
- [generic-function] **RESET-COUNTER** *COUNTER*

    Restore state of `COUNTER` to what it was just after
    creation.

<a id="x-28MGL-CORE-3A-40MGL-ATTRIBUTES-20MGL-PAX-3ASECTION-29"></a>
#### 6.3.1 Attributes

<a id="x-28MGL-CORE-3AATTRIBUTED-20CLASS-29"></a>
- [class] **ATTRIBUTED**

    This is a utility class that all counters subclass.
    The [`ATTRIBUTES`][cc37] plist can hold basically anything. Currently the
    attributes are only used when printing and they can be specified by
    the user. The monitor maker functions such as those in
    [Classification Monitors][c573] also add attributes of their own to the
    counters they create.
    
    With the `:PREPEND-ATTRIBUTES` initarg when can easily add new
    attributes without clobbering the those in the `:INITFORM`, (`:TYPE`
    "rmse") in this case.
    
        (princ (make-instance 'rmse-counter
                              :prepend-attributes '(:event "pred."
                                                    :dataset "test")))
        ;; pred. test rmse: 0.000e+0 (0)
        => #<RMSE-COUNTER pred. test rmse: 0.000e+0 (0)>


<a id="x-28MGL-CORE-3AATTRIBUTES-20-28MGL-PAX-3AACCESSOR-20MGL-CORE-3AATTRIBUTED-29-29"></a>
- [accessor] **ATTRIBUTES** *ATTRIBUTED (:ATTRIBUTES = NIL)*

    A plist of attribute keys and values.

<a id="x-28MGL-COMMON-3ANAME-20-28METHOD-20NIL-20-28MGL-CORE-3AATTRIBUTED-29-29-29"></a>
- [method] **NAME** *(ATTRIBUTED ATTRIBUTED)*

    Return a string assembled from the values of the [`ATTRIBUTES`][cc37] of
    `ATTRIBUTED`. If there are multiple entries with the same key, then
    they are printed near together.
    
    Values may be padded according to an enclosing
    [`WITH-PADDED-ATTRIBUTE-PRINTING`][2e8b].

<a id="x-28MGL-CORE-3AWITH-PADDED-ATTRIBUTE-PRINTING-20MGL-PAX-3AMACRO-29"></a>
- [macro] **WITH-PADDED-ATTRIBUTE-PRINTING** *(ATTRIBUTEDS) &BODY BODY*

    Note the width of values for each attribute key which is the number
    of characters in the value's [`PRINC-TO-STRING`][d397]'ed representation. In
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


<a id="x-28MGL-CORE-3ALOG-PADDED-20FUNCTION-29"></a>
- [function] **LOG-PADDED** *ATTRIBUTEDS*

    Log (see `LOG-MSG`) `ATTRIBUTEDS` non-escaped (as in [`PRINC`][f47b] or ~A) with
    the output being as table-like as possible.

<a id="x-28MGL-CORE-3A-40MGL-COUNTER-CLASSES-20MGL-PAX-3ASECTION-29"></a>
#### 6.3.2 Counter classes

In addition to the really basic ones here, also see
[Classification Counters][6598].

<a id="x-28MGL-CORE-3ABASIC-COUNTER-20CLASS-29"></a>
- [class] **BASIC-COUNTER** *[ATTRIBUTED][9715]*

    A simple counter whose [`ADD-TO-COUNTER`][62de] takes two
    additional parameters: an increment to the internal sums of called
    the [`NUMERATOR`][c1d9] and [`DENOMINATOR`][1abb]. [`COUNTER-VALUES`][20e8] returns two
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


<a id="x-28MGL-CORE-3ARMSE-COUNTER-20CLASS-29"></a>
- [class] **RMSE-COUNTER** *[BASIC-COUNTER][5979]*

    A [`BASIC-COUNTER`][5979] with whose nominator accumulates
    the square of some statistics. It has the attribute `:TYPE` "rmse".
    [`COUNTER-VALUES`][20e8] returns the square root of what `BASIC-COUNTER`'s
    `COUNTER-VALUES` would return.
    
        (let ((counter (make-instance 'rmse-counter)))
          (add-to-counter counter (+ (* 3 3) (* 4 4)) 2)
          counter)
        => #<RMSE-COUNTER rmse: 3.53553e+0 (2)>


<a id="x-28MGL-CORE-3ACONCAT-COUNTER-20CLASS-29"></a>
- [class] **CONCAT-COUNTER** *[ATTRIBUTED][9715]*

    A counter that simply concatenates
    sequences.
    
    \`\`\`cl-transcript
    (let ((counter (make-instance 'concat-counter)))
      (add-to-counter counter '(1 2 3) #(4 5))
      (add-to-counter counter '(6 7))
      (counter-values counter))
    => (1 2 3 4 5 6 7)
    \`\`\`\`

<a id="x-28MGL-CORE-3ACONCATENATION-TYPE-20-28MGL-PAX-3AREADER-20MGL-CORE-3ACONCAT-COUNTER-29-29"></a>
- [reader] **CONCATENATION-TYPE** *CONCAT-COUNTER (:CONCATENATION-TYPE = 'LIST)*

    A type designator suitable as the RESULT-TYPE
    argument to [`CONCATENATE`][8cb9].

<a id="x-28MGL-CORE-3A-40MGL-CLASSIFICATION-20MGL-PAX-3ASECTION-29"></a>
## 7 Classification

###### \[in package MGL-CORE\]
To be able to measure classification related quantities, we need to
define what the label of an instance is. Customization is possible
by implementing a method for a specific type of instance, but these
functions only ever appear as defaults that can be overridden.

<a id="x-28MGL-CORE-3ALABEL-INDEX-20GENERIC-FUNCTION-29"></a>
- [generic-function] **LABEL-INDEX** *INSTANCE*

    Return the label of `INSTANCE` as a non-negative
    integer.

<a id="x-28MGL-CORE-3ALABEL-INDEX-DISTRIBUTION-20GENERIC-FUNCTION-29"></a>
- [generic-function] **LABEL-INDEX-DISTRIBUTION** *INSTANCE*

    Return a one dimensional array of probabilities
    representing the distribution of labels. The probability of the
    label with [`LABEL-INDEX`][cc80] `I` is element at index `I` of the returned
    arrray.

The following two functions are basically the same as the previous
two, but in batch mode: they return a sequence of label indices or
distributions. These are called on results produced by models.
Implement these for a model and the monitor maker functions below
will automatically work. See FIXDOC: for bpn and boltzmann.

<a id="x-28MGL-CORE-3ALABEL-INDICES-20GENERIC-FUNCTION-29"></a>
- [generic-function] **LABEL-INDICES** *RESULTS*

    Return a sequence of label indices for `RESULTS`
    produced by some model for a batch of instances. This is akin to
    [`LABEL-INDEX`][cc80].

<a id="x-28MGL-CORE-3ALABEL-INDEX-DISTRIBUTIONS-20GENERIC-FUNCTION-29"></a>
- [generic-function] **LABEL-INDEX-DISTRIBUTIONS** *RESULT*

    Return a sequence of label index distributions for
    `RESULTS` produced by some model for a batch of instances. This is
    akin to [`LABEL-INDEX-DISTRIBUTION`][caec].

<a id="x-28MGL-CORE-3A-40MGL-CLASSIFICATION-MONITOR-20MGL-PAX-3ASECTION-29"></a>
### 7.1 Classification Monitors

The following functions return a list monitors. The monitors are
for events of signature (`INSTANCES` `MODEL`) such as those produced by
[`MONITOR-MODEL-RESULTS`][e50c] and its various model specific variations.
They are model-agnostic functions, extensible to new classifier
types. 

<a id="x-28MGL-CORE-3AMAKE-CLASSIFICATION-ACCURACY-MONITORS-20FUNCTION-29"></a>
- [function] **MAKE-CLASSIFICATION-ACCURACY-MONITORS** *MODEL &KEY OPERATION-MODE ATTRIBUTES (LABEL-INDEX-FN #'LABEL-INDEX)*

    Return a list of [`MONITOR`][7068] objects associated with
    [`CLASSIFICATION-ACCURACY-COUNTER`][430d]s. `LABEL-INDEX-FN` is a function
    like [`LABEL-INDEX`][cc80]. See that function for more.
    
    Implemented in terms of [`MAKE-CLASSIFICATION-ACCURACY-MONITORS*`][2aa3].

<a id="x-28MGL-CORE-3AMAKE-CROSS-ENTROPY-MONITORS-20FUNCTION-29"></a>
- [function] **MAKE-CROSS-ENTROPY-MONITORS** *MODEL &KEY OPERATION-MODE ATTRIBUTES (LABEL-INDEX-DISTRIBUTION-FN #'LABEL-INDEX-DISTRIBUTION)*

    Return a list of [`MONITOR`][7068] objects associated with
    [`CROSS-ENTROPY-COUNTER`][b186]s. `LABEL-INDEX-DISTRIBUTION-FN` is a
    function like [`LABEL-INDEX-DISTRIBUTION`][caec]. See that function for more.
    
    Implemented in terms of [`MAKE-CROSS-ENTROPY-MONITORS*`][e46f].

<a id="x-28MGL-CORE-3AMAKE-LABEL-MONITORS-20FUNCTION-29"></a>
- [function] **MAKE-LABEL-MONITORS** *MODEL &KEY OPERATION-MODE ATTRIBUTES (LABEL-INDEX-FN #'LABEL-INDEX) (LABEL-INDEX-DISTRIBUTION-FN #'LABEL-INDEX-DISTRIBUTION)*

    Return classification accuracy and cross-entropy monitors. See
    [`MAKE-CLASSIFICATION-ACCURACY-MONITORS`][911c] and
    [`MAKE-CROSS-ENTROPY-MONITORS`][6004] for a description of paramters.

The monitor makers above can be extended to support new classifier
types via the following generic functions.

<a id="x-28MGL-CORE-3AMAKE-CLASSIFICATION-ACCURACY-MONITORS-2A-20GENERIC-FUNCTION-29"></a>
- [generic-function] **MAKE-CLASSIFICATION-ACCURACY-MONITORS\*** *MODEL OPERATION-MODE LABEL-INDEX-FN ATTRIBUTES*

    Identical to [`MAKE-CLASSIFICATION-ACCURACY-MONITORS`][911c]
    bar the keywords arguments. Specialize this to add to support for
    new model types. The default implementation also allows for some
    extensibility: if [`LABEL-INDICES`][31ed] is defined on `MODEL`, then it will be
    used to extract label indices from model results.

<a id="x-28MGL-CORE-3AMAKE-CROSS-ENTROPY-MONITORS-2A-20GENERIC-FUNCTION-29"></a>
- [generic-function] **MAKE-CROSS-ENTROPY-MONITORS\*** *MODEL OPERATION-MODE LABEL-INDEX-DISTRIBUTION-FN ATTRIBUTES*

    Identical to [`MAKE-CROSS-ENTROPY-MONITORS`][6004] bar the
    keywords arguments. Specialize this to add to support for new model
    types. The default implementation also allows for some
    extensibility: if `LABEL-INDEX-DISTRIBUTIONS`([`0`][9385] [`1`][caec]) is defined on `MODEL`,
    then it will be used to extract label distributions from model
    results.

<a id="x-28MGL-CORE-3A-40MGL-CLASSIFICATION-MEASURER-20MGL-PAX-3ASECTION-29"></a>
### 7.2 Classification Measurers

The functions here compare some known good solution (also known as
*ground truth* or *target*) to a prediction or approximation and
return some measure of their \[dis\]similarity. They are model
independent, hence one has to extract the ground truths and
predictions first. Rarely used directly, they are mostly hidden
behind [Classification Monitors][c573].

<a id="x-28MGL-CORE-3AMEASURE-CLASSIFICATION-ACCURACY-20FUNCTION-29"></a>
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
    
    Note how the returned values are suitable for [`MULTIPLE-VALUE-CALL`][bf1a]
    with #'[`ADD-TO-COUNTER`][62de] and a [`CLASSIFICATION-ACCURACY-COUNTER`][430d].

<a id="x-28MGL-CORE-3AMEASURE-CROSS-ENTROPY-20FUNCTION-29"></a>
- [function] **MEASURE-CROSS-ENTROPY** *TRUTHS PREDICTIONS &KEY TRUTH-KEY PREDICTION-KEY (MIN-PREDICTION-PR 1.0d-15)*

    Return the sum of the cross-entropy between pairs of elements with
    the same index of `TRUTHS` and `PREDICTIONS`. `TRUTH-KEY` is a function
    that's when applied to an element of `TRUTHS` returns a sequence
    representing some kind of discrete target distribution (P in the
    definition below). `TRUTH-KEY` may be `NIL` which is equivalent to the
    [`IDENTITY`][8804] function. `PREDICTION-KEY` is the same kind of key for
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
    
    Note how the returned values are suitable for [`MULTIPLE-VALUE-CALL`][bf1a]
    with #'[`ADD-TO-COUNTER`][62de] and a [`CROSS-ENTROPY-COUNTER`][b186].

<a id="x-28MGL-CORE-3AMEASURE-ROC-AUC-20FUNCTION-29"></a>
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

<a id="x-28MGL-CORE-3AMEASURE-CONFUSION-20FUNCTION-29"></a>
- [function] **MEASURE-CONFUSION** *TRUTHS PREDICTIONS &KEY (TEST #'EQL) TRUTH-KEY PREDICTION-KEY WEIGHT*

    Create a [`CONFUSION-MATRIX`][60d2] from `TRUTHS` and `PREDICTIONS`.
    `TRUTHS` (keyed by `TRUTH-KEY`) is a sequence of class labels compared
    with `TEST` to another sequence of class labels in `PREDICTIONS` (keyed
    by `PREDICTION-KEY`). If `WEIGHT` is non-nil, then it is a function that
    returns the weight of an element of `TRUTHS`. Weighted cases add their
    weight to both counts (returned as the first and second values).
    
    Note how the returned confusion matrix can be added to another with
    [`ADD-TO-COUNTER`][62de].

<a id="x-28MGL-CORE-3A-40MGL-CLASSIFICATION-COUNTER-20MGL-PAX-3ASECTION-29"></a>
### 7.3 Classification Counters

<a id="x-28MGL-CORE-3ACLASSIFICATION-ACCURACY-COUNTER-20CLASS-29"></a>
- [class] **CLASSIFICATION-ACCURACY-COUNTER** *[BASIC-COUNTER][5979]*

    A [`BASIC-COUNTER`][5979] with "acc." as its `:TYPE`
    attribute and a [`PRINT-OBJECT`][eafc] method that prints percentages.

<a id="x-28MGL-CORE-3ACROSS-ENTROPY-COUNTER-20CLASS-29"></a>
- [class] **CROSS-ENTROPY-COUNTER** *[BASIC-COUNTER][5979]*

    A [`BASIC-COUNTER`][5979] with "xent" as its `:TYPE`
    attribute.

<a id="x-28MGL-CORE-3A-40MGL-CONFUSION-MATRIX-20MGL-PAX-3ASECTION-29"></a>
#### 7.3.1 Confusion Matrices

<a id="x-28MGL-CORE-3ACONFUSION-MATRIX-20CLASS-29"></a>
- [class] **CONFUSION-MATRIX**

    A confusion matrix keeps count of classification
    results. The correct class is called `target' and the output of the
    classifier is called`prediction'.

<a id="x-28MGL-CORE-3AMAKE-CONFUSION-MATRIX-20FUNCTION-29"></a>
- [function] **MAKE-CONFUSION-MATRIX** *&KEY (TEST #'EQL)*

    Classes are compared with `TEST`.

<a id="x-28MGL-CORE-3ASORT-CONFUSION-CLASSES-20GENERIC-FUNCTION-29"></a>
- [generic-function] **SORT-CONFUSION-CLASSES** *MATRIX CLASSES*

    Return a list of `CLASSES` sorted for presentation
    purposes.

<a id="x-28MGL-CORE-3ACONFUSION-CLASS-NAME-20GENERIC-FUNCTION-29"></a>
- [generic-function] **CONFUSION-CLASS-NAME** *MATRIX CLASS*

    Name of `CLASS` for presentation purposes.

<a id="x-28MGL-CORE-3ACONFUSION-COUNT-20GENERIC-FUNCTION-29"></a>
- [generic-function] **CONFUSION-COUNT** *MATRIX TARGET PREDICTION*

<a id="x-28MGL-CORE-3AMAP-CONFUSION-MATRIX-20GENERIC-FUNCTION-29"></a>
- [generic-function] **MAP-CONFUSION-MATRIX** *FN MATRIX*

    Call `FN` with `TARGET`, `PREDICTION`,
    [`COUNT`][5c66] paramaters for each cell in the confusion matrix. Cells with a
    zero count may be ommitted.

<a id="x-28MGL-CORE-3ACONFUSION-MATRIX-CLASSES-20GENERIC-FUNCTION-29"></a>
- [generic-function] **CONFUSION-MATRIX-CLASSES** *MATRIX*

    A list of all classes. The default is to collect
    classes from the counts. This can be overridden if, for instance,
    some classes are not present in the results.

<a id="x-28MGL-CORE-3ACONFUSION-MATRIX-ACCURACY-20FUNCTION-29"></a>
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

<a id="x-28MGL-CORE-3ACONFUSION-MATRIX-PRECISION-20FUNCTION-29"></a>
- [function] **CONFUSION-MATRIX-PRECISION** *MATRIX PREDICTION*

    Return the accuracy over the cases when the classifier said
    `PREDICTION`.

<a id="x-28MGL-CORE-3ACONFUSION-MATRIX-RECALL-20FUNCTION-29"></a>
- [function] **CONFUSION-MATRIX-RECALL** *MATRIX TARGET*

    Return the accuracy over the cases when the correct class is
    `TARGET`.

<a id="x-28MGL-CORE-3AADD-CONFUSION-MATRIX-20FUNCTION-29"></a>
- [function] **ADD-CONFUSION-MATRIX** *MATRIX RESULT-MATRIX*

    Add `MATRIX` into `RESULT-MATRIX`.

<a id="x-28MGL-CORE-3A-40MGL-FEATURES-20MGL-PAX-3ASECTION-29"></a>
## 8 Features

###### \[in package MGL-CORE\]
<a id="x-28MGL-CORE-3A-40MGL-FEATURE-SELECTION-20MGL-PAX-3ASECTION-29"></a>
### 8.1 Feature Selection

The following *scoring functions* all return an [`EQUAL`][96d0] hash table
that maps features to scores.

<a id="x-28MGL-CORE-3ACOUNT-FEATURES-20FUNCTION-29"></a>
- [function] **COUNT-FEATURES** *DOCUMENTS MAPPER &KEY (KEY #'IDENTITY)*

    Return scored features as an [`EQUAL`][96d0] hash table whose keys are
    features of `DOCUMENTS` and values are counts of occurrences of
    features. `MAPPER` takes a function and a document and calls function
    with features of the document.
    
    ```common-lisp
    (sort (alexandria:hash-table-alist
           (count-features '(("hello" "world")
                             ("this" "is" "our" "world"))
                           (lambda (fn document)
                             (map nil fn document))))
          #'string< :key #'car)
    => (("hello" . 1) ("is" . 1) ("our" . 1) ("this" . 1) ("world" . 2))
    ```


<a id="x-28MGL-CORE-3AFEATURE-LLRS-20FUNCTION-29"></a>
- [function] **FEATURE-LLRS** *DOCUMENTS MAPPER CLASS-FN &KEY (CLASSES (ALL-DOCUMENT-CLASSES DOCUMENTS CLASS-FN))*

    Return scored features as an [`EQUAL`][96d0] hash table whose keys are
    features of `DOCUMENTS` and values are their log likelihood ratios.
    `MAPPER` takes a function and a document and calls function with
    features of the document.
    
    ```common-lisp
    (sort (alexandria:hash-table-alist
           (feature-llrs '((:a "hello" "world")
                           (:b "this" "is" "our" "world"))
                         (lambda (fn document)
                           (map nil fn (rest document)))
                         #'first))
          #'string< :key #'car)
    => (("hello" . 2.6032386) ("is" . 2.6032386) ("our" . 2.6032386)
        ("this" . 2.6032386) ("world" . 4.8428774e-8))
    ```


<a id="x-28MGL-CORE-3AFEATURE-DISAMBIGUITIES-20FUNCTION-29"></a>
- [function] **FEATURE-DISAMBIGUITIES** *DOCUMENTS MAPPER CLASS-FN &KEY (CLASSES (ALL-DOCUMENT-CLASSES DOCUMENTS CLASS-FN))*

    Return scored features as an [`EQUAL`][96d0] hash table whose keys are
    features of `DOCUMENTS` and values are their *disambiguities*. `MAPPER`
    takes a function and a document and calls function with features of
    the document.
    
    From the paper 'Using Ambiguity Measure Feature Selection Algorithm
    for Support Vector Machine Classifier'.

<a id="x-28MGL-CORE-3A-40MGL-FEATURE-ENCODING-20MGL-PAX-3ASECTION-29"></a>
### 8.2 Feature Encoding

Features can rarely be fed directly to algorithms as is, they need
to be transformed in some way. Suppose we have a simple language
model that takes a single word as input and predicts the next word.
However, both input and output is to be encoded as float vectors of
length 1000. What we do is find the top 1000 words by some
measure (see [Feature Selection][1b5e]) and associate these words with
the integers in \[0..999\] (this is [`ENCODE`][fedd]ing). By using for
example [one-hot](http://en.wikipedia.org/wiki/One-hot) encoding, we
translate a word into a float vector when passing in the input. When
the model outputs the probability distribution of the next word, we
find the index of the max and find the word associated with it (this
is [`DECODE`][1339]ing)

<a id="x-28MGL-CORE-3AENCODE-20GENERIC-FUNCTION-29"></a>
- [generic-function] **ENCODE** *ENCODER DECODED*

    Encode `DECODED` with `ENCODER`. This interface is
    generic enough to be almost meaningless. See [`ENCODER/DECODER`][1beb] for a
    simple, [`MGL-NLP:BAG-OF-WORDS-ENCODER`][cbb4] for a slightly more involved
    example.
    
    If `ENCODER` is a function designator, then it's simply [`FUNCALL`][6b4a]ed
    with `DECODED`.

<a id="x-28MGL-CORE-3ADECODE-20GENERIC-FUNCTION-29"></a>
- [generic-function] **DECODE** *DECODER ENCODED*

    Decode `ENCODED` with `ENCODER`. For an `DECODER` /
    `ENCODER` pair, `(DECODE DECODER (ENCODE ENCODER OBJECT))` must be
    equal in some sense to `OBJECT`.
    
    If `DECODER` is a function designator, then it's simply [`FUNCALL`][6b4a]ed
    with `ENCODED`.

<a id="x-28MGL-CORE-3AENCODER-2FDECODER-20CLASS-29"></a>
- [class] **ENCODER/DECODER**

    Implements O(1) [`ENCODE`][fedd] and [`DECODE`][1339] by having an
    internal decoded-to-encoded and an encoded-to-decoded [`EQUAL`][96d0] hash
    table. `ENCODER/DECODER` objects can be saved and loaded (see
    [Persistence][29a1]) as long as the elements in the hash tables have
    read/write consitency.
    
    ```common-lisp
    (let ((indexer
            (make-indexer
             (alexandria:alist-hash-table '(("I" . 3) ("me" . 2) ("mine" . 1)))
             2)))
      (values (encode indexer "I")
              (encode indexer "me")
              (encode indexer "mine")
              (decode indexer 0)
              (decode indexer 1)
              (decode indexer 2)))
    => 0
    => 1
    => NIL
    => "I"
    => "me"
    => NIL
    ```


<a id="x-28MGL-CORE-3AMAKE-INDEXER-20FUNCTION-29"></a>
- [function] **MAKE-INDEXER** *SCORED-FEATURES N &KEY (START 0) (CLASS 'ENCODER/DECODER)*

    Take the top `N` features from `SCORED-FEATURES` (see
    [Feature Selection][1b5e]), assign indices to them starting from `START`.
    Return an [`ENCODER/DECODER`][1beb] (or another `CLASS`) that converts between
    objects and indices.

Also see [Bag of Words][0784].

<a id="x-28MGL-OPT-3A-40MGL-OPT-20MGL-PAX-3ASECTION-29"></a>
## 9 Gradient Based Optimization

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
[Gradient Descent][10e7] (with several variants) and [Conjugate Gradient][83e6] both of
which are first order methods (they do not need second order
gradients) but more can be added with the [Extension API][6a6f].

<a id="x-28MGL-OPT-3AMINIMIZE-20FUNCTION-29"></a>
- [function] **MINIMIZE** *OPTIMIZER GRADIENT-SOURCE &KEY (WEIGHTS (LIST-SEGMENTS GRADIENT-SOURCE)) (DATASET \*INFINITELY-EMPTY-DATASET\*)*

    Minimize the value of the real valued function represented by
    `GRADIENT-SOURCE` by updating some of its parameters in `WEIGHTS` (a `MAT`
    or a sequence of `MAT`s). Return `WEIGHTS`. `DATASET` (see
    [Datasets][109e]) is a set of unoptimized parameters of the same
    function. For example, `WEIGHTS` may be the weights of a neural
    network while `DATASET` is the training set consisting of inputs
    suitable for [`SET-INPUT`][0c9e]. The default
    `DATASET`, ([`*INFINITELY-EMPTY-DATASET*`][ad8f]) is suitable for when all
    parameters are optimized, so there is nothing left to come from the
    environment.
    
    Optimization terminates if `DATASET` is a sampler and it runs out or
    when some other condition met (see [`TERMINATION`][9006], for example). If
    `DATASET` is a [`SEQUENCE`][b9c1], then it is reused over and over again.
    
    Examples for various optimizers are provided in [Gradient Descent][10e7] and
    [Conjugate Gradient][83e6].

<a id="x-28MGL-OPT-3A-40MGL-OPT-ITERATIVE-OPTIMIZER-20MGL-PAX-3ASECTION-29"></a>
### 9.1 Iterative Optimizer

<a id="x-28MGL-OPT-3AITERATIVE-OPTIMIZER-20CLASS-29"></a>
- [class] **ITERATIVE-OPTIMIZER**

    An abstract base class of [Gradient Descent][10e7] and
    [Conjugate Gradient][83e6] based optimizers that iterate over instances until a
    termination condition is met.

<a id="x-28MGL-OPT-3AN-INSTANCES-20-28MGL-PAX-3AREADER-20MGL-OPT-3AITERATIVE-OPTIMIZER-29-29"></a>
- [reader] **N-INSTANCES** *ITERATIVE-OPTIMIZER (:N-INSTANCES = 0)*

    The number of instances this optimizer has seen so
    far. Incremented automatically during optimization.

<a id="x-28MGL-OPT-3ATERMINATION-20-28MGL-PAX-3AACCESSOR-20MGL-OPT-3AITERATIVE-OPTIMIZER-29-29"></a>
- [accessor] **TERMINATION** *ITERATIVE-OPTIMIZER (:TERMINATION = NIL)*

    If a number, it's the number of instances to train
    on in the sense of [`N-INSTANCES`][4c73]. If `N-INSTANCES` is equal or greater
    than this value optimization stops. If `TERMINATION` is `NIL`, then
    optimization will continue. If it is `T`, then optimization will
    stop. If it is a function of no arguments, then its return value
    is processed as if it was returned by `TERMINATION`.

<a id="x-28MGL-OPT-3AON-OPTIMIZATION-STARTED-20-28MGL-PAX-3AACCESSOR-20MGL-OPT-3AITERATIVE-OPTIMIZER-29-29"></a>
- [accessor] **ON-OPTIMIZATION-STARTED** *ITERATIVE-OPTIMIZER (:ON-OPTIMIZATION-STARTED = NIL)*

    An event hook with parameters `(OPTIMIZER
    GRADIENT-SOURCE N-INSTANCES)`. Called after initializations are
    performed (INITIALIZE-OPTIMIZER*, INITIALIZE-GRADIENT-SOURCE*) but
    before optimization is started.

<a id="x-28MGL-OPT-3AON-OPTIMIZATION-FINISHED-20-28MGL-PAX-3AACCESSOR-20MGL-OPT-3AITERATIVE-OPTIMIZER-29-29"></a>
- [accessor] **ON-OPTIMIZATION-FINISHED** *ITERATIVE-OPTIMIZER (:ON-OPTIMIZATION-FINISHED = NIL)*

    An event hook with parameters `(OPTIMIZER
    GRADIENT-SOURCE N-INSTANCES)`. Called when optimization has
    finished.

<a id="x-28MGL-OPT-3AON-N-INSTANCES-CHANGED-20-28MGL-PAX-3AACCESSOR-20MGL-OPT-3AITERATIVE-OPTIMIZER-29-29"></a>
- [accessor] **ON-N-INSTANCES-CHANGED** *ITERATIVE-OPTIMIZER (:ON-N-INSTANCES-CHANGED = NIL)*

    An event hook with parameters `(OPTIMIZER
    GRADIENT-SOURCE N-INSTANCES)`. Called when optimization of a batch
    of instances is done and [`N-INSTANCES`][4c73] is incremented.

Now let's discuss a few handy utilities.

<a id="x-28MGL-OPT-3AMONITOR-OPTIMIZATION-PERIODICALLY-20FUNCTION-29"></a>
- [function] **MONITOR-OPTIMIZATION-PERIODICALLY** *OPTIMIZER PERIODIC-FNS*

    For each periodic function in the list of `PERIODIC-FNS`, add a
    monitor to `OPTIMIZER`'s [`ON-OPTIMIZATION-STARTED`][ebd4],
    [`ON-OPTIMIZATION-FINISHED`][0072] and [`ON-N-INSTANCES-CHANGED`][4f0b] hooks. The
    monitors are simple functions that just call each periodic function
    with the event parameters (`OPTIMIZER` `GRADIENT-SOURCE` [`N-INSTANCES`][4c73]).
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
    prevents [`RESET-OPTIMIZATION-MONITORS`][ca09] from being called at the start
    of the optimization when the monitors are empty anyway.

<a id="x-28MGL-OPT-3ARESET-OPTIMIZATION-MONITORS-20GENERIC-FUNCTION-29"></a>
- [generic-function] **RESET-OPTIMIZATION-MONITORS** *OPTIMIZER GRADIENT-SOURCE*

    Report the state of [`MONITORS`][8f37] of
    `OPTIMIZER` and `GRADIENT-SOURCE` and reset their counters. See
    [`MONITOR-OPTIMIZATION-PERIODICALLY`][4528] for an example of how this is
    used.

<a id="x-28MGL-OPT-3ARESET-OPTIMIZATION-MONITORS-20-28METHOD-20NIL-20-28MGL-OPT-3AITERATIVE-OPTIMIZER-20T-29-29-29"></a>
- [method] **RESET-OPTIMIZATION-MONITORS** *(OPTIMIZER ITERATIVE-OPTIMIZER) GRADIENT-SOURCE*

    Log the counters of the monitors of `OPTIMIZER` and `GRADIENT-SOURCE`
    and reset them.

<a id="x-28MGL-OPT-3AREPORT-OPTIMIZATION-PARAMETERS-20GENERIC-FUNCTION-29"></a>
- [generic-function] **REPORT-OPTIMIZATION-PARAMETERS** *OPTIMIZER GRADIENT-SOURCE*

    A utility that's often called at the start of
    optimization (from [`ON-OPTIMIZATION-STARTED`][ebd4]). The default
    implementation logs the description of `GRADIENT-SOURCE` (as in
    [`DESCRIBE`][38a0]) and `OPTIMIZER` and calls `LOG-MAT-ROOM`.

<a id="x-28MGL-OPT-3A-40MGL-OPT-COST-20MGL-PAX-3ASECTION-29"></a>
### 9.2 Cost Function

The function being minimized is often called the *cost* or the
*loss* function.

<a id="x-28MGL-COMMON-3ACOST-20GENERIC-FUNCTION-29"></a>
- [generic-function] **COST** *MODEL*

    Return the value of the cost function being
    minimized. Calling this only makes sense in the context of an
    ongoing optimization (see [`MINIMIZE`][46a4]). The cost is that of a batch of
    instances.

<a id="x-28MGL-OPT-3AMAKE-COST-MONITORS-20FUNCTION-29"></a>
- [function] **MAKE-COST-MONITORS** *MODEL &KEY OPERATION-MODE ATTRIBUTES*

    Return a list of [`MONITOR`][7068] objects, each associated with one
    [`BASIC-COUNTER`][5979] with attribute `:TYPE` "cost". Implemented in terms of
    [`MAKE-COST-MONITORS*`][3815].

<a id="x-28MGL-OPT-3AMAKE-COST-MONITORS-2A-20GENERIC-FUNCTION-29"></a>
- [generic-function] **MAKE-COST-MONITORS\*** *MODEL OPERATION-MODE ATTRIBUTES*

    Identical to [`MAKE-COST-MONITORS`][46c2] bar the keywords
    arguments. Specialize this to add to support for new model types.

<a id="x-28MGL-GD-3A-40MGL-GD-20MGL-PAX-3ASECTION-29"></a>
### 9.3 Gradient Descent

###### \[in package MGL-GD\]
Gradient descent is a first-order optimization algorithm. Relying
completely on first derivatives, it does not even evaluate the
function to be minimized. Let's see how to minimize a numerical lisp
function with respect to some of its parameters.

<a id="x-28MGL-GD-3ASGD-2ELISP-20-28MGL-PAX-3AINCLUDE-20-23P-22-2Fhome-2Fmelisgl-2Fown-2Fmgl-2Fexample-2Fsgd-2Elisp-22-20-3AHEADER-NL-20-22-60-60-60commonlisp-22-20-3AFOOTER-NL-20-22-60-60-60-22-29-29"></a>
```commonlisp
(cl:defpackage :mgl-example-sgd
  (:use #:common-lisp #:mgl))

(in-package :mgl-example-sgd)

;;; Create an object representing the sine function.
(defparameter *diff-fn-1*
  (make-instance 'mgl-diffun:diffun
                 :fn #'sin
                 ;; We are going to optimize its only parameter.
                 :weight-indices '(0)))

;;; Minimize SIN. Note that there is no dataset involved because all
;;; parameters are being optimized.
(minimize (make-instance 'sgd-optimizer :termination 1000)
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
(minimize (make-instance 'sgd-optimizer :batch-size 10)
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
(minimize (make-instance 'sgd-optimizer :termination 1000)
          *diff-fn-2*
          :weights (make-mat 1)
          :dataset '((0) (1) (2) (3) (4) (5)))
;;; => A MAT with a single value of about 2.5.
```

We are going to see a number of accessors for optimizer paramaters.
In general, it's allowed to [`SETF`][17b7] real slot accessors (as opposed to
readers and writers) at any time during optimization and so is
defining a method on an optimizer subclass that computes the value
in any way. For example, to decay the learning rate on a per
mini-batch basis:

```commonlisp
(defmethod learning-rate ((optimizer my-sgd-optimizer))
  (* (slot-value optimizer 'learning-rate)
     (expt 0.998
           (/ (n-instances optimizer) 60000))))
```


<a id="x-28MGL-GD-3A-40MGL-GD-BATCH-GD-OPTIMIZER-20MGL-PAX-3ASECTION-29"></a>
#### 9.3.1 Batch Based Optimizers

First let's see everything common to all batch based optimizers,
then discuss [SGD Optimizer][25fd], [Adam Optimizer][bd13] and
[Normalized Batch Optimizer][0c91]. All batch based optimizers
are [`ITERATIVE-OPTIMIZER`][8da0]s, so see [Iterative Optimizer][779d]
too.

<a id="x-28MGL-GD-3ABATCH-GD-OPTIMIZER-20CLASS-29"></a>
- [class] **BATCH-GD-OPTIMIZER**

    Another abstract base class for gradient based
    optimizers tath updates all weights simultaneously after chewing
    through `BATCH-SIZE`([`0`][8916] [`1`][ba18] [`2`][c918]) inputs. See subclasses [`SGD-OPTIMIZER`][2a2f],
    [`ADAM-OPTIMIZER`][e0e6] and [`NORMALIZED-BATCH-GD-OPTIMIZER`][f6ae].
    
    [`PER-WEIGHT-BATCH-GD-OPTIMIZER`][5a43] may be a better choice when some
    weights can go unused for instance due to missing input values.

<a id="x-28MGL-COMMON-3ABATCH-SIZE-20-28MGL-PAX-3AACCESSOR-20MGL-GD-3A-3AGD-OPTIMIZER-29-29"></a>
- [accessor] **BATCH-SIZE** *GD-OPTIMIZER (:BATCH-SIZE = 1)*

    After having gone through `BATCH-SIZE` number of
    inputs, weights are updated. With `BATCH-SIZE` 1, one gets
    Stochastics Gradient Descent. With `BATCH-SIZE` equal to the number
    of instances in the dataset, one gets standard, 'batch' gradient
    descent. With `BATCH-SIZE` between these two extremes, one gets the
    most practical 'mini-batch' compromise.

<a id="x-28MGL-GD-3ALEARNING-RATE-20-28MGL-PAX-3AACCESSOR-20MGL-GD-3A-3AGD-OPTIMIZER-29-29"></a>
- [accessor] **LEARNING-RATE** *GD-OPTIMIZER (:LEARNING-RATE = 0.1)*

    This is the step size along the gradient. Decrease
    it if optimization diverges, increase it if it doesn't make
    progress.

<a id="x-28MGL-GD-3AMOMENTUM-20-28MGL-PAX-3AACCESSOR-20MGL-GD-3A-3AGD-OPTIMIZER-29-29"></a>
- [accessor] **MOMENTUM** *GD-OPTIMIZER (:MOMENTUM = 0)*

    A value in the \[0, 1) interval. `MOMENTUM` times the
    previous weight change is added to the gradient. 0 means no
    momentum.

<a id="x-28MGL-GD-3AMOMENTUM-TYPE-20-28MGL-PAX-3AREADER-20MGL-GD-3A-3AGD-OPTIMIZER-29-29"></a>
- [reader] **MOMENTUM-TYPE** *GD-OPTIMIZER (:MOMENTUM-TYPE = :NORMAL)*

    One of `:NORMAL`, `:NESTEROV` or `:NONE`. For pure
    optimization Nesterov's momentum may be better, but it may also
    increases chances of overfitting. Using `:NONE` is equivalent to 0
    momentum, but it also uses less memory. Note that with `:NONE`,
    [`MOMENTUM`][af05] is ignored even it it is non-zero.

<a id="x-28MGL-GD-3AWEIGHT-DECAY-20-28MGL-PAX-3AACCESSOR-20MGL-GD-3A-3AGD-OPTIMIZER-29-29"></a>
- [accessor] **WEIGHT-DECAY** *GD-OPTIMIZER (:WEIGHT-DECAY = 0)*

    An L2 penalty. It discourages large weights, much
    like a zero mean gaussian prior. `WEIGHT-DECAY` \* WEIGHT is added to
    the gradient to penalize large weights. It's as if the function
    whose minimum is sought had WEIGHT-DECAY\*sum\_i{0.5 \* WEIGHT\_i^2}
    added to it.

<a id="x-28MGL-GD-3AWEIGHT-PENALTY-20-28MGL-PAX-3AACCESSOR-20MGL-GD-3A-3AGD-OPTIMIZER-29-29"></a>
- [accessor] **WEIGHT-PENALTY** *GD-OPTIMIZER (:WEIGHT-PENALTY = 0)*

    An L1 penalty. It encourages sparsity.
    `SIGN`(WEIGHT) \* `WEIGHT-PENALTY` is added to the gradient pushing the
    weight towards negative infinity. It's as if the function whose
    minima is sought had WEIGHT-PENALTY\*sum\_i{abs(WEIGHT\_i)} added to
    it. Putting it on feature biases consitutes a sparsity constraint
    on the features.

<a id="x-28MGL-GD-3AUSE-SEGMENT-DERIVATIVES-P-20-28MGL-PAX-3AREADER-20MGL-GD-3A-3AGD-OPTIMIZER-29-29"></a>
- [reader] **USE-SEGMENT-DERIVATIVES-P** *GD-OPTIMIZER (:USE-SEGMENT-DERIVATIVES-P = NIL)*

    Save memory if both the gradient source (the model
    being optimized) and the optimizer support this feature. It works
    like this: the accumulator into which the gradient source is asked
    to place the derivatives of a segment will be [`SEGMENT-DERIVATIVES`][9a5b]
    of the segment. This allows the optimizer not to allocate an
    accumulator matrix into which the derivatives are summed.

<a id="x-28MGL-GD-3AAFTER-UPDATE-HOOK-20-28MGL-PAX-3AACCESSOR-20MGL-GD-3A-3AGD-OPTIMIZER-29-29"></a>
- [accessor] **AFTER-UPDATE-HOOK** *GD-OPTIMIZER (:AFTER-UPDATE-HOOK = NIL)*

    A list of functions with no arguments called after
    each weight update.

<a id="x-28MGL-GD-3ABEFORE-UPDATE-HOOK-20-28MGL-PAX-3AACCESSOR-20MGL-GD-3ABATCH-GD-OPTIMIZER-29-29"></a>
- [accessor] **BEFORE-UPDATE-HOOK** *BATCH-GD-OPTIMIZER (:BEFORE-UPDATE-HOOK = NIL)*

    A list of functions of no parameters. Each
    function is called just before a weight update takes place (after
    accumulated gradients have been divided the length of the batch).
    Convenient to hang some additional gradient accumulating code
    on.

<a id="x-28MGL-GD-3A-40MGL-GD-SGD-OPTIMIZER-20MGL-PAX-3ASECTION-29"></a>
##### SGD Optimizer

<a id="x-28MGL-GD-3ASGD-OPTIMIZER-20CLASS-29"></a>
- [class] **SGD-OPTIMIZER** *[BATCH-GD-OPTIMIZER][d94e]*

    With `BATCH-SIZE`([`0`][8916] [`1`][ba18] [`2`][c918]) 1 this is Stochastic Gradient
    Descent. With higher batch sizes, one gets mini-batch and Batch
    Gradient Descent.
    
    Assuming that `ACCUMULATOR` has the sum of gradients for a mini-batch,
    the weight update looks like this:
    
    $$
    \Delta\_w^{t+1} = momentum \* \Delta\_w^t
      + \frac{accumulator}{batchsize}
      + l\_2 w + l\_1 sign(w)
    $$
    
    $$
    w^{t+1} = w^{t} - learningrate \* \Delta\_w,
    $$
    
    which is the same as the more traditional formulation:
    
    $$
    \Delta\_w^{t+1} = momentum \* \Delta\_w^{t}
      + learningrate \* \left(\frac{\frac{df}{dw}}{batchsize}
                           + l\_2 w + l\_1 sign(w)\right)
    $$
    
    $$
    w^{t+1} = w^{t} - \Delta\_w,
    $$
    
    but the former works better when batch size, momentum or learning
    rate change during the course of optimization. The above is with
    normal momentum, Nesterov's momentum (see [`MOMENTUM-TYPE`][5611]) momentum is
    also available.
    
    See [Batch Based Optimizers][2c39] for the description of the various
    options common to all batch based optimizers.

<a id="x-28MGL-GD-3A-40MGL-GD-ADAM-OPTIMIZER-20MGL-PAX-3ASECTION-29"></a>
##### Adam Optimizer

<a id="x-28MGL-GD-3AADAM-OPTIMIZER-20CLASS-29"></a>
- [class] **ADAM-OPTIMIZER** *[BATCH-GD-OPTIMIZER][d94e]*

    Adam is a first-order stochasistic gradient descent
    optimizer. It maintains an internal estimation for the mean and raw
    variance of each derivative as exponential moving averages. The step
    it takes is basically `M/(sqrt(V)+E)` where `M` is the estimated
    mean, `V` is the estimated variance, and `E` is a small adjustment
    factor to prevent the gradient from blowing up. See version 5 of the
    [paper](http://arxiv.org/abs/1412.6980) for more.
    
    Note that using momentum is not supported with Adam. In fact, an
    error is signalled if it's not `:NONE`.
    
    See [Batch Based Optimizers][2c39] for the description of the various
    options common to all batch based optimizers.

<a id="x-28MGL-GD-3ALEARNING-RATE-20-28MGL-PAX-3AACCESSOR-20MGL-GD-3AADAM-OPTIMIZER-29-29"></a>
- [accessor] **LEARNING-RATE** *ADAM-OPTIMIZER (= 2.0e-4)*

    Same thing as [`LEARNING-RATE`][09ed] but with the default suggested by the Adam paper.

<a id="x-28MGL-GD-3AMEAN-DECAY-20-28MGL-PAX-3AACCESSOR-20MGL-GD-3AADAM-OPTIMIZER-29-29"></a>
- [accessor] **MEAN-DECAY** *ADAM-OPTIMIZER (:MEAN-DECAY = 0.9)*

    A number between 0 and 1 that determines how fast
    the estimated mean of derivatives is updated. 0 basically gives
    you `RMSPROP` (if [`VARIANCE-DECAY`][0900] is not too large) or AdaGrad (if
    `VARIANCE-DECAY` is close to 1 and the learning rate is annealed.
    This is $\beta\_1$ in the paper.

<a id="x-28MGL-GD-3AMEAN-DECAY-DECAY-20-28MGL-PAX-3AACCESSOR-20MGL-GD-3AADAM-OPTIMIZER-29-29"></a>
- [accessor] **MEAN-DECAY-DECAY** *ADAM-OPTIMIZER (:MEAN-DECAY-DECAY = (- 1 1.0d-7))*

    A value that should be close to 1. [`MEAN-DECAY`][011d] is
    multiplied by this value after each update. This is $\lambda$ in
    the paper.

<a id="x-28MGL-GD-3AVARIANCE-DECAY-20-28MGL-PAX-3AACCESSOR-20MGL-GD-3AADAM-OPTIMIZER-29-29"></a>
- [accessor] **VARIANCE-DECAY** *ADAM-OPTIMIZER (:VARIANCE-DECAY = 0.999)*

    A number between 0 and 1 that determines how fast
    the estimated variance of derivatives is updated. This is
    $\beta\_2$ in the paper.

<a id="x-28MGL-GD-3AVARIANCE-ADJUSTMENT-20-28MGL-PAX-3AACCESSOR-20MGL-GD-3AADAM-OPTIMIZER-29-29"></a>
- [accessor] **VARIANCE-ADJUSTMENT** *ADAM-OPTIMIZER (:VARIANCE-ADJUSTMENT = 1.0d-7)*

    Within the bowels of adam, the estimated mean is
    divided by the square root of the estimated variance (per weight)
    which can lead to numerical problems if the denominator is near
    zero. To avoid this, `VARIANCE-ADJUSTMENT`, which should be a small
    positive number, is added to the denominator. This is `epsilon` in
    the paper.

<a id="x-28MGL-GD-3A-40MGL-GD-NORMALIZED-BATCH-GD-OPTIMIZER-20MGL-PAX-3ASECTION-29"></a>
##### Normalized Batch Optimizer

<a id="x-28MGL-GD-3ANORMALIZED-BATCH-GD-OPTIMIZER-20CLASS-29"></a>
- [class] **NORMALIZED-BATCH-GD-OPTIMIZER** *[BATCH-GD-OPTIMIZER][d94e]*

    Like [`BATCH-GD-OPTIMIZER`][d94e] but keeps count of how many
    times each weight was used in the batch and divides the accumulated
    gradient by this count instead of dividing by `N-INSTANCES-IN-BATCH`.
    This only makes a difference if there are missing values in the
    learner that's being trained. The main feature that distuinguishes
    this class from [`PER-WEIGHT-BATCH-GD-OPTIMIZER`][5a43] is that batches end at
    same time for all weights.

<a id="x-28MGL-GD-3AN-WEIGHT-USES-IN-BATCH-20-28MGL-PAX-3AACCESSOR-20MGL-GD-3ANORMALIZED-BATCH-GD-OPTIMIZER-29-29"></a>
- [accessor] **N-WEIGHT-USES-IN-BATCH** *NORMALIZED-BATCH-GD-OPTIMIZER*

    Number of uses of the weight in its current batch.

<a id="x-28MGL-GD-3A-40MGL-GD-SEGMENTED-GD-OPTIMIZER-20MGL-PAX-3ASECTION-29"></a>
#### 9.3.2 Segmented GD Optimizer

<a id="x-28MGL-GD-3ASEGMENTED-GD-OPTIMIZER-20CLASS-29"></a>
- [class] **SEGMENTED-GD-OPTIMIZER**

    An optimizer that delegates training of segments to
    other optimizers. Useful to delegate training of different segments
    to different optimizers (capable of working with segmentables) or
    simply to not train all segments.

<a id="x-28MGL-GD-3ASEGMENTER-20-28MGL-PAX-3AREADER-20MGL-GD-3ASEGMENTED-GD-OPTIMIZER-29-29"></a>
- [reader] **SEGMENTER** *SEGMENTED-GD-OPTIMIZER (:SEGMENTER)*

    When this optimizer is initialized it loops over
    the segment of the learner with [`MAP-SEGMENTS`][2312]. `SEGMENTER` is a
    function that is called with each segment and returns an optimizer
    or `NIL`. Several segments may be mapped to the same optimizer.
    After the segment->optimizer mappings are collected, each
    optimizer is initialized by INITIALIZE-OPTIMIZER with the list of
    segments mapped to it.

<a id="x-28MGL-OPT-3ASEGMENTS-20-28MGL-PAX-3AREADER-20MGL-GD-3ASEGMENTED-GD-OPTIMIZER-29-29"></a>
- [reader] **SEGMENTS** *SEGMENTED-GD-OPTIMIZER*

[`SEGMENTED-GD-OPTIMIZER`][3ce0] inherits from [`ITERATIVE-OPTIMIZER`][8da0], so see
[Iterative Optimizer][779d] too.

<a id="x-28MGL-GD-3A-40MGL-GD-PER-WEIGHT-OPTIMIZATION-20MGL-PAX-3ASECTION-29"></a>
#### 9.3.3 Per-weight Optimization

<a id="x-28MGL-GD-3APER-WEIGHT-BATCH-GD-OPTIMIZER-20CLASS-29"></a>
- [class] **PER-WEIGHT-BATCH-GD-OPTIMIZER**

    This is much like [Batch Based Optimizers][2c39] but it
    is more clever about when to update weights. Basically every weight
    has its own batch independent from the batches of others. This has
    desirable properties. One can for example put two neural networks
    together without adding any connections between them and the
    learning will produce results equivalent to the separated case.
    Also, adding inputs with only missing values does not change
    anything.
    
    Due to its very non-batch nature, there is no CUDA implementation of
    this optimizer.

<a id="x-28MGL-GD-3AN-WEIGHT-USES-IN-BATCH-20-28MGL-PAX-3AACCESSOR-20MGL-GD-3APER-WEIGHT-BATCH-GD-OPTIMIZER-29-29"></a>
- [accessor] **N-WEIGHT-USES-IN-BATCH** *PER-WEIGHT-BATCH-GD-OPTIMIZER*

    Number of uses of the weight in its current batch.

<a id="x-28MGL-GD-3A-40MGL-GD-UTILITIES-20MGL-PAX-3ASECTION-29"></a>
#### 9.3.4 Utilities

<a id="x-28MGL-GD-3ACLIP-L2-NORM-20FUNCTION-29"></a>
- [function] **CLIP-L2-NORM** *MATS L2-UPPER-BOUND &KEY CALLBACK*

    Scale `MATS` so that their $L\_2$ norm does not exceed `L2-UPPER-BOUND`.
    
    Compute the norm of of `MATS` as if they were a single vector. If the
    norm is greater than `L2-UPPER-BOUND`, then scale each matrix
    destructively by the norm divided by `L2-UPPER-BOUND` and if non-NIL
    call the function `CALLBACK` with the scaling factor.

<a id="x-28MGL-GD-3AARRANGE-FOR-CLIPPING-GRADIENTS-20FUNCTION-29"></a>
- [function] **ARRANGE-FOR-CLIPPING-GRADIENTS** *BATCH-GD-OPTIMIZER L2-UPPER-BOUND &KEY CALLBACK*

    Make it so that the norm of the batch normalized gradients
    accumulated by `BATCH-GD-OPTIMIZER` is clipped to `L2-UPPER-BOUND`
    before every update. See [`CLIP-L2-NORM`][af6b].

<a id="x-28MGL-CG-3A-40MGL-CG-20MGL-PAX-3ASECTION-29"></a>
### 9.4 Conjugate Gradient

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


<a id="x-28MGL-CG-3ACG-20FUNCTION-29"></a>
- [function] **CG** *FN W &KEY (MAX-N-LINE-SEARCHES \*DEFAULT-MAX-N-LINE-SEARCHES\*) (MAX-N-EVALUATIONS-PER-LINE-SEARCH \*DEFAULT-MAX-N-EVALUATIONS-PER-LINE-SEARCH\*) (MAX-N-EVALUATIONS \*DEFAULT-MAX-N-EVALUATIONS\*) (SIG \*DEFAULT-SIG\*) (RHO \*DEFAULT-RHO\*) (INT \*DEFAULT-INT\*) (EXT \*DEFAULT-EXT\*) (RATIO \*DEFAULT-RATIO\*) SPARE-VECTORS*

    [`CG-OPTIMIZER`][ee97] passes each batch of data to this function with its
    [`CG-ARGS`][9749] passed on.
    
    Minimize a differentiable multivariate function with conjugate
    gradient. The Polak-Ribiere flavour of conjugate gradients is used
    to compute search directions, and a line search using quadratic and
    cubic polynomial approximations and the Wolfe-Powell stopping
    criteria is used together with the slope ratio method for guessing
    initial step sizes. Additionally a bunch of checks are made to make
    sure that exploration is taking place and that extrapolation will
    not be unboundedly large.
    
    `FN` is a function of two parameters: `WEIGHTS`([`0`][3c96] [`1`][a896]) and `DERIVATIVES`. `WEIGHTS`
    is a `MAT` of the same size as `W` that is where the search start from.
    `DERIVATIVES` is also a `MAT` of that size and it is where `FN` shall
    place the partial derivatives. `FN` returns the value of the function
    that is being minimized.
    
    `CG` performs a number of line searches and invokes `FN` at each step. A
    line search invokes `FN` at most `MAX-N-EVALUATIONS-PER-LINE-SEARCH`
    number of times and can succeed in improving the minimum by the
    sufficient margin or it can fail. Note, the even a failed line
    search may improve further and hence change the weights it's just
    that the improvement was deemed too small. `CG` stops when either:
    
    - two line searches fail in a row
    
    - `MAX-N-LINE-SEARCHES` is reached
    
    - `MAX-N-EVALUATIONS` is reached
    
    `CG` returns a `MAT` that contains the best weights, the minimum, the
    number of line searches performed, the number of succesful line
    searches and the number of evaluations.
    
    When using `MAX-N-EVALUATIONS` remember that there is an extra
    evaluation of `FN` before the first line search.
    
    `SPARE-VECTORS` is a list of preallocated `MAT`s of the same size as `W`.
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

<a id="x-28MGL-CG-3A-2ADEFAULT-INT-2A-20VARIABLE-29"></a>
- [variable] **\*DEFAULT-INT\*** *0.1*

    Don't reevaluate within `INT` of the limit of the current bracket.

<a id="x-28MGL-CG-3A-2ADEFAULT-EXT-2A-20VARIABLE-29"></a>
- [variable] **\*DEFAULT-EXT\*** *3*

    Extrapolate maximum `EXT` times the current step-size.

<a id="x-28MGL-CG-3A-2ADEFAULT-SIG-2A-20VARIABLE-29"></a>
- [variable] **\*DEFAULT-SIG\*** *0.1*

    `SIG` and `RHO` are the constants controlling the Wolfe-Powell
    conditions. `SIG` is the maximum allowed absolute ratio between
    previous and new slopes (derivatives in the search direction), thus
    setting `SIG` to low (positive) values forces higher precision in the
    line-searches.

<a id="x-28MGL-CG-3A-2ADEFAULT-RHO-2A-20VARIABLE-29"></a>
- [variable] **\*DEFAULT-RHO\*** *0.05*

    `RHO` is the minimum allowed fraction of the expected (from the slope
    at the initial point in the linesearch). Constants must satisfy 0 <
    `RHO` < `SIG` < 1.

<a id="x-28MGL-CG-3A-2ADEFAULT-RATIO-2A-20VARIABLE-29"></a>
- [variable] **\*DEFAULT-RATIO\*** *10*

    Maximum allowed slope ratio.

<a id="x-28MGL-CG-3A-2ADEFAULT-MAX-N-LINE-SEARCHES-2A-20VARIABLE-29"></a>
- [variable] **\*DEFAULT-MAX-N-LINE-SEARCHES\*** *NIL*

<a id="x-28MGL-CG-3A-2ADEFAULT-MAX-N-EVALUATIONS-PER-LINE-SEARCH-2A-20VARIABLE-29"></a>
- [variable] **\*DEFAULT-MAX-N-EVALUATIONS-PER-LINE-SEARCH\*** *20*

<a id="x-28MGL-CG-3A-2ADEFAULT-MAX-N-EVALUATIONS-2A-20VARIABLE-29"></a>
- [variable] **\*DEFAULT-MAX-N-EVALUATIONS\*** *NIL*

<a id="x-28MGL-CG-3ACG-OPTIMIZER-20CLASS-29"></a>
- [class] **CG-OPTIMIZER** *[ITERATIVE-OPTIMIZER][8da0]*

    Updates all weights simultaneously after chewing
    through `BATCH-SIZE`([`0`][8916] [`1`][ba18] [`2`][c918]) inputs.

<a id="x-28MGL-COMMON-3ABATCH-SIZE-20-28MGL-PAX-3AACCESSOR-20MGL-CG-3ACG-OPTIMIZER-29-29"></a>
- [accessor] **BATCH-SIZE** *CG-OPTIMIZER (:BATCH-SIZE)*

    After having gone through `BATCH-SIZE` number of
    instances, weights are updated. Normally, [`CG`][4ffb] operates on all
    available data, but it may be useful to introduce some noise into
    the optimization to reduce overfitting by using smaller batch
    sizes. If `BATCH-SIZE` is not set, it is initialized to the size of
    the dataset at the start of optimization.

<a id="x-28MGL-CG-3ACG-ARGS-20-28MGL-PAX-3AACCESSOR-20MGL-CG-3ACG-OPTIMIZER-29-29"></a>
- [accessor] **CG-ARGS** *CG-OPTIMIZER (:CG-ARGS = 'NIL)*

<a id="x-28MGL-CG-3AON-CG-BATCH-DONE-20-28MGL-PAX-3AACCESSOR-20MGL-CG-3ACG-OPTIMIZER-29-29"></a>
- [accessor] **ON-CG-BATCH-DONE** *CG-OPTIMIZER (:ON-CG-BATCH-DONE = NIL)*

    An event hook called when processing a conjugate
    gradient batch is done. The handlers on the hook are called with 8
    arguments:
    
        (optimizer gradient-source instances
         best-w best-f n-line-searches
         n-succesful-line-searches n-evaluations)
    
    The latter 5 of which are the return values of the [`CG`][4ffb] function.

<a id="x-28MGL-CG-3ALOG-CG-BATCH-DONE-20GENERIC-FUNCTION-29"></a>
- [generic-function] **LOG-CG-BATCH-DONE** *OPTIMIZER GRADIENT-SOURCE INSTANCES BEST-W BEST-F N-LINE-SEARCHES N-SUCCESFUL-LINE-SEARCHES N-EVALUATIONS*

    This is a function can be added to
    [`ON-CG-BATCH-DONE`][d10a]. The default implementation simply logs the event
    arguments.

<a id="x-28MGL-CG-3ASEGMENT-FILTER-20-28MGL-PAX-3AREADER-20MGL-CG-3ACG-OPTIMIZER-29-29"></a>
- [reader] **SEGMENT-FILTER** *CG-OPTIMIZER (:SEGMENT-FILTER = (CONSTANTLY T))*

    A predicate function on segments that filters out
    uninteresting segments. Called from [`INITIALIZE-OPTIMIZER*`][7c2f].

<a id="x-28MGL-OPT-3A-40MGL-OPT-EXTENSION-API-20MGL-PAX-3ASECTION-29"></a>
### 9.5 Extension API

<a id="x-28MGL-OPT-3A-40MGL-OPT-OPTIMIZER-20MGL-PAX-3ASECTION-29"></a>
#### 9.5.1 Implementing Optimizers

The following generic functions must be specialized for new
optimizer types.

<a id="x-28MGL-OPT-3AMINIMIZE-2A-20GENERIC-FUNCTION-29"></a>
- [generic-function] **MINIMIZE\*** *OPTIMIZER GRADIENT-SOURCE WEIGHTS DATASET*

    Called by [`MINIMIZE`][46a4] after [`INITIALIZE-OPTIMIZER*`][7c2f] and
    [`INITIALIZE-GRADIENT-SOURCE*`][dd95], this generic function is the main
    extension point for writing optimizers.

<a id="x-28MGL-OPT-3AINITIALIZE-OPTIMIZER-2A-20GENERIC-FUNCTION-29"></a>
- [generic-function] **INITIALIZE-OPTIMIZER\*** *OPTIMIZER GRADIENT-SOURCE WEIGHTS DATASET*

    Called automatically before training starts, this
    function sets up `OPTIMIZER` to be suitable for optimizing
    `GRADIENT-SOURCE`. It typically creates appropriately sized
    accumulators for the gradients.

<a id="x-28MGL-OPT-3ASEGMENTS-20GENERIC-FUNCTION-29"></a>
- [generic-function] **SEGMENTS** *OPTIMIZER*

    Several weight matrices known as *segments* can be
    optimized by a single optimizer. This function returns them as a
    list.

The rest are just useful for utilities for implementing
optimizers.

<a id="x-28MGL-OPT-3ATERMINATE-OPTIMIZATION-P-20FUNCTION-29"></a>
- [function] **TERMINATE-OPTIMIZATION-P** *N-INSTANCES TERMINATION*

    Utility function for subclasses of [`ITERATIVE-OPTIMIZER`][8da0]. It returns
    whether optimization is to be terminated based on `N-INSTANCES` and
    `TERMINATION` that are values of the respective accessors of
    `ITERATIVE-OPTIMIZER`.

<a id="x-28MGL-OPT-3ASET-N-INSTANCES-20FUNCTION-29"></a>
- [function] **SET-N-INSTANCES** *OPTIMIZER GRADIENT-SOURCE N-INSTANCES*

    Set [`N-INSTANCES`][4c73] of `OPTIMIZER` and
    fire [`ON-N-INSTANCES-CHANGED`][4f0b]. [`ITERATIVE-OPTIMIZER`][8da0] subclasses must
    call this to increment [`N-INSTANCES`][4c73].

<a id="x-28MGL-OPT-3ASEGMENT-SET-20CLASS-29"></a>
- [class] **SEGMENT-SET**

    This is a utility class for optimizers that have a
    list of [`SEGMENTS`][f00d] and (the weights being optimized) is able to copy
    back and forth between those segments and a single `MAT` (the
    accumulator).

<a id="x-28MGL-OPT-3ASEGMENTS-20-28MGL-PAX-3AREADER-20MGL-OPT-3ASEGMENT-SET-29-29"></a>
- [reader] **SEGMENTS** *SEGMENT-SET (:SEGMENTS)*

    A list of weight matrices.

<a id="x-28MGL-COMMON-3ASIZE-20-28MGL-PAX-3AREADER-20MGL-OPT-3ASEGMENT-SET-29-29"></a>
- [reader] **SIZE** *SEGMENT-SET*

    The sum of the sizes of the weight matrices of
    [`SEGMENTS`][f00d].

<a id="x-28MGL-OPT-3ADO-SEGMENT-SET-20MGL-PAX-3AMACRO-29"></a>
- [macro] **DO-SEGMENT-SET** *(SEGMENT &OPTIONAL START) SEGMENT-SET &BODY BODY*

    Iterate over [`SEGMENTS`][f00d] in `SEGMENT-SET`. If `START` is specified, the it
    is bound to the start index of `SEGMENT` within `SEGMENT-SET`. The start
    index is the sum of the sizes of previous segments.

<a id="x-28MGL-OPT-3ASEGMENT-SET-3C-MAT-20FUNCTION-29"></a>
- [function] **SEGMENT-SET\<-MAT** *SEGMENT-SET MAT*

    Copy the values of `MAT` to the weight matrices of `SEGMENT-SET` as if
    they were concatenated into a single `MAT`.

<a id="x-28MGL-OPT-3ASEGMENT-SET--3EMAT-20FUNCTION-29"></a>
- [function] **SEGMENT-SET-\>MAT** *SEGMENT-SET MAT*

    Copy the values of `SEGMENT-SET` to `MAT` as if they were concatenated
    into a single `MAT`.

<a id="x-28MGL-OPT-3A-40MGL-OPT-GRADIENT-SOURCE-20MGL-PAX-3ASECTION-29"></a>
#### 9.5.2 Implementing Gradient Sources

Weights can be stored in a multitude of ways. Optimizers need to
update weights, so it is assumed that weights are stored in any
number of `MAT` objects called segments.

The generic functions in this section must all be specialized for
new gradient sources except where noted.

<a id="x-28MGL-OPT-3AMAP-SEGMENTS-20GENERIC-FUNCTION-29"></a>
- [generic-function] **MAP-SEGMENTS** *FN GRADIENT-SOURCE*

    Apply `FN` to each segment of `GRADIENT-SOURCE`.

<a id="x-28MGL-OPT-3AMAP-SEGMENT-RUNS-20GENERIC-FUNCTION-29"></a>
- [generic-function] **MAP-SEGMENT-RUNS** *FN SEGMENT*

    Call `FN` with start and end of intervals of
    consecutive indices that are not missing in `SEGMENT`. Called by
    optimizers that support partial updates. The default implementation
    assumes that all weights are present. This only needs to be
    specialized if one plans to use an optimizer that knows how to deal
    unused/missing weights such as [`MGL-GD:NORMALIZED-BATCH-GD-OPTIMIZER`][f6ae]
    and `OPTIMIZER` [`MGL-GD:PER-WEIGHT-BATCH-GD-OPTIMIZER`][5a43].

<a id="x-28MGL-OPT-3ASEGMENT-WEIGHTS-20GENERIC-FUNCTION-29"></a>
- [generic-function] **SEGMENT-WEIGHTS** *SEGMENT*

    Return the weight matrix of `SEGMENT`. A segment
    doesn't need to be a `MAT` object itself. For example, it may be a
    `MGL-BM:CHUNK` of a [`MGL-BM:BM`][7e58] or a [`MGL-BP:LUMP`][c1ac] of a
    [`MGL-BP:BPN`][5187] whose [`NODES`][cc1c] slot holds the weights.

<a id="x-28MGL-OPT-3ASEGMENT-WEIGHTS-20-28METHOD-20NIL-20-28MGL-MAT-3AMAT-29-29-29"></a>
- [method] **SEGMENT-WEIGHTS** *(MAT MAT)*

    When the segment is really a `MAT`, then just return it.

<a id="x-28MGL-OPT-3ASEGMENT-DERIVATIVES-20GENERIC-FUNCTION-29"></a>
- [generic-function] **SEGMENT-DERIVATIVES** *SEGMENT*

    Return the derivatives matrix of `SEGMENT`. A segment
    doesn't need to be a `MAT` object itself. For example, it may be a
    `MGL-BM:CHUNK` of a [`MGL-BM:BM`][7e58] or a [`MGL-BP:LUMP`][c1ac] of a
    [`MGL-BP:BPN`][5187] whose DERIVATIVES slot holds the gradient.

<a id="x-28MGL-OPT-3ALIST-SEGMENTS-20FUNCTION-29"></a>
- [function] **LIST-SEGMENTS** *GRADIENT-SOURCE*

    A utility function that returns the list of segments from
    [`MAP-SEGMENTS`][2312] on `GRADIENT-SOURCE`.

<a id="x-28MGL-OPT-3AINITIALIZE-GRADIENT-SOURCE-2A-20GENERIC-FUNCTION-29"></a>
- [generic-function] **INITIALIZE-GRADIENT-SOURCE\*** *OPTIMIZER GRADIENT-SOURCE WEIGHTS DATASET*

    Called automatically before [`MINIMIZE*`][ae3d] is called,
    this function may be specialized if `GRADIENT-SOURCE` needs some kind
    of setup.

<a id="x-28MGL-OPT-3AINITIALIZE-GRADIENT-SOURCE-2A-20-28METHOD-20NIL-20-28T-20T-20T-20T-29-29-29"></a>
- [method] **INITIALIZE-GRADIENT-SOURCE\*** *OPTIMIZER GRADIENT-SOURCE WEIGHTS DATASET*

    The default method does nothing.

<a id="x-28MGL-OPT-3AACCUMULATE-GRADIENTS-2A-20GENERIC-FUNCTION-29"></a>
- [generic-function] **ACCUMULATE-GRADIENTS\*** *GRADIENT-SOURCE SINK BATCH MULTIPLIER VALUEP*

    Add `MULTIPLIER` times the sum of first-order
    gradients to accumulators of `SINK` (normally accessed with
    [`DO-GRADIENT-SINK`][20ca]) and if `VALUEP`, return the sum of values of the
    function being optimized for a `BATCH` of instances. `GRADIENT-SOURCE`
    is the object representing the function being optimized, `SINK` is
    gradient sink.
    
    Note the number of instances in `BATCH` may be larger than what
    `GRADIENT-SOURCE` process in one go (in the sense of say,
    [`MAX-N-STRIPES`][16c4]), so [`DO-BATCHES-FOR-MODEL`][faaa] or something like (`GROUP`
    `BATCH` `MAX-N-STRIPES`) can be handy.

<a id="x-28MGL-OPT-3A-40MGL-OPT-GRADIENT-SINK-20MGL-PAX-3ASECTION-29"></a>
#### 9.5.3 Implementing Gradient Sinks

Optimizers call [`ACCUMULATE-GRADIENTS*`][4bf1] on gradient sources. One
parameter of `ACCUMULATE-GRADIENTS*` is the `SINK`. A gradient sink
knows what accumulator matrix (if any) belongs to a segment. Sinks
are defined entirely by [`MAP-GRADIENT-SINK`][aabd].

<a id="x-28MGL-OPT-3AMAP-GRADIENT-SINK-20GENERIC-FUNCTION-29"></a>
- [generic-function] **MAP-GRADIENT-SINK** *FN SINK*

    Call `FN` of lambda list (`SEGMENT` `ACCUMULATOR`) on
    each segment and their corresponding accumulator `MAT` in `SINK`.

<a id="x-28MGL-OPT-3ADO-GRADIENT-SINK-20MGL-PAX-3AMACRO-29"></a>
- [macro] **DO-GRADIENT-SINK** *((SEGMENT ACCUMULATOR) SINK) &BODY BODY*

    A convenience macro on top of [`MAP-GRADIENT-SINK`][aabd].

<a id="x-28MGL-DIFFUN-3A-40MGL-DIFFUN-20MGL-PAX-3ASECTION-29"></a>
## 10 Differentiable Functions

###### \[in package MGL-DIFFUN\]
<a id="x-28MGL-DIFFUN-3ADIFFUN-20CLASS-29"></a>
- [class] **DIFFUN**

    `DIFFUN` dresses a lisp function (in its [`FN`][f491] slot) as
    a gradient source (see [Implementing Gradient Sources][c58b]) which allows it to
    be used in [`MINIMIZE`][46a4]. See the examples in [Gradient Descent][10e7] and
    [Conjugate Gradient][83e6].

<a id="x-28MGL-COMMON-3AFN-20-28MGL-PAX-3AREADER-20MGL-DIFFUN-3ADIFFUN-29-29"></a>
- [reader] **FN** *DIFFUN (:FN)*

    A real valued lisp function. It may have any
    number of parameters.

<a id="x-28MGL-DIFFUN-3APARAMETER-INDICES-20-28MGL-PAX-3AREADER-20MGL-DIFFUN-3ADIFFUN-29-29"></a>
- [reader] **PARAMETER-INDICES** *DIFFUN (:PARAMETER-INDICES = NIL)*

    The list of indices of parameters that we don't
    optimize. Values for these will come from the DATASET argument of
    [`MINIMIZE`][46a4].

<a id="x-28MGL-DIFFUN-3AWEIGHT-INDICES-20-28MGL-PAX-3AREADER-20MGL-DIFFUN-3ADIFFUN-29-29"></a>
- [reader] **WEIGHT-INDICES** *DIFFUN (:WEIGHT-INDICES = NIL)*

    The list of indices of parameters to be optimized,
    the values of which will come from the `WEIGHTS`
    argument of [`MINIMIZE`][46a4].

<a id="x-28MGL-BP-3A-40MGL-BP-20MGL-PAX-3ASECTION-29"></a>
## 11 Backpropagation Neural Networks

###### \[in package MGL-BP\]
<a id="x-28MGL-BP-3A-40MGL-BP-OVERVIEW-20MGL-PAX-3ASECTION-29"></a>
### 11.1 Backprop Overview

Backpropagation Neural Networks are just functions with lots of
parameters called *weights* and a layered structure when presented
as a [computational
graph](http://en.wikipedia.org/wiki/Automatic_differentiation). The
network is trained to [`MINIMIZE`][46a4] some kind of *loss function* whose
value the network computes.

In this implementation, a [`BPN`][5187] is assembled from several
[`LUMP`][c1ac]s (roughly corresponding to layers). Both feed-forward and
recurrent neural nets are supported ([`FNN`][9de4] and [`RNN`][b0f3], respectively).
`BPN`s can contain not only `LUMP`s but other `BPN`s, too. As we
see, networks are composite objects and the abstract base class for
composite and simple parts is called [`CLUMP`][a4fe].

<a id="x-28MGL-BP-3ACLUMP-20CLASS-29"></a>
- [class] **CLUMP**

    A `CLUMP` is a [`LUMP`][c1ac] or a [`BPN`][5187]. It represents
    a differentiable function. Arguments of clumps are given during
    instantiation. Some arguments are clumps themselves so they get
    permenantly wired together like this:
    
    ```commonlisp
    (->v*m (->input :size 10 :name 'input)
           (->weight :dimensions '(10 20) :name 'weight)
           :name 'activation)
    ```
    
    The above creates three clumps: the vector-matrix multiplication
    clumps called `ACTIVATION` which has a reference to its operands:
    `INPUT` and `WEIGHT`. Note that the example just defines a function, no
    actual computation has taken place, yet.
    
    This wiring of `CLUMP`s is how one builds feed-forward nets ([`FNN`][9de4]) or
    recurrent neural networks ([`RNN`][b0f3]) that are `CLUMP`s themselves so one
    can build nets in a hiearchical style if desired. Non-composite
    `CLUMP`s are called `LUMP` (note the loss of `C` that stands for
    composite). The various `LUMP` subtypes correspond to different layer
    types ([`->SIGMOID`][83f9], [`->DROPOUT`][441b], [`->RELU`][9d3a], [`->TANH`][5309], etc).

At this point, you may want to jump ahead to get a feel for how
things work by reading the [`FNN` Tutorial][6b38].

<a id="x-28MGL-BP-3A-40MGL-BP-EXTENSION-API-20MGL-PAX-3ASECTION-29"></a>
### 11.2 Clump API

These are mostly for extension purposes. About the only thing
needed from here for normal operation is [`NODES`][cc1c] when clamping inputs
or extracting predictions.

<a id="x-28MGL-BP-3ASTRIPEDP-20GENERIC-FUNCTION-29"></a>
- [generic-function] **STRIPEDP** *CLUMP*

    For efficiency, forward and backprop phases do
    their stuff in batch mode: passing a number of instances through the
    network in batches. Thus clumps must be able to store values of and
    gradients for each of these instances. However, some clumps produce
    the same result for each instance in a batch. These clumps are the
    weights, the parameters of the network. `STRIPEDP` returns true iff
    `CLUMP` does not represent weights (i.e. it's not a [`->WEIGHT`][b76f]).
    
    For striped clumps, their [`NODES`][cc1c] and [`DERIVATIVES`][a81b] are `MAT` objects with
    a leading dimension (number of rows in the 2d case) equal to the
    number of instances in the batch. Non-striped clumps have no
    restriction on their shape apart from what their usage dictates.

<a id="x-28MGL-COMMON-3ANODES-20GENERIC-FUNCTION-29"></a>
- [generic-function] **NODES** *OBJECT*

    Returns a `MAT` object representing the state or
    result of `OBJECT`. The first dimension of the returned matrix is
    equal to the number of stripes.

[`CLUMP`][a4fe]s' [`NODES`][cc1c] holds the result computed by the most recent
[`FORWARD`][c1ae]. For [`->INPUT`][f54e] lumps, this is where input values shall be
placed (see [`SET-INPUT`][0c9e]). Currently, the matrix is always two
dimensional but this restriction may go away in the future.

<a id="x-28MGL-BP-3ADERIVATIVES-20GENERIC-FUNCTION-29"></a>
- [generic-function] **DERIVATIVES** *CLUMP*

    Return the `MAT` object representing the partial
    derivatives of the function `CLUMP` computes. The returned partial
    derivatives were accumulated by previous [`BACKWARD`][5bd4] calls.
    
    This matrix is shaped like the matrix returned by [`NODES`][cc1c].

<a id="x-28MGL-BP-3AFORWARD-20GENERIC-FUNCTION-29"></a>
- [generic-function] **FORWARD** *CLUMP*

    Compute the values of the function represented by
    `CLUMP` for all stripes and place the results into [`NODES`][cc1c] of `CLUMP`.

<a id="x-28MGL-BP-3ABACKWARD-20GENERIC-FUNCTION-29"></a>
- [generic-function] **BACKWARD** *CLUMP*

    Compute the partial derivatives of the function
    represented by `CLUMP` and add them to [`DERIVATIVES`][a81b] of the
    corresponding argument clumps. The `DERIVATIVES` of `CLUMP` contains the
    sum of partial derivatives of all clumps by the corresponding
    output. This function is intended to be called after a [`FORWARD`][c1ae] pass.
    
    Take the [`->SIGMOID`][83f9] clump for example when the network is being
    applied to a batch of two instances `x1` and `x2`. `x1` and `x2` are
    set in the [`->INPUT`][f54e] lump X. The sigmoid computes `1/(1+exp(-x))`
    where `X` is its only argument clump.
    
        f(x) = 1/(1+exp(-x))
    
    When `BACKWARD` is called on the sigmoid lump, its `DERIVATIVES` is a
    2x1 `MAT` object that contains the partial derivatives of the loss
    function:
    
        dL(x1)/df
        dL(x2)/df
    
    Now the `BACKWARD` method of the sigmoid needs to add `dL(x1)/dx1` and
    `dL(x2)/dx2` to `DERIVATIVES` of `X`. Now, `dL(x1)/dx1 = dL(x1)/df *
    df(x1)/dx1` and the first term is what we have in `DERIVATIVES` of the
    sigmoid so it only needs to calculate the second term.

In addition to the above, clumps also have to support `SIZE`([`0`][85d3] [`1`][b99d]),
[`N-STRIPES`][8dd7], [`MAX-N-STRIPES`][16c4] (and the [`SETF`][17b7] methods of the latter two)
which can be accomplished just by inheriting from [`BPN`][5187], [`FNN`][9de4], [`RNN`][b0f3], or
a [`LUMP`][c1ac].

<a id="x-28MGL-BP-3A-40MGL-BPN-20MGL-PAX-3ASECTION-29"></a>
### 11.3 `BPN`s

<a id="x-28MGL-BP-3ABPN-20CLASS-29"></a>
- [class] **BPN** *[CLUMP][a4fe]*

    Abstract base class for [`FNN`][9de4] and [`RNN`][b0f3].

<a id="x-28MGL-CORE-3AN-STRIPES-20-28MGL-PAX-3AREADER-20MGL-BP-3ABPN-29-29"></a>
- [reader] **N-STRIPES** *BPN (:N-STRIPES = 1)*

    The current number of instances the network has.
    This is automatically set to the number of instances passed to
    [`SET-INPUT`][0c9e], so it rarely has to be manipulated directly although it
    can be set. When set `N-STRIPES` of all `CLUMPS`([`0`][f7c1] [`1`][a4fe]) get set to the same
    value.

<a id="x-28MGL-CORE-3AMAX-N-STRIPES-20-28MGL-PAX-3AREADER-20MGL-BP-3ABPN-29-29"></a>
- [reader] **MAX-N-STRIPES** *BPN (:MAX-N-STRIPES = NIL)*

    The maximum number of instances the network can
    operate on in parallel. Within [`BUILD-FNN`][606c] or [`BUILD-RNN`][764b], it defaults
    to `MAX-N-STRIPES` of that parent network, else it defaults to 1.
    When set `MAX-N-STRIPES` of all `CLUMPS`([`0`][f7c1] [`1`][a4fe]) get set to the same value.

<a id="x-28MGL-BP-3ACLUMPS-20-28MGL-PAX-3AREADER-20MGL-BP-3ABPN-29-29"></a>
- [reader] **CLUMPS** *BPN (:CLUMPS = (MAKE-ARRAY 0 :ELEMENT-TYPE 'CLUMP :ADJUSTABLE T :FILL-POINTER T))*

    A topological sorted adjustable array with a fill
    pointer that holds the clumps that make up the network. Clumps are
    added to it by [`ADD-CLUMP`][82d8] or, more often, automatically when within
    a [`BUILD-FNN`][606c] or [`BUILD-RNN`][764b]. Rarely needed, [`FIND-CLUMP`][175f] takes care of
    most uses.

<a id="x-28MGL-BP-3AFIND-CLUMP-20FUNCTION-29"></a>
- [function] **FIND-CLUMP** *NAME BPN &KEY (ERRORP T)*

    Find the clump with `NAME` among `CLUMPS`([`0`][f7c1] [`1`][a4fe]) of `BPN`. As always, names are
    compared with [`EQUAL`][96d0]. If not found, then return `NIL` or signal and
    error depending on `ERRORP`.

<a id="x-28MGL-BP-3AADD-CLUMP-20FUNCTION-29"></a>
- [function] **ADD-CLUMP** *CLUMP BPN*

    Add `CLUMP` to `BPN`. [`MAX-N-STRIPES`][16c4] of `CLUMP` gets set to that of `BPN`.
    It is an error to add a clump with a name already used by one of the
    [`CLUMPS`][f7c1] of `BPN`.

<a id="x-28MGL-BP-3A-40MGL-BP-TRAINING-20MGL-PAX-3ASECTION-29"></a>
#### 11.3.1 Training

[`BPN`][5187]s are trained to minimize the loss function they compute.
Before a `BPN` is passed to [`MINIMIZE`][46a4] (as its `GRADIENT-SOURCE`
argument), it must be wrapped in a [`BP-LEARNER`][00a0] object. `BP-LEARNER` has
[`MONITORS`][6202] slot which is used for example by
[`RESET-OPTIMIZATION-MONITORS`][e4f9].

Without the bells an whistles, the basic shape of training is this:

```commonlisp
(minimize optimizer (make-instance 'bp-learner :bpn bpn)
          :dataset dataset)
```


<a id="x-28MGL-BP-3ABP-LEARNER-20CLASS-29"></a>
- [class] **BP-LEARNER**

<a id="x-28MGL-BP-3ABPN-20-28MGL-PAX-3AREADER-20MGL-BP-3ABP-LEARNER-29-29"></a>
- [reader] **BPN** *BP-LEARNER (:BPN)*

    The `BPN` for which this [`BP-LEARNER`][00a0] provides the
    gradients.

<a id="x-28MGL-CORE-3AMONITORS-20-28MGL-PAX-3AACCESSOR-20MGL-BP-3ABP-LEARNER-29-29"></a>
- [accessor] **MONITORS** *BP-LEARNER (:MONITORS = NIL)*

    A list of [`MONITOR`][7068]s.

<a id="x-28MGL-BP-3A-40MGL-BP-MONITORING-20MGL-PAX-3ASECTION-29"></a>
#### 11.3.2 Monitoring

<a id="x-28MGL-BP-3AMONITOR-BPN-RESULTS-20FUNCTION-29"></a>
- [function] **MONITOR-BPN-RESULTS** *DATASET BPN MONITORS*

    For every batch (of size [`MAX-N-STRIPES`][16c4] of `BPN`) of instances in
    `DATASET`, set the batch as the next input with [`SET-INPUT`][0c9e], perform a
    [`FORWARD`][c1ae] pass and apply `MONITORS` to the `BPN` (with `APPLY-MONITORS`([`0`][989c] [`1`][bbdf])).
    Finally, return the counters of `MONITORS`. This is built on top of
    [`MONITOR-MODEL-RESULTS`][e50c].

<a id="x-28MGL-BP-3AMAKE-STEP-MONITOR-MONITORS-20FUNCTION-29"></a>
- [function] **MAKE-STEP-MONITOR-MONITORS** *RNN &KEY (COUNTER-VALUES-FN #'COUNTER-RAW-VALUES) (MAKE-COUNTER #'MAKE-STEP-MONITOR-MONITOR-COUNTER)*

    Return a list of monitors, one for every monitor in [`STEP-MONITORS`][71f9]
    of `RNN`. These monitors extract the results from their warp
    counterpairs with `COUNTER-VALUES-FN` and add them to their own
    counter that's created by `MAKE-COUNTER`. Wow. Ew. The idea is that
    one does something like this do monitor warped prediction:
    
    ```commonlisp
    (let ((*warp-time* t))
      (setf (step-monitors rnn)
            (make-cost-monitors rnn :attributes '(:event "warped pred.")))
      (monitor-bpn-results dataset rnn
                           ;; Just collect and reset the warp
                           ;; monitors after each batch of
                           ;; instances.
                           (make-step-monitor-monitors rnn)))
    ```


<a id="x-28MGL-BP-3AMAKE-STEP-MONITOR-MONITOR-COUNTER-20GENERIC-FUNCTION-29"></a>
- [generic-function] **MAKE-STEP-MONITOR-MONITOR-COUNTER** *STEP-COUNTER*

    In an [`RNN`][b0f3], `STEP-COUNTER` aggregates results of all
    the time steps during the processing of instances in the current
    batch. Return a new counter into which results from `STEP-COUNTER` can
    be accumulated when the processing of the batch is finished. The
    default implementation creates a copy of `STEP-COUNTER`.

<a id="x-28MGL-BP-3A-40MGL-FNN-20MGL-PAX-3ASECTION-29"></a>
#### 11.3.3 Feed-Forward Nets

[`FNN`][9de4] and [`RNN`][b0f3] have a lot in common (see their common superclass, [`BPN`][5187]).
There is very limited functionality that's specific to `FNN`s so let's
get them out of they way before we study a full example.

<a id="x-28MGL-BP-3AFNN-20CLASS-29"></a>
- [class] **FNN** *[BPN][5187]*

    A feed-forward neural net (as opposed to a
    recurrent one, see [`RNN`][b0f3]).

<a id="x-28MGL-BP-3ABUILD-FNN-20MGL-PAX-3AMACRO-29"></a>
- [macro] **BUILD-FNN** *(&KEY FNN (CLASS ''FNN) INITARGS MAX-N-STRIPES NAME) &BODY CLUMPS*

    Syntactic sugar to assemble `FNN`s from [`CLUMP`][a4fe]s. Like [`LET*`][2739], it is a
    sequence of bindings (of symbols to `CLUMP`s). The names of the clumps
    created default to the symbol of the binding. In case a clump is not
    bound to a symbol (because it was created in a nested expression),
    the local function `CLUMP` can be used to find the clump with the
    given name in the fnn being built. Example:
    
        (build-fnn ()
          (features (->input :size n-features))
          (biases (->weight :size n-features))
          (weights (->weight :size (* n-hiddens n-features)))
          (activations0 (->v*m :weights weights :x (clump 'features)))
          (activations (->+ :args (list biases activations0)))
          (output (->sigmoid :x activations)))


<a id="x-28MGL-BP-3A-40MGL-FNN-TUTORIAL-20MGL-PAX-3ASECTION-29"></a>
##### `FNN` Tutorial

Hopefully this example from `example/digit-fnn.lisp` illustrates
the concepts involved. If it's too dense despite the comments, then
read up on [Datasets][109e], [Gradient Based Optimization][c74a] and come back.

<a id="x-28MGL-BP-3ADIGIT-FNN-2ELISP-20-28MGL-PAX-3AINCLUDE-20-23P-22-2Fhome-2Fmelisgl-2Fown-2Fmgl-2Fexample-2Fdigit-fnn-2Elisp-22-20-3AHEADER-NL-20-22-60-60-60commonlisp-22-20-3AFOOTER-NL-20-22-60-60-60-22-29-29"></a>
```commonlisp
(cl:defpackage :mgl-example-digit-fnn
  (:use #:common-lisp #:mgl))

(in-package :mgl-example-digit-fnn)

;;; There are 10 possible digits used as inputs ...
(defparameter *n-inputs* 10)
;;; and we want to learn the rule that maps the input digit D to (MOD
;;; (1+ D) 3).
(defparameter *n-outputs* 3)

;;; We define a feed-forward net to be able to specialize how inputs
;;; are translated by adding a SET-INPUT method later.
(defclass digit-fnn (fnn)
  ())

;;; Build a DIGIT-FNN with a single hidden layer of rectified linear
;;; units and a softmax output.
(defun make-digit-fnn (&key (n-hiddens 5))
  (build-fnn (:class 'digit-fnn)
    (input (->input :size *n-inputs*))
    (hidden-activation (->activation input :size n-hiddens))
    (hidden (->relu hidden-activation))
    (output-activation (->activation hidden :size *n-outputs*))
    (output (->softmax-xe-loss output-activation))))

;;; This method is called with batches of 'instances' (input digits in
;;; this case) by MINIMIZE and also by MONITOR-BPN-RESULTS before
;;; performing a forward pass (i.e. computing the value of the
;;; function represented by the network). Its job is to encode the
;;; inputs by populating rows of the NODES matrix of the INPUT clump.
;;;
;;; Each input is encoded as a row of zeros with a single 1 at index
;;; determined by the input digit. This is called one-hot encoding.
;;; The TARGET could be encoded the same way, but instead we use the
;;; sparse option supported by TARGET of ->SOFTMAX-XE-LOSS.
(defmethod set-input (digits (fnn digit-fnn))
  (let* ((input (nodes (find-clump 'input fnn)))
         (output-lump (find-clump 'output fnn)))
    (fill! 0 input)
    (loop for i upfrom 0
          for digit in digits
          do (setf (mref input i digit) 1))
    (setf (target output-lump)
          (mapcar (lambda (digit)
                    (mod (1+ digit) *n-outputs*))
                  digits))))

;;; Train the network by minimizing the loss (cross-entropy here) with
;;; stochastic gradient descent.
(defun train-digit-fnn ()
  (let ((optimizer
          ;; First create the optimizer for MINIMIZE.
          (make-instance 'segmented-gd-optimizer
                         :segmenter
                         ;; We train each weight lump with the same
                         ;; parameters and, in fact, the same
                         ;; optimizer. But it need not be so, in
                         ;; general.
                         (constantly
                          (make-instance 'sgd-optimizer
                                         :learning-rate 1
                                         :momentum 0.9
                                         :batch-size 100))))
        (fnn (make-digit-fnn)))
    ;; The number of instances the FNN can work with in parallel. It's
    ;; usually equal to the batch size or is a its divisor.
    (setf (max-n-stripes fnn) 50)
    ;; Initialize all weights randomly.
    (map-segments (lambda (weights)
                    (gaussian-random! (nodes weights) :stddev 0.01))
                  fnn)
    ;; Arrange for training and test error to be logged.
    (monitor-optimization-periodically
     optimizer '((:fn log-test-error :period 10000)
                 (:fn reset-optimization-monitors :period 1000)))
    ;; Finally, start the optimization.
    (minimize optimizer
              ;; Dress FNN in a BP-LEARNER and attach monitors for the
              ;; cost to it. These monitors are going to be logged and
              ;; reset after every 100 training instance by
              ;; RESET-OPTIMIZATION-MONITORS above.
              (make-instance 'bp-learner
                             :bpn fnn
                             :monitors (make-cost-monitors
                                        fnn :attributes `(:event "train")))
              ;; Training stops when the sampler runs out (after 10000
              ;; instances).
              :dataset (make-sampler 10000))))

;;; Return a sampler object that produces MAX-N-SAMPLES number of
;;; random inputs (numbers between 0 and 9).
(defun make-sampler (max-n-samples)
  (make-instance 'function-sampler :max-n-samples max-n-samples
                 :generator (lambda () (random *n-inputs*))))

;;; Log the test error. Also, describe the optimizer and the bpn at
;;; the beginning of training. Called periodically during training
;;; (see above).
(defun log-test-error (optimizer learner)
  (when (zerop (n-instances optimizer))
    (describe optimizer)
    (describe (bpn learner)))
  (log-padded
   (monitor-bpn-results (make-sampler 1000) (bpn learner)
                        (make-cost-monitors
                         (bpn learner) :attributes `(:event "pred.")))))

#|

;;; Transcript follows:
(repeatably ()
  (let ((*log-time* nil))
    (train-digit-fnn)))
.. training at n-instances: 0
.. train cost: 0.000e+0 (0)
.. #<SEGMENTED-GD-OPTIMIZER {100E112E93}>
..  SEGMENTED-GD-OPTIMIZER description:
..    N-INSTANCES = 0
..    OPTIMIZERS = (#<SGD-OPTIMIZER
..                    #<SEGMENT-SET
..                      (#<->WEIGHT # :SIZE 15 1/1 :NORM 0.04473>
..                       #<->WEIGHT # :SIZE 3 1/1 :NORM 0.01850>
..                       #<->WEIGHT # :SIZE 50 1/1 :NORM 0.07159>
..                       #<->WEIGHT # :SIZE 5 1/1 :NORM 0.03056>)
..                      {100E335B73}>
..                    {100E06DF83}>)
..    SEGMENTS = (#<->WEIGHT (HIDDEN OUTPUT-ACTIVATION) :SIZE
..                  15 1/1 :NORM 0.04473>
..                #<->WEIGHT (:BIAS OUTPUT-ACTIVATION) :SIZE
..                  3 1/1 :NORM 0.01850>
..                #<->WEIGHT (INPUT HIDDEN-ACTIVATION) :SIZE
..                  50 1/1 :NORM 0.07159>
..                #<->WEIGHT (:BIAS HIDDEN-ACTIVATION) :SIZE
..                  5 1/1 :NORM 0.03056>)
..  
.. #<SGD-OPTIMIZER {100E06DF83}>
..  GD-OPTIMIZER description:
..    N-INSTANCES = 0
..    SEGMENT-SET = #<SEGMENT-SET
..                    (#<->WEIGHT (HIDDEN OUTPUT-ACTIVATION) :SIZE
..                       15 1/1 :NORM 0.04473>
..                     #<->WEIGHT (:BIAS OUTPUT-ACTIVATION) :SIZE
..                       3 1/1 :NORM 0.01850>
..                     #<->WEIGHT (INPUT HIDDEN-ACTIVATION) :SIZE
..                       50 1/1 :NORM 0.07159>
..                     #<->WEIGHT (:BIAS HIDDEN-ACTIVATION) :SIZE
..                       5 1/1 :NORM 0.03056>)
..                    {100E335B73}>
..    LEARNING-RATE = 1.00000e+0
..    MOMENTUM = 9.00000e-1
..    MOMENTUM-TYPE = :NORMAL
..    WEIGHT-DECAY = 0.00000e+0
..    WEIGHT-PENALTY = 0.00000e+0
..    N-AFTER-UPATE-HOOK = 0
..    BATCH-SIZE = 100
..  
..  BATCH-GD-OPTIMIZER description:
..    N-BEFORE-UPATE-HOOK = 0
..  #<DIGIT-FNN {100E11A423}>
..   BPN description:
..     CLUMPS = #(#<->INPUT INPUT :SIZE 10 1/50 :NORM 0.00000>
..                #<->ACTIVATION
..                  (HIDDEN-ACTIVATION :ACTIVATION) :STRIPES 1/50
..                  :CLUMPS 4>
..                #<->RELU HIDDEN :SIZE 5 1/50 :NORM 0.00000>
..                #<->ACTIVATION
..                  (OUTPUT-ACTIVATION :ACTIVATION) :STRIPES 1/50
..                  :CLUMPS 4>
..                #<->SOFTMAX-XE-LOSS OUTPUT :SIZE 3 1/50 :NORM 0.00000>)
..     N-STRIPES = 1
..     MAX-N-STRIPES = 50
..   pred. cost: 1.100d+0 (1000.00)
.. training at n-instances: 1000
.. train cost: 1.093d+0 (1000.00)
.. training at n-instances: 2000
.. train cost: 5.886d-1 (1000.00)
.. training at n-instances: 3000
.. train cost: 3.574d-3 (1000.00)
.. training at n-instances: 4000
.. train cost: 1.601d-7 (1000.00)
.. training at n-instances: 5000
.. train cost: 1.973d-9 (1000.00)
.. training at n-instances: 6000
.. train cost: 4.882d-10 (1000.00)
.. training at n-instances: 7000
.. train cost: 2.771d-10 (1000.00)
.. training at n-instances: 8000
.. train cost: 2.283d-10 (1000.00)
.. training at n-instances: 9000
.. train cost: 2.123d-10 (1000.00)
.. training at n-instances: 10000
.. train cost: 2.263d-10 (1000.00)
.. pred. cost: 2.210d-10 (1000.00)
..
==> (#<->WEIGHT (:BIAS HIDDEN-ACTIVATION) :SIZE 5 1/1 :NORM 2.94294>
-->  #<->WEIGHT (INPUT HIDDEN-ACTIVATION) :SIZE 50 1/1 :NORM 11.48995>
-->  #<->WEIGHT (:BIAS OUTPUT-ACTIVATION) :SIZE 3 1/1 :NORM 3.39103>
-->  #<->WEIGHT (HIDDEN OUTPUT-ACTIVATION) :SIZE 15 1/1 :NORM 11.39339>)

|#
```

<a id="x-28MGL-BP-3A-40MGL-RNN-20MGL-PAX-3ASECTION-29"></a>
#### 11.3.4 Recurrent Neural Nets

<a id="x-28MGL-BP-3A-40MGL-RNN-TUTORIAL-20MGL-PAX-3ASECTION-29"></a>
##### `RNN` Tutorial

Hopefully this example from `example/sum-sign-fnn.lisp` illustrates
the concepts involved. Make sure you are comfortable with
[`FNN` Tutorial][6b38] before reading this.

<a id="x-28MGL-BP-3ASUM-SIG-RNN-2ELISP-20-28MGL-PAX-3AINCLUDE-20-23P-22-2Fhome-2Fmelisgl-2Fown-2Fmgl-2Fexample-2Fsum-sign-rnn-2Elisp-22-20-3AHEADER-NL-20-22-60-60-60commonlisp-22-20-3AFOOTER-NL-20-22-60-60-60-22-29-29"></a>
```commonlisp
(cl:defpackage :mgl-example-sum-sign-rnn
  (:use #:common-lisp #:mgl))

(in-package :mgl-example-sum-sign-rnn)

;;; There is a single input at each time step...
(defparameter *n-inputs* 1)
;;; and we want to learn the rule that outputs the sign of the sum of
;;; inputs so far in the sequence.
(defparameter *n-outputs* 3)

;;; Generate a training example that's a sequence of random length
;;; between 1 and LENGTH. Elements of the sequence are lists of two
;;; elements:
;;;
;;; 1. The input for the network (a single random number).
;;;
;;; 2. The sign of the sum of inputs so far encoded as 0, 1, 2 (for
;;;    negative, zero and positive values). To add a twist, the sum is
;;;    reset whenever a negative input is seen.
(defun make-sum-sign-instance (&key (length 10))
  (let ((length (max 1 (random length)))
        (sum 0))
    (loop for i below length
          collect (let ((x (1- (* 2 (random 2)))))
                    (incf sum x)
                    (when (< x 0)
                      (setq sum x))
                    (list x (cond ((minusp sum) 0)
                                  ((zerop sum) 1)
                                  (t 2)))))))

;;; Build an RNN with a single lstm hidden layer and softmax output.
;;; For each time step, a SUM-SIGN-FNN will be instantiated.
(defun make-sum-sign-rnn (&key (n-hiddens 1))
  (build-rnn ()
    (build-fnn (:class 'sum-sign-fnn)
      (input (->input :size 1))
      (h (->lstm input :name 'h :size n-hiddens))
      (prediction (->softmax-xe-loss (->activation h :name 'prediction
                                                   :size *n-outputs*))))))

;;; We define this class to be able to specialize how inputs are
;;; translated by adding a SET-INPUT method later.
(defclass sum-sign-fnn (fnn)
  ())

;;; We have a batch of instances from MAKE-SUM-SIGN-INSTANCE for the
;;; RNN. This function is invoked with elements of these instances
;;; belonging to the same time step (i.e. at the same index) and sets
;;; the input and target up.
(defmethod set-input (instances (fnn sum-sign-fnn))
  (let ((input-nodes (nodes (find-clump 'input fnn))))
    (setf (target (find-clump 'prediction fnn))
          (loop for stripe upfrom 0
                for instance in instances
                collect
                ;; Sequences in the batch are not of equal length. The
                ;; RNN sends a NIL our way if a sequence has run out.
                (when instance
                  (destructuring-bind (input target) instance
                    (setf (mref input-nodes stripe 0) input)
                    target))))))

;;; Train the network by minimizing the loss (cross-entropy here) with
;;; the Adam optimizer.
(defun train-sum-sign-rnn ()
  (let ((rnn (make-sum-sign-rnn)))
    (setf (max-n-stripes rnn) 50)
    ;; Initialize the weights in the usual sqrt(1 / fan-in) style.
    (map-segments (lambda (weights)
                    (let* ((fan-in (mat-dimension (nodes weights) 0))
                           (limit (sqrt (/ 6 fan-in))))
                      (uniform-random! (nodes weights)
                                       :limit (* 2 limit))
                      (.+! (- limit) (nodes weights))))
                  rnn)
    (minimize (monitor-optimization-periodically
               (make-instance 'adam-optimizer
                              :learning-rate 0.2
                              :mean-decay 0.9
                              :mean-decay-decay 0.9
                              :variance-decay 0.9
                              :batch-size 100)
               '((:fn log-test-error :period 30000)
                 (:fn reset-optimization-monitors :period 3000)))
              (make-instance 'bp-learner
                             :bpn rnn
                             :monitors (make-cost-monitors rnn))
              :dataset (make-sampler 30000))))

;;; Return a sampler object that produces MAX-N-SAMPLES number of
;;; random inputs.
(defun make-sampler (max-n-samples &key (length 10))
  (make-instance 'function-sampler :max-n-samples max-n-samples
                 :generator (lambda ()
                              (make-sum-sign-instance :length length))))

;;; Log the test error. Also, describe the optimizer and the bpn at
;;; the beginning of training. Called periodically during training
;;; (see above).
(defun log-test-error (optimizer learner)
  (when (zerop (n-instances optimizer))
    (describe optimizer)
    (describe (bpn learner)))
  (let ((rnn (bpn learner)))
    (log-padded
     (append
      (monitor-bpn-results (make-sampler 1000) rnn
                           (make-cost-monitors
                            rnn :attributes '(:event "pred.")))
      ;; Same result in a different way: monitor predictions for
      ;; sequences up to length 20, but don't unfold the RNN
      ;; unnecessarily to save memory.
      (let ((*warp-time* t))
        (monitor-bpn-results (make-sampler 1000 :length 20) rnn
                             ;; Just collect and reset the warp
                             ;; monitors after each batch of
                             ;; instances.
                             (make-cost-monitors
                              rnn :attributes '(:event "warped pred."))))))
    ;; Verify that no further unfoldings took place.
    (assert (<= (length (clumps rnn)) 10)))
  (log-mat-room))

#|

;;; Transcript follows:
(let (;; Backprop nets do not need double float. Using single floats
      ;; is faster and needs less memory.
      (*default-mat-ctype* :float)
      ;; Enable moving data in and out of GPU memory so that the RNN
      ;; can work with sequences so long that the unfolded network
      ;; wouldn't otherwise fit in the GPU.
      (*cuda-window-start-time* 1)
      (*log-time* nil))
  ;; Seed the random number generators.
  (repeatably ()
    ;; Enable CUDA if available.
    (with-cuda* ()
      (train-sum-sign-rnn))))
.. training at n-instances: 0
.. cost: 0.000e+0 (0)
.. #<ADAM-OPTIMIZER {1006CD5663}>
..  GD-OPTIMIZER description:
..    N-INSTANCES = 0
..    SEGMENT-SET = #<SEGMENT-SET
..                    (#<->WEIGHT (H #) :SIZE 1 1/1 :NORM 1.73685>
..                     #<->WEIGHT (H #) :SIZE 1 1/1 :NORM 0.31893>
..                     #<->WEIGHT (#1=# #2=# :PEEPHOLE) :SIZE
..                       1 1/1 :NORM 1.81610>
..                     #<->WEIGHT (H #2#) :SIZE 1 1/1 :NORM 0.21965>
..                     #<->WEIGHT (#1# #3=# :PEEPHOLE) :SIZE
..                       1 1/1 :NORM 1.74939>
..                     #<->WEIGHT (H #3#) :SIZE 1 1/1 :NORM 0.40377>
..                     #<->WEIGHT (H PREDICTION) :SIZE
..                       3 1/1 :NORM 2.15898>
..                     #<->WEIGHT (:BIAS PREDICTION) :SIZE
..                       3 1/1 :NORM 2.94470>
..                     #<->WEIGHT (#1# #4=# :PEEPHOLE) :SIZE
..                       1 1/1 :NORM 0.97601>
..                     #<->WEIGHT (INPUT #4#) :SIZE 1 1/1 :NORM 0.65261>
..                     #<->WEIGHT (:BIAS #4#) :SIZE 1 1/1 :NORM 0.37653>
..                     #<->WEIGHT (INPUT #1#) :SIZE 1 1/1 :NORM 0.92334>
..                     #<->WEIGHT (:BIAS #1#) :SIZE 1 1/1 :NORM 0.01609>
..                     #<->WEIGHT (INPUT #5=#) :SIZE 1 1/1 :NORM 1.09995>
..                     #<->WEIGHT (:BIAS #5#) :SIZE 1 1/1 :NORM 1.41244>
..                     #<->WEIGHT (INPUT #6=#) :SIZE 1 1/1 :NORM 0.40475>
..                     #<->WEIGHT (:BIAS #6#) :SIZE 1 1/1 :NORM 1.75358>)
..                    {1006CD8753}>
..    LEARNING-RATE = 2.00000e-1
..    MOMENTUM = NONE
..    MOMENTUM-TYPE = :NONE
..    WEIGHT-DECAY = 0.00000e+0
..    WEIGHT-PENALTY = 0.00000e+0
..    N-AFTER-UPATE-HOOK = 0
..    BATCH-SIZE = 100
..  
..  BATCH-GD-OPTIMIZER description:
..    N-BEFORE-UPATE-HOOK = 0
..  
..  ADAM-OPTIMIZER description:
..    MEAN-DECAY-RATE = 1.00000e-1
..    MEAN-DECAY-RATE-DECAY = 9.00000e-1
..    VARIANCE-DECAY-RATE = 1.00000e-1
..    VARIANCE-ADJUSTMENT = 1.00000d-7
..  #<RNN {10047C77E3}>
..   BPN description:
..     CLUMPS = #(#<SUM-SIGN-FNN :STRIPES 1/50 :CLUMPS 4>
..                #<SUM-SIGN-FNN :STRIPES 1/50 :CLUMPS 4>)
..     N-STRIPES = 1
..     MAX-N-STRIPES = 50
..   
..   RNN description:
..     MAX-LAG = 1
..   pred.        cost: 1.223e+0 (4455.00)
.. warped pred. cost: 1.228e+0 (9476.00)
.. Foreign memory usage:
.. foreign arrays: 162 (used bytes: 39,600)
.. CUDA memory usage:
.. device arrays: 114 (used bytes: 220,892, pooled bytes: 19,200)
.. host arrays: 162 (used bytes: 39,600)
.. host->device copies: 6,164, device->host copies: 4,490
.. training at n-instances: 3000
.. cost: 3.323e-1 (13726.00)
.. training at n-instances: 6000
.. cost: 3.735e-2 (13890.00)
.. training at n-instances: 9000
.. cost: 1.012e-2 (13872.00)
.. training at n-instances: 12000
.. cost: 3.026e-3 (13953.00)
.. training at n-instances: 15000
.. cost: 9.267e-4 (13948.00)
.. training at n-instances: 18000
.. cost: 2.865e-4 (13849.00)
.. training at n-instances: 21000
.. cost: 8.893e-5 (13758.00)
.. training at n-instances: 24000
.. cost: 2.770e-5 (13908.00)
.. training at n-instances: 27000
.. cost: 8.514e-6 (13570.00)
.. training at n-instances: 30000
.. cost: 2.705e-6 (13721.00)
.. pred.        cost: 1.426e-6 (4593.00)
.. warped pred. cost: 1.406e-6 (9717.00)
.. Foreign memory usage:
.. foreign arrays: 216 (used bytes: 52,800)
.. CUDA memory usage:
.. device arrays: 148 (used bytes: 224,428, pooled bytes: 19,200)
.. host arrays: 216 (used bytes: 52,800)
.. host->device copies: 465,818, device->host copies: 371,990
..
==> (#<->WEIGHT (H (H :OUTPUT)) :SIZE 1 1/1 :NORM 0.10624>
-->  #<->WEIGHT (H (H :CELL)) :SIZE 1 1/1 :NORM 0.94460>
-->  #<->WEIGHT ((H :CELL) (H :FORGET) :PEEPHOLE) :SIZE 1 1/1 :NORM 0.61312>
-->  #<->WEIGHT (H (H :FORGET)) :SIZE 1 1/1 :NORM 0.38093>
-->  #<->WEIGHT ((H :CELL) (H :INPUT) :PEEPHOLE) :SIZE 1 1/1 :NORM 1.17956>
-->  #<->WEIGHT (H (H :INPUT)) :SIZE 1 1/1 :NORM 0.88011>
-->  #<->WEIGHT (H PREDICTION) :SIZE 3 1/1 :NORM 49.93808>
-->  #<->WEIGHT (:BIAS PREDICTION) :SIZE 3 1/1 :NORM 10.98112>
-->  #<->WEIGHT ((H :CELL) (H :OUTPUT) :PEEPHOLE) :SIZE 1 1/1 :NORM 0.67996>
-->  #<->WEIGHT (INPUT (H :OUTPUT)) :SIZE 1 1/1 :NORM 0.65251>
-->  #<->WEIGHT (:BIAS (H :OUTPUT)) :SIZE 1 1/1 :NORM 10.23003>
-->  #<->WEIGHT (INPUT (H :CELL)) :SIZE 1 1/1 :NORM 5.98116>
-->  #<->WEIGHT (:BIAS (H :CELL)) :SIZE 1 1/1 :NORM 0.10681>
-->  #<->WEIGHT (INPUT (H :FORGET)) :SIZE 1 1/1 :NORM 4.46301>
-->  #<->WEIGHT (:BIAS (H :FORGET)) :SIZE 1 1/1 :NORM 1.57195>
-->  #<->WEIGHT (INPUT (H :INPUT)) :SIZE 1 1/1 :NORM 0.36401>
-->  #<->WEIGHT (:BIAS (H :INPUT)) :SIZE 1 1/1 :NORM 8.63833>)

|#
```

<a id="x-28MGL-BP-3ARNN-20CLASS-29"></a>
- [class] **RNN** *[BPN][5187]*

    A recurrent neural net (as opposed to a
    feed-forward one. It is typically built with [`BUILD-RNN`][764b] that's no
    more than a shallow convenience macro.
    
    An `RNN` takes instances as inputs that are sequences of variable
    length. At each time step, the next unprocessed elements of these
    sequences are set as input until all input sequences in the batch
    run out. To be able to perform backpropagation, all intermediate
    [`LUMP`][c1ac]s must be kept around, so the recursive connections are
    transformed out by
    [unfolding](http://en.wikipedia.org/wiki/Backpropagation_through_time)
    the network. Just how many lumps this means depends on the length of
    the sequences.
    
    When an `RNN` is created, `MAX-LAG + 1` [`BPN`][5187]s are instantiated so
    that all weights are present and one can start training it.

<a id="x-28MGL-BP-3AUNFOLDER-20-28MGL-PAX-3AREADER-20MGL-BP-3ARNN-29-29"></a>
- [reader] **UNFOLDER** *RNN (:UNFOLDER)*

    The `UNFOLDER` of an [`RNN`][b0f3] is function of no arguments
    that builds and returns a [`BPN`][5187]. The unfolder is allowed to create
    networks with arbitrary topology even different ones for different
    [`TIME-STEP`][6e96]s with the help of [`LAG`][ff5a], or nested `RNN`s. Weights of
    the same name are shared between the folds. That is, if a [`->WEIGHT`][b76f]
    lump were to be created and a weight lump of the same name already
    exists, then the existing lump will be added to the `BPN` created by
    `UNFOLDER`.

<a id="x-28MGL-BP-3AMAX-LAG-20-28MGL-PAX-3AREADER-20MGL-BP-3ARNN-29-29"></a>
- [reader] **MAX-LAG** *RNN (:MAX-LAG = 1)*

    The networks built by [`UNFOLDER`][8e53] may contain new
    weights up to time step `MAX-LAG`. Beyond that point, all weight
    lumps must be reappearances of weight lumps with the same name at
    previous time steps. Most recurrent networks reference only the
    state of lumps at the previous time step (with the function [`LAG`][ff5a]),
    hence the default of 1. But it is possible to have connections to
    arbitrary time steps. The maximum connection lag must be specified
    when creating the [`RNN`][b0f3].

<a id="x-28MGL-BP-3ACUDA-WINDOW-START-TIME-20-28MGL-PAX-3AACCESSOR-20MGL-BP-3ARNN-29-29"></a>
- [accessor] **CUDA-WINDOW-START-TIME** *RNN (:CUDA-WINDOW-START-TIME = \*CUDA-WINDOW-START-TIME\*)*

    Due to unfolding, the memory footprint of an [`RNN`][b0f3]
    is almost linear in the number of time steps (i.e. the max
    sequence length). For prediction, this is addressed by
    [Time Warp][d0e3]. For training, we cannot discard results of
    previous time steps because they are needed for backpropagation,
    but we can at least move them out of GPU memory if they are not
    going to be used for a while and copy them back before they are
    needed. Obviously, this is only relevant if CUDA is being used.
    
    If `CUDA-WINDOW-START-TIME` is `NIL`, then this feature is turned off.
    Else, during training, at `CUDA-WINDOW-START-TIME` or later time
    steps, matrices belonging to non-weight lumps may be forced out of
    GPU memory and later brought back as neeeded.
    
    This feature is implemented in terms of
    `MGL-MAT:WITH-SYNCING-CUDA-FACETS` that uses CUDA host memory (also
    known as *page-locked* or *pinned memory*) to do asynchronous
    copies concurrently with normal computation. The consequence of
    this is that it is now main memory usage that's unbounded which
    toghether with page-locking makes it a potent weapon to bring a
    machine to a halt. You were warned.

<a id="x-28MGL-BP-3A-2ACUDA-WINDOW-START-TIME-2A-20VARIABLE-29"></a>
- [variable] **\*CUDA-WINDOW-START-TIME\*** *NIL*

    The default for [`CUDA-WINDOW-START-TIME`][f573].

<a id="x-28MGL-BP-3ABUILD-RNN-20MGL-PAX-3AMACRO-29"></a>
- [macro] **BUILD-RNN** *(&KEY RNN (CLASS ''RNN) NAME INITARGS MAX-N-STRIPES (MAX-LAG 1)) &BODY BODY*

    Create an `RNN` with `MAX-N-STRIPES` and `MAX-LAG` whose [`UNFOLDER`][8e53] is `BODY`
    wrapped in a lambda. Bind symbol given as the `RNN` argument to the
    `RNN` object so that `BODY` can see it.

<a id="x-28MGL-BP-3ALAG-20FUNCTION-29"></a>
- [function] **LAG** *NAME &KEY (LAG 1) RNN PATH*

    In `RNN` or if it's `NIL` the `RNN` being extended with another
    [`BPN`][5187] (called *unfolding*), look up the [`CLUMP`][a4fe] with `NAME` in the `BPN`
    that's `LAG` number of time steps before the `BPN` being added. If this
    function is called from [`UNFOLDER`][8e53] of an `RNN` (which is what happens
    behind the scene in the body of [`BUILD-RNN`][764b]), then it returns an
    opaque object representing a lagged connection to a clump, else it
    returns the `CLUMP` itself.
    
    FIXDOC: `PATH`

<a id="x-28MGL-BP-3ATIME-STEP-20FUNCTION-29"></a>
- [function] **TIME-STEP** *&KEY (RNN \*RNN\*)*

    Return the time step `RNN` is currently executing or being unfolded for.
    It is 0 when the `RNN` is being unfolded for the first time.

<a id="x-28MGL-CORE-3ASET-INPUT-20-28METHOD-20NIL-20-28T-20MGL-BP-3ARNN-29-29-29"></a>
- [method] **SET-INPUT** *INSTANCES (RNN RNN)*

    `RNN`s operate on batches of instances just like [`FNN`][9de4]s. But the
    instances here are like datasets: sequences or samplers and they are
    turned into sequences of batches of instances with
    `MAP-DATASETS`([`0`][765c] [`1`][17d3]) `:IMPUTE` `NIL`. The batch of instances at index 2 is
    clamped onto the [`BPN`][5187] at time step 2 with `SET-INPUT`.
    
    When the input sequences in the batch are not of the same length,
    already exhausted sequences will produce `NIL` (due to `:IMPUTE` `NIL`)
    above. When such a `NIL` is clamped with `SET-INPUT` on a `BPN` of the
    `RNN`, `SET-INPUT` must set the [`IMPORTANCE`][038e] of the ->ERROR lumps to 0
    else training would operate on the noise left there by previous
    invocations.

<a id="x-28MGL-BP-3A-40MGL-RNN-TIME-WARP-20MGL-PAX-3ASECTION-29"></a>
##### Time Warp

The unbounded memory usage of [`RNN`][b0f3]s with one [`BPN`][5187] allocated per
time step can become a problem. For training, where the gradients
often have to be backpropagated from the last time step to the very
beginning, this is hard to solve but with [`CUDA-WINDOW-START-TIME`][f573] the
limit is no longer GPU memory.

For prediction on the other hand, one doesn't need to keep old steps
around indefinitely: they can be discarded when future time steps
will never reference them again.

<a id="x-28MGL-BP-3A-2AWARP-TIME-2A-20VARIABLE-29"></a>
- [variable] **\*WARP-TIME\*** *NIL*

    Controls whether warping is enabled (see [Time Warp][d0e3]). Don't
    enable it for training, as it would make backprop impossible.

<a id="x-28MGL-BP-3AWARPED-TIME-20FUNCTION-29"></a>
- [function] **WARPED-TIME** *&KEY (RNN \*RNN\*) (TIME (TIME-STEP :RNN RNN)) (LAG 0)*

    Return the index of the [`BPN`][5187] in `CLUMPS`([`0`][f7c1] [`1`][a4fe]) of `RNN` whose task it is to
    execute computation at `(- (TIME-STEP RNN) LAG)`. This is normally
    the same as [`TIME-STEP`][6e96] (disregarding `LAG`). That is, `CLUMPS` can be
    indexed by `TIME-STEP` to get the `BPN`. However, when [`*WARP-TIME*`][ed4f] is
    true, execution proceeds in a cycle as the structure of the network
    allows.
    
    Suppose we have a typical `RNN` that only ever references the previous
    time step so its [`MAX-LAG`][084d] is 1. Its [`UNFOLDER`][8e53] returns `BPN`s of
    identical structure bar a shift in their time lagged connections
    except for the very first, so [`WARP-START`][d6e0] and [`WARP-LENGTH`][51d5] are both 1.
    If `*WARP-TIME*` is `NIL`, then the mapping from `TIME-STEP` to the `BPN` in
    `CLUMPS` is straightforward:
    
        time:   |  0 |  1 |  2 |  3 |  4 |  5
        --------+----+----+----+----+----+----
        warped: |  0 |  1 |  2 |  3 |  4 |  5
        --------+----+----+----+----+----+----
        bpn:    | b0 | b1 | b2 | b3 | b4 | b5
    
    When `*WARP-TIME*` is true, we reuse the `B1` - `B2` bpns in a loop:
    
        time:   |  0 |  1 |  2 |  3 |  4 |  5
        --------+----+----+----+----+----+----
        warped: |  0 |  1 |  2 |  1 |  2 |  1
        --------+----+----+----+----+----+----
        bpn:    | b0 | b1 | b2 | b1*| b2 | b1*
    
    `B1*` is the same `BPN` as `B1`, but its connections created by `LAG` go
    through warped time and end up referencing `B2`. This way, memory
    consumption is independent of the number time steps needed to
    process a sequence or make predictions.
    
    To be able to pull this trick off `WARP-START` and `WARP-LENGTH` must be
    specified when the `RNN` is instantiated. In general, with
    `*WARP-TIME*` `(+ WARP-START (MAX 2 WARP-LENGTH))` bpns are needed.
    The 2 comes from the fact that with cycle length 1 a bpn would need
    to takes its input from itself which is problematic because it has
    [`NODES`][cc1c] for only one set of values.

<a id="x-28MGL-BP-3AWARP-START-20-28MGL-PAX-3AREADER-20MGL-BP-3ARNN-29-29"></a>
- [reader] **WARP-START** *RNN (:WARP-START = 1)*

    The [`TIME-STEP`][6e96] from which [`UNFOLDER`][8e53] will create
    [`BPN`][5187]s that essentially repeat every [`WARP-LENGTH`][51d5] steps.

<a id="x-28MGL-BP-3AWARP-LENGTH-20-28MGL-PAX-3AREADER-20MGL-BP-3ARNN-29-29"></a>
- [reader] **WARP-LENGTH** *RNN (:WARP-LENGTH = 1)*

    An integer such that the [`BPN`][5187] [`UNFOLDER`][8e53] creates at
    time step `I` (where `(<= WARP-START I)`) is identical to the `BPN`
    created at time step `(+ WARP-START (MOD (- I WARP-START)
    WARP-LENGTH))` except for a shift in its time lagged
    connections.

<a id="x-28MGL-BP-3ASTEP-MONITORS-20-28MGL-PAX-3AACCESSOR-20MGL-BP-3ARNN-29-29"></a>
- [accessor] **STEP-MONITORS** *RNN (:STEP-MONITORS = NIL)*

    During training, unfolded [`BPN`][5187]s corresponding to
    previous time steps may be expensive to get at because they are no
    longer in GPU memory. This consideration also applies to making
    prediction with the additional caveat that with [`*WARP-TIME*`][ed4f] true,
    previous states are discarded so it's not possible to gather
    statistics after [`FORWARD`][c1ae] finished.
    
    Add monitor objects to this slot and they will be automatically
    applied to the [`RNN`][b0f3] after each step when `FORWARD`ing the `RNN`
    during training or prediction. To be able to easily switch between
    sets of monitors, in addition to a list of monitors this can be a
    symbol or a function, too. If it's a symbol, then its a designator
    for its [`SYMBOL-VALUE`][920f]. If it's a function, then it must have no
    arguments and it's a designator for its return value.

<a id="x-28MGL-BP-3A-40MGL-BP-LUMPS-20MGL-PAX-3ASECTION-29"></a>
### 11.4 Lumps

<a id="x-28MGL-BP-3A-40MGL-BP-LUMP-20MGL-PAX-3ASECTION-29"></a>
#### 11.4.1 Lump Base Class

<a id="x-28MGL-BP-3ALUMP-20CLASS-29"></a>
- [class] **LUMP** *[CLUMP][a4fe]*

    A `LUMP` is a simple, layerlike component of a neural
    network. There are many kinds of lumps, each of which performs a
    specific operation or just stores inputs and weights. By convention,
    the names of lumps start with the prefix `->`. Defined as classes,
    they also have a function of the same name as the class to create
    them easily. These maker functions typically have keyword arguments
    corresponding to initargs of the class, with some (mainly the input
    lumps) turned into normal positional arguments. So instead of having
    to do
    
        (make-instance '->tanh :x some-input :name 'my-tanh)
    
    one can simply write
    
        (->tanh some-input :name 'my-tanh)
    
    Lumps instantiated in any way within a [`BUILD-FNN`][606c] or [`BUILD-RNN`][764b] are
    automatically added to the network being built.
    
    A lump has its own [`NODES`][cc1c] and [`DERIVATIVES`][a81b] matrices allocated for it
    in which the results of the forward and backward passes are stored.
    This is in contrast to a [`BPN`][5187] whose `NODES` and `DERIVATIVES`
    are those of its last constituent [`CLUMP`][a4fe].
    
    Since lumps almost always live within a `BPN`, their
    [`N-STRIPES`][07fb] and [`MAX-N-STRIPES`][91a3] are
    handled automagically behind the scenes.

<a id="x-28MGL-COMMON-3ASIZE-20-28MGL-PAX-3AREADER-20MGL-BP-3ALUMP-29-29"></a>
- [reader] **SIZE** *LUMP (:SIZE)*

    The number of values in a single stripe.

<a id="x-28MGL-COMMON-3ADEFAULT-VALUE-20-28MGL-PAX-3AREADER-20MGL-BP-3ALUMP-29-29"></a>
- [reader] **DEFAULT-VALUE** *LUMP (:DEFAULT-VALUE = 0)*

    Upon creation or resize the lump's nodes get
    filled with this value.

<a id="x-28MGL-BP-3ADEFAULT-SIZE-20GENERIC-FUNCTION-29"></a>
- [generic-function] **DEFAULT-SIZE** *LUMP*

    Return a default for the [`SIZE`][85d3] of
    `LUMP` if one is not supplied at instantiation. The value is often
    computed based on the sizes of the inputs. This function is for
    implementing new lump types.

<a id="x-28MGL-COMMON-3ANODES-20-28MGL-PAX-3AREADER-20MGL-BP-3ALUMP-29-29"></a>
- [reader] **NODES** *LUMP (= NIL)*

    The values computed by the lump in the forward
    pass are stored here. It is an `N-STRIPES * SIZE` matrix that has
    storage allocated for `MAX-N-STRIPES * SIZE` elements for
    non-weight lumps. [`->WEIGHT`][b76f] lumps have no stripes nor restrictions
    on their shape.

<a id="x-28MGL-BP-3ADERIVATIVES-20-28MGL-PAX-3AREADER-20MGL-BP-3ALUMP-29-29"></a>
- [reader] **DERIVATIVES** *LUMP*

    The derivatives computed in the backward pass are
    stored here. This matrix is very much like [`NODES`][d699]
    in shape and size.

<a id="x-28MGL-BP-3A-40MGL-BP-INPUTS-20MGL-PAX-3ASECTION-29"></a>
#### 11.4.2 Inputs

<a id="x-28MGL-BP-3A-40MGL-BP-INPUT-LUMP-20MGL-PAX-3ASECTION-29"></a>
##### Input Lump

<a id="x-28MGL-BP-3A--3EINPUT-20CLASS-29"></a>
- [class] **-\>INPUT** *[-\>DROPOUT][441b]*

    A lump that has no input lumps, does not change its
    values in the forward pass (except when [`DROPOUT`][e7f6] is non-zero), and does not compute derivatives. *Clamp*
    inputs on [`NODES`][cc1c] of input lumps in [`SET-INPUT`][0c9e].
    
    For convenience, `->INPUT` can perform dropout itself although it
    defaults to no dropout.
    
    ```common-lisp
    (->input :size 10 :name 'some-input)
    ==> #<->INPUT SOME-INPUT :SIZE 10 1/1 :NORM 0.00000>
    ```


<a id="x-28MGL-BP-3ADROPOUT-20-28MGL-PAX-3AACCESSOR-20MGL-BP-3A--3EINPUT-29-29"></a>
- [accessor] **DROPOUT** *-\>INPUT (= NIL)*

    See [`DROPOUT`][2481].

<a id="x-28MGL-BP-3A-40MGL-BP-EMBEDDING-LUMP-20MGL-PAX-3ASECTION-29"></a>
##### Embedding Lump

This lump is like an input and a simple activation molded together
in the name of efficiency.

<a id="x-28MGL-BP-3A--3EEMBEDDING-20CLASS-29"></a>
- [class] **-\>EMBEDDING** *[LUMP][c1ac]*

    Select rows of `WEIGHTS`([`0`][3c96] [`1`][a896]), one row for each index in
    [`INPUT-ROW-INDICES`][5e58]. This lump is equivalent to adding an [`->INPUT`][f54e] lump
    with a one hot encoding scheme and a [`->V*M`][dbc4] lump on top of it, but it
    is more efficient in execution and in memory usage, because it works
    with a sparse representation of the input.
    
    The `SIZE`([`0`][85d3] [`1`][b99d]) of this lump is the number of columns of `WEIGHTS` which is
    determined automatically.
    
    ```common-lisp
    (->embedding :weights (->weight :name 'embedding-weights
                                    :dimensions '(3 5))
                 :name 'embeddings)
    ==> #<->EMBEDDING EMBEDDINGS :SIZE 5 1/1 :NORM 0.00000>
    ```


<a id="x-28MGL-COMMON-3AWEIGHTS-20-28MGL-PAX-3AREADER-20MGL-BP-3A--3EEMBEDDING-29-29"></a>
- [reader] **WEIGHTS** *-\>EMBEDDING (:WEIGHTS)*

    A weight lump whose rows indexed by
    [`INPUT-ROW-INDICES`][5e58] are copied to the output of this lump.

<a id="x-28MGL-BP-3AINPUT-ROW-INDICES-20-28MGL-PAX-3AREADER-20MGL-BP-3A--3EEMBEDDING-29-29"></a>
- [reader] **INPUT-ROW-INDICES** *-\>EMBEDDING (:INPUT-ROW-INDICES)*

    A sequence of batch size length of row indices. To
    be set in [`SET-INPUT`][0c9e].

<a id="x-28MGL-BP-3A-40MGL-BP-WEIGHT-LUMP-20MGL-PAX-3ASECTION-29"></a>
#### 11.4.3 Weight Lump

<a id="x-28MGL-BP-3A--3EWEIGHT-20CLASS-29"></a>
- [class] **-\>WEIGHT** *[LUMP][c1ac]*

    A set of optimizable parameters of some kind. When
    a [`BPN`][5187] is is trained (see [Training][0d82]) the [`NODES`][cc1c] of weight lumps
    will be changed. Weight lumps perform no computation.
    
    Weights can be created by specifying the total size or the
    dimensions:
    
    ```common-lisp
    (dimensions (->weight :size 10 :name 'w))
    => (1 10)
    (dimensions (->weight :dimensions '(5 10) :name 'w))
    => (5 10)
    ```


<a id="x-28MGL-BP-3ADIMENSIONS-20-28MGL-PAX-3AREADER-20MGL-BP-3A--3EWEIGHT-29-29"></a>
- [reader] **DIMENSIONS** *-\>WEIGHT (:DIMENSIONS)*

    [`NODES`][cc1c] and [`DERIVATIVES`][a81b] of this lump will be
    allocated with these dimensions.

<a id="x-28MGL-BP-3AWITH-WEIGHTS-COPIED-20MGL-PAX-3AMACRO-29"></a>
- [macro] **WITH-WEIGHTS-COPIED** *(FROM-BPN) &BODY BODY*

    In `BODY` [`->WEIGHT`][b76f] will first look up if a weight lump of the same
    name exists in `FROM-BPN` and return that, or else create a weight
    lump normally. If `FROM-BPN` is `NIL`, then no weights are copied.

<a id="x-28MGL-BP-3A-40MGL-BP-ACTIVATIONS-20MGL-PAX-3ASECTION-29"></a>
#### 11.4.4 Activations

<a id="x-28MGL-BP-3A-40MGL-BP-ACTIVATION-SUBNET-20MGL-PAX-3ASECTION-29"></a>
##### Activation Subnet

So we have some inputs. Usually the next step is to multiply the
input vector with a weight matrix and add biases. This can be done
directly with ->+, [`->V*M`][dbc4] and [`->WEIGHT`][b76f], but it's more convenient to
use activation subnets to reduce the clutter.

<a id="x-28MGL-BP-3A--3EACTIVATION-20CLASS-29"></a>
- [class] **-\>ACTIVATION** *[BPN][5187]*

    Activation subnetworks are built by the function
    [`->ACTIVATION`][b602] and they have a number of lumps hidden inside them.
    Ultimately, this subnetwork computes a sum like `sum_i x_i * W_i +
    sum_j y_j .* V_j + biases` where `x_i` are input lumps, `W_i` are
    dense matrices representing connections, while `V_j` are peephole
    connection vectors that are mulitplied in an elementwise manner with
    their corresponding input `y_j`.

<a id="x-28MGL-BP-3A--3EACTIVATION-20FUNCTION-29"></a>
- [function] **-\>ACTIVATION** *INPUTS &KEY (NAME (GENSYM)) SIZE PEEPHOLES (ADD-BIAS-P T)*

    Create a subnetwork of class [`->ACTIVATION`][7162] that computes the over
    activation from dense connection from lumps in `INPUTS`, and
    elementwise connection from lumps in `PEEPHOLES`. Create new [`->WEIGHT`][b76f]
    lumps as necessary. `INPUTS` and `PEEPHOLES` can be a single lump or a
    list of lumps. Finally, if `ADD-BIAS-P`, then add an elementwise bias
    too. `SIZE` must be specified explicitly, because it is not possible
    to determine it unless there are peephole connections.
    
    ```common-lisp
    (->activation (->input :size 10 :name 'input) :name 'h1 :size 4)
    ==> #<->ACTIVATION (H1 :ACTIVATION) :STRIPES 1/1 :CLUMPS 4>
    ```
    
    This is the basic workhorse of neural networks which takes care of
    the linear transformation whose results and then fed to some
    non-linearity ([`->SIGMOID`][83f9], [`->TANH`][5309], etc).
    
    The name of the subnetwork clump is `(,NAME :ACTIVATION)`. The bias
    weight lump (if any) is named `(:BIAS ,NAME)`. Dense connection
    weight lumps are named are named after the input and `NAME`: `(,(NAME
    INPUT) ,NAME)`, while peepholes weight lumps are named `(,(NAME
    INPUT) ,NAME :PEEPHOLE)`. This is useful to know if, for example,
    they are to be initialized differently.

<a id="x-28MGL-BP-3A-40MGL-BP-BATCH-NORMALIZATION-20MGL-PAX-3ASECTION-29"></a>
##### Batch-Normalization

<a id="x-28MGL-BP-3A--3EBATCH-NORMALIZED-20CLASS-29"></a>
- [class] **-\>BATCH-NORMALIZED** *[LUMP][c1ac]*

    This is an implementation of v3 of the [Batch
    Normalization paper](http://arxiv.org/abs/1502.03167). The output of
    `->BATCH-NORMALIZED` is its input normalized so that for all elements
    the mean across stripes is zero and the variance is 1. That is, the
    mean of the batch is subtracted from the inputs and they are
    rescaled by their sample stddev. Actually, after the normalization
    step the values are rescaled and shifted (but this time with learnt
    parameters) in order to keep the representational power of the model
    the same. The primary purpose of this lump is to speed up learning,
    but it also acts as a regularizer. See the paper for the details.
    
    To normalize the output of `LUMP` without no additional
    regularizer effect:
    
    ```commonlisp
    (->batch-normalized lump :batch-size :use-population)
    ```
    
    The above uses an exponential moving average to estimate the mean
    and variance of batches and these estimations are used at both
    training and test time. In contrast to this, the published version
    uses the sample mean and variance of the current batch at training
    time which injects noise into the process. The noise is higher for
    lower batch sizes and has a regularizing effect. This is the default
    behavior (equivalent to `:BATCH-SIZE NIL`):
    
    ```commonlisp
    (->batch-normalized lump)
    ```
    
    For performance reasons one may wish to process a higher number of
    instances in a batch (in the sense of [`N-STRIPES`][8dd7]) and get the
    regularization effect associated with a lower batch size. This is
    possible by setting `:BATCH-SIZE` to a divisor of the the number of
    stripes. Say, the number of stripes is 128, but we want as much
    regularization as we would get with 32:
    
    ```commonlisp
    (->batch-normalized lump :batch-size 32)
    ```
    
    The primary input of `->BATCH-NORMALIZED` is often an `->ACTIVATION`([`0`][7162] [`1`][b602]) and
    its output is fed into an activation function (see
    [Activation Functions][5d86]).

<a id="x-28MGL-BP-3ABATCH-NORMALIZATION-20-28MGL-PAX-3AREADER-20MGL-BP-3A--3EBATCH-NORMALIZED-29-29"></a>
- [reader] **BATCH-NORMALIZATION** *-\>BATCH-NORMALIZED (:NORMALIZATION)*

    The [`->BATCH-NORMALIZATION`][c469] of this lump. May be
    shared between multiple [`->BATCH-NORMALIZED`][9da9] lumps.
    
    Batch normalization is special in that it has state apart from the
    computed results ([`NODES`][cc1c]) and its derivatives ([`DERIVATIVES`][a81b]). This
    state is the estimated mean and variance of its inputs and they
    are encapsulated by `->BATCH-NORMALIZATION`.
    
    If `NORMALIZATION` is not given at instantiation, then a new
    `->BATCH-NORMALIZATION` object will be created automatically,
    passing `:BATCH-SIZE`, `:VARIANCE-ADJUSTMENT`, and `:POPULATION-DECAY`
    arguments on to `->BATCH-NORMALIZATION`. See [`BATCH-SIZE`][c918], [`VARIANCE-ADJUSTMENT`][aa86] and [`POPULATION-DECAY`][46c4]. New scale and shift weight lumps will be
    created with names:
    
        `(,name :scale)
        `(,name :shift)
    
    where `NAME` is the `NAME`([`0`][0414] [`1`][f3bc]) of this lump.
    
    This default behavior covers the use-case where the statistics
    kept by `->BATCH-NORMALIZATION` are to be shared only between time
    steps of an [`RNN`][b0f3].

<a id="x-28MGL-BP-3A--3EBATCH-NORMALIZATION-20CLASS-29"></a>
- [class] **-\>BATCH-NORMALIZATION** *[-\>WEIGHT][b76f]*

    The primary purpose of this class is to hold the
    estimated mean and variance of the inputs to be normalized and allow
    them to be shared between multiple [`->BATCH-NORMALIZED`][9da9] lumps that
    carry out the computation. These estimations are saved and loaded by
    [`SAVE-STATE`][c102] and [`LOAD-STATE`][6bd7].
    
    ```commonlisp
    (->batch-normalization (->weight :name '(h1 :scale) :size 10)
                           (->weight :name '(h1 :shift) :size 10)
                           :name '(h1 :batch-normalization))
    ```


<a id="x-28MGL-COMMON-3ASCALE-20-28MGL-PAX-3AREADER-20MGL-BP-3A--3EBATCH-NORMALIZATION-29-29"></a>
- [reader] **SCALE** *-\>BATCH-NORMALIZATION (:SCALE)*

    A weight lump of the same size as [`SHIFT`][7960]. This is
    $\gamma$ in the paper.

<a id="x-28MGL-BP-3ASHIFT-20-28MGL-PAX-3AREADER-20MGL-BP-3A--3EBATCH-NORMALIZATION-29-29"></a>
- [reader] **SHIFT** *-\>BATCH-NORMALIZATION (:SHIFT)*

    A weight lump of the same size as [`SCALE`][c9d1]. This is
    $\beta$ in the paper.

<a id="x-28MGL-COMMON-3ABATCH-SIZE-20-28MGL-PAX-3AREADER-20MGL-BP-3A--3EBATCH-NORMALIZATION-29-29"></a>
- [reader] **BATCH-SIZE** *-\>BATCH-NORMALIZATION (:BATCH-SIZE = NIL)*

    Normally all stripes participate in the batch.
    Lowering the number of stripes may increase the regularization
    effect, but it also makes the computation less efficient. By
    setting `BATCH-SIZE` to a divisor of [`N-STRIPES`][8dd7] one can decouple the
    concern of efficiency from that of regularization. The default
    value, `NIL`, is equivalent to `N-STRIPES`. `BATCH-SIZE` only affects
    training.
    
    With the special value `:USE-POPULATION`, instead of the mean and
    the variance of the current batch, use the population statistics
    for normalization. This effectively cancels the regularization
    effect, leaving only the faster learning.

<a id="x-28MGL-GD-3AVARIANCE-ADJUSTMENT-20-28MGL-PAX-3AREADER-20MGL-BP-3A--3EBATCH-NORMALIZATION-29-29"></a>
- [reader] **VARIANCE-ADJUSTMENT** *-\>BATCH-NORMALIZATION (:VARIANCE-ADJUSTMENT = 1.0e-4)*

    A small positive real number that's added to the
    sample variance. This is $\epsilon$ in the paper.

<a id="x-28MGL-BP-3APOPULATION-DECAY-20-28MGL-PAX-3AREADER-20MGL-BP-3A--3EBATCH-NORMALIZATION-29-29"></a>
- [reader] **POPULATION-DECAY** *-\>BATCH-NORMALIZATION (:POPULATION-DECAY = 0.99)*

    While training, an exponential moving average of
    batch means and standard deviances (termed *population
    statistics*) is updated. When making predictions, normalization is
    performed using these statistics. These population statistics are
    persisted by [`SAVE-STATE`][c102].

<a id="x-28MGL-BP-3A--3EBATCH-NORMALIZED-ACTIVATION-20FUNCTION-29"></a>
- [function] **-\>BATCH-NORMALIZED-ACTIVATION** *INPUTS &KEY (NAME (GENSYM)) SIZE PEEPHOLES BATCH-SIZE VARIANCE-ADJUSTMENT POPULATION-DECAY*

    A utility functions that creates and wraps an `->ACTIVATION`([`0`][7162] [`1`][b602]) in
    [`->BATCH-NORMALIZED`][9da9] and with its [`BATCH-NORMALIZATION`][eaf1] the two weight
    lumps for the scale and shift
    parameters. `(->BATCH-NORMALIZED-ACTIVATION INPUTS :NAME 'H1 :SIZE
    10)` is equivalent to:
    
    ```commonlisp
    (->batch-normalized (->activation inputs :name 'h1 :size 10 :add-bias-p nil)
                        :name '(h1 :batch-normalized-activation))
    ```
    
    Note how biases are turned off since normalization will cancel them
    anyway (but a shift is added which amounts to the same effect).

<a id="x-28MGL-BP-3A-40MGL-BP-ACTIVATION-FUNCTIONS-20MGL-PAX-3ASECTION-29"></a>
#### 11.4.5 Activation Functions

Now we are moving on to the most important non-linearities to which
activations are fed.

<a id="x-28MGL-BP-3A-40MGL-BP-SIGMOID-LUMP-20MGL-PAX-3ASECTION-29"></a>
##### Sigmoid Lump

<a id="x-28MGL-BP-3A--3ESIGMOID-20CLASS-29"></a>
- [class] **-\>SIGMOID** *[-\>DROPOUT][441b] [LUMP][c1ac]*

    Applies the `1/(1 + e^{-x})` function elementwise
    to its inputs. This is one of the classic non-linearities for neural
    networks.
    
    For convenience, `->SIGMOID` can perform dropout itself although it
    defaults to no dropout.
    
    ```common-lisp
    (->sigmoid (->activation (->input :size 10) :size 5) :name 'this)
    ==> #<->SIGMOID THIS :SIZE 5 1/1 :NORM 0.00000>
    ```
    
    The `SIZE`([`0`][85d3] [`1`][b99d]) of this lump is the size of its input which is determined
    automatically.

<a id="x-28MGL-BP-3ADROPOUT-20-28MGL-PAX-3AACCESSOR-20MGL-BP-3A--3ESIGMOID-29-29"></a>
- [accessor] **DROPOUT** *-\>SIGMOID (= NIL)*

    See [`DROPOUT`][2481].

<a id="x-28MGL-BP-3A-40MGL-BP-TANH-LUMP-20MGL-PAX-3ASECTION-29"></a>
##### Tanh Lump

<a id="x-28MGL-BP-3A--3ETANH-20CLASS-29"></a>
- [class] **-\>TANH** *[LUMP][c1ac]*

    Applies the [`TANH`][7582] function to its input in an
    elementwise manner. The `SIZE`([`0`][85d3] [`1`][b99d]) of this lump is the size of its input
    which is determined automatically.

<a id="x-28MGL-BP-3A-40MGL-BP-SCALED-TANH-LUMP-20MGL-PAX-3ASECTION-29"></a>
##### Scaled Tanh Lump

<a id="x-28MGL-BP-3A--3ESCALED-TANH-20CLASS-29"></a>
- [class] **-\>SCALED-TANH** *[LUMP][c1ac]*

    Pretty much like [`TANH`][7582] but its input and output is
    scaled in such a way that the variance of its output is close to 1
    if the variance of its input is close to 1 which is a nice property
    to combat vanishing gradients. The actual function is `1.7159 *
    tanh(2/3 * x)`. The `SIZE`([`0`][85d3] [`1`][b99d]) of this lump is the size of its input which
    is determined automatically.

<a id="x-28MGL-BP-3A-40MGL-BP-RELU-LUMP-20MGL-PAX-3ASECTION-29"></a>
##### Relu Lump

We are somewhere around year 2007 by now.

<a id="x-28MGL-BP-3A--3ERELU-20CLASS-29"></a>
- [class] **-\>RELU** *[LUMP][c1ac]*

    `max(0,x)` activation function. Be careful, relu
    units can get stuck in the off state: if they move to far to
    negative territory it can be very difficult to get out of it. The
    `SIZE`([`0`][85d3] [`1`][b99d]) of this lump is the size of its input which is determined
    automatically.

<a id="x-28MGL-BP-3A-40MGL-BP-MAX-LUMP-20MGL-PAX-3ASECTION-29"></a>
##### Max Lump

We are in about year 2011.

<a id="x-28MGL-BP-3A--3EMAX-20CLASS-29"></a>
- [class] **-\>MAX** *[LUMP][c1ac]*

    This is basically maxout without dropout (see
    http://arxiv.org/abs/1302.4389). It groups its inputs by
    [`GROUP-SIZE`][59dd], and outputs the maximum of each group.
    The `SIZE`([`0`][85d3] [`1`][b99d]) of the output is automatically calculated, it is the size
    of the input divided by [`GROUP-SIZE`][59dd].
    
    ```common-lisp
    (->max (->input :size 120) :group-size 3 :name 'my-max)
    ==> #<->MAX MY-MAX :SIZE 40 1/1 :NORM 0.00000 :GROUP-SIZE 3>
    ```
    
    The advantage of `->MAX` over [`->RELU`][9d3a] is that flow gradient is never
    stopped so there is no problem of units getting stuck in off
    state.

<a id="x-28MGL-COMMON-3AGROUP-SIZE-20-28MGL-PAX-3AREADER-20MGL-BP-3A--3EMAX-29-29"></a>
- [reader] **GROUP-SIZE** *-\>MAX (:GROUP-SIZE)*

    The number of inputs in each group.

<a id="x-28MGL-BP-3A-40MGL-BP-MIN-LUMP-20MGL-PAX-3ASECTION-29"></a>
##### Min Lump

<a id="x-28MGL-BP-3A--3EMIN-20CLASS-29"></a>
- [class] **-\>MIN** *[LUMP][c1ac]*

    Same as [`->MAX`][f652], but it computes the [`MIN`][40e2] of groups.
    Rarely useful.

<a id="x-28MGL-COMMON-3AGROUP-SIZE-20-28MGL-PAX-3AREADER-20MGL-BP-3A--3EMIN-29-29"></a>
- [reader] **GROUP-SIZE** *-\>MIN (:GROUP-SIZE)*

    The number of inputs in each group.

<a id="x-28MGL-BP-3A-40MGL-BP-MAX-CHANNEL-LUMP-20MGL-PAX-3ASECTION-29"></a>
##### Max-Channel Lump

<a id="x-28MGL-BP-3A--3EMAX-CHANNEL-20CLASS-29"></a>
- [class] **-\>MAX-CHANNEL** *[LUMP][c1ac]*

    Called LWTA (Local Winner Take All) or
    Channel-Out (see http://arxiv.org/abs/1312.1909) in the literature
    it is basically [`->MAX`][f652], but instead of producing one output per
    group, it just produces zeros for all unit but the one with the
    maximum value in the group. This allows the next layer to get some
    information about the path along which information flowed. The `SIZE`([`0`][85d3] [`1`][b99d])
    of this lump is the size of its input which is determined
    automatically.

<a id="x-28MGL-COMMON-3AGROUP-SIZE-20-28MGL-PAX-3AREADER-20MGL-BP-3A--3EMAX-CHANNEL-29-29"></a>
- [reader] **GROUP-SIZE** *-\>MAX-CHANNEL (:GROUP-SIZE)*

    The number of inputs in each group.

<a id="x-28MGL-BP-3A-40MGL-BP-LOSSES-20MGL-PAX-3ASECTION-29"></a>
#### 11.4.6 Losses

Ultimately, we need to tell the network what to learn which means
that the loss function to be minimized needs to be constructed as
part of the network.

<a id="x-28MGL-BP-3A-40MGL-BP-LOSS-LUMP-20MGL-PAX-3ASECTION-29"></a>
##### Loss Lump

<a id="x-28MGL-BP-3A--3ELOSS-20CLASS-29"></a>
- [class] **-\>LOSS** *[-\>SUM][edcf]*

    Calculate the loss for the instances in the batch.
    The main purpose of this lump is to provide a training signal.
    
    An error lump is usually a leaf in the graph of lumps (i.e. there
    are no other lumps whose input is this one). The special thing about
    error lumps is that 1 (but see [`IMPORTANCE`][038e]) is added automatically to
    their derivatives. Error lumps have exactly one node (per stripe)
    whose value is computed as the sum of nodes in their input lump.

<a id="x-28MGL-BP-3AIMPORTANCE-20-28MGL-PAX-3AACCESSOR-20MGL-BP-3A--3ELOSS-29-29"></a>
- [accessor] **IMPORTANCE** *-\>LOSS (:IMPORTANCE = NIL)*

    This is to support weighted instances. That is
    when not all training instances are equally important. If non-NIL,
    a 1d `MAT` with the importances of stripes of the batch. When
    `IMPORTANCE` is given (typically in [`SET-INPUT`][0c9e]), then instead of
    adding 1 to the derivatives of all stripes, `IMPORTANCE` is added
    elemtwise.

<a id="x-28MGL-BP-3A-40MGL-BP-SQUARED-DIFFERENCE-LUMP-20MGL-PAX-3ASECTION-29"></a>
##### Squared Difference Lump

In regression, the squared error loss is most common. The squared
error loss can be constructed by combining [`->SQUARED-DIFFERENCE`][e8d2] with
a [`->LOSS`][2171].

<a id="x-28MGL-BP-3A--3ESQUARED-DIFFERENCE-20CLASS-29"></a>
- [class] **-\>SQUARED-DIFFERENCE** *[LUMP][c1ac]*

    This lump takes two input lumps and calculates
    their squared difference `(x - y)^2` in an elementwise manner. The
    `SIZE`([`0`][85d3] [`1`][b99d]) of this lump is automatically determined from the size of its
    inputs. This lump is often fed into [`->LOSS`][2171] that sums the squared
    differences and makes it part of the function to be minimized.
    
    ```common-lisp
    (->loss (->squared-difference (->activation (->input :size 100)
                                                :size 10)
                                  (->input :name 'target :size 10))
            :name 'squared-error)
    ==> #<->LOSS SQUARED-ERROR :SIZE 1 1/1 :NORM 0.00000>
    ```
    
    Currently this lump is not CUDAized, but it will copy data from the
    GPU if it needs to.

<a id="x-28MGL-BP-3A-40MGL-BP-SOFTMAX-XE-LOSS-LUMP-20MGL-PAX-3ASECTION-29"></a>
##### Softmax Cross-Entropy Loss Lump

<a id="x-28MGL-BP-3A--3ESOFTMAX-XE-LOSS-20CLASS-29"></a>
- [class] **-\>SOFTMAX-XE-LOSS** *[LUMP][c1ac]*

    A specialized lump that computes the softmax of its
    input in the forward pass and backpropagates a cross-entropy loss.
    The advantage of doing these together is numerical stability. The
    total cross-entropy is the sum of cross-entropies per group of
    [`GROUP-SIZE`][a437] elements:
    
    $$
    XE(x) = - \sum\_{i=1,g} t\_i \ln(s\_i),
    $$
    
    where `g` is the number of classes ([`GROUP-SIZE`][a437]), `t_i` are the targets (i.e. the true
    probabilities of the class, often all zero but one), `s_i` is the
    output of softmax calculated from input `X`:
    
    $$
    s\_i = {softmax}(x\_1, x\_2, ..., x\_g) =
      \frac{e^x\_i}{\sum\_{j=1,g} e^x\_j}
    $$
    
    In other words, in the forward phase this lump takes input `X`,
    computes its elementwise [`EXP`][45be], normalizes each group of
    [`GROUP-SIZE`][a437] elements to sum to 1 to get
    the softmax which is the result that goes into [`NODES`][cc1c]. In the
    backward phase, there are two sources of gradients: the lumps that
    use the output of this lump as their input (currently not
    implemented and would result in an error) and an implicit
    cross-entropy loss.
    
    One can get the cross-entropy calculated in the most recent forward
    pass by calling [`COST`][410c] on this lump.
    
    This is the most common loss function for classification. In fact,
    it is nearly ubiquitous. See the [`FNN` Tutorial][6b38] and the
    [`RNN` Tutorial][9700] for how this loss and [`SET-INPUT`][0c9e] work together.

<a id="x-28MGL-COMMON-3AGROUP-SIZE-20-28MGL-PAX-3AREADER-20MGL-BP-3A--3ESOFTMAX-XE-LOSS-29-29"></a>
- [reader] **GROUP-SIZE** *-\>SOFTMAX-XE-LOSS (:GROUP-SIZE)*

    The number of elements in a softmax group. This is
    the number of classes for classification. Often `GROUP-SIZE` is
    equal to `SIZE`([`0`][85d3] [`1`][b99d]) (it is the default), but in general the only
    constraint is that `SIZE` is a multiple of `GROUP-SIZE`.

<a id="x-28MGL-COMMON-3ATARGET-20-28MGL-PAX-3AACCESSOR-20MGL-BP-3A--3ESOFTMAX-XE-LOSS-29-29"></a>
- [accessor] **TARGET** *-\>SOFTMAX-XE-LOSS (:TARGET = NIL)*

    Set in [`SET-INPUT`][0c9e], this is either a `MAT` of the same
    size as the input lump `X` or if the target is very sparse, this
    can also be a sequence of batch size length that contains the
    index value pairs of non-zero entries:
    
        (;; first instance in batch has two non-zero targets
         (;; class 10 has 30% expected probability
          (10 . 0.3)
          ;; class 2 has 70% expected probability
          (2 .  0.7))
         ;; second instance in batch puts 100% on class 7
         7
         ;; more instances in the batch follow
         ...)
    
    Actually, in the rare case where [`GROUP-SIZE`][a437] is not `SIZE`([`0`][85d3] [`1`][b99d]) (i.e. there are several softmax
    normalization groups for every example), the length of the above
    target sequence is `BATCH-SIZE`([`0`][8916] [`1`][ba18] [`2`][c918]) \* N-GROUPS. Indices are always
    relative to the start of the group.
    
    If [`GROUP-SIZE`][a437] is large (for example,
    in neural language models with a huge number of words), using
    sparse targets can make things go much faster, because calculation
    of the derivative is no longer quadratic.
    
    Giving different weights to training instances is implicitly
    supported. While target values in a group should sum to 1,
    multiplying all target values with a weight `W` is equivalent to
    training that `W` times on the same example.

<a id="x-28MGL-BP-3AENSURE-SOFTMAX-TARGET-MATRIX-20FUNCTION-29"></a>
- [function] **ENSURE-SOFTMAX-TARGET-MATRIX** *SOFTMAX-XE-LOSS N*

    Set [`TARGET`][b5c7] of `SOFTMAX-XE-LOSS` to a `MAT` capable of holding the dense
    target values for `N` stripes.

<a id="x-28MGL-BP-3A-40MGL-BP-STOCHASTICITY-20MGL-PAX-3ASECTION-29"></a>
#### 11.4.7 Stochasticity

<a id="x-28MGL-BP-3A-40MGL-BP-DROPOUT-LUMP-20MGL-PAX-3ASECTION-29"></a>
##### Dropout Lump

<a id="x-28MGL-BP-3A--3EDROPOUT-20CLASS-29"></a>
- [class] **-\>DROPOUT** *[LUMP][c1ac]*

    The output of this lump is identical to its input,
    except it randomly zeroes out some of them during training which act
    as a very strong regularizer. See Geoffrey Hinton's 'Improving
    neural networks by preventing co-adaptation of feature
    detectors'.
    
    The `SIZE`([`0`][85d3] [`1`][b99d]) of this lump is the size of its input which is determined
    automatically.

<a id="x-28MGL-BP-3ADROPOUT-20-28MGL-PAX-3AACCESSOR-20MGL-BP-3A--3EDROPOUT-29-29"></a>
- [accessor] **DROPOUT** *-\>DROPOUT (:DROPOUT = 0.5)*

    If non-NIL, then in the forward pass zero out each
    node in this chunk with `DROPOUT` probability.

<a id="x-28MGL-BP-3A-40MGL-BP-GAUSSIAN-RANDOM-LUMP-20MGL-PAX-3ASECTION-29"></a>
##### Gaussian Random Lump

<a id="x-28MGL-BP-3A--3EGAUSSIAN-RANDOM-20CLASS-29"></a>
- [class] **-\>GAUSSIAN-RANDOM** *[LUMP][c1ac]*

    This lump has no input, it produces normally
    distributed independent random numbers with [`MEAN`][d96a] and [`VARIANCE`][404c] (or
    [`VARIANCE-FOR-PREDICTION`][80e2]). This is useful building block for noise
    based regularization methods.
    
    ```common-lisp
    (->gaussian-random :size 10 :name 'normal :mean 1 :variance 2)
    ==> #<->GAUSSIAN-RANDOM NORMAL :SIZE 10 1/1 :NORM 0.00000>
    ```


<a id="x-28MGL-BP-3AMEAN-20-28MGL-PAX-3AACCESSOR-20MGL-BP-3A--3EGAUSSIAN-RANDOM-29-29"></a>
- [accessor] **MEAN** *-\>GAUSSIAN-RANDOM (:MEAN = 0)*

    The mean of the normal distribution.

<a id="x-28MGL-BP-3AVARIANCE-20-28MGL-PAX-3AACCESSOR-20MGL-BP-3A--3EGAUSSIAN-RANDOM-29-29"></a>
- [accessor] **VARIANCE** *-\>GAUSSIAN-RANDOM (:VARIANCE = 1)*

    The variance of the normal distribution.

<a id="x-28MGL-BP-3AVARIANCE-FOR-PREDICTION-20-28MGL-PAX-3AACCESSOR-20MGL-BP-3A--3EGAUSSIAN-RANDOM-29-29"></a>
- [accessor] **VARIANCE-FOR-PREDICTION** *-\>GAUSSIAN-RANDOM (:VARIANCE-FOR-PREDICTION = 0)*

    If not `NIL`, then this value overrides [`VARIANCE`][404c]
    when not in training (i.e. when making predictions).

<a id="x-28MGL-BP-3A-40MGL-BP-SAMPLE-BINARY-LUMP-20MGL-PAX-3ASECTION-29"></a>
##### Binary Sampling Lump

<a id="x-28MGL-BP-3A--3ESAMPLE-BINARY-20CLASS-29"></a>
- [class] **-\>SAMPLE-BINARY** *[LUMP][c1ac]*

    Treating values of its input as probabilities,
    sample independent binomials. Turn true into 1 and false into 0. The
    `SIZE`([`0`][85d3] [`1`][b99d]) of this lump is determined automatically from the size of its
    input.
    
    ```common-lisp
    (->sample-binary (->input :size 10) :name 'binarized-input)
    ==> #<->SAMPLE-BINARY BINARIZED-INPUT :SIZE 10 1/1 :NORM 0.00000>
    ```


<a id="x-28MGL-BP-3A-40MGL-BP-ARITHMETIC-20MGL-PAX-3ASECTION-29"></a>
#### 11.4.8 Arithmetic

<a id="x-28MGL-BP-3A-40MGL-BP-SUM-LUMP-20MGL-PAX-3ASECTION-29"></a>
##### Sum Lump

<a id="x-28MGL-BP-3A--3ESUM-20CLASS-29"></a>
- [class] **-\>SUM** *[LUMP][c1ac]*

    Computes the sum of all nodes of its input per
    stripe. This `SIZE`([`0`][85d3] [`1`][b99d]) of this lump is always 1.

<a id="x-28MGL-BP-3A-40MGL-BP-V-2AM-LUMP-20MGL-PAX-3ASECTION-29"></a>
##### Vector-Matrix Multiplication Lump

<a id="x-28MGL-BP-3A--3EV-2AM-20CLASS-29"></a>
- [class] **-\>V\*M** *[LUMP][c1ac]*

    Perform `X * WEIGHTS` where `X` (the input) is of
    size `M` and `WEIGHTS`([`0`][3c96] [`1`][a896]) is a [`->WEIGHT`][b76f] whose single stripe is taken to
    be of dimensions `M x N` stored in row major order. `N` is the size
    of this lump. If [`TRANSPOSE-WEIGHTS-P`][533e] then `WEIGHTS` is `N x M` and `X
    * WEIGHTS'` is computed.

<a id="x-28MGL-COMMON-3AWEIGHTS-20-28MGL-PAX-3AREADER-20MGL-BP-3A--3EV-2AM-29-29"></a>
- [reader] **WEIGHTS** *-\>V\*M (:WEIGHTS)*

    A [`->WEIGHT`][b76f] lump.

<a id="x-28MGL-BP-3ATRANSPOSE-WEIGHTS-P-20-28MGL-PAX-3AREADER-20MGL-BP-3A--3EV-2AM-29-29"></a>
- [reader] **TRANSPOSE-WEIGHTS-P** *-\>V\*M (:TRANSPOSE-WEIGHTS-P = NIL)*

    Determines whether the input is multiplied by
    `WEIGHTS`([`0`][3c96] [`1`][a896]) or its transpose.

<a id="x-28MGL-BP-3A-40MGL-BP--2B-LUMP-20MGL-PAX-3ASECTION-29"></a>
##### Elementwise Addition Lump

<a id="x-28MGL-BP-3A--3E-2B-20CLASS-29"></a>
- [class] **-\>+** *[LUMP][c1ac]*

    Performs elementwise addition on its input lumps.
    The `SIZE`([`0`][85d3] [`1`][b99d]) of this lump is automatically determined from the size of
    its inputs if there is at least one. If one of the inputs is a
    [`->WEIGHT`][b76f] lump, then it is added to every stripe.
    
    ```common-lisp
    (->+ (list (->input :size 10) (->weight :size 10 :name 'bias))
         :name 'plus)
    ==> #<->+ PLUS :SIZE 10 1/1 :NORM 0.00000>
    ```


<a id="x-28MGL-BP-3A-40MGL-BP--2A-LUMP-20MGL-PAX-3ASECTION-29"></a>
##### Elementwise Multiplication Lump

<a id="x-28MGL-BP-3A--3E-2A-20CLASS-29"></a>
- [class] **-\>\*** *[LUMP][c1ac]*

    Performs elementwise multiplication on its two
    input lumps. The `SIZE`([`0`][85d3] [`1`][b99d]) of this lump is automatically determined from
    the size of its inputs. Either input can be a [`->WEIGHT`][b76f] lump.
    
    ```common-lisp
    (->* (->input :size 10) (->weight :size 10 :name 'scale)
         :name 'mult)
    ==> #<->* MULT :SIZE 10 1/1 :NORM 0.00000>
    ```


<a id="x-28MGL-BP-3A-40MGL-BP-ABS-LUMP-20MGL-PAX-3ASECTION-29"></a>
##### Abs Lump

<a id="x-28MGL-BP-3A--3EABS-20CLASS-29"></a>
- [class] **-\>ABS** *[LUMP][c1ac]*

<a id="x-28MGL-BP-3A-40MGL-BP-EXP-LUMP-20MGL-PAX-3ASECTION-29"></a>
##### Exp Lump

<a id="x-28MGL-BP-3A--3EEXP-20CLASS-29"></a>
- [class] **-\>EXP** *[LUMP][c1ac]*

<a id="x-28MGL-BP-3A-40MGL-BP-NORMALIZED-LUMP-20MGL-PAX-3ASECTION-29"></a>
##### Normalized Lump

<a id="x-28MGL-BP-3A--3ENORMALIZED-20CLASS-29"></a>
- [class] **-\>NORMALIZED** *[LUMP][c1ac]*

<a id="x-28MGL-BP-3A-40MGL-BP-RNN-OPERATIONS-20MGL-PAX-3ASECTION-29"></a>
#### 11.4.9 Operations for `RNN`s

<a id="x-28MGL-BP-3A-40MGL-BP-LSTM-SUBNET-20MGL-PAX-3ASECTION-29"></a>
##### LSTM Subnet

<a id="x-28MGL-BP-3A--3ELSTM-20CLASS-29"></a>
- [class] **-\>LSTM** *[BPN][5187]*

    Long-Short Term Memory subnetworks are built by the
    function [`->LSTM`][2823] and they have many lumps hidden inside them. These
    lumps are packaged into a subnetwork to reduce clutter.

<a id="x-28MGL-BP-3A--3ELSTM-20FUNCTION-29"></a>
- [function] **-\>LSTM** *INPUTS &KEY NAME CELL-INIT OUTPUT-INIT SIZE (ACTIVATION-FN '-\>ACTIVATION) (GATE-FN '-\>SIGMOID) (INPUT-FN '-\>TANH) (OUTPUT-FN '-\>TANH) (PEEPHOLES T)*

    Create an LSTM layer consisting of input, forget, output gates with
    which input, cell state and output are scaled. Lots of lumps are
    created, the final one representing to output of the LSTM has `NAME`.
    The rest of the lumps are named automatically based on `NAME`. This
    function returns only the output lump (`m`), but all created lumps
    are added automatically to the [`BPN`][5187] being built.
    
    There are many papers and tutorials on LSTMs. This version is well
    described in "Long Short-Term Memory Recurrent Neural Network
    Architectures for Large Scale Acoustic Modeling" (2014, Hasim Sak,
    Andrew Senior, Francoise Beaufays). Using the notation from that
    paper:
    
    $$
    i\_t = s(W\_{ix} x\_t + W\_{im} m\_{t-1} + W\_{ic} \odot
    c\_{t-1} + b\_i)
    $$
    
    $$
    f\_t = s(W\_{fx} x\_t + W\_{fm} m\_{t-1} + W\_{fc} \odot
    c\_{t-1} + b\_f)
    $$
    
    $$
    c\_t = f\_t \odot c\_{t-1} + i\_t \odot g(W\_{cx} x\_t +
    W\_{cm} m\_{t-1} + b\_c)
    $$
    
    $$
    o\_t = s(W\_{ox} x\_t + W\_{om} m\_{t-1} + W\_{oc} \odot
    c\_t + b\_o)
    $$
    
    $$
    m\_t = o\_t \odot h(c\_t),
    $$
    
    where `i`, `f`, and `o` are the input, forget and output gates. `c`
    is the cell state and `m` is the actual output.
    
    Weight matrices for connections from `c` (`W_ic`, `W_fc` and `W_oc`)
    are diagonal and represented by just the vector of diagonal values.
    These connections are only added if `PEEPHOLES` is true.
    
    A notable difference from the paper is that in addition to being a
    single lump, `x_t` (`INPUTS`) can also be a list of lumps. Whenever
    some activation is to be calculated based on `x_t`, it is going to
    be the sum of individual activations. For example, `W_ix * x_t` is
    really `sum_j W_ijx * inputs_j`.
    
    If `CELL-INIT` is non-NIL, then it must be a [`CLUMP`][a4fe] of `SIZE` form which
    stands for the initial state of the value cell (`c_{-1}`). `CELL-INIT`
    being `NIL` is equivalent to the state of all zeros.
    
    `ACTIVATION-FN` defaults to `->ACTIVATION`([`0`][7162] [`1`][b602]), but it can be for example
    [`->BATCH-NORMALIZED-ACTIVATION`][0f0f]. In general, functions like the
    aforementioned two with signature like (`INPUTS` `&KEY` `NAME` `SIZE`
    `PEEPHOLES`) can be passed as `ACTIVATION-FN`.

<a id="x-28MGL-BP-3A-40MGL-BP-SEQ-BARRIER-LUMP-20MGL-PAX-3ASECTION-29"></a>
##### Sequence Barrier Lump

<a id="x-28MGL-BP-3A--3ESEQ-BARRIER-20CLASS-29"></a>
- [class] **-\>SEQ-BARRIER** *[LUMP][c1ac]*

    In an [`RNN`][b0f3], processing of stripes (instances in the
    batch) may require different number of time step so the final state
    for stripe 0 is in stripe 0 of some lump L at time step 7, while for
    stripe 1 it is in stripe 1 of sump lump L at time step 42.
    
    This lump copies the per-stripe states from different lumps into a
    single lump so that further processing can take place (typically
    when the `RNN` is embedded in another network).
    
    The `SIZE`([`0`][85d3] [`1`][b99d]) of this lump is automatically set to the size of the lump
    returned by `(FUNCALL SEQ-ELT-FN 0)`.

<a id="x-28MGL-BP-3ASEQ-ELT-FN-20-28MGL-PAX-3AREADER-20MGL-BP-3A--3ESEQ-BARRIER-29-29"></a>
- [reader] **SEQ-ELT-FN** *-\>SEQ-BARRIER (:SEQ-ELT-FN)*

    A function of an `INDEX` argument that returns the
    lump with that index in some sequence.

<a id="x-28MGL-BP-3ASEQ-INDICES-20-28MGL-PAX-3AACCESSOR-20MGL-BP-3A--3ESEQ-BARRIER-29-29"></a>
- [accessor] **SEQ-INDICES** *-\>SEQ-BARRIER*

    A sequence of length batch size of indices. The
    element at index `I` is the index to be passed to [`SEQ-ELT-FN`][29c0] to
    find the lump whose stripe `I` is copied to stripe `I` of this
    this lump.

<a id="x-28MGL-BP-3A-40MGL-BP-UTILITIES-20MGL-PAX-3ASECTION-29"></a>
### 11.5 Utilities

<a id="x-28MGL-BP-3ARENORMALIZE-ACTIVATIONS-20FUNCTION-29"></a>
- [function] **RENORMALIZE-ACTIVATIONS** *-\>V\*M-LUMPS L2-UPPER-BOUND*

    If the l2 norm of the incoming weight vector of the a unit is
    larger than `L2-UPPER-BOUND` then renormalize it to `L2-UPPER-BOUND`.
    The list of `->V*M-LUMPS` is assumed to be eventually fed to the same
    lump.
    
    To use it, group the activation clumps into the same GD-OPTIMIZER
    and hang this function on [`AFTER-UPDATE-HOOK`][124f], that latter of which is
    done for you [`ARRANGE-FOR-RENORMALIZING-ACTIVATIONS`][8b55].
    
    See "Improving neural networks by preventing co-adaptation of
    feature detectors (Hinton, 2012)",
    <http://arxiv.org/pdf/1207.0580.pdf>.

<a id="x-28MGL-BP-3AARRANGE-FOR-RENORMALIZING-ACTIVATIONS-20FUNCTION-29"></a>
- [function] **ARRANGE-FOR-RENORMALIZING-ACTIVATIONS** *BPN OPTIMIZER L2-UPPER-BOUND*

    By pushing a lambda to [`AFTER-UPDATE-HOOK`][124f] of `OPTIMIZER` arrange for
    all weights beings trained by `OPTIMIZER` to be renormalized (as in
    [`RENORMALIZE-ACTIVATIONS`][c7fa] with `L2-UPPER-BOUND`).
    
    It is assumed that if the weights either belong to an activation
    lump or are simply added to the activations (i.e. they are biases).

<a id="x-28MGL-3A-40MGL-BM-20MGL-PAX-3ASECTION-29"></a>
## 12 Boltzmann Machines


<a id="x-28MGL-3A-40MGL-GP-20MGL-PAX-3ASECTION-29"></a>
## 13 Gaussian Processes


<a id="x-28MGL-NLP-3A-40MGL-NLP-20MGL-PAX-3ASECTION-29"></a>
## 14 Natural Language Processing

###### \[in package MGL-NLP\]
This in nothing more then a couple of utilities for now which may
grow into a more serious toolset for NLP eventually.

<a id="x-28MGL-NLP-3AMAKE-N-GRAM-MAPPEE-20FUNCTION-29"></a>
- [function] **MAKE-N-GRAM-MAPPEE** *FUNCTION N*

    Make a function of a single argument that's suitable as the
    function argument to a mapper function. It calls `FUNCTION` with every
    `N` element.
    
    ```common-lisp
    (map nil (make-n-gram-mappee #'print 3) '(a b c d e))
    ..
    .. (A B C) 
    .. (B C D) 
    .. (C D E) 
    ```


<a id="x-28MGL-NLP-3ABLEU-20FUNCTION-29"></a>
- [function] **BLEU** *CANDIDATES REFERENCES &KEY CANDIDATE-KEY REFERENCE-KEY (N 4)*

    Compute the [BLEU score](http://en.wikipedia.org/wiki/BLEU) for
    bilingual CORPUS. `BLEU` measures how good a translation is compared
    to human reference translations.
    
    `CANDIDATES` (keyed by `CANDIDATE-KEY`) and `REFERENCES` (keyed by
    `REFERENCE-KEY`) are sequences of sentences. A sentence is a sequence
    of words. Words are compared with [`EQUAL`][96d0], and may be any kind of
    object (not necessarily strings).
    
    Currently there is no support for multiple reference translations. `N`
    determines the largest n-grams to consider.
    
    The first return value is the `BLEU` score (between 0 and 1, not as a
    percentage). The second value is the sum of the lengths of
    `CANDIDATES` divided by the sum of the lengths of `REFERENCES` (or `NIL`,
    if the denominator is 0). The third is a list of n-gram
    precisions (also between 0 and 1 or `NIL`), one for each element in
    \[1..`N`\].
    
    This is basically a reimplementation of
    [multi-bleu.perl](https://github.com/moses-smt/mosesdecoder/blob/master/scripts/generic/multi-bleu.perl).
    
    ```common-lisp
    (bleu '((1 2 3 4) (a b))
          '((1 2 3 4) (1 2)))
    => 0.8408964
    => 1
    => (;; 1-gram precision: 4/6
        2/3
        ;; 2-gram precision: 3/4
        3/4
        ;; 3-gram precision: 2/2
        1
        ;; 4-gram precision: 1/1
        1)
    ```


<a id="x-28MGL-NLP-3A-40MGL-NLP-BAG-OF-WORDS-20MGL-PAX-3ASECTION-29"></a>
### 14.1 Bag of Words

<a id="x-28MGL-NLP-3ABAG-OF-WORDS-ENCODER-20CLASS-29"></a>
- [class] **BAG-OF-WORDS-ENCODER**

    [`ENCODE`][fedd] all features of a document with a sparse
    vector. Get the features of document from `MAPPER`, encode each
    feature with [`FEATURE-ENCODER`][96d0f]. `FEATURE-ENCODER` may return `NIL` if the
    feature is not used. The result is a vector of encoded-feature/value
    conses. encoded-features are unique (under [`ENCODED-FEATURE-TEST`][21ca])
    within the vector but are in no particular order.
    
    Depending on `KIND`, value is calculated in various ways:
    
    - For `:FREQUENCY` it is the number of times the corresponding feature
    was found in `DOCUMENT`.
    
    - For `:BINARY` it is always 1.
    
    - For `:NORMALIZED-FREQUENCY` and `:NORMALIZED-BINARY` are like the
      unnormalized counterparts except that as the final step values in
      the assembled sparse vector are normalized to sum to 1.
    
    - Finally, `:COMPACTED-BINARY` is like `:BINARY` but the return values
      is not a vector of conses, but a vector of element-type
      [`ENCODED-FEATURE-TYPE`][d443].
    
    ```common-lisp
    (let* ((feature-indexer
             (make-indexer
              (alexandria:alist-hash-table '(("I" . 3) ("me" . 2) ("mine" . 1)))
              2))
           (bag-of-words-encoder
             (make-instance 'bag-of-words-encoder
                            :feature-encoder feature-indexer
                            :feature-mapper (lambda (fn document)
                                              (map nil fn document))
                            :kind :frequency)))
      (encode bag-of-words-encoder '("All" "through" "day" "I" "me" "mine"
                                     "I" "me" "mine" "I" "me" "mine")))
    => #((0 . 3.0d0) (1 . 3.0d0))
    ```


<a id="x-28MGL-NLP-3AFEATURE-ENCODER-20-28MGL-PAX-3AREADER-20MGL-NLP-3ABAG-OF-WORDS-ENCODER-29-29"></a>
- [reader] **FEATURE-ENCODER** *BAG-OF-WORDS-ENCODER (:FEATURE-ENCODER)*

<a id="x-28MGL-NLP-3AFEATURE-MAPPER-20-28MGL-PAX-3AREADER-20MGL-NLP-3ABAG-OF-WORDS-ENCODER-29-29"></a>
- [reader] **FEATURE-MAPPER** *BAG-OF-WORDS-ENCODER (:FEATURE-MAPPER)*

<a id="x-28MGL-NLP-3AENCODED-FEATURE-TEST-20-28MGL-PAX-3AREADER-20MGL-NLP-3ABAG-OF-WORDS-ENCODER-29-29"></a>
- [reader] **ENCODED-FEATURE-TEST** *BAG-OF-WORDS-ENCODER (:ENCODED-FEATURE-TEST = #'EQL)*

<a id="x-28MGL-NLP-3AENCODED-FEATURE-TYPE-20-28MGL-PAX-3AREADER-20MGL-NLP-3ABAG-OF-WORDS-ENCODER-29-29"></a>
- [reader] **ENCODED-FEATURE-TYPE** *BAG-OF-WORDS-ENCODER (:ENCODED-FEATURE-TYPE = T)*

<a id="x-28MGL-NLP-3ABAG-OF-WORDS-KIND-20-28MGL-PAX-3AREADER-20MGL-NLP-3ABAG-OF-WORDS-ENCODER-29-29"></a>
- [reader] **BAG-OF-WORDS-KIND** *BAG-OF-WORDS-ENCODER (:KIND = :BINARY)*

  [0072]: #x-28MGL-OPT-3AON-OPTIMIZATION-FINISHED-20-28MGL-PAX-3AACCESSOR-20MGL-OPT-3AITERATIVE-OPTIMIZER-29-29 "MGL-OPT:ON-OPTIMIZATION-FINISHED (MGL-PAX:ACCESSOR MGL-OPT:ITERATIVE-OPTIMIZER)"
  [0078]: #x-28MGL-CORE-3AINSTANCE-TO-EXECUTOR-PARAMETERS-20GENERIC-FUNCTION-29 "MGL-CORE:INSTANCE-TO-EXECUTOR-PARAMETERS GENERIC-FUNCTION"
  [00a0]: #x-28MGL-BP-3ABP-LEARNER-20CLASS-29 "MGL-BP:BP-LEARNER CLASS"
  [00ee]: #x-28MGL-3A-40MGL-LINKS-20MGL-PAX-3ASECTION-29 "Links"
  [011d]: #x-28MGL-GD-3AMEAN-DECAY-20-28MGL-PAX-3AACCESSOR-20MGL-GD-3AADAM-OPTIMIZER-29-29 "MGL-GD:MEAN-DECAY (MGL-PAX:ACCESSOR MGL-GD:ADAM-OPTIMIZER)"
  [038e]: #x-28MGL-BP-3AIMPORTANCE-20-28MGL-PAX-3AACCESSOR-20MGL-BP-3A--3ELOSS-29-29 "MGL-BP:IMPORTANCE (MGL-PAX:ACCESSOR MGL-BP:->LOSS)"
  [0414]: #x-28MGL-COMMON-3ANAME-20-28METHOD-20NIL-20-28MGL-CORE-3AATTRIBUTED-29-29-29 "MGL-COMMON:NAME (METHOD NIL (MGL-CORE:ATTRIBUTED))"
  [0784]: #x-28MGL-NLP-3A-40MGL-NLP-BAG-OF-WORDS-20MGL-PAX-3ASECTION-29 "Bag of Words"
  [07c7]: #x-28MGL-CORE-3A-40MGL-CONFUSION-MATRIX-20MGL-PAX-3ASECTION-29 "Confusion Matrices"
  [07fb]: #x-28MGL-CORE-3AN-STRIPES-20-28MGL-PAX-3AREADER-20MGL-BP-3ABPN-29-29 "MGL-CORE:N-STRIPES (MGL-PAX:READER MGL-BP:BPN)"
  [084d]: #x-28MGL-BP-3AMAX-LAG-20-28MGL-PAX-3AREADER-20MGL-BP-3ARNN-29-29 "MGL-BP:MAX-LAG (MGL-PAX:READER MGL-BP:RNN)"
  [08ac]: #x-28MGL-DATASET-3AGENERATOR-20-28MGL-PAX-3AREADER-20MGL-DATASET-3AFUNCTION-SAMPLER-29-29 "MGL-DATASET:GENERATOR (MGL-PAX:READER MGL-DATASET:FUNCTION-SAMPLER)"
  [0900]: #x-28MGL-GD-3AVARIANCE-DECAY-20-28MGL-PAX-3AACCESSOR-20MGL-GD-3AADAM-OPTIMIZER-29-29 "MGL-GD:VARIANCE-DECAY (MGL-PAX:ACCESSOR MGL-GD:ADAM-OPTIMIZER)"
  [0933]: #x-28MGL-BP-3AMONITOR-BPN-RESULTS-20FUNCTION-29 "MGL-BP:MONITOR-BPN-RESULTS FUNCTION"
  [09ed]: #x-28MGL-GD-3ALEARNING-RATE-20-28MGL-PAX-3AACCESSOR-20MGL-GD-3A-3AGD-OPTIMIZER-29-29 "MGL-GD:LEARNING-RATE (MGL-PAX:ACCESSOR MGL-GD::GD-OPTIMIZER)"
  [0ba7]: #x-28MGL-CORE-3A-40MGL-CLASSIFICATION-MEASURER-20MGL-PAX-3ASECTION-29 "Classification Measurers"
  [0c91]: #x-28MGL-GD-3A-40MGL-GD-NORMALIZED-BATCH-GD-OPTIMIZER-20MGL-PAX-3ASECTION-29 "Normalized Batch Optimizer"
  [0c9e]: #x-28MGL-CORE-3ASET-INPUT-20GENERIC-FUNCTION-29 "MGL-CORE:SET-INPUT GENERIC-FUNCTION"
  [0d6a]: #x-28MGL-NLP-3A-40MGL-NLP-20MGL-PAX-3ASECTION-29 "Natural Language Processing"
  [0d82]: #x-28MGL-BP-3A-40MGL-BP-TRAINING-20MGL-PAX-3ASECTION-29 "Training"
  [0f0f]: #x-28MGL-BP-3A--3EBATCH-NORMALIZED-ACTIVATION-20FUNCTION-29 "MGL-BP:->BATCH-NORMALIZED-ACTIVATION FUNCTION"
  [109e]: #x-28MGL-DATASET-3A-40MGL-DATASET-20MGL-PAX-3ASECTION-29 "Datasets"
  [10e7]: #x-28MGL-GD-3A-40MGL-GD-20MGL-PAX-3ASECTION-29 "Gradient Descent"
  [117a]: http://www.lispworks.com/documentation/HyperSpec/Body/f_open.htm "OPEN FUNCTION"
  [124f]: #x-28MGL-GD-3AAFTER-UPDATE-HOOK-20-28MGL-PAX-3AACCESSOR-20MGL-GD-3A-3AGD-OPTIMIZER-29-29 "MGL-GD:AFTER-UPDATE-HOOK (MGL-PAX:ACCESSOR MGL-GD::GD-OPTIMIZER)"
  [1339]: #x-28MGL-CORE-3ADECODE-20GENERIC-FUNCTION-29 "MGL-CORE:DECODE GENERIC-FUNCTION"
  [1355]: #x-28MGL-BP-3A-40MGL-FNN-20MGL-PAX-3ASECTION-29 "Feed-Forward Nets"
  [16c4]: #x-28MGL-CORE-3AMAX-N-STRIPES-20GENERIC-FUNCTION-29 "MGL-CORE:MAX-N-STRIPES GENERIC-FUNCTION"
  [175f]: #x-28MGL-BP-3AFIND-CLUMP-20FUNCTION-29 "MGL-BP:FIND-CLUMP FUNCTION"
  [17b7]: http://www.lispworks.com/documentation/HyperSpec/Body/m_setf.htm "SETF MGL-PAX:MACRO"
  [17d3]: #x-28MGL-DATASET-3AMAP-DATASET-20FUNCTION-29 "MGL-DATASET:MAP-DATASET FUNCTION"
  [1abb]: http://www.lispworks.com/documentation/HyperSpec/Body/f_numera.htm "DENOMINATOR FUNCTION"
  [1b5e]: #x-28MGL-CORE-3A-40MGL-FEATURE-SELECTION-20MGL-PAX-3ASECTION-29 "Feature Selection"
  [1beb]: #x-28MGL-CORE-3AENCODER-2FDECODER-20CLASS-29 "MGL-CORE:ENCODER/DECODER CLASS"
  [1cab]: #x-28MGL-DATASET-3AMAX-N-SAMPLES-20-28MGL-PAX-3AACCESSOR-20MGL-DATASET-3AFUNCTION-SAMPLER-29-29 "MGL-DATASET:MAX-N-SAMPLES (MGL-PAX:ACCESSOR MGL-DATASET:FUNCTION-SAMPLER)"
  [207b]: #x-28MGL-BP-3A-40MGL-BP-INPUTS-20MGL-PAX-3ASECTION-29 "Inputs"
  [20ca]: #x-28MGL-OPT-3ADO-GRADIENT-SINK-20MGL-PAX-3AMACRO-29 "MGL-OPT:DO-GRADIENT-SINK MGL-PAX:MACRO"
  [20e8]: #x-28MGL-CORE-3ACOUNTER-VALUES-20GENERIC-FUNCTION-29 "MGL-CORE:COUNTER-VALUES GENERIC-FUNCTION"
  [2171]: #x-28MGL-BP-3A--3ELOSS-20CLASS-29 "MGL-BP:->LOSS CLASS"
  [21ca]: #x-28MGL-NLP-3AENCODED-FEATURE-TEST-20-28MGL-PAX-3AREADER-20MGL-NLP-3ABAG-OF-WORDS-ENCODER-29-29 "MGL-NLP:ENCODED-FEATURE-TEST (MGL-PAX:READER MGL-NLP:BAG-OF-WORDS-ENCODER)"
  [2312]: #x-28MGL-OPT-3AMAP-SEGMENTS-20GENERIC-FUNCTION-29 "MGL-OPT:MAP-SEGMENTS GENERIC-FUNCTION"
  [2481]: #x-28MGL-BP-3ADROPOUT-20-28MGL-PAX-3AACCESSOR-20MGL-BP-3A--3EDROPOUT-29-29 "MGL-BP:DROPOUT (MGL-PAX:ACCESSOR MGL-BP:->DROPOUT)"
  [24aa]: #x-28MGL-CORE-3A-40MGL-FEATURE-ENCODING-20MGL-PAX-3ASECTION-29 "Feature Encoding"
  [25fd]: #x-28MGL-GD-3A-40MGL-GD-SGD-OPTIMIZER-20MGL-PAX-3ASECTION-29 "SGD Optimizer"
  [2739]: http://www.lispworks.com/documentation/HyperSpec/Body/s_let_l.htm "LET* MGL-PAX:MACRO"
  [2823]: #x-28MGL-BP-3A--3ELSTM-20FUNCTION-29 "MGL-BP:->LSTM FUNCTION"
  [2981]: #x-28MGL-DIFFUN-3A-40MGL-DIFFUN-20MGL-PAX-3ASECTION-29 "Differentiable Functions"
  [29a1]: #x-28MGL-CORE-3A-40MGL-PERSISTENCE-20MGL-PAX-3ASECTION-29 "Persistence"
  [29c0]: #x-28MGL-BP-3ASEQ-ELT-FN-20-28MGL-PAX-3AREADER-20MGL-BP-3A--3ESEQ-BARRIER-29-29 "MGL-BP:SEQ-ELT-FN (MGL-PAX:READER MGL-BP:->SEQ-BARRIER)"
  [2a2f]: #x-28MGL-GD-3ASGD-OPTIMIZER-20CLASS-29 "MGL-GD:SGD-OPTIMIZER CLASS"
  [2aa3]: #x-28MGL-CORE-3AMAKE-CLASSIFICATION-ACCURACY-MONITORS-2A-20GENERIC-FUNCTION-29 "MGL-CORE:MAKE-CLASSIFICATION-ACCURACY-MONITORS* GENERIC-FUNCTION"
  [2c39]: #x-28MGL-GD-3A-40MGL-GD-BATCH-GD-OPTIMIZER-20MGL-PAX-3ASECTION-29 "Batch Based Optimizers"
  [2e8b]: #x-28MGL-CORE-3AWITH-PADDED-ATTRIBUTE-PRINTING-20MGL-PAX-3AMACRO-29 "MGL-CORE:WITH-PADDED-ATTRIBUTE-PRINTING MGL-PAX:MACRO"
  [2fe9]: #x-28MGL-BP-3A-40MGL-BP-ARITHMETIC-20MGL-PAX-3ASECTION-29 "Arithmetic"
  [3045]: #x-28MGL-BP-3A-40MGL-BP-LUMP-20MGL-PAX-3ASECTION-29 "Lump Base Class"
  [31ed]: #x-28MGL-CORE-3ALABEL-INDICES-20GENERIC-FUNCTION-29 "MGL-CORE:LABEL-INDICES GENERIC-FUNCTION"
  [331b]: #x-28MGL-CORE-3AMAKE-EXECUTOR-WITH-PARAMETERS-20GENERIC-FUNCTION-29 "MGL-CORE:MAKE-EXECUTOR-WITH-PARAMETERS GENERIC-FUNCTION"
  [332e]: #x-28MGL-3A-40MGL-BM-20MGL-PAX-3ASECTION-29 "Boltzmann Machines"
  [376e]: http://www.lispworks.com/documentation/HyperSpec/Body/f_apply.htm "APPLY FUNCTION"
  [3815]: #x-28MGL-OPT-3AMAKE-COST-MONITORS-2A-20GENERIC-FUNCTION-29 "MGL-OPT:MAKE-COST-MONITORS* GENERIC-FUNCTION"
  [38a0]: http://www.lispworks.com/documentation/HyperSpec/Body/f_descri.htm "DESCRIBE FUNCTION"
  [3c96]: #x-28MGL-COMMON-3AWEIGHTS-20-28MGL-PAX-3AREADER-20MGL-BP-3A--3EEMBEDDING-29-29 "MGL-COMMON:WEIGHTS (MGL-PAX:READER MGL-BP:->EMBEDDING)"
  [3ce0]: #x-28MGL-GD-3ASEGMENTED-GD-OPTIMIZER-20CLASS-29 "MGL-GD:SEGMENTED-GD-OPTIMIZER CLASS"
  [3f9f]: #x-28MGL-RESAMPLE-3A-40MGL-RESAMPLE-CV-BAGGING-20MGL-PAX-3ASECTION-29 "CV Bagging"
  [401f]: #x-28MGL-DATASET-3AFINISHEDP-20GENERIC-FUNCTION-29 "MGL-DATASET:FINISHEDP GENERIC-FUNCTION"
  [404c]: #x-28MGL-BP-3AVARIANCE-20-28MGL-PAX-3AACCESSOR-20MGL-BP-3A--3EGAUSSIAN-RANDOM-29-29 "MGL-BP:VARIANCE (MGL-PAX:ACCESSOR MGL-BP:->GAUSSIAN-RANDOM)"
  [40e2]: http://www.lispworks.com/documentation/HyperSpec/Body/f_max_m.htm "MIN FUNCTION"
  [410c]: #x-28MGL-COMMON-3ACOST-20GENERIC-FUNCTION-29 "MGL-COMMON:COST GENERIC-FUNCTION"
  [430d]: #x-28MGL-CORE-3ACLASSIFICATION-ACCURACY-COUNTER-20CLASS-29 "MGL-CORE:CLASSIFICATION-ACCURACY-COUNTER CLASS"
  [441b]: #x-28MGL-BP-3A--3EDROPOUT-20CLASS-29 "MGL-BP:->DROPOUT CLASS"
  [443c]: #x-28MGL-3A-40MGL-CODE-ORGANIZATION-20MGL-PAX-3ASECTION-29 "Code Organization"
  [4476]: #x-28MGL-CORE-3A-40MGL-EXECUTORS-20MGL-PAX-3ASECTION-29 "Executors"
  [4528]: #x-28MGL-OPT-3AMONITOR-OPTIMIZATION-PERIODICALLY-20FUNCTION-29 "MGL-OPT:MONITOR-OPTIMIZATION-PERIODICALLY FUNCTION"
  [45be]: http://www.lispworks.com/documentation/HyperSpec/Body/f_exp_e.htm "EXP FUNCTION"
  [46a4]: #x-28MGL-OPT-3AMINIMIZE-20FUNCTION-29 "MGL-OPT:MINIMIZE FUNCTION"
  [46c2]: #x-28MGL-OPT-3AMAKE-COST-MONITORS-20FUNCTION-29 "MGL-OPT:MAKE-COST-MONITORS FUNCTION"
  [46c4]: #x-28MGL-BP-3APOPULATION-DECAY-20-28MGL-PAX-3AREADER-20MGL-BP-3A--3EBATCH-NORMALIZATION-29-29 "MGL-BP:POPULATION-DECAY (MGL-PAX:READER MGL-BP:->BATCH-NORMALIZATION)"
  [4a8e]: #x-28MGL-3A-40MGL-GLOSSARY-20MGL-PAX-3ASECTION-29 "Glossary"
  [4bf1]: #x-28MGL-OPT-3AACCUMULATE-GRADIENTS-2A-20GENERIC-FUNCTION-29 "MGL-OPT:ACCUMULATE-GRADIENTS* GENERIC-FUNCTION"
  [4c73]: #x-28MGL-OPT-3AN-INSTANCES-20-28MGL-PAX-3AREADER-20MGL-OPT-3AITERATIVE-OPTIMIZER-29-29 "MGL-OPT:N-INSTANCES (MGL-PAX:READER MGL-OPT:ITERATIVE-OPTIMIZER)"
  [4f0b]: #x-28MGL-OPT-3AON-N-INSTANCES-CHANGED-20-28MGL-PAX-3AACCESSOR-20MGL-OPT-3AITERATIVE-OPTIMIZER-29-29 "MGL-OPT:ON-N-INSTANCES-CHANGED (MGL-PAX:ACCESSOR MGL-OPT:ITERATIVE-OPTIMIZER)"
  [4f0e]: #x-28MGL-BP-3A-40MGL-BP-MONITORING-20MGL-PAX-3ASECTION-29 "Monitoring"
  [4ffb]: #x-28MGL-CG-3ACG-20FUNCTION-29 "MGL-CG:CG FUNCTION"
  [5003]: http://www.lispworks.com/documentation/HyperSpec/Body/f_vector.htm "VECTOR FUNCTION"
  [5187]: #x-28MGL-BP-3ABPN-20CLASS-29 "MGL-BP:BPN CLASS"
  [51d5]: #x-28MGL-BP-3AWARP-LENGTH-20-28MGL-PAX-3AREADER-20MGL-BP-3ARNN-29-29 "MGL-BP:WARP-LENGTH (MGL-PAX:READER MGL-BP:RNN)"
  [51f7]: #x-28MGL-BP-3A-40MGL-BP-RNN-OPERATIONS-20MGL-PAX-3ASECTION-29 "Operations for `RNN`s"
  [5293]: #x-28MGL-RESAMPLE-3ASPLIT-FOLD-2FCONT-20FUNCTION-29 "MGL-RESAMPLE:SPLIT-FOLD/CONT FUNCTION"
  [5309]: #x-28MGL-BP-3A--3ETANH-20CLASS-29 "MGL-BP:->TANH CLASS"
  [533e]: #x-28MGL-BP-3ATRANSPOSE-WEIGHTS-P-20-28MGL-PAX-3AREADER-20MGL-BP-3A--3EV-2AM-29-29 "MGL-BP:TRANSPOSE-WEIGHTS-P (MGL-PAX:READER MGL-BP:->V*M)"
  [5611]: #x-28MGL-GD-3AMOMENTUM-TYPE-20-28MGL-PAX-3AREADER-20MGL-GD-3A-3AGD-OPTIMIZER-29-29 "MGL-GD:MOMENTUM-TYPE (MGL-PAX:READER MGL-GD::GD-OPTIMIZER)"
  [56b2]: #x-28MGL-BP-3A-40MGL-BP-OVERVIEW-20MGL-PAX-3ASECTION-29 "Backprop Overview"
  [5748]: #x-28MGL-OPT-3A-40MGL-OPT-OPTIMIZER-20MGL-PAX-3ASECTION-29 "Implementing Optimizers"
  [5752]: #x-28MGL-CORE-3ACOUNTER-20-28MGL-PAX-3AREADER-20MGL-CORE-3AMONITOR-29-29 "MGL-CORE:COUNTER (MGL-PAX:READER MGL-CORE:MONITOR)"
  [592c]: http://www.lispworks.com/documentation/HyperSpec/Body/f_list_.htm "LIST FUNCTION"
  [5979]: #x-28MGL-CORE-3ABASIC-COUNTER-20CLASS-29 "MGL-CORE:BASIC-COUNTER CLASS"
  [59c2]: #x-28MGL-RESAMPLE-3A-40MGL-RESAMPLE-MISC-20MGL-PAX-3ASECTION-29 "Miscellaneous Operations"
  [59dd]: #x-28MGL-COMMON-3AGROUP-SIZE-20-28MGL-PAX-3AREADER-20MGL-BP-3A--3EMAX-29-29 "MGL-COMMON:GROUP-SIZE (MGL-PAX:READER MGL-BP:->MAX)"
  [5a43]: #x-28MGL-GD-3APER-WEIGHT-BATCH-GD-OPTIMIZER-20CLASS-29 "MGL-GD:PER-WEIGHT-BATCH-GD-OPTIMIZER CLASS"
  [5bd4]: #x-28MGL-BP-3ABACKWARD-20GENERIC-FUNCTION-29 "MGL-BP:BACKWARD GENERIC-FUNCTION"
  [5c66]: http://www.lispworks.com/documentation/HyperSpec/Body/f_countc.htm "COUNT FUNCTION"
  [5d86]: #x-28MGL-BP-3A-40MGL-BP-ACTIVATION-FUNCTIONS-20MGL-PAX-3ASECTION-29 "Activation Functions"
  [5ded]: #x-28MGL-RESAMPLE-3ASPLIT-FOLD-2FMOD-20FUNCTION-29 "MGL-RESAMPLE:SPLIT-FOLD/MOD FUNCTION"
  [5e58]: #x-28MGL-BP-3AINPUT-ROW-INDICES-20-28MGL-PAX-3AREADER-20MGL-BP-3A--3EEMBEDDING-29-29 "MGL-BP:INPUT-ROW-INDICES (MGL-PAX:READER MGL-BP:->EMBEDDING)"
  [5fdc]: #x-28MGL-CORE-3AMAP-BATCHES-FOR-MODEL-20FUNCTION-29 "MGL-CORE:MAP-BATCHES-FOR-MODEL FUNCTION"
  [6004]: #x-28MGL-CORE-3AMAKE-CROSS-ENTROPY-MONITORS-20FUNCTION-29 "MGL-CORE:MAKE-CROSS-ENTROPY-MONITORS FUNCTION"
  [606c]: #x-28MGL-BP-3ABUILD-FNN-20MGL-PAX-3AMACRO-29 "MGL-BP:BUILD-FNN MGL-PAX:MACRO"
  [60b3]: #x-28MGL-3A-40MGL-GP-20MGL-PAX-3ASECTION-29 "Gaussian Processes"
  [60d2]: #x-28MGL-CORE-3ACONFUSION-MATRIX-20CLASS-29 "MGL-CORE:CONFUSION-MATRIX CLASS"
  [60e3]: #x-28MGL-CORE-3A-40MGL-CLASSIFICATION-20MGL-PAX-3ASECTION-29 "Classification"
  [6202]: #x-28MGL-CORE-3AMONITORS-20-28MGL-PAX-3AACCESSOR-20MGL-BP-3ABP-LEARNER-29-29 "MGL-CORE:MONITORS (MGL-PAX:ACCESSOR MGL-BP:BP-LEARNER)"
  [627a]: #x-28MGL-RESAMPLE-3AFRACTURE-STRATIFIED-20FUNCTION-29 "MGL-RESAMPLE:FRACTURE-STRATIFIED FUNCTION"
  [62de]: #x-28MGL-CORE-3AADD-TO-COUNTER-20GENERIC-FUNCTION-29 "MGL-CORE:ADD-TO-COUNTER GENERIC-FUNCTION"
  [6598]: #x-28MGL-CORE-3A-40MGL-CLASSIFICATION-COUNTER-20MGL-PAX-3ASECTION-29 "Classification Counters"
  [6872]: #x-28MGL-BP-3A-40MGL-BP-WEIGHT-LUMP-20MGL-PAX-3ASECTION-29 "Weight Lump"
  [6a6f]: #x-28MGL-OPT-3A-40MGL-OPT-EXTENSION-API-20MGL-PAX-3ASECTION-29 "Extension API"
  [6b38]: #x-28MGL-BP-3A-40MGL-FNN-TUTORIAL-20MGL-PAX-3ASECTION-29 "`FNN` Tutorial"
  [6b4a]: http://www.lispworks.com/documentation/HyperSpec/Body/f_funcal.htm "FUNCALL FUNCTION"
  [6bd7]: #x-28MGL-CORE-3ALOAD-STATE-20FUNCTION-29 "MGL-CORE:LOAD-STATE FUNCTION"
  [6c7f]: http://www.lispworks.com/documentation/HyperSpec/Body/f_mod_r.htm "MOD FUNCTION"
  [6da5]: #x-28MGL-CORE-3A-40MGL-ATTRIBUTES-20MGL-PAX-3ASECTION-29 "Attributes"
  [6e96]: #x-28MGL-BP-3ATIME-STEP-20FUNCTION-29 "MGL-BP:TIME-STEP FUNCTION"
  [6f82]: #x-28MGL-RESAMPLE-3AFRACTURE-20FUNCTION-29 "MGL-RESAMPLE:FRACTURE FUNCTION"
  [7068]: #x-28MGL-CORE-3AMONITOR-20CLASS-29 "MGL-CORE:MONITOR CLASS"
  [7162]: #x-28MGL-BP-3A--3EACTIVATION-20CLASS-29 "MGL-BP:->ACTIVATION CLASS"
  [71f9]: #x-28MGL-BP-3ASTEP-MONITORS-20-28MGL-PAX-3AACCESSOR-20MGL-BP-3ARNN-29-29 "MGL-BP:STEP-MONITORS (MGL-PAX:ACCESSOR MGL-BP:RNN)"
  [7582]: http://www.lispworks.com/documentation/HyperSpec/Body/f_sinh_.htm "TANH FUNCTION"
  [764b]: #x-28MGL-BP-3ABUILD-RNN-20MGL-PAX-3AMACRO-29 "MGL-BP:BUILD-RNN MGL-PAX:MACRO"
  [765c]: #x-28MGL-DATASET-3AMAP-DATASETS-20FUNCTION-29 "MGL-DATASET:MAP-DATASETS FUNCTION"
  [779d]: #x-28MGL-OPT-3A-40MGL-OPT-ITERATIVE-OPTIMIZER-20MGL-PAX-3ASECTION-29 "Iterative Optimizer"
  [7960]: #x-28MGL-BP-3ASHIFT-20-28MGL-PAX-3AREADER-20MGL-BP-3A--3EBATCH-NORMALIZATION-29-29 "MGL-BP:SHIFT (MGL-PAX:READER MGL-BP:->BATCH-NORMALIZATION)"
  [7a28]: #x-28MGL-BP-3A-40MGL-BP-EXTENSION-API-20MGL-PAX-3ASECTION-29 "Clump API"
  [7bc3]: #x-28MGL-DATASET-3A-40MGL-SAMPLER-20MGL-PAX-3ASECTION-29 "Samplers"
  [7c2f]: #x-28MGL-OPT-3AINITIALIZE-OPTIMIZER-2A-20GENERIC-FUNCTION-29 "MGL-OPT:INITIALIZE-OPTIMIZER* GENERIC-FUNCTION"
  [7e58]: http://www.lispworks.com/documentation/HyperSpec/Body/t_class.htm "CLASS CLASS"
  [7ee3]: #x-28MGL-CORE-3A-40MGL-COUNTER-CLASSES-20MGL-PAX-3ASECTION-29 "Counter classes"
  [80e2]: #x-28MGL-BP-3AVARIANCE-FOR-PREDICTION-20-28MGL-PAX-3AACCESSOR-20MGL-BP-3A--3EGAUSSIAN-RANDOM-29-29 "MGL-BP:VARIANCE-FOR-PREDICTION (MGL-PAX:ACCESSOR MGL-BP:->GAUSSIAN-RANDOM)"
  [8148]: #x-28MGL-CORE-3AREAD-STATE-20FUNCTION-29 "MGL-CORE:READ-STATE FUNCTION"
  [82d8]: #x-28MGL-BP-3AADD-CLUMP-20FUNCTION-29 "MGL-BP:ADD-CLUMP FUNCTION"
  [83e6]: #x-28MGL-CG-3A-40MGL-CG-20MGL-PAX-3ASECTION-29 "Conjugate Gradient"
  [83f9]: #x-28MGL-BP-3A--3ESIGMOID-20CLASS-29 "MGL-BP:->SIGMOID CLASS"
  [85d3]: #x-28MGL-COMMON-3ASIZE-20-28MGL-PAX-3AREADER-20MGL-BP-3ALUMP-29-29 "MGL-COMMON:SIZE (MGL-PAX:READER MGL-BP:LUMP)"
  [86fd]: #x-28MGL-RESAMPLE-3ASAMPLE-FROM-20FUNCTION-29 "MGL-RESAMPLE:SAMPLE-FROM FUNCTION"
  [871e]: #x-28MGL-BP-3A-40MGL-RNN-20MGL-PAX-3ASECTION-29 "Recurrent Neural Nets"
  [8788]: #x-28MGL-BP-3A-40MGL-BP-20MGL-PAX-3ASECTION-29 "Backpropagation Neural Networks"
  [8804]: http://www.lispworks.com/documentation/HyperSpec/Body/f_identi.htm "IDENTITY FUNCTION"
  [8916]: #x-28MGL-COMMON-3ABATCH-SIZE-20-28MGL-PAX-3AACCESSOR-20MGL-CG-3ACG-OPTIMIZER-29-29 "MGL-COMMON:BATCH-SIZE (MGL-PAX:ACCESSOR MGL-CG:CG-OPTIMIZER)"
  [8b55]: #x-28MGL-BP-3AARRANGE-FOR-RENORMALIZING-ACTIVATIONS-20FUNCTION-29 "MGL-BP:ARRANGE-FOR-RENORMALIZING-ACTIVATIONS FUNCTION"
  [8cb8]: #x-28MGL-RESAMPLE-3ASPLIT-STRATIFIED-20FUNCTION-29 "MGL-RESAMPLE:SPLIT-STRATIFIED FUNCTION"
  [8cb9]: http://www.lispworks.com/documentation/HyperSpec/Body/f_concat.htm "CONCATENATE FUNCTION"
  [8da0]: #x-28MGL-OPT-3AITERATIVE-OPTIMIZER-20CLASS-29 "MGL-OPT:ITERATIVE-OPTIMIZER CLASS"
  [8dd7]: #x-28MGL-CORE-3AN-STRIPES-20GENERIC-FUNCTION-29 "MGL-CORE:N-STRIPES GENERIC-FUNCTION"
  [8e53]: #x-28MGL-BP-3AUNFOLDER-20-28MGL-PAX-3AREADER-20MGL-BP-3ARNN-29-29 "MGL-BP:UNFOLDER (MGL-PAX:READER MGL-BP:RNN)"
  [8f37]: #x-28MGL-CORE-3AMONITORS-20GENERIC-FUNCTION-29 "MGL-CORE:MONITORS GENERIC-FUNCTION"
  [9006]: #x-28MGL-OPT-3ATERMINATION-20-28MGL-PAX-3AACCESSOR-20MGL-OPT-3AITERATIVE-OPTIMIZER-29-29 "MGL-OPT:TERMINATION (MGL-PAX:ACCESSOR MGL-OPT:ITERATIVE-OPTIMIZER)"
  [9105]: #x-28MGL-BP-3A-40MGL-BP-ACTIVATIONS-20MGL-PAX-3ASECTION-29 "Activations"
  [911c]: #x-28MGL-CORE-3AMAKE-CLASSIFICATION-ACCURACY-MONITORS-20FUNCTION-29 "MGL-CORE:MAKE-CLASSIFICATION-ACCURACY-MONITORS FUNCTION"
  [9192]: #x-28MGL-3A-40MGL-OVERVIEW-20MGL-PAX-3ASECTION-29 "Overview"
  [91a3]: #x-28MGL-CORE-3AMAX-N-STRIPES-20-28MGL-PAX-3AREADER-20MGL-BP-3ABPN-29-29 "MGL-CORE:MAX-N-STRIPES (MGL-PAX:READER MGL-BP:BPN)"
  [91f3]: #x-28MGL-BP-3A-40MGL-BP-UTILITIES-20MGL-PAX-3ASECTION-29 "Utilities"
  [920f]: http://www.lispworks.com/documentation/HyperSpec/Body/f_symb_5.htm "SYMBOL-VALUE FUNCTION"
  [9358]: http://www.lispworks.com/documentation/HyperSpec/Body/t_mod.htm "MOD TYPE"
  [9385]: #x-28MGL-CORE-3ALABEL-INDEX-DISTRIBUTIONS-20GENERIC-FUNCTION-29 "MGL-CORE:LABEL-INDEX-DISTRIBUTIONS GENERIC-FUNCTION"
  [93a7]: #x-28MGL-BP-3A-40MGL-BP-LOSSES-20MGL-PAX-3ASECTION-29 "Losses"
  [9524]: #x-28MGL-RESAMPLE-3ACROSS-VALIDATE-20FUNCTION-29 "MGL-RESAMPLE:CROSS-VALIDATE FUNCTION"
  [95fe]: #x-28MGL-CORE-3AWRITE-STATE-20FUNCTION-29 "MGL-CORE:WRITE-STATE FUNCTION"
  [9641]: #x-28MGL-BP-3A-40MGL-BP-LUMPS-20MGL-PAX-3ASECTION-29 "Lumps"
  [96d0]: http://www.lispworks.com/documentation/HyperSpec/Body/f_equal.htm "EQUAL FUNCTION"
  [96d0f]: #x-28MGL-NLP-3AFEATURE-ENCODER-20-28MGL-PAX-3AREADER-20MGL-NLP-3ABAG-OF-WORDS-ENCODER-29-29 "MGL-NLP:FEATURE-ENCODER (MGL-PAX:READER MGL-NLP:BAG-OF-WORDS-ENCODER)"
  [9700]: #x-28MGL-BP-3A-40MGL-RNN-TUTORIAL-20MGL-PAX-3ASECTION-29 "`RNN` Tutorial"
  [9715]: #x-28MGL-CORE-3AATTRIBUTED-20CLASS-29 "MGL-CORE:ATTRIBUTED CLASS"
  [9749]: #x-28MGL-CG-3ACG-ARGS-20-28MGL-PAX-3AACCESSOR-20MGL-CG-3ACG-OPTIMIZER-29-29 "MGL-CG:CG-ARGS (MGL-PAX:ACCESSOR MGL-CG:CG-OPTIMIZER)"
  [989a]: #x-28MGL-GD-3A-40MGL-GD-SEGMENTED-GD-OPTIMIZER-20MGL-PAX-3ASECTION-29 "Segmented GD Optimizer"
  [989c]: #x-28MGL-CORE-3AAPPLY-MONITORS-20FUNCTION-29 "MGL-CORE:APPLY-MONITORS FUNCTION"
  [98f9]: http://www.lispworks.com/documentation/HyperSpec/Body/t_list.htm "LIST TYPE"
  [9a5b]: #x-28MGL-OPT-3ASEGMENT-DERIVATIVES-20GENERIC-FUNCTION-29 "MGL-OPT:SEGMENT-DERIVATIVES GENERIC-FUNCTION"
  [9d3a]: #x-28MGL-BP-3A--3ERELU-20CLASS-29 "MGL-BP:->RELU CLASS"
  [9da9]: #x-28MGL-BP-3A--3EBATCH-NORMALIZED-20CLASS-29 "MGL-BP:->BATCH-NORMALIZED CLASS"
  [9de4]: #x-28MGL-BP-3AFNN-20CLASS-29 "MGL-BP:FNN CLASS"
  [a077]: #x-28MGL-CORE-3ACOUNTER-20GENERIC-FUNCTION-29 "MGL-CORE:COUNTER GENERIC-FUNCTION"
  [a1d4]: http://www.lispworks.com/documentation/HyperSpec/Body/f_eq.htm "EQ FUNCTION"
  [a210]: #x-28MGL-OPT-3A-40MGL-OPT-GRADIENT-SINK-20MGL-PAX-3ASECTION-29 "Implementing Gradient Sinks"
  [a39b]: #x-28MGL-RESAMPLE-3A-40MGL-RESAMPLE-20MGL-PAX-3ASECTION-29 "Resampling"
  [a437]: #x-28MGL-COMMON-3AGROUP-SIZE-20-28MGL-PAX-3AREADER-20MGL-BP-3A--3ESOFTMAX-XE-LOSS-29-29 "MGL-COMMON:GROUP-SIZE (MGL-PAX:READER MGL-BP:->SOFTMAX-XE-LOSS)"
  [a4fe]: #x-28MGL-BP-3ACLUMP-20CLASS-29 "MGL-BP:CLUMP CLASS"
  [a81b]: #x-28MGL-BP-3ADERIVATIVES-20GENERIC-FUNCTION-29 "MGL-BP:DERIVATIVES GENERIC-FUNCTION"
  [a884]: #x-28MGL-GD-3A-40MGL-GD-PER-WEIGHT-OPTIMIZATION-20MGL-PAX-3ASECTION-29 "Per-weight Optimization"
  [a896]: #x-28MGL-COMMON-3AWEIGHTS-20-28MGL-PAX-3AREADER-20MGL-BP-3A--3EV-2AM-29-29 "MGL-COMMON:WEIGHTS (MGL-PAX:READER MGL-BP:->V*M)"
  [aa2e]: #x-28MGL-BP-3A-40MGL-BP-STOCHASTICITY-20MGL-PAX-3ASECTION-29 "Stochasticity"
  [aa86]: #x-28MGL-GD-3AVARIANCE-ADJUSTMENT-20-28MGL-PAX-3AREADER-20MGL-BP-3A--3EBATCH-NORMALIZATION-29-29 "MGL-GD:VARIANCE-ADJUSTMENT (MGL-PAX:READER MGL-BP:->BATCH-NORMALIZATION)"
  [aabd]: #x-28MGL-OPT-3AMAP-GRADIENT-SINK-20GENERIC-FUNCTION-29 "MGL-OPT:MAP-GRADIENT-SINK GENERIC-FUNCTION"
  [ad8f]: #x-28MGL-DATASET-3A-2AINFINITELY-EMPTY-DATASET-2A-20VARIABLE-29 "MGL-DATASET:*INFINITELY-EMPTY-DATASET* VARIABLE"
  [ada2]: #x-28MGL-CORE-3A-40MGL-PARAMETERIZED-EXECUTOR-CACHE-20MGL-PAX-3ASECTION-29 "Parameterized Executor Cache"
  [ae3d]: #x-28MGL-OPT-3AMINIMIZE-2A-20GENERIC-FUNCTION-29 "MGL-OPT:MINIMIZE* GENERIC-FUNCTION"
  [aee6]: #x-28MGL-RESAMPLE-3ASAMPLE-STRATIFIED-20FUNCTION-29 "MGL-RESAMPLE:SAMPLE-STRATIFIED FUNCTION"
  [af05]: #x-28MGL-GD-3AMOMENTUM-20-28MGL-PAX-3AACCESSOR-20MGL-GD-3A-3AGD-OPTIMIZER-29-29 "MGL-GD:MOMENTUM (MGL-PAX:ACCESSOR MGL-GD::GD-OPTIMIZER)"
  [af6b]: #x-28MGL-GD-3ACLIP-L2-NORM-20FUNCTION-29 "MGL-GD:CLIP-L2-NORM FUNCTION"
  [afff]: http://www.lispworks.com/documentation/HyperSpec/Body/f_zerop.htm "ZEROP FUNCTION"
  [b01b]: #x-28MGL-CORE-3AMAP-OVER-EXECUTORS-20GENERIC-FUNCTION-29 "MGL-CORE:MAP-OVER-EXECUTORS GENERIC-FUNCTION"
  [b0f3]: #x-28MGL-BP-3ARNN-20CLASS-29 "MGL-BP:RNN CLASS"
  [b186]: #x-28MGL-CORE-3ACROSS-ENTROPY-COUNTER-20CLASS-29 "MGL-CORE:CROSS-ENTROPY-COUNTER CLASS"
  [b5c7]: #x-28MGL-COMMON-3ATARGET-20-28MGL-PAX-3AACCESSOR-20MGL-BP-3A--3ESOFTMAX-XE-LOSS-29-29 "MGL-COMMON:TARGET (MGL-PAX:ACCESSOR MGL-BP:->SOFTMAX-XE-LOSS)"
  [b602]: #x-28MGL-BP-3A--3EACTIVATION-20FUNCTION-29 "MGL-BP:->ACTIVATION FUNCTION"
  [b647]: #x-28MGL-RESAMPLE-3A-40MGL-RESAMPLE-BAGGING-20MGL-PAX-3ASECTION-29 "Bagging"
  [b76f]: #x-28MGL-BP-3A--3EWEIGHT-20CLASS-29 "MGL-BP:->WEIGHT CLASS"
  [b99d]: #x-28MGL-COMMON-3ASIZE-20-28MGL-PAX-3AREADER-20MGL-OPT-3ASEGMENT-SET-29-29 "MGL-COMMON:SIZE (MGL-PAX:READER MGL-OPT:SEGMENT-SET)"
  [b9c1]: http://www.lispworks.com/documentation/HyperSpec/Body/t_seq.htm "SEQUENCE TYPE"
  [ba18]: #x-28MGL-COMMON-3ABATCH-SIZE-20-28MGL-PAX-3AACCESSOR-20MGL-GD-3A-3AGD-OPTIMIZER-29-29 "MGL-COMMON:BATCH-SIZE (MGL-PAX:ACCESSOR MGL-GD::GD-OPTIMIZER)"
  [ba91]: #x-28MGL-RESAMPLE-3ASTRATIFY-20FUNCTION-29 "MGL-RESAMPLE:STRATIFY FUNCTION"
  [bbdf]: #x-28MGL-CORE-3AAPPLY-MONITOR-20GENERIC-FUNCTION-29 "MGL-CORE:APPLY-MONITOR GENERIC-FUNCTION"
  [bd13]: #x-28MGL-GD-3A-40MGL-GD-ADAM-OPTIMIZER-20MGL-PAX-3ASECTION-29 "Adam Optimizer"
  [bdf9]: #x-28MGL-DATASET-3AN-SAMPLES-20-28MGL-PAX-3AREADER-20MGL-DATASET-3AFUNCTION-SAMPLER-29-29 "MGL-DATASET:N-SAMPLES (MGL-PAX:READER MGL-DATASET:FUNCTION-SAMPLER)"
  [be8d]: #x-28MGL-DATASET-3A-40MGL-SAMPLER-FUNCTION-SAMPLER-20MGL-PAX-3ASECTION-29 "Function Sampler"
  [be95]: #x-28MGL-CORE-3A-40MGL-COUNTER-20MGL-PAX-3ASECTION-29 "Counters"
  [bf1a]: http://www.lispworks.com/documentation/HyperSpec/Body/s_multip.htm "MULTIPLE-VALUE-CALL MGL-PAX:MACRO"
  [c102]: #x-28MGL-CORE-3ASAVE-STATE-20FUNCTION-29 "MGL-CORE:SAVE-STATE FUNCTION"
  [c1ac]: #x-28MGL-BP-3ALUMP-20CLASS-29 "MGL-BP:LUMP CLASS"
  [c1ae]: #x-28MGL-BP-3AFORWARD-20GENERIC-FUNCTION-29 "MGL-BP:FORWARD GENERIC-FUNCTION"
  [c1d9]: http://www.lispworks.com/documentation/HyperSpec/Body/f_numera.htm "NUMERATOR FUNCTION"
  [c40e]: #x-28MGL-GD-3A-40MGL-GD-UTILITIES-20MGL-PAX-3ASECTION-29 "Utilities"
  [c469]: #x-28MGL-BP-3A--3EBATCH-NORMALIZATION-20CLASS-29 "MGL-BP:->BATCH-NORMALIZATION CLASS"
  [c573]: #x-28MGL-CORE-3A-40MGL-CLASSIFICATION-MONITOR-20MGL-PAX-3ASECTION-29 "Classification Monitors"
  [c58b]: #x-28MGL-OPT-3A-40MGL-OPT-GRADIENT-SOURCE-20MGL-PAX-3ASECTION-29 "Implementing Gradient Sources"
  [c701]: #x-28MGL-CORE-3A-40MGL-MONITOR-20MGL-PAX-3ASECTION-29 "Monitors"
  [c74a]: #x-28MGL-OPT-3A-40MGL-OPT-20MGL-PAX-3ASECTION-29 "Gradient Based Optimization"
  [c7fa]: #x-28MGL-BP-3ARENORMALIZE-ACTIVATIONS-20FUNCTION-29 "MGL-BP:RENORMALIZE-ACTIVATIONS FUNCTION"
  [c8db]: #x-28MGL-CORE-3A-40MGL-FEATURES-20MGL-PAX-3ASECTION-29 "Features"
  [c918]: #x-28MGL-COMMON-3ABATCH-SIZE-20-28MGL-PAX-3AREADER-20MGL-BP-3A--3EBATCH-NORMALIZATION-29-29 "MGL-COMMON:BATCH-SIZE (MGL-PAX:READER MGL-BP:->BATCH-NORMALIZATION)"
  [c9d1]: #x-28MGL-COMMON-3ASCALE-20-28MGL-PAX-3AREADER-20MGL-BP-3A--3EBATCH-NORMALIZATION-29-29 "MGL-COMMON:SCALE (MGL-PAX:READER MGL-BP:->BATCH-NORMALIZATION)"
  [ca09]: #x-28MGL-OPT-3ARESET-OPTIMIZATION-MONITORS-20GENERIC-FUNCTION-29 "MGL-OPT:RESET-OPTIMIZATION-MONITORS GENERIC-FUNCTION"
  [caec]: #x-28MGL-CORE-3ALABEL-INDEX-DISTRIBUTION-20GENERIC-FUNCTION-29 "MGL-CORE:LABEL-INDEX-DISTRIBUTION GENERIC-FUNCTION"
  [cbb4]: #x-28MGL-NLP-3ABAG-OF-WORDS-ENCODER-20CLASS-29 "MGL-NLP:BAG-OF-WORDS-ENCODER CLASS"
  [cc1c]: #x-28MGL-COMMON-3ANODES-20GENERIC-FUNCTION-29 "MGL-COMMON:NODES GENERIC-FUNCTION"
  [cc37]: #x-28MGL-CORE-3AATTRIBUTES-20-28MGL-PAX-3AACCESSOR-20MGL-CORE-3AATTRIBUTED-29-29 "MGL-CORE:ATTRIBUTES (MGL-PAX:ACCESSOR MGL-CORE:ATTRIBUTED)"
  [cc80]: #x-28MGL-CORE-3ALABEL-INDEX-20GENERIC-FUNCTION-29 "MGL-CORE:LABEL-INDEX GENERIC-FUNCTION"
  [cd3b]: #x-28MGL-CORE-3A-40MGL-MEASURER-20MGL-PAX-3ASECTION-29 "Measurers"
  [d0e3]: #x-28MGL-BP-3A-40MGL-RNN-TIME-WARP-20MGL-PAX-3ASECTION-29 "Time Warp"
  [d10a]: #x-28MGL-CG-3AON-CG-BATCH-DONE-20-28MGL-PAX-3AACCESSOR-20MGL-CG-3ACG-OPTIMIZER-29-29 "MGL-CG:ON-CG-BATCH-DONE (MGL-PAX:ACCESSOR MGL-CG:CG-OPTIMIZER)"
  [d1e0]: #x-28MGL-BP-3A-40MGL-BPN-20MGL-PAX-3ASECTION-29 "`BPN`s"
  [d397]: http://www.lispworks.com/documentation/HyperSpec/Body/f_wr_to_.htm "PRINC-TO-STRING FUNCTION"
  [d3b2]: #x-28MGL-CORE-3APARAMETERIZED-EXECUTOR-CACHE-MIXIN-20CLASS-29 "MGL-CORE:PARAMETERIZED-EXECUTOR-CACHE-MIXIN CLASS"
  [d443]: #x-28MGL-NLP-3AENCODED-FEATURE-TYPE-20-28MGL-PAX-3AREADER-20MGL-NLP-3ABAG-OF-WORDS-ENCODER-29-29 "MGL-NLP:ENCODED-FEATURE-TYPE (MGL-PAX:READER MGL-NLP:BAG-OF-WORDS-ENCODER)"
  [d699]: #x-28MGL-COMMON-3ANODES-20-28MGL-PAX-3AREADER-20MGL-BP-3ALUMP-29-29 "MGL-COMMON:NODES (MGL-PAX:READER MGL-BP:LUMP)"
  [d6e0]: #x-28MGL-BP-3AWARP-START-20-28MGL-PAX-3AREADER-20MGL-BP-3ARNN-29-29 "MGL-BP:WARP-START (MGL-PAX:READER MGL-BP:RNN)"
  [d94e]: #x-28MGL-GD-3ABATCH-GD-OPTIMIZER-20CLASS-29 "MGL-GD:BATCH-GD-OPTIMIZER CLASS"
  [d96a]: #x-28MGL-BP-3AMEAN-20-28MGL-PAX-3AACCESSOR-20MGL-BP-3A--3EGAUSSIAN-RANDOM-29-29 "MGL-BP:MEAN (MGL-PAX:ACCESSOR MGL-BP:->GAUSSIAN-RANDOM)"
  [dbc4]: #x-28MGL-BP-3A--3EV-2AM-20CLASS-29 "MGL-BP:->V*M CLASS"
  [dbc7]: #x-28-22mgl-22-20ASDF-2FSYSTEM-3ASYSTEM-29 '"mgl" ASDF/SYSTEM:SYSTEM'
  [dd95]: #x-28MGL-OPT-3AINITIALIZE-GRADIENT-SOURCE-2A-20GENERIC-FUNCTION-29 "MGL-OPT:INITIALIZE-GRADIENT-SOURCE* GENERIC-FUNCTION"
  [e0e6]: #x-28MGL-GD-3AADAM-OPTIMIZER-20CLASS-29 "MGL-GD:ADAM-OPTIMIZER CLASS"
  [e2a3]: http://www.lispworks.com/documentation/HyperSpec/Body/t_vector.htm "VECTOR TYPE"
  [e46f]: #x-28MGL-CORE-3AMAKE-CROSS-ENTROPY-MONITORS-2A-20GENERIC-FUNCTION-29 "MGL-CORE:MAKE-CROSS-ENTROPY-MONITORS* GENERIC-FUNCTION"
  [e4b0]: http://www.lispworks.com/documentation/HyperSpec/Body/f_ensu_1.htm "ENSURE-DIRECTORIES-EXIST FUNCTION"
  [e4f9]: #x-28MGL-OPT-3ARESET-OPTIMIZATION-MONITORS-20-28METHOD-20NIL-20-28MGL-OPT-3AITERATIVE-OPTIMIZER-20T-29-29-29 "MGL-OPT:RESET-OPTIMIZATION-MONITORS (METHOD NIL (MGL-OPT:ITERATIVE-OPTIMIZER T))"
  [e50c]: #x-28MGL-CORE-3AMONITOR-MODEL-RESULTS-20FUNCTION-29 "MGL-CORE:MONITOR-MODEL-RESULTS FUNCTION"
  [e668]: #x-28MGL-CORE-3A-40MGL-MONITORING-20MGL-PAX-3ASECTION-29 "Monitoring"
  [e746]: #x-28MGL-OPT-3A-40MGL-OPT-COST-20MGL-PAX-3ASECTION-29 "Cost Function"
  [e7ea]: #x-28MGL-3A-40MGL-DEPENDENCIES-20MGL-PAX-3ASECTION-29 "Dependencies"
  [e7f6]: #x-28MGL-BP-3ADROPOUT-20-28MGL-PAX-3AACCESSOR-20MGL-BP-3A--3EINPUT-29-29 "MGL-BP:DROPOUT (MGL-PAX:ACCESSOR MGL-BP:->INPUT)"
  [e8d2]: #x-28MGL-BP-3A--3ESQUARED-DIFFERENCE-20CLASS-29 "MGL-BP:->SQUARED-DIFFERENCE CLASS"
  [eaf1]: #x-28MGL-BP-3ABATCH-NORMALIZATION-20-28MGL-PAX-3AREADER-20MGL-BP-3A--3EBATCH-NORMALIZED-29-29 "MGL-BP:BATCH-NORMALIZATION (MGL-PAX:READER MGL-BP:->BATCH-NORMALIZED)"
  [eafc]: http://www.lispworks.com/documentation/HyperSpec/Body/f_pr_obj.htm "PRINT-OBJECT GENERIC-FUNCTION"
  [eb05]: #x-28MGL-CORE-3AMEASURER-20-28MGL-PAX-3AREADER-20MGL-CORE-3AMONITOR-29-29 "MGL-CORE:MEASURER (MGL-PAX:READER MGL-CORE:MONITOR)"
  [ebd4]: #x-28MGL-OPT-3AON-OPTIMIZATION-STARTED-20-28MGL-PAX-3AACCESSOR-20MGL-OPT-3AITERATIVE-OPTIMIZER-29-29 "MGL-OPT:ON-OPTIMIZATION-STARTED (MGL-PAX:ACCESSOR MGL-OPT:ITERATIVE-OPTIMIZER)"
  [ed4f]: #x-28MGL-BP-3A-2AWARP-TIME-2A-20VARIABLE-29 "MGL-BP:*WARP-TIME* VARIABLE"
  [edcf]: #x-28MGL-BP-3A--3ESUM-20CLASS-29 "MGL-BP:->SUM CLASS"
  [ee97]: #x-28MGL-CG-3ACG-OPTIMIZER-20CLASS-29 "MGL-CG:CG-OPTIMIZER CLASS"
  [f00d]: #x-28MGL-OPT-3ASEGMENTS-20GENERIC-FUNCTION-29 "MGL-OPT:SEGMENTS GENERIC-FUNCTION"
  [f17b]: #x-28MGL-RESAMPLE-3A-40MGL-RESAMPLE-CROSS-VALIDATION-20MGL-PAX-3ASECTION-29 "Cross-validation"
  [f257]: #x-28MGL-CORE-3A-40MGL-CORE-20MGL-PAX-3ASECTION-29 "Core"
  [f3bc]: #x-28MGL-COMMON-3ANAME-20-28MGL-PAX-3AREADER-20MGL-DATASET-3AFUNCTION-SAMPLER-29-29 "MGL-COMMON:NAME (MGL-PAX:READER MGL-DATASET:FUNCTION-SAMPLER)"
  [f466]: http://www.lispworks.com/documentation/HyperSpec/Body/f_length.htm "LENGTH FUNCTION"
  [f47b]: http://www.lispworks.com/documentation/HyperSpec/Body/f_wr_pr.htm "PRINC FUNCTION"
  [f491]: #x-28MGL-COMMON-3AFN-20-28MGL-PAX-3AREADER-20MGL-DIFFUN-3ADIFFUN-29-29 "MGL-COMMON:FN (MGL-PAX:READER MGL-DIFFUN:DIFFUN)"
  [f54e]: #x-28MGL-BP-3A--3EINPUT-20CLASS-29 "MGL-BP:->INPUT CLASS"
  [f573]: #x-28MGL-BP-3ACUDA-WINDOW-START-TIME-20-28MGL-PAX-3AACCESSOR-20MGL-BP-3ARNN-29-29 "MGL-BP:CUDA-WINDOW-START-TIME (MGL-PAX:ACCESSOR MGL-BP:RNN)"
  [f652]: #x-28MGL-BP-3A--3EMAX-20CLASS-29 "MGL-BP:->MAX CLASS"
  [f6ae]: #x-28MGL-GD-3ANORMALIZED-BATCH-GD-OPTIMIZER-20CLASS-29 "MGL-GD:NORMALIZED-BATCH-GD-OPTIMIZER CLASS"
  [f790]: #x-28MGL-RESAMPLE-3A-40MGL-RESAMPLE-PARTITIONS-20MGL-PAX-3ASECTION-29 "Partitions"
  [f7aa]: #x-28MGL-3A-40MGL-INTRODUCTION-20MGL-PAX-3ASECTION-29 "Introduction"
  [f7c1]: #x-28MGL-BP-3ACLUMPS-20-28MGL-PAX-3AREADER-20MGL-BP-3ABPN-29-29 "MGL-BP:CLUMPS (MGL-PAX:READER MGL-BP:BPN)"
  [f956]: #x-28MGL-DATASET-3ASAMPLE-20GENERIC-FUNCTION-29 "MGL-DATASET:SAMPLE GENERIC-FUNCTION"
  [f98e]: #x-28MGL-CORE-3ADO-EXECUTORS-20MGL-PAX-3AMACRO-29 "MGL-CORE:DO-EXECUTORS MGL-PAX:MACRO"
  [faaa]: #x-28MGL-CORE-3ADO-BATCHES-FOR-MODEL-20MGL-PAX-3AMACRO-29 "MGL-CORE:DO-BATCHES-FOR-MODEL MGL-PAX:MACRO"
  [fedd]: #x-28MGL-CORE-3AENCODE-20GENERIC-FUNCTION-29 "MGL-CORE:ENCODE GENERIC-FUNCTION"
  [ff5a]: #x-28MGL-BP-3ALAG-20FUNCTION-29 "MGL-BP:LAG FUNCTION"
  [ff82]: #x-28MGL-CORE-3A-40MGL-MODEL-STRIPE-20MGL-PAX-3ASECTION-29 "Batch Processing"

* * *
###### \[generated by [MGL-PAX](https://github.com/melisgl/mgl-pax)\]
