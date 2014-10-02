# MGL Manual

###### \[in package MGL\]
## mgl ASDF System Details

- Version: 0.0.8
- Description: MGL is a machine learning library for backpropagation
  neural networks, boltzmann machines, gaussian processes and more.
- Licence: MIT, see COPYING.
- Author: Gábor Melis
- Mailto: [mega@retes.hu](mailto:mega@retes.hu)
- Homepage: [http://quotenil.com](http://quotenil.com)

## Overview

MGL is a Common Lisp machine learning library by [Gábor
Melis](http://quotenil.com) with some parts originally contributed
by Ravenpack International. It implements:

- Backpropagation networks (BPN)- Dropout

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


### Features

In general, the focus is on power and performance not on ease of use.
For example, it's possible to:

- control the order of presentation of training examples,

- vary learning rate depending on time, state of the trainer object,

- track all kinds of statistics during training,
etc.

Perhaps one day there will be a cookie cutter interface with
restricted functionality if a reasonable compromise is found between
power and utility.

### Dependencies

MGL used to rely on [LLA](https://github.com/tpapp/lla) to
interface to BLAS and LAPACK. That's mostly history by now, but
configuration of foreign libraries is still done via LLA. See the
README in LLA on how to set things up. Note that these days OpenBLAS
is easier to set up and just as fast as ATLAS.

[CL-CUDA](https://github.com/takagi/cl-cuda) is a dependency for
which the NVIDIA CUDA Toolkit needs to be installed, but MGL is
fully functional even if there is no cuda capable gpu installed. See
the MGL-MAT:WITH-CUDA\* macro for how to use it.

### Tests

Run the built in tests 
with:

    (ASDF:OOS 'ASDF:TEST-OP '#:MGL)

Note, that most of the tests are rather stochastic and can fail once
in a while.

### Bundled Software

With [MGL-PAX](https://github.com/melisgl/mgl-pax) and
[MGL-MAT](https://github.com/melisgl/mgl-mat) libraries split off
there remains only a single library bundled with MGL which does
not depend on the rest of MGL:

- MGL-GNUPLOT, a plotting library.

There is also MGL-VISUALS which does depend on MGL.

## Basic Concepts

MODEL, training set, test set, validation set,
sample/instance/example.

## Datasets

Ultimately machine learning is about creating models of some
domain. The observations in modelled domain are called *instances*.
Sets of instances are called *datasets*. Datasets are used when
fitting a model or when making predictions.

Implementationally speaking, an instance is typically represented by
a set of numbers which is called *feature vector*. A dataset is a
CL:SEQUENCE of such instances or a GENERATOR object that produces
instances.

## Resampling

###### \[in package MGL-RESAMPLE\]
The focus of this package is on resampling methods such as
cross-validation and bagging which can be used for model evaluation,
model selection, and also as a simple form of ensembling. Data
partitioning and sampling functions are also provided because they
tend to be used together with resampling.

### Partitions

The following functions partition a dataset (currently only
SEQUENCEs are supported) into a number of partitions. For each
element in the original dataset there is exactly one partition that
contains it.

- [function] FRACTURE FRACTIONS SEQ &KEY WEIGHT

    Partition SEQ into a number of subsequences. FRACTIONS is either a
    positive integer or a list of non-negative real numbers. WEIGHT is
    NIL or a function that returns a non-negative real number when
    called with an element from SEQ. If FRACTIONS is a positive integer
    then return a list of that many subsequences with equal sum of
    weights bar rounding errors, else partition SEQ into subsequences,
    where the sum of weights of subsequence I is proportional to element
    I of FRACTIONS. If WEIGHT is NIL, then it's element is assumed to
    have the same weight.
    
    To split into 5 sequences:
    
        (fracture 5 '(0 1 2 3 4 5 6 7 8 9))
        => ((0 1) (2 3) (4 5) (6 7) (8 9))
    
    To split into two sequences whose lengths are proportional to 2 and
    3:
    
        (fracture '(2 3) '(0 1 2 3 4 5 6 7 8 9))
        => ((0 1 2 3) (4 5 6 7 8 9))


- [function] STRATIFY SEQ &KEY (KEY #'IDENTITY) (TEST #'EQL)

    Return the list of strata of SEQ. SEQ is sequence of elements for
    which the function KEY returns the class they belong to. Such
    classes are opaque objects compared for equality with TEST. A
    stratum is a sequence of elements with the same (under TEST) KEY.
    
        (stratify '(0 1 2 3 4 5 6 7 8 9) :key #'evenp)
        => ((0 2 4 6 8) (1 3 5 7 9))


- [function] FRACTURE-STRATIFIED FRACTIONS SEQ &KEY (KEY #'IDENTITY) (TEST #'EQL) WEIGHT

    Similar to FRACTURE, but also makes sure that keys are evenly
    distributed among the partitions (see STRATIFY). It can be useful
    for classification tasks to partition the data set while keeping the
    distribution of classes the same.
    
    Note that the sets returned are not in random order. In fact, they
    are sorted internally by KEY.
    
    For example, to make two splits with approximately the same number
    of even and odd numbers:
    
        (fracture-stratified 2 '(0 1 2 3 4 5 6 7 8 9) :key #'evenp)
        => ((0 2 1 3) (4 6 8 5 7 9))


### Cross-validation

- [function] CROSS-VALIDATE DATA FN &KEY (N-FOLDS 5) (FOLDS (ALEXANDRIA.0.DEV:IOTA N-FOLDS)) (SPLIT-FN #'SPLIT-FOLD/MOD) PASS-FOLD

    Map FN over the FOLDS of DATA split with SPLIT-FN and collect the
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
    the trained model and/or its score on TEST. Also, sometimes one may
    want to do only some of the folds and remember which ones they were:
    
        (cross-validate '(0 1 2 3 4)
                        (lambda (fold test training)
                         (list :fold fold test training))
                        :folds '(2 3)
                        :pass-fold t)
        => ((:fold 2 (2) (0 1 3 4))
            (:fold 3 (3) (0 1 2 4)))
    
    Finally, the way the data is split can be customized. By default
    SPLIT-FOLD/MOD is called with the arguments DATA, the fold (from
    among FOLDS) and N-FOLDS. SPLIT-FOLD/MOD returns two values which
    are then passed on to FN. One can use SPLIT-FOLD/CONT or
    SPLIT-STRATIFIED or any other function that works with these
    arguments. The only real constraint is that FN has to take as many
    arguments (plus the fold argument if PASS-FOLD) as SPLIT-FN
    returns.

- [function] SPLIT-FOLD/MOD SEQ FOLD N-FOLDS

    Partition SEQ into two sequences: one with elements of SEQ with
    indices whose remainder is FOLD when divided with N-FOLDS, and a
    second one with the rest. The second one is the larger set. The
    order of elements remains stable. This function is suitable as the
    SPLIT-FN argument of CROSS-VALIDATE.

- [function] SPLIT-FOLD/CONT SEQ FOLD N-FOLDS

    Imagine dividing SEQ into N-FOLDS subsequences of the same
    size (bar rounding). Return the subsequence of index FOLD as the
    first value and the all the other subsequences concatenated into one
    as the second value. The order of elements remains stable. This
    function is suitable as the SPLIT-FN argument of CROSS-VALIDATE.

- [function] SPLIT-STRATIFIED SEQ FOLD N-FOLDS &KEY (KEY #'IDENTITY) (TEST #'EQL) WEIGHT

    Split SEQ into N-FOLDS partitions (as in FRACTURE-STRATIFIED).
    Return the partition of index FOLD as the first value, and the
    concatenation of the rest as the second value. This function is
    suitable as the SPLIT-FN argument of CROSS-VALIDATE (mostly likely
    as a closure with KEY, TEST, WEIGHT bound).

### Bagging

- [function] BAG SEQ FN &KEY (RATIO 1) N WEIGHT (REPLACEMENT T) KEY (TEST #'EQL) (RANDOM-STATE *RANDOM-STATE*)

    Sample from SEQ with SAMPLE-FROM (passing RATIO, WEIGHT,
    REPLACEMENT), or SAMPLE-STRATIFIED if KEY is not NIL. Call FN with
    the sample. If N is NIL then keep repeating this until FN performs a
    non-local exit. Else N must be a non-negative integer, N iterations
    will be performed, the primary values returned by FN collected into
    a list and returned. See SAMPLE-FROM and SAMPLE-STRATIFIED for
    examples.

- [function] SAMPLE-FROM RATIO SEQ &KEY WEIGHT REPLACEMENT (RANDOM-STATE *RANDOM-STATE*)

    Return a sequence constructed by sampling with or without
    REPLACEMENT from SEQ. The sum of weights in the result sequence will
    approximately be the sum of weights of SEQ times RATIO. If WEIGHT is
    NIL then elements are assumed to have equal weights, else WEIGHT
    should return a non-negative real number when called with an element
    of SEQ.
    
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


- [function] SAMPLE-STRATIFIED RATIO SEQ &KEY WEIGHT REPLACEMENT (KEY #'IDENTITY) (TEST #'EQL) (RANDOM-STATE *RANDOM-STATE*)

    Like SAMPLE-FROM but makes sure that the weighted proportion of
    classes in the result is approximately the same as the proportion in
    SEQ. See STRATIFY for the description of KEY and TEST.

### CV Bagging

- [function] BAG-CV DATA FN &KEY N (N-FOLDS 5) (FOLDS (ALEXANDRIA.0.DEV:IOTA N-FOLDS)) (SPLIT-FN #'SPLIT-FOLD/MOD) PASS-FOLD (RANDOM-STATE *RANDOM-STATE*)

    Perform cross-validation on different shuffles of DATA N times and
    collect the results. Since CROSS-VALIDATE collects the return values
    of FN, the return value of this function is a list of lists of FN
    results. If N is NIL, don't collect anything just keep doing
    repeated CVs until FN performs an non-local exit.
    
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

### Miscellaneous Operations

- [function] SPREAD-STRATA SEQ &KEY (KEY #'IDENTITY) (TEST #'EQL)

    Return a sequence that's a reordering of SEQ such that elements
    belonging to different strata (under KEY and TEST, see STRATIFY) are
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


- [function] ZIP-EVENLY SEQS &KEY RESULT-TYPE

    Make a single sequence out of the sequences in SEQS so that in the
    returned sequence indices of elements belonging to the same source
    sequence are spread evenly across the whole range. The result is a
    list is RESULT-TYPE is LIST, it's a vector if RESULT-TYPE is VECTOR.
    If RESULT-TYPE is NIL, then it's determined by the type of the first
    sequence in SEQS.
    
        (zip-evenly '((0 2 4) (1 3)))
        => (0 1 2 3 4)


## Optimization


## Backprogation Neural Networks


## Boltzmann Machines


## Gaussian Processes


* * *
###### \[generated by [MGL-PAX](https://github.com/melisgl/mgl-pax)\]
