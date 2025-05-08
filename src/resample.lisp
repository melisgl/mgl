(in-package :mgl-resample)

(defsection @mgl-resample (:title "Resampling")
  "The focus of this package is on resampling methods such as
  cross-validation and bagging which can be used for model evaluation,
  model selection, and also as a simple form of ensembling. Data
  partitioning and sampling functions are also provided because they
  tend to be used together with resampling."
  (@mgl-resample-shuffling section)
  (@mgl-resample-partitions section)
  (@mgl-resample-cross-validation section)
  (@mgl-resample-bagging section)
  (@mgl-resample-cv-bagging section)
  (@mgl-resample-misc section))


(defsection @mgl-resample-shuffling (:title "Shuffling")
  (shuffle function)
  (shuffle! function))

(defun shuffle (seq)
  "Copy of SEQ and shuffle it using Fisher-Yates algorithm."
  (if (listp seq)
      (coerce (shuffle-vector! (coerce seq 'vector)) 'list)
      (shuffle-vector! (copy-seq seq))))

(defun shuffle! (seq)
  "Shuffle SEQ using Fisher-Yates algorithm."
  (if (listp seq)
      (coerce (shuffle-vector! (coerce seq 'vector)) 'list)
      (shuffle-vector! seq)))

(defun shuffle-vector! (vector)
  (loop for idx downfrom (1- (length vector)) to 1
        for other = (random (1+ idx))
        do (unless (= idx other)
             (rotatef (aref vector idx) (aref vector other))))
  vector)


(defsection @mgl-resample-partitions (:title "Partitions")
  "The following functions partition a dataset (currently only
  SEQUENCEs are supported) into a number of partitions. For each
  element in the original dataset there is exactly one partition that
  contains it."
  (fracture function)
  (stratify function)
  (fracture-stratified function))

(defun fracture (fractions seq &key weight)
  "Partition SEQ into a number of subsequences. FRACTIONS is either a
  positive integer or a list of non-negative real numbers. WEIGHT is
  NIL or a function that returns a non-negative real number when
  called with an element from SEQ. If FRACTIONS is a positive integer
  then return a list of that many subsequences with equal sum of
  weights bar rounding errors, else partition SEQ into subsequences,
  where the sum of weights of subsequence I is proportional to element
  I of FRACTIONS. If WEIGHT is NIL, then it's element is assumed to
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
  ```"
  (let* ((length (length seq))
         (weights-total (if weight (reduce #'+ seq :key weight) length))
         (fractions (if (numberp fractions)
                        (make-list fractions :initial-element 1)
                        fractions)))
    (let ((fractions-total (reduce #'+ fractions))
          (n-fractions (length fractions))
          (start 0)
          (weights-sum 0))
      (loop for fraction-index below n-fractions
            for fraction in fractions
            for fractions-sum = fraction then (+ fractions-sum fraction)
            for weights-sum-limit = (* weights-total (/ fractions-sum
                                                        fractions-total))
            collect
            (subseq seq start
                    (if (= fraction-index (1- n-fractions))
                        ;; The last split absorbs rounding errors.
                        length
                        (multiple-value-setq (start weights-sum)
                          (find-enough-weights seq start weight weights-sum
                                               weights-sum-limit))))))))

(defun find-enough-weights (seq start weight weights-sum weights-sum-limit)
  (let ((i start)
        (weights-sum weights-sum))
    (map nil (lambda (x)
               (let ((w (if weight (funcall weight x) 1)))
                 (when (<= weights-sum-limit (+ weights-sum w))
                   (return-from find-enough-weights
                     (if (< (abs (- weights-sum-limit (+ weights-sum w)))
                            (abs (- weights-sum-limit weights-sum)))
                         (values (1+ i) (+ weights-sum w))
                         (values i weights-sum))))
                 (incf weights-sum w)
                 (incf i)))
         (subseq seq start))))

(defun stratify (seq &key (key #'identity) (test #'eql))
  "Return the list of strata of SEQ. SEQ is a sequence of elements for
  which the function KEY returns the class they belong to. Such
  classes are opaque objects compared for equality with TEST. A
  stratum is a sequence of elements with the same (under TEST) KEY.

  ```cl-transcript
  (stratify '(0 1 2 3 4 5 6 7 8 9) :key #'evenp)
  => ((0 2 4 6 8) (1 3 5 7 9))
  ```"
  (if (member test (list 'eq #'eq 'eql #'eql 'equal #'equal))
      (let ((h (make-hash-table :test test))
            (classes ()))
        (map nil (lambda (x)
                   (let ((class (funcall key x)))
                     (pushnew class classes :test test)
                     (push x (gethash class h))))
             seq)
        ;; Get the elements of the hash table in a reproducible
        ;; order.
        (loop for class in (reverse classes)
              collect (nreverse (gethash class h))))
      (let ((keys (collect-distinct seq :key key :test test)))
        (if (zerop (length keys))
            ()
            (loop for k in keys
                  collect
                  (let ((elements
                          (coerce
                           (remove-if-not (lambda (x)
                                            (funcall test k
                                                     (funcall key x)))
                                          seq)
                           'vector)))
                    elements))))))

(defun collect-distinct (seq &key (key #'identity) (test #'eql))
  (let ((result ()))
    (map nil
         (lambda (x)
           (pushnew (funcall key x) result :test test))
         seq)
    (nreverse result)))

(defun fracture-stratified (fractions seq &key (key #'identity)
                            (test #'eql) weight)
  "Similar to FRACTURE, but also makes sure that keys are evenly
  distributed among the partitions (see STRATIFY). It can be useful
  for classification tasks to partition the data set while keeping the
  distribution of classes the same.

  Note that the sets returned are not in random order. In fact, they
  are sorted internally by KEY.

  For example, to make two splits with approximately the same number
  of even and odd numbers:

  ```cl-transcript
  (fracture-stratified 2 '(0 1 2 3 4 5 6 7 8 9) :key #'evenp)
  => ((0 2 1 3) (4 6 8 5 7 9))
  ```"
  (let ((stratum-partitions
          (mapcar (lambda (elements)
                    (fracture fractions elements :weight weight))
                  (stratify seq :key key :test test))))
    (loop for i below (length (elt stratum-partitions 0))
          collect (apply #'concatenate
                         (if (listp seq)
                             'list
                             `(vector ,(array-element-type seq)))
                         (mapcar (lambda (splits)
                                   (elt splits i))
                                 stratum-partitions)))))


(defsection @mgl-resample-cross-validation (:title "Cross-validation")
  (cross-validate function)
  (split-fold/mod function)
  (split-fold/cont function)
  (split-stratified function))

(defun cross-validate (data fn &key (n-folds 5)
                       (folds (alexandria:iota n-folds))
                       (split-fn #'split-fold/mod) pass-fold)
  "Map FN over the FOLDS of DATA split with SPLIT-FN and collect the
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
  the trained model and/or its score on TEST. Also, sometimes one may
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
  SPLIT-FOLD/MOD is called with the arguments DATA, the fold (from
  among FOLDS) and N-FOLDS. SPLIT-FOLD/MOD returns two values which
  are then passed on to FN. One can use SPLIT-FOLD/CONT or
  SPLIT-STRATIFIED or any other function that works with these
  arguments. The only real constraint is that FN has to take as many
  arguments (plus the fold argument if PASS-FOLD) as SPLIT-FN
  returns."
  (assert (every (lambda (fold)
                   (and (<= 0 fold) (< fold n-folds)))
                 folds))
  (map 'list (lambda (fold)
               (if pass-fold
                   (multiple-value-call fn fold
                     (funcall split-fn data fold n-folds))
                   (multiple-value-call fn
                     (funcall split-fn data fold n-folds))))
       folds))

(defun split-fold/mod (seq fold n-folds)
  "Partition SEQ into two sequences: one with elements of SEQ with
  indices whose remainder is FOLD when divided with N-FOLDS, and a
  second one with the rest. The second one is the larger set. The
  order of elements remains stable. This function is suitable as the
  SPLIT-FN argument of CROSS-VALIDATE."
  (assert (<= 0 fold (1- n-folds)))
  (split-by-index seq (lambda (i)
                        (= fold (mod i n-folds)))))

(defun split-fold/cont (seq fold n-folds)
  "Imagine dividing SEQ into N-FOLDS subsequences of the same
  size (bar rounding). Return the subsequence of index FOLD as the
  first value and the all the other subsequences concatenated into one
  as the second value. The order of elements remains stable. This
  function is suitable as the SPLIT-FN argument of CROSS-VALIDATE."
  (assert (<= 0 fold (1- n-folds)))
  (let ((fold-length (/ (length seq) n-folds)))
    (split-by-index seq (lambda (i)
                          (= fold (floor i fold-length))))))

(defun split-by-index (seq pred)
  "Partition SEQ into two sequences: one with the elements with
  indices for which PRED returns true, one with the rest. The order of
  elements remains is stable within the two splits."
  (let ((true-seq ())
        (false-seq ())
        (i 0))
    (map nil (lambda (x)
               (if (funcall pred i)
                   (push x true-seq)
                   (push x false-seq))
               (incf i))
         seq)
    (let ((true-seq (nreverse true-seq))
          (false-seq (nreverse false-seq)))
      (if (listp seq)
          (values true-seq false-seq)
          (values (coerce true-seq 'vector) (coerce false-seq 'vector))))))

(defun split-stratified (seq fold n-folds &key (key #'identity) (test #'eql)
                         weight)
  "Split SEQ into N-FOLDS partitions (as in FRACTURE-STRATIFIED).
  Return the partition of index FOLD as the first value, and the
  concatenation of the rest as the second value. This function is
  suitable as the SPLIT-FN argument of CROSS-VALIDATE (mostly likely
  as a closure with KEY, TEST, WEIGHT bound)."
  (let ((strata (fracture-stratified n-folds seq :key key :test test
                                     :weight weight)))
    (values (elt strata fold)
            (apply #'concatenate (if (listp seq) 'list 'vector)
                   (append (subseq strata 0 fold)
                           (subseq strata (1+ fold)))))))


(defsection @mgl-resample-bagging (:title "Bagging")
  (bag function)
  (sample-from function)
  (sample-stratified function))

(defun bag (seq fn &key (ratio 1) n weight (replacement t) key (test #'eql)
            (random-state *random-state*))
  "Sample from SEQ with SAMPLE-FROM (passing RATIO, WEIGHT,
  REPLACEMENT), or SAMPLE-STRATIFIED if KEY is not NIL. Call FN with
  the sample. If N is NIL then keep repeating this until FN performs a
  non-local exit. Else N must be a non-negative integer, N iterations
  will be performed, the primary values returned by FN collected into
  a list and returned. See SAMPLE-FROM and SAMPLE-STRATIFIED for
  examples."
  (flet ((foo ()
           (if key
               (funcall fn (sample-stratified ratio seq :weight weight
                                              :replacement replacement
                                              :key key :test test
                                              :random-state random-state))
               (funcall fn (sample-from ratio seq :weight weight
                                        :replacement replacement
                                        :random-state random-state)))))
    (if n
        (loop repeat n collect (foo))
        (loop (foo)))))

(defun sample-from (ratio seq &key weight replacement
                    (random-state *random-state*))
  "Return a sequence constructed by sampling with or without
  REPLACEMENT from SEQ. The sum of weights in the result sequence will
  approximately be the sum of weights of SEQ times RATIO. If WEIGHT is
  NIL then elements are assumed to have equal weights, else WEIGHT
  should return a non-negative real number when called with an element
  of SEQ.

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
  ```"
  (if replacement
      (sample-with-replacement* ratio seq :weight weight
                                :random-state random-state)
      (sample-without-replacement* ratio seq :weight weight
                                   :random-state random-state)))

(defun sample-with-replacement* (ratio seq &key weight
                                 (random-state *random-state*))
  (let* ((seq* (coerce seq 'vector))
         (n (length seq*))
         (sum-weights (if weight
                          (loop for element across seq*
                                sum (funcall weight element))
                          n))
         (limit (* sum-weights ratio))
         (sum 0)
         (bag ()))
    (loop while (< sum limit) do
      (let ((element (aref seq* (random n random-state))))
        (push element bag)
        (incf sum (if weight (funcall weight element) 1))))
    (if (listp seq)
        (nreverse bag)
        (coerce (nreverse bag) 'vector))))

(defun sample-without-replacement* (ratio seq &key weight
                                    (random-state *random-state*))
  (let* ((seq* (let ((*random-state* random-state))
                 (shuffle! (coerce-to-fresh-vector seq))))
         (n (length seq*))
         (sum-weights (if weight
                          (loop for element across seq*
                                sum (funcall weight element))
                          n))
         (limit (* sum-weights ratio))
         (sum 0)
         (bag ()))
    (loop for i below n
          while (< sum limit)
          do (let ((element (aref seq* i)))
               (push element bag)
               (incf sum (if weight (funcall weight element) 1))))
    (if (listp seq)
        (nreverse bag)
        (coerce (nreverse bag) 'vector))))

(defun coerce-to-fresh-vector (seq)
  (if (listp seq)
      (coerce seq 'vector)
      (copy-seq seq)))

(defun sample-stratified (ratio seq &key weight replacement
                          (key #'identity) (test #'eql)
                          (random-state *random-state*))
  "Like SAMPLE-FROM but makes sure that the weighted proportion of
  classes in the result is approximately the same as the proportion in
  SEQ. See STRATIFY for the description of KEY and TEST."
  (let ((per-key-bags
          (mapcar (lambda (elements)
                    (sample-from ratio elements
                                 :weight weight :replacement replacement
                                 :random-state random-state))
                  (stratify seq :key key :test test))))
    (apply #'concatenate (if (listp seq)
                             'list
                             `(vector ,(array-element-type seq)))
           per-key-bags)))


(defsection @mgl-resample-cv-bagging (:title "CV Bagging")
  (bag-cv function))

(defun bag-cv (data fn &key n (n-folds 5)
               (folds (alexandria:iota n-folds))
               (split-fn #'split-fold/mod) pass-fold
               (random-state *random-state*))
  "Perform cross-validation on different shuffles of DATA N times and
  collect the results. Since CROSS-VALIDATE collects the return values
  of FN, the return value of this function is a list of lists of FN
  results. If N is NIL, don't collect anything just keep doing
  repeated CVs until FN performs a non-local exit.

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
  estimate for each example which gets refined by further CVs."
  (bag data
       (lambda (samples)
         (cross-validate samples fn :n-folds n-folds :folds folds
                         :split-fn split-fn :pass-fold pass-fold))
       :ratio 1 :n n :replacement nil
       :random-state random-state))


(defsection @mgl-resample-misc (:title "Miscellaneous Operations")
  (spread-strata function)
  (zip-evenly function))

(defun spread-strata (seq &key (key #'identity) (test #'eql))
  "Return a sequence that's a reordering of SEQ such that elements
  belonging to different strata (under KEY and TEST, see STRATIFY) are
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
  ```"
  (zip-evenly (stratify seq :key key :test test)
              :result-type (if (listp seq) 'list 'vector)))

(defun zip-evenly (seqs &key result-type)
  "Make a single sequence out of the sequences in SEQS so that in the
  returned sequence indices of elements belonging to the same source
  sequence are spread evenly across the whole range. The result is a
  list is RESULT-TYPE is LIST, it's a vector if RESULT-TYPE is VECTOR.
  If RESULT-TYPE is NIL, then it's determined by the type of the first
  sequence in SEQS.

  ```cl-transcript
  (zip-evenly '((0 2 4) (1 3)))
  => (0 1 2 3 4)
  ```"
  (let* ((n (length seqs))
         (lengths (map 'vector #'length seqs))
         (seq-indices (make-array n :initial-element 0))
         (result-length (reduce #'+ lengths))
         (result (make-array result-length))
         (result-index 0))
    (flet ((find-next ()
             (let ((min nil)
                   (min-i nil))
               (loop for i below n
                     for index across seq-indices
                     for length across lengths
                     do (let ((x (/ index length)))
                          (when (or (null min) (< x min))
                            (setq min x)
                            (setq min-i i))))
               min-i)))
      (let ((seqs (map 'vector (lambda (seq)
                                 (coerce seq 'vector))
                       seqs)))
        (loop while (< result-index result-length)
              do (let ((i (find-next)))
                   (setf (aref result result-index)
                         (aref (aref seqs i) (aref seq-indices i)))
                   (incf result-index)
                   (incf (aref seq-indices i))))))
    (if (and (not (eq result-type 'vector)) (listp (first seqs)))
        (coerce result 'list)
        result)))
