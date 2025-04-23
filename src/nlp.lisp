(in-package :mgl-nlp)

(named-readtables:in-readtable pythonic-string-reader:pythonic-string-syntax)

(defsection @mgl-nlp (:title "Natural Language Processing")
  "This in nothing more then a couple of utilities for now which may
  grow into a more serious toolset for NLP eventually."
  (make-n-gram-mappee function)
  (bleu function)
  (@mgl-nlp-bag-of-words section))

(defun make-n-gram-mappee (function n)
  "Make a function of a single argument that's suitable as the
  function argument to a mapper function. It calls FUNCTION with every
  N element.

  ```cl-transcript
  (map nil (make-n-gram-mappee #'print 3) '(a b c d e))
  ..
  .. (A B C) 
  .. (B C D) 
  .. (C D E) 
  ```"
  (let ((previous-values '()))
    (lambda (x)
      (push x previous-values)
      (when (< n (length previous-values))
        (setf previous-values (subseq previous-values 0 n)))
      (when (= n (length previous-values))
        (funcall function (reverse previous-values))))))

(defun bleu (candidates references &key candidate-key reference-key (n 4))
  "Compute the [\\BLEU score](http://en.wikipedia.org/wiki/BLEU) for
  bilingual CORPUS. \\BLEU measures how good a translation is compared
  to human reference translations.

  CANDIDATES (keyed by CANDIDATE-KEY) and REFERENCES (keyed by
  REFERENCE-KEY) are sequences of sentences. A sentence is a sequence
  of words. Words are compared with EQUAL, and may be any kind of
  object (not necessarily strings).

  Currently there is no support for multiple reference translations. N
  determines the largest n-grams to consider.

  The first return value is the BLEU score (between 0 and 1, not as a
  percentage). The second value is the sum of the lengths of
  CANDIDATES divided by the sum of the lengths of REFERENCES (or NIL,
  if the denominator is 0). The third is a list of n-gram
  precisions (also between 0 and 1 or NIL), one for each element in
  \\[1..`N`].

  This is basically a reimplementation of
  [multi-bleu.perl](https://github.com/moses-smt/mosesdecoder/blob/master/scripts/generic/multi-bleu.perl).

  ```cl-transcript
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
  ```"
  (flet ((count-n-grams (seq n)
           (let ((counts (make-hash-table :test #'equal)))
             (map nil (make-n-gram-mappee (lambda (n-gram)
                                            (incf (gethash n-gram counts 0)))
                                          n)
                  seq)
             counts))
         (sum-clipped (candidate-n-gram-counts reference-n-gram-counts)
           (let ((sum 0))
             (maphash (lambda (n-gram candidate-count)
                        (let ((reference-count
                                (gethash n-gram reference-n-gram-counts 0)))
                          (incf sum (min candidate-count reference-count))))
                      candidate-n-gram-counts)
             sum))
         (sum-unclipped (n-gram-counts)
           (let ((sum 0))
             (maphash (lambda (n-gram count)
                        (declare (ignore n-gram))
                        (incf sum count))
                      n-gram-counts)
             sum)))
    (let ((sum-clipped (make-array n :initial-element 0))
          (sum-unclipped (make-array n :initial-element 0))
          (sum-length-candidate 0)
          (sum-length-reference 0))
      (map nil (lambda (candidate reference)
                 (let ((candidate (apply-key candidate-key candidate))
                       (reference (apply-key reference-key reference)))
                   (incf sum-length-candidate (length candidate))
                   (incf sum-length-reference (length reference))
                   (loop for i upfrom 1 upto n
                         do (let ((candidate-n-gram-counts
                                    (count-n-grams candidate i))
                                  (reference-n-gram-counts
                                    (count-n-grams reference i)))
                              (incf (aref sum-clipped (1- i))
                                    (sum-clipped candidate-n-gram-counts
                                                 reference-n-gram-counts))
                              (incf (aref sum-unclipped (1- i))
                                    (sum-unclipped candidate-n-gram-counts))))))
           candidates references)
      (let ((brevity-penalty
              (if (zerop sum-length-candidate)
                  0
                  (exp (min 0 (- 1 (/ sum-length-reference
                                      sum-length-candidate))))))
            (precisions (loop for i below n
                              collect (let ((clipped (aref sum-clipped i))
                                            (unclipped (aref sum-unclipped i)))
                                        ;; Handle the edge cases.
                                        (cond ((zerop unclipped)
                                               nil)
                                              (t
                                               (/ clipped unclipped)))))))
        (values (if (or (some #'null precisions)
                        (some #'zerop precisions))
                    0
                    (* brevity-penalty
                       (exp (/ (loop for p in precisions sum (log p)) n))))
                (if (zerop sum-length-reference)
                    nil
                    (/ sum-length-candidate sum-length-reference))
                precisions)))))


(defsection @mgl-nlp-bag-of-words (:title "Bag of Words")
  (bag-of-words-encoder class)
  (feature-encoder (reader bag-of-words-encoder))
  (feature-mapper (reader bag-of-words-encoder))
  (encoded-feature-test (reader bag-of-words-encoder))
  (encoded-feature-type (reader bag-of-words-encoder))
  (bag-of-words-kind (reader bag-of-words-encoder)))

(defclass bag-of-words-encoder ()
  ((feature-encoder :initarg :feature-encoder :reader feature-encoder)
   (feature-mapper :initarg :feature-mapper :reader feature-mapper)
   (encoded-feature-test :initform #'eql :initarg :encoded-feature-test
                         :reader encoded-feature-test)
   (encoded-feature-type :initform t :initarg :encoded-feature-type
                         :reader encoded-feature-type)
   (kind :initform :binary :initarg :kind :reader bag-of-words-kind
         :type (member :binary :frequency
                       :normalized-binary :normalized-frequency)))
  (:documentation """ENCODE all features of a document with a sparse
  vector. Get the features of document from MAPPER, encode each
  feature with FEATURE-ENCODER. FEATURE-ENCODER may return NIL if the
  feature is not used. The result is a vector of encoded-feature/value
  conses. encoded-features are unique (under ENCODED-FEATURE-TEST)
  within the vector but are in no particular order.

  Depending on KIND, value is calculated in various ways:

  - For :FREQUENCY it is the number of times the corresponding feature
  was found in DOCUMENT.

  - For :BINARY it is always 1.

  - For :NORMALIZED-FREQUENCY and :NORMALIZED-BINARY are like the
    unnormalized counterparts except that as the final step values in
    the assembled sparse vector are normalized to sum to 1.

  - Finally, :COMPACTED-BINARY is like :BINARY but the return values
    is not a vector of conses, but a vector of element-type
    ENCODED-FEATURE-TYPE.

  ```cl-transcript
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
  ```"""))

(defmethod encode ((encoder bag-of-words-encoder) decoded)
  (let ((feature-encoder (feature-encoder encoder)))
    (encode/bag-of-words decoded
                         (feature-mapper encoder)
                         (lambda (feature)
                           (encode feature-encoder feature))
                         :kind (bag-of-words-kind encoder)
                         :encoded-feature-type (encoded-feature-type encoder))))

(defun encode/bag-of-words (document mapper feature-encoder &key (kind :binary)
                            (encoded-feature-type t))
  (assert (member kind '(:binary :frequency
                         :normalized-binary :normalized-frequency
                         :compacted-binary)))
  (let ((v (make-array 20 :adjustable t :fill-pointer 0)))
    (funcall mapper
             (lambda (feature)
               (let ((index (funcall feature-encoder feature)))
                 (when index
                   (let ((pos (position index v :key #'car)))
                     (if pos
                         (incf (cdr (aref v pos)))
                         (vector-push-extend (cons index #.(flt 1)) v))))))
             document)
    (when (member kind '(:binary :normalized-binary :compacted-binary))
      (loop for x across v
            do (setf (cdr x) #.(flt 1))))
    (when (member kind '(:normalized-binary :normalized-frequency))
      (let ((sum (loop for x across v summing (cdr x))))
        (map-into v (lambda (x)
                      (cons (car x)
                            (/ (cdr x) sum)))
                  v)))
    (let ((r (stable-sort v #'< :key #'car)))
      (if (eq kind :compacted-binary)
          (compact-binary-feature-vector r encoded-feature-type)
          r))))

(defun compact-binary-feature-vector (feature-vector element-type)
  (make-array (length feature-vector)
              :element-type element-type
              :initial-contents (map 'vector #'car feature-vector)))
