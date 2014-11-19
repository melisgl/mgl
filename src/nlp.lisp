(in-package :mgl-nlp)

(defsection @mgl-nlp (:title "Natural Language Processing")
  "This in nothing more then a couple of utilities for now which may
  grow into a more serious toolset for NLP eventually."
  (make-n-gram-mappee function)
  (bleu function))

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

(defun bleu (corpus &key candidate-key reference-key (n 4))
  "Compute the [BLEU score](http://en.wikipedia.org/wiki/BLEU) for
  bilingual CORPUS. BLEU measures how good a translation is compared
  to human reference translations.

  CORPUS must be a sequence. CANDIDATE-KEY is function called with
  elements of CORPUS and returns a sequence of words (i.e. a tokenized
  translation). REFERENCE-KEY is similar but it returns the reference
  translation. Words are compared with EQUAL, and may be any kind of
  object (not necessarily strings).

  Currently there is no support for multiple reference translations. N
  determines the largest n-grams to consider.

  The first return value is the BLEU score (between 0 and 1, not as a
  percentage), the second value is the brevity penalty and the third
  is a list n-gram precisions (also between 0 and 1 or NIL), one for
  each element in [1..N].

  This is basically a reimplementation of
  [multi-bleu.perl](https://github.com/moses-smt/mosesdecoder/blob/master/scripts/generic/multi-bleu.perl).

  ```cl-transcript
  (bleu '(((1 2 3 4) (1 2 3 4))
          ((a b) (1 2)))
        :candidate-key #'first
        :reference-key #'second)
  => 0.8408964
  => 1.0
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
      (map nil (lambda (sentence)
                 (let ((candidate (funcall candidate-key sentence))
                       (reference (funcall reference-key sentence)))
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
           corpus)
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
                brevity-penalty
                precisions)))))
