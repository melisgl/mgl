(in-package :mgl-core)

;;;; FIXME: add weight support: INSTANCE-WEIGHT? STRIPED-WEIGHTS?

(defsection @mgl-classification (:title "Classification")
  "To be able to measure classification related quantities, we need to
  define what the label of an instance is. Customization is possible
  by implementing a method for a specific type of instance, but these
  functions only ever appear as defaults that can be overridden."
  (label-index generic-function)
  (label-index-distribution generic-function)
  "The following two functions are basically the same as the previous
  two, but in batch mode: they return a sequence of label indices or
  distributions. These are called on results produced by models.
  Implement these for a model and the monitor maker functions below
  will automatically work. See FIXDOC: for bpn and boltzmann."
  (label-indices generic-function)
  (label-index-distributions generic-function)
  (@mgl-classification-monitor section)
  (@mgl-classification-measurer section)
  (@mgl-classification-counter section))

(defgeneric label-index (instance)
  (:documentation "Return the label of INSTANCE as a non-negative
  integer."))

(defgeneric label-index-distribution (instance)
  (:documentation "Return a one dimensional array of probabilities
  representing the distribution of labels. The probability of the
  label with LABEL-INDEX `I` is element at index `I` of the returned
  arrray."))

(defgeneric label-indices (results)
  (:documentation "Return a sequence of label indices for RESULTS
  produced by some model for a batch of instances. This is akin to
  LABEL-INDEX."))

(defgeneric label-index-distributions (result)
  (:documentation "Return a sequence of label index distributions for
  RESULTS produced by some model for a batch of instances. This is
  akin to LABEL-INDEX-DISTRIBUTION."))

(defsection @mgl-classification-monitor (:title "Classification Monitors")
  "The following functions return a list monitors. The monitors are
  for events of signature (INSTANCES MODEL) such as those produced by
  MONITOR-MODEL-RESULTS and its various model specific variations.
  They are model-agnostic functions, extensible to new classifier
  types. "
  (make-classification-accuracy-monitors function)
  (make-cross-entropy-monitors function)
  (make-label-monitors function)
  "The monitor makers above can be extended to support new classifier
  types via the following generic functions."
  (make-classification-accuracy-monitors* generic-function)
  (make-cross-entropy-monitors* generic-function))

;;;; FIXME, FIXDOC: OPERATION-MODE is currently unused and undocumented.

(defun make-classification-accuracy-monitors
    (model &key operation-mode attributes (label-index-fn #'label-index))
  "Return a list of MONITOR objects associated with
  [CLASSIFICATION-ACCURACY-COUNTER][]s. LABEL-INDEX-FN is a function
  like LABEL-INDEX. See that function for more.

  Implemented in terms of MAKE-CLASSIFICATION-ACCURACY-MONITORS*."
  (make-classification-accuracy-monitors* model operation-mode label-index-fn
                                          attributes))

(defun make-cross-entropy-monitors (model &key operation-mode attributes
                                    (label-index-distribution-fn
                                     #'label-index-distribution))
  "Return a list of MONITOR objects associated with
  [CROSS-ENTROPY-COUNTER][]s. LABEL-INDEX-DISTRIBUTION-FN is a
  function like LABEL-INDEX-DISTRIBUTION. See that function for more.

  Implemented in terms of MAKE-CROSS-ENTROPY-MONITORS*."
  (make-cross-entropy-monitors* model operation-mode label-index-distribution-fn
                                attributes))

(defun make-label-monitors (model &key operation-mode attributes
                            (label-index-fn #'label-index)
                            (label-index-distribution-fn
                             #'label-index-distribution))
  "Return classification accuracy and cross-entropy monitors. See
  MAKE-CLASSIFICATION-ACCURACY-MONITORS and
  MAKE-CROSS-ENTROPY-MONITORS for a description of paramters."
  (append (make-classification-accuracy-monitors* model operation-mode
                                                  label-index-fn attributes)
          (make-cross-entropy-monitors* model operation-mode
                                        label-index-distribution-fn
                                        attributes)))

(defgeneric make-classification-accuracy-monitors*
    (model operation-mode label-index-fn attributes)
  (:documentation "Identical to MAKE-CLASSIFICATION-ACCURACY-MONITORS
  bar the keywords arguments. Specialize this to add to support for
  new model types. The default implementation also allows for some
  extensibility: if LABEL-INDICES is defined on MODEL, then it will be
  used to extract label indices from model results.")
  (:method (object operation-mode label-index-fn attributes)
    (when (applies-to-p #'label-indices object)
      (list
       (make-instance
        'monitor
        :measurer (lambda (instances result)
                    (declare (ignore result))
                    (measure-classification-accuracy
                     instances (label-indices object)
                     :truth-key label-index-fn
                     :test #'=))
        :counter (make-instance
                  'classification-accuracy-counter
                  :prepend-attributes (append attributes
                                         `(:component ,(name object)))))))))

(defgeneric make-cross-entropy-monitors*
    (model operation-mode label-index-distribution-fn attributes)
  (:documentation "Identical to MAKE-CROSS-ENTROPY-MONITORS bar the
  keywords arguments. Specialize this to add to support for new model
  types. The default implementation also allows for some
  extensibility: if LABEL-INDEX-DISTRIBUTIONS is defined on MODEL,
  then it will be used to extract label distributions from model
  results.")
  (:method (object operation-mode label-index-distribution-fn attributes)
    (when (applies-to-p #'label-index-distributions object)
      (list
       (make-instance
        'monitor
        :measurer (lambda (instances result)
                    (declare (ignore result))
                    (measure-cross-entropy
                     instances (label-index-distributions object)
                     :truth-key label-index-distribution-fn))
        :counter (make-instance
                  'cross-entropy-counter
                  :prepend-attributes `(,@attributes
                                        :component ,(name object))))))))


(defsection @mgl-classification-measurer (:title "Classification Measurers")
  "The functions here compare some known good solution (also known as
  _ground truth_ or _target_) to a prediction or approximation and
  return some measure of their \\[dis]similarity. They are model
  independent, hence one has to extract the ground truths and
  predictions first. Rarely used directly, they are mostly hidden
  behind @MGL-CLASSIFICATION-MONITOR."
  (measure-classification-accuracy function)
  (measure-cross-entropy function)
  (measure-roc-auc function)
  (measure-confusion function))

(defun measure-classification-accuracy (truths predictions &key (test #'eql)
                                        truth-key prediction-key weight)
  "Return the number of correct classifications and as the second
  value the number of instances (equal to length of TRUTHS in the
  non-weighted case). TRUTHS (keyed by TRUTH-KEY) is a sequence of
  opaque class labels compared with TEST to another sequence of
  classes labels in PREDICTIONS (keyed by PREDICTION-KEY). If WEIGHT
  is non-nil, then it is a function that returns the weight of an
  element of TRUTHS. Weighted cases add their weight to both
  counts (returned as the first and second values) instead of 1 as in
  the non-weighted case.

  Note how the returned values are suitable for MULTIPLE-VALUE-CALL
  with #'ADD-TO-COUNTER and a CLASSIFICATION-ACCURACY-COUNTER."
  (assert (= (length truths) (length predictions)))
  (let ((n-correct-classifications 0)
        (n 0))
    (map nil (lambda (truth prediction)
               (let ((truth (apply-key truth-key truth))
                     (prediction (apply-key prediction-key prediction))
                     (weight (if weight (funcall weight truth) 1)))
                 (when (funcall test truth prediction)
                   (incf n-correct-classifications weight))
                 (incf n weight)))
         truths predictions)
    ;; FIXEXT: is this better?
    #+nil
    (make-instance 'classification-accuracy-counter
                   :sum-errors n-correct-classifications
                   :n-sum-errors n)
    (values n-correct-classifications n)))

(defun measure-cross-entropy (truths predictions &key truth-key
                              prediction-key
                              (min-prediction-pr #.(expt 10d0 -15)))
  "Return the sum of the cross-entropy between pairs of elements with
  the same index of TRUTHS and PREDICTIONS. TRUTH-KEY is a function
  that's when applied to an element of TRUTHS returns a sequence
  representing some kind of discrete target distribution (P in the
  definition below). TRUTH-KEY may be NIL which is equivalent to the
  IDENTITY function. PREDICTION-KEY is the same kind of key for
  PREDICTIONS, but the sequence it returns represents a distribution
  that approximates (Q below) the true one.

  Cross-entropy of the true and approximating distributions is defined
  as:

      cross-entropy(p,q) = - sum_i p(i) * log(q(i))

  of which this function returns the sum over the pairs of elements of
  TRUTHS and PREDICTIONS keyed by TRUTH-KEY and PREDICTION-KEY.

  Due to the logarithm, if q(i) is close to zero, we run into
  numerical problems. To prevent this, all q(i) that are less than
  MIN-PREDICTION-PR are treated as if they were MIN-PREDICTION-PR.

  The second value returned is the sum of p(i) over all TRUTHS and all
  `I`. This is normally equal to `(LENGTH TRUTHS)`, since elements of
  TRUTHS represent a probability distribution, but this is not
  enforced which allows relative importance of elements to be
  controlled.

  The third value returned is a plist that maps each index occurring
  in the distribution sequences to a list of two elements:

       sum_j p_j(i) * log(q_j(i))

  and

      sum_j p_j(i)

  where `J` indexes into TRUTHS and PREDICTIONS.

      (measure-cross-entropy '((0 1 0)) '((0.1 0.7 0.2)))
      => 0.35667497
         1
         (2 (0.0 0)
          1 (0.35667497 1)
          0 (0.0 0))

  Note how the returned values are suitable for MULTIPLE-VALUE-CALL
  with #'ADD-TO-COUNTER and a CROSS-ENTROPY-COUNTER."
  (assert (= (length truths) (length predictions)))
  (let ((sum 0)
        (sum-weights 0)
        (label-errors ()))
    (map nil (lambda (truth prediction)
               (let ((truth (if truth-key (funcall truth-key truth) truth))
                     (prediction
                       (if prediction-key
                           (funcall prediction-key prediction)
                           prediction)))
                 (let ((label 0))
                   (map nil (lambda (true-pr prediction-pr)
                              (let ((err (- (* true-pr
                                               (log (max min-prediction-pr
                                                         prediction-pr))))))
                                (incf sum err)
                                (incf sum-weights true-pr)
                                (let ((label-error
                                        (or (getf label-errors label)
                                            (setf (getf label-errors label)
                                                  (list 0 0)))))
                                  (incf (first label-error) err)
                                  (incf (second label-error) true-pr)))
                              (incf label))
                        truth prediction))))
         truths predictions)
    (values (if (zerop sum-weights)
                0
                (/ sum sum-weights))
            sum-weights label-errors)))

(defun measure-roc-auc (predictions pred &key (key #'identity) weight)
  "Return the area under the ROC curve for PREDICTIONS representing
  predictions for a binary classification problem. PRED is a predicate
  function for deciding whether a prediction belongs to the so called
  positive class. KEY returns a number for each element which is the
  predictor's idea of how much that element is likely to belong to the
  class, although it's not necessarily a probability.

  If WEIGHT is NIL, then all elements of PREDICTIONS count as 1
  towards the unnormalized sum within AUC. Else WEIGHT must be a
  function like KEY, but it should return the importance (a positive
  real number) of elements. If the weight of an prediction is 2 then
  it's as if there were another identical copy of that prediction in
  PREDICTIONS.

  The algorithm is based on algorithm 2 in the paper 'An introduction
  to ROC analysis' by Tom Fawcett.

  ROC AUC is equal to the probability of a randomly chosen positive
  having higher KEY (score) than a randomly chosen negative element.
  With equal scores in mind, a more precise version is: AUC is the
  expectation of the above probability over all possible sequences
  sorted by scores."
  (let ((sum 0)
        (seq (stable-sort (copy-seq predictions) #'> :key key))
        (prev-pos-count 0)
        (prev-neg-count 0)
        (pos-count 0)
        (neg-count 0)
        (prev-score nil))
    (flet ((add ()
             ;; Called once per 'batch', i.e. examples with equal
             ;; scores.
             (incf sum (*
                        ;; The number of negative examples in the
                        ;; current batch.
                        (- neg-count prev-neg-count)
                        ;; The number of positive examples occurring
                        ;; earlier than the negatives examples in the
                        ;; batch. With positive examples in the batch
                        ;; we take the expectation over random orders.
                        (/ (+ pos-count prev-pos-count) 2)))))
      (map nil
           (lambda (x)
             (let ((score (funcall key x))
                   (w (if weight (funcall weight x) 1)))
               (when (or (null prev-score) (/= prev-score score))
                 (add)
                 (setq prev-score score
                       prev-pos-count pos-count
                       prev-neg-count neg-count))
               (if (funcall pred x)
                   (incf pos-count w)
                   (incf neg-count w))))
           seq)
      (add))
    (if (or (zerop pos-count) (zerop neg-count))
        (values nil nil nil)
        (values (/ sum pos-count neg-count) pos-count neg-count))))

(defun measure-confusion (truths predictions &key (test #'eql)
                          truth-key prediction-key weight)
  "Create a CONFUSION-MATRIX from TRUTHS and PREDICTIONS.
  TRUTHS (keyed by TRUTH-KEY) is a sequence of class labels compared
  with TEST to another sequence of class labels in PREDICTIONS (keyed
  by PREDICTION-KEY). If WEIGHT is non-nil, then it is a function that
  returns the weight of an element of TRUTHS. Weighted cases add their
  weight to both counts (returned as the first and second values).

  Note how the returned confusion matrix can be added to another with
  ADD-TO-COUNTER."
  (assert (= (length truths) (length predictions)))
  (let ((confusions (make-confusion-matrix :test test)))
    (map nil (lambda (truth prediction)
               (let ((truth (apply-key truth-key truth))
                     (prediction (apply-key prediction-key prediction))
                     (weight (if weight (funcall weight truth) 1)))
                 (incf (confusion-count confusions truth prediction)
                       weight)))
         truths predictions)
    confusions))


(defsection @mgl-classification-counter (:title "Classification Counters")
  (classification-accuracy-counter class)
  (cross-entropy-counter class)
  (@mgl-confusion-matrix section))

(defclass classification-accuracy-counter (basic-counter)
  ((attributes :initform '(:type "acc.")))
  (:documentation "A BASIC-COUNTER with \"acc.\" as its :TYPE
  attribute and a PRINT-OBJECT method that prints percentages."))

(defmethod print-object ((counter classification-accuracy-counter) stream)
  (maybe-print-unreadable-object (counter stream :type t)
    (print-name counter stream)
    (multiple-value-bind (e c) (counter-values counter)
      (if e
          (format stream "~,2F% (~D)" (* 100 e) c)
          (format stream "~A (~D)" e c)))))

;;; FIXDOC
(defclass cross-entropy-counter (basic-counter)
  ((attributes :initform '(:type "xent"))
   (per-label-counters
    :initform (make-hash-table)
    :initarg :per-label-counters
    :reader per-label-counters
    :documentation "A hash table mapping labels to the cross entropy
    counters for samples with that label."))
  (:documentation "A BASIC-COUNTER with \"xent\" as its :TYPE
  attribute."))

(defvar *print-label-counters* nil)

;;; FIXME: COUNTER-VALUES

(defmethod print-object ((counter cross-entropy-counter) stream)
  (maybe-print-unreadable-object (counter stream :type t)
    (let ((*print-escape* nil))
      (call-next-method))
    (when *print-label-counters*
      (loop for cons in (stable-sort (alexandria:hash-table-alist
                                      (per-label-counters counter))
                                     #'< :key #'car)
            do (destructuring-bind (label . counter) cons
                 (format stream ",~_ (label: ~S, ~A)" label
                         (with-output-to-string (stream)
                           (let ((*print-escape* nil))
                             (print-object counter stream)))))))))

(defmethod add-to-counter ((counter cross-entropy-counter) &rest args)
  ;; PER-INDEX-CE is optional because we end up calling this method
  ;; for the labels.
  (destructuring-bind (sum-ce sum-weights &optional per-index-ce) args
    (call-next-method counter sum-ce sum-weights)
    (loop for (label (sum-ce sum-weights)) on per-index-ce by #'cddr
          do (let ((counter-for-label
                     (or (gethash label (per-label-counters counter))
                         (setf (gethash label (per-label-counters counter))
                               (make-instance 'cross-entropy-counter)))))
               (add-to-counter counter-for-label sum-ce sum-weights)))))

(defmethod reset-counter ((counter cross-entropy-counter))
  (call-next-method)
  (map nil #'reset-counter
       (mapcar #'cdr (alexandria:hash-table-alist
                      (per-label-counters counter)))))


;;; FIXDOC
(defsection @mgl-confusion-matrix (:title "Confusion Matrices")
  (confusion-matrix class)
  (make-confusion-matrix function)
  (sort-confusion-classes generic-function)
  (confusion-class-name generic-function)
  (confusion-count generic-function)
  (map-confusion-matrix generic-function)
  (confusion-matrix-classes generic-function)
  (confusion-matrix-accuracy function)
  (confusion-matrix-precision function)
  (confusion-matrix-recall function)
  (add-confusion-matrix function))

(defclass confusion-matrix ()
  ((counts :initform (make-hash-table) :initarg :counts :reader counts))
  (:documentation "A confusion matrix keeps count of classification
  results. The correct class is called `target' and the output of the
  classifier is called `prediction'. Classes are compared with
  EQUAL."))

(defun make-confusion-matrix (&key (test #'eql))
  (make-instance 'confusion-matrix
                 :counts (make-hash-table :test test)))

(defgeneric sort-confusion-classes (matrix classes)
  (:documentation "Return a list of CLASSES sorted for presentation
  purposes.")
  (:method ((matrix confusion-matrix) classes)
    (stable-sort (copy-seq classes) #'string<=
                 :key (lambda (class)
                        (confusion-class-name matrix class)))))

(defgeneric confusion-class-name (matrix class)
  (:documentation "Name of CLASS for presentation purposes.")
  (:method ((matrix confusion-matrix) class)
    (princ-to-string class)))

(defgeneric confusion-count (matrix target prediction)
  (:method ((matrix confusion-matrix) target prediction)
    (gethash (cons target prediction) (counts matrix) 0)))

(defgeneric (setf confusion-count) (count matrix target prediction)
  (:method (count (matrix confusion-matrix) target prediction)
    (setf (gethash (cons target prediction) (counts matrix)) count)))

(defgeneric map-confusion-matrix (fn matrix)
  (:documentation "Call FN with TARGET, PREDICTION, COUNT paramaters
  for each cell in the confusion matrix. Cells with a zero count may
  be ommitted.")
  (:method (fn (matrix confusion-matrix))
    (maphash (lambda (key value)
               (funcall fn (car key) (cdr key) value))
             (counts matrix))))

(defgeneric confusion-matrix-classes (matrix)
  (:documentation "A list of all classes. The default is to collect
  classes from the counts. This can be overridden if, for instance,
  some classes are not present in the results.")
  (:method ((matrix confusion-matrix))
    (let ((all-classes ()))
      (map-confusion-matrix (lambda (target prediction count)
                              (declare (ignore count))
                              (pushnew target all-classes :test #'equal)
                              (pushnew prediction all-classes :test #'equal))
                            matrix)
      all-classes)))

(defun confusion-matrix-accuracy (matrix &key filter)
  "Return the overall accuracy of the results in MATRIX. It's computed
  as the number of correctly classified cases (hits) divided by the
  name of cases. Return the number of hits and the number of cases as
  the second and third value. If FILTER function is given, then call
  it with the target and the prediction of the cell. Disregard cell
  for which FILTER returns NIL.

  Precision and recall can be easily computed by giving the right
  filter, although those are provided in separate convenience
  functions."
  (let ((n-hits 0)
        (total 0))
    (map-confusion-matrix (lambda (target prediction count)
                            (when (or (null filter)
                                      (funcall filter target prediction))
                              (when (eql target prediction)
                                (incf n-hits count))
                              (incf total count)))
                          matrix)
    (if (zerop total)
        (values nil 0 0)
        (values (/ n-hits total) n-hits total))))

(defun confusion-matrix-precision (matrix prediction)
  "Return the accuracy over the cases when the classifier said
  PREDICTION."
  (confusion-matrix-accuracy matrix
                             :filter (lambda (target prediction2)
                                       (declare (ignore target))
                                       (eql prediction prediction2))))

(defun confusion-matrix-recall (matrix target)
  "Return the accuracy over the cases when the correct class is
  TARGET."
  (confusion-matrix-accuracy matrix
                             :filter (lambda (target2 prediction)
                                       (declare (ignore prediction))
                                       (eql target target2))))

(defun add-confusion-matrix (matrix result-matrix)
  "Add MATRIX into RESULT-MATRIX."
  (map-confusion-matrix (lambda (target prediction count)
                          (incf (confusion-count result-matrix target
                                                 prediction)
                                count))
                        matrix)
  result-matrix)

(defmethod print-object ((matrix confusion-matrix) stream)
  (flet ((->% (x)
           (if x
               (format nil "~,2F%" (* 100 x))
               nil)))
    (let ((all-classes
            (sort-confusion-classes matrix
                                    (confusion-matrix-classes matrix))))
      (print-unreadable-object (matrix stream :type t)
        (terpri stream)
        (print-table
         `(("" ,@(mapcar (lambda (class)
                           (confusion-class-name matrix class))
                         all-classes)
               "Recall")
           ,@(loop
               for target in all-classes
               collect (append (cons (confusion-class-name matrix target)
                                     (loop for prediction in all-classes
                                           collect (confusion-count
                                                    matrix target prediction)))
                               (list (->% (confusion-matrix-recall matrix
                                                                   target)))))
           ("Precision"
            ,@(loop for prediction in all-classes
                    collect (->% (confusion-matrix-precision matrix
                                                             prediction)))
            ""))
         :stream stream)
        (format stream "Accuracy: ~A"
                (->% (confusion-matrix-accuracy matrix)))))
    matrix))


;;;; ROC

;;;; FIXDEAD

#|

(defclass collecting-counter (counter)
  ((elements
    :initform (make-array 0 :adjustable t :fill-pointer 0)
    :initarg :elements
    :accessor elements)))

(defmethod reset-counter ((counter collecting-counter))
  (setf (elements counter) (make-array 0 :adjustable t :fill-pointer 0)))

(defclass roc-auc-counter (collecting-counter)
  ((attributes :initform '(:type "roc-auc"))
   (class-label :initform (error "CLASS-LABEL must be provided.")
                :initarg :class-label :reader class-label)
   (class-index :initform (error "CLASS-INDEX must be provided.")
                :initarg :class-index :reader class-index)
   (example-fn :initform #'car :initarg :example-fn :reader example-fn)
   (label-fn :initform nil :initarg :label-fn :reader label-fn)
   (confidences-fn
    :initform #'cdr :initarg :confidences-fn :reader confidences-fn)))

(defmethod print-counter ((counter roc-auc-counter) stream)
  (multiple-value-bind (e c) (get-error counter)
    (if e
        (format stream "~,5F (~D)" e c)
        (format stream "~A (~D)" e c))))

(defmethod add-error ((counter roc-auc-counter) elements n)
  (assert (= n (length elements)))
  (let ((elements* (elements counter)))
    (loop for x in elements do (vector-push-extend x elements*))))

(defmethod get-error ((counter roc-auc-counter))
  (let ((class-label (class-label counter))
        (class-index (class-index counter))
        (example-fn (example-fn counter))
        (label-fn (label-fn counter))
        (confidences-fn (confidences-fn counter)))
    (measure-roc-auc (elements counter)
                     (lambda (element)
                       (let ((example (funcall example-fn element)))
                         (eql (funcall label-fn example) class-label)))
                     :key (lambda (element)
                            (elt (funcall confidences-fn element)
                                 class-index)))))

(defun make-chunk-reconstruction-roc-auc-monitors
    (chunks &key chunk-filter class-label class-index)
  (loop for chunk in (mgl-bm::remove-if* chunk-filter chunks)
        for measurer = (maybe-make-classification-confidence-collector chunk)
        when measurer
          collect
          (make-instance 'monitor
                         :measurer measurer
                         :counter
                         (make-instance 'roc-auc-counter
                                        :prepend-name
                                        (format nil "chunk ~A (class ~A/~A)"
                                                (name chunk)
                                                class-label
                                                class-index)
                                        :class-label class-label
                                        :class-index class-index))))

(defun make-bm-reconstruction-roc-auc-monitors
    (bm &key chunk-filter)
  "Return a list of counter, measurer conses to keep track of
  classification-accuracys suitable for BM-MEAN-FIELD-ERRORS."
  (make-chunk-reconstruction-classification-accuracy-monitors
   (chunks bm) :chunk-filter chunk-filter))

|#
