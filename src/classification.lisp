(in-package :mgl-train)

(defgeneric label (object)
  (:documentation "Return the label of object as an index. This is a
special case of LABEL-DISTRIBUTION."))

(defgeneric label-distribution (striped  stripe object)
  (:documentation "Return an FLT-VECTOR that represents our knowledge
of the distribution of the true label of OBJECT.")
  (:method (striped stripe object)
    nil))

(defclass labeled () ()
  (:documentation "Mixin for chunks/whatever that hold labels. In the
simplest case you need to make sure that LABEL and STRIPE-LABEL work
on examples and striped things of interest.

For instance in a BM, SOFTMAX-LABEL-CHUNK inherits from LABELED and
SOFTMAX-CHUNK. STRIPE-LABEL is implemented as taking the index of the
prediction with the maximum probability. Thus, only LABEL is left to
be implemented on the training examples.

Once set up, COUNT-MISCLASSIFICATIONS can be called directly or one
can work with counters and measurers."))

(defun labeledp (object)
  (typep object 'labeled))

(defgeneric stripe-label (striped stripe)
  (:documentation "Return the label of STRIPE in STRIPED. Typically
computed by finding the label with the maximum probability."))

(defun count-misclassifications (examples striped
                                 &key (label-fn #'label)
                                 (stripe-label-fn #'stripe-label))
  "Return the number of classification errors and the number of
examples. The length of EXAMPLES must be equal to the number of
stripes in STRIPED. LABEL-FN takes an example and returns its label
that compared by EQL to what STRIPE-LABEL-FN returns for STRIPED and
the index of the stripe. This is a measurer function."
  (assert (= (length examples) (n-stripes striped)))
  (let ((n-misclassifications 0)
        (n 0))
    (loop for example in examples
          for stripe upfrom 0
          do (let ((label (funcall label-fn example)))
               (when label
                 (when (/= label (funcall stripe-label-fn striped stripe))
                   (incf n-misclassifications))
                 (incf n))))
    (values n-misclassifications n)))

(defgeneric maybe-make-misclassification-measurer (obj)
  (:documentation "Return a function of one parameter that is invoked
when OBJ has the predicted label(s) computed and it counts
misclassifications. Return NIL if OBJ contains no labels.")
  (:method (obj)
    (values nil nil))
  (:method ((labeled labeled))
    (lambda (examples learner)
      (declare (ignore learner))
      (count-misclassifications examples labeled))))

(defgeneric classification-confidences (striped stripe))

(defun measure-cross-entropy (examples striped
                              &key (label-fn #'label)
                              (label-distribution-fn #'label-distribution)
                              (confidence-fn #'classification-confidences))
  "Return the sum of the cross entropy between the confidences and the
distribution (1 at the label of the class) and the number of examples.
The length of EXAMPLES must be equal to the number of stripes in
STRIPED. LABEL-FN takes an example and returns its label. This is a
measurer function."
  (assert (= (length examples) (n-stripes striped)))
  (let ((sum 0)
        (sum-weights #.(flt 0))
        (label-errors ()))
    (loop for example in examples
          for stripe upfrom 0
          for confidences = (funcall confidence-fn striped stripe)
          do (let ((distribution (funcall label-distribution-fn
                                          striped stripe example)))
               (cond (distribution
                      (assert (= (length confidences) (length distribution)))
                      (loop for prediction across confidences
                            for target across distribution
                            for label upfrom 0
                            do (let ((err (- (* target
                                                (log (max #.(expt 10d0 -15)
                                                          prediction))))))
                                 (incf sum err)
                                 (incf sum-weights target)
                                 (let ((label-error
                                         (or (getf label-errors label)
                                             (setf (getf label-errors label)
                                                   (list #.(flt 0) 0)))))
                                   (incf (first label-error) err)
                                   (incf (second label-error) target)))))
                     (t
                      (let ((label (funcall label-fn example)))
                        (when label
                          (let ((err (- (* (log
                                            (max #.(expt 10d0 -15)
                                                 (aref confidences label)))))))
                            (incf sum err)
                            (incf sum-weights)
                            (let ((label-error
                                    (or (getf label-errors label)
                                        (setf (getf label-errors label)
                                              (list #.(flt 0) 0)))))
                              (incf (first label-error) err)
                              (incf (second label-error))))))))))
    (values (list sum label-errors)
            sum-weights)))

(defgeneric maybe-make-cross-entropy-measurer (obj)
  (:documentation "Return a function of one parameter that is invoked
when OBJ has the predicted label(s) computed and it measures cross
entropy error. Return NIL if OBJ contains no labels.")
  (:method (obj)
    (values nil nil))
  (:method ((labeled labeled))
    (lambda (examples learner)
      (declare (ignore learner))
      (measure-cross-entropy examples labeled))))

(defclass cross-entropy-counter (error-counter)
  ((name :initform '("cross entropy"))
   (per-label-counters
    :initform (make-hash-table)
    :initarg :per-label-counters
    :reader per-label-counters
    :documentation "A hash table mapping labels to the cross entropy
counters for samples with that label.")))

(defvar *print-label-counters* nil)

(defmethod print-counter ((counter cross-entropy-counter) stream)
  (multiple-value-bind (e c) (get-error counter)
    (if e
        (format stream "~,5E (~D)" e c)
        (format stream "~A (~D)" e c)))
  (when *print-label-counters*
    (loop for cons in (sort (alexandria:hash-table-alist
                             (per-label-counters counter))
                            #'< :key #'car)
          do (destructuring-bind (label . counter) cons
               (format stream ",~_ (label: ~S, ~A)" label
                       (with-output-to-string (stream)
                         (print-counter counter stream)))))))

(defmethod add-error ((counter cross-entropy-counter) (err list) n)
  (destructuring-bind (overall-err label-errors) err
    (call-next-method counter overall-err n)
    (loop for (label (err n)) on label-errors by #'cddr
          do (let ((counter-for-label
                     (or (gethash label (per-label-counters counter))
                         (setf (gethash label (per-label-counters counter))
                               (make-instance 'cross-entropy-counter)))))
               (add-error counter-for-label err n)))))

(defmethod reset-counter ((counter cross-entropy-counter))
  (call-next-method)
  (map nil #'reset-counter
       (mapcar #'cdr (alexandria:hash-table-alist
                      (per-label-counters counter)))))


;;;; ROC

(defun roc-auc (seq pred &key (key #'identity) weight)
  "Return the area under the ROC curve for the dataset represented by
SEQ. PRED is a predicate function for deciding whether an element of
SEQ belongs to the class in question. KEY returns a number for each
element which is the predictor's idea of how much that element is
likely to belong to the class, it's not necessarily a probability.

If WEIGHT is NIL, then all elements of SEQ count as 1 towards the
unnormalized sum within AUC. Else WEIGHT must be a function like KEY,
but it should return the importance (a positive real number) of
elements. If the weight of an element is 2 then it's as if there were
two instances of that element in SEQ.

The algorithm is based on algorithm 2 in the paper 'An introduction to
ROC analysis' by Tom Fawcett."
  ;; AUC is equal to the probability of a randomly chosen positive
  ;; having higher KEY (score) than a randomly chosen negative
  ;; element. With equal scores in mind, a more precise version is:
  ;; AUC is the expectation of the above probability over all possible
  ;; sequences sorted by scores.
  (let ((sum 0)
        (seq (sort (copy-seq seq) #'> :key key))
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

(defclass collecting-counter (counter)
  ((elements
    :initform (make-array 0 :adjustable t :fill-pointer 0)
    :initarg :elements
    :accessor elements)))

(defmethod reset-counter ((counter collecting-counter))
  (setf (elements counter) (make-array 0 :adjustable t :fill-pointer 0)))

(defclass roc-auc-counter (collecting-counter)
  ((name :initform '("auc"))
   (class-label :initform (error "CLASS-LABEL must be provided.")
                :initarg :class-label :reader class-label)
   (class-index :initform (error "CLASS-INDEX must be provided.")
                :initarg :class-index :reader class-index)
   (example-fn :initform #'car :initarg :example-fn :reader example-fn)
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
        (confidences-fn (confidences-fn counter)))
    (roc-auc (elements counter)
             (lambda (element)
               (let ((example (funcall example-fn element)))
                 (eql (label example) class-label)))
             :key (lambda (element)
                    (elt (funcall confidences-fn element)
                         class-index)))))

(defun collect-classification-confidences
    (examples striped &key (confidence-fn #'classification-confidences))
  "Return the sequence of prediction confidences for EXAMPLES as
measured on STRIPED when number of classification errors and the
number of examples. The length of EXAMPLES must be equal to the number
of stripes in STRIPED. LABEL-FN takes an example and returns its label
that compared by EQL to what STRIPE-LABEL-FN returns for STRIPED and
the index of the stripe. This is a measurer function."
  (assert (= (length examples) (n-stripes striped)))
  (values (loop for example in examples
                for stripe upfrom 0
                for confidence = (funcall confidence-fn striped stripe)
                collect (cons example confidence))
          (length examples)))

(defgeneric maybe-make-classification-confidence-collector (obj)
  (:documentation "Return a collector function of (examples learner)
args that is invoked when OBJ has the predicted label(s) computed and
it collects a sequence of confidences (one confidence list per
example). Return NIL if OBJ contains no labels.")
  (:method (obj)
    (values nil nil))
  (:method ((labeled labeled))
    (lambda (examples learner)
      (declare (ignore learner))
      (collect-classification-confidences examples labeled))))

(defun make-chunk-reconstruction-roc-auc-counters-and-measurers
    (chunks &key chunk-filter class-label class-index)
  (loop for chunk in (mgl-bm::remove-if* chunk-filter chunks)
        for measurer = (maybe-make-classification-confidence-collector chunk)
        when measurer
        collect
        (cons (make-instance 'roc-auc-counter
                             :prepend-name (format nil "chunk ~A (class ~A/~A)"
                                                   (name chunk)
                                                   class-label
                                                   class-index)
                             :class-label class-label
                             :class-index class-index)
              measurer)))

#+nil
(defun make-bm-reconstruction-roc-auc-counters-and-measurers
    (bm &key chunk-filter)
  "Return a list of counter, measurer conses to keep track of
misclassifications suitable for BM-MEAN-FIELD-ERRORS."
  (make-chunk-reconstruction-misclassification-counters-and-measurers
   (chunks bm) :chunk-filter chunk-filter))
