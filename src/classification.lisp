(in-package :mgl-train)

(defgeneric label (object))

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
  (let ((n 0))
    (loop for example in examples
          for stripe upfrom 0
          do (unless (= (funcall label-fn example)
                        (funcall stripe-label-fn striped stripe))
               (incf n)))
    (values n (length examples))))

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
                              (confidence-fn #'classification-confidences))
  "Return the sum of the cross entropy between the real and the  the  and the number of
examples. The length of EXAMPLES must be equal to the number of
stripes in STRIPED. LABEL-FN takes an example and returns its label
that compared by EQL to what STRIPE-LABEL-FN returns for STRIPED and
the index of the stripe. This is a measurer function."
  (assert (= (length examples) (n-stripes striped)))
  (let ((sum 0))
    (loop for example in examples
          for stripe upfrom 0
          for confidences = (funcall confidence-fn striped stripe)
          do (incf sum (- (log (max #.(expt 10d0 -15)
                                    (aref confidences (funcall label-fn example)))))))
    (values sum (length examples))))

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
  ((name :initform '("cross entropy"))))


;;;; ROC

(defun roc-auc (seq pred &key (key #'identity))
  "Return the area under the ROC curve for the dataset represented by
SEQ. PRED is a predicate function for deciding whether an element of
SEQ belongs to the class in question. KEY returns the a number for
each element which is the predictor's idea of how much that element is
likely to belong to the class, it's not necessarily a probability."
  ;; AUC is equal to the probability of a randomly chosen positive
  ;; having higher KEY (score) than a randomly chosen negative
  ;; element.
  (let ((sum 0)
        (seq (sort (copy-seq seq) #'> :key key))
        (prev-pos-count 0)
        (prev-neg-count 0)
        (pos-count 0)
        (neg-count 0)
        (prev-score nil))
    (flet ((add ()
             (incf sum (*
                        ;; In regions of equal scores go with the
                        ;; expectation.
                        (/ (+ pos-count prev-pos-count) 2)
                        (- neg-count prev-neg-count)))))
      (map nil
           (lambda (x)
             (let ((score (funcall key x)))
               (when (or (null prev-score) (/= prev-score score))
                 (add)
                 (setq prev-score score
                       prev-pos-count pos-count
                       prev-neg-count neg-count)))
             (if (funcall pred x)
                 (incf pos-count)
                 (incf neg-count)))
           seq)
      (add))
    (if (or (zerop pos-count) (zerop neg-count))
        (values nil nil nil)
        (values (/ sum pos-count neg-count) pos-count neg-count))))

(defun test-roc-auc ()
  (assert (= 8/9
             (roc-auc '((9 . t) (8 . t) (7 . nil) (6 . t) (5 . nil) (4 . nil))
                      #'cdr :key #'car))))

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

