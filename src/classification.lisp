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
can call work with counters and measurers."))

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
