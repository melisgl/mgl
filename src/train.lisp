(in-package :mgl-train)

;;;; Interface

(defgeneric sample (sampler)
  (:documentation "The SAMPLER - if not FINISHEDP - returns on object
that represents a sample from the world to be experienced or in other
words simply something the can be used as input for the learning."))

(defgeneric finishedp (sampler)
  (:documentation "See if SAMPLER has run out of examples."))

(defgeneric initialize-trainer (trainer learner)
  (:documentation "To be called before training starts this function
sets up TRAINER to be suitable for LEARNER. Normally called
automatically from a :BEFORE method on TRAIN."))

(defgeneric train (sampler trainer learner)
  (:method :before (sampler trainer learner)
           (declare (ignore sampler))
           (initialize-trainer trainer learner))
  (:documentation "Train LEARNER with TRAINER on the examples from
SAMPLER. Before that TRAINER is initialized for LEARNER with
INITIALIZE-TRAINER. Training continues until SAMPLER is finished."))

(defgeneric train-batch (batch trainer learner)
  (:documentation "Called by TRAIN. Useful to hang an around method on
to monitor progress."))

(defgeneric set-input (samples learner)
  (:documentation "Set SAMPLES as inputs in LEARNER. SAMPLES is always
a sequence of examples even for learners not capable of batch
operation."))

(defgeneric n-inputs-until-update (trainer)
  (:documentation "Return the largest number of inputs guaranteed not
to cause a change in the learner being trained."))


;;;; Samplers

(defclass function-sampler ()
  ((sampler :initarg :sampler :accessor sampler)))

(defmethod finishedp ((sampler function-sampler))
  nil)

(defmethod sample ((sampler function-sampler))
  (funcall (sampler sampler)))

(defclass counting-sampler ()
  ((n-samples :initform 0 :initarg :n-inputs :accessor n-samples)
   (max-n-samples :initform nil :initarg :max-n-samples
                  :accessor max-n-samples))
  (:documentation "Keep track of how many samples have been generated
and say FINISHEDP if it's not less than MAX-N-INPUTS (that is
optional)."))

(defmethod finishedp ((sampler counting-sampler))
  (let ((max-n-samples (max-n-samples sampler)))
    (and max-n-samples
         (<= max-n-samples (n-samples sampler)))))

(defmethod sample ((sampler counting-sampler))
  (incf (n-samples sampler))
  (call-next-method))

(defclass counting-function-sampler (counting-sampler function-sampler) ())

(defun sample-batch (sampler max-size)
  "Return a sequence of samples of length at most MAX-SIZE or less if
SAMPLER runs out."
  (loop repeat max-size
        while (not (finishedp sampler))
        collect (sample sampler)))


;;;; Error counter

(defclass error-counter ()
  ((sum-errors
    :initform #.(flt 0) :reader sum-errors
    :documentation "The sum of errors.")
   (n-sum-errors
    :initform 0 :reader n-sum-errors
    :documentation "The total number of observations whose errors
contributed to SUM-ERROR.")))

(defclass rmse-counter (error-counter) ())

(defgeneric add-error (counter err n)
  (:documentation "Add ERR to SUM-ERROR and N to N-SUM-ERRORS.")
  (:method ((counter error-counter) err n)
    (incf (slot-value counter 'sum-errors) err)
    (incf (slot-value counter 'n-sum-errors) n)))

(defgeneric reset-counter (counter)
  (:method ((counter error-counter))
    (with-slots (sum-errors n-sum-errors) counter
      (setf sum-errors #.(flt 0))
      (setf n-sum-errors 0))))

(defgeneric get-error (counter)
  (:method ((counter error-counter))
    (with-slots (sum-errors n-sum-errors) counter
      (values (if (zerop n-sum-errors)
                  nil
                  (/ sum-errors n-sum-errors))
              n-sum-errors)))
  (:method ((counter rmse-counter))
    (multiple-value-bind (e n) (call-next-method)
      (if e
          (values (sqrt e) n)
          nil))))


;;;; Stripes

(defgeneric max-n-stripes (learner)
  (:documentation "The number of examples with which the learner is
capable of dealing simultaneously."))

(defgeneric set-max-n-stripes (max-n-stripes object)
  (:documentation "Allocate the necessary stuff to allow for N-STRIPES
number of examples to be worked with simultaneously."))

(defsetf max-n-stripes (object) (store)
  `(set-max-n-stripes ,store ,object))

(defgeneric n-stripes (learner)
  (:documentation "The number of examples with which the learner is
currently dealing."))

(defgeneric set-n-stripes (n-stripes object)
  (:documentation "Set the number of stripes \(out of MAX-N-STRIPES)
that are in use."))

(defsetf n-stripes (object) (store)
  `(set-n-stripes ,store ,object))

(defgeneric stripe-start (stripe obj))
(defgeneric stripe-end (stripe obj))

(eval-when (:compile-toplevel :load-toplevel :execute)
  (defun stripe-binding (stripe obj start &optional end)
    (with-gensyms (%stripe %obj)
      `((,%stripe ,stripe)
        (,%obj ,obj)
        (,start (the index (stripe-start ,%stripe ,%obj)))
        ,@(when end `((,end (the index (stripe-end ,%stripe ,%obj)))))))))

(defmacro with-stripes (specs &body body)
  `(let* ,(mapcan (lambda (spec) (apply #'stripe-binding spec))
                  specs)
     ,@body))


;;;; Various accessor type generic functions share by packages.

(defgeneric name (object))
(defgeneric size (object))
(defgeneric nodes (object))
(defgeneric default-value (object))
(defgeneric group-size (object))
(defgeneric batch-size (object))
(defgeneric n-inputs (object))
