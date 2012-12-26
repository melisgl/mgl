(in-package :mgl-train)

;;;; Various accessor type generic functions shared by packages

(defgeneric name (object))
(defgeneric size (object))
(defgeneric nodes (object))
(defgeneric default-value (object))
(defgeneric group-size (object))
(defgeneric batch-size (object))
(defgeneric n-inputs (object))


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
  (:method :around (sampler trainer learner)
           (declare (ignore sampler))
           (initialize-trainer trainer learner)
           (call-next-method))
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

(defmethod print-object ((sampler counting-sampler) stream)
  (pprint-logical-block (stream ())
    (print-unreadable-object (sampler stream :type t :identity t)
      (format stream "~S/~S" (n-samples sampler) (max-n-samples sampler))))
  sampler)

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

(defclass counter ()
  ((name :initform () :initarg :name :reader name)))

(defmethod initialize-instance :after ((counter counter) &key
                                       (prepend-name nil prepend-name-p)
                                       &allow-other-keys)
  (when prepend-name-p
    (push prepend-name (slot-value counter 'name))))

(defclass error-counter (counter)
  ((sum-errors
    :initform #.(flt 0) :reader sum-errors
    :documentation "The sum of errors.")
   (n-sum-errors
    :initform 0 :reader n-sum-errors
    :documentation "The total number of observations whose errors
contributed to SUM-ERROR.")))

(defgeneric print-counter (counter stream))

(defmethod print-counter ((counter error-counter) stream)
  (multiple-value-bind (e c) (get-error counter)
    (if e
        (format stream "~,5E (~D)" e c)
        (format stream "~A (~D)" e c))))

(defclass misclassification-counter (error-counter)
  ((name :initform '("classification accuracy"))))

(defmethod print-counter ((counter misclassification-counter) stream)
  (multiple-value-bind (e c) (get-error counter)
    (if e
        (format stream "~,2F% (~D)"
                (* 100 (- 1 e)) c)
        (format stream "~A (~D)" e c))))

(defclass rmse-counter (error-counter)
  ((name :initform '("rmse"))))

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
                  0
                  (/ sum-errors n-sum-errors))
              n-sum-errors)))
  (:method ((counter rmse-counter))
    (multiple-value-bind (e n) (call-next-method)
      (values (sqrt e) n))))

(defmethod print-object ((counter counter) stream)
  (pprint-logical-block (stream ())
    (flet ((foo ()
             (when (slot-boundp counter 'name)
               (format stream "~{~A~^ ~:_~}: ~:_"
                       (alexandria:ensure-list (name counter))))
             (print-counter counter stream)))
      (if *print-escape*
          (print-unreadable-object (counter stream :type t)
            (foo))
          (foo))))
  counter)


;;;; Stripes
;;;;
;;;; For batch processing, objects (typically inputs or objects that
;;;; hold [intermediate] results of a computation) can be striped.
;;;; Each stripe is identified by an index in [0,MAX-N-STRIPES-1].

(defgeneric max-n-stripes (object)
  (:documentation "The number of stripes with which the OBJECT is
capable of dealing simultaneously."))

(defgeneric set-max-n-stripes (max-n-stripes object)
  (:documentation "Allocate the necessary stuff to allow for N-STRIPES
number of stripes to be worked with simultaneously in OBJECT."))

(defsetf max-n-stripes (object) (store)
  `(set-max-n-stripes ,store ,object))

(defgeneric n-stripes (object)
  (:documentation "The number of stripes currently present in OBJECT.
This is at most MAX-N-STRIPES."))

(defgeneric set-n-stripes (n-stripes object)
  (:documentation "Set the number of stripes \(out of MAX-N-STRIPES)
that are in use in OBJECT."))

(defsetf n-stripes (object) (store)
  `(set-n-stripes ,store ,object))

(defgeneric stripe-start (stripe object)
  (:documentation "Return the start of STRIPE in OBJECT, that's
usually an index into some kind of storage that backs OBJECT."))

(defgeneric stripe-end (stripe object)
  (:documentation "Return the end of STRIPE in OBJECT, that's usually
an index into some kind of storage that backs OBJECT."))

(eval-when (:compile-toplevel :load-toplevel :execute)
  (defun stripe-binding (stripe object start &optional end)
    (with-gensyms (%stripe %object)
      `((,%stripe ,stripe)
        (,%object ,object)
        (,start (the index (stripe-start ,%stripe ,%object)))
        ,@(when end `((,end (the index (stripe-end ,%stripe ,%object)))))))))

(defmacro with-stripes (specs &body body)
  "Bind start and optionally end indices of belonging to stripes in
striped objects.

 (WITH-STRIPE ((STRIPE1 OBJECT1 START1 END1)
               (STRIPE2 OBJECT2 START2 END2)
               ...)
  ...)"
  `(let* ,(mapcan (lambda (spec) (apply #'stripe-binding spec))
                  specs)
     ,@body))


;;;; Collecting errors

(defun map-batches-for-learner (fn sampler learner)
  "Call FN with batches of samples suitable for LEARNER. The number of
samples in a batch is MAX-N-STRIPES of LEARNER or less if SAMPLER runs
out."
  (loop until (finishedp sampler) do
    (funcall fn (sample-batch sampler (max-n-stripes learner)))))

(defmacro do-batches-for-learner ((samples (sampler learner)) &body body)
  "Convenience macro over MAP-BATCHES-FOR-LEARNER."
  `(map-batches-for-learner (lambda (,samples) ,@body) ,sampler ,learner))

(defun add-measured-error (counter-and-measurer &rest args)
  (multiple-value-call #'add-error
    (car counter-and-measurer)
    (apply (cdr counter-and-measurer) args)))

(defun apply-counters-and-measurers (counters-and-measurers &rest args)
  "Add the errors measured by the measurers to the counters."
  (map nil
       (lambda (counter-and-measurer)
         (apply #'add-measured-error counter-and-measurer args))
       counters-and-measurers)
  counters-and-measurers)

(defun collect-batch-errors (fn sampler learner counters-and-measurers)
  "Sample from SAMPLER until it runs out. Call FN with each batch of samples.
COUNTERS-AND-MEASURERS is a sequence of conses of a counter and
function. The function takes one parameter: a sequence of samples and
is called after each call to FN. Measurers return two values: the
cumulative error and the counter, suitable as the second and third
argument to ADD-ERROR. Finally, return the counters. Return the list
of counters from COUNTERS-AND-MEASURERS."
  (when counters-and-measurers
    (do-batches-for-learner (samples (sampler learner))
      (funcall fn samples)
      (apply-counters-and-measurers counters-and-measurers samples learner)))
  (map 'list #'car counters-and-measurers))
