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

(defgeneric train (sampler trainer learner &key &allow-other-keys)
  (:method :before (sampler trainer learner &rest args)
           (declare (ignore sampler args))
           (initialize-trainer trainer learner))
  (:method (sampler trainer learner &rest args)
    (loop until (finishedp sampler) do
          (apply #'train-one (sample sampler) trainer learner args))
    learner)
  (:documentation "Train LEARNER with TRAINER on the examples from
SAMPLER. Before that TRAINER is initialized for LEARNER with
INITIALIZE-TRAINER. Training continues until SAMPLER is finished. For
trainers that inherit from BATCH-TRAINER BATCH-SIZE number of samples
are collected and passed to TRAIN-BATCH. For other trainers samples
are passed one by one to TRAIN-ONE."))

(defgeneric set-input (sample learner)
  (:documentation "Set SAMPLE as INPUT in LEARNER."))

(defgeneric train-one (sample trainer learner &key &allow-other-keys)
  (:documentation "Train LEARNER by TRAINER on one SAMPLE. This method
usually starts with calling SET-INPUT."))


;;;; Batch training

(defclass batch-trainer ()
  ((batch-size :initarg :batch-size :accessor batch-size))
  (:documentation "Batch trainers collect BATCH-SIZE number of samples
and learn and train on them as a whole, possibly multiple times. Note
that some trainers that really work on batches are not subclassed from
this as they only use each sample once so TRAIN-ONE suffices for their
needs."))

(defgeneric train-batch (batch trainer learner &key &allow-other-keys)
  (:documentation "Batch is a sequence of samples that can be used as
input for LEARNER any number of times and in any order desired."))

(defmethod train (sampler (trainer batch-trainer) learner &rest args)
  (loop until (finishedp sampler) do
        (let ((batch (loop repeat (batch-size trainer)
                           until (finishedp sampler)
                           collect (sample sampler))))
          (apply #'train-batch batch trainer learner args))))


;;;; Samplers

(defclass function-sampler ()
  ((sampler :initarg :sampler :accessor sampler)))

(defmethod finishedp ((sampler function-sampler))
  (declare (ignore sampler))
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
