(in-package mgl-dataset)

(defsection @mgl-dataset (:title "Dataset")
  "Ultimately machine learning is about creating models of some
  domain. The observations in the modelled domain are called
  _instances_. Sets of instances are called _datasets_. Datasets are
  used when fitting a model or when making predictions.

  Implementationally speaking, an instance is typically represented by
  a set of numbers which is called _feature vector_. A dataset is a
  SEQUENCE of such instances or a @MGL-SAMPLER object that produces
  instances."
  (@mgl-sampler section))

;;;; Rename to generator?

(defsection @mgl-sampler (:title "Sampler")
  "Some algorithms do not need random access to the entire dataset and
  can work with a stream observations. Samplers are simple generators
  providing two functions: SAMPLE and FINISHEDP."
  (sample generic-function)
  (finishedp generic-function)
  (list-samples function)
  (make-sequence-sampler function)
  (*infinitely-empty-dataset* variable)
  (@mgl-sampler-function-sampler section))

(defgeneric sample (sampler)
  (:documentation "If not SAMPLER has not run out of data (see
  FINISHEDP) SAMPLE returns an object that represents a sample from
  the world to be experienced or, in other words, simply something the
  can be used as input for training or prediction."))

(defgeneric finishedp (sampler)
  (:documentation "See if SAMPLER has run out of examples."))

(defun list-samples (sampler max-size)
  "Return a list of samples of length at most MAX-SIZE or less if
  SAMPLER runs out."
  (loop repeat max-size
        while (not (finishedp sampler))
        collect (sample sampler)))

(defun make-sequence-sampler (seq)
  "A simple sampler that returns elements of SEQ once, in order."
  (make-instance 'function-sampler
                 :max-n-samples (length seq)
                 :generator (make-seq-generator seq)))


(defsection @mgl-sampler-function-sampler (:title "Function Sampler")
  (function-sampler class)
  (generator (reader function-sampler))
  (max-n-samples (accessor function-sampler))
  (name (reader function-sampler))
  (n-samples (reader function-sampler)))


(defclass function-sampler ()
  ((generator
    :initarg :generator
    :reader generator
    :documentation "A generator function of no arguments that returns
    the next sample.")
   (n-samples
    :initform 0
    :initarg :n-samples
    :reader n-samples
    :documentation "")
   (max-n-samples
    :initform nil
    :initarg :max-n-samples
    :accessor max-n-samples)
   (name
    :initform nil
    :initarg :name
    :reader name
    :documentation "An arbitrary object naming the sampler. Only used
    for printing the sampler object."))
  (:documentation "A sampler with a function in its GENERATOR that
  produces a stream of samples which may or may not be finite
  depending on MAX-N-SAMPLES. FINISHEDP returns T iff MAX-N-SAMPLES is
  non-nil, and it's not greater than the number of samples
  generated (N-SAMPLES).

      (list-samples (make-instance 'function-sampler
                                   :generator (lambda ()
                                                (random 10))
                                   :max-n-samples 5)
                    10)
      => (3 5 2 3 3)"))

(defmethod finishedp ((sampler function-sampler))
  (let ((max-n-samples (max-n-samples sampler)))
    (and max-n-samples
         (<= max-n-samples (n-samples sampler)))))

(defmethod sample ((sampler function-sampler))
  (incf (slot-value sampler 'n-samples))
  (funcall (generator sampler)))

(defmethod print-object ((sampler function-sampler) stream)
  (pprint-logical-block (stream ())
    (print-unreadable-object (sampler stream :type t)
      (when (name sampler)
        (format stream "~S " (name sampler)))
      (when (max-n-samples sampler)
        (format stream "~S/~S" (n-samples sampler) (max-n-samples sampler)))))
  sampler)

(defvar *infinitely-empty-dataset* (make-instance 'function-sampler
                                                  :generator (constantly nil)
                                                  :name "infinitely empty")
  "This is the default dataset for MGL-OPT:MINIMIZE. It's an infinite
  stream of NILs.")
