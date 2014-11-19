(in-package mgl-dataset)

(defsection @mgl-dataset (:title "Datasets")
  "An instance can often be any kind of object of the user's choice.
  It is typically represented by a set of numbers which is called a
  feature vector or by a structure holding the feature vector, the
  label, etc. A dataset is a SEQUENCE of such instances or a
  @MGL-SAMPLER object that produces instances."
  (map-dataset function)
  (map-datasets function)
  (@mgl-sampler section))

(defun ensure-sampler (dataset)
  (if (typep dataset 'sequence)
      (make-sequence-sampler dataset :max-n-samples (length dataset))
      dataset))

(defun map-dataset (fn dataset)
  "Call FN with each instance in DATASET. This is basically equivalent
  to iterating over the elements of a sequence or a sampler (see
  @MGL-SAMPLER)."
  (let ((sampler (ensure-sampler dataset)))
    (loop until (finishedp sampler) do
      (funcall fn (sample sampler)))))

(defun map-datasets (fn datasets &key (impute nil imputep))
  "Call FN with a list of instances, one from each dataset in
  DATASETS. Return nothing. If IMPUTE is specified then iterate until
  the largest dataset is consumed imputing IMPUTE for missing values.
  If IMPUTE is not specified then iterate until the smallest dataset
  runs out.

  ```cl-transcript
  (map-datasets #'prin1 '((0 1 2) (:a :b)))
  .. (0 :A)(1 :B)
  
  (map-datasets #'prin1 '((0 1 2) (:a :b)) :impute nil)
  .. (0 :A)(1 :B)(2 NIL)
  ```

  It is of course allowed to mix sequences with samplers:

  ```cl-transcript
  (map-datasets #'prin1
                (list '(0 1 2)
                      (make-sequence-sampler '(:a :b) :max-n-samples 2)))
  .. (0 :A)(1 :B)
  ```"
  (let ((samplers (map 'vector #'ensure-sampler datasets)))
    (loop
      (let* ((seen-finished-sampler-p nil)
             (seen-live-sampler-p nil)
             (instances
               (map 'list (lambda (sampler)
                            (cond ((finishedp sampler)
                                   (setq seen-finished-sampler-p t)
                                   impute)
                                  (t
                                   (setq seen-live-sampler-p t)
                                   (sample sampler))))
                    samplers)))
        (when (or (not seen-live-sampler-p)
                  (and (not imputep)
                       seen-finished-sampler-p))
          (return))
        (funcall fn instances)))
    (values)))

(defsection @mgl-sampler (:title "Samplers")
  "Some algorithms do not need random access to the entire dataset and
  can work with a stream observations. Samplers are simple generators
  providing two functions: SAMPLE and FINISHEDP."
  (sample generic-function)
  (finishedp generic-function)
  (list-samples function)
  (make-sequence-sampler function)
  (make-random-sampler function)
  (*infinitely-empty-dataset* variable)
  (@mgl-sampler-function-sampler section))

(defgeneric sample (sampler)
  (:documentation "If SAMPLER has not run out of data (see FINISHEDP)
  SAMPLE returns an object that represents a sample from the world to
  be experienced or, in other words, simply something the can be used
  as input for training or prediction. It is not allowed to call
  SAMPLE if SAMPLER is FINISHEDP."))

(defgeneric finishedp (sampler)
  (:documentation "See if SAMPLER has run out of examples."))

(defun list-samples (sampler max-size)
  "Return a list of samples of length at most MAX-SIZE or less if
  SAMPLER runs out."
  (loop repeat max-size
        while (not (finishedp sampler))
        collect (sample sampler)))

(defun make-sequence-sampler (seq &key max-n-samples)
  "Create a sampler that returns elements of SEQ in their original
  order. If MAX-N-SAMPLES is non-nil, then at most MAX-N-SAMPLES are
  sampled."
  (make-instance 'function-sampler
                 :max-n-samples max-n-samples
                 :generator (make-sequence-generator seq)))

(defun make-random-sampler (seq &key max-n-samples
                            (reorder #'mgl-resample:shuffle))
  "Create a sampler that returns elements of SEQ in random order. If
  MAX-N-SAMPLES is non-nil, then at most MAX-N-SAMPLES are sampled.
  The first pass over a shuffled copy of SEQ, and this copy is
  reshuffled whenever the sampler reaches the end of it. Shuffling is
  performed by calling the REORDER function."
  (make-instance 'function-sampler
                 :max-n-samples max-n-samples
                 :generator (make-random-generator seq :reorder reorder)))


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
