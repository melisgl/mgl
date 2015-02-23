(in-package :mgl-core)

(defsection @mgl-core (:title "Core")
  (@mgl-persistence section)
  (@mgl-model-stripe section)
  (@mgl-executors section))


(defsection @mgl-persistence (:title "Persistence")
  (load-state function)
  (save-state function)
  (read-state function)
  (write-state function)
  (read-state* generic-function)
  (write-state* generic-function))

(defun load-state (filename object)
  "Load weights of OBJECT from FILENAME. Return OBJECT."
  (with-open-file (stream filename
                          #+sbcl :element-type #+sbcl :default)
    (read-state object stream)
    (assert (= (file-position stream) (file-length stream)) ()
            "LOAD-STATE left ~D bytes unread."
            (- (file-length stream) (file-position stream)))
    object))

(defun save-state (filename object &key (if-exists :error)
                   (ensure t))
  "Save weights of OBJECT to FILENAME. If ENSURE, then
  ENSURE-DIRECTORIES-EXIST is called on FILENAME. IF-EXISTS is passed
  on to OPEN. Return OBJECT."
  (when ensure
    (ensure-directories-exist filename))
  (with-open-file (stream filename :direction :output
                          :if-does-not-exist :create
                          :if-exists if-exists
                          #+sbcl :element-type #+sbcl :default)
    (write-state object stream)))

(defun read-state (object stream)
  "Read the weights of OBJECT from the bivalent STREAM where weights
  mean the learnt parameters. There is currently no sanity checking of
  data which will most certainly change in the future together with
  the serialization format. Return OBJECT."
  (read-state* object stream (make-hash-table))
  object)

(defun write-state (object stream)
  "Write weight of OBJECT to the bivalent STREAM. Return OBJECT."
  (write-state* object stream (make-hash-table))
  object)

(defgeneric read-state* (object stream context)
  (:documentation "This is the extension point for READ-STATE. It is
  guaranteed that primary READ-STATE* methods will be called only once
  for each OBJECT (under EQ). CONTEXT is an opaque object and must be
  passed on to any recursive READ-STATE* calls.")
  (:method :around (object stream context)
    (unless (gethash object context)
      (setf (gethash object context) t)
      (call-next-method))))

(defgeneric write-state* (object stream context)
  (:documentation "This is the extension point for WRITE-STATE. It is
  guaranteed that primary WRITE-STATE* methods will be called only
  once for each OBJECT (under EQ). CONTEXT is an opaque object and must
  be passed on to any recursive WRITE-STATE* calls.")
  (:method :around (object stream context)
    (unless (gethash object context)
      (setf (gethash object context) t)
      (call-next-method))))


(defsection @mgl-model-stripe (:title "Batch Processing")
  "Processing instances one by one during training or prediction can
  be slow. The models that support batch processing for greater
  efficiency are said to be _striped_.

  Typically, during or after creating a model, one sets MAX-N-STRIPES
  on it a positive integer. When a batch of instances is to be fed to
  the model it is first broken into subbatches of length that's at
  most MAX-N-STRIPES. For each subbatch, SET-INPUT (FIXDOC) is called
  and a before method takes care of setting N-STRIPES to the actual
  number of instances in the subbatch. When MAX-N-STRIPES is set
  internal data structures may be resized which is an expensive
  operation. Setting N-STRIPES is a comparatively cheap operation,
  often implemented as matrix reshaping.

  Note that for models made of different parts (for example,
  [MGL-BP:BPN][CLASS] consists of [MGL-BP:LUMP][]s) , setting these
  values affects the constituent parts, but one should never change
  the number stripes of the parts directly because that would lead to
  an internal inconsistency in the model."
  (max-n-stripes generic-function)
  (set-max-n-stripes generic-function)
  (n-stripes generic-function)
  (set-n-stripes generic-function)
  (with-stripes macro)
  (stripe-start generic-function)
  (stripe-end generic-function)
  (set-input generic-function)
  (map-batches-for-model function)
  (do-batches-for-model macro))

(defgeneric max-n-stripes (object)
  (:documentation "The number of stripes with which the OBJECT is
  capable of dealing simultaneously. "))

(defgeneric set-max-n-stripes (max-n-stripes object)
  (:documentation "Allocate the necessary stuff to allow for
  MAX-N-STRIPES number of stripes to be worked with simultaneously in
  OBJECT. This is called when MAX-N-STRIPES is SETF'ed."))

(defsetf max-n-stripes (object) (store)
  `(set-max-n-stripes ,store ,object))

(defgeneric n-stripes (object)
  (:documentation "The number of stripes currently present in OBJECT.
  This is at most MAX-N-STRIPES."))

(defgeneric set-n-stripes (n-stripes object)
  (:documentation "Set the number of stripes (out of MAX-N-STRIPES)
  that are in use in OBJECT. This is called when N-STRIPES is
  SETF'ed."))

(defsetf n-stripes (object) (store)
  `(set-n-stripes ,store ,object))

(defmacro with-stripes (specs &body body)
  "Bind start and optionally end indices belonging to stripes in
  striped objects.

      (WITH-STRIPES ((STRIPE1 OBJECT1 START1 END1)
                     (STRIPE2 OBJECT2 START2)
                     ...)
       ...)

  This is how one's supposed to find the index range corresponding to
  the Nth input in an input lump of a bpn:

       (with-stripes ((n input-lump start end))
         (loop for i upfrom start below end
               do (setf (mref (nodes input-lump) i) 0d0)))

  Note how the input lump is striped, but the matrix into which we are
  indexing (NODES) is not known to WITH-STRIPES. In fact, for lumps
  the same stripe indices work with NODES and MGL-BP:DERIVATIVES."
  `(let* ,(mapcan (lambda (spec) (apply #'stripe-binding spec))
                  specs)
     ,@body))

(eval-when (:compile-toplevel :load-toplevel :execute)
  (defun stripe-binding (stripe object start &optional end)
    (alexandria:with-gensyms (%stripe %object)
      `((,%stripe ,stripe)
        (,%object ,object)
        (,start (the index (stripe-start ,%stripe ,%object)))
        ,@(when end `((,end (the index (stripe-end ,%stripe ,%object)))))))))

(defgeneric stripe-start (stripe object)
  (:documentation "Return the start index of STRIPE in some array or
  matrix of OBJECT."))

(defgeneric stripe-end (stripe object)
  (:documentation "Return the end index (exclusive) of STRIPE in some
  array or matrix of OBJECT."))

(defgeneric set-input (instances model)
  (:documentation "Set INSTANCES as inputs in MODEL. SAMPLES is always
  a SEQUENCE of instances even for models not capable of batch
  operation. It sets N-STRIPES to (LENGTH INSTANCES) in a :BEFORE
  method."))

(defun map-batches-for-model (fn dataset model)
  "Call FN with batches of instances from DATASET suitable for MODEL.
  The number of instances in a batch is MAX-N-STRIPES of MODEL or less
  if there are no more instances left."
  (let ((sampler (if (typep dataset 'sequence)
                     (make-sequence-sampler dataset
                                            :max-n-samples (length dataset))
                     dataset)))
    (loop until (finishedp sampler) do
      (funcall fn (list-samples sampler (max-n-stripes model))))))

(defmacro do-batches-for-model ((batch (dataset model)) &body body)
  "Convenience macro over MAP-BATCHES-FOR-MODEL."
  `(map-batches-for-model (lambda (,batch) ,@body) ,dataset ,model))


(defsection @mgl-executors (:title "Executors")
  (map-over-executors generic-function)
  (do-executors macro)
  (@mgl-parameterized-executor-cache section))

(defgeneric map-over-executors (fn instances prototype-executor)
  (:documentation "Divide INSTANCES between executors that perform the
  same function as PROTOTYPE-EXECUTOR and call FN with the instances
  and the executor for which the instances are.

  Some objects conflate function and call: the forward pass of a
  [MGL-BP:BPN][class] computes output from inputs so it is like a
  function but it also doubles as a function call in the sense that
  the bpn (function) object changes state during the computation of
  the output. Hence not even the forward pass of a bpn is thread safe.
  There is also the restriction that all inputs must be of the same
  size.

  For example, if we have a function that builds bpn a for an input of
  a certain size, then we can create a factory that creates bpns for a
  particular call. The factory probably wants keep the weights the
  same though. In @MGL-PARAMETERIZED-EXECUTOR-CACHE,
  MAKE-EXECUTOR-WITH-PARAMETERS is this factory.

  Parallelization of execution is another possibility
  MAP-OVER-EXECUTORS allows, but there is no prebuilt solution for it,
  yet.

  The default implementation simply calls FN with INSTANCES and
  PROTOTYPE-EXECUTOR.")
  (:method (fn instances object)
    (funcall fn instances object)))

(defmacro do-executors ((instances object) &body body)
  "Convenience macro on top of MAP-OVER-EXECUTORS."
  `(map-over-executors (lambda (,instances ,object)
                         ,@body)
                       ,instances ,object))


(defsection @mgl-parameterized-executor-cache
    (:title "Parameterized Executor Cache")
  (parameterized-executor-cache-mixin class)
  (make-executor-with-parameters generic-function)
  (instance-to-executor-parameters generic-function))

(defclass parameterized-executor-cache-mixin ()
  ((executor-cache
    :initform (make-hash-table :test #'equal)
    :reader executor-cache))
  (:documentation "Mix this into a model, implement
  INSTANCE-TO-EXECUTOR-PARAMETERS and MAKE-EXECUTOR-WITH-PARAMETERS
  and DO-EXECUTORS will be to able build executors suitable for
  different instances. The canonical example is using a BPN to compute
  the means and convariances of a gaussian process. Since each
  instance is made of a variable number of observations, the size of
  the input is not constant, thus we have a bpn (an executor) for each
  input dimension (the parameters)."))

(defgeneric make-executor-with-parameters (parameters cache)
  (:documentation "Create a new executor for PARAMETERS. CACHE is a
  PARAMETERIZED-EXECUTOR-CACHE-MIXIN. In the BPN gaussian process
  example, PARAMETERS would be a list of input dimensions."))

(defgeneric instance-to-executor-parameters (instance cache)
  (:documentation "Return the parameters for an executor able to
  handle INSTANCE. Called by MAP-OVER-EXECUTORS on CACHE (that's a
  PARAMETERIZED-EXECUTOR-CACHE-MIXIN). The returned parameters are
  keys in an EQUAL parameters->executor hash table."))

(defun lookup-executor-cache (parameters cache)
  (gethash parameters (executor-cache cache)))

(defun insert-into-executor-cache (parameters cache value)
  (setf (gethash parameters (executor-cache cache)) value))

(defmethod map-over-executors (fn instances
                               (c parameterized-executor-cache-mixin))
  (trivially-map-over-executors fn instances c))

(defun trivially-map-over-executors (fn instances obj)
  (let ((executor-to-instances (make-hash-table)))
    (dolist (instance instances)
      (let ((executor (find-one-executor instance obj)))
        (push instance (gethash executor executor-to-instances))))
    (maphash (lambda (executor instances)
               (funcall fn instances executor))
             executor-to-instances)))

(defgeneric find-one-executor (instance obj)
  (:method (instance obj)
    nil)
  (:method :around (instance obj)
    (or (call-next-method)
        obj))
  (:method (instance (cached parameterized-executor-cache-mixin))
    (let ((parameters (instance-to-executor-parameters instance cached)))
      (or (lookup-executor-cache parameters cached)
          (let ((executor (make-executor-with-parameters parameters cached)))
            (when executor
              (insert-into-executor-cache parameters cached executor))
            executor)))))
