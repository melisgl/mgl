;;;; TODO:
;;;;
;;;; * factored RBM: weight matrix is A*B?
;;;;
;;;; * conditioning chunks are not really visible nor hidden
;;;;
;;;; * semi-restricted (connections between visibles)
;;;;
;;;; * higher order rbm
;;;;
;;;; * training with conjugate gradient: is it possible to calculate
;;;; the cost function?

(in-package :mgl-rbm)

;;;; Chunk

(defclass chunk ()
  ((name :initform (gensym) :initarg :name :reader name)
   (inputs
    :type (or flt-vector null) :reader inputs
    :documentation "This is where SET-INPUT shall store the input \(it
may also set INDICES-PRESENT). It is NIL in CONSTANT-CHUNKS.")
   (nodes
    :type flt-vector :reader nodes
    :documentation "A value for each node in the chunk. First
activations are put here (inputs*weights) then the mean of the
probability distribution is calculated from the activation and finally
\(optionally) a sample is taken from the probability distribution. All
these values are stored in this vector.")
   (indices-present
    :initform nil :initarg :indices-present :type (or null index-vector)
    :accessor indices-present
    :documentation "NIL or a simple vector of array indices into the
layer's NODES. Need not be ordered."))
  (:documentation "Base class for different chunks. A chunk is a set
of nodes of the same type."))

(declaim (inline chunk-size))
(defun chunk-size (chunk)
  (length (the flt-vector (nodes chunk))))

(defmethod print-object ((chunk chunk) stream)
  (print-unreadable-object (chunk stream :type t :identity t)
    (format stream "~S ~S" (name chunk) (chunk-size chunk)))
  chunk)

(defun ->chunk (chunk-designator chunks)
  (if (typep chunk-designator 'chunk)
      chunk-designator
      (or (find chunk-designator chunks :key #'name :test #'equal)
          (error "Cannot find chunk ~S." chunk-designator))))

(defmacro do-chunk ((var chunk) &body body)
  "Iterate over the indices of nodes of CHUNK skipping missing ones."
  (with-gensyms (%chunk %indices-present %size)
    `(let* ((,%chunk ,chunk)
            (,%indices-present (indices-present ,%chunk)))
       (if ,%indices-present
           (locally (declare (type index-vector ,%indices-present))
             (loop for ,var across ,%indices-present
                   do (progn ,@body)))
           (let ((,%size (chunk-size ,%chunk)))
             (declare (type index ,%size))
             (loop for ,var fixnum below ,%size
                   do ,@body))))))

(defclass conditioning-chunk (chunk) ()
  (:documentation "Nodes in CONDITIONING-CHUNK never change their
values on their own so they are to be clamped. Including this chunk in
the visible layer allows `conditional' RBMs."))

(defun conditioning-chunk-p (chunk)
  (typep chunk 'conditioning-chunk))

(defmethod initialize-instance :after ((chunk chunk)
                                       &key (size 1) &allow-other-keys)
  (setf (slot-value chunk 'nodes) (make-flt-array size))
  (setf (slot-value chunk 'inputs)
        (if (typep chunk 'constant-chunk)
            nil
            (make-flt-array size))))

(defclass constant-chunk (conditioning-chunk)
  ((value :initform #.(flt 1) :initarg :value :type flt :reader value))
  (:documentation "A special kind of CONDITIONING-CHUNK whose NODES
are always VALUE. This conveniently allows biases in the opposing
layer."))

(defmethod initialize-instance :after ((chunk constant-chunk)
                                       &key &allow-other-keys)
  (fill (nodes chunk) (value chunk)))

(defclass sigmoid-chunk (chunk) ()
  (:documentation "Nodes in a sigmoid chunk have two possible samples:
0 and 1. The probability of a node being on is given by the sigmoid of
its activation."))

(defclass gaussian-chunk (chunk) ()
  (:documentation "Nodes are real valued. The sample of a node is its
activation plus guassian noise of unit variance."))

(defclass normalized-group-chunk ()
  ((scale
    :initform #.(flt 1) :initarg :scale :accessor scale :type flt
    :documentation "The sum of the means after normalization. Can be
changed during training, for instance when clamping.")
   (group-size
    :initform (error "GROUP-SIZE must be specified.")
    :initarg :group-size
    :reader group-size))
  (:documentation "Means are normalized to SCALE within groups of
GROUP-SIZE."))

(defclass exp-normalized-group-chunk (normalized-group-chunk) ()
  (:documentation "Means are normalized (EXP ACTIVATION)."))

(defclass softmax-chunk (chunk exp-normalized-group-chunk) ()
  (:documentation "Binary units with normalized (EXP ACTIVATION)
firing probabilities representing a multinomial distribution. That is,
samples have exactly one 1 in each group of GROUP-SIZE."))

(defclass constrained-poisson-chunk (chunk exp-normalized-group-chunk) ()
  (:documentation "Poisson units with normalized (EXP ACTIVATION) means."))

(defgeneric set-chunk-mean (chunk)
  (:documentation "Set NODES of CHUNK to the means of the probability
distribution. When called NODES contains the activations.")
  (:method ((chunk conditioning-chunk)))
  (:method ((chunk sigmoid-chunk))
    (let ((nodes (nodes chunk)))
      (declare (type flt-vector nodes))
      (do-chunk (i chunk)
        (setf (aref nodes i)
              (sigmoid (aref nodes i))))))
  (:method ((chunk gaussian-chunk))
    ;; nothing to do: NODES already contains the activation
    )
  (:method ((chunk normalized-group-chunk))
    ;; NODES is already set up, only normalization within groups of
    ;; GROUP-SIZE remains.
    (let ((nodes (nodes chunk))
          (scale (scale chunk))
          (group-size (group-size chunk)))
      (declare (type flt-vector nodes)
               (type flt scale)
               (type index group-size))
      (assert (zerop (mod (chunk-size chunk) group-size)))
      (do-chunk (i chunk)
        ;; this assumes that nodes in the same group have values at
        ;; the same time
        (when (zerop (mod i group-size))
          (let ((sum #.(flt 0)))
            (declare (type flt sum) (optimize (speed 3)))
            (loop for j upfrom i below (+ i group-size)
                  do (incf sum (aref nodes j)))
            (setq sum (/ sum scale))
            (loop for j upfrom i below (+ i group-size)
                  do (setf (aref nodes j)
                           (/ (aref nodes j) sum))))))))
  (:method ((chunk exp-normalized-group-chunk))
    (let ((nodes (nodes chunk)))
      (declare (type flt-vector nodes))
      (do-chunk (i chunk)
        (setf (aref nodes i)
              (exp (aref nodes i)))))
    (call-next-method)))

(defgeneric sample-chunk (chunk)
  (:documentation "Sample from the probability distribution of CHUNK
whose means are in NODES.")
  (:method ((chunk conditioning-chunk)))
  (:method ((chunk sigmoid-chunk))
    (let ((nodes (nodes chunk)))
      (declare (type flt-vector nodes))
      (do-chunk (i chunk)
        (setf (aref nodes i)
              (binarize-randomly (aref nodes i))))))
  (:method ((chunk gaussian-chunk))
    (let ((nodes (nodes chunk)))
      (declare (type flt-vector nodes))
      (do-chunk (i chunk)
        (setf (aref nodes i)
              (+ (aref nodes i)
                 (gaussian-random-1))))))
  (:method ((chunk softmax-chunk))
    (let ((nodes (nodes chunk))
          (group-size (group-size chunk)))
      (declare (type flt-vector nodes)
               (type index group-size)
               (optimize (speed 3)))
      (do-chunk (i chunk)
        (when (zerop (mod i group-size))
          (let ((x (random #.(flt 1))))
            (declare (type flt x))
            (loop for j upfrom i below (+ i group-size) do
                  (when (minusp (decf x (aref nodes j)))
                    (fill nodes #.(flt 0) :start i :end (+ i group-size))
                    (setf (aref nodes j) #.(flt 1))
                    (return))))))))
  (:method ((chunk constrained-poisson-chunk))
    (error "Not implemented yet.")
    #+nil
    (let ((nodes (nodes chunk)))
      (declare (type flt-vector nodes)
               (optimize (speed 3)))
      (do-chunk (i chunk)
        (setf (aref nodes i) (poisson (aref nodes i)))))))


;;;; RBM

(defclass rbm ()
  ((visible-chunks :initarg :visible-chunks :type list :reader visible-chunks)
   (hidden-chunks :initarg :hidden-chunks :type list :reader hidden-chunks)
   (clouds :initarg :clouds :type list :reader clouds))
  (:documentation "An RBM is a network of two layers of nodes. By
convention one is called `visible' and the other `hidden'. Connections
between nodes are symmetrical and there are no intralayer connections.

Layers consists of chunks and chunks of opposing layers can be
connected. A set of connections is called a `cloud'. Currently only
fully connected clouds are supported and this restriction makes it
easy to generate a backprop network from an RBM."))

(defstruct (cloud (:conc-name "") (:constructor %make-cloud))
  cloud-name
  (visible-chunk nil :type chunk)
  (hidden-chunk nil :type chunk)
  (weights nil :type flt-vector))

(defmethod name ((cloud cloud))
  (cloud-name cloud))

(defun make-cloud (&key name visible-chunk hidden-chunk weights)
  (unless weights
    (setq weights (make-flt-array (* (chunk-size visible-chunk)
                                     (chunk-size hidden-chunk))))
    (map-into weights (lambda () (flt (* 0.01 (gaussian-random-1))))))
  (%make-cloud :cloud-name name
               :visible-chunk visible-chunk
               :hidden-chunk hidden-chunk
               :weights weights))

(defmacro do-clouds ((cloud rbm) &body body)
  `(dolist (,cloud (clouds ,rbm))
     ,@body))

(defmacro do-cloud-runs (((start end) cloud) &body body)
  "Iterate over consecutive runs of weights present in CLOUD."
  (with-gensyms (%cloud %hidden-chunk-size %index)
    `(let ((,%cloud ,cloud))
       (if (indices-present (visible-chunk ,%cloud))
           (let ((,%hidden-chunk-size (chunk-size (hidden-chunk ,%cloud))))
             (do-chunk (,%index (visible-chunk ,%cloud))
               (let* ((,start (#.*the* index (* ,%index ,%hidden-chunk-size)))
                      (,end (#.*the* index (+ ,start ,%hidden-chunk-size))))
                 ,@body)))
           (let ((,start 0)
                 (,end (length (weights ,%cloud))))
             ,@body)))))

(eval-when (:compile-toplevel :load-toplevel :execute)
  (defun make-do-cloud/hidden (hidden-index index hidden-chunk-size
                               offset body)
    `(do ((,hidden-index 0 (#.*the* index (1+ ,hidden-index)))
          (,index ,offset (#.*the* index (1+ ,index))))
         ((>= ,hidden-index ,hidden-chunk-size))
       ,@body)))

(defmacro do-cloud/visible ((visible-index cloud) &body body)
  (with-gensyms (%cloud %hidden-chunk-size %offset)
    `(let* ((,%cloud ,cloud)
            (,%hidden-chunk-size (chunk-size (hidden-chunk ,%cloud))))
       (declare (type index ,%hidden-chunk-size))
       (do-chunk (,visible-index (visible-chunk ,%cloud))
         (let ((,%offset (#.*the* index
                                  (* ,visible-index ,%hidden-chunk-size))))
           (macrolet ((do-cloud/hidden ((hidden-index index) &body body)
                        (make-do-cloud/hidden hidden-index index
                                              ',%hidden-chunk-size ',%offset
                                              body)))
             ,@body))))))

(defun find-cloud (name rbm &key errorp)
  "Find the cloud in RBM whose name is EQUAL to NAME. Raise and error
if not found and ERRORP."
  (or (find name (clouds rbm) :key #'name :test #'equal)
      (if errorp
          (error "Cannot find cloud ~S." name)
          nil)))

(defun ->cloud (cloud-designator rbm)
  (if (typep cloud-designator 'cloud)
      cloud-designator
      (find-cloud cloud-designator rbm :errorp t)))

(defun ->clouds (visible-chunks hidden-chunks cloud-specs)
  (let ((clouds
         (loop for spec in cloud-specs
               collect
               (if (typep spec 'cloud)
                   spec
                   (destructuring-bind (&key name visible-chunk hidden-chunk)
                       spec
                     (make-cloud
                      :name name
                      :visible-chunk (->chunk visible-chunk visible-chunks)
                      :hidden-chunk (->chunk hidden-chunk hidden-chunks)))))))
    (unless (unique-names-p clouds)
      (error "Name conflict among clouds: ~S." clouds))
    clouds))

(defun unique-names-p (list)
  (= (length (remove-duplicates (mapcar #'name list) :test #'equal))
     (length (mapcar #'name list))))

(defun default-clouds (visible-chunks hidden-chunks)
  "Return a list of cloud specifications suitable for MAKE-RBM. Put a
cloud between each pair of visible and hidden chunk unless they are
both conditioning chunks. The names of the clouds are two element
lists of the names of the visible and hidden chunks."
  (let ((clouds '()))
    (dolist (visible-chunk visible-chunks)
      (dolist (hidden-chunk hidden-chunks)
        (unless (and (conditioning-chunk-p visible-chunk)
                     (conditioning-chunk-p hidden-chunk))
          (push `(:name ,(list (name visible-chunk) (name hidden-chunk))
                  :visible-chunk ,(name visible-chunk)
                  :hidden-chunk ,(name hidden-chunk))
                clouds))))
    clouds))

(defmethod initialize-instance :after
    ((rbm rbm) &key visible-chunks hidden-chunks
     (clouds (default-clouds visible-chunks hidden-chunks)))
  "Return an RBM that consists of VISIBLE-CHUNKS, HIDDEN-CHUNKS and
CLOUDS of weights. Where CLOUDS is a list of cloud specifications.
Names of chunks and clouds shall be unique by EQUAL."
  (unless (unique-names-p visible-chunks)
    (error "Name conflict among visible chunks: ~S." visible-chunks))
  (unless (unique-names-p hidden-chunks)
    (error "Name conflict among hidden chunks ~S." hidden-chunks))
  (setf (slot-value rbm 'clouds)
        (->clouds visible-chunks hidden-chunks clouds))
  rbm)

(defun hijack-means-to-activation (rbm to-visible/hidden)
  "Set the chunks of TO-VISIBLE/HIDDEN layer of RBM to the activations
calculated from the other layer's nodes. Skip chunks that don't need
activations."
  (declare (optimize (speed 3) #.*no-array-bounds-check*))
  (cond ((eq :hidden to-visible/hidden)
         (dolist (chunk (hidden-chunks rbm))
           (unless (conditioning-chunk-p chunk)
             (fill (the flt-vector (nodes chunk)) #.(flt 0))))
         (do-clouds (cloud rbm)
           (unless (conditioning-chunk-p (hidden-chunk cloud))
             (let ((weights (weights cloud))
                   (from (nodes (visible-chunk cloud)))
                   (to (nodes (hidden-chunk cloud))))
               (declare (type flt-vector weights from to))
               (if (and (use-blas-p (length weights))
                        (null (indices-present (visible-chunk cloud))))
                   (funcall (intern #.(symbol-name 'dgemv)
                                    (find-package 'blas))
                            "N" (length to) (length from)
                            1d0 weights (length to)
                            from 1
                            1d0 to 1)
                   (do-cloud/visible (i cloud)
                     (let ((x (aref from i)))
                       (unless (zerop x)
                         (do-cloud/hidden (j weight-index)
                           (incf (aref to j)
                                 (* x (aref weights weight-index))))))))))))
        (t
         (dolist (chunk (visible-chunks rbm))
           (unless (conditioning-chunk-p chunk)
             (let ((nodes (the flt-vector (nodes chunk))))
               (if (indices-present chunk)
                   (do-chunk (i chunk)
                     (setf (aref nodes i) #.(flt 0)))
                   (fill nodes #.(flt 0))))))
         (do-clouds (cloud rbm)
           (unless (conditioning-chunk-p (visible-chunk cloud))
             (let ((weights (weights cloud))
                   (from (nodes (hidden-chunk cloud)))
                   (to (nodes (visible-chunk cloud))))
               (declare (type flt-vector weights from to))
               (if (and (use-blas-p (length weights))
                        (null (indices-present (visible-chunk cloud))))
                   (funcall (intern #.(symbol-name 'dgemv)
                                    (find-package 'blas))
                            "T" (length from) (length to)
                            1d0 weights (length from)
                            from 1
                            1d0 to 1)
                   (do-cloud/visible (i cloud)
                     (let ((sum #.(flt 0)))
                       (declare (type flt sum))
                       (do-cloud/hidden (j weight-index)
                         (incf sum (* (aref from j)
                                      (aref weights weight-index))))
                       (incf (aref to i) sum))))))))))

(defun set-visible-mean (rbm)
  "Set NODES of the chunks in the visible layer to the means of their
respective probability distribution assuming NODES contains the
activations."
  (hijack-means-to-activation rbm :visible)
  (map nil #'set-chunk-mean (visible-chunks rbm)))

(defun set-hidden-mean (rbm)
  "Set NODES of the chunks in the hidden layer to the means of their
respective probability distribution assuming NODES contains the
activations."
  (hijack-means-to-activation rbm :hidden)
  (map nil #'set-chunk-mean (hidden-chunks rbm)))

(defun sample-visible (rbm)
  "Generate samples from the probability distribution defined by the
chunk type and the mean that resides in NODES."
  (map nil #'sample-chunk (visible-chunks rbm)))

(defun sample-hidden (rbm)
  "Generate samples from the probability distribution defined by the
chunk type and the mean that resides in NODES."
  (map nil #'sample-chunk (hidden-chunks rbm)))


;;;; Integration with train and gradient descent

(defclass rbm-trainer (segmented-trainer)
  ((sample-visible-p
    :initform nil
    :initarg :sample-visible-p
    :accessor sample-visible-p
    :documentation "Controls whether visible nodes are sampled during
the learning or the mean field is used instead.")
   (sample-hidden-p
    :initform t
    :initarg :sample-hidden-p
    :accessor sample-hidden-p
    :documentation "Controls whether hidden nodes are sampled during
the learning or the mean field is used instead.")
   (n-gibbs
    :type (integer 1)
    :initform 1
    :initarg :n-gibbs
    :accessor n-gibbs
    :documentation "The number of steps of Gibbs sampling to perform.")))

(defmethod map-segments (fn (rbm rbm))
  (map nil fn (clouds rbm)))

(defmethod segment-weights ((cloud cloud))
  (values (weights cloud) 0 (length (weights cloud))))

(defmethod map-segment-runs (fn (cloud cloud))
  (do-cloud-runs ((start end) cloud)
    (funcall fn start end)))

(defmethod train-one (sample (trainer rbm-trainer) rbm &key)
  (set-input sample rbm)
  (positive-phase trainer rbm)
  (negative-phase trainer rbm)
  (call-next-method))

(defun copy-inputs-to-nodes (rbm)
  (map nil (lambda (chunk)
             (locally (declare (optimize (speed 3)))
               (when (inputs chunk)
                 (replace (the flt-vector (nodes chunk))
                          (the flt-vector (inputs chunk))))))
       (visible-chunks rbm)))

(defmethod set-input :around (sample (rbm rbm))
  (unwind-protect (call-next-method)
    (copy-inputs-to-nodes rbm)))


;;;; Training implementation

(defgeneric accumulate-positive-phase-statistics (trainer rbm)
  (:method ((trainer rbm-trainer) rbm)
    (declare (ignore rbm))
    (do-segment-gradient-accumulators ((cloud acc-start products) trainer)
      (let ((v1 (nodes (visible-chunk cloud)))
            (v2 (nodes (hidden-chunk cloud))))
        (declare (type flt-vector v1 v2 products)
                 (optimize (speed 3) #.*no-array-bounds-check*))
        ;; We cloud use (BLAS:DGER (LENGTH V1) (LENGTH V2) -1D0 V1 1
        ;; V2 1 PRODUCTS (LENGTH V1)) as in SUBTRACT-PRODUCT but it
        ;; seems to result in a slowdown most likely due to the UNLESS
        ;; (ZEROP x) test being effective as typical data sets tend to
        ;; have many zeros in the input.
        (special-case (zerop acc-start)
          (do-cloud/visible (i cloud)
            (let ((x (aref v1 i)))
              (unless (zerop x)
                (do-cloud/hidden (j weight-index)
                  (decf (aref products (#.*the*
                                        index
                                        (+ acc-start weight-index)))
                        (* x (aref v2 j)))))))))))
  (:documentation "Subtract the product of the nodes of the two layers
of each cloud to the appropriate gradient accumlator. This is the
first term of contrastive divergence learning rule."))

;;; Takes N-GIBBS as parameter because it may change between
;;; invocations. Currently it is always 1, though.
(defgeneric accumulate-negative-phase-statistics (trainer rbm n-gibbs)
  (:method ((trainer rbm-trainer) rbm n-gibbs)
    (declare (ignore rbm)
             (type index n-gibbs))
    (do-segment-gradient-accumulators
        ((cloud acc-start accumulator1 accumulator2) trainer)
      (let ((v1 (nodes (visible-chunk cloud)))
            (v2 (nodes (hidden-chunk cloud)))
            (products (or accumulator2 accumulator1)))
        (declare (type flt-vector v1 v2 products))
        (if (and (use-blas-p (length products))
                 (null (indices-present (visible-chunk cloud))))
            (funcall (intern #.(symbol-name 'dger) (find-package 'blas))
                     (length v2) (length v1) (/ (flt n-gibbs))
                     v2 1 v1 1 products (length v2))
            (locally (declare (optimize (speed 3) #.*no-array-bounds-check*))
              (special-case (zerop acc-start)
                (if (= 1 n-gibbs)
                    (do-cloud/visible (i cloud)
                      (let ((x (aref v1 i)))
                        (unless (zerop x)
                          (do-cloud/hidden (j weight-index)
                            (incf (aref products
                                        (#.*the*
                                         index
                                         (+ acc-start weight-index)))
                                  (* x (aref v2 j)))))))
                    (let ((value (/ (flt n-gibbs))))
                      (do-cloud/visible (i cloud)
                        (let ((x (aref v1 i)))
                          (unless (zerop x)
                            (let ((x (* x value)))
                              (do-cloud/hidden (j weight-index)
                                (incf (aref products
                                            (#.*the*
                                             index
                                             (+ acc-start weight-index)))
                                      (* x (aref v2 j))))))))))))))))
  (:documentation "Add the product of the nodes of the two layers of
each cloud to the appropriate gradient accumlator. This is the second
term of contrastive divergence learning rule."))

(defgeneric positive-phase (trainer rbm)
  (:method (trainer rbm)
    (set-hidden-mean rbm)
    (accumulate-positive-phase-statistics trainer rbm)
    (when (sample-hidden-p trainer)
      (sample-hidden rbm))))

#+nil
(defun print-nodes-by-chunk (network)
  (map nil (lambda (chunk)
             (unless (typep chunk 'mgl-rbm:conditioning-chunk)
               (format t "~S~%  " chunk)
               (let ((array (mgl-rbm:nodes chunk))
                     (n 0))
                 (mgl-rbm::do-chunk (i chunk)
                   (format t "~S:~,5F " i (aref array i))
                   (when (and *print-length*
                              (<= *print-length* (incf n)))
                     (return))))
               (terpri)))
       (append (mgl-rbm:visible-chunks network)
               (mgl-rbm:hidden-chunks network))))

(defgeneric negative-phase (trainer rbm)
  (:method ((trainer rbm-trainer) rbm)
    (let ((n-gibbs (n-gibbs trainer))
          (sample-visible-p (sample-visible-p trainer))
          (sample-hidden-p (sample-hidden-p trainer)))
      (assert (plusp n-gibbs))
      (loop for i below n-gibbs do
            (set-visible-mean rbm)
            (when sample-visible-p
              (sample-visible rbm))
            (set-hidden-mean rbm)
            (when (and sample-hidden-p (/= i (1- n-gibbs)))
              (sample-hidden rbm)))
      (accumulate-negative-phase-statistics trainer rbm 1))))


;;;; I/O

(defmethod write-weights ((cloud cloud) stream)
  (mgl-util::write-double-float-vector (weights cloud) stream))

(defmethod write-weights ((rbm rbm) stream)
  (dolist (cloud (clouds rbm))
    (write-weights cloud stream)))

(defmethod read-weights ((cloud cloud) stream)
  (mgl-util::read-double-float-vector (weights cloud) stream))

(defmethod read-weights ((rbm rbm) stream)
  (dolist (cloud (clouds rbm))
    (read-weights cloud stream)))


;;;; Convenience

(defun layer-error (chunks)
  "Return the squared norm of INPUTS - NODES."
  (let ((sum #.(flt 0))
        (n 0))
    (declare (type flt sum) (type index n) (optimize (speed 3)))
    (dolist (chunk chunks)
      (unless (conditioning-chunk-p chunk)
        (let ((inputs (inputs chunk))
              (nodes (nodes chunk)))
          (declare (type flt-vector inputs nodes))
          (do-chunk (i chunk)
            (let ((x (aref inputs i))
                  (y (aref nodes i)))
              (incf sum (expt (- x y) 2))
              (incf n))))))
    (values sum n)))

(defgeneric get-squared-error (rbm)
  (:method ((rbm rbm))
    (set-hidden-mean rbm)
    (set-visible-mean rbm)
    (layer-error (visible-chunks rbm))))
