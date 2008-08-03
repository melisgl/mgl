(in-package :mgl-rbm)

;;;; Chunk

(defclass chunk ()
  ((name :initform (gensym) :initarg :name :reader name)
   (inputs
    :type (or matlisp:real-matrix null) :reader inputs
    :documentation "This is where SET-INPUT saves the input for later
use by RECONSTRUCTION-ERROR, INPUTS->NODES. It is NIL in
CONSTANT-CHUNKS.")
   (nodes
    :type matlisp:real-matrix :reader nodes
    :documentation "A value for each node in the chunk. First,
activations are put here (weights*inputs) then the mean of the
probability distribution is calculated from the activation and finally
\(optionally) a sample is taken from the probability distribution. All
these values are stored in this vector. This is also where SET-INPUT
is supposed to clamp the values.")
   (indices-present
    :initform nil :initarg :indices-present :type (or null index-vector)
    :accessor indices-present
    :documentation "NIL or a simple vector of array indices into the
layer's NODES. Need not be ordered. SET-INPUT sets it. Note that if it
is non-NIL then N-STRIPES must be 1.")
   (default-value
    :initform #.(flt 0) :initarg :default-value :type flt
    :reader default-value
    :documentation "Upon creation or resize the chunk's nodes get
filled with this value."))
  (:documentation "Base class for different chunks. A chunk is a set
of nodes of the same type."))

(declaim (inline chunk-size))
(defun chunk-size (chunk)
  (the index (values (matlisp:nrows (nodes chunk)))))

(defmethod size ((chunk chunk))
  (chunk-size chunk))

(declaim (inline chunk-n-stripes))
(defun chunk-n-stripes (chunk)
  (the index (values (matlisp:ncols (nodes chunk)))))

(defmethod n-stripes ((chunk chunk))
  (chunk-n-stripes chunk))

(declaim (inline chunk-max-n-stripes))
(defun chunk-max-n-stripes (chunk)
  (let ((nodes (nodes chunk)))
    (the index (/ (length (storage nodes))
                  (matlisp:nrows nodes)))))

(defmethod max-n-stripes ((chunk chunk))
  (chunk-max-n-stripes chunk))

(defmethod stripe-start (stripe (chunk chunk))
  (* stripe (chunk-size chunk)))

(defmethod stripe-end (stripe (chunk chunk))
  (* (1+ stripe) (chunk-size chunk)))

(defmethod print-object ((chunk chunk) stream)
  (print-unreadable-object (chunk stream :type t :identity t)
    (format stream "~S ~S(~S/~S)" (name chunk) (chunk-size chunk)
            (chunk-n-stripes chunk) (chunk-max-n-stripes chunk)))
  chunk)

(defun use-blas-on-chunk-p (cost chunk)
  (assert (or (null (indices-present chunk))
              (= 1 (chunk-n-stripes chunk))))
  (and
   ;; several stripes or cost is high => blas
   (or (< 1 (chunk-n-stripes chunk))
       (use-blas-p cost))
   ;; there is no missing value support in blas
   (null (indices-present chunk))))

(defun ->chunk (chunk-designator chunks)
  (if (typep chunk-designator 'chunk)
      chunk-designator
      (or (find chunk-designator chunks :key #'name :test #'equal)
          (error "Cannot find chunk ~S." chunk-designator))))

(defvar *current-stripe*)

(defmacro do-stripes ((chunk) &body body)
  (with-gensyms (%chunk %stripe)
    `(let ((,%chunk ,chunk))
       (assert (locally (declare (optimize (speed 1)))
                 (or (not (indices-present ,%chunk))
                     (= 1 (chunk-n-stripes ,%chunk)))))
       (dotimes (,%stripe (chunk-n-stripes ,%chunk))
         (let ((*current-stripe* ,%stripe))
           ,@body)))))

(defmacro do-chunk ((index chunk) &body body)
  "Iterate over the indices of nodes of CHUNK skipping missing ones."
  (with-gensyms (%chunk %indices-present %size)
    `(let* ((,%chunk ,chunk)
            (,%indices-present (indices-present ,%chunk)))
       (if ,%indices-present
           (locally (declare (type index-vector ,%indices-present))
             (loop for ,index across ,%indices-present
                   do (progn ,@body)))
           (let ((,%size (chunk-size ,%chunk)))
             (declare (type index ,%size))
             (loop for ,index fixnum
                   upfrom (locally (declare (optimize (speed 1)))
                            (the index (* *current-stripe* ,%size)))
                   below (locally (declare (optimize (speed 1)))
                           (the index (+ ,%size (* *current-stripe* ,%size))))
                   do ,@body))))))

(defun fill-chunk (chunk value)
  (declare (type flt value))
  (if (use-blas-on-chunk-p (cost-of-fill (nodes chunk)) chunk)
      (let ((nodes (storage (nodes chunk))))
        (do-stripes (chunk)
          (do-chunk (i chunk)
            (setf (aref nodes i) value))))
      (matlisp:fill-matrix (nodes chunk) value)))

(defclass conditioning-chunk (chunk) ()
  (:documentation "Nodes in CONDITIONING-CHUNK never change their
values on their own so they are to be clamped. Including this chunk in
the visible layer allows `conditional' RBMs."))

(defun conditioning-chunk-p (chunk)
  (typep chunk 'conditioning-chunk))

(defun resize-chunk (chunk size max-n-stripes)
  (unless (and (slot-boundp chunk 'nodes)
               (= size (chunk-size chunk))
               (= max-n-stripes (chunk-max-n-stripes chunk)))
    (setf (slot-value chunk 'nodes)
          (matlisp:make-real-matrix size max-n-stripes))
    (fill-chunk chunk (default-value chunk))
    (setf (slot-value chunk 'inputs)
          (if (typep chunk 'constant-chunk)
              nil
              (matlisp:make-real-matrix size max-n-stripes)))))

(defmethod set-n-stripes (n-stripes (chunk chunk))
  (set-ncols (nodes chunk) n-stripes)
  (when (inputs chunk)
    (set-ncols (inputs chunk) n-stripes))
  n-stripes)

(defmethod set-max-n-stripes (max-n-stripes (chunk chunk))
  (resize-chunk chunk (chunk-size chunk) max-n-stripes)
  max-n-stripes)

(defmethod initialize-instance :after ((chunk chunk)
                                       &key (size 1) (max-n-stripes 1)
                                       &allow-other-keys)
  (resize-chunk chunk size max-n-stripes))

(defclass constant-chunk (conditioning-chunk)
  ((default-value :initform #.(flt 1)))
  (:documentation "A special kind of CONDITIONING-CHUNK whose NODES
are always DEFAULT-VALUE. This conveniently allows biases in the
opposing layer."))

(defmethod initialize-instance :after ((chunk constant-chunk)
                                       &key &allow-other-keys)
  (fill-chunk chunk (default-value chunk)))

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
    (let ((nodes (storage (nodes chunk))))
      (do-stripes (chunk)
        (do-chunk (i chunk)
          (setf (aref nodes i)
                (sigmoid (aref nodes i)))))))
  (:method ((chunk gaussian-chunk))
    ;; nothing to do: NODES already contains the activation
    )
  (:method ((chunk normalized-group-chunk))
    ;; NODES is already set up, only normalization within groups of
    ;; GROUP-SIZE remains.
    (let ((nodes (storage (nodes chunk)))
          (scale (scale chunk))
          (group-size (group-size chunk)))
      (declare (type flt scale)
               (type index group-size))
      (assert (zerop (mod (chunk-size chunk) group-size)))
      (do-stripes (chunk)
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
                             (/ (aref nodes j) sum)))))))))
  (:method ((chunk exp-normalized-group-chunk))
    (let ((nodes (storage (nodes chunk))))
      (do-stripes (chunk)
        (do-chunk (i chunk)
          (setf (aref nodes i)
                (exp (aref nodes i))))))
    (call-next-method)))

(defgeneric sample-chunk (chunk)
  (:documentation "Sample from the probability distribution of CHUNK
whose means are in NODES.")
  (:method ((chunk conditioning-chunk)))
  (:method ((chunk sigmoid-chunk))
    (let ((nodes (storage (nodes chunk))))
      (do-stripes (chunk)
        (do-chunk (i chunk)
          (setf (aref nodes i)
                (binarize-randomly (aref nodes i)))))))
  (:method ((chunk gaussian-chunk))
    (let ((nodes (storage (nodes chunk))))
      (do-stripes (chunk)
        (do-chunk (i chunk)
          (setf (aref nodes i)
                (+ (aref nodes i)
                   (gaussian-random-1)))))))
  (:method ((chunk softmax-chunk))
    (let ((nodes (storage (nodes chunk)))
          (group-size (group-size chunk)))
      (declare (type index group-size)
               (optimize (speed 3)))
      (do-stripes (chunk)
        (do-chunk (i chunk)
          (when (zerop (mod i group-size))
            (let ((x (random #.(flt 1))))
              (declare (type flt x))
              (loop for j upfrom i below (+ i group-size) do
                    (when (minusp (decf x (aref nodes j)))
                      (fill nodes #.(flt 0) :start i :end (+ i group-size))
                      (setf (aref nodes j) #.(flt 1))
                      (return)))))))))
  (:method ((chunk constrained-poisson-chunk))
    (error "Not implemented yet.")
    #+nil
    (let ((nodes (storage (nodes chunk))))
      (declare (optimize (speed 3)))
      (do-stripes (chunk)
        (do-chunk (i chunk)
          (setf (aref nodes i) (poisson (aref nodes i))))))))


;;;; Cloud

(defclass cloud ()
  ((name :initarg :name :reader name)
   (visible-chunk :type chunk :initarg :visible-chunk :reader visible-chunk)
   (hidden-chunk :type chunk :initarg :hidden-chunk :reader hidden-chunk)))

(defmethod print-object ((cloud cloud) stream)
  (print-unreadable-object (cloud stream :type t :identity t)
    (when (slot-boundp cloud 'name)
      (format stream "~S" (name cloud))))
  cloud)

(defmethod set-n-stripes (n-stripes (cloud cloud)))
(defmethod set-max-n-stripes (max-n-stripes (cloud cloud)))

(defgeneric activate-cloud (cloud to-visible/hidden)
  (:documentation "Add the activations to the nodes of chunk to either
the visible or the hidden chunk of CLOUD according to
TO-VISIBLE/HIDDEN that can be :VISIBLE or :HIDDEN. In the simplest
case this is add weights (of CLOUD) * nodes (of the visible chunk) to
the nodes of the hidden chunk."))

(defgeneric accumulate-cloud-statistics (cloud trainer addp)
  (:documentation "Add or subtract the derivative of positive phase
term of contrastive divergence from the accumulator of TRAINER
corresponding to CLOUD."))


;;;; Full cloud

(defclass full-cloud (cloud)
  ((weights
    :type matlisp:real-matrix :initarg :weights :reader weights
    :documentation "In Matlisp, chunks are represented as column
vectors \(disregarding the more than one stripe case). If the visible
chunk is Nx1 and the hidden is Mx1 then the weight matrix is MxN.
Hidden = hidden + weights * visible. Visible = visible + weights^T *
hidden. Looking directly at the underlying Lisp array
\(MATLISP::STORE), it's all transposed.")))

(defmethod initialize-instance :after ((cloud full-cloud) &key &allow-other-keys)
  (unless (slot-boundp cloud 'weights)
    (setf (slot-value cloud 'weights)
          (matlisp:make-real-matrix (chunk-size (hidden-chunk cloud))
                                    (chunk-size (visible-chunk cloud))))
    (map-into (storage (weights cloud))
              (lambda () (flt (* 0.01 (gaussian-random-1)))))))

(defmacro do-cloud-runs (((start end) cloud) &body body)
  "Iterate over consecutive runs of weights present in CLOUD."
  (with-gensyms (%cloud %hidden-chunk-size %index)
    `(let ((,%cloud ,cloud))
       (if (indices-present (visible-chunk ,%cloud))
           (let ((,%hidden-chunk-size (chunk-size (hidden-chunk ,%cloud))))
             (do-stripes ((visible-chunk ,%cloud))
               (do-chunk (,%index (visible-chunk ,%cloud))
                 (let* ((,start (the! index (* ,%index ,%hidden-chunk-size)))
                        (,end (the! index (+ ,start ,%hidden-chunk-size))))
                   ,@body))))
           (let ((,start 0)
                 (,end (matlisp:number-of-elements (weights ,%cloud))))
             ,@body)))))

(eval-when (:compile-toplevel :load-toplevel :execute)
  (defun make-do-cloud/hidden (hidden-index index hidden-chunk-size
                               offset body)
    `(do ((,hidden-index 0 (the! index (1+ ,hidden-index)))
          (,index ,offset (the! index (1+ ,index))))
         ((>= ,hidden-index ,hidden-chunk-size))
       ,@body)))

(defmacro do-cloud/visible ((visible-index cloud) &body body)
  (with-gensyms (%cloud %hidden-chunk-size %offset)
    `(let* ((,%cloud ,cloud)
            (,%hidden-chunk-size (chunk-size (hidden-chunk ,%cloud))))
       (declare (type index ,%hidden-chunk-size))
       (assert (= 1 (chunk-n-stripes (visible-chunk ,%cloud))))
       (do-stripes ((visible-chunk ,%cloud))
         (do-chunk (,visible-index (visible-chunk ,%cloud))
           (let ((,%offset (the! index
                                 (* ,visible-index ,%hidden-chunk-size))))
             (macrolet ((do-cloud/hidden ((hidden-index index) &body body)
                          (make-do-cloud/hidden hidden-index index
                                                ',%hidden-chunk-size ',%offset
                                                body)))
               ,@body)))))))

(defmethod activate-cloud ((cloud full-cloud) to-visible/hidden)
  (declare (optimize (speed 3) #.*no-array-bounds-check*))
  (if (eq to-visible/hidden :hidden)
      (let ((weights (weights cloud))
            (from (nodes (visible-chunk cloud)))
            (to (nodes (hidden-chunk cloud))))
        (if (use-blas-on-chunk-p (cost-of-gemm weights from :nn)
                                 (visible-chunk cloud))
            (matlisp:gemm! (flt 1) weights from (flt 1) to)
            (let ((weights (storage weights))
                  (from (storage from))
                  (to (storage to)))
              (do-cloud/visible (i cloud)
                (let ((x (aref from i)))
                  (unless (zerop x)
                    (do-cloud/hidden (j weight-index)
                      (incf (aref to j)
                            (* x (aref weights weight-index))))))))))
      (let ((weights (weights cloud))
            (from (nodes (hidden-chunk cloud)))
            (to (nodes (visible-chunk cloud))))
        (if (use-blas-on-chunk-p (cost-of-gemm weights from :tn)
                                 (visible-chunk cloud))
            (matlisp:gemm! (flt 1) weights from (flt 1) to :tn)
            (let ((weights (storage weights))
                  (from (storage from))
                  (to (storage to)))
              (do-cloud/visible (i cloud)
                (let ((sum #.(flt 0)))
                  (declare (type flt sum))
                  (do-cloud/hidden (j weight-index)
                    (incf sum (* (aref from j)
                                 (aref weights weight-index))))
                  (incf (aref to i) sum))))))))

(defmethod accumulate-cloud-statistics ((cloud full-cloud) trainer addp)
  (with-segment-gradient-accumulator ((start accumulator accumulator2)
                                      (cloud trainer))
    (when (and accumulator start)
      (let ((v1 (nodes (visible-chunk cloud)))
            (v2 (nodes (hidden-chunk cloud)))
            (accumulator (if (or addp (null accumulator2))
                             accumulator
                             accumulator2)))
        (if (and (zerop start)
                 (use-blas-on-chunk-p (cost-of-gemm v2 v1 :nt)
                                      (visible-chunk cloud)))
            (matlisp:gemm! (flt (if addp 1 -1)) v2 v1 (flt 1)
                           (reshape2 accumulator
                                     (matlisp:nrows v2)
                                     (matlisp:nrows v1))
                           :nt)
            (let ((v1 (storage v1))
                  (v2 (storage v2))
                  (accumulator (storage accumulator)))
              (declare (optimize (speed 3) #.*no-array-bounds-check*))
              (if addp
                  (special-case (zerop start)
                    (do-cloud/visible (i cloud)
                      (let ((x (aref v1 i)))
                        (unless (zerop x)
                          (do-cloud/hidden (j weight-index)
                            (incf (aref accumulator
                                        (the! index (+ start weight-index)))
                                  (* x (aref v2 j))))))))
                  (special-case (zerop start)
                    (do-cloud/visible (i cloud)
                      (let ((x (aref v1 i)))
                        (unless (zerop x)
                          (do-cloud/hidden (j weight-index)
                            (decf (aref accumulator
                                        (the! index (+ start weight-index)))
                                  (* x (aref v2 j)))))))))))))))

(defmethod map-segments (fn (cloud full-cloud))
  (funcall fn cloud))

(defmethod segment-weights ((cloud full-cloud))
  (values (storage (weights cloud)) 0
          (length (storage (weights cloud)))))

(defmethod map-segment-runs (fn (cloud full-cloud))
  (do-cloud-runs ((start end) cloud)
    (funcall fn start end)))

(defmethod write-weights ((cloud full-cloud) stream)
  (write-double-float-vector (storage (weights cloud)) stream))

(defmethod read-weights ((cloud full-cloud) stream)
  (read-double-float-vector (storage (weights cloud)) stream))


;;;; Factored cloud

(defclass factored-cloud (cloud)
  ((cloud-a
    :type full-cloud :initarg :cloud-a :reader cloud-a
    :documentation "A full cloud whose hidden chunk is the same as the
hidden chunk of this cloud and whose visible chunk is the same as the
hidden chunk of CLOUD-B.")
   (cloud-b
    :type full-cloud :initarg :cloud-b :reader cloud-b
    :documentation "A full cloud whose visible chunk is the same as
the visible chunk of this cloud and whose hidden chunk is the same as
the visible chunk of CLOUD-A."))
  (:documentation "Like FULL-CLOUD but the weight matrix is factored
into a product of two matrices: A*B. At activation time, HIDDEN +=
A*B*VISIBLE."))

(defclass factored-cloud-shared-chunk (chunk) ())

(defmethod initialize-instance :after ((cloud factored-cloud)
                                       &key common-rank
                                       &allow-other-keys)
  (assert (typep common-rank '(integer 1)))
  (let ((shared (make-instance 'factored-cloud-shared-chunk
                               :size common-rank
                               :name `(name cloud))))
    (setf (slot-value cloud 'cloud-a)
          (make-instance 'full-cloud
                         :name (list (name cloud) :a)
                         :visible-chunk shared
                         :hidden-chunk (hidden-chunk cloud)))
    (setf (slot-value cloud 'cloud-b)
          (make-instance 'full-cloud
                         :name (list (name cloud) :b)
                         :visible-chunk (visible-chunk cloud)
                         :hidden-chunk shared))))

(defun factored-cloud-shared-chunk (cloud)
  (visible-chunk (cloud-a cloud)))

(defun common-rank (cloud)
  (chunk-size (factored-cloud-shared-chunk cloud)))

(defmethod set-n-stripes (n-stripes (cloud factored-cloud))
  (setf (n-stripes (factored-cloud-shared-chunk cloud)) n-stripes))

(defmethod set-max-n-stripes (max-n-stripes (cloud factored-cloud))
  (setf (max-n-stripes (factored-cloud-shared-chunk cloud)) max-n-stripes))

(defmethod activate-cloud ((cloud factored-cloud) to-visible/hidden)
  ;; Normal chunks are zeroed by HIJACK-MEANS-TO-ACTIVATION.
  (fill-chunk (factored-cloud-shared-chunk cloud) #.(flt 0))
  (cond ((eq to-visible/hidden :hidden)
         (activate-cloud (cloud-b cloud) to-visible/hidden)
         (activate-cloud (cloud-a cloud) to-visible/hidden))
        (t
         (activate-cloud (cloud-a cloud) to-visible/hidden)
         (activate-cloud (cloud-b cloud) to-visible/hidden))))

(defmethod accumulate-cloud-statistics ((cloud factored-cloud) trainer addp)
  (let* ((visible-chunk (visible-chunk cloud))
         (v (nodes visible-chunk))
         (h (nodes (hidden-chunk cloud)))
         (a (weights (cloud-a cloud)))
         (b (weights (cloud-b cloud)))
         (n-stripes (n-stripes (visible-chunk cloud)))
         (c (matlisp:nrows b))
         (shared (factored-cloud-shared-chunk cloud))
         (v* (storage v))
         (shared* (storage (nodes shared))))
    (with-segment-gradient-accumulator ((start accumulator accumulator2)
                                        ((cloud-a cloud) trainer))
      (when (and accumulator start)
        ;; dCD/dA ~= h*v'*B'
        (let ((x (reshape2 (nodes shared) n-stripes c))
              (accumulator (if (or addp (null accumulator2))
                               accumulator
                               accumulator2)))
          (if (and (zerop start)
                   (null (indices-present (visible-chunk cloud))))
              (matlisp:gemm! (flt 1) v b (flt 0) x :tt)
              (progn
                (assert (= 1 (n-stripes visible-chunk)))
                (let ((b* (storage b)))
                  (declare (optimize (speed 3) #.*no-array-bounds-check*))
                  (matlisp:fill-matrix x (flt 0))
                  (do-stripes (visible-chunk)
                    (do-chunk (i visible-chunk)
                      (let ((v*i (aref v* i)))
                        (loop for j of-type index upfrom 0 below c
                              for bi of-type index
                              upfrom (the! index (* i c)) do
                              (incf (aref shared* j) (* v*i (aref b* bi))))))))))
          (matlisp:gemm! (flt (if addp 1 -1)) h x
                         (flt 1) (reshape2 accumulator
                                           (matlisp:nrows a)
                                           (matlisp:ncols a))))))
    (with-segment-gradient-accumulator ((start accumulator accumulator2)
                                        ((cloud-b cloud) trainer))
      (when (and accumulator start)
        ;; dCD/dB ~= A'*h*v'
        (let ((x (reshape2 (nodes shared) c n-stripes))
              (accumulator (if (or addp (null accumulator2))
                               accumulator
                               accumulator2)))
          (matlisp:gemm! (flt 1) a h (flt 0) x :tn)
          (if (and (zerop start)
                   (null (indices-present (visible-chunk cloud))))
              (matlisp:gemm! (flt (if addp 1 -1)) x v
                             (flt 1) (reshape2 accumulator
                                               (matlisp:nrows b)
                                               (matlisp:ncols b))
                             :nt)
              (progn
                (assert (= 1 (n-stripes visible-chunk)))
                (let ((acc* (storage accumulator)))
                  (declare (optimize (speed 3) #.*no-array-bounds-check*))
                  (do-stripes (visible-chunk)
                    (do-chunk (i visible-chunk)
                      (let ((v*i (* (flt (if addp 1 -1)) (aref v* i))))
                        (loop for j of-type index upfrom 0 below c
                              for acc-i of-type index
                              upfrom (the! index
                                           (+ start (the! index (* i c)))) do
                              (incf (aref acc* acc-i)
                                    (* v*i (aref shared* j)))))))))))))))

(defmethod map-segments (fn (cloud factored-cloud))
  (funcall fn (cloud-a cloud))
  (funcall fn (cloud-b cloud)))

(defmethod write-weights ((cloud factored-cloud) stream)
  (write-weights (cloud-a cloud) stream)
  (write-weights (cloud-b cloud) stream))

(defmethod read-weights ((cloud factored-cloud) stream)
  (read-weights (cloud-a cloud) stream)
  (read-weights (cloud-b cloud) stream))


;;;; RBM

(defclass rbm ()
  ((visible-chunks :initarg :visible-chunks :type list :reader visible-chunks)
   (hidden-chunks :initarg :hidden-chunks :type list :reader hidden-chunks)
   (clouds :initarg :clouds :type list :reader clouds)
   (dbn :initform nil :type (or null dbn) :reader dbn)
   (max-n-stripes :initform 1 :initarg :max-n-stripes :reader max-n-stripes))
  (:documentation "An RBM is a network of two layers of nodes. By
convention one is called `visible' and the other `hidden'. Connections
between nodes are symmetrical and there are no intralayer connections.

Layers consist of chunks and chunks of opposing layers can be
connected. A set of connections is called a `cloud'. Currently only
fully connected clouds are supported."))

(defmacro do-clouds ((cloud rbm) &body body)
  `(dolist (,cloud (clouds ,rbm))
     ,@body))

(defmethod n-stripes ((rbm rbm))
  (n-stripes (first (visible-chunks rbm))))

(defmethod set-n-stripes (n-stripes (rbm rbm))
  (dolist (chunk (visible-chunks rbm))
    (setf (n-stripes chunk) n-stripes))
  (dolist (chunk (hidden-chunks rbm))
    (setf (n-stripes chunk) n-stripes))
  (do-clouds (cloud rbm)
    (setf (n-stripes cloud) n-stripes)))

(defmethod set-max-n-stripes (max-n-stripes (rbm rbm))
  (setf (slot-value rbm 'max-n-stripes) max-n-stripes)
  (dolist (chunk (visible-chunks rbm))
    (setf (max-n-stripes chunk) max-n-stripes))
  (dolist (chunk (hidden-chunks rbm))
    (setf (max-n-stripes chunk) max-n-stripes))
  (do-clouds (cloud rbm)
    (setf (max-n-stripes cloud) max-n-stripes)))

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
  (flet ((name* (chunk-or-name)
           (if (typep chunk-or-name 'chunk)
               (name chunk-or-name)
               chunk-or-name)))
    (let ((clouds
           (loop for spec in cloud-specs
                 collect
                 (if (typep spec 'cloud)
                     spec
                     (multiple-value-bind (known unknown)
                         (split-plist spec '(:class :name
                                             :visible-chunk :hidden-chunk))
                       (destructuring-bind (&key (class 'full-cloud)
                                                 visible-chunk hidden-chunk
                                                 (name
                                                  (list (name* visible-chunk)
                                                        (name* hidden-chunk))))
                           known
                         (apply #'make-instance
                                class
                                :name name
                                :visible-chunk (->chunk visible-chunk
                                                        visible-chunks)
                                :hidden-chunk (->chunk hidden-chunk
                                                       hidden-chunks)
                                unknown)))))))
      (unless (unique-names-p clouds)
        (error "Name conflict among clouds: ~S." clouds))
      clouds)))

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
          (push `(:visible-chunk ,(name visible-chunk)
                  :hidden-chunk ,(name hidden-chunk))
                clouds))))
    clouds))

(defun merge-cloud-specs (specs default-specs)
  "Take DEFAULT-SPECS, a list of cloud specs, remove those that are
between chunks that have a spec in SPECS and add SPECS. If a spec has
CLASS NIL then remove it as well."
  (labels ((visible-name (spec)
             (if (listp spec)
                 (getf spec :visible-chunk)
                 (name (visible-chunk spec))))
           (hidden-name (spec)
             (if (listp spec)
                 (getf spec :hidden-chunk)
                 (name (hidden-chunk spec))))
           (match (spec1 spec2)
             (and (equal (visible-name spec1)
                         (visible-name spec2))
                  (equal (hidden-name spec1)
                         (hidden-name spec2)))))
    (remove-if (lambda (spec)
                 (null (getf spec :class 'full-cloud)))
               (append (remove-if (lambda (spec)
                                    (some (lambda (spec1)
                                            (match spec spec1))
                                          specs))
                                  default-specs)
                       specs))))

(defmethod initialize-instance :after
    ((rbm rbm) &key visible-chunks hidden-chunks
     (default-clouds (default-clouds visible-chunks hidden-chunks))
     clouds)
  "Return an RBM that consists of VISIBLE-CHUNKS, HIDDEN-CHUNKS and
CLOUDS of weights. Where CLOUDS is a list of cloud specifications.
Names of chunks and clouds shall be unique by EQUAL."
  (unless (unique-names-p visible-chunks)
    (error "Name conflict among visible chunks: ~S." visible-chunks))
  (unless (unique-names-p hidden-chunks)
    (error "Name conflict among hidden chunks ~S." hidden-chunks))
  (setf (slot-value rbm 'clouds)
        (->clouds visible-chunks hidden-chunks
                  (merge-cloud-specs clouds default-clouds)))
  ;; make sure chunks have the same MAX-N-STRIPES
  (setf (max-n-stripes rbm) (max-n-stripes rbm)))

(defun hijack-means-to-activation (rbm to-visible/hidden)
  "Set the chunks of TO-VISIBLE/HIDDEN layer of RBM to the activations
calculated from the other layer's nodes. Skip chunks that don't need
activations."
  (cond ((eq :hidden to-visible/hidden)
         (dolist (chunk (hidden-chunks rbm))
           (unless (conditioning-chunk-p chunk)
             (fill-chunk chunk #.(flt 0))))
         (do-clouds (cloud rbm)
           (unless (conditioning-chunk-p (hidden-chunk cloud))
             (activate-cloud cloud to-visible/hidden))))
        (t
         (dolist (chunk (visible-chunks rbm))
           (unless (conditioning-chunk-p chunk)
             (fill-chunk chunk #.(flt 0))))
         (do-clouds (cloud rbm)
           (unless (conditioning-chunk-p (visible-chunk cloud))
             (activate-cloud cloud to-visible/hidden))))))

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

(defmethod map-segments (fn (rbm rbm))
  (map nil (lambda (cloud)
             (map-segments fn cloud))
       (clouds rbm)))

(defmethod write-weights ((rbm rbm) stream)
  (dolist (cloud (clouds rbm))
    (write-weights cloud stream)))

(defmethod read-weights ((rbm rbm) stream)
  (dolist (cloud (clouds rbm))
    (read-weights cloud stream)))


;;;; Integration with train and gradient descent

(defclass rbm-trainer (segmented-gd-trainer)
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

(defmethod train (sampler (trainer rbm-trainer) (rbm rbm))
  (while (not (finishedp sampler))
    (train-batch (sample-batch sampler (n-inputs-until-update trainer))
                 trainer rbm)))

(defmethod train-batch (batch (trainer rbm-trainer) (rbm rbm))
  (loop for samples in (group batch (max-n-stripes rbm))
        do (set-input samples rbm)
        (positive-phase trainer rbm)
        (negative-phase trainer rbm))
  (maybe-update-weights trainer (length batch)))

(defun inputs->nodes (rbm)
  "Copy the previously clamped INPUTS to NODES as if SET-INPUT were
called with the same parameters."
  (map nil (lambda (chunk)
             (let ((inputs (inputs chunk)))
               (when inputs
                 (if (use-blas-on-chunk-p (cost-of-copy inputs) chunk)
                     (matlisp:copy! (inputs chunk) (nodes chunk))
                     (let ((inputs (storage (inputs chunk)))
                           (nodes (storage (nodes chunk))))
                       (declare (optimize (speed 3)))
                       (do-stripes (chunk)
                         (do-chunk (i chunk)
                           (setf (aref nodes i) (aref inputs i)))))))))
       (visible-chunks rbm)))

(defun nodes->inputs (rbm)
  "Copy NODES to INPUTS."
  (map nil (lambda (chunk)
             (let ((inputs (inputs chunk)))
               (when inputs
                 (if (use-blas-on-chunk-p (cost-of-copy inputs) chunk)
                     (matlisp:copy! (nodes chunk) (inputs chunk))
                     (let ((inputs (storage (inputs chunk)))
                           (nodes (storage (nodes chunk))))
                       (declare (optimize (speed 3)))
                       (do-stripes (chunk)
                         (do-chunk (i chunk)
                           (setf (aref inputs i) (aref nodes i)))))))))
       (visible-chunks rbm)))


;;;; Training implementation

(defgeneric accumulate-positive-phase-statistics (trainer rbm)
  (:method ((trainer rbm-trainer) rbm)
    (do-clouds (cloud rbm)
      (accumulate-cloud-statistics cloud trainer nil))))

(defgeneric accumulate-negative-phase-statistics (trainer rbm)
  (:method ((trainer rbm-trainer) rbm)
    (do-clouds (cloud rbm)
      (accumulate-cloud-statistics cloud trainer t))))

(defgeneric positive-phase (trainer rbm)
  (:method (trainer rbm)
    (set-hidden-mean rbm)
    (accumulate-positive-phase-statistics trainer rbm)))

(defgeneric negative-phase (trainer rbm)
  (:method ((trainer rbm-trainer) rbm)
    (let ((n-gibbs (n-gibbs trainer))
          (sample-visible-p (sample-visible-p trainer))
          (sample-hidden-p (sample-hidden-p trainer)))
      (assert (plusp n-gibbs))
      (loop for i below n-gibbs do
            (when sample-hidden-p
              (sample-hidden rbm))
            (set-visible-mean rbm)
            (when sample-visible-p
              (sample-visible rbm))
            (set-hidden-mean rbm))
      (accumulate-negative-phase-statistics trainer rbm))))


;;;; Convenience

(defun reconstruction-error (rbm)
  "Return the squared norm of INPUTS - NODES not considering constant
or conditioning chunks that aren't reconstructed in any case. The
second value returned is the number of nodes that contributed to the
error."
  (let ((sum #.(flt 0))
        (n 0))
    (declare (type flt sum) (type index n) (optimize (speed 3)))
    (dolist (chunk (visible-chunks rbm))
      (unless (conditioning-chunk-p chunk)
        (let ((inputs (storage (inputs chunk)))
              (nodes (storage (nodes chunk))))
          (do-stripes (chunk)
            (do-chunk (i chunk)
              (let ((x (aref inputs i))
                    (y (aref nodes i)))
                (incf sum (expt (- x y) 2))
                (incf n)))))))
    (values sum n)))

(defgeneric get-squared-error (rbm)
  (:method ((rbm rbm))
    (set-hidden-mean rbm)
    (set-visible-mean rbm)
    (reconstruction-error rbm)))
