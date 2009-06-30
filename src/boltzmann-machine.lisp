(in-package :mgl-bm)

;;;; Chunk

(defclass chunk ()
  ((name :initform (gensym) :initarg :name :reader name)
   (nodes
    :type matlisp:real-matrix :reader nodes
    :documentation "A value for each node in the chunk. First,
activations are put here (weights*inputs) then the mean of the
probability distribution is calculated from the activation and finally
\(optionally) a sample is taken from the probability distribution. All
these values are stored in this vector. This is also where SET-INPUT
is supposed to clamp the values. Note that not only the values in the
matrix but also the matrix object itself can change when the owning
network is used.")
   (old-nodes
    :type matlisp:real-matrix :reader old-nodes
    :documentation "The previous value of each node. Used to provide
parallel computation semanctics when there are intralayer connections.
Swapped with NODES at times.")
   (inputs
    :type (or matlisp:real-matrix null) :reader inputs
    :documentation "This is where SET-INPUT saves the input for later
use by RECONSTRUCTION-ERROR, INPUTS->NODES. It is NIL in
CONSTANT-CHUNKS.")
   (cache-static-activations-p
    :initform nil
    :initarg :cache-static-activations-p
    :reader cache-static-activations-p
    :documentation "Controls whether activations that do not change
between SET-INPUT calls \(i.e. they come from conditioning chunks
including biases) are cached.")
   (static-activations-cached-p
    :initform nil :accessor static-activations-cached-p)
   (static-activations
    :type (or matlisp:real-matrix null) :reader static-activations
    :documentation "Of the same size as NODES. This is where
activations coming from biases and conditioning chunks are between two
SET-INPUT calls. Gets more useful with a lot of conditioning and more
activations, for instance, with high N-GIBBS.")
   (indices-present
    :initform nil :initarg :indices-present :type (or null index-vector)
    :accessor indices-present
    :documentation "NIL or a simple vector of array indices into the
layer's NODES. Need not be ordered. SET-INPUT sets it. Note, that if
it is non-NIL then N-STRIPES must be 1.")
   (default-value
    :initform #.(flt 0) :initarg :default-value :type flt
    :reader default-value
    :documentation "Upon creation or resizing, the chunk's nodes get
filled with this value."))
  (:documentation "A chunk is a set of nodes of the same type in a
Boltzmann Machine. This is an abstract base class."))

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

(declaim (inline mat-max-n-stripes))
(defun mat-max-n-stripes (mat)
  (the index (/ (length (storage mat))
                (matlisp:nrows mat))))

(declaim (inline chunk-max-n-stripes))
(defun chunk-max-n-stripes (chunk)
  (mat-max-n-stripes (nodes chunk)))

(defmethod max-n-stripes ((chunk chunk))
  (chunk-max-n-stripes chunk))

(defmethod stripe-start (stripe (chunk chunk))
  (* stripe (chunk-size chunk)))

(defmethod stripe-end (stripe (chunk chunk))
  (* (1+ stripe) (chunk-size chunk)))

(defmethod print-object ((chunk chunk) stream)
  (print-unreadable-object (chunk stream :type t :identity t)
    (format stream "~S ~S(~S/~S)" (ignore-errors (name chunk))
            (ignore-errors (chunk-size chunk))
            (ignore-errors (chunk-n-stripes chunk))
            (ignore-errors (chunk-max-n-stripes chunk))))
  chunk)

;;; Currently the lisp code handles only the single stripe case and
;;; blas cannot deal with no missing values.
(defun check-stripes (chunk)
  (let ((indices-present (indices-present chunk)))
    (assert (or (null indices-present)
                (= 1 (chunk-n-stripes chunk))))))

(defun use-blas-on-chunk-p (cost chunk)
  (check-stripes chunk)
  (cond ((indices-present chunk)
         ;; there is no missing value support in blas
         nil)
        (t
         ;; several stripes or cost is high => blas
         (or (< 1 (chunk-n-stripes chunk))
             (use-blas-p cost)))))

(defun ->chunk (chunk-designator chunks)
  (if (typep chunk-designator 'chunk)
      chunk-designator
      (or (find chunk-designator chunks :key #'name :test #'equal)
          (error "Cannot find chunk ~S." chunk-designator))))

(defvar *current-stripe*)

(defmacro do-stripes ((chunk &optional (stripe (gensym))) &body body)
  (with-gensyms (%chunk)
    `(let ((,%chunk ,chunk))
       (check-stripes ,%chunk)
       (dotimes (,stripe (chunk-n-stripes ,%chunk))
         (let ((*current-stripe* ,stripe))
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

(defun fill-chunk (chunk value &key allp)
  (declare (type flt value))
  (if (or allp (use-blas-on-chunk-p (cost-of-fill (nodes chunk)) chunk))
      (matlisp:fill-matrix (nodes chunk) value)
      (let ((nodes (storage (nodes chunk))))
        (declare (optimize (speed 3) #.*no-array-bounds-check*))
        (do-stripes (chunk)
          (do-chunk (i chunk)
            (setf (aref nodes i) value))))))

(defun zero-chunk (chunk)
  (fill-chunk chunk #.(flt 0)))

(defun zero-chunks (chunks)
  (map nil #'zero-chunk chunks))

(defun sum-chunk-nodes-and-old-nodes (chunk node-weight old-node-weight)
  (matlisp:scal! (flt node-weight) (nodes chunk))
  (matlisp:scal! (flt old-node-weight) (old-nodes chunk))
  (matlisp:m+! (old-nodes chunk) (nodes chunk)))

(defun sum-nodes-and-old-nodes (chunks node-weight old-node-weight)
  (map nil (lambda (chunk)
             (sum-chunk-nodes-and-old-nodes chunk node-weight old-node-weight))
       chunks))

(defclass conditioning-chunk (chunk) ()
  (:documentation "Nodes in CONDITIONING-CHUNK never change their
values on their own so they are to be clamped. Including this chunk in
the visible layer allows `conditional' RBMs."))

(defun conditioning-chunk-p (chunk)
  (typep chunk 'conditioning-chunk))

(defgeneric resize-chunk (chunk size max-n-stripes)
  (:method ((chunk chunk) size max-n-stripes)
    (unless (and (slot-boundp chunk 'nodes)
                 (= size (chunk-size chunk))
                 (= max-n-stripes (chunk-max-n-stripes chunk)))
      (setf (slot-value chunk 'nodes)
            (matlisp:make-real-matrix size max-n-stripes))
      (setf (slot-value chunk 'old-nodes)
            (if (conditioning-chunk-p chunk)
                (nodes chunk)
                (matlisp:make-real-matrix size max-n-stripes)))
      (fill-chunk chunk (default-value chunk) :allp t)
      (setf (slot-value chunk 'inputs)
            (if (typep chunk 'conditioning-chunk)
                nil
                (matlisp:make-real-matrix size max-n-stripes)))
      (setf (slot-value chunk 'static-activations)
            (if (cache-static-activations-p chunk)
                (matlisp:make-real-matrix size max-n-stripes)
                nil)))))

(defmethod set-n-stripes (n-stripes (chunk chunk))
  (set-ncols (nodes chunk) n-stripes)
  (set-ncols (old-nodes chunk) n-stripes)
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
  (fill-chunk chunk (default-value chunk) :allp t))

(defclass sigmoid-chunk (chunk) ()
  (:documentation "Nodes in a sigmoid chunk have two possible samples:
0 and 1. The probability of a node being on is given by the sigmoid of
its activation."))

(defclass gaussian-chunk (chunk) ()
  (:documentation "Nodes are real valued. The sample of a node is its
activation plus guassian noise of unit variance."))

(defclass normalized-group-chunk (chunk)
  ((scale
    :initform #.(flt 1) :type (or flt flt-vector)
    :initarg :scale :accessor scale
    :documentation "The sum of the means after normalization. Can be
changed during training, for instance when clamping. If it is a vector
then its length must be MAX-N-STRIPES which automatically maintained
when changing the number of stripes.")
   (group-size
    :initform (error "GROUP-SIZE must be specified.")
    :initarg :group-size
    :reader group-size))
  (:documentation "Means are normalized to SCALE within groups of
GROUP-SIZE."))

(defmethod resize-chunk ((chunk normalized-group-chunk) size max-n-stripes)
  (call-next-method)
  (when (and (typep (scale chunk) 'flt-vector)
             (/= (max-n-stripes chunk) (length (scale chunk))))
    (setf (scale chunk) (make-flt-array (max-n-stripes chunk)))))

(defclass exp-normalized-group-chunk (normalized-group-chunk) ()
  (:documentation "Means are normalized (EXP ACTIVATION)."))

(defclass softmax-chunk (exp-normalized-group-chunk) ()
  (:documentation "Binary units with normalized (EXP ACTIVATION)
firing probabilities representing a multinomial distribution. That is,
samples have exactly one 1 in each group of GROUP-SIZE."))

(defclass constrained-poisson-chunk (exp-normalized-group-chunk) ()
  (:documentation "Poisson units with normalized (EXP ACTIVATION) means."))

(defclass temporal-chunk (conditioning-chunk)
  ((hidden-source-chunk
    :initarg :hidden-source-chunk
    :reader hidden-source-chunk)
   (next-node-inputs :reader next-node-inputs)
   (has-inputs-p :initform nil :reader has-inputs-p))
  (:documentation "After a SET-HIDDEN-MEAN, the means of
HIDDEN-SOURCE-CHUNK are stored in NEXT-NODE-INPUTS and on the next
SET-INPUT copied onto NODES. If there are multiple SET-HIDDEN-MEAN
calls between two SET-INPUT calls then only the first set of values
are remembered."))

(defmethod resize-chunk ((chunk temporal-chunk) size max-n-stripes)
  (call-next-method)
  (unless (and (slot-boundp chunk 'next-node-inputs)
               (= size (matlisp:nrows (next-node-inputs chunk)))
               (= max-n-stripes (mat-max-n-stripes (next-node-inputs chunk))))
    (setf (slot-value chunk 'next-node-inputs)
          (matlisp:make-real-matrix size max-n-stripes))))

(defun copy-chunk-nodes (chunk from to)
  (if (use-blas-on-chunk-p (cost-of-copy from) chunk)
      (matlisp:copy! from to)
      (let ((from (storage from))
            (to (storage to)))
        (declare (optimize (speed 3)))
        (do-stripes (chunk)
          (do-chunk (i chunk)
            (setf (aref to i) (aref from i)))))))

(defun maybe-remember (chunk)
  (unless (has-inputs-p chunk)
    (let ((hidden (hidden-source-chunk chunk)))
      (assert (null (indices-present hidden)))
      (copy-chunk-nodes chunk (nodes hidden) (next-node-inputs chunk))
      (setf (slot-value chunk 'has-inputs-p) t))))

(defun maybe-use-remembered (chunk)
  (when (has-inputs-p chunk)
    (setf (indices-present chunk) nil)
    (copy-chunk-nodes chunk (next-node-inputs chunk) (nodes chunk))
    (setf (slot-value chunk 'has-inputs-p) nil)))

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
      (declare (type (or flt flt-vector) scale)
               (type index group-size))
      (assert (zerop (mod (chunk-size chunk) group-size)))
      (do-stripes (chunk stripe)
        (let ((scale (if (typep scale 'flt) scale (aref scale stripe))))
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
                               (/ (aref nodes j) sum))))))))))
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
    (let ((nodes (storage (nodes chunk))))
      (do-stripes (chunk)
        (do-chunk (i chunk)
          (setf (aref nodes i) (flt (poisson-random (aref nodes i)))))))))


;;;; Cloud

(defclass cloud ()
  ((name :initarg :name :reader name)
   (chunk1 :type chunk :initarg :chunk1 :reader chunk1)
   (chunk2 :type chunk :initarg :chunk2 :reader chunk2)
   (scale1
    :type flt :initform #.(flt 1) :initarg :scale1 :reader scale1
    :documentation "When CHUNK1 is being activated count activations
coming from this cloud multiplied by SCALE1.")
   (scale2
    :type flt :initform #.(flt 1) :initarg :scale2 :reader scale2
    :documentation "When CHUNK2 is being activated count activations
coming from this cloud multiplied by SCALE2."))
  (:documentation "A set of connections between two chunks. The chunks
may be the same, be both visible or both hidden subject to constraints
imposed by the type of boltzmann machine the cloud is part of."))

(defmethod print-object ((cloud cloud) stream)
  (print-unreadable-object (cloud stream :type t :identity t)
    (when (slot-boundp cloud 'name)
      (format stream "~S" (name cloud))))
  cloud)

(defmethod set-n-stripes (n-stripes (cloud cloud)))
(defmethod set-max-n-stripes (max-n-stripes (cloud cloud)))

(defgeneric activate-cloud (cloud reversep)
  (:documentation "From OLD-NODES of CHUNK1 calculate the activations
of CHUNK2 and _add_ them to NODES of CHUNK2. If REVERSEP then swap the
roles of the chunks. In the simplest case it adds weights (of CLOUD) *
OLD-NODES (of CHUNK1) to the nodes of the hidden chunk."))

(defgeneric accumulate-cloud-statistics (trainer cloud multiplier)
  (:documentation "Take the accumulator of TRAINER that corresponds to
CLOUD and add MULTIPLIER times the cloud statistics of [persistent]
contrastive divergence."))

(defun hijack-means-to-activation (chunks clouds)
  "Set NODES of CHUNKS to the activations calculated from CLOUDS. Skip
chunks that don't need activations."
  (flet ((to-cache-p (chunk)
           (and (cache-static-activations-p chunk)
                (not (static-activations-cached-p chunk)))))
    ;; Zero activations or copy cached activations coming from
    ;; conditioning chunks.
    (dolist (chunk chunks)
      (unless (conditioning-chunk-p chunk)
        (if (static-activations-cached-p chunk)
            (copy-chunk-nodes chunk (static-activations chunk) (nodes chunk))
            (fill-chunk chunk #.(flt 0)))))
    ;; Calculate the activations coming from conditioning chunks if
    ;; they are to be cached.
    (flet ((foo (to-chunk from-chunk cloud)
             (when (and (to-cache-p to-chunk)
                        (conditioning-chunk-p from-chunk))
               (activate-cloud cloud (eq (chunk2 cloud) from-chunk)))))
      (dolist (cloud clouds)
        (cond ((and (eq (chunk1 cloud) (chunk2 cloud))
                    (member (chunk1 cloud) chunks))
               (foo (chunk1 cloud) (chunk2 cloud) cloud))
              ((member (chunk1 cloud) chunks)
               (foo (chunk1 cloud) (chunk2 cloud) cloud))
              ((member (chunk2 cloud) chunks)
               (foo (chunk2 cloud) (chunk1 cloud) cloud)))))
    ;; Remember those.
    (dolist (chunk chunks)
      (when (to-cache-p chunk)
        (copy-chunk-nodes chunk (nodes chunk) (static-activations chunk))
        (setf (static-activations-cached-p chunk) t)))
    ;; By now chunks with STATIC-ACTIVATIONS-CACHED-P have the bias
    ;; activations in NODES. Do the non-cached activations.
    (flet ((foo (to-chunk from-chunk cloud)
             (unless (or (conditioning-chunk-p to-chunk)
                         (and (static-activations-cached-p to-chunk)
                              (conditioning-chunk-p from-chunk)))
               (activate-cloud cloud (eq (chunk2 cloud) from-chunk)))))
      (dolist (cloud clouds)
        (cond ((and (eq (chunk1 cloud) (chunk2 cloud))
                    (member (chunk1 cloud) chunks))
               (foo (chunk1 cloud) (chunk2 cloud) cloud))
              ((member (chunk1 cloud) chunks)
               (foo (chunk1 cloud) (chunk2 cloud) cloud))
              ((member (chunk2 cloud) chunks)
               (foo (chunk2 cloud) (chunk1 cloud) cloud)))))))

;;; See if both ends of CLOUD are among CHUNKS.
(defun both-cloud-ends-in-p (cloud chunks)
  (and (member (chunk1 cloud) chunks)
       (member (chunk2 cloud) chunks)))

(defgeneric zero-weight-to-self (cloud)
  (:documentation "In a BM W_{i,i} is always zero."))

(defmethod activate-cloud :before (cloud reversep)
  (zero-weight-to-self cloud))

;;; Return the chunk of CLOUD that's among CHUNKS and the other chunk
;;; of CLOUD as the second value.
(defun cloud-chunk-among-chunks (cloud chunks)
  (cond ((member (chunk1 cloud) chunks)
         (values (chunk1 cloud) (chunk2 cloud)))
        ((member (chunk2 cloud) chunks)
         (values (chunk2 cloud) (chunk1 cloud)))
        (t
         (values nil nil))))

(defun cloud-between-chunks-p (cloud chunks1 chunks2)
  (or (and (member (chunk1 cloud) chunks1)
           (member (chunk2 cloud) chunks2))
      (and (member (chunk1 cloud) chunks2)
           (member (chunk2 cloud) chunks1))))


;;;; Full cloud

(defclass full-cloud (cloud)
  ((weights
    :type matlisp:real-matrix :initarg :weights :reader weights
    :documentation "In Matlisp, chunks are represented as column
vectors \(disregarding the multi-striped case). If the visible chunk
is Nx1 and the hidden is Mx1 then the weight matrix is MxN. Hidden =
hidden + weights * visible. Visible = visible + weights^T * hidden.
Looking directly at the underlying Lisp array \(MATLISP::STORE), it's
all transposed.")))

(defmethod initialize-instance :after ((cloud full-cloud)
                                       &key &allow-other-keys)
  (unless (slot-boundp cloud 'weights)
    (setf (slot-value cloud 'weights)
          (matlisp:make-real-matrix (chunk-size (chunk2 cloud))
                                    (chunk-size (chunk1 cloud))))
    (map-into (storage (weights cloud))
              (lambda () (flt (* 0.01 (gaussian-random-1)))))))

(defmacro do-cloud-runs (((start end) cloud) &body body)
  "Iterate over consecutive runs of weights present in CLOUD."
  (with-gensyms (%cloud %chunk2-size %index)
    `(let ((,%cloud ,cloud))
       (if (indices-present (chunk1 ,%cloud))
           (let ((,%chunk2-size (chunk-size (chunk2 ,%cloud))))
             (do-stripes ((chunk1 ,%cloud))
               (do-chunk (,%index (chunk1 ,%cloud))
                 (let* ((,start (the! index (* ,%index ,%chunk2-size)))
                        (,end (the! index (+ ,start ,%chunk2-size))))
                   ,@body))))
           (let ((,start 0)
                 (,end (matlisp:number-of-elements (weights ,%cloud))))
             ,@body)))))

(eval-when (:compile-toplevel :load-toplevel :execute)
  (defun make-do-cloud/chunk2 (chunk2-index index chunk2-size
                               offset body)
    `(do ((,chunk2-index 0 (the! index (1+ ,chunk2-index)))
          (,index ,offset (the! index (1+ ,index))))
         ((>= ,chunk2-index ,chunk2-size))
       ,@body)))

(defmacro do-cloud/chunk1 ((chunk1-index cloud) &body body)
  (with-gensyms (%cloud %chunk2-size %offset)
    `(let* ((,%cloud ,cloud)
            (,%chunk2-size (chunk-size (chunk2 ,%cloud))))
       (declare (type index ,%chunk2-size))
       (when (indices-present (chunk2 ,%cloud))
         (error "CHUNK2 cannot have INDICES-PRESENT."))
       (do-stripes ((chunk1 ,%cloud))
         (do-chunk (,chunk1-index (chunk1 ,%cloud))
           (let ((,%offset (the! index
                                 (* ,chunk1-index ,%chunk2-size))))
             (macrolet ((do-cloud/chunk2 ((chunk2-index index) &body body)
                          (make-do-cloud/chunk2 chunk2-index index
                                                ',%chunk2-size ',%offset
                                                body)))
               ,@body)))))))

(defmethod zero-weight-to-self ((cloud full-cloud))
  (when (eq (chunk1 cloud) (chunk2 cloud))
    (let ((weights (weights cloud)))
      (loop for i below (chunk-size (chunk1 cloud)) do
            (setf (matlisp:matrix-ref weights i i) #.(flt 0))))))

(defmethod activate-cloud ((cloud full-cloud) reversep)
  (declare (optimize (speed 3) #.*no-array-bounds-check*))
  (if (not reversep)
      (let ((weights (weights cloud))
            (from (old-nodes (chunk1 cloud)))
            (to (nodes (chunk2 cloud)))
            (scale (scale2 cloud)))
        (declare (type flt scale))
        (if (use-blas-on-chunk-p (cost-of-gemm weights from :nn)
                                 (chunk1 cloud))
            (matlisp:gemm! scale weights from (flt 1) to)
            (let ((weights (storage weights))
                  (from (storage from))
                  (to (storage to)))
              (declare (type flt-vector weights from to))
              (do-cloud/chunk1 (i cloud)
                (let ((x (aref from i)))
                  (unless (zerop x)
                    (setq x (* x scale))
                    (do-cloud/chunk2 (j weight-index)
                      (incf (aref to j)
                            (* x (aref weights weight-index))))))))))
      (let ((weights (weights cloud))
            (from (old-nodes (chunk2 cloud)))
            (to (nodes (chunk1 cloud)))
            (scale (scale1 cloud)))
        (declare (type flt scale))
        (if (use-blas-on-chunk-p (cost-of-gemm weights from :tn)
                                 (chunk1 cloud))
            (matlisp:gemm! scale weights from (flt 1) to :tn)
            (let ((weights (storage weights))
                  (from (storage from))
                  (to (storage to)))
              (declare (type flt-vector weights from to))
              (do-cloud/chunk1 (i cloud)
                (let ((sum #.(flt 0)))
                  (declare (type flt sum))
                  (do-cloud/chunk2 (j weight-index)
                    (incf sum (* (aref from j)
                                 (aref weights weight-index))))
                  (incf (aref to i) (* sum scale)))))))))

(defmethod accumulate-cloud-statistics (trainer (cloud full-cloud) multiplier)
  (declare (type flt multiplier))
  (with-segment-gradient-accumulator ((start accumulator)
                                      (cloud trainer))
    (when (and accumulator start)
      (let ((v1 (nodes (chunk1 cloud)))
            (v2 (nodes (chunk2 cloud))))
        (if (and (zerop start)
                 (use-blas-on-chunk-p (cost-of-gemm v2 v1 :nt)
                                      (chunk1 cloud)))
            (matlisp:gemm! multiplier v2 v1 (flt 1)
                           (reshape2 accumulator
                                     (matlisp:nrows v2)
                                     (matlisp:nrows v1))
                           :nt)
            (let ((v1 (storage v1))
                  (v2 (storage v2))
                  (accumulator (storage accumulator)))
              (declare (optimize (speed 3) #.*no-array-bounds-check*))
              (cond ((= multiplier (flt 1))
                     (special-case (zerop start)
                       (do-cloud/chunk1 (i cloud)
                         (let ((x (aref v1 i)))
                           (unless (zerop x)
                             (do-cloud/chunk2 (j weight-index)
                               (incf (aref accumulator
                                           (the! index (+ start weight-index)))
                                     (* x (aref v2 j)))))))))
                    ((= multiplier (flt -1))
                     (special-case (zerop start)
                       (do-cloud/chunk1 (i cloud)
                         (let ((x (aref v1 i)))
                           (unless (zerop x)
                             (do-cloud/chunk2 (j weight-index)
                               (decf (aref accumulator
                                           (the! index (+ start weight-index)))
                                     (* x (aref v2 j)))))))))
                    (t
                     (special-case (zerop start)
                       (do-cloud/chunk1 (i cloud)
                         (let ((x (* multiplier (aref v1 i))))
                           (unless (zerop x)
                             (do-cloud/chunk2 (j weight-index)
                               (incf (aref accumulator
                                           (the! index (+ start weight-index)))
                                     (* x (aref v2 j))))))))))))))))

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

;;; HACK: subclass CONDITIONING-CHUNK just to make OLD-NODES and NODES
;;; the same.
(defclass factored-cloud-shared-chunk (conditioning-chunk) ())

(defmethod initialize-instance :after ((cloud factored-cloud) &key rank
                                       &allow-other-keys)
  (assert (typep rank '(or (integer 1) null)))
  (unless (and (slot-boundp cloud 'cloud-a)
               (slot-boundp cloud 'cloud-b))
    (let ((shared (make-instance 'factored-cloud-shared-chunk
                                 :size rank
                                 :name (list (name cloud) :shared))))
      (setf (slot-value cloud 'cloud-a)
            (make-instance 'full-cloud
                           :name (list (name cloud) :a)
                           :chunk1 shared
                           :chunk2 (chunk2 cloud)
                           :scale2 (scale2 cloud)))
      (setf (slot-value cloud 'cloud-b)
            (make-instance 'full-cloud
                           :name (list (name cloud) :b)
                           :chunk1 (chunk1 cloud)
                           :chunk2 shared
                           :scale1 (scale1 cloud))))))

(defun factored-cloud-shared-chunk (cloud)
  (chunk1 (cloud-a cloud)))

(defun rank (cloud)
  (chunk-size (factored-cloud-shared-chunk cloud)))

(defmethod set-n-stripes (n-stripes (cloud factored-cloud))
  (setf (n-stripes (factored-cloud-shared-chunk cloud)) n-stripes))

(defmethod set-max-n-stripes (max-n-stripes (cloud factored-cloud))
  (setf (max-n-stripes (factored-cloud-shared-chunk cloud)) max-n-stripes))

(defmethod zero-weight-to-self ((cloud factored-cloud))
  (when (eq (chunk1 cloud) (chunk2 cloud))
    (error "ZERO-WEIGHT-TO-SELF not implemented for FACTORED-CLOUD")))

(defmethod activate-cloud ((cloud factored-cloud) reversep)
  ;; Normal chunks are zeroed by HIJACK-MEANS-TO-ACTIVATION.
  (fill-chunk (factored-cloud-shared-chunk cloud) #.(flt 0))
  (cond ((not reversep)
         (activate-cloud (cloud-b cloud) reversep)
         (activate-cloud (cloud-a cloud) reversep))
        (t
         (activate-cloud (cloud-a cloud) reversep)
         (activate-cloud (cloud-b cloud) reversep))))

(defmethod accumulate-cloud-statistics (trainer (cloud factored-cloud)
                                        multiplier)
  (declare (type flt multiplier))
  (let* ((chunk1 (chunk1 cloud))
         (v (nodes chunk1))
         (h (nodes (chunk2 cloud)))
         (a (weights (cloud-a cloud)))
         (b (weights (cloud-b cloud)))
         (n-stripes (n-stripes (chunk1 cloud)))
         (c (matlisp:nrows b))
         (shared (factored-cloud-shared-chunk cloud))
         (v* (storage v))
         (shared* (storage (nodes shared))))
    (check-stripes chunk1)
    (with-segment-gradient-accumulator ((start accumulator)
                                        ((cloud-a cloud) trainer))
      (when (and accumulator start)
        ;; dCD/dA ~= h*v'*B'
        (let ((x (reshape2 (nodes shared) n-stripes c)))
          (if (and (zerop start)
                   (null (indices-present chunk1)))
              (matlisp:gemm! (flt 1) v b (flt 0) x :tt)
              (let ((b* (storage b)))
                (declare (optimize (speed 3) #.*no-array-bounds-check*))
                (matlisp:fill-matrix x (flt 0))
                (do-stripes (chunk1)
                  (do-chunk (i chunk1)
                    (let ((v*i (aref v* i)))
                      (unless (zerop v*i)
                        (loop for j of-type index upfrom 0 below c
                              for bi of-type index
                              upfrom (the! index (* i c)) do
                              (incf (aref shared* j)
                                    (* v*i (aref b* bi))))))))))
          (matlisp:gemm! multiplier h x
                         (flt 1) (reshape2 accumulator
                                           (matlisp:nrows a)
                                           (matlisp:ncols a))))))
    (with-segment-gradient-accumulator ((start accumulator)
                                        ((cloud-b cloud) trainer))
      (when (and accumulator start)
        ;; dCD/dB ~= A'*h*v'
        (let ((x (reshape2 (nodes shared) c n-stripes)))
          (matlisp:gemm! (flt 1) a h (flt 0) x :tn)
          (if (and (zerop start)
                   (null (indices-present (chunk1 cloud))))
              (matlisp:gemm! multiplier x v
                             (flt 1) (reshape2 accumulator
                                               (matlisp:nrows b)
                                               (matlisp:ncols b))
                             :nt)
              (let ((acc* (storage accumulator)))
                (declare (optimize (speed 3) #.*no-array-bounds-check*))
                (do-stripes (chunk1)
                  (do-chunk (i chunk1)
                    (let ((v*i (* multiplier (aref v* i))))
                      (unless (zerop v*i)
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


;;;; Boltzmann Machine

(defclass bm ()
  ((visible-chunks
    :type list :initarg :visible-chunks :reader visible-chunks
    :documentation "A list of CHUNKs whose values come from the
outside world: SET-INPUT sets them.")
   (hidden-chunks
    :type list :initarg :hidden-chunks :reader hidden-chunks
    :documentation "A list of CHUNKs that are not directly observed.
Disjunct from VISIBLE-CHUNKS.")
   (clouds
    :type list :initform '(:merge) :initarg :clouds :reader clouds
    :documentation "Normally, a list of CLOUDS representing the
connections between chunks. During initialization cloud specs are
allowed in the list.")
   (has-hidden-to-hidden-p :reader has-hidden-to-hidden-p)
   (has-visible-to-visible-p :reader has-visible-to-visible-p)
   (max-n-stripes :initform 1 :initarg :max-n-stripes :reader max-n-stripes))
  (:documentation "The network is assembled from CHUNKS (nodes of the
same behaviour) and CLOUDs (connections between two chunks). To
instantiate, arrange for VISIBLE-CHUNKS, HIDDEN-CHUNKS, CLOUDS (either
as initargs or initforms) to be set.

Usage of CLOUDS is slightly tricky: you may pass a list of CLOUD
objects connected to chunks in this network. Alternatively, a cloud
spec may stand for a cloud. Also, the initial value of CLOUDS is
merged with the default cloud spec list before the final cloud spec
list is instantiated. The default cloud spec list is what
FULL-CLOUDS-EVERYWHERE returns for VISIBLE-CHUNKS and HIDDEN-CHUNKS.
See MERGE-CLOUD-SPECS for the gory details. The initform, '(:MERGE),
simply leaves the default cloud specs alone."))

(defgeneric find-chunk (name object &key errorp)
  (:documentation "Find the chunk in OBJECT whose name is EQUAL to
NAME. Signal an error if not found and ERRORP.")
  (:method (name (bm bm) &key errorp)
    (or (find name (visible-chunks bm) :key #'name :test #'equal)
        (find name (hidden-chunks bm) :key #'name :test #'equal)
        (if errorp
            (error "Cannot find chunk ~S." name)
            nil))))

(defmacro do-clouds ((cloud bm) &body body)
  `(dolist (,cloud (clouds ,bm))
     ,@body))

(defmethod n-stripes ((bm bm))
  (n-stripes (first (visible-chunks bm))))

(defmethod set-n-stripes (n-stripes (bm bm))
  (dolist (chunk (visible-chunks bm))
    (setf (n-stripes chunk) n-stripes))
  (dolist (chunk (hidden-chunks bm))
    (setf (n-stripes chunk) n-stripes))
  (do-clouds (cloud bm)
    (setf (n-stripes cloud) n-stripes)))

(defmethod set-max-n-stripes (max-n-stripes (bm bm))
  (setf (slot-value bm 'max-n-stripes) max-n-stripes)
  (dolist (chunk (visible-chunks bm))
    (setf (max-n-stripes chunk) max-n-stripes))
  (dolist (chunk (hidden-chunks bm))
    (setf (max-n-stripes chunk) max-n-stripes))
  (do-clouds (cloud bm)
    (setf (max-n-stripes cloud) max-n-stripes)))

(defgeneric find-cloud (name object &key errorp)
  (:documentation "Find the cloud in OBJECT whose name is EQUAL to
NAME. Signal an error if not found and ERRORP.")
  (:method (name (bm bm) &key errorp)
    (or (find name (clouds bm) :key #'name :test #'equal)
        (if errorp
            (error "Cannot find cloud ~S." name)
            nil))))

(defun ->cloud (cloud-designator bm)
  (if (typep cloud-designator 'cloud)
      cloud-designator
      (find-cloud cloud-designator bm :errorp t)))

(defun ->clouds (chunks cloud-specs)
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
                         (split-plist spec '(:class :name :chunk1 :chunk2))
                       (destructuring-bind (&key (class 'full-cloud)
                                                 chunk1 chunk2
                                                 (name
                                                  (list (name* chunk1)
                                                        (name* chunk2))))
                           known
                         (apply #'make-instance
                                class
                                :name name
                                :chunk1 (->chunk chunk1 chunks)
                                :chunk2 (->chunk chunk2 chunks)
                                unknown)))))))
      (when (name-clashes clouds)
        (error "Name conflict among clouds: ~S." (name-clashes clouds)))
      clouds)))

(defun name-clashes (list)
  (let ((names (mapcar #'name list)))
    (set-difference names
                    (remove-duplicates names :test #'equal)
                    :test #'equal)))

(defun full-clouds-everywhere (visible-chunks hidden-chunks)
  "Return a list of cloud specifications suitable for instantiating an
BM. Put a cloud between each pair of visible and hidden chunks unless
they are both conditioning chunks. The names of the clouds are two
element lists of the names of the visible and hidden chunks."
  (let ((clouds '()))
    (dolist (visible-chunk visible-chunks)
      (dolist (hidden-chunk hidden-chunks)
        (unless (and (conditioning-chunk-p visible-chunk)
                     (conditioning-chunk-p hidden-chunk))
          (push `(:chunk1 ,(name visible-chunk)
                  :chunk2 ,(name hidden-chunk))
                clouds))))
    (nreverse clouds)))

(defun merge-cloud-specs (specs default-specs)
  "Combine cloud SPECS and DEFAULT-SPECS. If the first element of
SPECS is :MERGE then merge them else return SPECS. Merging
concatenates them but removes those specs from DEFAULT-SPECS that are
between chunks that have a spec in SPECS. If a spec has CLASS NIL then
it is removed as well. A cloud spec at minimum specifies the name of
the chunks it connects:

  (:chunk1 inputs :chunk2 features)

in which case it defaults to be a FULL-CLOUD. If that is not desired
then the class can be specified:

  (:chunk1 inputs :chunk2 features :class factored-cloud)

To remove a cloud from DEFAULT-SPECS use :CLASS NIL:

  (:chunk1 inputs :chunk2 features :class nil)

Other initargs are passed as is to MAKE-INSTANCE:

  (:chunk1 inputs :chunk2 features :class factored-cloud :rank 10)

You may also pass a CLOUD object as a spec."
  (labels ((chunk1-name (spec)
             (if (listp spec)
                 (getf spec :chunk1)
                 (name (chunk1 spec))))
           (chunk2-name (spec)
             (if (listp spec)
                 (getf spec :chunk2)
                 (name (chunk2 spec))))
           (match (spec1 spec2)
             (and (equal (chunk1-name spec1)
                         (chunk1-name spec2))
                  (equal (chunk2-name spec1)
                         (chunk2-name spec2)))))
    (if (eq :merge (first specs))
        (let ((specs (rest specs)))
          (remove-if (lambda (spec)
                       (and (not (typep spec 'cloud))
                            (null (getf spec :class 'full-cloud))))
                     (append (remove-if (lambda (spec)
                                          (some (lambda (spec1)
                                                  (match spec spec1))
                                                specs))
                                        default-specs)
                             specs)))
        specs)))

(defmethod initialize-instance :after ((bm bm) &key &allow-other-keys)
  "Return an BM that consists of VISIBLE-CHUNKS, HIDDEN-CHUNKS and
CLOUDS of weights where CLOUDS is a list of cloud specifications.
Names of chunks and clouds must be unique under EQUAL. CLOUDS is
merged with DEFAULT-CLOUDS. DEFAULT-CLOUDS defaults to connecting all
visible and hidden chunks with FULL-CLOUDS without any intralayer
connection. See MERGE-CLOUD-SPECS on the semantics of merging."
  (let* ((visible-chunks (visible-chunks bm))
         (hidden-chunks (hidden-chunks bm))
         (name-clashes (name-clashes (append visible-chunks hidden-chunks))))
    (when name-clashes
      (error "Name conflict among chunks ~S." name-clashes))
    (unless (every (lambda (obj) (typep obj 'cloud)) (clouds bm))
      (setf (slot-value bm 'clouds)
            (->clouds (append visible-chunks hidden-chunks)
                      (merge-cloud-specs (clouds bm)
                                         (full-clouds-everywhere
                                          visible-chunks
                                          hidden-chunks)))))
    ;; make sure chunks have the same MAX-N-STRIPES
    (setf (max-n-stripes bm) (max-n-stripes bm))
    (setf (slot-value bm 'has-visible-to-visible-p)
          (not (not (some (lambda (cloud)
                            (both-cloud-ends-in-p cloud visible-chunks))
                          (clouds bm)))))
    (setf (slot-value bm 'has-hidden-to-hidden-p)
          (not (not (some (lambda (cloud)
                            (both-cloud-ends-in-p cloud hidden-chunks))
                          (clouds bm)))))))

(defun swap-nodes (chunks)
  (dolist (chunk chunks)
    (rotatef (slot-value chunk 'nodes)
             (slot-value chunk 'old-nodes))))

(defun set-visible-mean (bm)
  "Set NODES of the chunks in the visible layer to the means of their
respective probability distributions."
  (swap-nodes (visible-chunks bm))
  (swap-nodes (hidden-chunks bm))
  (hijack-means-to-activation (visible-chunks bm) (clouds bm))
  (map nil #'set-chunk-mean (visible-chunks bm))
  ;; Hiddens did not change. Simply swap them back.
  (swap-nodes (hidden-chunks bm)))

(defun set-hidden-mean (bm)
  "Set NODES of the chunks in the hidden layer to the means of their
respective probability distributions."
  (swap-nodes (visible-chunks bm))
  (swap-nodes (hidden-chunks bm))
  (hijack-means-to-activation (hidden-chunks bm) (clouds bm))
  (map nil #'set-chunk-mean (hidden-chunks bm))
  ;; Visibles did not change. Simply swap them back.
  (swap-nodes (visible-chunks bm))
  (dolist (chunk (visible-chunks bm))
    (when (typep chunk 'temporal-chunk)
      (maybe-remember chunk))))

(defun sample-visible (bm)
  "Generate samples from the probability distribution defined by the
chunk type and the mean that resides in NODES."
  (map nil #'sample-chunk (visible-chunks bm)))

(defun sample-hidden (bm)
  "Generate samples from the probability distribution defined by the
chunk type and the mean that resides in NODES."
  (map nil #'sample-chunk (hidden-chunks bm)))

(defmethod set-input :around (samples (bm bm))
  (setf (n-stripes bm) (length samples))
  (flet ((clear-static-activations ()
           (dolist (chunk (visible-chunks bm))
             (setf (static-activations-cached-p chunk) nil))
           (dolist (chunk (hidden-chunks bm))
             (setf (static-activations-cached-p chunk) nil))))
    (unwind-protect
         ;; Do any clamping specific to this BM.
         (progn
           (dolist (chunk (visible-chunks bm))
             (when (typep chunk 'temporal-chunk)
               (maybe-use-remembered chunk)))
           ;; STATIC-ACTIVATIONS-CACHED-P is cleared only before
           ;; (CALL-NEXT-METHOD), therefore it is expected that it
           ;; does not activate the bm (as in SET-HIDDEN-MEAN,
           ;; SET-VISIBLE-MEAN) before writing the final values to all
           ;; conditioning chunks.
           (clear-static-activations)
           (call-next-method))
      ;; Then remember the inputs.
      (nodes->inputs bm))))

(defmethod map-segments (fn (bm bm))
  (map nil (lambda (cloud)
             (map-segments fn cloud))
       (clouds bm)))

(defmethod write-weights ((bm bm) stream)
  (dolist (cloud (clouds bm))
    (write-weights cloud stream)))

(defmethod read-weights ((bm bm) stream)
  (dolist (cloud (clouds bm))
    (read-weights cloud stream)))


;;;; Deep Boltzmann Machine

(defclass dbm (bm)
  ((layers
    :initarg :layers :type list :reader layers
    :documentation "A list of layers from bottom up. A layer is a list
of chunks. The layers partition the set of all chunks in the BM.
Chunks with no connections to layers below are visible (including
constant and conditioning) chunks. The layered structure is used in
the single, bottom-up approximate inference pass. When instantiating a
DBM, VISIBLE-CHUNKS and HIDDEN-CHUNKS are inferred from LAYERS and
CLOUDS.")
   (clouds-up-to-layers
    :type list :reader clouds-up-to-layers
    :documentation "Each element of this list is a list of clouds
connected from below to the layer of the same index."))
  (:documentation "A Deep Boltzmann Machine. See \"Deep Boltzmann
Machines\" by Ruslan Salakhutdinov and Geoffrey Hinton at
<http://www.cs.toronto.edu/~hinton/absps/dbm.pdf>.

To instantiate, set up LAYERS and CLOUDS but not VISIBLE-CHUNKS and
HIDDEN-CHUNKS, because contrary to how initialization works in the
superclass (BM), the values of these slots are inferred from LAYERS
and CLOUDS: chunks without a connection from below are visible while
the rest are hidden.

The default cloud spec list is computed by calling
FULL-CLOUDS-EVERYWHERE-BETWEEN-LAYERS on LAYERS."))

(defun full-clouds-everywhere-between-layers (layers)
  (loop for (layer1 layer2) on layers
        while layer2
        append (full-clouds-everywhere layer1 layer2)))

;;; See if CHUNK has a cloud among CLOUDS that connects it to any of
;;; CHUNKS.
(defun connects-to-p (chunk chunks clouds)
  (some (lambda (cloud)
          (if (typep cloud 'cloud)
              (or (and (eq chunk (chunk1 cloud))
                       (member (chunk2 cloud) chunks))
                  (and (eq chunk (chunk2 cloud))
                       (member (chunk1 cloud) chunks)))
              ;; Same thing for cloud specs.
              (let ((chunk1-name (getf cloud :chunk1))
                    (chunk2-name (getf cloud :chunk2))
                    (chunk-name (name chunk)))
                (or (and (eq chunk-name chunk1-name)
                         (member chunk2-name chunks
                                 :key #'name :test #'equal))
                    (and (eq chunk-name chunk2-name)
                         (member chunk1-name chunks
                                 :key #'name :test #'equal))))))
        clouds))

(defmethod initialize-instance :around ((dbm dbm) &rest initargs
                                        &key &allow-other-keys)
  ;; We need LAYERS and CLOUDS in order to infer visible/hidden
  ;; chunks, so compute clouds here.
  ;;
  ;; LAYERS might have an initform in a subclass or be passed as an
  ;; initarg. Call SHARED-INITIALIZE for the slots we are interested
  ;; in, so that they are initialized to whatever value takes
  ;; precedence.
  (apply #'shared-initialize dbm '(layers clouds visible-chunks hidden-chunks)
         initargs)
  (when (or (slot-boundp dbm 'visible-chunks)
            (slot-boundp dbm 'hidden-chunks))
    (error "Don't supply VISIBLE-CHUNKS and HIDDEN-CHUNKS for DBMs."))
  (let ((clouds (clouds dbm))
        (layers (layers dbm))
        (visible-chunks ())
        (hidden-chunks ()))
    ;; Merge clouds, at this point it may contain cloud specs or cloud
    ;; objects. Specs will be resolved in due time by the next method.
    (setq clouds
          (merge-cloud-specs clouds
                             (full-clouds-everywhere-between-layers layers)))
    ;; Infer VISIBLE-CHUNKS, HIDDEN-CHUNKS from LAYERS and CLOUDS.
    (dolist (layer (layers dbm))
      (let ((layer-visible-chunks ())
            (layer-hidden-chunks ()))
        (dolist (chunk layer)
          (if (or (connects-to-p chunk visible-chunks clouds)
                  (connects-to-p chunk hidden-chunks clouds))
              (push chunk layer-hidden-chunks)
              (push chunk layer-visible-chunks)))
        (setq visible-chunks (append visible-chunks
                                     (reverse layer-visible-chunks)))
        (setq hidden-chunks (append hidden-chunks
                                    (reverse layer-hidden-chunks)))))
    (apply #'call-next-method dbm
           :clouds clouds
           :visible-chunks visible-chunks
           :hidden-chunks hidden-chunks
           initargs)))

;;; Check that there are no clouds between non-adjacent layers. FIXME:
;;; should intralyer connections be allowed?
(defun check-dbm-clouds (dbm)
  (let ((bad-clouds (set-difference (clouds dbm)
                                    (apply #'append (clouds-up-to-layers dbm)))))
    (when bad-clouds
      (error "In ~A some clouds are between non-adjecent layers: ~A"
             dbm bad-clouds))))

(defmethod initialize-instance :after ((dbm dbm) &key &allow-other-keys)
  (setf (slot-value dbm 'clouds-up-to-layers)
        (loop for layer-below = () then layer
              for layer in (layers dbm)
              collect (remove-if-not
                       (lambda (cloud)
                         (cloud-between-chunks-p cloud layer-below layer))
                       (clouds dbm))))
  (check-dbm-clouds dbm))

(defun up-dbm (dbm)
  "Do a single upward pass in DBM, performing approximate inference.
Disregard intralayer and downward connections, double activations to
chunks having upward connections."
  (loop for (layer-below layer layer-above) on (layers dbm)
        for (clouds-up-to-layer clouds-up-to-layer-above)
        on (rest (clouds-up-to-layers dbm))
        while layer
        do (swap-nodes layer-below)
        (hijack-means-to-activation layer clouds-up-to-layer)
        ;; Double activations of chunks in LAYER that have connections
        ;; to LAYER-ABOVE.
        (dolist (chunk layer)
          (when (and (not (conditioning-chunk-p chunk))
                     (connects-to-p chunk layer-above
                                    clouds-up-to-layer-above))
            (matlisp:scal! #.(flt 2) (nodes chunk))))
        (map nil #'set-chunk-mean layer)
        (swap-nodes layer-below)))

(defun down-dbm (dbm)
  "Do a single downward pass in DBM, propagating the mean-field much
like performing approximate inference, but in the other direction.
Disregard intralayer and upward connections, double activations to
chunks having downward connections."
  (loop for (layer-above layer layer-below) on (reverse (layers dbm))
        for (clouds-down-to-layer clouds-down-to-layer-below)
        on (reverse (clouds-up-to-layers dbm))
        while layer
        do (swap-nodes layer-above)
        (hijack-means-to-activation layer clouds-down-to-layer)
        ;; Double activations of chunks in LAYER that have connections
        ;; to LAYER-BELOW.
        (dolist (chunk layer)
          (when (and (not (conditioning-chunk-p chunk))
                     (connects-to-p chunk layer-below
                                    clouds-down-to-layer-below))
            (matlisp:scal! #.(flt 2) (nodes chunk))))
        (map nil #'set-chunk-mean layer)
        (swap-nodes layer-above)))


;;;; DBM->DBN

(define-slots-not-to-be-copied 'dbm->dbn chunk
  nodes old-nodes inputs
  cache-static-activations-p static-activations-cached-p static-activations
  indices-present)

(defmethod copy-object-extra-initargs ((context (eql 'dbm->dbn)) (chunk chunk))
  `(:size ,(chunk-size chunk)
    :max-n-stripes ,(max-n-stripes chunk)))

(define-slots-not-to-be-copied 'dbm->dbn temporal-chunk
  next-node-inputs has-inputs-p)

(define-slots-to-be-shallow-copied 'dbm->dbn full-cloud
  weights)

(define-slots-not-to-be-copied 'dbm->dbn bm
  max-n-stripes)

(defun copy-dbm-chunk-to-dbn (chunk)
  (copy 'dbm->dbn chunk))

;;; C1 <-W-> C2: C1 * W -> C2, C1 <- W^T * C2
;;;
;;; C1 <-W-> C2 <-> C3: C1 * W * 2 -> C2, C1 <- W^T * C2
;;;
;;; C0 <-> C1 <-W-> C2: C1 * W -> C2, C1 <- 2 * W^T * C2
;;;
;;; C0 <-> C1 <-W-> C2 <-> C3: C1 * W * 2 -> C2, C1 <- 2 * W^T * C2
;;;
;;; In short, double activation from the cloud if the target chunk has
;;; input from another layer.
(defun copy-dbm-cloud-to-dbn (cloud clouds layer-below layer1 layer2 layer-above)
  (let ((chunk1 (chunk1 cloud))
        (chunk2 (chunk2 cloud))
        (copy (copy 'dbm->dbn cloud)))
    (when (and (member chunk1 layer1)
               (connects-to-p chunk1 layer-below clouds))
      (setf (slot-value copy 'scale1) (flt 2)))
    (when (and (member chunk2 layer2)
               (connects-to-p chunk2 layer-above clouds))
      (setf (slot-value copy 'scale2) (flt 2)))
    (when (and (member chunk2 layer1)
               (connects-to-p chunk2 layer-below clouds))
      (setf (slot-value copy 'scale2) (flt 2)))
    (when (and (member chunk1 layer2)
               (connects-to-p chunk1 layer-above clouds))
      (setf (slot-value copy 'scale1) (flt 2)))
    copy))

(defun dbm->dbn (dbm &key (rbm-class 'rbm) (dbn-class 'dbn)
                 dbn-initargs)
  "Convert DBM to a DBN by discarding intralayer connections and
doubling activations of clouds where necessary. If a chunk does not
have input from below then scale its input from above by 2; similarly,
if a chunk does not have input from above then scale its input from
below by 2. By default, weights are shared between clouds and their
copies.

For now, unrolling the resulting DBN to a BPN is not supported."
  (let* ((clouds (clouds dbm))
         (rbms (with-copying
                 (loop
                  for layer-below = nil then layer1
                  for (layer1 layer2 layer-above) on (layers dbm)
                  while layer2
                  collect
                  (flet ((copy-cloud (cloud)
                           (copy-dbm-cloud-to-dbn cloud clouds
                                                  layer-below
                                                  layer1 layer2
                                                  layer-above))
                         (cloud-between-layers-p (cloud)
                           (cloud-between-chunks-p
                            cloud layer1 layer2)))
                    (make-instance rbm-class
                                   :visible-chunks (mapcar
                                                    #'copy-dbm-chunk-to-dbn
                                                    layer1)
                                   :hidden-chunks (mapcar
                                                   #'copy-dbm-chunk-to-dbn
                                                   layer2)
                                   :clouds (mapcar #'copy-cloud
                                                   (remove-if-not
                                                    #'cloud-between-layers-p
                                                    clouds))))))))
    (apply #'make-instance dbn-class
           :rbms rbms
           dbn-initargs)))


;;;; Restricted Boltzmann Machine

(defclass rbm (bm)
  ((dbn :initform nil :type (or null dbn) :reader dbn))
  (:documentation "An RBM is a BM with no intralayer connections. An
RBM when trained with PCD behaves the same as a BM with the same
chunks, clouds but it can also be trained by contrastive
divergence (see RBM-CD-TRAINER) and stacked in a DBN."))

(defmethod initialize-instance :after ((rbm rbm) &key &allow-other-keys)
  (when (has-visible-to-visible-p rbm)
    (error "An RBM cannot have visible to visible connections."))
  (when (has-hidden-to-hidden-p rbm)
    (error "An RBM cannot have hidden to hidden connections.")))


;;;; Integration with gradient descent

;;; Base class for BM trainers with a positive and negative phase (CD
;;; and PCD).
(defclass segmented-gd-bm-trainer (segmented-gd-trainer) ())

(defgeneric accumulate-positive-phase-statistics (trainer bm &key multiplier)
  (:method ((trainer segmented-gd-bm-trainer) bm &key (multiplier (flt 1)))
    (do-clouds (cloud bm)
      (accumulate-cloud-statistics trainer cloud (flt (* -1 multiplier))))))

(defgeneric accumulate-negative-phase-statistics (trainer bm &key multiplier)
  (:method ((trainer segmented-gd-bm-trainer) bm &key (multiplier (flt 1)))
    (do-clouds (cloud bm)
      (accumulate-cloud-statistics trainer cloud (flt multiplier)))))

(defmethod train (sampler (trainer segmented-gd-bm-trainer) (bm bm))
  (while (not (finishedp sampler))
    (train-batch (sample-batch sampler (n-inputs-until-update trainer))
                 trainer bm)))

(defgeneric positive-phase (batch trainer bm))

(defgeneric negative-phase (batch trainer bm))


;;;; Common base class for MCMC based BM trainers

(defclass bm-mcmc-parameters ()
  ((visible-sampling
    :initform nil
    :initarg :visible-sampling
    :accessor visible-sampling
    :documentation "Controls whether visible nodes are sampled during
the learning or the mean field is used instead.")
   (hidden-sampling
    :initform :half-hearted
    :type (member nil :half-hearted t)
    :initarg :hidden-sampling
    :accessor hidden-sampling
    :documentation "Controls whether and how hidden nodes are sampled
during the learning or mean field is used instead. :HALF-HEARTED, the
default value, samples the hiddens but uses the hidden means to
calculate the effect of the positive phase on the gradient.")
   (n-gibbs
    :type (integer 1)
    :initform 1
    :initarg :n-gibbs
    :accessor n-gibbs
    :documentation "The number of steps of Gibbs sampling to perform."))
  (:documentation "Paramaters for Markov Chain Monte Carlo based
trainers for BMs."))


;;;; Contrastive Divergence (CD) learning for RBMs

(defclass rbm-cd-trainer (segmented-gd-bm-trainer bm-mcmc-parameters)
  ()
  (:documentation "A contrastive divergence based trainer for RBMs."))

(defmethod train-batch (batch (trainer rbm-cd-trainer) (rbm rbm))
  (loop for samples in (group batch (max-n-stripes rbm))
        do (set-input samples rbm)
        (positive-phase batch trainer rbm)
        (negative-phase batch trainer rbm))
  (maybe-update-weights trainer (length batch)))

(defmethod positive-phase (batch (trainer rbm-cd-trainer) (rbm rbm))
  (set-hidden-mean rbm)
  (ecase (hidden-sampling trainer)
    ((nil) (accumulate-positive-phase-statistics trainer rbm))
    ((:half-hearted)
     (accumulate-positive-phase-statistics trainer rbm)
     (sample-hidden rbm))
    ((t)
     (sample-hidden rbm)
     (accumulate-positive-phase-statistics trainer rbm))))

(defmethod negative-phase (batch (trainer rbm-cd-trainer) (rbm rbm))
  (let ((visible-sampling (visible-sampling trainer))
        (hidden-sampling (hidden-sampling trainer)))
    (loop for i below (n-gibbs trainer) do
          (when (and (not (zerop i)) hidden-sampling)
            (sample-hidden rbm))
          (set-visible-mean rbm)
          (when visible-sampling
            (sample-visible rbm))
          (set-hidden-mean rbm))
    (accumulate-negative-phase-statistics trainer rbm)))


;;;; Persistent Contrastive Divergence (PCD) learning

(define-slots-not-to-be-copied 'pcd chunk
  nodes old-nodes inputs
  cache-static-activations-p static-activations-cached-p static-activations
  indices-present)

(defmethod copy-object-extra-initargs ((context (eql 'pcd)) (chunk chunk))
  `(:size ,(chunk-size chunk)
    :max-n-stripes ,(max-n-stripes chunk)))

(define-slots-not-to-be-copied 'pcd temporal-chunk
  next-node-inputs has-inputs-p)

(define-slots-to-be-shallow-copied 'pcd full-cloud
  weights)

(define-slots-not-to-be-copied 'pcd bm
  max-n-stripes)

(define-slots-not-to-be-copied 'pcd dbm
  visible-chunks hidden-chunks)

(define-slots-to-be-shallow-copied 'pcd rbm
  dbn)

(defclass bm-pcd-trainer (segmented-gd-bm-trainer bm-mcmc-parameters)
  ((normal-chains
    :type bm
    :accessor normal-chains
    :documentation "The BM being trained.")
   (persistent-chains
    :type bm
    :accessor persistent-chains
    :documentation "A BM that keeps the states of the persistent
chains (each stripe is a chain), initialized from the BM being trained
by COPY with 'PCD as the context. Suitable for training BM and
RBM.")))

;;;; Damped mean field updates.

(defun settle-visible-mean-field (bm n-iterations damping-factor
                                  &key initial-pass-done-p)
  "Set visible means, multiply them with DAMPING-FACTOR (an FLT
between 0 and 1). Repeat this N-ITERATIONS times."
  (declare (type flt damping-factor))
  (let ((visible-chunks (remove-if #'conditioning-chunk-p
                                   (visible-chunks bm))))
    (unless initial-pass-done-p
      (zero-chunks visible-chunks)
      (set-visible-mean bm))
    (when (has-visible-to-visible-p bm)
      (loop for i below n-iterations do
            (set-visible-mean bm)
            (sum-nodes-and-old-nodes visible-chunks
                                     (flt damping-factor)
                                     (flt (- 1 damping-factor)))))))

(defun settle-hidden-mean-field (bm n-iterations damping-factor
                                 &key initial-pass-done-p)
  "Set hidden means, multiply them with DAMPING-FACTOR (an FLT
between 0 and 1). Repeat this N-ITERATIONS times."
  (declare (type flt damping-factor))
  (let ((hidden-chunks (remove-if #'conditioning-chunk-p
                                  (hidden-chunks bm))))
    (unless initial-pass-done-p
      (zero-chunks hidden-chunks)
      (set-hidden-mean bm))
    (when (has-hidden-to-hidden-p bm)
      (loop for i below n-iterations do
            (set-hidden-mean bm)
            (sum-nodes-and-old-nodes hidden-chunks
                                     (flt damping-factor)
                                     (flt (- 1 damping-factor)))))))

(defgeneric set-hidden-mean-in-pcd (trainer bm)
  (:documentation "Called in the positive phase during PCD training.
For an RBM it trivially calls SET-HIDDEN-MEAN, while for a BM it calls
SETTLE-HIDDEN-MEAN-FIELD with 10 iterations and 0.1 as the damping
factor. To change the parameters (or how the mean field is computed),
write a specialized method.")
  (:method ((trainer bm-pcd-trainer) (bm bm))
    (settle-hidden-mean-field bm 10 (flt 0.1)))
  (:method ((trainer bm-pcd-trainer) (dbm dbm))
    (up-dbm dbm)
    (settle-hidden-mean-field dbm 10 (flt 0.1) :initial-pass-done-p t))
  (:method ((trainer bm-pcd-trainer) (rbm rbm))
    (set-hidden-mean rbm)))

(defmethod initialize-trainer ((trainer bm-pcd-trainer) (bm bm))
  (setf (normal-chains trainer) bm)
  (setf (persistent-chains trainer) (copy 'pcd bm))
  (call-next-method))

(defmethod train-batch (batch (trainer bm-pcd-trainer) (bm bm))
  (loop for samples in (group batch (max-n-stripes bm))
        do (set-input samples bm)
        (positive-phase batch trainer bm))
  (negative-phase batch trainer (persistent-chains trainer))
  (maybe-update-weights trainer (length batch)))

;;; If CLOUD is in the persistent chain, that is, it's a copy of a
;;; cloud in the normal BM then use the orignal as that's what the
;;; SEGMENTED-GD-BM-TRAINER was initialized with (and they, of course,
;;; share the weights).
(defmethod find-segment-gradient-accumulator (cloud (trainer bm-pcd-trainer))
  (if (find cloud (clouds (persistent-chains trainer)))
      (find-segment-gradient-accumulator (find-cloud (name cloud)
                                                     (normal-chains trainer))
                                         trainer)
      (call-next-method)))

(defmethod positive-phase (batch (trainer bm-pcd-trainer) (bm bm))
  (set-hidden-mean-in-pcd trainer bm)
  (when (eq t (hidden-sampling trainer))
    (sample-hidden bm))
  (accumulate-positive-phase-statistics trainer bm))

(defmethod negative-phase (batch (trainer bm-pcd-trainer) bm)
  (let ((visible-sampling (visible-sampling trainer))
        (hidden-sampling (hidden-sampling trainer)))
    (loop repeat (n-gibbs trainer) do
          (when hidden-sampling
            (sample-hidden bm))
          (set-visible-mean bm)
          (when visible-sampling
            (sample-visible bm))
          (set-hidden-mean bm)))
  (accumulate-negative-phase-statistics
   trainer bm
   ;; The number of persistent chains (or fantasy particles), that is,
   ;; N-STRIPES of PERSISTENT-CHAINS is not necessarily the same as
   ;; the batch size. Normalize so that positive and negative phase
   ;; has the same weight.
   :multiplier (/ (length batch)
                  (n-stripes (persistent-chains trainer)))))


;;;; Convenience, utilities

(defun inputs->nodes (bm)
  "Copy the previously clamped INPUTS to NODES as if SET-INPUT were
called with the same parameters."
  (map nil (lambda (chunk)
             (let ((inputs (inputs chunk)))
               (when inputs
                 (copy-chunk-nodes chunk inputs (nodes chunk)))))
       (visible-chunks bm)))

(defun nodes->inputs (bm)
  "Copy NODES to INPUTS."
  (map nil (lambda (chunk)
             (let ((inputs (inputs chunk)))
               (when inputs
                 (copy-chunk-nodes chunk (nodes chunk) inputs))))
       (visible-chunks bm)))

(defun reconstruction-error (bm)
  "Return the squared norm of INPUTS - NODES not considering constant
or conditioning chunks that aren't reconstructed in any case. The
second value returned is the number of nodes that contributed to the
error."
  (let ((sum #.(flt 0))
        (n 0))
    (declare (type flt sum) (type index n) (optimize (speed 3)))
    (dolist (chunk (visible-chunks bm))
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
