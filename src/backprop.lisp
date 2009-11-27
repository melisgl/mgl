;;;; Backpropagation for feed-forward networks.
;;;;
;;;; There is no constraint on the network topology except that it
;;;; must be acyclic. Networks can be defined in a compact form by
;;;; describing `lumps' of nodes. All nodes in a lump have the same
;;;; transfer function.
;;;;
;;;; It turns out that on sizable problems, such as the mnist one,
;;;; performance is memory bound. Not being able to do batch
;;;; processing made v0.0.1 very slow. To allow the memory/cache aware
;;;; BLAS 3 routines work their magic, ``stripes'' were added.

(in-package :mgl-bp)

;;;; Lump

(defvar *bpn-being-built* nil)
(defvar *next-lump-name* nil)

(defun next-lump-name ()
  (prog1 (or *next-lump-name* (gensym))
    (setf *next-lump-name* nil)))

(defgeneric default-size (lump)
  (:method (lump)
    (or (slot-boundp lump 'size)
        (error "Can't compute size for ~S." lump))))

(defclass lump ()
  ((name :initform (next-lump-name) :type symbol :initarg :name :reader name)
   (size :type index :initarg :size :reader size)
   (n-stripes :initform 1 :type index :reader n-stripes)
   (max-n-stripes
    :initform 1 :type index :initarg :max-n-stripes
    :reader max-n-stripes)
   (same-stripes-p
    :initform nil :reader same-stripes-p
    :documentation "Non-NIL iff all stripes are the same. If true, it
effectively overrides both N-STRIPES and MAX-N-STRIPES and there is
only one column in NODES and DERIVATIVES. Set up by the lump itself
taking its inputs into account. Notably, WEIGHT-LUMPS always have
SAME-STRIPES-P T.")
   (nodes
    :initform nil :type (or matlisp:real-matrix null) :reader nodes
    :documentation "The values of the nodes. All nodes have values. It
is a SIZE x N-STRIPES matrix that can be enlarged to SIZE x
MAX-N-STRIPES by setting N-STRIPES.")
   (derivatives
    :initform nil :type (or matlisp:real-matrix null) :reader derivatives
    :documentation "Derivatives of nodes, input node derivatives are
not calculated. A matrix of the same dimension as NODES.")
   (default-value
       :initform #.(flt 0) :initarg :default-value :type flt
       :reader default-value
       :documentation "Upon creation or resize the lump's nodes get
filled with this value.")
   (indices-to-calculate
    :initform nil :initarg :indices-to-calculate :type (or null index-vector)
    :accessor indices-to-calculate
    :documentation "NIL or a simple vector of array indices into this
lump's range (i.e. in the 0 (1- SIZE) interval). Need not be ordered.
If not NIL the node's value is not calculated and its derivatives are
not propagated unless it is in INDICES-TO-CALCULATE. It has no effect
subsequent lumps: they may use values that have not been recalculated.
The primary use-case is to temporarily mask out an uninteresting part
of the network.")))

(defun limit-stripes (lump n)
  (if (same-stripes-p lump)
      (min 1 n)
      n))

;;; The effective number of stripes.
(defun n-stripes* (lump)
  (limit-stripes lump (n-stripes lump)))

;;; The effective maximum number of stripes, i.e. the number of
;;; allocated columns of NODES and DERIVATIVES.
(defun max-n-stripes* (lump)
  (limit-stripes lump (max-n-stripes lump)))

;;; The maximum number of columns for which MAT has storage allocated.
(defun max-ncols (mat)
  (the index (/ (length (storage mat))
                (matlisp:nrows mat))))

(defmethod print-object ((lump lump) stream)
  (pprint-logical-block (stream ())
    (print-unreadable-object (lump stream :type t)
      (format stream "~S ~:_~S ~:_~S" (name lump) :size
              (if (slot-boundp lump 'size)
                  (size lump)
                  ':unbound))
      (format stream "*~S(~S/~S)" (n-stripes* lump) (n-stripes lump)
              (max-n-stripes lump))
      (when (indices-to-calculate lump)
        (format stream " ~:_(~S)" (length (indices-to-calculate lump))))))
  lump)

(defmethod set-n-stripes (n-stripes (lump lump))
  (assert (<= 0 n-stripes (max-n-stripes lump)))
  (setf (slot-value lump 'n-stripes) n-stripes)
  (set-ncols (nodes lump) (n-stripes* lump))
  (set-ncols (derivatives lump) (n-stripes* lump))
  n-stripes)

(defmethod set-max-n-stripes (max-n-stripes (lump lump))
  (let ((old-max-n-stripes* (max-n-stripes* lump))
        (size (size lump)))
    (setf (slot-value lump 'max-n-stripes) max-n-stripes
          (slot-value lump 'n-stripes) (min (n-stripes lump) max-n-stripes))
    (let ((max-n-stripes* (max-n-stripes* lump)))
      (cond ((zerop max-n-stripes*)
             (when (nodes lump)
               (setf (slot-value lump 'nodes) nil
                     (slot-value lump 'derivatives) nil)))
            ((or (/= max-n-stripes* old-max-n-stripes*)
                 (null (nodes lump)))
             (setf (slot-value lump 'nodes)
                   (matlisp:make-real-matrix size max-n-stripes*))
             (matlisp:fill-matrix (nodes lump) (default-value lump))
             (setf (slot-value lump 'derivatives)
                   (matlisp:make-real-matrix size max-n-stripes*))
             ;; make sure the number of column is set properly
             (set-n-stripes (n-stripes lump) lump)))))
  max-n-stripes)

(defmethod initialize-instance :after ((lump lump) &key &allow-other-keys)
  (unless (slot-boundp lump 'size)
    (setf (slot-value lump 'size) (default-size lump)))
  ;; ensure that the matrices are allocated
  (setf (max-n-stripes lump) (max-n-stripes lump))
  (when *bpn-being-built*
    (add-lump lump *bpn-being-built*)))

(defmethod stripe-start (stripe (lump lump))
  (assert (<= 0 stripe (1- (n-stripes lump))))
  (* (if (same-stripes-p lump)
         0
         stripe)
     (size lump)))

(defmethod stripe-end (stripe (lump lump))
  (+ (stripe-start stripe lump)
     (size lump)))

(defgeneric transfer-lump (lump))
(defgeneric derive-lump (lump))


;;;; Data lumps

(defclass data-lump (lump) ())
(defclass input-lump (data-lump) ())
(defclass weight-lump (data-lump)
  ((same-stripes-p :initform t)))
(defclass constant-lump (data-lump)
  ((default-value :initform #.(flt 1))))

(defmethod transfer-lump ((lump data-lump)))

(defmethod derive-lump ((lump data-lump)))


;;;; BPN

(defclass bpn ()
  ((lumps
    :initform (make-array 0 :element-type 'lump :adjustable t :fill-pointer t)
    :type (array lump (*)) :reader lumps
    :documentation "Lumps in reverse order")
   (max-n-stripes
    :initform 1 :type index :initarg :max-n-stripes
    :reader max-n-stripes)))

(defmethod print-object ((bpn bpn) stream)
  (pprint-logical-block (stream ())
    (print-unreadable-object (bpn stream :type t :identity t)
      (unless (zerop (length (lumps bpn)))
        (format stream "~S ~:_~S ~:_~S ~:_~S" :n-stripes (n-stripes bpn)
                :max-n-stripes (max-n-stripes bpn)))))
  bpn)

(define-descriptions (bpn bpn)
  lumps n-stripes max-n-stripes)

(defmethod n-stripes ((bpn bpn))
  (n-stripes (aref (lumps bpn) 0)))

(defmethod set-n-stripes (n-stripes (bpn bpn))
  (loop for lump across (lumps bpn)
        do (setf (n-stripes lump) n-stripes)))

(defmethod set-max-n-stripes (max-n-stripes (bpn bpn))
  (setf (slot-value bpn 'max-n-stripes) max-n-stripes)
  (loop for lump across (lumps bpn)
        do (setf (max-n-stripes lump) max-n-stripes)))

(defmethod initialize-instance :after ((bpn bpn) &key &allow-other-keys)
  ;; make sure lumps have the same MAX-N-STRIPES
  (setf (max-n-stripes bpn) (max-n-stripes bpn)))

(defun find-lump (name bpn &key errorp)
  (or (find name (lumps bpn) :key #'name :test #'equal)
      (if errorp
          (error "Cannot find lump ~S." name)
          nil)))

(defmethod set-input :around (samples (bpn bpn))
  (setf (n-stripes bpn) (length samples))
  (call-next-method))

(defun add-lump (lump bpn)
  "Add LUMP to BPN. MAX-N-STRIPES of LUMP gets set to equal that of
the previous last, non-weight lump of BPN."
  (when (find-lump (name lump) bpn)
    (error "Cannot add ~S: ~%
            a lump of same name has already been added to this network." lump))
  (setf (max-n-stripes lump) (max-n-stripes bpn))
  (vector-push-extend lump (slot-value bpn 'lumps))
  lump)

(defmacro build-bpn ((&key (class ''bpn) initargs
                           (max-n-stripes 1)) &body lumps)
  (let ((bindings
         (mapcar (lambda (lump)
                   (destructuring-bind (symbol init-form) lump
                     `(,symbol (let ((*next-lump-name* ',symbol))
                                 (make-instance ',(first init-form)
                                                ,@(rest init-form))))))
                 lumps)))
    `(let ((*bpn-being-built* (apply #'make-instance ,class
                                     :max-n-stripes ,max-n-stripes
                                     ,initargs)))
       (flet ((lump (name)
                (find-lump name *bpn-being-built* :errorp t)))
         (let* ,bindings
           (declare (ignorable ,@(mapcar #'first bindings)))
           ;; prevent warning if LUMP goes unused
           #'lump))
       *bpn-being-built*)))

(defun ->lump (bpn lump-spec)
  (if (typep lump-spec 'lump)
      lump-spec
      (find-lump lump-spec bpn :errorp t)))

(defun forward-bpn (bpn &key from-lump to-lump)
  "Propagate the values from the already clamped inputs."
  (declare (optimize (debug 2)))
  (let ((from-lump (if from-lump (->lump bpn from-lump) nil))
        (to-lump (if to-lump (->lump bpn to-lump) nil))
        (seen-from-lump-p (not from-lump)))
    (loop for lump across (lumps bpn)
          do (when (eq lump from-lump) (setq seen-from-lump-p t))
          do (when seen-from-lump-p (transfer-lump lump))
          until (eq lump to-lump))))

;;; Derivatives of weights are left alone to let them accumulate which
;;; is useful in batches such as when training by conjugate gradient.
(defun zero-non-weight-derivatives (bpn &key (last-lump nil last-lump-p))
  (let ((lumps (lumps bpn))
        (last-lump (if last-lump-p (->lump bpn last-lump) nil)))
    (loop for i downfrom (1- (length lumps)) downto 0
          for lump = (aref lumps i)
          until (and last-lump-p (eq last-lump lump))
          when (not (typep lump 'weight-lump))
          do (matlisp:fill-matrix (derivatives lump) (flt 0)))))

(defun backward-bpn (bpn &key (last-lump nil last-lump-p))
  "Accumulate derivatives of weights."
  (let* ((lumps (lumps bpn))
         (last-lump (if last-lump-p (->lump bpn last-lump) nil)))
    (apply #'zero-non-weight-derivatives bpn
           (if last-lump-p
               (list :last-lump last-lump)
               nil))
    (loop for i downfrom (1- (length lumps)) downto 0
          for lump = (aref lumps i)
          until (and last-lump-p (eq last-lump lump))
          do (derive-lump lump))))


;;;; Train

(defclass base-bp-trainer ()
  ((first-trained-lump :reader first-trained-lump)))

(defmethod initialize-trainer ((trainer base-bp-trainer) segmentable)
  (setf (slot-value trainer 'first-trained-lump) nil)
  (call-next-method))

(defmethod map-segments (fn (bpn bpn))
  (map nil
       (lambda (lump)
         (when (typep lump 'weight-lump)
           (funcall fn lump)))
       (lumps bpn)))

(defmethod segment-weights ((lump lump))
  (let ((nodes (nodes lump)))
    (values (storage nodes) 0 (matlisp:number-of-elements nodes))))

(defun first-trained-weight-lump (trainer bpn)
  "Much time can be wasted computing derivatives of non-trained weight
lumps. Return the first one that TRAINER trains."
  (or (slot-value trainer 'first-trained-lump)
      (setf (slot-value trainer 'first-trained-lump)
            (find-if (lambda (lump)
                       (member lump (segments trainer)))
                     (lumps bpn)))))

(defmethod cost (bpn)
  "Return the sum of costs for all active stripes. The cost of a
stripe is the IMPORTANCE weighted sum of the error nodes. The second
value is the number of stripes."
  (let ((sum (flt 0)))
    (loop for lump across (lumps bpn) do
          (when (typep lump 'error-node)
            (let ((nodes (nodes lump)))
              (incf sum (* (importance lump)
                           (if (same-stripes-p lump)
                               (n-stripes lump)
                               1)
                           (sum-elements nodes))))))
    (values sum (n-stripes bpn))))

(defgeneric compute-derivatives (samples trainer bpn)
  (:method (samples trainer bpn)
    (set-input samples bpn)
    (forward-bpn bpn)
    (backward-bpn bpn :last-lump (first-trained-weight-lump trainer bpn))))


;;;; CG trainer

(defclass cg-bp-trainer (base-bp-trainer cg-trainer) ())

(defun segment-set-derivatives->weights (segment-set weights)
  (declare (type flt-vector weights)
           (optimize (speed 3)))
  (do-segment-set (lump :start-in-segment-set start-in-segment-set) segment-set
    (replace weights (storage (derivatives lump))
             :start1 start-in-segment-set
             :start2 0 :end2 (matlisp:number-of-elements (derivatives lump)))))

(defmethod compute-batch-cost-and-derive (samples trainer (bpn bpn))
  (let ((cost #.(flt 0)))
    (do-segment-set (lump) (segment-set trainer)
      (let ((derivatives (derivatives lump)))
        (matlisp:fill-matrix derivatives (flt 0))))
    (loop for batch in (group samples (max-n-stripes bpn))
          do (compute-derivatives batch trainer bpn)
          (incf cost (cost bpn)))
    ;; By now the weight derivatives have accumulated, see
    ;; ZERO-NON-WEIGHT-DERIVATIVES.
    (segment-set-derivatives->weights (segment-set trainer)
                                      (accumulator trainer))
    cost))


;;;; Gradient descent trainer

(defclass bp-trainer (base-bp-trainer segmented-gd-trainer) ())

(defun add-and-forget-derivatives (trainer bpn)
  ;; It would be as easy as:
  ;;
  ;;  (let ((ones (matlisp:ones (n-stripes bpn) 1)))
  ;;    (do-segment-gradient-accumulators ((lump acc-start accumulator) trainer)
  ;;      (matlisp:gemm (flt 1) (derivatives lump) ones (flt 1) accumulator)
  ;;      (matlisp:fill-matrix (derivatives lump) (flt 0))))
  ;;
  ;; save for the one-to-many segment-trainer and segment run
  ;; complications.
  (do-segment-gradient-accumulators ((lump acc-start accumulator) trainer)
    (let ((accumulator* (storage accumulator))
          (derivatives* (storage (derivatives lump))))
      (map-segment-runs
       (lambda (start1 end1)
         (declare (type index start1 end1))
         (assert (<= (+ acc-start end1) (length accumulator*)))
         (assert (typep lump 'weight-lump))
         (with-stripes ((0 lump ls le))
           (let ((a (+ ls start1))
                 (b (+ ls end1))
                 (c (+ acc-start start1)))
             (declare (type index a b c)
                      (optimize (speed 3) #.*no-array-bounds-check*))
             (assert (< a le))
             (assert (<= b le))
             (loop for i of-type index upfrom a below b
                   for j of-type index upfrom c
                   do (incf (aref accumulator* j) (aref derivatives* i))))))
       lump)))
  ;; All weight derivatives must be zeroed, even the ones not being
  ;; trained on to avoid overflows.
  (loop for lump across (lumps bpn)
        do (when (typep lump 'weight-lump)
             (matlisp:fill-matrix (derivatives lump) (flt 0)))))

(defmethod train (sampler (trainer bp-trainer) (bpn bpn))
  (while (not (finishedp sampler))
    (train-batch (sample-batch sampler (n-inputs-until-update trainer))
                 trainer bpn)))

(defmethod train-batch (batch (trainer bp-trainer) (bpn bpn))
  (loop for samples in (group batch (max-n-stripes bpn))
        do (compute-derivatives samples trainer bpn))
  (add-and-forget-derivatives trainer bpn)
  (maybe-update-weights trainer (length batch)))


;;;; I/O

(defmethod write-weights ((lump weight-lump) stream)
  (write-double-float-vector (storage (nodes lump)) stream))

(defmethod read-weights ((lump weight-lump) stream)
  (read-double-float-vector (storage (nodes lump)) stream))

(defmethod write-weights ((bpn bpn) stream)
  (map-segments (lambda (weights)
                  (write-weights weights stream))
                bpn))

(defmethod read-weights ((bpn bpn) stream)
  (map-segments (lambda (weights)
                  (read-weights weights stream))
                bpn))


;;;; NORMALIZED-LUMP

(defclass normalized-lump (lump)
  ((x :initarg :x :reader x :documentation "Input comes from here.")
   (group-size :initarg :group-size :reader group-size)
   (scale
    :initform #.(flt 1) :type (or flt flt-vector)
    :initarg :scale :accessor scale
    :documentation "The sum of nodes after normalization. Can be
changed during training, for instance when clamping. If it is a vector
then its length must be MAX-N-STRIPES which automatically
maintained.")))

(defmethod default-size ((lump normalized-lump))
  (size (x lump)))

(defmethod set-max-n-stripes (max-n-stripes (lump normalized-lump))
  (call-next-method)
  (when (and (typep (scale lump) 'flt-vector)
             (/= (max-n-stripes lump) (length (scale lump))))
    (setf (scale lump) (make-flt-array (max-n-stripes lump))))
  max-n-stripes)

(defmethod transfer-lump ((lump normalized-lump))
  (let* ((x (x lump))
         (group-size (group-size lump))
         (scale (scale lump))
         (x* (storage (nodes x)))
         (to* (storage (nodes lump))))
    (declare (type index group-size)
             (type (or flt flt-vector) scale))
    (assert (= (size lump) (size x)))
    (assert (= (n-stripes lump) (n-stripes x)))
    (loop for stripe of-type index below (n-stripes* lump) do
          (let ((scale (if (typep scale 'flt) scale (aref scale stripe))))
            (with-stripes ((stripe lump ls le)
                           (stripe x xs xe))
              (loop for li upfrom ls below le
                    for xi upfrom xs below xe
                    for i upfrom 0
                    do
                    (when (zerop (mod i group-size))
                      (let ((sum #.(flt 0)))
                        (declare (type flt sum)
                                 (optimize (speed 3)))
                        (loop for j upfrom xi below (+ xi group-size)
                              do (incf sum (aref x* j)))
                        (setq sum (/ sum scale))
                        (loop for xj upfrom xi below (+ xi group-size)
                              for lj upfrom li below (+ li group-size)
                              do (setf (aref to* lj)
                                       (/ (aref x* xj) sum)))))))))))

(defmethod derive-lump ((lump normalized-lump))
  (let* ((x (x lump))
         (group-size (group-size lump))
         (scale (scale lump))
         (x* (storage (nodes x)))
         (xd* (storage (derivatives x)))
         (ld* (storage (derivatives lump))))
    (declare (type index group-size)
             (type (or flt flt-vector) scale))
    (assert (= (size lump) (size x)))
    (assert (= (n-stripes lump) (n-stripes x)))
    (loop for stripe of-type index below (n-stripes* lump) do
          (let ((scale (if (typep scale 'flt) scale (aref scale stripe))))
            (with-stripes ((stripe lump ls le)
                           (stripe x xs xe))
              (loop for li of-type index upfrom ls below le by group-size
                    for xi of-type index upfrom xs below xe by group-size
                    do
                    (let ((sum #.(flt 0))
                          (lie (+ li group-size))
                          (xie (+ xi group-size)))
                      (declare (type flt sum)
                               (optimize (speed 3)))
                      (loop for xj upfrom xi below xie
                            do (incf sum (aref x* xj)))
                      (let ((sum-square (expt sum 2)))
                        ;; derive by xj
                        (loop for xj of-type index upfrom xi below xie do
                              (loop for lk upfrom li below lie
                                    for xk upfrom xi below xie do
                                    (if (= xk xj)
                                        (incf (aref xd* xj)
                                              (* (aref ld* lk)
                                                 scale
                                                 (/ (- sum (aref x* xj))
                                                    sum-square)))
                                        (decf (aref xd* xj)
                                              (* (aref ld* lk)
                                                 scale
                                                 (/ (aref x* xk)
                                                    sum-square))))))))))))))


;;;; Activation lump

(defclass activation-lump (lump)
  ((weights :type weight-lump :initarg :weights :reader weights)
   (x :initarg :x :reader x :documentation "Input comes from here.")
   (transpose-weights-p :initform nil :initarg :transpose-weights-p
                        :reader transpose-weights-p))
  (:documentation "Perform WEIGHTS*X where X is of size N and WEIGHTS
is a WEIGHT-LUMP whose single stripe is taken to be of dimensions M x
N stored in column major order. M is the size of this lump. If
TRANSPOSE-WEIGHTS-P then WEIGHTS is N x M and WEIGHTS'*X is
computed."))

(defmethod default-size ((lump activation-lump))
  (/ (size (weights lump))
     (size (x lump))))

(defmethod transfer-lump ((lump activation-lump))
  (let* ((x (x lump))
         (weights (weights lump))
         (nx (size x))
         (nl (/ (size weights)
                nx)))
    ;; FIXME:
    (assert (null (same-stripes-p x)))
    (if (transpose-weights-p lump)
        (matlisp:gemm! (flt 1) (reshape2 (nodes weights) nx nl) (nodes x)
                       (flt 0) (nodes lump) :tn)
        (matlisp:gemm! (flt 1) (reshape2 (nodes weights) nl nx) (nodes x)
                       (flt 0) (nodes lump)))))

(defmethod derive-lump ((lump activation-lump))
  (let* ((x (x lump))
         (weights (weights lump))
         (nx (size x))
         (nl (/ (size weights)
                nx)))
    (if (transpose-weights-p lump)
        ;; dx += w*a
        (matlisp:gemm! (flt 1) (reshape2 (nodes weights) nx nl)
                       (derivatives lump)
                       (flt 1) (derivatives x))
        ;; dx += w'*a
        (matlisp:gemm! (flt 1) (reshape2 (nodes weights) nl nx)
                       (derivatives lump)
                       (flt 1) (derivatives x) :tn))
    (if (transpose-weights-p lump)
        ;; dw += x*a'
        (matlisp:gemm! (flt 1) (nodes x) (derivatives lump)
                       (flt 1) (reshape2 (derivatives weights) nx nl) :nt)
        ;; dw += a*x'
        (matlisp:gemm! (flt 1) (derivatives lump) (nodes x)
                       (flt 1) (reshape2 (derivatives weights) nl nx) :nt))))


;;;; Node type library

(defclass ->+ (lump)
  ((args :initarg :args :reader args)))

(defmethod default-size ((lump ->+))
  (size (first (args lump))))

(defmethod transfer-lump ((lump ->+))
  (let* ((to (nodes lump))
         (n-stripes* (n-stripes* lump))
         (ones (matlisp:ones 1 n-stripes*)))
    (matlisp:fill-matrix to (flt 0))
    (dolist (arg (args lump))
      (cond ((= n-stripes* (n-stripes* arg))
             (matlisp:m+! (nodes arg) to))
            (t
             (assert (same-stripes-p arg))
             (matlisp:gemm! (flt 1) (nodes arg) ones (flt 1) to))))))

(defmethod derive-lump ((lump ->+))
  (let* ((derivatives (derivatives lump))
         (n-stripes* (n-stripes* lump))
         (ones (matlisp:ones n-stripes* 1)))
    (dolist (arg (args lump))
      (cond ((= n-stripes* (n-stripes* arg))
             (matlisp:m+! (derivatives lump) (derivatives arg)))
            (t
             (assert (same-stripes-p arg))
             (matlisp:gemm! (flt 1) derivatives ones
                            (flt 1) (derivatives arg)))))))

(defclass ->sum (lump)
  ((x :initarg :x :reader x))
  (:documentation "Sum of all nodes \(per stripe)."))

(defmethod default-size ((lump ->sum))
  1)

(defmethod transfer-lump ((lump ->sum))
  (let ((x (x lump)))
    (assert (= (n-stripes lump) (n-stripes x)))
    (let ((x* (storage (nodes x)))
          (to* (storage (nodes lump))))
      (declare (optimize (speed 3) #.*no-array-bounds-check*))
      (loop for stripe of-type index below (n-stripes* lump) do
            (with-stripes ((stripe x xs xe))
              (setf (aref to* stripe)
                    (let ((sum (flt 0)))
                      (declare (type flt sum))
                      (loop for xi upfrom xs below xe
                            do (incf sum (aref x* xi)))
                      sum)))))))

(defmethod derive-lump ((lump ->sum))
  (let ((x (x lump)))
    (assert (= (n-stripes lump) (n-stripes x)))
    (let ((xd* (storage (derivatives x)))
          (derivatives* (storage (derivatives lump))))
      (declare (optimize (speed 3) #.*no-array-bounds-check*))
      (loop for stripe of-type index below (n-stripes* lump) do
            (let ((d (aref derivatives* stripe)))
              (with-stripes ((stripe x xs xe))
                (loop for xi upfrom xs below xe
                      do (incf (aref xd* xi) d))))))))

(defclass ->linear (lump)
  ((x :initarg :x :reader x)
   (y :initarg :y :reader y)))

(defmethod default-size ((lump ->linear))
  1)

(defmethod transfer-lump ((lump ->linear))
  (let ((x (x lump))
        (y (y lump)))
    (assert (= (size lump) (size x) (size y)))
    (assert (= (n-stripes lump) (n-stripes x) (n-stripes y)))
    (let ((x* (storage (nodes x)))
          (y* (storage (nodes y)))
          (to* (storage (nodes lump))))
      (declare (optimize (speed 3) #.*no-array-bounds-check*))
      (loop for stripe of-type index below (n-stripes* lump) do
            (with-stripes ((stripe x xs xe)
                           (stripe y ys ye))
              (setf (aref to* stripe)
                    (let ((sum (flt 0)))
                      (declare (type flt sum))
                      (loop for xi upfrom xs below xe
                            for yi upfrom ys below ye
                            do (incf sum (* (aref x* xi)
                                            (aref y* yi))))
                      sum)))))))

(defmethod derive-lump ((lump ->linear))
  (let ((x (x lump))
        (y (y lump)))
    (assert (= (size lump) (size x) (size y)))
    (assert (= (n-stripes lump) (n-stripes x) (n-stripes y)))
    (let ((x* (storage (nodes x)))
          (xd* (storage (derivatives x)))
          (y* (storage (nodes y)))
          (yd* (storage (derivatives y)))
          (derivatives* (storage (derivatives lump))))
      (declare (optimize (speed 3) #.*no-array-bounds-check*))
      (loop for stripe of-type index below (n-stripes* lump) do
            (let ((d (aref derivatives* stripe)))
              (with-stripes ((stripe x xs xe)
                             (stripe y ys ye))
                (loop for xi upfrom xs below xe
                      for yi upfrom ys below ye
                      do
                      (incf (aref xd* xi)
                            (* d (aref y* yi)))
                      (incf (aref yd* yi)
                            (* d (aref x* xi))))))))))

(defclass ->sigmoid (lump)
  ((x :initarg :x :reader x)))

(defmethod default-size ((lump ->sigmoid))
  (size (x lump)))

(defmethod transfer-lump ((lump ->sigmoid))
  (let ((x (x lump)))
    (assert (= (size lump) (size x)))
    (let ((x* (storage (nodes x)))
          (l* (storage (nodes lump))))
      (declare (optimize (speed 3) #.*no-array-bounds-check*))
      (loop for stripe of-type index below (n-stripes* lump) do
            (with-stripes ((stripe lump ls le)
                           (stripe x xs xe))
              (loop for li upfrom ls below le
                    for xi upfrom xs below xe
                    do (setf (aref l* li) (sigmoid (aref x* xi)))))))))

(defmethod derive-lump ((lump ->sigmoid))
  (let ((x (x lump)))
    (assert (= (size lump) (size x)))
    (let ((xd* (storage (derivatives x)))
          (l* (storage (nodes lump)))
          (ld* (storage (derivatives lump))))
      (declare (optimize (speed 3) #.*no-array-bounds-check*))
      (loop for stripe of-type index below (n-stripes* lump) do
            (with-stripes ((stripe lump ls le)
                           (stripe x xs xe))
              (loop for li upfrom ls below le
                    for xi upfrom xs below xe
                    do (incf (aref xd* li)
                             (let ((s (aref l* li)))
                               (* (aref ld* li)
                                  s (- 1 s))))))))))

(defclass ->exp (lump)
  ((x :initarg :x :reader x)))

(defmethod default-size ((lump ->exp))
  (size (x lump)))

(defmethod transfer-lump ((lump ->exp))
  (let ((x (x lump)))
    (assert (= (size lump) (size x)))
    (let ((x* (storage (nodes x)))
          (l* (storage (nodes lump))))
      (declare (optimize (speed 3) #.*no-array-bounds-check*))
      (loop for stripe of-type index below (n-stripes* lump) do
            (with-stripes ((stripe lump ls le)
                           (stripe x xs xe))
              (loop for li upfrom ls below le
                    for xi upfrom xs below xe
                    do (setf (aref l* li) (exp (aref x* xi)))))))))

(defmethod derive-lump ((lump ->exp))
  (let ((x (x lump)))
    (assert (= (size lump) (size x)))
    (let ((xd* (storage (derivatives x)))
          (l* (storage (nodes lump)))
          (ld* (storage (derivatives lump))))
      (declare (optimize (speed 3) #.*no-array-bounds-check*))
      (loop for stripe of-type index below (n-stripes* lump) do
            (with-stripes ((stripe lump ls le)
                           (stripe x xs xe))
              (loop for li upfrom ls below le
                    for xi upfrom xs below xe
                    do (incf (aref xd* li) (* (aref ld* li)
                                              (aref l* li)))))))))

(defclass ->sum-squared-error (lump)
  ((x :initarg :x :reader x)
   (y :initarg :y :reader y)))

(defmethod default-size ((lump ->sum-squared-error))
  1)

(defmethod transfer-lump ((lump ->sum-squared-error))
  (let ((x (x lump))
        (y (y lump)))
    (assert (= (size x) (size y)))
    (assert (= (n-stripes lump) (n-stripes x) (n-stripes y)))
    (let ((x* (storage (nodes x)))
          (y* (storage (nodes y)))
          (to* (storage (nodes lump))))
      (declare (optimize (speed 3) #.*no-array-bounds-check*))
      (loop for stripe of-type index below (n-stripes* lump) do
            (with-stripes ((stripe x xs xe)
                           (stripe y ys ye))
              (setf (aref to* stripe)
                    (let ((sum (flt 0)))
                      (declare (type flt sum))
                      (loop for xi upfrom xs below xe
                            for yi upfrom ys below ye
                            do (incf sum (expt (- (aref x* xi)
                                                  (aref y* yi))
                                               2)))
                      sum)))))))

(defmethod derive-lump ((lump ->sum-squared-error))
  (let ((x (x lump))
        (y (y lump)))
    (assert (= (size x) (size y)))
    (assert (= (n-stripes lump) (n-stripes x) (n-stripes y)))
    (let ((x* (storage (nodes x)))
          (xd* (storage (derivatives x)))
          (y* (storage (nodes y)))
          (yd* (storage (derivatives y)))
          (derivatives* (storage (derivatives lump))))
      (declare (optimize (speed 3) #.*no-array-bounds-check*))
      (loop for stripe of-type index below (n-stripes* lump) do
            (let ((d (aref derivatives* stripe)))
              (with-stripes ((stripe x xs xe)
                             (stripe y ys ye))
                (loop for xi upfrom xs below xe
                      for yi upfrom ys below ye
                      do
                      (incf (aref xd* xi)
                            (* d 2 (- (aref x* xi)
                                      (aref y* yi))))
                      (incf (aref yd* yi)
                            (* d 2 (- (aref y* yi)
                                      (aref x* xi)))))))))))

(defclass ->cross-entropy (lump) ())

#+nil
(defmethod transfer-lump ((lump ->cross-entropy))
  (destructuring-bind (x y) (args lump)
    (assert (= (size lump) (size x) (size y)))
    (assert (= (n-stripes lump) (n-stripes x) (n-stripes y)))
    (let ((x* (storage (nodes x)))
          (y* (storage (nodes y)))
          (to* (storage (nodes lump)))
          (size (size x)))
      (declare (type index size)
               (optimize (speed 3) #.*no-array-bounds-check*))
      (loop for stripe of-type index below (n-stripes* lump) do
            (setf (aref to* stripe)
                  (- (loop for i of-type index upfrom (* stripe size)
                           below (* (1+ stripe) size)
                           summing (* (aref x* i)
                                      (the! flt (log (aref y* i)))))))))))

#+nil
(defmethod derive-lump ((lump ->cross-entropy))
  (destructuring-bind (x y) (args lump)
    (assert (= (size lump) (size x) (size y)))
    (assert (= (n-stripes lump) (n-stripes x) (n-stripes y)))
    (let ((x* (storage (nodes x)))
          (xd* (storage (derivatives x)))
          (y* (storage (nodes y)))
          (yd* (storage (derivatives y)))
          (derivatives* (storage (derivatives lump)))
          (size (size x)))
      (declare (type index size)
               (optimize (speed 3) #.*no-array-bounds-check*))
      (loop for stripe of-type index below (n-stripes* lump) do
            (let ((d (aref derivatives* stripe)))
              (loop for i of-type index upfrom (* stripe size)
                    below (* (1+ stripe) size)
                    do
                    (incf (aref xd* i)
                          (* d (- (log (aref y* i)))))
                    (incf (aref yd* i)
                          (* d 2 (- (/ (aref x* i)
                                       (aref y* i)))))))))))


;;;; ERROR-NODE

(defclass error-node (->sum)
  ((importance
    :initform (flt 1) :initarg :importance :accessor importance
    :documentation "Error nodes have their incoming derivative set to
IMPORTANCE no matter what other nodes depend on them."))
  (:documentation "An error node is usually a leaf in the graph of
lumps. Contrary to non-error leaf lumps it gets a non-zero derivative:
IMPORTANCE. Error lumps have exactly one node \(in each stripe) whose
value is computed as the sum of nodes in the X parameter lump."))

(defmethod default-size ((lump error-node))
  1)

(defmethod derive-lump :around ((lump error-node))
  (matlisp:fill-matrix (derivatives lump) (importance lump))
  (call-next-method))


;;;; CROSS-ENTROPY-SOFTMAX-LUMP

(defclass cross-entropy-softmax-lump (lump)
  ((group-size :initarg :group-size :reader group-size)
   (x :initarg :x :reader x :documentation "This is the input lump.")
   (softmax
    :reader softmax
    :documentation "A matrix of the same size as X, EXP'ed and
normalized in groups of GROUP-SIZE.")
   (target
    :initarg :target :reader target
    :documentation "A lump of the same size as INPUT-LUMP that is the
T in -sum_{k}target_k*ln(x_k) which the the cross entropy error.")
   (normalized-lump :reader normalized-lump))
  (:documentation "A specialized lump that is equivalent to hooking
->EXP with NORMALIZED-LUMP and ->CROSS-ENTROPY but is numerically
stable. See <http://groups.google.com/group/comp.ai.neural-nets/msg/a7594ebea01fef04?dmode=source>

It has two parameters X and TARGET. In the transfer phase it computes
the EXP of each input node and normalizes them as if by
NORMALIZED-LUMP. These intermediate values are placed into SOFTMAX.
The value node K is nodes_k = - target_k * ln(softmax_k). Since the
sum of this is cross entropy: - sum_k target_k * ln(softmax_k), simply
plug this lump into an ERROR-NODE.

In the derive phase it computes the cross entropy error of the
normalized input: d(-sum_k{target_k * ln(softmax_k)})/dx_k = sum_j{
target_j * (softmax_k - KDELjk)} which is equal to softmax_k -
target_k if target sums to 1."))

(defmethod default-size ((lump cross-entropy-softmax-lump))
  (size (x lump)))

(defun ensure-softmax (lump)
  (unless (and (slot-boundp lump 'softmax)
               (= (length (storage (nodes (x lump))))
                  (length (storage (softmax lump)))))
    (setf (slot-value lump 'softmax)
          (matlisp:copy (nodes (x lump)))))
  (softmax lump))

(defmethod transfer-lump ((lump cross-entropy-softmax-lump))
  (let* ((x (x lump))
         (group-size (group-size lump))
         (x* (storage (nodes x)))
         (softmax* (storage (ensure-softmax lump)))
         (target (target lump))
         (target* (storage (nodes target)))
         (to* (storage (nodes lump))))
    (declare (type index group-size))
    (loop for stripe of-type index below (n-stripes* lump) do
          (with-stripes ((stripe lump ls le)
                         (stripe x xs xe)
                         (stripe target ts te))
            (loop for li upfrom ls below le
                  for xi upfrom xs below xe
                  for ti upfrom ts below te
                  for i upfrom 0
                  do
                  (when (zerop (mod i group-size))
                    (let ((sum #.(flt 0)))
                      (declare (type flt sum)
                               (optimize (speed 3)))
                      (loop for xj upfrom xi below (+ xi group-size)
                            do (incf sum (exp (aref x* xj))))
                      (loop for lj upfrom li below (+ li group-size)
                            for xj upfrom xi below (+ xi group-size)
                            for tj upfrom ti below (+ ti group-size)
                            do (let ((s (/ (exp (aref x* xj)) sum)))
                                 (declare (type positive-flt s))
                                 (setf (aref softmax* lj) s)
                                 (setf (aref to* lj)
                                       (- (* (aref target* tj)
                                             (the flt (log s))))))))))))))

(defmethod derive-lump ((lump cross-entropy-softmax-lump))
  (let* ((x (x lump))
         (group-size (group-size lump))
         (xd* (storage (derivatives x)))
         (softmax (softmax lump))
         (softmax* (storage softmax))
         (target (target lump))
         (target* (storage (nodes target)))
         (d* (storage (derivatives lump))))
    (declare (type index group-size))
    ;; FIXME: target derivative not calculated
    (assert (typep target 'input-lump))
    (loop for stripe of-type index below (n-stripes* lump) do
          (with-stripes ((stripe lump ls le)
                         (stripe x xs xe)
                         (stripe target ts te))
            ;; loop on groups
            (loop
             for lg of-type index upfrom ls below le by group-size
             for tg of-type index upfrom ts below te by group-size
             for xg of-type index upfrom xs below xe by group-size
             do
             ;; calc d(XENT)/dx_j
             (loop
              for lj of-type index upfrom lg below (+ lg group-size)
              for tj of-type index upfrom tg below (+ tg group-size)
              for xj of-type index upfrom xg below (+ xg group-size)
              do
              ;; Since we cannot be sure that x_i sum to one and all
              ;; elements D* are equal (which is the case if this is
              ;; hooked into an error node directly), we cannot take
              ;; the shortcut of d(XENT)/dx_j = softmax_j - target_j.
              ;; Instead, we must calculate
              ;; sum_i{target_i*(softmax_j-KDEL_ij)} where KDEL is the
              ;; Kronecker delta.
              (locally
                  (declare (optimize (speed 3)))
                (loop
                 for li upfrom lg below (+ lg group-size)
                 for ti upfrom tg below (+ tg group-size)
                 do (incf (aref xd* xj)
                          (* (aref d* li)
                             (aref target* ti)
                             (- (aref softmax* lj)
                                (if (= ti tj)
                                    #.(flt 1)
                                    #.(flt 0)))))))))))))
