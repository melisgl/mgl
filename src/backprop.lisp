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
(defvar *in-training-p* nil)

(defun next-lump-name ()
  (prog1 (or *next-lump-name* (gensym))
    (setf *next-lump-name* nil)))

(defgeneric default-size (lump)
  (:method (lump)
    (or (slot-boundp lump 'size)
        (error "Can't compute size for ~S." lump))))

(defclass lump ()
  ((name :initform (next-lump-name) :initarg :name :reader name)
   (size :type index :initarg :size :reader size)
   (n-stripes :initform 1 :type index :reader n-stripes)
   (max-n-stripes
    :initform 1 :type index :initarg :max-n-stripes
    :reader max-n-stripes)
   (same-stripes-p
    :initform nil :initarg :same-stripes-p :reader same-stripes-p
    :documentation "Non-NIL iff all stripes are the same. If true, it
effectively overrides both N-STRIPES and MAX-N-STRIPES and there is
only one column in NODES and DERIVATIVES. Set up by the lump itself
taking its inputs into account. Notably, ->WEIGHTS always have
SAME-STRIPES-P T.")
   (nodes
    :initform nil :type (or flt-vector null) :reader nodes
    :documentation "The values of the nodes. All nodes have values. It
is conceptually a N-STRIPES x SIZE matrix that can be enlarged to
MAX-N-STRIPES x SIZE by setting N-STRIPES.")
   (derivatives
    :initform nil :type (or flt-vector null) :reader derivatives
    :documentation "Derivatives of nodes, input node derivatives are
not calculated. A 1d array representing a matrix of the same dimension
as NODES.")
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

(declaim (inline nodes*))
(defun nodes* (lump)
  (the flt-vector (values (nodes lump))))

(declaim (inline derivatives*))
(defun derivatives* (lump)
  (the flt-vector (values (derivatives lump))))

(defmacro deflump (name direct-superclasses direct-slots &rest options)
  (destructuring-bind (name maker-name)
      (if (listp name) name (list name name))
    `(progn
       (defclass ,name ,direct-superclasses ,direct-slots ,@options)
       ;; FIXME: extract initarg names for programmer convenience
       (defun ,maker-name (&rest args)
         (apply #'make-instance ',name args)))))

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

(defun norm (v)
  (sqrt (loop for i below (length v)
              sum (expt (aref v i) 2))))

(defmethod print-object ((lump lump) stream)
  (pprint-logical-block (stream ())
    (print-unreadable-object (lump stream :type t)
      (format stream "~S ~:_~S ~:_~S" (name lump) :size
              (if (slot-boundp lump 'size)
                  (size lump)
                  :unbound))
      (format stream "*~S(~S/~S) :norm ~,5F" (n-stripes* lump) (n-stripes lump)
              (max-n-stripes lump) (ignore-errors (norm (nodes lump))))
      (when (indices-to-calculate lump)
        (format stream " ~:_(~S)" (length (indices-to-calculate lump))))))
  lump)

(defmethod set-n-stripes (n-stripes (lump lump))
  (assert (<= 0 n-stripes (max-n-stripes lump)))
  (setf (slot-value lump 'n-stripes) n-stripes)
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
                   (make-flt-array (* max-n-stripes* size)))
             (fill! (default-value lump) (nodes lump))
             (setf (slot-value lump 'derivatives)
                   (make-flt-array (* max-n-stripes* size)))
             ;; make sure the number of columns is set properly
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
(defgeneric set-input-done (lump)
  (:method (lump)))


;;;; Data lumps

(defclass data-lump (lump) ())

(deflump (->weight ->weight*) (data-lump)
  ((same-stripes-p :initform t)))

(defvar *lumps-to-copy* ())

(defmacro with-weights-copied ((from-bpn) &body body)
  "In BODY ->WEIGHT will first look up if a weight lump of the same
name exists in FROM-BPN and return that, or else create a weight lump
normally. If FROM-BPN is NIL, then weights are copied."
  (alexandria:with-gensyms (%from-bpn)
    `(let* ((,%from-bpn ,from-bpn)
            (*lumps-to-copy*
              (when ,%from-bpn
                (remove-if-not (lambda (lump)
                                 (typep lump '->weight))
                               (lumps ,%from-bpn)))))
       ,@body)))

(defun ->weight (&rest args)
  (let ((to-be-copied (find mgl-bp::*next-lump-name* *lumps-to-copy*
                            :key #'name :test #'equal)))
    (cond (to-be-copied
           (when *bpn-being-built*
             (add-lump to-be-copied *bpn-being-built*))
           to-be-copied)
          (t
           (apply #'->weight* args)))))

(deflump ->constant (data-lump)
  ((default-value :initform #.(flt 1))))

(defmethod transfer-lump ((lump data-lump)))

(defmethod derive-lump ((lump data-lump)))

(deflump ->input (data-lump)
  ((running-stats :reader running-stats)
   (update-stats-p :initform nil :initarg :update-stats-p
                   :accessor update-stats-p)
   (normalize-with-stats-p :initform nil :initarg :normalize-with-stats-p
                           :accessor normalize-with-stats-p)
   (normalized-cap :initform nil :initarg :normalized-cap
                   :accessor normalized-cap)))

(defmethod set-input-done ((lump ->input))
  (let ((nodes (nodes* lump))
        (n-stripes* (n-stripes* lump))
        (size (size lump)))
    (when (update-stats-p lump)
      (unless (slot-boundp lump 'running-stats)
        (setf (slot-value lump 'running-stats)
              (coerce (loop repeat (size lump)
                            collect (make-instance 'running-stat))
                      'vector)))
      (let ((running-stats (running-stats lump)))
        (dotimes (i size)
          (let ((running-stat (aref running-stats i)))
            (dotimes (j n-stripes*)
              (add-to-running-stat (aref nodes (+ i (* j size)))
                                   running-stat))))))
    (when (normalize-with-stats-p lump)
      (assert (slot-boundp lump 'running-stats))
      (let ((running-stats (running-stats lump))
            (cap (normalized-cap lump)))
        (dotimes (i size)
          (let* ((running-stat (aref running-stats i))
                 (mean (running-stat-mean running-stat))
                 (stddev (sqrt (running-stat-variance running-stat))))
            (dotimes (j n-stripes*)
              (let* ((index (+ i (* j size)))
                     (x (if (zerop stddev)
                            (- (aref nodes index) mean)
                            (/ (- (aref nodes index) mean)
                               stddev))))
                (setf (aref nodes index)
                      (cond ((= least-negative-flt x)
                             #.(flt 0))
                            ((and cap (< cap x))
                             cap)
                            ((and cap (< x (- cap)))
                             (- cap))
                            (t
                             x)))))))))))


;;;; ->ERROR

(deflump ->error (->sum)
  ((importance
    :initform nil
    :initarg :importance
    :accessor importance
    :documentation "If non-NIL, an FLT-VECTOR of n-stripes."))
  (:documentation "An error node is usually a leaf in the graph of
lumps. Contrary to non-error leaf lumps it gets a non-zero derivative:
1. Error lumps have exactly one node \(in each stripe) whose value is
computed as the sum of nodes in the X parameter lump."))

(defmethod default-size ((lump ->error))
  1)

(defmethod derive-lump :around ((lump ->error))
  (if (importance lump)
      (replace (derivatives lump) (importance lump))
      (fill! (flt 1) (derivatives lump)))
  (call-next-method))


;;;; BPN

(defclass bpn ()
  ((lumps
    :initform (make-array 0 :element-type 'lump :adjustable t :fill-pointer t)
    :initarg :lumps
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
  (call-next-method)
  (loop for lump across (lumps bpn) do
    (set-input-done lump)))

(defun add-lump (lump bpn)
  "Add LUMP to BPN. MAX-N-STRIPES of LUMP gets set to equal that of
the previous last, non-weight lump of BPN."
  (when (find-lump (name lump) bpn)
    (error "Cannot add ~S: ~%
            a lump of same name has already been added to this network." lump))
  (setf (max-n-stripes lump) (max-n-stripes bpn))
  (vector-push-extend lump (slot-value bpn 'lumps) 1)
  lump)

(defun remove-lump (lump bpn)
  (setf (slot-value bpn 'lumps)
        (delete lump (lumps bpn)))
  lump)

(defmacro build-bpn ((&key (class ''bpn) initargs
                        (max-n-stripes 1)) &body lumps)
  "Syntactic sugar to assemble BPNs from lumps. Like LET* it is a
sequence of bindings (of symbols to lumps). The names of the lumps
created default to the symbol of the binding. In case a lump is not
bound to a symbol (because it was created in a nested expression), the
local function LUMP finds the lump with the given name in the bpn
being built. Example:

  (mgl-bp:build-bpn ()
    (features (mgl-bp:->input :size n-features))
    (biases (mgl-bp:->weight :size n-features))
    (weights (mgl-bp:->weight :size (* n-hiddens n-features)))
    (activations0 (mgl-bp:->activation :weights weights :x features))
    (activations (mgl-bp:->+ :args (list biases activations0)))
    (output (mgl-bp:->sigmoid :x activations)))"
  (let ((bindings
          (mapcar (lambda (lump)
                    (destructuring-bind (symbol init-form) lump
                      `(,symbol (let ((*next-lump-name* ',symbol))
                                  (,(first init-form)
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

(defun copy-bpn-weights (from-bpn to-bpn &key error-if-no-match-p)
  (loop for from-lump across (lumps from-bpn)
        do (when (typep from-lump '->weight)
             (let ((to-lump (find-lump (name from-lump) to-bpn
                                       :errorp error-if-no-match-p)))
               (assert (= (size to-lump) (size from-lump)))
               (setf (slot-value to-lump 'nodes) (nodes from-lump))))))

(defun forward-bpn (bpn &key from-lump to-lump end-lump)
  "Propagate the values from the already clamped inputs."
  (declare (optimize (debug 2)))
  (let ((from-lump (if from-lump (->lump bpn from-lump) nil))
        (to-lump (if to-lump (->lump bpn to-lump) nil))
        (seen-from-lump-p (not from-lump)))
    (loop for lump across (lumps bpn)
          until (eq lump end-lump)
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
          when (not (typep lump '->weight))
          do (fill! (flt 0) (derivatives lump)))))

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
         (when (typep lump '->weight)
           (funcall fn lump)))
       (lumps bpn)))

(defmethod segment-weights ((lump lump))
  (let ((nodes (nodes lump)))
    (values nodes 0 (length nodes))))

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
stripe is the sum of the error nodes. The second value is the number
of stripes."
  (let ((sum (flt 0)))
    (loop for lump across (lumps bpn) do
          (when (typep lump '->error)
            (let ((nodes (nodes lump)))
              (incf sum (* (if (same-stripes-p lump)
                               (n-stripes lump)
                               1)
                           (clnu:sum nodes))))))
    (values sum (n-stripes bpn))))

(defgeneric compute-derivatives (samples trainer bpn)
  (:method (samples trainer bpn)
    (let ((*in-training-p* t))
      (set-input samples bpn)
      (forward-bpn bpn)
      (backward-bpn bpn :last-lump (first-trained-weight-lump trainer bpn)))))


;;;; CG trainer

(defclass cg-bp-trainer (base-bp-trainer cg-trainer) ())

(defun segment-set-derivatives->weights (segment-set weights)
  (declare (type flt-vector weights)
           (optimize (speed 3)))
  (do-segment-set (lump :start-in-segment-set start-in-segment-set) segment-set
    (let ((derivatives (derivatives* lump)))
      (declare (type flt-vector derivatives))
      (replace weights derivatives
               :start1 start-in-segment-set
               :start2 0 :end2 (length derivatives)))))

(defmethod compute-batch-cost-and-derive (samples trainer (bpn bpn))
  (let ((cost #.(flt 0)))
    (do-segment-set (lump) (segment-set trainer)
      (let ((derivatives (derivatives lump)))
        (fill! (flt 0) derivatives)))
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
  ;;  (let ((ones (ones (n-stripes bpn) 1)))
  ;;    (do-segment-gradient-accumulators ((lump acc-start accumulator) trainer)
  ;;      (lla:gemm! (flt 1) (derivatives lump) ones (flt 1) accumulator)
  ;;      (fill! (flt 0) (derivatives lump))))
  ;;
  ;; save for the one-to-many segment-trainer and segment run
  ;; complications.
  (do-segment-gradient-accumulators ((lump acc-start accumulator) trainer)
    (let ((derivatives (derivatives* lump)))
      (map-segment-runs
       (lambda (start1 end1)
         (declare (type index start1 end1))
         (assert (<= (+ acc-start end1) (length accumulator)))
         (assert (typep lump '->weight))
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
                   do (incf (aref accumulator j) (aref derivatives i))))))
       lump)))
  ;; All weight derivatives must be zeroed, even the ones not being
  ;; trained on to avoid overflows.
  (loop for lump across (lumps bpn)
        do (when (typep lump '->weight)
             (fill! (flt 0) (derivatives lump)))))

(defmethod train (sampler (trainer bp-trainer) (bpn bpn))
  (while (not (finishedp sampler))
    (let ((samples (sample-batch sampler (n-inputs-until-update trainer))))
      (do-executors (samples bpn)
        (train-batch samples trainer bpn)))))

(defmethod train-batch (batch (trainer bp-trainer) (bpn bpn))
  (loop for samples in (group batch (max-n-stripes bpn))
        do (compute-derivatives samples trainer bpn))
  (add-and-forget-derivatives trainer bpn)
  (maybe-update-weights trainer (length batch)))


;;;; I/O

(defmethod write-weights ((lump ->weight) stream)
  (write-double-float-array (nodes lump) stream))

(defmethod read-weights ((lump ->weight) stream)
  (read-double-float-array (nodes lump) stream))

(defmethod write-weights ((bpn bpn) stream)
  (map-segments (lambda (weights)
                  (write-weights weights stream))
                bpn))

(defmethod read-weights ((bpn bpn) stream)
  (map-segments (lambda (weights)
                  (read-weights weights stream))
                bpn))


;;;; ->NORMALIZED

(deflump ->normalized (lump)
  ((x :initarg :x :reader x :documentation "Input comes from here.")
   (group-size :initarg :group-size :reader group-size)
   (scale
    :initform #.(flt 1)
    :type (or flt flt-vector)
    :initarg :scale :accessor scale
    :documentation "The sum of nodes after normalization. Can be
changed during training, for instance when clamping. If it is a vector
then its length must be MAX-N-STRIPES which automatically
maintained.")))

(defmethod default-size ((lump ->normalized))
  (size (x lump)))

(defmethod set-max-n-stripes (max-n-stripes (lump ->normalized))
  (call-next-method)
  (when (and (typep (scale lump) 'flt-vector)
             (/= (max-n-stripes lump) (length (scale lump))))
    (setf (scale lump) (make-flt-array (max-n-stripes lump))))
  max-n-stripes)

(defmethod transfer-lump ((lump ->normalized))
  (let* ((x (x lump))
         (group-size (group-size lump))
         (scale (scale lump))
         (x* (nodes* x))
         (to* (nodes* lump)))
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

(defmethod derive-lump ((lump ->normalized))
  (let* ((x (x lump))
         (group-size (group-size lump))
         (scale (scale lump))
         (x* (nodes* x))
         (xd* (derivatives* x))
         (ld* (derivatives* lump)))
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

(deflump ->activation (lump)
  ((weights :type ->weight :initarg :weights :reader weights)
   (x :initarg :x :reader x :documentation "Input comes from here.")
   (transpose-weights-p :initform nil :initarg :transpose-weights-p
                        :reader transpose-weights-p))
  (:documentation "Perform X*WEIGHTS where X is of size M and WEIGHTS
is a ->WEIGHT whose single stripe is taken to be of dimensions M x N
stored in column major order. N is the size of this lump. If
TRANSPOSE-WEIGHTS-P then WEIGHTS is N x M and X*WEIGHTS' is
computed."))

(defmethod default-size ((lump ->activation))
  (/ (size (weights lump))
     (size (x lump))))

(defmethod transfer-lump ((lump ->activation))
  (let* ((x (x lump))
         (weights (weights lump))
         (n-stripes (n-stripes lump))
         (nx (size x))
         (nl (/ (size weights)
                nx)))
    ;; FIXME:
    (assert (not (same-stripes-p x)))
    (if (transpose-weights-p lump)
        (lla:gemm! (flt 1) (nodes x)
                    (aops:reshape (nodes weights) (list nl nx))
                    (flt 0) (nodes lump)
                    :transpose-b? t :lda nx :ldc nl
                    :m n-stripes :n nl :k nx)
        (lla:gemm! (flt 1) (nodes x)
                    (aops:reshape (nodes weights) (list nx nl))
                    (flt 0) (nodes lump)
                    :lda nx :ldc nl :m n-stripes :n nl :k nx))))

(defmethod derive-lump ((lump ->activation))
  (let* ((x (x lump))
         (weights (weights lump))
         (n-stripes (n-stripes lump))
         (nx (size x))
         (nl (/ (size weights)
                nx)))
    ;; FIXME: transform RESHAPE out
    (if (transpose-weights-p lump)
        ;; dx += a*w
        (lla:gemm! (flt 1) (derivatives lump)
                    (aops:reshape (nodes weights) (list nl nx))
                    (flt 1) (derivatives x)
                    :lda nl :ldc nx
                    :m n-stripes :n nx :k nl)
        ;; dx += a*w'
        (lla:gemm! (flt 1) (derivatives lump)
                    (aops:reshape (nodes weights) (list nx nl))
                    (flt 1) (derivatives x)
                    :transpose-b? t :lda nl :ldc nx
                    :m n-stripes :n nx :k nl))
    (if (transpose-weights-p lump)
        ;; dw += a'*x
        (lla:gemm! (flt 1) (derivatives lump) (nodes x)
                    (flt 1) (aops:reshape (derivatives weights) (list nl nx))
                    :transpose-a? t
                    :lda nl :ldb nx :ldc nx
                    :m nl :n nx :k n-stripes)
        ;; dw += x'*a
        (lla:gemm! (flt 1) (nodes x) (derivatives lump)
                    (flt 1) (aops:reshape (derivatives weights) (list nx nl))
                    :transpose-a? t
                    :lda nx :ldb nl :ldc nl
                    :m nx :n nl :k n-stripes))))


;;;; Node type library

(deflump ->rep (lump)
  ((x :initarg :x :reader x)
   (n :initarg :n :reader n)))

(defmethod default-size ((lump ->rep))
  (* (n lump) (size (x lump))))

(defmethod transfer-lump ((lump ->rep))
  (let ((x (x lump)))
    (assert (= (n-stripes lump) (n-stripes x)))
    (let ((x* (nodes* x))
          (to* (nodes* lump))
          (n (n lump))
          (xn (size x)))
      (declare (optimize (speed 3) #.*no-array-bounds-check*)
               (type index n xn))
      (loop for stripe of-type index below (n-stripes* lump) do
        (with-stripes ((stripe x xs xe)
                       (stripe lump ls le))
          (declare (ignore xe le))
          (dotimes (i xn)
            (let ((v (aref x* (+ xs i))))
              (loop for li of-type index upfrom (+ ls i) by xn
                    repeat n
                    do (setf (aref to* li) v)))))))))

(defmethod derive-lump ((lump ->rep))
  (let ((x (x lump)))
    (assert (= (n-stripes lump) (n-stripes x)))
    (let ((xd* (derivatives* x))
          (d* (derivatives* lump))
          (n (n lump))
          (xn (size x)))
      (declare (optimize (speed 3) #.*no-array-bounds-check*)
               (type index n xn))
      (loop for stripe of-type index below (n-stripes* lump) do
        (with-stripes ((stripe x xs xe)
                       (stripe lump ls le))
          (declare (ignore xe le))
          (dotimes (i xn)
            (let ((sum (flt 0)))
              (loop for li of-type index upfrom (+ ls i) by xn
                    repeat n
                    do (incf sum (aref d* li)))
              (setf (aref xd* (+ xs i)) sum))))))))


(deflump ->stretch (lump)
  ((x :initarg :x :reader x)
   (n :initarg :n :reader n)))

(defmethod default-size ((lump ->stretch))
  (* (n lump) (size (x lump))))

(defmethod transfer-lump ((lump ->stretch))
  (let ((x (x lump)))
    (assert (= (n-stripes lump) (n-stripes x)))
    (let ((x* (nodes* x))
          (l* (nodes* lump))
          (n (n lump)))
      (declare (optimize (speed 3) #.*no-array-bounds-check*)
               (type index n))
      (loop for stripe of-type index below (n-stripes* lump) do
        (with-stripes ((stripe x xs xe)
                       (stripe lump ls le))
          (declare (ignore le))
          (let ((li ls))
            (loop for xi upfrom xs below xe
                  do (let ((v (aref x* xi)))
                       (loop repeat n
                             do (setf (aref l* li) v)
                                (incf li))))))))))

(defmethod derive-lump ((lump ->stretch))
  (let ((x (x lump)))
    (assert (= (n-stripes lump) (n-stripes x)))
    (let ((xd* (derivatives* x))
          (d* (derivatives* lump))
          (n (n lump)))
      (declare (optimize (speed 3) #.*no-array-bounds-check*)
               (type index n))
      (loop for stripe of-type index below (n-stripes* lump) do
        (with-stripes ((stripe x xs xe)
                       (stripe lump ls le))
          (declare (ignore le))
          (let ((li ls))
            (loop for xi upfrom xs below xe
                  do (let ((sum (flt 0)))
                       (loop repeat n
                             do (incf sum (aref d* li))
                                (incf li))
                       (incf (aref xd* xi) sum)))))))))


(deflump ->+ (lump)
  ((args :initarg :args :reader args)))

(defmethod default-size ((lump ->+))
  (size (first (args lump))))

(defmethod transfer-lump ((lump ->+))
  (let* ((to (nodes lump))
         (n-stripes* (n-stripes* lump))
         (ones (make-flt-array (list n-stripes* 1) :initial-element (flt 1))))
    (fill! (flt 0) to)
    (dolist (arg (args lump))
      (cond ((= n-stripes* (n-stripes* arg))
             (lla:axpy! 1 (nodes arg) to))
            (t
             (assert (same-stripes-p arg))
             (lla:gemm! (flt 1) ones (nodes arg) (flt 1) to
                         :m n-stripes* :n (size lump) :k 1
                         :ldb (size arg) :ldc (size lump)))))))

(defmethod derive-lump ((lump ->+))
  (let* ((derivatives (derivatives* lump))
         (n-stripes* (n-stripes* lump))
         (ones (make-flt-array (list 1 n-stripes*) :initial-element (flt 1))))
    (dolist (arg (args lump))
      (cond ((= n-stripes* (n-stripes* arg))
             (lla:axpy! 1 (derivatives lump) (derivatives arg)))
            (t
             (assert (same-stripes-p arg))
             (lla:gemm! (flt 1) ones derivatives (flt 1) (derivatives arg)
                         :m 1 :n (size arg) :k n-stripes*
                         :ldb (size lump) :ldc (size arg)))))))


(deflump ->* (lump)
  ((x :initarg :x :reader x)
   (y :initarg :y :reader y)))

(defmethod default-size ((lump ->*))
  (size (x lump)))

(defmethod transfer-lump ((lump ->*))
  (let* ((to (nodes lump))
         (n-stripes* (n-stripes* lump))
         (x (x lump))
         (y (y lump)))
    (cond ((= n-stripes* (n-stripes* x))
           (cond ((numberp y)
                  (fill! (flt 0) to)
                  (lla:axpy! y (nodes x) to))
                 (t
                  (let ((x* (nodes x))
                        (y* (nodes y)))
                    (dotimes (i (* n-stripes* (size lump)))
                      (setf (aref to i) (* (aref x* i) (aref y* i))))))))
          (t
           (assert nil)))))

(defmethod derive-lump ((lump ->*))
  (let* ((d* (derivatives* lump))
         (n-stripes* (n-stripes* lump))
         (x (x lump))
         (xd* (derivatives* x))
         (y (y lump)))
    (cond ((= n-stripes* (n-stripes* x))
           (cond ((numberp y)
                  (dotimes (i (* n-stripes* (size lump)))
                    (incf (aref xd* i) (* y (aref d* i)))))
                 (t
                  (let ((x* (nodes* x))
                        (y* (nodes* y))
                        (yd* (derivatives* y)))
                    (dotimes (i (* n-stripes* (size lump)))
                      (incf (aref xd* i) (* (aref y* i) (aref d* i))))
                    (dotimes (i (* n-stripes* (size lump)))
                      (incf (aref yd* i) (* (aref x* i) (aref d* i))))))))
          (t
           (assert nil)))))

(deflump ->sum (lump)
  ((x :initarg :x :reader x))
  (:documentation "Sum of all nodes \(per stripe)."))

(defmethod default-size ((lump ->sum))
  1)

(defmethod transfer-lump ((lump ->sum))
  (let ((x (x lump)))
    (assert (= (n-stripes lump) (n-stripes x)))
    (let ((x* (nodes* x))
          (to* (nodes* lump)))
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
    (let ((xd* (derivatives* x))
          (derivatives* (derivatives* lump)))
      (declare (optimize (speed 3) #.*no-array-bounds-check*))
      (loop for stripe of-type index below (n-stripes* lump) do
        (let ((d (aref derivatives* stripe)))
          (with-stripes ((stripe x xs xe))
            (loop for xi upfrom xs below xe
                  do (incf (aref xd* xi) d))))))))

(deflump ->abs (lump)
  ((x :initarg :x :reader x)))

(defmethod default-size ((lump ->abs))
  (size (x lump)))

(defmethod transfer-lump ((lump ->abs))
  (let ((x (x lump)))
    (assert (= (size lump) (size x)))
    (let ((x* (nodes* x))
          (l* (nodes* lump)))
      (declare (optimize (speed 3) #.*no-array-bounds-check*))
      (loop for stripe of-type index below (n-stripes* lump) do
        (with-stripes ((stripe lump ls le)
                       (stripe x xs xe))
          (loop for li upfrom ls below le
                for xi upfrom xs below xe
                do (setf (aref l* li) (abs (aref x* xi)))))))))

(defmethod derive-lump ((lump ->abs))
  (let ((x (x lump)))
    (assert (= (size lump) (size x)))
    (let ((x* (nodes* x))
          (xd* (derivatives* x))
          (ld* (derivatives* lump)))
      (declare (optimize (speed 3) #.*no-array-bounds-check*))
      (loop for stripe of-type index below (n-stripes* lump) do
        (with-stripes ((stripe lump ls le)
                       (stripe x xs xe))
          (loop for li upfrom ls below le
                for xi upfrom xs below xe
                do (incf (aref xd* xi) (* (aref ld* li)
                                          ;; make sure it doesn't get
                                          ;; stuck at 0
                                          (if (minusp (aref x* xi))
                                              (flt -1)
                                              (flt 1))))))))))

(deflump ->linear (lump)
  ((x :initarg :x :reader x)
   (y :initarg :y :reader y)))

(defmethod default-size ((lump ->linear))
  1)

(defmethod transfer-lump ((lump ->linear))
  (let ((x (x lump))
        (y (y lump)))
    (assert (= 1 (size lump)))
    (assert (= (size x) (size y)))
    (assert (= (n-stripes lump) (n-stripes x) (n-stripes y)))
    (let ((x* (nodes* x))
          (y* (nodes* y))
          (to* (nodes* lump)))
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
    (assert (= 1 (size lump)))
    (assert (= (size x) (size y)))
    (assert (= (n-stripes lump) (n-stripes x) (n-stripes y)))
    (let ((x* (nodes* x))
          (xd* (derivatives* x))
          (y* (nodes* y))
          (yd* (derivatives* y))
          (derivatives* (derivatives* lump)))
      (declare (optimize (speed 3) #.*no-array-bounds-check*))
      (loop for stripe of-type index below (n-stripes* lump) do
        (let ((d (aref derivatives* stripe)))
          (with-stripes ((stripe x xs xe)
                         (stripe y ys ye))
            (loop for xi upfrom xs below xe
                  for yi upfrom ys below ye
                  do (incf (aref xd* xi)
                           (* d (aref y* yi)))
                     (incf (aref yd* yi)
                           (* d (aref x* xi))))))))))

(deflump ->sigmoid (lump)
  ((x :initarg :x :reader x)
   (dropout
    :initform nil :initarg :dropout :reader dropout
    :documentation "If non-NIL, then in the forward pass zero out each
node in this chunk with DROPOUT probability. See Geoffrey Hinton's
'Improving neural networks by preventing co-adaptation of feature
detectors'.")))

(defmethod default-size ((lump ->sigmoid))
  (size (x lump)))

(defmethod transfer-lump ((lump ->sigmoid))
  (let ((x (x lump)))
    (assert (= (size lump) (size x)))
    (let ((x* (nodes* x))
          (l* (nodes* lump))
          (dropout (dropout lump)))
      (declare (optimize (speed 3) #.*no-array-bounds-check*)
               (type (or flt null) dropout))
      (loop for stripe of-type index below (n-stripes* lump) do
        (with-stripes ((stripe lump ls le)
                       (stripe x xs xe))
          (loop for li upfrom ls below le
                for xi upfrom xs below xe
                do (setf (aref l* li)
                         (if dropout
                             (if *in-training-p*
                                 (if (try-chance dropout)
                                     #.(flt 0)
                                     (sigmoid (aref x* xi)))
                                 (* (- #.(flt 1) dropout)
                                    (sigmoid (aref x* xi))))
                             (sigmoid (aref x* xi))))))))))

(defmethod derive-lump ((lump ->sigmoid))
  (let ((x (x lump)))
    (assert (= (size lump) (size x)))
    (let ((xd* (derivatives* x))
          (l* (nodes* lump))
          (ld* (derivatives* lump)))
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

#+nil
(defmethod transfer-lump ((lump ->sigmoid))
  (let ((x (x lump)))
    (assert (= (size lump) (size x)))
    (let ((x* (nodes* x))
          (l* (nodes* lump))
          (dropout (dropout lump)))
      (declare (optimize (speed 3) #.*no-array-bounds-check*)
               (type (or flt null) dropout))
      (loop for stripe of-type index below (n-stripes* lump) do
        (with-stripes ((stripe lump ls le)
                       (stripe x xs xe))
          (loop for li upfrom ls below le
                for xi upfrom xs below xe
                do (setf (aref l* li)
                         (if dropout
                             (if *in-training-p*
                                 (if (try-chance dropout)
                                     #.(flt 0)
                                     (/ (sigmoid (aref x* xi))
                                        (- #.(flt 1) dropout)))
                                 (sigmoid (aref x* xi)))
                             (sigmoid (aref x* xi))))))))))

#+nil
(defmethod derive-lump ((lump ->sigmoid))
  (let ((x (x lump)))
    (assert (= (size lump) (size x)))
    (let ((x* (nodes* x))
          (xd* (derivatives* x))
          (l* (nodes* lump))
          (ld* (derivatives* lump))
          (dropout (dropout lump)))
      (declare (optimize (speed 3) #.*no-array-bounds-check*)
               (type (or flt null) dropout))
      (loop for stripe of-type index below (n-stripes* lump) do
        (with-stripes ((stripe lump ls le)
                       (stripe x xs xe))
          (loop for li upfrom ls below le
                for xi upfrom xs below xe
                do (unless (zerop (aref l* li))
                     (incf (aref xd* li)
                           (let ((s (sigmoid (aref x* xi))))
                             #+nil
                             (* (aref ld* li)
                                s (- 1 s))
                             (/ (* (aref ld* li)
                                   s (- 1 s))
                                (- #.(flt 1) dropout)))))))))))

(deflump ->stochastic-sigmoid (lump)
  ((x :initarg :x :reader x)))

(defmethod default-size ((lump ->stochastic-sigmoid))
  (size (x lump)))

(defmethod transfer-lump ((lump ->stochastic-sigmoid))
  (let ((x (x lump)))
    (assert (= (size lump) (size x)))
    (let ((x* (nodes* x))
          (l* (nodes* lump)))
      (declare (optimize (speed 3) #.*no-array-bounds-check*))
      (loop for stripe of-type index below (n-stripes* lump) do
        (with-stripes ((stripe lump ls le)
                       (stripe x xs xe))
          (loop for li upfrom ls below le
                for xi upfrom xs below xe
                do (setf (aref l* li)
                         (if *in-training-p*
                             (binarize-randomly (sigmoid (aref x* xi)))
                             (sigmoid (aref x* xi))))))))))

(defmethod derive-lump ((lump ->stochastic-sigmoid))
  (let ((x (x lump)))
    (assert (= (size lump) (size x)))
    (let ((x* (nodes* x))
          (xd* (derivatives* x))
          (l* (nodes* lump))
          (ld* (derivatives* lump)))
      (declare (optimize (speed 3) #.*no-array-bounds-check*))
      (loop for stripe of-type index below (n-stripes* lump) do
        (with-stripes ((stripe lump ls le)
                       (stripe x xs xe))
          (loop for li upfrom ls below le
                for xi upfrom xs below xe
                do (unless (zerop (aref l* li))
                     (incf (aref xd* li)
                           (let ((s (sigmoid (aref x* xi))))
                             (* (aref ld* li)
                                s (- 1 s)))))))))))

(deflump ->scaled-tanh (lump)
  ((x :initarg :x :reader x)))

(defmethod default-size ((lump ->scaled-tanh))
  (size (x lump)))

(defmethod transfer-lump ((lump ->scaled-tanh))
  (let ((x (x lump)))
    (assert (= (size lump) (size x)))
    (let ((x* (nodes* x))
          (l* (nodes* lump)))
      (declare (optimize (speed 3) #.*no-array-bounds-check*))
      (loop for stripe of-type index below (n-stripes* lump) do
            (with-stripes ((stripe lump ls le)
                           (stripe x xs xe))
              (loop for li upfrom ls below le
                    for xi upfrom xs below xe
                    do (setf (aref l* li) (scaled-tanh (aref x* xi)))))))))

(defmethod derive-lump ((lump ->scaled-tanh))
  (let ((x (x lump)))
    (assert (= (size lump) (size x)))
    (let ((x* (nodes* x))
          (xd* (derivatives* x))
          (ld* (derivatives* lump)))
      (declare (optimize (speed 3) #.*no-array-bounds-check*))
      (loop for stripe of-type index below (n-stripes* lump) do
            (with-stripes ((stripe lump ls le)
                           (stripe x xs xe))
              (loop for li upfrom ls below le
                    for xi upfrom xs below xe do
                    (incf (aref xd* xi)
                          (* (aref ld* li)
                             #.(flt (/ 7137 6239))
                             (expt (sech (* #.(flt 2/3) (aref x* xi)))
                                   2)))))))))
(deflump ->rectified (lump)
  ((x :initarg :x :reader x)
   (noisyp :initform nil :initarg :noisyp :accessor noisyp))
  (:documentation "max(0,x) activation function. If NOISYP then add
normal(0,sigmoid(x)) noise to x."))

(defmethod default-size ((lump ->rectified))
  (size (x lump)))

(defmethod transfer-lump ((lump ->rectified))
  (let ((x (x lump)))
    (assert (= (size lump) (size x)))
    (let ((x* (nodes* x))
          (l* (nodes* lump))
          (noisyp (noisyp lump)))
      (declare (optimize (speed 3) #.*no-array-bounds-check*))
      (loop for stripe of-type index below (n-stripes* lump) do
        (with-stripes ((stripe lump ls le)
                       (stripe x xs xe))
          (loop for li upfrom ls below le
                for xi upfrom xs below xe
                do (setf (aref l* li)
                         (max #.(flt 0)
                              (let ((xi (aref x* xi)))
                                (+ xi
                                   (if noisyp
                                       (* (the! flt (gaussian-random-1))
                                          (sqrt (max #.(flt 0.0001)
                                                     (sigmoid xi))))
                                       #.(flt 0))))))))))))

(defmethod derive-lump ((lump ->rectified))
  (let ((x (x lump)))
    (assert (= (size lump) (size x)))
    (let ((xd* (derivatives* x))
          (l* (nodes* lump))
          (ld* (derivatives* lump)))
      (declare (optimize (speed 3) #.*no-array-bounds-check*))
      (loop for stripe of-type index below (n-stripes* lump) do
        (with-stripes ((stripe lump ls le)
                       (stripe x xs xe))
          (loop for li upfrom ls below le
                for xi upfrom xs below xe
                do (when (plusp (aref l* li))
                     (incf (aref xd* li) (aref ld* li)))))))))

(deflump ->softplus (lump)
  ((x :initarg :x :reader x))
  (:documentation "log(1+exp(x))) activation function."))

(defmethod default-size ((lump ->softplus))
  (size (x lump)))

(defmethod transfer-lump ((lump ->softplus))
  (let ((x (x lump)))
    (assert (= (size lump) (size x)))
    (let ((x* (nodes* x))
          (l* (nodes* lump)))
      (declare (optimize (speed 3) #.*no-array-bounds-check*))
      (loop for stripe of-type index below (n-stripes* lump) do
        (with-stripes ((stripe lump ls le)
                       (stripe x xs xe))
          (loop for li upfrom ls below le
                for xi upfrom xs below xe
                do (setf (aref l* li)
                         (let ((xi (aref x* xi)))
                           (if (< xi (flt 300))
                               (log (1+ (exp xi)))
                               xi)))))))))

(defmethod derive-lump ((lump ->softplus))
  (let ((x (x lump)))
    (assert (= (size lump) (size x)))
    (let ((x* (nodes* x))
          (xd* (derivatives* x))
          (l* (nodes* lump))
          (ld* (derivatives* lump)))
      (declare (optimize (speed 3) #.*no-array-bounds-check*))
      (loop for stripe of-type index below (n-stripes* lump) do
        (with-stripes ((stripe lump ls le)
                       (stripe x xs xe))
          (loop for li upfrom ls below le
                for xi upfrom xs below xe
                do (when (plusp (aref l* li))
                     (incf (aref xd* li)
                           (* (aref ld* li)
                              (let ((xi (aref x* xi)))
                                (if (< xi #.(flt 300))
                                    (sigmoid xi)
                                    #.(flt 1))))))))))))


(deflump ->exp (lump)
  ((x :initarg :x :reader x)))

(defmethod default-size ((lump ->exp))
  (size (x lump)))

(defmethod transfer-lump ((lump ->exp))
  (let ((x (x lump)))
    (assert (= (size lump) (size x)))
    (let ((x* (nodes* x))
          (l* (nodes* lump)))
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
    (let ((xd* (derivatives* x))
          (l* (nodes* lump))
          (ld* (derivatives* lump)))
      (declare (optimize (speed 3) #.*no-array-bounds-check*))
      (loop for stripe of-type index below (n-stripes* lump) do
            (with-stripes ((stripe lump ls le)
                           (stripe x xs xe))
              (loop for li upfrom ls below le
                    for xi upfrom xs below xe
                    do (incf (aref xd* li) (* (aref ld* li)
                                              (aref l* li)))))))))


(declaim (inline rough-exponential))
(defun rough-exponential (x &key signal-variance length-scale (roughness 2))
  (+ (* (abs signal-variance)
        (exp (* #.(flt -0.5)
                (if (zerop x)
                    #.(flt 0)
                    (expt (abs (/ x length-scale))
                          roughness)))))))

(declaim (inline derive-rough-exponential))
(defun derive-rough-exponential (x &key signal-variance length-scale
                                     (roughness 2))
  ;; d/dx(s^2*exp(-0.5*abs(x/l)^r)+b^2)
  (let* ((a0 (abs (/ x length-scale)))
         (a1 (if (zerop x) (flt 0) (expt a0 roughness)))
         (a2 (exp (* -0.5 a1)))
         (a3 (* #.(flt 0.5) roughness (abs signal-variance) a2)))
    (values
     ;; d/dx
     (if (zerop x)
         (flt 0)
         (- (/ (* a3 a1) x)))
     ;; d/dv
     (* (sign signal-variance) a2)
     ;; d/dl
     (/ (* a3 a1) length-scale)
     ;; d/r
     (if (zerop x)
         (flt 0)
         (* #.(flt -0.25) (abs signal-variance) a2 a1 (* 2 (log a0)))))))

(deflump ->rough-exponential (lump)
  ((x :initarg :x :reader x)
   (signal-variance :initarg :signal-variance :reader signal-variance)
   (length-scale :initarg :length-scale :reader length-scale)
   (roughness :initarg :roughness :reader roughness)))

(defmethod default-size ((lump ->rough-exponential))
  (size (x lump)))

(defmethod transfer-lump ((lump ->rough-exponential))
  (let ((x (x lump))
        (sv (signal-variance lump))
        (lsc (length-scale lump))
        (r (roughness lump)))
    (assert (= (size lump) (size x)))
    (let ((l* (nodes* lump))
          (x* (nodes* x))
          (sv* (nodes* sv))
          (lsc* (nodes* lsc))
          (r* (nodes* r)))
      (declare (optimize (speed 3) #.*no-array-bounds-check*))
      (loop for stripe of-type index below (n-stripes* lump) do
        (with-stripes ((stripe lump ls le)
                       (stripe x xs xe)
                       (stripe sv svs sve)
                       (stripe lsc lscs lsce)
                       (stripe r rs re))
          (loop for li upfrom ls below le
                for xi upfrom xs below xe
                for svi upfrom svs below sve
                for lsci upfrom lscs below lsce
                for ri upfrom rs below re
                do (setf (aref l* li)
                         (rough-exponential (aref x* xi)
                                            :signal-variance (aref sv* svi)
                                            :length-scale (aref lsc* lsci)
                                            :roughness (aref r* ri)))))))))

(defmethod derive-lump ((lump ->rough-exponential))
  (let ((x (x lump))
        (sv (signal-variance lump))
        (lsc (length-scale lump))
        (r (roughness lump)))
    (assert (= (size lump) (size x)))
    (let ((x* (nodes* x))
          (sv* (nodes* sv))
          (lsc* (nodes* lsc))
          (r* (nodes* r))
          (ld* (derivatives* lump))
          (xd* (derivatives* x))
          (svd* (derivatives* sv))
          (lscd* (derivatives* lsc))
          (rd* (derivatives* r)))
      (declare (optimize (speed 3) #.*no-array-bounds-check*))
      (loop for stripe of-type index below (n-stripes* lump) do
        (with-stripes ((stripe lump ls le)
                       (stripe x xs xe)
                       (stripe sv svs sve)
                       (stripe lsc lscs lsce)
                       (stripe r rs re))
          (loop for li upfrom ls below le
                for xi upfrom xs below xe
                for svi upfrom svs below sve
                for lsci upfrom lscs below lsce
                for ri upfrom rs below re
                do (let ((d (aref ld* li)))
                     (multiple-value-bind (dx dsv dlsc dr)
                         (derive-rough-exponential
                          (aref x* xi)
                          :signal-variance (aref sv* svi)
                          :length-scale (aref lsc* lsci)
                          :roughness (aref r* ri))
                       (incf (aref xd* xi) (* d dx))
                       (incf (aref svd* svi) (* d dsv))
                       (incf (aref lscd* lsci) (* d dlsc))
                       (incf (aref rd* ri) (* d dr))))))))))


(deflump ->periodic (lump)
  ((x :initarg :x :reader x)
   (period :initarg :period :reader period)))

(defmethod default-size ((lump ->periodic))
  (size (x lump)))

(defmethod transfer-lump ((lump ->periodic))
  (let ((x (x lump))
        (pe (period lump)))
    (assert (= (size lump) (size x)))
    (let ((l* (nodes* lump))
          (x* (nodes* x))
          (pe* (nodes* pe)))
      ;; (declare (optimize (speed 3) #.*no-array-bounds-check*))
      (loop for stripe of-type index below (n-stripes* lump) do
        (with-stripes ((stripe lump ls le)
                       (stripe x xs xe)
                       (stripe pe pes pee))
          (loop for li upfrom ls below le
                for xi upfrom xs below xe
                for pei upfrom pes below pee
                do (setf (aref l* li)
                         (sin (* pi (/ (aref x* xi)
                                       (aref pe* pei)))))))))))

(defmethod derive-lump ((lump ->periodic))
  (let ((x (x lump))
        (pe (period lump)))
    (assert (= (size lump) (size x)))
    (let ((ld* (derivatives* lump))
          (x* (nodes* x))
          (xd* (derivatives* x))
          (pe* (nodes* pe))
          (ped* (derivatives* pe)))
      ;; (declare (optimize (speed 3) #.*no-array-bounds-check*))
      (loop for stripe of-type index below (n-stripes* lump) do
        (with-stripes ((stripe lump ls le)
                       (stripe x xs xe)
                       (stripe pe pes pee))
          (loop for li upfrom ls below le
                for xi upfrom xs below xe
                for pei upfrom pes below pee
                do (let* ((xv (aref x* xi))
                          (pev (aref pe* pei))
                          (d (aref ld* li))
                          (a (cos (/ (* pi xv)
                                     pev))))
                     (incf (aref xd* xi)
                           (* d (/ (* pi a)
                                   pev)))
                     (incf (aref ped* pei)
                           (* d (- (/ (* pi xv a)
                                      (expt pev 2))))))))))))


(deflump ->ref (lump)
  ((index :initarg :index :reader index)
   (into :initarg :into :reader into)
   (drop-negative-index-p
    :initform nil
    :initarg :drop-negative-index-p
    :reader drop-negative-index-p)))

(defmethod default-size ((lump ->ref))
  (size (index lump)))

(defmethod transfer-lump ((lump ->ref))
  (let* ((l* (nodes* lump))
         (index (index lump))
         (index* (nodes* index))
         (into (into lump))
         (into* (nodes* into))
         (n (size into))
         (drop-negative-index-p (drop-negative-index-p lump)))
    (assert (= (size lump) (size index)))
    (loop for stripe of-type index below (n-stripes* lump) do
      (with-stripes ((stripe lump ls le)
                     (stripe index index-s index-e)
                     (stripe into into-s into-e))
        (declare (ignore into-e))
        (loop for li upfrom ls below le
              for index-i upfrom index-s below index-e
              do (let ((into-i (round (aref index* index-i))))
                   (assert (and (or drop-negative-index-p (<= 0 into-i))
                                (< into-i n)))
                   (when (<= 0 into-i)
                     (setf (aref l* li)
                           (aref into* (+ into-s into-i))))))))))

(defmethod derive-lump ((lump ->ref))
  (let* ((d* (derivatives* lump))
         (index (index lump))
         (index* (nodes* index))
         (into (into lump))
         (intod* (derivatives* into)))
    (assert (= (size lump) (size index)))
    (assert (typep index '->input))
    (loop for stripe of-type index below (n-stripes* lump) do
      (with-stripes ((stripe lump ls le)
                     (stripe index index-s index-e)
                     (stripe into into-s into-e))
        (declare (ignore into-e))
        (loop for li upfrom ls below le
              for index-i upfrom index-s below index-e
              do (let ((into-i (round (aref index* index-i))))
                   (when (<= 0 into-i)
                     (incf (aref intod* (+ into-s into-i))
                           (aref d* li)))))))))


(deflump ->sum-squared-error (lump)
  ((x :initarg :x :reader x)
   (y :initarg :y :reader y)))

(defmethod default-size ((lump ->sum-squared-error))
  1)

(defmethod transfer-lump ((lump ->sum-squared-error))
  (let ((x (x lump))
        (y (y lump)))
    (assert (= (size x) (size y)))
    (assert (= (n-stripes lump) (n-stripes x) (n-stripes y)))
    (let ((x* (nodes* x))
          (y* (nodes* y))
          (to* (nodes* lump)))
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
    (let ((x* (nodes* x))
          (xd* (derivatives* x))
          (y* (nodes* y))
          (yd* (derivatives* y))
          (derivatives* (derivatives* lump)))
      (declare (optimize (speed 3) #.*no-array-bounds-check*))
      (loop for stripe of-type index below (n-stripes* lump) do
        (let ((d (aref derivatives* stripe)))
          (with-stripes ((stripe x xs xe)
                         (stripe y ys ye))
            (loop for xi upfrom xs below xe
                  for yi upfrom ys below ye
                  do (incf (aref xd* xi)
                           (* d 2 (- (aref x* xi)
                                     (aref y* yi))))
                     (incf (aref yd* yi)
                           (* d 2 (- (aref y* yi)
                                     (aref x* xi)))))))))))

(deflump ->squared-error (lump)
  ((x :initarg :x :reader x)
   (y :initarg :y :reader y)))

(defmethod default-size ((lump ->squared-error))
  (size (x lump)))

(defmethod transfer-lump ((lump ->squared-error))
  (let ((x (x lump))
        (y (y lump)))
    (assert (= (size x) (size y)))
    (assert (= (n-stripes lump) (n-stripes x) (n-stripes y)))
    (let ((x* (nodes* x))
          (y* (nodes* y))
          (to* (nodes* lump)))
      (declare (optimize (speed 3) #.*no-array-bounds-check*))
      (loop for stripe of-type index below (n-stripes* lump) do
        (with-stripes ((stripe x xs xe)
                       (stripe y ys ye)
                       (stripe lump ls le))
          (loop for xi upfrom xs below xe
                for yi upfrom ys below ye
                for li upfrom ls below le
                do (setf (aref to* li)
                         (expt (- (aref x* xi)
                                  (aref y* yi))
                               2))))))))

(defmethod derive-lump ((lump ->squared-error))
  (let ((x (x lump))
        (y (y lump)))
    (assert (= (size x) (size y)))
    (assert (= (n-stripes lump) (n-stripes x) (n-stripes y)))
    (let ((x* (nodes* x))
          (xd* (derivatives* x))
          (y* (nodes* y))
          (yd* (derivatives* y))
          (derivatives* (derivatives* lump)))
      (declare (optimize (speed 3) #.*no-array-bounds-check*))
      (loop for stripe of-type index below (n-stripes* lump) do
        (with-stripes ((stripe x xs xe)
                       (stripe y ys ye)
                       (stripe lump ls le))
          (loop for xi upfrom xs below xe
                for yi upfrom ys below ye
                for li upfrom ls below le
                do (incf (aref xd* xi)
                         (* (aref derivatives* li)
                            2 (- (aref x* xi)
                                 (aref y* yi))))
                   (incf (aref yd* yi)
                         (* (aref derivatives* li)
                            2 (- (aref y* yi)
                                 (aref x* xi))))))))))
(deflump ->dropout (lump)
  ((x :initarg :x :reader x)
   (dropout
    :type (or null flt)
    :initform (flt 0.5) :initarg :dropout :reader dropout
    :documentation "If non-NIL, then in the forward pass zero out each
node in this chunk with DROPOUT probability. See Geoffrey Hinton's
'Improving neural networks by preventing co-adaptation of feature
detectors'.")))

(defmethod default-size ((lump ->dropout))
  (size (x lump)))

(defmethod transfer-lump ((lump ->dropout))
  (let ((x (x lump)))
    (assert (= (size lump) (size x)))
    (let ((x* (nodes* x))
          (l* (nodes* lump))
          (dropout (dropout lump)))
      (declare (optimize (speed 3) #.*no-array-bounds-check*)
               (type (or flt null) dropout))
      (loop for stripe of-type index below (n-stripes* lump) do
        (with-stripes ((stripe lump ls le)
                       (stripe x xs xe))
          (loop for li upfrom ls below le
                for xi upfrom xs below xe
                do (setf (aref l* li)
                         (if dropout
                             (if *in-training-p*
                                 (if (try-chance dropout)
                                     (if (zerop (random 2))
                                         least-negative-flt
                                         least-positive-flt)
                                     (aref x* xi))
                                 (* (- #.(flt 1) dropout)
                                    (aref x* xi)))
                             (aref x* xi)))))))))

(defmethod derive-lump ((lump ->dropout))
  (let ((x (x lump)))
    (assert (= (size lump) (size x)))
    (let ((xd* (derivatives* x))
          (l* (nodes* lump))
          (ld* (derivatives* lump)))
      (declare (optimize (speed 3) #.*no-array-bounds-check*))
      (loop for stripe of-type index below (n-stripes* lump) do
        (with-stripes ((stripe lump ls le)
                       (stripe x xs xe))
          (loop for li upfrom ls below le
                for xi upfrom xs below xe
                do ;; KLUDGE: I'm not sure using these two as dropout
                   ;; markers is safe.
                   (when (and (/= least-negative-flt (aref l* li))
                              (/= least-positive-flt (aref l* li)))
                     (incf (aref xd* li) (aref ld* li)))))))))


;;;; ->MAX

(deflump ->max (->normalized)
  ((group-size :initarg :group-size :reader group-size)))

(defmethod default-size ((lump ->max))
  (/ (size (x lump)) (group-size lump)))

(defmethod transfer-lump ((lump ->max))
  (let* ((x (x lump))
         (group-size (group-size lump))
         (x* (nodes* x))
         (to* (nodes* lump)))
    (declare (type index group-size))
    (loop for stripe of-type index below (n-stripes* lump) do
      (with-stripes ((stripe lump ls le)
                     (stripe x xs xe))
        (loop for li upfrom ls below le do
          (setf (aref to* li) most-negative-flt))
        (loop for xi upfrom xs below xe
              for i upfrom 0
              do (let ((li (+ ls (floor i group-size))))
                   (setf (aref to* li) (max (aref to* li)
                                            (aref x* xi)))))))))

(defmethod derive-lump ((lump ->max))
  (let* ((x (x lump))
         (group-size (group-size lump))
         (to* (nodes* lump))
         (x* (nodes* x))
         (xd* (derivatives* x))
         (d* (derivatives* lump)))
    (declare (type index group-size))
    (loop for stripe of-type index below (n-stripes* lump) do
      (with-stripes ((stripe lump ls le)
                     (stripe x xs xe))
        (declare (ignore le))
        (loop for xi upfrom xs below xe
              for i upfrom 0
              do (let ((li (+ ls (floor i group-size))))
                   (when (= (aref to* li)
                            (aref x* xi))
                     (incf (aref xd* xi)
                           (aref d* li)))))))))


;;;; ->SOFTMAX

(deflump ->softmax (->normalized)
  ())

(defmethod default-size ((lump ->softmax))
  (size (x lump)))

(defmethod transfer-lump ((lump ->softmax))
  (let* ((x (x lump))
         (group-size (group-size lump))
         (x* (nodes* x))
         (to* (nodes* lump))
         (scale (scale lump)))
    (declare (type index group-size)
             (type flt scale))
    (loop for stripe of-type index below (n-stripes* lump) do
      (with-stripes ((stripe lump ls le)
                     (stripe x xs xe))
        (loop for li upfrom ls below le
              for xi upfrom xs below xe
              for i upfrom 0
              do (when (zerop (mod i group-size))
                   (let ((max most-negative-flt)
                         (sum #.(flt 0)))
                     (declare (type flt max sum)
                              (optimize (speed 3)))
                     ;; It's more stable numerically to subtract the
                     ;; max from elements in the group before
                     ;; exponentiating.
                     (loop for xj upfrom xi below (+ xi group-size)
                           do (setq max (max max (aref x* xj))))
                     (loop for xj upfrom xi below (+ xi group-size)
                           do (incf sum (exp (- (aref x* xj) max))))
                     (setq sum (/ sum scale))
                     (loop for lj upfrom li below (+ li group-size)
                           for xj upfrom xi below (+ xi group-size)
                           for i below group-size
                           do (let ((s (/ (exp (- (aref x* xj) max)) sum)))
                                (declare (type positive-flt s))
                                (setf (aref to* lj) s))))))))))

(defmethod derive-lump ((lump ->softmax))
  (let* ((x (x lump))
         (group-size (group-size lump))
         (to* (nodes* lump))
         (xd* (derivatives* x))
         (d* (derivatives* lump))
         (scale (scale lump))
         (scale^2 (* scale scale)))
    (declare (type index group-size)
             (type flt scale))
    (loop for stripe of-type index below (n-stripes* lump) do
      (with-stripes ((stripe lump ls le)
                     (stripe x xs xe))
        (loop
          for lg of-type index upfrom ls below le by group-size
          for xg of-type index upfrom xs below xe by group-size
          do (locally
                 (declare (optimize (speed 3))))
             (loop for lj upfrom lg below (+ lg group-size)
                   for xj upfrom xg below (+ xg group-size)
                   do (let ((e^x (aref to* lj)))
                        (declare (type flt e^x))
                        (incf (aref xd* xj)
                              (* (aref d* lj)
                                 (/ (* e^x (- scale e^x))
                                    scale^2))))))))))


(deflump ->cross-entropy (lump) ())

#+nil
(defmethod transfer-lump ((lump ->cross-entropy))
  (destructuring-bind (x y) (args lump)
    (assert (= (size lump) (size x) (size y)))
    (assert (= (n-stripes lump) (n-stripes x) (n-stripes y)))
    (let ((x* (nodes* x))
          (y* (nodes* y))
          (to* (nodes* lump))
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
    (let ((x* (nodes* x))
          (xd* (derivatives* x))
          (y* (nodes* y))
          (yd* (derivatives* y))
          (derivatives* (derivatives* lump))
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


;;;; ->CROSS-ENTROPY-SOFTMAX

(deflump ->cross-entropy-softmax (lump)
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
   (class-weights
    :initform nil
    :initarg :class-weights
    :accessor class-weights
    :documentation "If non-NIL, an FLT-VECTOR of GROUP-SIZE. Useful
TARGET's distribution is different on the training and test sets. Just
set the w_i to test_frequency_i/training_frequency_i.")
   (normalized-lump :reader normalized-lump))
  (:documentation "A specialized lump that is equivalent to hooking
->EXP with NORMALIZED-LUMP and ->CROSS-ENTROPY but is numerically
stable. See <http://groups.google.com/group/comp.ai.neural-nets/msg/a7594ebea01fef04?dmode=source>

It has two parameters X and TARGET. In the transfer phase it computes
the EXP of each input node and normalizes them as if by
NORMALIZED-LUMP. These intermediate values are placed into SOFTMAX.
The value node K is nodes_k = - target_k * ln(softmax_k). Since the
sum of this is cross entropy: - sum_k target_k * ln(softmax_k), simply
plug this lump into an ->ERROR.

In the derive phase it computes the cross entropy error of the
normalized input: d(-sum_k{target_k * ln(softmax_k)})/dx_k = sum_j{
target_j * (softmax_k - KDELjk)} which is equal to softmax_k -
target_k if target sums to 1."))

(defmethod default-size ((lump ->cross-entropy-softmax))
  (size (x lump)))

(defun ensure-softmax (lump)
  (unless (and (slot-boundp lump 'softmax)
               (= (length (nodes* (x lump)))
                  (length (softmax lump))))
    (setf (slot-value lump 'softmax)
          (alexandria:copy-array (nodes* (x lump)))))
  (softmax lump))

(defmethod transfer-lump ((lump ->cross-entropy-softmax))
  (let* ((x (x lump))
         (group-size (group-size lump))
         (x* (nodes* x))
         (softmax* (ensure-softmax lump))
         (target (target lump))
         (target* (nodes* target))
         (to* (nodes* lump))
         (class-weights (class-weights lump)))
    (declare (type index group-size)
             (type flt-vector softmax*)
             (type (or flt-vector null) class-weights))
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
                    (let ((max most-negative-flt)
                          (sum #.(flt 0)))
                      (declare (type flt max sum)
                               (optimize (speed 3)))
                      ;; It's more stable numerically to subtract the
                      ;; max from elements in the group before
                      ;; exponentiating.
                      (loop for xj upfrom xi below (+ xi group-size)
                            do (setq max (max max (aref x* xj))))
                      (loop for xj upfrom xi below (+ xi group-size)
                            do (incf sum (exp (- (aref x* xj) max))))
                      (loop for lj upfrom li below (+ li group-size)
                            for xj upfrom xi below (+ xi group-size)
                            for tj upfrom ti below (+ ti group-size)
                            for i below group-size
                            do (let ((s (/ (exp (- (aref x* xj) max)) sum)))
                                 (declare (type positive-flt s))
                                 (setf (aref softmax* lj) s)
                                 (setf (aref to* lj)
                                       (- (* (if class-weights
                                                 (aref class-weights i)
                                                 #.(flt 1))
                                             (aref target* tj)
                                             (the flt (log s))))))))))))))

(defmethod derive-lump ((lump ->cross-entropy-softmax))
  (let* ((x (x lump))
         (group-size (group-size lump))
         (xd* (derivatives* x))
         (softmax (softmax lump))
         (softmax* softmax)
         (target (target lump))
         (target* (nodes* target))
         (d* (derivatives* lump))
         (class-weights (class-weights lump)))
    (declare (type index group-size)
             (type flt-vector softmax)
             (type (or flt-vector null) class-weights))
    ;; FIXME: target derivative not calculated
    (assert (typep target '->input))
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
                 for i below group-size
                 do (incf (aref xd* xj)
                          (* (aref d* li)
                             (if class-weights
                                 (aref class-weights i)
                                 #.(flt 1))
                             (aref target* ti)
                             (- (aref softmax* lj)
                                (if (= ti tj)
                                    #.(flt 1)
                                    #.(flt 0)))))))))))))

(defmethod classification-confidences ((lump ->cross-entropy-softmax) stripe)
  (with-stripes ((stripe lump start end))
    (subseq (softmax lump) start end)))

(defmethod label-distribution ((lump ->cross-entropy-softmax) stripe object)
  (declare (ignore object))
  (let ((target (target lump))
        (class-weights (class-weights lump)))
    (with-stripes ((stripe target ts te))
      (let ((d (subseq (nodes* target) ts te)))
        (if class-weights
            (map-into d #'* d class-weights)
            d)))))


;;;; Utilities

(defun collect-bpn-errors (sampler bpn &key counters-and-measurers)
  (collect-batch-errors (lambda (samples)
                          (do-executors (samples bpn)
                            (set-input samples bpn)
                            (forward-bpn bpn)))
                        sampler bpn counters-and-measurers))

;;; If the l2 norm of the incoming weight vector of the same unit is
;;; larger than L2-UPPER-BOUND then renormalize it L2-UPPER-BOUND. The
;;; list of ->ACTIVATIONS is assumed to be eventually fed to the
;;; same lump.
;;;
;;; To use it, group the activation lumps into the same GD-TRAINER and
;;; hang this function on AFTER-UPDATE-HOOK.
;;;
;;; See "Improving neural networks by preventing co-adaptation of
;;; feature detectors (Hinton, 2012)",
;;; <http://arxiv.org/pdf/1207.0580.pdf>.
(defun renormalize-activations (->activations l2-upper-bound)
  (when (and ->activations l2-upper-bound)
    (let ((n-outputs (size (first ->activations)))
          (n-normalized 0))
      ;; For each output unit get the l2 norm of all its activations.
      (dotimes (output n-outputs)
        (let ((sum (flt 0)))
          (dolist (lump ->activations)
            (let* ((weights (if (typep lump '->activation)
                                (weights lump)
                                lump))
                   (weights-size (size weights))
                   (weights* (nodes* weights))
                   (n-inputs (/ weights-size n-outputs)))
              ;; Iterate over the activations for the same output
              ;; unit. Note that the weight matrix is in column major
              ;; mode.
              (if (and (typep lump '->activation)
                       (transpose-weights-p lump))
                  ;; The weights for the same output unit are in a
                  ;; column.
                  (let ((i (* output n-inputs)))
                    (loop repeat n-inputs do
                      (incf sum (expt (aref weights* i) 2))
                      (incf i)))
                  ;; The weights for the same output unit are in a
                  ;; row.
                  (let ((i output))
                    (loop repeat n-inputs do
                      (incf sum (expt (aref weights* i) 2))
                      (incf i n-outputs))))))
          (setq sum (sqrt sum))
          ;; If the constraint is violated, iterate over the same
          ;; units and normalize them to L2-UPPER-BOUND.
          #+nil
          (format t "not renormalizing (~,5F -> ~,5F) ~S~%"
                  sum l2-upper-bound ->activations)
          (when (< l2-upper-bound sum)
            (let ((div (/ sum l2-upper-bound)))
              (incf n-normalized)
              #+nil
              (format t "renormalizing (~,5F -> ~,5F) ~S~%"
                      sum l2-upper-bound ->activations)
              (dolist (lump ->activations)
                (let* ((weights (if (typep lump '->activation)
                                    (weights lump)
                                    lump))
                       (weights-size (size weights))
                       (weights* (nodes* weights))
                       (n-inputs (/ weights-size n-outputs)))
                  (if (and (typep lump '->activation)
                           (transpose-weights-p lump))
                      (let ((i (* output n-inputs)))
                        (loop repeat n-inputs do
                          (setf (aref weights* i) (/ (aref weights* i) div))
                          (incf i)))
                      (let ((i output))
                        (loop repeat n-inputs do
                          (setf (aref weights* i) (/ (aref weights* i) div))
                          (incf i n-outputs))))))
              #+nil
              (format t "renormalized (~,5F -> ~,5F) ~S~%"
                      sum l2-upper-bound ->activations)))))
      #+nil
      (when (plusp n-normalized)
        (format t "normalized weights for ~S neurons in ~S~%"
                n-normalized ->activations)))))
