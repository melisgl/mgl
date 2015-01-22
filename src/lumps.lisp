(in-package :mgl-bp)

;;;; Lump

(defgeneric default-size (lump)
  (:method (lump)
    (or (slot-boundp lump 'size)
        (error "Can't compute size for ~S." lump))))

(defclass-now lump (clump)
  ((size :type index :initarg :size :reader size)
   (nodes
    :initform nil :type mat :reader nodes
    :documentation "The values of the nodes. All nodes have values. It
    is conceptually a N-STRIPES x SIZE matrix that can be enlarged to
    MAX-N-STRIPES x SIZE by setting N-STRIPES.")
   (derivatives
    :type mat :reader derivatives
    :documentation "Derivatives of nodes, input node derivatives are
    not calculated. A 1d array representing a matrix of the same
    dimension as NODES.")
   (default-value
    :initform 0 :initarg :default-value :type real
    :reader default-value
    :documentation "Upon creation or resize the lump's nodes get
    filled with this value.")
   (shared-with-clump
    :initform nil
    :initarg :shared-with-clump
    :reader shared-with-clump)))

(defmethod n-stripes ((lump lump))
  (let ((nodes (nodes lump)))
    (if nodes
        (mat-dimension nodes 0)
        1)))

(defmethod max-n-stripes ((lump lump))
  (let ((nodes (nodes lump)))
    (cond (nodes
           (/ (mat-max-size nodes)
              (mat-dimension nodes 1)))
          (*bpn-being-built*
           (max-n-stripes *bpn-being-built*))
          (t
           1))))

(defmethod print-object ((lump lump) stream)
  (pprint-logical-block (stream ())
    (print-unreadable-object (lump stream :type t)
      (format stream "~S ~:_~S ~:_~S" (name lump) :size
              (if (slot-boundp lump 'size)
                  (size lump)
                  :unbound))
      (let ((mgl-cube:*let-input-through-p* t))
        (format stream " ~S/~S :norm ~,5F" (n-stripes lump) (max-n-stripes lump)
                (ignore-errors (nrm2 (nodes lump)))))))
  lump)

(defmethod set-n-stripes (n-stripes (lump lump))
  (assert (<= 0 n-stripes (max-n-stripes lump)))
  (let ((dimensions (list n-stripes (mat-dimension (nodes lump) 1))))
    (reshape! (nodes lump) dimensions)
    (when (derivatives lump)
      (reshape! (derivatives lump) dimensions)))
  n-stripes)

(defmethod set-max-n-stripes (max-n-stripes (lump lump))
  (let ((shared-with-clump (shared-with-clump lump)))
    (cond (shared-with-clump
           (assert (= max-n-stripes (max-n-stripes shared-with-clump)))
           (setf (slot-value lump 'nodes) (nodes shared-with-clump))
           (when (slot-boundp shared-with-clump 'derivatives)
             (setf (slot-value lump 'derivatives)
                   (derivatives shared-with-clump))))
          (t
           (let ((old-max-n-stripes (max-n-stripes lump))
                 (size (size lump)))
             (let ((n-stripes (n-stripes lump)))
               (cond ((zerop max-n-stripes)
                      (when (nodes lump)
                        (setf (slot-value lump 'nodes) nil
                              (slot-value lump 'derivatives) nil)))
                     ((or (/= max-n-stripes old-max-n-stripes)
                          (null (nodes lump)))
                      (setf (slot-value lump 'nodes)
                            (make-mat (list n-stripes size)
                                      :max-size (* max-n-stripes size)
                                      :initial-element (default-value lump)))
                      ;; ->SOFTMAX-XE-LOSS doesn't use DERIVATIVES and
                      ;; sets this to NIL to save memory. Don't create
                      ;; it then.
                      (when (or (not (slot-boundp lump 'derivatives))
                                (not (null (slot-value lump 'derivatives))))
                        (setf (slot-value lump 'derivatives)
                              (make-mat (list n-stripes size)
                                        :max-size (* max-n-stripes
                                                     size)))))))))))
  max-n-stripes)

(defmethod initialize-instance :after ((lump lump) &key &allow-other-keys)
  (unless (slot-boundp lump 'size)
    (setf (slot-value lump 'size) (default-size lump))))

(defmethod stripe-start (stripe (lump lump))
  (* (if (stripedp lump)
         (progn
           (assert (<= 0 stripe (1- (n-stripes lump))))
           stripe)
         0)
     (size lump)))

(defmethod stripe-end (stripe (lump lump))
  (+ (stripe-start stripe lump)
     (size lump)))

(defmethod segment-weights ((lump lump))
  (nodes lump))

;;; Only weights are segments. Nothing to do for other lumps.
(defmethod map-segments (fn (lump lump)))


;;;; Data lumps

(defclass-now data-lump (lump) ())

(defclass-now ->weight (data-lump)
  ((dimensions :initarg dimensions :reader dimensions)))

(defmaker (->weight ->weight*))

(defmethod initialize-instance :around ((weight ->weight) &key dimensions size
                                        &allow-other-keys)
  (setf (slot-value weight 'dimensions)
        (if dimensions
            (alexandria:ensure-list dimensions)
            (list 1 size)))
  (unless size
    (setf (slot-value weight 'size) (reduce #'* (dimensions weight))))
  (call-next-method))

(defmethod stripedp ((lump ->weight))
  nil)

(defmethod n-stripes ((weight ->weight))
  1)

(defmethod max-n-stripes ((weight ->weight))
  1)

(defmethod set-n-stripes (n-stripes (weight ->weight))
  (declare (ignore n-stripes)))

(defmethod set-max-n-stripes (max-n-stripes (weight ->weight))
  (assert (null (shared-with-clump weight)))
  (cond ((or (null (nodes weight))
             (/= (mat-max-size (nodes weight))
                 (reduce #'* (dimensions weight))))
         (setf (slot-value weight 'nodes)
               (make-mat (dimensions weight)
                         :initial-element (default-value weight)))
         (setf (slot-value weight 'derivatives)
               (make-mat (dimensions weight)
                         :initial-element (default-value weight))))
        (t
         (setf (slot-value weight 'nodes)
               (reshape! (nodes weight) (dimensions weight)))
         (setf (slot-value weight 'derivatives)
               (reshape! (derivatives weight) (dimensions weight)))))
  max-n-stripes)

(defvar *lumps-to-copy* ())

(defun call-with-weights-copied (from-clump fn)
  (let ((*lumps-to-copy* (when from-clump (list-segments from-clump))))
    (funcall fn)))

(defmacro with-weights-copied ((from-bpn) &body body)
  "In BODY ->WEIGHT will first look up if a weight lump of the same
  name exists in FROM-BPN and return that, or else create a weight
  lump normally. If FROM-BPN is NIL, then weights are copied."
  `(call-with-weights-copied ,from-bpn (lambda () ,@body)))

(defun ->weight (&rest args)
  (assert (getf args :name) ()
          "->WEIGHT lumps must be named explicitly to allow weight ~
          sharing to work.")
  (let* ((name (getf args :name))
         (to-be-copied (find name *lumps-to-copy* :key #'name :test #'name=)))
    (cond (to-be-copied
           (when *bpn-being-built*
             (add-clump to-be-copied *bpn-being-built*))
           to-be-copied)
          (t
           (apply #'->weight* args)))))

(defmethod map-segments (fn (lump ->weight))
  (funcall fn lump))

(defmethod write-weights ((lump ->weight) stream)
  (write-mat (nodes lump) stream))

(defmethod read-weights ((lump ->weight) stream)
  (read-mat (nodes lump) stream))

(defclass-now ->constant (data-lump)
  ((default-value :initform 1)))

(defmaker ->constant)

(defmethod default-size ((lump ->constant))
  1)

(defmethod forward ((lump data-lump)))

(defmethod backward ((lump data-lump)))


(defclass-now ->dropout (lump)
  ((x :initarg :x :reader x)
   (dropout
    :type (or null real)
    :initform 0.5 :initarg :dropout :reader dropout
    :documentation "If non-NIL, then in the forward pass zero out each
    node in this chunk with DROPOUT probability. See Geoffrey Hinton's
    'Improving neural networks by preventing co-adaptation of feature
    detectors'.")
   (mask :initform nil :reader mask)))

(defmaker ->dropout)

(defmethod default-size ((lump ->dropout))
  (size (x lump)))

(defmethod forward ((lump ->dropout))
  (let ((x (x lump))
        (dropout (dropout lump)))
    (assert (= (size lump) (size x)))
    (assert (stripedp lump))
    (assert (stripedp x))
    ;; Some subclasses (->INPUT, for instance) set X to be the same as
    ;; LUMP.
    (unless (eq x lump)
      (copy! (nodes x) (nodes lump)))
    (when dropout
      (let ((dropout (coerce-to-ctype dropout)))
        (if *in-training-p*
            (let ((mask (ensure-mask lump)))
              (dropout! (nodes lump) mask dropout))
            (scal! (- 1 dropout) (nodes lump)))))))

(defun ensure-mask (lump)
  (when (dropout lump)
    (let ((x (nodes (x lump))))
      (setf (slot-value lump 'mask)
            (if (mask lump)
                (adjust! (mask lump) (mat-size x) 0)
                (make-mat (mat-size x))))))
  (mask lump))

(defun dropout! (x mask dropout-probability &key (n (mat-size x)))
  (declare (type real dropout-probability)
           (type index n))
  (uniform-random! mask)
  (if (use-cuda-p)
      (cuda-dropout-xorwow x n mask dropout-probability
                           :grid-dim (list (ceiling n 256) 1 1)
                           :block-dim (list 256 1 1))
      (lisp-dropout x (mat-displacement x) n mask (mat-displacement mask)
                    dropout-probability)))

(define-lisp-kernel (lisp-dropout)
    ((x :mat :io) (start-x index) (n index) (mask :mat :io) (start-mask index)
     (dropout-probability single-float))
  (loop for xi of-type index upfrom start-x below (the! index (+ start-x n))
        for mi of-type index upfrom start-mask
        do (cond ((< (aref mask mi) dropout-probability)
                  (setf (aref x xi) 0.0)
                  (setf (aref mask mi) 0.0))
                 (t
                  (setf (aref mask mi) 1.0)))))

(define-cuda-kernel (cuda-dropout-xorwow)
    (void ((x :mat :io) (n int) (mask :mat :io) (dropout-probability float)))
  (let ((i (+ (* block-dim-x block-idx-x) thread-idx-x)))
    (when (< i n)
      (if (< (aref mask i) dropout-probability)
          (progn
            (set (aref x i) 0.0)
            (set (aref mask i) 0.0))
          (set (aref mask i) 1.0)))))

(defmethod backward ((lump ->dropout))
  (let* ((x (x lump))
         (dropout (dropout lump))
         (xd (derivatives x))
         (ld (derivatives lump))
         (mask (mask lump)))
    (assert (= (size lump) (size x)))
    (if (not dropout)
        (axpy! 1 ld xd :n (* (size lump) (n-stripes lump)))
        (geem! 1 mask ld 1 xd))))


;;;; ->INPUT

(defclass-now ->input (->dropout data-lump)
  ((dropout :initform nil)))

(defmaker ->input)

(defmethod forward ((lump ->input))
  (setf (slot-value lump 'x) lump)
  (call-next-method))

;;; Do nothing. In clumpicular, prevent the method for ->DROPOUT from
;;; being called.
(defmethod backward ((lump ->input)))


(defclass-now ->multiply-with-gaussian (lump)
  ((x :initarg :x :reader x)
   (variance
    :type (or null real)
    :initform nil :initarg :variance :reader variance)
   (multipliers :initform nil :reader multipliers)))

(defmaker ->multiply-with-gaussian)

(defmethod default-size ((lump ->multiply-with-gaussian))
  (size (x lump)))

(defun ensure-multipliers (lump)
  (when (variance lump)
    (let ((x (nodes (x lump))))
      (setf (slot-value lump 'multipliers)
            (if (multipliers lump)
                (adjust! (multipliers lump) (mat-size x) 0)
                (make-mat (mat-size x))))))
  (multipliers lump))

(defun mgn! (l x multipliers variance)
  (gaussian-random! multipliers :mean 1 :stddev (sqrt variance))
  (geem! 1 multipliers x 0 l))

(defmethod forward ((lump ->multiply-with-gaussian))
  (if *in-training-p*
      (mgn! (nodes lump) (nodes (x lump)) (ensure-multipliers lump)
            (variance lump))
      (copy! (nodes (x lump)) (nodes lump))))

(defmethod backward ((lump ->multiply-with-gaussian))
  (let* ((x (x lump))
         (xd (derivatives x))
         (ld (derivatives lump))
         (multipliers (multipliers lump)))
    (assert (= (size lump) (size x)))
    (geem! 1 multipliers ld 1 xd)))


(defclass-now ->sample-binary (lump)
  ((x :initarg :x :reader x)
   (randoms :initform nil :reader randoms)))

(defmaker ->sample-binary)

(defmethod default-size ((lump ->sample-binary))
  (size (x lump)))

(defun ensure-randoms (lump)
  (let ((x (nodes (x lump))))
    (setf (slot-value lump 'randoms)
          (if (randoms lump)
              (adjust! (randoms lump) (mat-size x) 0)
              (make-mat (mat-size x))))))

(defmethod forward ((lump ->sample-binary))
  (let ((x (x lump)))
    (assert (= (size lump) (size x)))
    (assert (stripedp lump))
    (assert (stripedp x))
    (unless (eq x lump)
      (copy! (nodes x) (nodes lump)))
    (when *in-training-p*
      (let ((randoms (ensure-randoms lump)))
        (uniform-random! randoms)
        (.<! randoms (nodes lump))))))

(defmethod backward ((lump ->sample-binary))
  (geem! 1 (derivatives lump) (nodes lump) 1 (derivatives (x lump))))


;;;; ->SUM

(defclass-now ->sum (lump)
  ((x :initarg :x :reader x))
  (:documentation "Sum of all nodes per stripe)."))

(defmaker ->sum)

(defmethod default-size ((lump ->sum))
  1)

(defmethod forward ((lump ->sum))
  (sum! (nodes (x lump)) (nodes lump) :axis 1))

(defmethod backward ((lump ->sum))
  (with-ones (ones (list 1 (size lump)))
    (gemm! 1 (derivatives lump) ones 1 (derivatives (x lump)))))


;;;; ->ERROR

;;;; FIXME: rename to ->loss?
(defclass-now ->error (->sum)
  ((importance
    :initform nil
    :initarg :importance
    :accessor importance
    :documentation "If non-NIL, an FLT-VECTOR of n-stripes."))
  (:documentation "An error node is usually a leaf in the graph of
  lumps. Contrary to non-error leaf lumps it gets a non-zero
  derivative: 1. Error lumps have exactly one node \(in each stripe)
  whose value is computed as the sum of nodes in the X parameter
  lump."))

(defmaker ->error)

(defmethod default-size ((lump ->error))
  1)

(defmethod forward :around ((lump ->error))
  (call-next-method)
  (when (importance lump)
    (.*! (importance lump) (nodes lump))))

(defmethod backward :around ((lump ->error))
  (if (importance lump)
      (axpy! 1 (importance lump) (derivatives lump))
      (.+! 1 (derivatives lump)))
  (call-next-method))

(defmethod cost ((lump ->error))
  (assert (stripedp lump))
  (let ((sum 0)
        (sum-importances 0))
    (with-facets ((nodes ((nodes lump) 'backing-array :direction :input)))
      (let ((importances (importance lump)))
        (loop for i below (n-stripes lump)
              do (incf sum (aref nodes i))
                 (incf sum-importances (if importances
                                           (mref importances i)
                                           1)))))
    (values sum sum-importances)))


;;;; ->NORMALIZED

(defclass-now ->normalized (lump)
  ((x :initarg :x :reader x :documentation "Input comes from here.")
   (group-size :initarg :group-size :reader group-size)
   (scale
    :initform 1
    :type (or real array)
    :initarg :scale :accessor scale
    :documentation "The sum of nodes after normalization. Can be
    changed during training, for instance when clamping. If it is a
    vector then its length must be MAX-N-STRIPES which automatically
    maintained.")))

(defmaker ->normalized)

(defmethod default-size ((lump ->normalized))
  (size (x lump)))

(defmethod set-max-n-stripes (max-n-stripes (lump ->normalized))
  (call-next-method)
  (when (and (typep (scale lump) 'mat)
             (/= (max-n-stripes lump) (mat-size (scale lump))))
    (setf (scale lump) (make-mat (max-n-stripes lump))))
  max-n-stripes)

(defmethod forward ((lump ->normalized))
  (let* ((x (x lump))
         (group-size (group-size lump))
         (scale (scale lump)))
    (declare (type index group-size)
             (type (or real array) scale))
    (assert (= (size lump) (size x)))
    (assert (= (n-stripes lump) (n-stripes x)))
    (with-facets ((x* ((nodes x) 'backing-array :direction :input))
                  (to* ((nodes lump) 'backing-array :direction :output)))
      (loop for stripe of-type index below (n-stripes lump) do
        (let ((scale (if (typep scale 'real) scale (mref scale stripe))))
          (with-stripes ((stripe lump ls le)
                         (stripe x xs xe))
            (loop for li upfrom ls below le
                  for xi upfrom xs below xe
                  for i upfrom 0
                  do (when (zerop (mod i group-size))
                       (let ((sum 0))
                         (loop for j upfrom xi below (+ xi group-size)
                               do (incf sum (aref x* j)))
                         (setq sum (/ sum scale))
                         (loop for xj upfrom xi below (+ xi group-size)
                               for lj upfrom li below (+ li group-size)
                               do (setf (aref to* lj)
                                        (/ (aref x* xj) sum))))))))))))

(defmethod backward ((lump ->normalized))
  (let* ((x (x lump))
         (group-size (group-size lump))
         (scale (scale lump)))
    (declare (type index group-size)
             (type (or real array) scale))
    (assert (= (size lump) (size x)))
    (assert (= (n-stripes lump) (n-stripes x)))
    (with-facets ((x* ((nodes x) 'backing-array :direction :input))
                  (xd* ((derivatives x) 'backing-array :direction :io))
                  (ld* ((derivatives lump) 'backing-array :direction :input)))
      (loop for stripe of-type index below (n-stripes lump) do
        (let ((scale (if (typep scale 'real) scale (aref scale stripe))))
          (with-stripes ((stripe lump ls le)
                         (stripe x xs xe))
            (loop for li of-type index upfrom ls below le by group-size
                  for xi of-type index upfrom xs below xe by group-size
                  do (let ((sum 0)
                           (lie (+ li group-size))
                           (xie (+ xi group-size)))
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
                                                   sum-square)))))))))))))))


(defclass ->activation (bpn)
  ())

(defun ->activation (&key name size inputs peepholes (add-bias-p t)
                     (bpn *bpn-being-built*))
  (assert name () "NAME argument must be supplied for ->ACTIVATION.")
  (when (or add-bias-p inputs peepholes)
    (build-fnn (:name (list name :activation) :class '->activation)
      ;; To save memory, which is especially critical in a long RNN,
      ;; we make ->MM and ->* below use the NODES and DERIVATIVES of
      ;; this ->+ lump. In the forward pass they add their results to
      ;; the shared nodes (instead of setting it) and in the backward
      ;; pass all args of ->+ have the same derivative so it works out
      ;; fine.
      (shared-with-clump
       (->+ :name (list :sum name)
            :size size
            :args (if add-bias-p
                      (list (->weight :name (list :bias name) :size size))
                      ())))
      (ignored
       (progn
         (dolist (input inputs)
           (let* ((input (->clump bpn input))
                  (name (list (name input) name))
                  (w (->weight :name name :size (* size (size input)))))
             (->mm :name (list name :activation)
                   :x input :weights w
                   :shared-with-clump shared-with-clump)))
         (dolist (peephole peepholes)
           (let* ((peephole (->clump bpn peephole))
                  (name (list (name peephole) name :peephole))
                  (w (->weight :name name :size size)))
             (->* :name (list name :activation) :x peephole :y w
                  :shared-with-clump shared-with-clump)
             (assert (= size (size peephole)) ()
                     "Size of peephole input lump ~S is not ~S."
                     peephole size))))))))


;;;; Vector-matrix multiplication lump (per-stripe)

(defclass-now ->mm (lump)
  ((weights :type ->weight :initarg :weights :reader weights)
   (x :initarg :x :reader x :documentation "Input comes from here.")
   (transpose-weights-p :initform nil :initarg :transpose-weights-p
                        :reader transpose-weights-p))
  (:documentation "Perform X*WEIGHTS where X is of size M and WEIGHTS
  is a ->WEIGHT whose single stripe is taken to be of dimensions M x N
  stored in row major order. N is the size of this lump. If
  TRANSPOSE-WEIGHTS-P then WEIGHTS is N x M and X*WEIGHTS' is
  computed."))

(defmaker ->mm)

(defmethod initialize-instance :after ((lump ->mm) &key
                                       &allow-other-keys)
  (assert (= (* (size lump) (size (x lump)))
             (size (weights lump))))
  (setf (slot-value (weights lump) 'dimensions)
        (if (transpose-weights-p lump)
            (list (size lump) (size (x lump)))
            (list (size (x lump)) (size lump))))
  ;; force reshaping
  (setf (max-n-stripes (weights lump)) (max-n-stripes (weights lump))))

(defmethod default-size ((lump ->mm))
  (/ (size (weights lump))
     (size (x lump))))

(defmethod forward ((lump ->mm))
  (let* ((x (x lump))
         (weights (weights lump))
         (n-stripes (n-stripes lump))
         (nx (size x))
         (nl (/ (size weights) nx))
         (output-scale (if (shared-with-clump lump) 1 0)))
    ;; FIXEXT:
    (assert (stripedp x))
    (if (transpose-weights-p lump)
        ;; a = x*w'
        (gemm! 1 (nodes x) (nodes weights)
               output-scale (nodes lump)
               :transpose-b? t :lda nx :ldb nx :ldc nl
               :m n-stripes :n nl :k nx)
        ;; a = x*w
        (gemm! 1 (nodes x) (nodes weights)
               output-scale (nodes lump)
               :lda nx :ldb nl :ldc nl
               :m n-stripes :n nl :k nx))))

(defmethod backward ((lump ->mm))
  (let* ((x (x lump))
         (weights (weights lump))
         (n-stripes (n-stripes lump))
         (nx (size x))
         (nl (/ (size weights)
                nx))
         (x* (nodes x))
         (dx* (derivatives x))
         (w* (nodes weights))
         (dw* (derivatives weights))
         (dl* (derivatives lump)))
    (if (transpose-weights-p lump)
        ;; dx += da*w
        (gemm! 1 dl* w* 1 dx*
               :lda nl :ldb nx :ldc nx
               :m n-stripes :n nx :k nl)
        ;; dx += da*w'
        (gemm! 1 dl* w* 1 dx*
               :transpose-b? t :lda nl :ldb nl :ldc nx
               :m n-stripes :n nx :k nl))
    (if (transpose-weights-p lump)
        ;; dw += da'*x
        (gemm! 1 dl* x* 1 dw*
               :transpose-a? t
               :lda nl :ldb nx :ldc nx
               :m nl :n nx :k n-stripes)
        ;; dw += x'*da
        (gemm! 1 x* dl* 1 dw*
               :transpose-a? t
               :lda nx :ldb nl :ldc nl
               :m nx :n nl :k n-stripes))))


;;;; Node type library

(defclass-now ->rep (lump)
  ((x :initarg :x :reader x)
   (n :initarg :n :reader n)))

(defmaker ->rep)

(defmethod default-size ((lump ->rep))
  (* (n lump) (size (x lump))))

(defmethod forward ((lump ->rep))
  (let ((x (x lump)))
    ;; (assert (= (n-stripes lump) (n-stripes x)))
    (let ((n (n lump))
          (xn (size x)))
      (declare (optimize (speed 3) #.*no-array-bounds-check*)
               (type index n xn))
      (with-facets ((x* ((nodes x) 'backing-array :direction :input
                         :type flt-vector))
                    (to* ((nodes lump) 'backing-array :direction :output
                          :type flt-vector)))
        (loop for stripe of-type index below (n-stripes lump) do
          (with-stripes ((stripe x xs)
                         (stripe lump ls))
            (dotimes (i xn)
              (let ((v (aref x* (the! index (+ xs i)))))
                (loop for li of-type index upfrom (+ ls i) by xn
                      repeat n
                      do (setf (aref to* li) v))))))))))

(defmethod backward ((lump ->rep))
  (let ((x (x lump)))
    ;; (assert (= (n-stripes lump) (n-stripes x)))
    (let ((n (n lump))
          (xn (size x)))
      (declare (optimize (speed 3) #.*no-array-bounds-check*)
               (type index n xn))
      (with-facets ((xd* ((derivatives x) 'backing-array :direction :io
                          :type flt-vector))
                    (d* ((derivatives lump) 'backing-array :direction :input
                         :type flt-vector)))
        (loop for stripe of-type index below (n-stripes lump) do
          (with-stripes ((stripe x xs)
                         (stripe lump ls))
            (dotimes (i xn)
              (let ((sum (flt 0)))
                (loop for li of-type index upfrom (+ ls i) by xn
                      repeat n
                      do (incf sum (aref d* li)))
                (incf (aref xd* (+ xs i)) sum)))))))))


(defclass-now ->stretch (lump)
  ((x :initarg :x :reader x)
   (n :initarg :n :reader n)))

(defmaker ->stretch)

(defmethod default-size ((lump ->stretch))
  (* (n lump) (size (x lump))))

(defmethod forward ((lump ->stretch))
  (let ((x (x lump)))
    (assert (= (n-stripes lump) (n-stripes x)))
    (let ((n (n lump)))
      (declare (optimize (speed 3) #.*no-array-bounds-check*)
               (type index n))
      (with-facets ((x* ((nodes x) 'backing-array :direction :input
                         :type flt-vector))
                    (l* ((nodes lump) 'backing-array :direction :output
                         :type flt-vector)))
        (loop for stripe of-type index below (n-stripes lump) do
          (with-stripes ((stripe x xs xe)
                         (stripe lump ls))
            (let ((li ls))
              (loop for xi upfrom xs below xe
                    do (let ((v (aref x* xi)))
                         (loop repeat n
                               do (setf (aref l* li) v)
                                  (incf li)))))))))))

(defmethod backward ((lump ->stretch))
  (let ((x (x lump)))
    (assert (= (n-stripes lump) (n-stripes x)))
    (let ((n (n lump)))
      (declare (optimize (speed 3) #.*no-array-bounds-check*)
               (type index n))
      (with-facets ((xd* ((derivatives x) 'backing-array :direction :io
                          :type flt-vector))
                    (d* ((derivatives lump) 'backing-array :direction :input
                         :type flt-vector)))
        (loop for stripe of-type index below (n-stripes lump) do
          (with-stripes ((stripe x xs xe)
                         (stripe lump ls))
            (let ((li ls))
              (loop for xi upfrom xs below xe
                    do (let ((sum (flt 0)))
                         (loop repeat n
                               do (incf sum (aref d* li))
                                  (incf li))
                         (incf (aref xd* xi) sum))))))))))


(defclass-now ->+ (lump)
  ((args :initarg :args :reader args)))

(defmaker ->+)

(defmethod default-size ((lump ->+))
  (if (slot-boundp lump 'size)
      (slot-value lump 'size)
      (size (first (args lump)))))

(defmethod forward ((lump ->+))
  (let* ((nodes (nodes lump))
         (n-stripes (n-stripes lump)))
    (fill! 0 nodes)
    (with-ones (ones n-stripes)
      (dolist (arg (args lump))
        (cond ((stripedp arg)
               (axpy! 1 (nodes arg) nodes))
              (t
               (let ((l* (nodes lump))
                     (arg* (nodes arg)))
                 (gemm! 1 ones arg* 1 l*
                        :m n-stripes :n (size lump) :k 1
                        :lda 1 :ldb (size arg) :ldc (size lump)))))))))

(defmethod backward ((lump ->+))
  (let* ((n-stripes (n-stripes lump))
         (dl (derivatives lump)))
    (with-ones (ones n-stripes)
      (dolist (arg (args lump))
        (let ((darg (derivatives arg)))
          (cond ((stripedp arg)
                 (axpy! 1 dl darg))
                (t
                 (gemm! 1 ones dl 1 darg
                        :m 1 :n (size arg) :k n-stripes
                        :lda n-stripes :ldb (size lump)
                        :ldc (size arg)))))))))


(defclass-now ->* (lump)
  ((x :initarg :x :reader x)
   (y :initarg :y :reader y)))

(defmaker ->*)

(defmethod default-size ((lump ->*))
  (size (x lump)))

(defmethod forward ((lump ->*))
  (let* ((to (nodes lump))
         (x (x lump))
         (y (y lump))
         (output-scale (if (shared-with-clump lump) 1 0)))
    (cond ((and (stripedp x)
                (stripedp y))
           (cond ((numberp y)
                  (scal! output-scale to)
                  (axpy! y (nodes x) to))
                 (t
                  (with-shape-and-displacement ((nodes x) (mat-dimensions to))
                    (with-shape-and-displacement ((nodes y) (mat-dimensions to))
                      (geem! 1 (nodes x) (nodes y) output-scale to))))))
          ((and (not (stripedp x))
                (not (stripedp y)))
           (assert nil () "Not implemented."))
          ((not (stripedp x))
           (geerv! 1 (nodes y) (nodes x) output-scale to))
          (t
           (with-shape-and-displacement ((nodes x) (mat-dimensions to))
             (geerv! 1 (nodes x) (nodes y) output-scale to))))))

(defmethod backward ((lump ->*))
  (let* ((x (x lump))
         (y (y lump)))
    (cond ((and (stripedp x)
                (stripedp y))
           (cond ((numberp y)
                  (axpy! y (derivatives lump) (derivatives x)))
                 (t
                  (let ((dl (derivatives lump)))
                    (with-shape-and-displacement ((nodes x)
                                                  (mat-dimensions dl))
                      (with-shape-and-displacement ((nodes y)
                                                    (mat-dimensions dl))
                        (with-shape-and-displacement ((derivatives x)
                                                      (mat-dimensions dl))
                          (with-shape-and-displacement ((derivatives y)
                                                        (mat-dimensions dl))
                            (geem! 1 dl (nodes y)
                                   1 (derivatives x))
                            (geem! 1 dl (nodes x)
                                   1 (derivatives y))))))))))
          ((and (not (stripedp x))
                (not (stripedp y)))
           (assert nil () "Not implemented."))
          ((not (stripedp x))
           (with-thread-cached-mat (tmp (mat-dimensions (nodes y)))
             (geem! 1 (derivatives lump) (nodes y) 0 tmp)
             (sum! tmp (derivatives x) :axis 0 :beta 1)
             (geerv! 1 (derivatives lump) (nodes x) 1 (derivatives y))))
          (t
           (let ((dl (derivatives lump)))
             (with-shape-and-displacement ((nodes x)
                                           (mat-dimensions dl))
               (with-shape-and-displacement ((derivatives x)
                                             (mat-dimensions dl))
                 (with-thread-cached-mat (tmp (mat-dimensions (nodes x)))
                   (geem! 1 dl (nodes x) 0 tmp)
                   (sum! tmp (derivatives y) :axis 0 :beta 1)
                   (geerv! 1 dl (nodes y) 1 (derivatives x))))))))))


(defclass-now ->abs (lump)
  ((x :initarg :x :reader x)))

(defmaker ->abs)

(defmethod default-size ((lump ->abs))
  (size (x lump)))

(defmethod forward ((lump ->abs))
  (let ((x (x lump)))
    (assert (= (size lump) (size x)))
    (with-facets ((x* ((nodes x) 'backing-array :direction :input
                       :type flt-vector))
                  (l* ((nodes lump) 'backing-array :direction :output
                       :type flt-vector)))
      (declare (optimize (speed 3) #.*no-array-bounds-check*))
      (loop for stripe of-type index below (n-stripes lump) do
        (with-stripes ((stripe lump ls le)
                       (stripe x xs xe))
          (loop for li upfrom ls below le
                for xi upfrom xs below xe
                do (setf (aref l* li) (abs (aref x* xi)))))))))

(defmethod backward ((lump ->abs))
  (let ((x (x lump)))
    (assert (= (size lump) (size x)))
    (with-facets ((x* ((nodes x) 'backing-array :direction :input
                       :type flt-vector))
                  (xd* ((derivatives x) 'backing-array :direction :io
                        :type flt-vector))
                  (ld* ((derivatives lump) 'backing-array :direction :input
                        :type flt-vector)))
      (declare (optimize (speed 3) #.*no-array-bounds-check*))
      (loop for stripe of-type index below (n-stripes lump) do
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


(defclass-now ->sin (->dropout lump)
  ((dropout :initform nil)))

(defmaker ->sin)

(defmethod default-size ((lump ->sin))
  (size (x lump)))

(define-cuda-kernel (cuda-sin!)
    (void ((x :mat :input) (n int) (y :mat :output)))
  (let ((stride (* block-dim-x grid-dim-x)))
    (do ((i (+ (* block-dim-x block-idx-x) thread-idx-x)
            (+ i stride)))
        ((>= i n))
      (let ((e (aref x i)))
        (set (aref y i) (sin e))))))

(define-lisp-kernel (lisp-sin!)
    ((x :mat :input) (start-x index) (n index) (y :mat :output) (start-y index))
  (loop for xi of-type index upfrom start-x
          below (the! index (+ start-x n))
        for yi of-type index upfrom start-y
        do (setf (aref y yi)
                 (let ((xe (aref x xi)))
                   (sin xe)))))

(defun sin! (x y)
  (let ((n (mat-size x)))
    (assert (= n (mat-size y)))
    (if (use-cuda-p)
        (multiple-value-bind (block-dim grid-dim) (choose-1d-block-and-grid n 4)
          (cuda-sin! x n y :grid-dim grid-dim :block-dim block-dim))
        (lisp-sin! x (mat-displacement x) n y (mat-displacement y)))))

(defmethod forward ((lump ->sin))
  (let ((x (x lump)))
    (assert (= (size lump) (size x)))
    (sin! (nodes x) (nodes lump))))

(define-cuda-kernel (cuda-sin-derivative!)
    (void ((x :mat :input) (n int) (ld :mat :input) (xd :mat :io)))
  (let ((stride (* block-dim-x grid-dim-x)))
    (do ((i (+ (* block-dim-x block-idx-x) thread-idx-x)
            (+ i stride)))
        ((>= i n))
      (set (aref xd i) (+ (aref xd i)
                          (* (aref ld i) (cos (aref x i))))))))

(define-lisp-kernel (lisp-sin-derivative!)
    ((x :mat :input) (start-x index) (n index)
     (ld :mat :input) (start-ld index)
     (xd :mat :io) (start-xd index))
  (loop for xi of-type index upfrom start-x
          below (the! index (+ start-x n))
        for ldi of-type index upfrom start-ld
        for xdi of-type index upfrom start-xd
        do (incf (aref xd xdi) (* (aref ld ldi) (aref x xi)))))

(defun sin-derivative! (x ld xd)
  (let ((n (mat-size x)))
    (assert (= n (mat-size ld)))
    (assert (= n (mat-size xd)))
    (if (use-cuda-p)
        (multiple-value-bind (block-dim grid-dim) (choose-1d-block-and-grid n 4)
          (cuda-sin-derivative! x n ld xd :grid-dim grid-dim
                                :block-dim block-dim))
        (lisp-sin-derivative! x (mat-displacement x) n
                              ld (mat-displacement ld)
                              xd (mat-displacement xd)))))

(defmethod backward ((lump ->sin))
  (let ((x (x lump)))
    (assert (= (size lump) (size x)))
    (sin-derivative! (nodes x) (derivatives lump) (derivatives x))))


(defclass-now ->sigmoid (->dropout lump)
  ((dropout :initform nil)))

(defmaker ->sigmoid)

(defmethod default-size ((lump ->sigmoid))
  (size (x lump)))

(define-cuda-kernel (cuda-sigmoid!)
    (void ((x :mat :input) (n int) (y :mat :output)))
  (let ((stride (* block-dim-x grid-dim-x)))
    (do ((i (+ (* block-dim-x block-idx-x) thread-idx-x)
            (+ i stride)))
        ((>= i n))
      (let ((e (aref x i)))
        (set (aref y i) (/ 1.0 (+ 1.0 (exp (- e)))))))))

(define-lisp-kernel (lisp-sigmoid!)
    ((x :mat :input) (start-x index) (n index) (y :mat :output) (start-y index))
  (loop for xi of-type index upfrom start-x
          below (the! index (+ start-x n))
        for yi of-type index upfrom start-y
        do (setf (aref y yi)
                 (let ((xe (aref x xi)))
                   (/ (1+ (with-zero-on-underflow (xe) (exp (- xe)))))))))

(defun sigmoid! (x y)
  (let ((n (mat-size x)))
    (assert (= n (mat-size y)))
    (if (use-cuda-p)
        (multiple-value-bind (block-dim grid-dim) (choose-1d-block-and-grid n 4)
          (cuda-sigmoid! x n y :grid-dim grid-dim :block-dim block-dim))
        (lisp-sigmoid! x (mat-displacement x) n y (mat-displacement y)))))

(defmethod forward ((lump ->sigmoid))
  (let ((x (x lump)))
    (assert (= (size lump) (size x)))
    (sigmoid! (nodes x) (nodes lump))
    ;; Handle dropout.
    (let ((x (x lump)))
      (setf (slot-value lump 'x) lump)
      (unwind-protect
           (call-next-method)
        (setf (slot-value lump 'x) x)))))

(define-cuda-kernel (cuda-sigmoid-derivative!)
    (void ((l :mat :input) (n int) (ld :mat :input) (xd :mat :io)))
  (let ((stride (* block-dim-x grid-dim-x)))
    (do ((i (+ (* block-dim-x block-idx-x) thread-idx-x)
            (+ i stride)))
        ((>= i n))
      (let ((s (aref l i)))
        (set (aref xd i) (+ (aref xd i)
                            (* (aref ld i)
                               s (- 1.0 s))))))))

(define-lisp-kernel (lisp-sigmoid-derivative!)
    ((l :mat :input) (start-l index) (n index)
     (ld :mat :input) (start-ld index)
     (xd :mat :io) (start-xd index))
  (loop for li of-type index upfrom start-l
          below (the! index (+ start-l n))
        for ldi of-type index upfrom start-ld
        for xdi of-type index upfrom start-xd
        do (incf (aref xd xdi)
                 ;; If dropped out, S is 0 and the derivative is fine.
                 (let ((s (aref l li)))
                   (* (aref ld ldi)
                      s (- 1 s))))))

(defun sigmoid-derivative! (l ld xd)
  (let ((n (mat-size l)))
    (assert (= n (mat-size ld)))
    (assert (= n (mat-size xd)))
    (if (use-cuda-p)
        (multiple-value-bind (block-dim grid-dim) (choose-1d-block-and-grid n 4)
          (cuda-sigmoid-derivative! l n ld xd :grid-dim grid-dim
                                    :block-dim block-dim))
        (lisp-sigmoid-derivative! l (mat-displacement l) n
                                  ld (mat-displacement ld)
                                  xd (mat-displacement xd)))))

(defmethod backward ((lump ->sigmoid))
  (let ((x (x lump)))
    (assert (= (size lump) (size x)))
    (sigmoid-derivative! (nodes lump) (derivatives lump) (derivatives x))))


(defclass-now ->tanh (lump)
  ((x :initarg :x :reader x)))

(defmaker ->tanh)

(defmethod default-size ((lump ->tanh))
  (size (x lump)))

(defmethod forward ((lump ->tanh))
  (let ((x (x lump)))
    (assert (= (size lump) (size x)))
    (tanh! (nodes x) (nodes lump))))

(defun tanh! (x y)
  (let ((n (mat-size x)))
    (assert (= n (mat-size y)))
    (if (use-cuda-p)
        (multiple-value-bind (block-dim grid-dim) (choose-1d-block-and-grid n 4)
          (cuda-tanh! x n y :grid-dim grid-dim :block-dim block-dim))
        (lisp-tanh! x (mat-displacement x) n y (mat-displacement y)))))

(define-cuda-kernel (cuda-tanh!)
    (void ((x :mat :input) (n int) (y :mat :output)))
  (let ((stride (* block-dim-x grid-dim-x)))
    (do ((i (+ (* block-dim-x block-idx-x) thread-idx-x)
            (+ i stride)))
        ((>= i n))
      (let ((xe (aref x i)))
        (set (aref y i) (tanh xe))))))

(define-lisp-kernel (lisp-tanh!)
    ((x :mat :input) (start-x index) (n index) (y :mat :output) (start-y index))
  (loop for xi of-type index upfrom start-x
          below (the! index (+ start-x n))
        for yi of-type index upfrom start-y
        do (let ((xe (aref x xi)))
             (setf (aref y yi) (tanh xe)))))

(defmethod backward ((lump ->tanh))
  (let ((x (x lump)))
    (assert (= (size lump) (size x)))
    (tanh-derivative! (nodes x) (derivatives lump) (derivatives x))))

(defun tanh-derivative! (x ld xd)
  (let ((n (mat-size x)))
    (assert (= n (mat-size ld)))
    (assert (= n (mat-size xd)))
    (if (use-cuda-p)
        (multiple-value-bind (block-dim grid-dim) (choose-1d-block-and-grid n 4)
          (cuda-tanh-derivative! x n ld xd :grid-dim grid-dim
                                    :block-dim block-dim))
        (lisp-tanh-derivative! x (mat-displacement x) n
                                  ld (mat-displacement ld)
                                  xd (mat-displacement xd)))))

(define-cuda-kernel (cuda-tanh-derivative!)
    (void ((x :mat :input) (n int) (ld :mat :input) (xd :mat :io)))
  (let ((stride (* block-dim-x grid-dim-x)))
    (do ((i (+ (* block-dim-x block-idx-x) thread-idx-x)
            (+ i stride)))
        ((>= i n))
      (let ((xe (aref x i)))
        (set (aref xd i)
             (+ (aref xd i)
                (* (aref ld i)
                   (expt (/ (cosh xe)) 2.0))))))))

(define-lisp-kernel (lisp-tanh-derivative!)
    ((x :mat :input) (start-l index) (n index)
     (ld :mat :input) (start-ld index)
     (xd :mat :io) (start-xd index))
  (loop for li of-type index upfrom start-l
          below (the! index (+ start-l n))
        for ldi of-type index upfrom start-ld
        for xdi of-type index upfrom start-xd
        do (incf (aref xd xdi)
                 (let ((xe (aref x xdi)))
                   (* (aref ld ldi)
                      (expt (/ (cosh xe)) 2))))))


(defclass-now ->scaled-tanh (lump)
  ((x :initarg :x :reader x)))

(defmaker ->scaled-tanh)

(defmethod default-size ((lump ->scaled-tanh))
  (size (x lump)))

(defmethod forward ((lump ->scaled-tanh))
  (let ((x (x lump)))
    (assert (= (size lump) (size x)))
    (scaled-tanh! (nodes x) (nodes lump))))

(defun scaled-tanh! (x y)
  (let ((n (mat-size x)))
    (assert (= n (mat-size y)))
    (if (use-cuda-p)
        (multiple-value-bind (block-dim grid-dim) (choose-1d-block-and-grid n 4)
          (cuda-scaled-tanh! x n y :grid-dim grid-dim :block-dim block-dim))
        (lisp-scaled-tanh! x (mat-displacement x) n y (mat-displacement y)))))

(define-cuda-kernel (cuda-scaled-tanh!)
    (void ((x :mat :input) (n int) (y :mat :output)))
  (let ((stride (* block-dim-x grid-dim-x)))
    (do ((i (+ (* block-dim-x block-idx-x) thread-idx-x)
            (+ i stride)))
        ((>= i n))
      (let ((xe (aref x i)))
        (set (aref y i) (* 1.7159 (tanh (* 0.6666666 xe))))))))

(define-lisp-kernel (lisp-scaled-tanh!)
    ((x :mat :input) (start-x index) (n index) (y :mat :output) (start-y index))
  (loop for xi of-type index upfrom start-x
          below (the! index (+ start-x n))
        for yi of-type index upfrom start-y
        do (let ((xe (aref x xi)))
             (setf (aref y yi) (* 1.7159 (tanh (* 0.6666666 xe)))))))

(defmethod backward ((lump ->scaled-tanh))
  (let ((x (x lump)))
    (assert (= (size lump) (size x)))
    (scaled-tanh-derivative! (nodes x) (derivatives lump) (derivatives x))))

(defun scaled-tanh-derivative! (x ld xd)
  (let ((n (mat-size x)))
    (assert (= n (mat-size ld)))
    (assert (= n (mat-size xd)))
    (if (use-cuda-p)
        (multiple-value-bind (block-dim grid-dim) (choose-1d-block-and-grid n 4)
          (cuda-scaled-tanh-derivative! x n ld xd :grid-dim grid-dim
                                    :block-dim block-dim))
        (lisp-scaled-tanh-derivative! x (mat-displacement x) n
                                  ld (mat-displacement ld)
                                  xd (mat-displacement xd)))))

(define-cuda-kernel (cuda-scaled-tanh-derivative!)
    (void ((x :mat :input) (n int) (ld :mat :input) (xd :mat :io)))
  (let ((stride (* block-dim-x grid-dim-x)))
    (do ((i (+ (* block-dim-x block-idx-x) thread-idx-x)
            (+ i stride)))
        ((>= i n))
      (let ((xe (aref x i)))
        (set (aref xd i)
             (+ (aref xd i)
                (* (aref ld i)
                   1.1439333
                   (expt (/ (cosh (* 0.6666667 xe))) 2.0))))))))

(define-lisp-kernel (lisp-scaled-tanh-derivative!)
    ((x :mat :input) (start-l index) (n index)
     (ld :mat :input) (start-ld index)
     (xd :mat :io) (start-xd index))
  (loop for li of-type index upfrom start-l
          below (the! index (+ start-l n))
        for ldi of-type index upfrom start-ld
        for xdi of-type index upfrom start-xd
        do (incf (aref xd xdi)
                 (let ((xe (aref x xdi)))
                   (* (aref ld ldi)
                      1.1439333
                      (expt (/ (cosh (* 0.6666667 xe))) 2))))))


(defclass-now ->rectified (lump)
  ((x :initarg :x :reader x)
   #+nil
   (noisyp :initform nil :initarg :noisyp :accessor noisyp))
  (:documentation "max(0,x) activation function. If NOISYP then add
  normal(0,sigmoid(x)) noise to x."))

(defmaker ->rectified)

(defmethod default-size ((lump ->rectified))
  (size (x lump)))

(defmethod forward ((lump ->rectified))
  (let ((x (x lump)))
    (assert (= (size lump) (size x)))
    (rectify! (nodes x) (nodes lump))))

(defun rectify! (x y &key (n (mat-size x)))
  (assert (eq (mat-ctype x) (mat-ctype y)))
  (assert (<= n (mat-size x)))
  (assert (<= n (mat-size y)))
  (if (use-cuda-p)
      (cuda-rectify x y n
                    :grid-dim (list (ceiling n 256) 1 1)
                    :block-dim (list 256 1 1))
      (lisp-rectify x (mat-displacement x) y (mat-displacement y) n)))

(define-lisp-kernel (lisp-rectify)
    ((x :mat :input) (start-x index) (y :mat :output) (start-y index) (n index))
  (loop for xi of-type index upfrom start-x below (the! index (+ start-x n))
        for yi of-type index upfrom start-y
        do (let ((xie (aref x xi)))
             (if (< xie 0.0)
                 (setf (aref y yi) 0.0)
                 (setf (aref y yi) xie)))))

(define-cuda-kernel (cuda-rectify)
    (void ((x :mat :input) (y :mat :output) (n int)))
  (let ((i (+ (* block-dim-x block-idx-x) thread-idx-x)))
    (when (< i n)
      (let ((xi (aref x i)))
        (if (< xi 0.0)
            (set (aref y i) 0.0)
            (set (aref y i) xi))))))

(defmethod backward ((lump ->rectified))
  (let* ((x (x lump))
         (xd (derivatives x))
         (ln (nodes lump))
         (ld (derivatives lump))
         (n (mat-size (nodes lump))))
    (assert (= (size lump) (size x)))
    (if (use-cuda-p)
        (cuda-rectify-derivative xd ln ld n
                                 :grid-dim (list (ceiling n 256) 1 1)
                                 :block-dim (list 256 1 1))
        (lisp-rectify-derivative xd (mat-displacement xd)
                                 ln (mat-displacement ln)
                                 ld (mat-displacement ld)
                                 n))))

(define-lisp-kernel (lisp-rectify-derivative)
    ((xd :mat :io) (start-xd index) (l :mat :input) (start-l index)
     (ld :mat :input) (start-ld index) (n index))
  (loop for xdi of-type index upfrom start-xd below (the! index (+ start-xd n))
        for li of-type index upfrom start-l
        for ldi of-type index upfrom start-ld
        do (when (< 0.0 (aref l li))
             (incf (aref xd xdi) (aref ld ldi)))))

(define-cuda-kernel (cuda-rectify-derivative)
    (void ((xd :mat :io) (l :mat :input) (ld :mat :input) (n int)))
  (let ((i (+ (* block-dim-x block-idx-x) thread-idx-x)))
    (when (< i n)
      (when (< 0.0 (aref l i))
        (set (aref xd i) (+ (aref xd i)
                            (aref ld i)))))))


(defclass-now ->identity (lump)
  ((x :initarg :x :reader x)
   (derivative-limit :initform nil :initarg :derivative-limit
                     :reader derivative-limit)))

(defmaker ->identity)

(defmethod default-size ((lump ->identity))
  (size (x lump)))

(defmethod forward ((lump ->identity))
  (copy! (nodes (x lump)) (nodes lump)))

(defmethod backward ((lump ->identity))
  (let ((limit (derivative-limit lump)))
    (when limit
      (assert (plusp limit))
      (.min! limit (derivatives lump))
      (.max! (- limit) (derivatives lump))))
  (axpy! 1 (derivatives lump) (derivatives (x lump))))


(defclass-now ->split-sign (lump)
  ((x :initarg :x :reader x)))

(defmaker ->split-sign)

(defmethod default-size ((lump ->split-sign))
  (* 2 (size (x lump))))

(define-cuda-kernel (cuda-sign-split)
    (void ((x :mat :input) (y :mat :output) (n int)))
  (let ((i (+ (* block-dim-x block-idx-x) thread-idx-x)))
    (when (< i n)
      (let ((xi (aref x i)))
        (if (< xi 0.0)
            (progn
              (set (aref y (* 2 i)) 0.0)
              (set (aref y (+ (* 2 i) 1)) (- xi)))
            (progn
              (set (aref y (* 2 i)) xi)
              (set (aref y (+ (* 2 i) 1)) 0.0)))))))

(defun sign-split! (x y &key (n (mat-size x)))
  (assert (eq (mat-ctype x) (mat-ctype y)))
  (assert (<= n (mat-size x)))
  (assert (<= n (mat-size y)))
  (if (use-cuda-p)
      (cuda-sign-split x y n
                    :grid-dim (list (ceiling n 256) 1 1)
                    :block-dim (list 256 1 1))
      (assert nil)
      #+nil
      (with-facets ((x* (x 'backing-array :direction :input
                           :type flt-vector))
                    (y* (y 'backing-array :direction :output
                           :type flt-vector)))
        (dotimes (i n)
          (setf (aref y* i) (max #.(flt 0) (aref x* i)))))))

(defmethod forward ((lump ->split-sign))
  (let ((x (x lump)))
    (assert (= (size lump) (* 2 (size x))))
    (sign-split! (nodes x) (nodes lump))))

(define-cuda-kernel (cuda-sign-split-derivative)
    (void ((xd :mat :io) (x :mat :input) (ld :mat :input) (n int)))
  (let ((i (+ (* block-dim-x block-idx-x) thread-idx-x)))
    (when (< i n)
      (if (< (aref x i) 0.0)
          (set (aref xd i) (- (aref xd i)
                              (aref ld (+ (* 2 i) 1))))
          (set (aref xd i) (+ (aref xd i)
                              (aref ld (* 2 i))))))))

(defmethod backward ((lump ->split-sign))
  (let* ((x (x lump))
         (xd (derivatives x))
         (ld (derivatives lump))
         (n (mat-size (nodes x))))
    (assert (= (size lump) (* 2 (size x))))
    (if (use-cuda-p)
        (cuda-sign-split-derivative xd (nodes x) ld n
                                    :grid-dim (list (ceiling n 256) 1 1)
                                    :block-dim (list 256 1 1))
        (assert nil)
        #+nil
        (with-facets ((xd* ((derivatives x) 'backing-array :direction :io
                            :type flt-vector))
                      (l* ((nodes lump) 'backing-array :direction :input
                           :type flt-vector))
                      (ld* ((derivatives lump) 'backing-array :direction :input
                            :type flt-vector)))
          (declare (optimize (speed 3) #.*no-array-bounds-check*))
          (loop for stripe of-type index below (n-stripes lump) do
            (with-stripes ((stripe lump ls le)
                           (stripe x xs xe))
              (loop for li upfrom ls below le
                    for xi upfrom xs below xe
                    do (when (plusp (aref l* li))
                         (incf (aref xd* xi) (aref ld* li))))))))))


(defclass-now ->softplus (lump)
  ((x :initarg :x :reader x))
  (:documentation "log(1+exp(x))) activation function."))

(defmaker ->softplus)

(defmethod default-size ((lump ->softplus))
  (size (x lump)))

(defmethod forward ((lump ->softplus))
  (let ((x (x lump)))
    (assert (= (size lump) (size x)))
    (with-facets ((x* ((nodes x) 'backing-array :direction :input
                       :type flt-vector))
                  (l* ((nodes lump) 'backing-array :direction :output
                       :type flt-vector)))
      (declare (optimize (speed 3) #.*no-array-bounds-check*))
      (loop for stripe of-type index below (n-stripes lump) do
        (with-stripes ((stripe lump ls le)
                       (stripe x xs xe))
          (loop for li upfrom ls below le
                for xi upfrom xs below xe
                do (setf (aref l* li)
                         (let ((xi (aref x* xi)))
                           (if (< xi (flt 300))
                               (log (1+ (exp xi)))
                               xi)))))))))

(defmethod backward ((lump ->softplus))
  (let ((x (x lump)))
    (assert (= (size lump) (size x)))
    (with-facets ((x* ((nodes x) 'backing-array :direction :input
                       :type flt-vector))
                  (xd* ((derivatives x) 'backing-array :direction :io
                        :type flt-vector))
                  (l* ((nodes lump) 'backing-array :direction :input
                       :type flt-vector))
                  (ld* ((derivatives lump) 'backing-array :direction :input
                        :type flt-vector)))
      (declare (optimize (speed 3) #.*no-array-bounds-check*))
      (loop for stripe of-type index below (n-stripes lump) do
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


(defclass-now ->exp (lump)
  ((x :initarg :x :reader x)))

(defmaker ->exp)

(defmethod default-size ((lump ->exp))
  (size (x lump)))

(defmethod forward ((lump ->exp))
  (let ((x (x lump)))
    (assert (= (size lump) (size x)))
    (with-facets ((x* ((nodes x) 'backing-array :direction :input))
                  (l* ((nodes lump) 'backing-array :direction :output)))
      #+nil (declare (optimize (speed 3) #.*no-array-bounds-check*))
      (loop for stripe of-type index below (n-stripes lump) do
        (with-stripes ((stripe lump ls le)
                       (stripe x xs xe))
          (loop for li upfrom ls below le
                for xi upfrom xs below xe
                do (setf (aref l* li) (exp (aref x* xi)))))))))

(defmethod backward ((lump ->exp))
  (let ((x (x lump)))
    (assert (= (size lump) (size x)))
    (with-facets ((xd* ((derivatives x) 'backing-array :direction :io))
                  (l* ((nodes lump) 'backing-array :direction :input))
                  (ld* ((derivatives lump) 'backing-array :direction :input)))
      #+nil (declare (optimize (speed 3) #.*no-array-bounds-check*))
      (loop for stripe of-type index below (n-stripes lump) do
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

(defclass-now ->rough-exponential (lump)
  ((x :initarg :x :reader x)
   (signal-variance :initarg :signal-variance :reader signal-variance)
   (length-scale :initarg :length-scale :reader length-scale)
   (roughness :initarg :roughness :reader roughness)))

(defmaker ->rough-exponential)

(defmethod default-size ((lump ->rough-exponential))
  (size (x lump)))

(defmethod forward ((lump ->rough-exponential))
  (let ((x (x lump))
        (sv (signal-variance lump))
        (lsc (length-scale lump))
        (r (roughness lump)))
    (assert (= (size lump) (size x)))
    (with-facets ((l* ((nodes lump) 'backing-array :direction :output
                       :type flt-vector))
                  (x* ((nodes x) 'backing-array :direction :input
                       :type flt-vector))
                  (sv* ((nodes sv) 'backing-array :direction :input
                        :type flt-vector))
                  (lsc* ((nodes lsc) 'backing-array :direction :input
                         :type flt-vector))
                  (r* ((nodes r) 'backing-array :direction :input
                       :type flt-vector)))
      (declare (optimize (speed 3) #.*no-array-bounds-check*))
      (loop for stripe of-type index below (n-stripes lump) do
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

(defmethod backward ((lump ->rough-exponential))
  (let ((x (x lump))
        (sv (signal-variance lump))
        (lsc (length-scale lump))
        (r (roughness lump)))
    (assert (= (size lump) (size x)))
    (with-facets ((x* ((nodes x) 'backing-array :direction :input
                       :type flt-vector))
                  (sv* ((nodes sv) 'backing-array :direction :input
                        :type flt-vector))
                  (lsc* ((nodes lsc) 'backing-array :direction :input
                         :type flt-vector))
                  (r* ((nodes r) 'backing-array :direction :input
                       :type flt-vector))
                  (ld* ((derivatives lump) 'backing-array :direction :input
                        :type flt-vector))
                  (xd* ((derivatives x) 'backing-array :direction :io
                        :type flt-vector))
                  (svd* ((derivatives sv) 'backing-array :direction :io
                         :type flt-vector))
                  (lscd* ((derivatives lsc) 'backing-array :direction :io
                          :type flt-vector))
                  (rd* ((derivatives r) 'backing-array :direction :io
                        :type flt-vector)))
      (declare (optimize (speed 3) #.*no-array-bounds-check*))
      (loop for stripe of-type index below (n-stripes lump) do
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


(defclass-now ->periodic (lump)
  ((x :initarg :x :reader x)
   (period :initarg :period :reader period)))

(defmaker ->periodic)

(defmethod default-size ((lump ->periodic))
  (size (x lump)))

(defmethod forward ((lump ->periodic))
  (let ((x (x lump))
        (pe (period lump)))
    (assert (= (size lump) (size x)))
    (with-facets ((l* ((nodes lump) 'backing-array :direction :output
                       :type flt-vector))
                  (x* ((nodes x) 'backing-array :direction :input
                       :type flt-vector))
                  (pe* ((nodes pe) 'backing-array :direction :input
                        :type flt-vector)))
      ;; (declare (optimize (speed 3) #.*no-array-bounds-check*))
      (loop for stripe of-type index below (n-stripes lump) do
        (with-stripes ((stripe lump ls le)
                       (stripe x xs xe)
                       (stripe pe pes pee))
          (loop for li upfrom ls below le
                for xi upfrom xs below xe
                for pei upfrom pes below pee
                do (setf (aref l* li)
                         (sin (* #.(flt pi) (/ (aref x* xi)
                                               (aref pe* pei)))))))))))

(defmethod backward ((lump ->periodic))
  (let ((x (x lump))
        (pe (period lump)))
    (assert (= (size lump) (size x)))
    (with-facets ((ld* ((derivatives lump) 'backing-array :direction :input
                        :type flt-vector))
                  (x* ((nodes x) 'backing-array :direction :input
                       :type flt-vector))
                  (xd* ((derivatives x) 'backing-array :direction :io
                        :type flt-vector))
                  (pe* ((nodes pe) 'backing-array :direction :input
                        :type flt-vector))
                  (ped* ((derivatives pe) 'backing-array :direction :io
                         :type flt-vector)))
      ;; (declare (optimize (speed 3) #.*no-array-bounds-check*))
      (loop for stripe of-type index below (n-stripes lump) do
        (with-stripes ((stripe lump ls le)
                       (stripe x xs xe)
                       (stripe pe pes pee))
          (loop for li upfrom ls below le
                for xi upfrom xs below xe
                for pei upfrom pes below pee
                do (let* ((xv (aref x* xi))
                          (pev (aref pe* pei))
                          (d (aref ld* li))
                          (a (cos (/ (* #.(flt pi) xv)
                                     pev))))
                     (incf (aref xd* xi)
                           (* d (/ (* #.(flt pi) a)
                                   pev)))
                     (incf (aref ped* pei)
                           (* d (- (/ (* #.(flt pi) xv a)
                                      (expt pev 2))))))))))))


(defclass-now ->ref (lump)
  ((index :initarg :index :reader index)
   (into :initarg :into :reader into)
   (drop-negative-index-p
    :initform nil
    :initarg :drop-negative-index-p
    :reader drop-negative-index-p)))

(defmaker ->ref)

(defmethod default-size ((lump ->ref))
  (size (index lump)))

(defmethod forward ((lump ->ref))
  (let* ((index (index lump))
         (into (into lump))
         (n (size into))
         (drop-negative-index-p (drop-negative-index-p lump)))
    (with-facets ((l* ((nodes lump) 'backing-array :direction :output
                       :type flt-vector))
                  (index* ((nodes index) 'backing-array :direction :input
                           :type flt-vector))
                  (into* ((nodes into) 'backing-array :direction :input
                          :type flt-vector)))
      (assert (= (size lump) (size index)))
      (loop for stripe of-type index below (n-stripes lump) do
        (with-stripes ((stripe lump ls le)
                       (stripe index index-s index-e)
                       (stripe into into-s))
          (loop for li upfrom ls below le
                for index-i upfrom index-s below index-e
                do (let ((into-i (round (aref index* index-i))))
                     (assert (and (or drop-negative-index-p (<= 0 into-i))
                                  (< into-i n)))
                     (when (<= 0 into-i)
                       (setf (aref l* li)
                             (aref into* (+ into-s into-i)))))))))))

(defmethod backward ((lump ->ref))
  (let ((index (index lump))
        (into (into lump)))
    (assert (= (size lump) (size index)))
    (assert (typep index '->input))
    (with-facets ((d* ((derivatives lump) 'backing-array :direction :input
                       :type flt-vector))
                  (index* ((nodes index) 'backing-array :direction :input
                           :type flt-vector))
                  (intod* ((derivatives into) 'backing-array :direction :io
                           :type flt-vector)))
      (loop for stripe of-type index below (n-stripes lump) do
        (with-stripes ((stripe lump ls le)
                       (stripe index index-s index-e)
                       (stripe into into-s))
          (loop for li upfrom ls below le
                for index-i upfrom index-s below index-e
                do (let ((into-i (round (aref index* index-i))))
                     (when (<= 0 into-i)
                       (incf (aref intod* (+ into-s into-i))
                             (aref d* li))))))))))


(defclass-now ->embedding (lump)
  ((weights
    :initarg :weights :reader weights
    :documentation "A weight lump whose rows indexed by ROWS are
    copied to the output of this lump.")
   (rows
    :initarg :rows :accessor rows
    :documentation "A sequence of row indices."))
  (:documentation "If the input is one hot encoded and it's only
  multiplied with a matrix, then it may be more efficient in execution
  and in memory usage to only store the hot index and copy the
  appropriate row to the output."))

(defmaker ->embedding)

(defmethod default-size ((lump ->embedding))
  (mat-dimension (nodes (weights lump)) 1))

(defmethod forward ((lump ->embedding))
  (let* ((weights (nodes (weights lump)))
         (rows (rows lump))
         (nodes (nodes lump)))
    (let ((stripe 0))
      (map nil (lambda (row)
                 (if row
                     (with-shape-and-displacement (weights)
                       (with-shape-and-displacement (nodes)
                         (copy! (reshape-to-row-matrix! weights row)
                                (reshape-to-row-matrix! nodes stripe))))
                     (with-shape-and-displacement (nodes)
                       (fill! 0 (reshape-to-row-matrix! nodes stripe))))
                 (incf stripe))
           rows))))

(defmethod backward ((lump ->embedding))
  (let* ((wd (derivatives (weights lump)))
         (rows (rows lump))
         (ld (derivatives lump)))
    (let ((stripe 0))
      (map nil (lambda (row)
                 (when row
                   (with-shape-and-displacement (wd)
                     (with-shape-and-displacement (ld)
                       (axpy! 1 (reshape-to-row-matrix! ld stripe)
                              (reshape-to-row-matrix! wd row)))))
                 (incf stripe))
           rows))))


(defclass-now ->sum-squared-error (lump)
  ((x :initarg :x :reader x)
   (y :initarg :y :reader y)))

(defmaker ->sum-squared-error)

(defmethod default-size ((lump ->sum-squared-error))
  1)

(defmethod forward ((lump ->sum-squared-error))
  (let ((x (x lump))
        (y (y lump)))
    (assert (= (size x) (size y)))
    (assert (= (n-stripes lump) (n-stripes x) (n-stripes y)))
    (with-facets ((x* ((nodes x) 'backing-array :direction :input))
                  (y* ((nodes y) 'backing-array :direction :input))
                  (to* ((nodes lump) 'backing-array :direction :output)))
      (loop for stripe of-type index below (n-stripes lump) do
        (with-stripes ((stripe x xs xe)
                       (stripe y ys ye))
          (setf (aref to* stripe)
                (let ((sum 0))
                  (loop for xi upfrom xs below xe
                        for yi upfrom ys below ye
                        do (incf sum (expt (- (aref x* xi)
                                              (aref y* yi))
                                           2)))
                  sum)))))))

(defmethod backward ((lump ->sum-squared-error))
  (let ((x (x lump))
        (y (y lump)))
    (assert (= (size x) (size y)))
    (assert (= (n-stripes lump) (n-stripes x) (n-stripes y)))
    (with-facets ((x* ((nodes x) 'backing-array :direction :input))
                  (xd* ((derivatives x) 'backing-array :direction :io))
                  (y* ((nodes y) 'backing-array :direction :input))
                  (yd* ((derivatives y) 'backing-array :direction :io))
                  (ld* ((derivatives lump) 'backing-array :direction :input)))
      (loop for stripe of-type index below (n-stripes lump) do
        (let ((d (aref ld* stripe)))
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


(defclass-now ->squared-error (lump)
  ((x :initarg :x :reader x)
   (y :initarg :y :reader y)))

(defmaker ->squared-error)

(defmethod default-size ((lump ->squared-error))
  (size (x lump)))

(defmethod forward ((lump ->squared-error))
  (let ((x (x lump))
        (y (y lump)))
    (assert (= (size x) (size y)))
    (assert (= (n-stripes lump) (n-stripes x) (n-stripes y)))
    (with-facets ((x* ((nodes x) 'backing-array :direction :input
                       :type flt-vector))
                  (y* ((nodes y) 'backing-array :direction :input
                       :type flt-vector))
                  (to* ((nodes lump) 'backing-array :direction :output
                        :type flt-vector)))
      (declare (optimize (speed 3) #.*no-array-bounds-check*))
      (loop for stripe of-type index below (n-stripes lump) do
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

(defmethod backward ((lump ->squared-error))
  (let ((x (x lump))
        (y (y lump)))
    (assert (= (size x) (size y)))
    (assert (= (n-stripes lump) (n-stripes x) (n-stripes y)))
    (with-facets ((x* ((nodes x) 'backing-array :direction :input
                       :type flt-vector))
                  (xd* ((derivatives x) 'backing-array :direction :io
                        :type flt-vector))
                  (y* ((nodes y) 'backing-array :direction :input
                       :type flt-vector))
                  (yd* ((derivatives y) 'backing-array :direction :io
                        :type flt-vector))
                  (ld* ((derivatives lump) 'backing-array :direction :input
                        :type flt-vector)))
      (declare (optimize (speed 3) #.*no-array-bounds-check*))
      (loop for stripe of-type index below (n-stripes lump) do
        (with-stripes ((stripe x xs xe)
                       (stripe y ys ye)
                       (stripe lump ls le))
          (loop for xi upfrom xs below xe
                for yi upfrom ys below ye
                for li upfrom ls below le
                do (incf (aref xd* xi)
                         (* (aref ld* li)
                            2 (- (aref x* xi)
                                 (aref y* yi))))
                   (incf (aref yd* yi)
                         (* (aref ld* li)
                            2 (- (aref y* yi)
                                 (aref x* xi))))))))))


;;;; ->MAX

(defclass-now ->max (lump)
  ((x :initarg :x :reader x :documentation "Input comes from here.")
   (group-size :initarg :group-size :reader group-size)))

(defmaker ->max)

(defmethod default-size ((lump ->max))
  (/ (size (x lump)) (group-size lump)))

(defmethod forward ((lump ->max))
  (let* ((x (x lump))
         (group-size (group-size lump))
         (n (mat-size (nodes lump))))
    (if (use-cuda-p)
        (cuda-max group-size (nodes x) n (nodes lump)
                  :grid-dim (list (ceiling n 256) 1 1)
                  :block-dim (list 256 1 1))
        (lisp-max group-size (nodes x) n (nodes lump)))))

(define-lisp-kernel (lisp-max)
    ((group-size index) (x :mat :input) (n index) (y :mat :output))
  (locally (declare (optimize (speed 1)))
    (loop for group-start of-type index upfrom 0 below n by group-size
          for yi upfrom 0
          do (let ((group-end (the! index (+ group-start group-size)))
                   (max most-positive-single-float))
               (declare (type single-float max)
                        (optimize (speed 3)))
               (loop for i upfrom group-start below group-end
                     do (setq max (max max (aref x i))))
               (setf (aref y yi) max)))))

(define-cuda-kernel (cuda-max)
    (void ((group-size int) (x :mat :input) (n int) (y :mat :output)))
  (let ((k (+ (* block-dim-x block-idx-x) thread-idx-x)))
    (when (< k n)
      (let* ((i (* group-size k))
             (max (aref x i)))
        (do ((a 1 (+ a 1)))
            ((>= a group-size))
          (let ((xe (aref x (+ i a))))
            (when (< xe max)
              (set max xe))))
        (set (aref y k) max)))))

(defmethod backward ((lump ->max))
  (let* ((x (x lump))
         (group-size (group-size lump))
         (n (mat-size (nodes lump))))
    (if (use-cuda-p)
        (cuda-max-derivative group-size (nodes x) n (nodes lump)
                             (derivatives lump) (derivatives x)
                             :grid-dim (list (ceiling n 256) 1 1)
                             :block-dim (list 256 1 1))
        (lisp-max-derivative group-size (nodes x) n (nodes lump)
                             (derivatives lump) (derivatives x)))))

(define-lisp-kernel (lisp-max-derivative)
    ((group-size index) (x :mat :input) (n index) (l :mat :input)
     (ld :mat :input) (xd :mat :io))
  (locally (declare (optimize (speed 1)))
    (loop for group-start of-type index upfrom 0 below n by group-size
          for li upfrom 0
          do (let ((group-end (the! index (+ group-start group-size)))
                   (max (aref l li)))
               (declare (type single-float max)
                        (optimize (speed 3)))
               (loop for i upfrom group-start below group-end
                     do (when (= max (aref x i))
                          (incf (aref xd i) (aref ld li))))))))

(define-cuda-kernel (cuda-max-derivative)
    (void ((group-size int) (x :mat :input) (n int) (l :mat :input)
           (ld :mat :input) (xd :mat :io)))
  (let ((k (+ (* block-dim-x block-idx-x) thread-idx-x)))
    (when (< k n)
      (let ((i (* group-size k)))
        (do ((a 0 (+ a 1)))
            ((>= a group-size))
          (let ((ia (+ i a)))
            (when (= (aref x ia) (aref l k))
              (set (aref xd ia)
                   (+ (aref xd ia)
                      (aref ld k))))))))))


;;;; ->MIN

(defclass-now ->min (lump)
  ((x :initarg :x :reader x :documentation "Input comes from here.")
   (group-size :initarg :group-size :reader group-size)))

(defmaker ->min)

(defmethod default-size ((lump ->min))
  (/ (size (x lump)) (group-size lump)))

(defmethod forward ((lump ->min))
  (let* ((x (x lump))
         (group-size (group-size lump))
         (n (mat-size (nodes lump))))
    (if (use-cuda-p)
        (cuda-min group-size (nodes x) n (nodes lump)
                  :grid-dim (list (ceiling n 256) 1 1)
                  :block-dim (list 256 1 1))
        (lisp-min group-size (nodes x) n (nodes lump)))))

(define-lisp-kernel (lisp-min)
    ((group-size index) (x :mat :input) (n index) (y :mat :output))
  (locally (declare (optimize (speed 1)))
    (loop for group-start of-type index upfrom 0 below n by group-size
          for yi upfrom 0
          do (let ((group-end (the! index (+ group-start group-size)))
                   (min most-negative-single-float))
               (declare (type single-float min)
                        (optimize (speed 3)))
               (loop for i upfrom group-start below group-end
                     do (setq min (min min (aref x i))))
               (setf (aref y yi) min)))))

(define-cuda-kernel (cuda-min)
    (void ((group-size int) (x :mat :input) (n int) (y :mat :output)))
  (let ((k (+ (* block-dim-x block-idx-x) thread-idx-x)))
    (when (< k n)
      (let* ((i (* group-size k))
             (min (aref x i)))
        (do ((a 1 (+ a 1)))
            ((>= a group-size))
          (let ((xe (aref x (+ i a))))
            (when (< xe min)
              (set min xe))))
        (set (aref y k) min)))))

(defmethod backward ((lump ->min))
  (let* ((x (x lump))
         (group-size (group-size lump))
         (n (mat-size (nodes lump))))
    (if (use-cuda-p)
        (cuda-min-derivative group-size (nodes x) n (nodes lump)
                             (derivatives lump) (derivatives x)
                             :grid-dim (list (ceiling n 256) 1 1)
                             :block-dim (list 256 1 1))
        (lisp-min-derivative group-size (nodes x) n (nodes lump)
                             (derivatives lump) (derivatives x)))))

(define-lisp-kernel (lisp-min-derivative)
    ((group-size index) (x :mat :input) (n index) (l :mat :input)
     (ld :mat :input) (xd :mat :io))
  (locally (declare (optimize (speed 1)))
    (loop for group-start of-type index upfrom 0 below n by group-size
          for li upfrom 0
          do (let ((group-end (the! index (+ group-start group-size)))
                   (min (aref l li)))
               (declare (type single-float min)
                        (optimize (speed 3)))
               (loop for i upfrom group-start below group-end
                     do (when (= min (aref x i))
                          (incf (aref xd i) (aref ld li))))))))

(define-cuda-kernel (cuda-min-derivative)
    (void ((group-size int) (x :mat :input) (n int) (l :mat :input)
           (ld :mat :input) (xd :mat :io)))
  (let ((k (+ (* block-dim-x block-idx-x) thread-idx-x)))
    (when (< k n)
      (let ((i (* group-size k)))
        (do ((a 0 (+ a 1)))
            ((>= a group-size))
          (let ((ia (+ i a)))
            (when (= (aref x ia) (aref l k))
              (set (aref xd ia)
                   (+ (aref xd ia)
                      (aref ld k))))))))))


;;;; ->MAX-CHANNEL

(defclass-now ->max-channel (lump)
  ((x :initarg :x :reader x :documentation "Input comes from here.")
   (group-size :initarg :group-size :reader group-size)))

(defmaker ->max-channel)

(defmethod default-size ((lump ->max-channel))
  (size (x lump)))

(defmethod forward ((lump ->max-channel))
  (let* ((x (x lump))
         (group-size (group-size lump)))
    (declare (type index group-size))
    (if (use-cuda-p)
        (let ((n (/ (mat-size (nodes lump)) group-size)))
          (cuda-max-channel group-size (nodes x) n (nodes lump)
                            :grid-dim (list (ceiling n 256) 1 1)
                            :block-dim (list 256 1 1)))
        (lisp-max-channel
         group-size (nodes x) (mat-displacement (nodes x)) (mat-size (nodes x))
         (nodes lump) (mat-displacement (nodes lump))))))

(define-lisp-kernel (lisp-max-channel)
    ((group-size index)
     (x :mat :input) (start-x index) (n index)
     (y :mat :output) (start-y index))
  (loop for xi of-type index upfrom start-x
          below (the! index (+ start-x n)) by group-size
        for yi of-type index upfrom start-y by group-size
        do (let ((max (aref x xi)))
             (do ((a 1 (+ a 1)))
                 ((>= a group-size))
               (let ((xe (aref x (+ xi a))))
                 (when (< max xe)
                   (setq max xe))))
             (do ((a 0 (+ a 1)))
                 ((>= a group-size))
               (let ((xe (aref x (+ xi a))))
                 (if (= max xe)
                     (setf (aref y (+ yi a)) xe)
                     (setf (aref y (+ yi a)) 0.0)))))))

(define-cuda-kernel (cuda-max-channel)
    (void ((group-size int) (x :mat :input) (n int) (y :mat :output)))
  (let ((k (+ (* block-dim-x block-idx-x) thread-idx-x)))
    (when (< k n)
      (let* ((i (* group-size k))
             (max (aref x i)))
        (do ((a 1 (+ a 1)))
            ((>= a group-size))
          (let ((xe (aref x (+ i a))))
            (when (< max xe)
              (set max xe))))
        (do ((a 0 (+ a 1)))
            ((>= a group-size))
          (let ((xe (aref x (+ i a))))
            (if (= max xe)
                (set (aref y (+ i a)) xe)
                (set (aref y (+ i a)) 0.0))))))))

(defmethod backward ((lump ->max-channel))
  (let* ((x (x lump))
         (group-size (group-size lump)))
    (declare (type index group-size))
    (if (use-cuda-p)
        (let ((n (/ (mat-size (nodes lump)) group-size)))
          (cuda-max-channel-derivative group-size (nodes x) n
                                       (derivatives lump) (derivatives x)
                                       :grid-dim (list (ceiling n 256) 1 1)
                                       :block-dim (list 256 1 1)))
        (lisp-max-channel-derivative
         group-size (nodes x) (mat-displacement (nodes x)) (mat-size (nodes x))
         (derivatives lump) (mat-displacement (derivatives lump))
         (derivatives x) (mat-displacement (derivatives x))))))

(define-lisp-kernel (lisp-max-channel-derivative)
    ((group-size index)
     (x :mat :input) (start-x index) (n index)
     (ld :mat :input) (start-ld index)
     (xd :mat :io) (start-xd index))
  (loop for xi of-type index upfrom start-x
          below (the! index (+ start-x n)) by group-size
        for ldi of-type index upfrom start-ld by group-size
        for xdi of-type index upfrom start-xd by group-size
        do (let ((max (aref x xi)))
             (do ((a 1 (+ a 1)))
                 ((>= a group-size))
               (let ((xe (aref x (+ xi a))))
                 (when (< max xe)
                   (setq max xe))))
             (do ((a 0 (+ a 1)))
                 ((>= a group-size))
               (let ((xe (aref x (+ xi a))))
                 (when (= max xe)
                   (setf (aref xd (+ xdi a))
                         (+ (aref xd (+ xdi a))
                            (aref ld (+ ldi a))))))))))

(define-cuda-kernel (cuda-max-channel-derivative)
    (void ((group-size int) (x :mat :input) (n int)
           (ld :mat :input) (xd :mat :io)))
  (let ((k (+ (* block-dim-x block-idx-x) thread-idx-x)))
    (when (< k n)
      (let* ((i (* group-size k))
             (max (aref x i)))
        (do ((a 1 (+ a 1)))
            ((>= a group-size))
          (let ((xe (aref x (+ i a))))
            (when (< max xe)
              (set max xe))))
        (do ((a 0 (+ a 1)))
            ((>= a group-size))
          (let* ((ia (+ i a))
                 (xe (aref x ia)))
            (when (= max xe)
              (set (aref xd ia)
                   (+ (aref xd ia)
                      (aref ld ia))))))))))


;;;; ->SOFTMAX

(defclass-now ->softmax (->normalized)
  ())

(defmaker ->softmax)

(defmethod default-size ((lump ->softmax))
  (size (x lump)))

(defmethod forward ((lump ->softmax))
  (let* ((x (x lump))
         (group-size (group-size lump))
         (scale (scale lump)))
    (declare (type index group-size)
             (type flt scale))
    (with-facets ((x* ((nodes x) 'backing-array :direction :input
                       :type flt-vector))
                  (to* ((nodes lump) 'backing-array :direction :output
                        :type flt-vector)))
      (loop for stripe of-type index below (n-stripes lump) do
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
                                  (setf (aref to* lj) s)))))))))))

(defmethod backward ((lump ->softmax))
  (let* ((x (x lump))
         (group-size (group-size lump))
         (scale (scale lump)))
    (declare (type index group-size)
             (type flt scale))
    (assert (= (flt 1) scale))
    (with-facets ((to* ((nodes lump) 'backing-array :direction :output
                        :type flt-vector))
                  (xd* ((derivatives x) 'backing-array :direction :io
                        :type flt-vector))
                  (d* ((derivatives lump) 'backing-array :direction :input
                       :type flt-vector)))
      (loop for stripe of-type index below (n-stripes lump) do
        (with-stripes ((stripe lump ls le)
                       (stripe x xs xe))
          (loop
            for lg of-type index upfrom ls below le by group-size
            for xg of-type index upfrom xs below xe by group-size
            do (locally
                   (declare (optimize (speed 3))))
               ;; d(softmax(x)_i)/dx_k = softmax(x)_i * (K_ik - softmax(x)_k)
               (dotimes (i group-size)
                 (let* ((li (+ lg i))
                        (dli (aref d* li))
                        (li* (aref to* li)))
                   (dotimes (k group-size)
                     (let ((xk (+ xg k)))
                       (incf (aref xd* xk)
                             (* dli
                                (if (= i k)
                                    (* li* (- #.(flt 1) li*))
                                    (* li* (- (aref to* (+ lg k)))))))))))))))))


;;;; ->SOFTMAX-XE-LOSS

;;; FIXME: DERIVATIVES is unused currently. Save memory by not
;;; creating it?
(defclass-now ->softmax-xe-loss (lump)
  ((group-size :initarg :group-size :reader group-size)
   (x :initarg :x :reader x :documentation "This is the input lump.")
   (target
    :initform nil :initarg :target :accessor target
    :documentation "A MAT of the same size as the input lump `X`. The
    cross entropy is calculated based on target and actual (`X`)
    values: `-sum_{k}target_k*ln(x_k)`.

    If the target is very sparse, this can be a sequence of batch size
    length that contains the index value pairs of non-zero entries:

        (;; first instance in batch has to non-zero targets
         (;; class 10 has 30% expected probability
          (10 . 0.3)
          ;; class 2 has 70% expected probability
          (2 .  0.7))
         ;; second instance in batch puts 100% on class 7
         7
         ;; more instance in the batch follow
         ...)

    Actually, in the rare case where GROUP-SIZE is not SIZE (i.e.
    there are several softmax normalization groups for every example),
    the length of the above target sequence is BATCH-SIZE * N-GROUPS.
    Indices are always relative to the start of the group.

    If GROUP-SIZE is large (for example, in neural language models
    with a huge number of words), this can make things go much faster,
    because calculation of the derivative is no longer quadratic.")
   ;; Make sure SET-MAX-N-STRIPES doesn't create DERIVATIVES. We don't
   ;; use it anyway.
   (derivatives :initform nil))
  (:documentation "A specialized lump that is equivalent to hooking
  ->EXP with NORMALIZED-LUMP and ->CROSS-ENTROPY but is numerically
  stable. See
  <http://groups.google.com/group/comp.ai.neural-nets/msg/a7594ebea01fef04?dmode=source>

  It has two parameters X and TARGET. In the forward phase it
  computes the EXP of each input node and normalizes them as if by
  NORMALIZED-LUMP. These intermediate values are placed into SOFTMAX.
  The value node K is nodes_k = - target_k * ln(softmax_k). Since the
  sum of this is cross entropy: - sum_k target_k * ln(softmax_k),
  simply plug this lump into an ->ERROR.

  In the derive phase it computes the cross entropy error of the
  normalized input: d(-sum_k{target_k * ln(softmax_k)})/dx_k = sum_j{
  target_j * (softmax_k - KDELjk)} which is equal to softmax_k -
  target_k if target sums to 1."))

(defmaker ->softmax-xe-loss)

(defmethod default-size ((lump ->softmax-xe-loss))
  (size (x lump)))

(defmethod initialize-instance :after ((lump ->softmax-xe-loss)
                                       &key &allow-other-keys)
  (unless (slot-boundp lump 'group-size)
    (setf (slot-value lump 'group-size) (size lump))))

(defun ensure-softmax-target-matrix (lump n)
  (setf (target lump)
        (if (typep (target lump) 'mat)
            (adjust! (target lump) (list n (size lump)) 0)
            (make-mat (list n (size lump))
                      :max-size (* (max-n-stripes lump) (size lump))))))

(eval-when (:compile-toplevel :load-toplevel :execute)
  (defparameter *n-softmax-threads* 128))

(defmethod forward ((lump ->softmax-xe-loss))
  (let* ((x (nodes (x lump)))
         (group-size (group-size lump))
         (softmax (nodes lump))
         (n (* (n-stripes lump) (size lump))))
    (if (use-cuda-p)
        (cuda-softmax-xe group-size x n softmax
                         :grid-dim (list (/ n group-size) 1 1)
                         :block-dim (list *n-softmax-threads* 1 1))
        (lisp-softmax-xe group-size x n softmax))))

(define-lisp-kernel (lisp-softmax-xe)
    ((group-size index) (x :mat :input) (n index) (softmax :mat :output))
  (locally (declare (optimize (speed 1)))
    (loop for group-start of-type index upfrom 0 below n by group-size
          do (let ((group-end (the! index (+ group-start group-size)))
                   (max most-negative-single-float)
                   (sum 0.0))
               (declare (type single-float max sum)
                        (optimize (speed 3)))
               ;; It's more stable numerically to subtract the
               ;; max from elements in the group before
               ;; exponentiating.
               (loop for i upfrom group-start below group-end
                     do (setq max (max max (aref x i))))
               (loop for i upfrom group-start below group-end
                     do (incf sum (exp (- (aref x i) max))))
               (loop for i upfrom group-start below group-end
                     for class-i below group-size
                     do (let ((s (/ (exp (- (aref x i) max))
                                    sum)))
                          (setf (aref softmax i) s)))))))

;;; This implementation was translated from SoftMax.cu in cunn.
(define-cuda-kernel (cuda-softmax-xe)
    (void ((group-size int) (input :mat :input) (n int) (output :mat :output)))
  (with-shared-memory ((buffer float #.(1+ *n-softmax-threads*)))
    (let* ((k block-idx-x)
           (base-index (* k group-size))
           (i-start (+ base-index thread-idx-x))
           (i-end (+ base-index group-size))
           (i-step block-dim-x))
      (set (aref buffer thread-idx-x) #.most-negative-single-float)
      ;; start working on the max
      (do ((i i-start (+ i i-step)))
          ((<= i-end i))
        (let ((z (aref input i)))
          (when (< (aref buffer thread-idx-x) z)
            (set (aref buffer thread-idx-x) z))))
      (syncthreads)
      ;; reduce
      (when (= thread-idx-x 0)
        (let ((max-k #.most-negative-single-float))
          (do ((i 0 (1+ i)))
              ((<= block-dim-x i))
            (when (< max-k (aref buffer i))
              (set max-k (aref buffer i))))
          (set (aref buffer #.*n-softmax-threads*) max-k)))
      (syncthreads)
      ;; start working on the sum
      (let ((max-k (aref buffer #.*n-softmax-threads*)))
        (set (aref buffer thread-idx-x) 0.0)
        (do ((i i-start (+ i i-step)))
            ((<= i-end i))
          (let ((z (exp (- (aref input i) max-k))))
            (set (aref buffer thread-idx-x)
                 (+ (aref buffer thread-idx-x) z))
            (set (aref output i) z))))
      (syncthreads)
      ;; reduce
      (when (= thread-idx-x 0)
        (let ((sum-k 0.0))
          (do ((i 0 (1+ i)))
              ((<= block-dim-x i))
            (set sum-k (+ sum-k (aref buffer i))))
          (set (aref buffer #.*n-softmax-threads*) sum-k)))
      (syncthreads)
      ;; softmax
      (let ((sum-k (aref buffer #.*n-softmax-threads*)))
        (do ((i i-start (+ i i-step)))
            ((<= i-end i))
          (set (aref output i) (/ (aref output i) sum-k)))))))

(defmacro do-sparse-targets (((group-start target-index target-value)
                              targets group-size)
                             &body body)
  (alexandria:once-only (group-size)
    `(flet ((foo (,target-index ,target-value ,group-start)
              (assert (and (<= 0 ,target-index) (< ,target-index ,group-size))
                      () "Sparse target ~S is not in [0, ~S)."
                      ,target-index ,group-size)
              (let ((,target-value (coerce-to-ctype ,target-value)))
                ,@body)))
       (let ((,group-start 0))
         (map nil (lambda (targets)
                    (if (typep targets 'index)
                        (foo targets 1.0 group-start)
                        (loop for entry in targets
                              do (destructuring-bind (index . value) entry
                                   (foo index value group-start))))
                    (incf group-start ,group-size))
              ,targets)))))

(defmethod backward ((lump ->softmax-xe-loss))
  ;; FIXME: We ignore the derivatives of the softmax itself for now,
  ;; only do the backprop for XE.
  (let* ((x (x lump))
         (dx (derivatives x))
         (group-size (group-size lump))
         (softmax (nodes lump))
         (target (target lump))
         (n (* (n-stripes lump) (size lump))))
    (if (typep target 'sequence)
        (do-sparse-targets ((group-start target-index target-value)
                            target group-size)
          (if (use-cuda-p)
              (multiple-value-bind (block-dim grid-dim)
                  (choose-1d-block-and-grid group-size 4)
                (cuda-softmax-xe-derivative/sparse
                 group-start group-size
                 dx softmax (+ group-start target-index) target-value
                 :grid-dim grid-dim :block-dim block-dim))
              (lisp-softmax-xe-derivative/sparse
               group-start group-size dx softmax
               (+ group-start target-index) target-value)))
        (if (use-cuda-p)
            (multiple-value-bind (block-dim grid-dim)
                (choose-1d-block-and-grid n 4)
              (cuda-softmax-xe-derivative
               group-size dx n target softmax
               :grid-dim grid-dim :block-dim block-dim))
            (lisp-softmax-xe-derivative
             group-size dx n target softmax)))))

(define-lisp-kernel (lisp-softmax-xe-derivative)
    ((group-size index) (xd :mat :io) (n index)
     (target :mat :input) (softmax :mat :input))
  (locally (declare (optimize (speed 1)))
    (loop for group-start of-type index upfrom 0 below n by group-size
          do ;; calc d(XENT)/dx_j
             (let ((group-end (the! index (+ group-start group-size))))
               (loop
                 for j of-type index upfrom group-start below group-end
                 do ;; Since we cannot be sure that x_i sum to one and
                    ;; all elements D are equal (which is the case if
                    ;; this is hooked into an error node directly), we
                    ;; cannot take the shortcut of d(XENT)/dx_j =
                    ;; softmax_j - target_j. Instead, we must
                    ;; calculate sum_i{target_i*(softmax_j-KDEL_ij)}
                    ;; where KDEL is the Kronecker delta.
                    (locally (declare (optimize (speed 3)))
                      (loop
                        for i upfrom group-start below group-end
                        do (incf (aref xd j)
                                 (* (aref target i)
                                    (- (aref softmax j)
                                       (if (= i j)
                                           1.0
                                           0.0)))))))))))

(define-cuda-kernel (cuda-softmax-xe-derivative)
    (void ((group-size int) (xd :mat :io) (n int)
           (target :mat :input) (softmax :mat :input)))
  (let ((stride (* block-dim-x grid-dim-x)))
    (do ((ia (+ (* block-dim-x block-idx-x) thread-idx-x)
             (+ ia stride)))
        ((>= ia n))
      (let* ((a (mod ia group-size))
             (i (- ia a))
             (softmax-ia (aref softmax ia))
             (sum 0.0))
        (do ((b 0 (+ b 1)))
            ((>= b group-size))
          (let* ((ib (+ i b))
                 (target-ib (aref target ib)))
            (when (/= 0.0 target-ib)
              (set sum (+ sum (* target-ib
                                 (- softmax-ia
                                    (if (= a b)
                                        1.0
                                        0.0))))))))
        (set (aref xd ia) (+ (aref xd ia) sum))))))

(define-lisp-kernel (lisp-softmax-xe-derivative/sparse)
    ((group-start index) (group-size index) (xd :mat :io) (softmax :mat :input)
     (target-index index) (target-value single-float))
  (locally (declare (optimize (speed 1)))
    (let ((group-end (the! index (+ group-start group-size))))
      (loop
        for j of-type index upfrom group-start below group-end
        do (locally (declare (optimize (speed 3)))
             (incf (aref xd j)
                   (* target-value
                      (- (aref softmax j)
                         (if (= j target-index)
                             1.0
                             0.0)))))))))

(define-cuda-kernel (cuda-softmax-xe-derivative/sparse)
    (void ((group-start int) (group-size int)
           (xd :mat :io) (softmax :mat :input)
           (target-index int) (target-value float)))
  (let ((stride (* block-dim-x grid-dim-x))
        (target-softmax (aref softmax target-index)))
    (do ((i-in-group (+ (* block-dim-x block-idx-x) thread-idx-x)
                     (+ i-in-group stride)))
        ((>= i-in-group group-size))
      (let ((i (+ group-start i-in-group)))
        (set (aref xd i) (+ (aref xd i)
                            (* target-value
                               (- (aref softmax i)
                                  (if (= target-index i)
                                      1.0
                                      0.0)))))))))

(defmethod label-indices ((lump ->softmax-xe-loss))
  (max-row-positions (nodes lump)))

(defmethod label-index-distributions ((lump ->softmax-xe-loss))
  (rows-to-arrays (nodes lump)))

(defmethod cost ((lump ->softmax-xe-loss))
  ;; FIXME: only do this once after FORWARD
  (let ((target (target lump)))
    (if (typep target 'sequence)
        (let ((sum 0)
              (sum-target 0))
          (with-facets ((softmax ((nodes lump) 'backing-array
                                  :direction :input)))
            (do-sparse-targets ((group-start target-index target-value)
                                target (group-size lump))
              (let ((prediction (aref softmax (+ group-start target-index))))
                (incf sum (* (log prediction) target-value))
                (incf sum-target target-value))))
          (values (- sum) sum-target))
        (with-thread-cached-mat (tmp (mat-dimensions (nodes lump)))
          (copy! (nodes lump) tmp)
          (.log! tmp)
          (.*! target tmp)
          (values (asum tmp) (asum target))))))


;;;; LSTM

(defun ->lstm (&key name inputs cell-init output-init n-cells
               (gate-fn '->sigmoid) (input-fn '->tanh)
               (output-fn '->tanh)
               (peepholes t))
  "Create an LSTM layer consisting of input, forget, output gates with
  which input, cell state and output are scaled. Lots of lumps are
  created, the final one representing to output of the LSTM has NAME.
  The rest of the lumps are named automatically based on NAME. This
  function returns only the output lump (`m`), but all created lumps
  are added automatically to the BPN being built.

  There are many papers and tutorials on LSTMs. This version is well
  described in \"Long Short-Term Memory Recurrent Neural Network
  Architectures for Large Scale Acoustic Modeling\" (2014, Hasim Sak,
  Andrew Senior, Francoise Beaufays). Using the notation from that
  paper:

      i_t = s(W_ix * x_t + W_im * m_{t_1} + W_ic .* c_{t-1} + b_i)
      f_t = s(W_fx * x_t + W_fm * m_{t_1} + W_fc .* c_{t-1} + b_f)
      c_t = f_t .* c_{t-1} + i_t .* g(W_cx * x_t + W_cm  * m_{t-1} + b_c)
      o_t = s(W_ox * x_t + W_om * m_{t-1} + W_oc .* c_t + b_o)
      m_t = o_t .* h(c_t)

  ... where `i`, `f`, and `o` are the input, forget and output gates.
  `c` is the cell state and `m` is the actual output.

  Weight matrices for connections from `c` (`W_ic`, `W_fc` and `W_oc`)
  diagonal and are represented by just the vector of diagonal values.
  These connections are only added if PEEPHOLES is true.

  A notable difference from the paper is that `x_t` (the input) is
  actually represented by a list of lumps in INPUTS. Whenever some
  activation is to be calculated based on `x_t`, it is going to be the
  sum of individual activations. For example, `W_ix * x_t` is really
  `sum_j W_ijx * inputs_j`.

  If CELL-INIT is non-NIL, then it must be a CLUMP of SIZE N-CELLS
  form which stands for the initial state of the value cell (c_{-1}).
  CELL-INIT being NIL is equivalent to the state of all zeros."
  (check-type n-cells index)
  (let* ((input-gate-name `(,name :input))
         (forget-gate-name `(,name :forget))
         (output-gate-name `(,name :output))
         (cell-name `(,name :cell))
         (output-name name))
    (labels ((add (x list)
               (if x (cons x list) list))
             (lagged-cell ()
               (if (zerop (time-step))
                   cell-init
                   (lag cell-name)))
             (lagged-output ()
               (if (zerop (time-step))
                   output-init
                   (lag output-name))))
      (build-fnn (:name name :class '->lstm)
        ;; i_t = s(W_ix * x_t + W_im * m_{t_1} + W_ic .* c_{t-1} + b_i)
        (input-gate
         (funcall
          gate-fn :name input-gate-name
          :x (->activation :name input-gate-name :size n-cells
                           :inputs (add (lagged-output) inputs)
                           :peepholes (when peepholes
                                        (add (lagged-cell) ())))))
        ;; f_t = s(W_fx * x_t + W_fm * m_{t_1} + W_fc .* c_{t-1} + b_f)
        (forget-gate
         (funcall
          gate-fn :name forget-gate-name
          :x (->activation :name forget-gate-name
                           :size n-cells
                           :inputs (add (lagged-output) inputs)
                           :peepholes (when peepholes
                                        (add (lagged-cell) ())))))
        ;; c_t = f_t .* c_{t-1} + i_t .* g(W_cx * x_t + W_cm * m_{t-1} + b_c)
        (cell
         ;; Save memory by sharing.
         (let ((shared-with-clump
                 (->+ :name cell-name
                      :size n-cells
                      :args ())))
           (when (lagged-cell)
             (->* :x forget-gate :y (lagged-cell)
                  :shared-with-clump shared-with-clump))
           (->* :x input-gate
                :y (funcall
                    input-fn
                    :x (->activation
                        :name cell-name
                        :size n-cells
                        :inputs (add (lagged-output) inputs)))
                :shared-with-clump shared-with-clump)
           shared-with-clump))
        ;; o_t = s(W_ox * x_t + W_om * m_{t-1} + W_oc .* c_t + b_o)
        (output-gate
         (funcall
          gate-fn :name output-gate-name
          :x (->activation :name output-gate-name :size n-cells
                           :inputs (add (lagged-output) inputs)
                           :peepholes (when peepholes
                                        (list cell)))))
        ;; m_t = o_t .* h(c_t)
        (output
         (->* :name output-name
              :x output-gate
              :y (funcall output-fn :x cell)))))))

(defclass ->lstm (bpn)
  ())


(defclass-now ->seq-barrier (lump)
  ((seq-elt-fn :initarg :seq-elt-fn :reader seq-elt-fn)
   (seq-lengths :accessor seq-lengths)))

(defmaker ->seq-barrier)

(defmethod default-size ((lump ->seq-barrier))
  (size (funcall (seq-elt-fn lump) 0)))

(defmethod forward ((lump ->seq-barrier))
  ;; For each row of NODES, there is an input sequence of some length.
  ;; Look up the clump at the end of the sequence and copy its
  ;; corresponding row to NODES.
  (let ((nodes (nodes lump))
        (size (size lump))
        (seq-lengths (seq-lengths lump))
        (seq-elt-fn (seq-elt-fn lump))
        (stripe 0))
    (map-displacements
     (lambda (nodes)
       (let* ((seq-length (pop seq-lengths))
              (end-clump (funcall seq-elt-fn (1- seq-length)))
              (end-nodes (nodes end-clump)))
         (assert (< stripe (n-stripes end-clump)))
         (with-shape-and-displacement (end-nodes size (mat-displacement nodes))
           (copy! end-nodes nodes)))
       (incf stripe))
     nodes size)))

(defmethod backward ((lump ->seq-barrier))
  (let ((derivatives (derivatives lump))
        (size (size lump))
        (seq-lengths (seq-lengths lump))
        (seq-elt-fn (seq-elt-fn lump)))
    (map-displacements
     (lambda (derivatives)
       (let* ((seq-length (pop seq-lengths))
              (end-clump (funcall seq-elt-fn (1- seq-length)))
              (end-derivatives (derivatives end-clump)))
         (with-shape-and-displacement (end-derivatives size
                                       (mat-displacement derivatives))
           (axpy! 1 derivatives end-derivatives))))
     derivatives size)))


;;;; RENORMALIZE-ACTIVATIONS

(defun renormalize-activations (->mms l2-upper-bound)
  "If the l2 norm of the incoming weight vector of the a unit is
  larger than L2-UPPER-BOUND then renormalize it to L2-UPPER-BOUND.
  The list of ->MMS is assumed to be eventually fed to the
  same lump.

  To use it, group the activation clumps into the same GD-OPTIMIZER and
  hang this function on AFTER-UPDATE-HOOK, that latter of which is
  done for you ARRANGE-FOR-RENORMALIZING-ACTIVATIONS.

  See \"Improving neural networks by preventing co-adaptation of
  feature detectors (Hinton, 2012)\",
  <http://arxiv.org/pdf/1207.0580.pdf>."
  (when (and ->mms l2-upper-bound)
    (renormalize-mats
     (loop for lump in ->mms
           collect (let ((weights (etypecase lump
                                    (->mm (weights lump))
                                    (->weight lump))))
                     (list (nodes weights)
                           (if (and (typep lump '->mm)
                                    (transpose-weights-p lump))
                               :row
                               :column))))
     l2-upper-bound)))

(defun arrange-for-renormalizing-activations (bpn optimizer l2-upper-bound)
  "By pushing a lambda to AFTER-UPDATE-HOOK of OPTIMIZER arrange for
  all weights beings trained by OPTIMIZER to be renormalized (as in
  RENORMALIZE-ACTIVATIONS with L2-UPPER-BOUND).

  It is assumed that if the weights either belong to an activation
  lump or are simply added to the activations (i.e. they are biases)."
  (push (let ((->mms nil)
              (firstp t))
          (lambda ()
            (when firstp
              (setq ->mms
                    (loop for lump in (segments optimizer)
                          collect (or (find-activation-lump-for-weight lump bpn)
                                      lump)))
              (setq firstp nil))
            (renormalize-activations ->mms l2-upper-bound)))
        (after-update-hook optimizer)))

(defun find-activation-lump-for-weight (->weight bpn)
  ;; FIXME: this iteration is broken for nested bpns.
  (loop for lump across (clumps bpn) do
    (when (and (typep lump '->mm)
               (eq (weights lump) ->weight))
      (return lump))))

(defun renormalize-mats (mat-and-row/column-list l2-upper-bound)
  (let ((n (mat-and-row/column-sum-size mat-and-row/column-list))
        (firstp t)
        (l2-upper-bound (coerce-to-ctype l2-upper-bound)))
    (with-thread-cached-mat (sums (list n 1))
      (loop for (mat row/column) in mat-and-row/column-list
            do (with-thread-cached-mat
                   (square (mat-dimensions mat)
                           ;; no initialization?
                           :initial-element nil)
                 (copy! mat square)
                 (.square! square)
                 (assert (= n (ecase row/column
                                ((:row) (mat-dimension mat 0))
                                ((:column) (mat-dimension mat 1)))))
                 (ecase row/column
                   ((:row)
                    (sum! square sums :axis 1 :beta (if firstp 0 1)))
                   ((:column)
                    (sum! square sums :axis 0 :beta (if firstp 0 1)))))
               (setq firstp nil))
      (when sums
        (.sqrt! sums)
        (loop for (mat row/column) in mat-and-row/column-list
              do (ecase row/column
                   ((:row)
                    (maybe-renormalize-rows mat l2-upper-bound sums))
                   ((:column)
                    (maybe-renormalize-columns mat l2-upper-bound sums))))))))

(defun mat-and-row/column-sum-size (mat-and-row/column-list)
  (if mat-and-row/column-list
      (destructuring-bind (mat row/column) (first mat-and-row/column-list)
        (ecase row/column
          ((:row) (mat-dimension mat 0))
          ((:column) (mat-dimension mat 1))))
      0))

(defun maybe-renormalize-rows (mat l2-upper-bound norms)
  (let ((n-rows (mat-dimension mat 0))
        (n-columns (mat-dimension mat 1)))
    (declare (type fixnum n-rows n-columns))
    (assert (= n-rows (mat-size norms)))
    (if (use-cuda-p)
        (let ((n n-rows))
          (cuda-maybe-renormalize-rows
           mat n-rows n-columns l2-upper-bound norms
           :grid-dim (list (ceiling n 256) 1 1)
           :block-dim (list 256 1 1)))
        (lisp-maybe-renormalize-rows mat (mat-displacement mat)
                                     n-rows n-columns
                                     l2-upper-bound
                                     norms (mat-displacement norms)))))

(define-lisp-kernel (lisp-maybe-renormalize-rows)
    ((x :mat :io) (start-x index) (n-rows index) (n-columns index)
     (l2-upper-bound single-float) (norms :mat :input) (start-norms index))
  (loop for norms-i of-type index upfrom start-norms
          below (the! index (+ start-norms n-rows))
        do (let ((norm (aref norms norms-i)))
             (when (< l2-upper-bound norm)
               (let ((scale (/ l2-upper-bound (+ norm 0.0000001)))
                     (row-start (+ start-x (the! index (* norms-i n-columns)))))
                 (do ((j 0 (+ j 1)))
                     ((>= j n-columns))
                   (let ((j (the! index (+ row-start j))))
                     (setf (aref x j) (* scale (aref x j))))))))))

(define-cuda-kernel (cuda-maybe-renormalize-rows)
    (void ((x :mat :io) (n-rows int) (n-columns int)
           (l2-upper-bound float) (norms :mat :input)))
  (let ((i (+ (* block-dim-x block-idx-x) thread-idx-x)))
    (when (< i n-rows)
      (let ((norm (aref norms i)))
        (when (< l2-upper-bound norm)
          (let ((scale (/ l2-upper-bound (+ norm 0.0000001)))
                (row-start (* i n-columns)))
            (do ((j 0 (+ j 1)))
                ((>= j n-columns))
              (set (aref x (+ row-start j))
                   (* scale (aref x (+ row-start j)))))))))))

(defun maybe-renormalize-columns (mat l2-upper-bound norms)
  (let ((n-rows (mat-dimension mat 0))
        (n-columns (mat-dimension mat 1)))
    (declare (type fixnum n-rows n-columns))
    (assert (= n-columns (mat-size norms)))
    (if (use-cuda-p)
        (let ((n n-columns))
          (cuda-maybe-renormalize-columns
           mat n-rows n-columns l2-upper-bound norms
           :grid-dim (list (ceiling n 256) 1 1)
           :block-dim (list 256 1 1)))
        (lisp-maybe-renormalize-rows mat (mat-displacement mat)
                                     n-rows n-columns l2-upper-bound
                                     norms (mat-displacement norms)))))

(define-lisp-kernel (lisp-maybe-renormalize-columns)
    ((x :mat :io) (start-x index) (n-rows index) (n-columns index)
     (l2-upper-bound single-float) (norms :mat :input) (start-norms index))
  (loop for norms-i of-type index upfrom start-norms
          below (the! index (+ start-norms n-rows))
        do (let ((norm (aref norms norms-i)))
             (when (< l2-upper-bound norm)
               (let ((scale (/ l2-upper-bound (+ norm 0.0000001)))
                     (k (the! index (+ start-x norms-i))))
                 (declare (type index k))
                 (dotimes (i n-rows)
                   (setf (aref x k) (* scale (aref x k)))
                   (setq k (the! index (+ k n-columns)))))))))

(define-cuda-kernel (cuda-maybe-renormalize-columns)
    (void ((x :mat :io) (n-rows int) (n-columns int)
           (l2-upper-bound float) (norms :mat :input)))
  (let ((i (+ (* block-dim-x block-idx-x) thread-idx-x)))
    (when (< i n-columns)
      (let ((norm (aref norms i)))
        (when (< l2-upper-bound norm)
          (let ((scale (/ l2-upper-bound (+ norm 0.0000001)))
                (k i))
            (do ((j 0 (+ j 1)))
                ((>= j n-rows))
              (set (aref x k) (* scale (aref x k)))
              (set k (+ k n-columns)))))))))


(defun clip-gradients (mats l2-upper-bound &key callback)
  (let ((sum 0))
    (map nil (lambda (mat)
               (incf sum (expt (nrm2 mat) 2)))
         mats)
    (let ((norm (sqrt sum)))
      (when (< l2-upper-bound norm)
        (let ((scale (/ l2-upper-bound norm)))
          (when callback
            (funcall callback scale))
          (map nil (lambda (mat)
                     (scal! scale mat))
               mats)
          scale)))))

(defun arrange-for-clipping-gradients (batch-gd-optimizer l2-upper-bound
                                       &key callback)
  (push (lambda ()
          (clip-gradients (list (mgl-gd::accumulator batch-gd-optimizer))
                          l2-upper-bound :callback callback))
        (before-update-hook batch-gd-optimizer))
  batch-gd-optimizer)
