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
;;;; BLAS 3 routines work their magic, `stripes' were added.

(in-package :mgl-bp)

(defsection @mgl-bp (:title "Backpropagation Neural Networks")
  "")

(defun make-cost-monitors (model &key operation-mode attributes)
  (make-cost-monitors* model operation-mode attributes))

(defgeneric make-cost-monitors* (mode operation-mode attributes))

;;;; Lump

(defvar *bpn-being-built* nil)

(defvar *in-training-p* nil)

(defvar *next-lump-name* nil)

(defun next-lump-name ()
  (prog1 (or *next-lump-name* (gensym))
    (setf *next-lump-name* nil)))

(defgeneric default-size (lump)
  (:method (lump)
    (or (slot-boundp lump 'size)
        (error "Can't compute size for ~S." lump))))

(defclass-now lump ()
  ((name :initform (next-lump-name) :initarg :name :reader name)
   (size :type index :initarg :size :reader size)
   (same-stripes-p
    :initform nil :initarg :same-stripes-p :reader same-stripes-p
    :documentation "Non-NIL iff all stripes are the same. If true, it
    effectively overrides both N-STRIPES and MAX-N-STRIPES and there
    is only one row in NODES and DERIVATIVES. Set up by the lump
    itself taking its inputs into account. Notably, ->WEIGHTs always
    have SAME-STRIPES-P T.")
   (nodes
    :initform nil :type mat :reader nodes
    :documentation "The values of the nodes. All nodes have values. It
    is conceptually a N-STRIPES x SIZE matrix that can be enlarged to
    MAX-N-STRIPES x SIZE by setting N-STRIPES.")
   (derivatives
    :initform nil :type mat :reader derivatives
    :documentation "Derivatives of nodes, input node derivatives are
    not calculated. A 1d array representing a matrix of the same
    dimension as NODES.")
   (default-value
    :initform #.(flt 0) :initarg :default-value :type flt
    :reader default-value
    :documentation "Upon creation or resize the lump's nodes get
    filled with this value.")))

(defmethod n-stripes ((lump lump))
  (let ((nodes (nodes lump)))
    (if nodes
        (mat-dimension nodes 0)
        1)))

(defmethod max-n-stripes ((lump lump))
  (let ((nodes (nodes lump)))
    (if nodes
        (/ (mat-max-size nodes)
           (mat-dimension nodes 1))
        1)))

(defun limit-stripes (lump n)
  (if (same-stripes-p lump)
      (min 1 n)
      n))

;;; The effective number of stripes.
(defun n-stripes* (lump)
  (limit-stripes lump (n-stripes lump)))

;;; The effective maximum number of stripes, i.e. the number of
;;; allocated rows of NODES and DERIVATIVES.
(defun max-n-stripes* (lump)
  (limit-stripes lump (max-n-stripes lump)))

(defun norm (v)
  (with-facets ((v* (v 'backing-array :direction :input :type flt-vector)))
    (let* ((sum (flt 0))
           (start (mat-displacement v))
           (end (+ start (mat-size v))))
      (locally
          (declare (optimize speed))
        (loop for i of-type index upfrom start below end
              do (incf sum (expt (aref v* i) 2))))
      (sqrt sum))))

(defmethod print-object ((lump lump) stream)
  (pprint-logical-block (stream ())
    (print-unreadable-object (lump stream :type t)
      (format stream "~S ~:_~S ~:_~S" (name lump) :size
              (if (slot-boundp lump 'size)
                  (size lump)
                  :unbound))
      (let ((mgl-cube:*let-input-through-p* t))
        (format stream "*~S(~S/~S) :norm ~,5F" (n-stripes* lump)
                (n-stripes lump) (max-n-stripes lump)
                (ignore-errors (norm (nodes lump)))))))
  lump)

(defmethod set-n-stripes (n-stripes (lump lump))
  (let ((n-stripes (limit-stripes lump n-stripes)))
    (assert (<= 0 n-stripes (max-n-stripes lump)))
    (let ((dimensions (list n-stripes (mat-dimension (nodes lump) 1))))
      (reshape! (nodes lump) dimensions)
      (reshape! (derivatives lump) dimensions)))
  n-stripes)

(defmethod set-max-n-stripes (max-n-stripes (lump lump))
  (let ((old-max-n-stripes* (max-n-stripes* lump))
        (size (size lump)))
    (let ((n-stripes* (n-stripes* lump))
          (max-n-stripes* (limit-stripes lump max-n-stripes)))
      (cond ((zerop max-n-stripes*)
             (when (nodes lump)
               (setf (slot-value lump 'nodes) nil
                     (slot-value lump 'derivatives) nil)))
            ((or (/= max-n-stripes* old-max-n-stripes*)
                 (null (nodes lump)))
             (setf (slot-value lump 'nodes)
                   (make-mat (list n-stripes* size)
                             :ctype flt-ctype
                             :max-size (* max-n-stripes* size)
                             :initial-element (default-value lump)))
             (setf (slot-value lump 'derivatives)
                   (make-mat (list n-stripes* size)
                             :ctype flt-ctype
                             :max-size (* max-n-stripes* size)))))))
  max-n-stripes)

(defmethod initialize-instance :after ((lump lump) &key &allow-other-keys)
  (unless (slot-boundp lump 'size)
    (setf (slot-value lump 'size) (default-size lump))))

(defmethod initialize-instance :around ((lump lump) &key &allow-other-keys)
  ;; ensure that the matrices are allocated
  (call-next-method)
  (setf (max-n-stripes lump) (max-n-stripes lump))
  (when *bpn-being-built*
    (add-lump lump *bpn-being-built*)))

(defmethod stripe-start (stripe (lump lump))
  (* (if (same-stripes-p lump)
         0
         (progn
           (assert (<= 0 stripe (1- (n-stripes lump))))
           stripe))
     (size lump)))

(defmethod stripe-end (stripe (lump lump))
  (+ (stripe-start stripe lump)
     (size lump)))

(defgeneric transfer-lump (lump))
(defgeneric derive-lump (lump))
(defgeneric set-input-done (lump)
  (:method (lump)))


;;;; Data lumps

(defclass-now data-lump (lump) ())

(defclass-now ->weight (data-lump)
  ((same-stripes-p :initform t)
   (dimensions :initarg dimensions :reader dimensions)))

(defmaker (->weight ->weight*))

(defmethod initialize-instance :after ((weight ->weight) &key dimensions size
                                       &allow-other-keys)
  (setf (slot-value weight 'dimensions)
        (if dimensions
            (alexandria:ensure-list dimensions)
            (list 1 size)))
  (unless size
    (setf (slot-value weight 'size) (reduce #'* (dimensions weight)))))

(defmethod n-stripes ((weight ->weight))
  1)

(defmethod max-n-stripes ((weight ->weight))
  1)

(defmethod set-n-stripes (n-stripes (weight ->weight))
  (declare (ignore n-stripes)))

(defmethod set-max-n-stripes (max-n-stripes (weight ->weight))
  (declare (ignore max-n-stripes))
  (when (or (null (nodes weight))
            (not (equal (dimensions weight)
                        (mat-dimensions (nodes weight)))))
    (setf (slot-value weight 'nodes)
          (make-instance 'mat
                         :ctype flt-ctype
                         :dimensions (dimensions weight)
                         :initial-element (default-value weight)))
    (setf (slot-value weight 'derivatives)
          (make-instance 'mat
                         :ctype flt-ctype
                         :dimensions (dimensions weight)))))

(defvar *lumps-to-copy* ())

(defmacro with-weights-copied ((from-bpn) &body body)
  "In BODY ->WEIGHT will first look up if a weight lump of the same
  name exists in FROM-BPN and return that, or else create a weight
  lump normally. If FROM-BPN is NIL, then weights are copied."
  (alexandria:with-gensyms (%from-bpn)
    `(let* ((,%from-bpn ,from-bpn)
            (*lumps-to-copy*
              (when ,%from-bpn
                (remove-if-not (lambda (lump)
                                 (typep lump '->weight))
                               (lumps ,%from-bpn)))))
       ,@body)))

(defun ->weight (&rest args)
  (let ((to-be-copied (find *next-lump-name* *lumps-to-copy*
                            :key #'name :test #'equal)))
    (cond (to-be-copied
           (when *bpn-being-built*
             (add-lump to-be-copied *bpn-being-built*))
           to-be-copied)
          (t
           (apply #'->weight* args)))))

(defclass-now ->constant (data-lump)
  ((default-value :initform #.(flt 1))))

(defmaker ->constant)

(defmethod default-size ((lump ->constant))
  1)

(defmethod transfer-lump ((lump data-lump)))

(defmethod derive-lump ((lump data-lump)))


(defclass-now ->dropout (lump)
  ((x :initarg :x :reader x)
   (dropout
    :type (or null flt)
    :initform nil :initarg :dropout :reader dropout
    :documentation "If non-NIL, then in the forward pass zero out each
    node in this chunk with DROPOUT probability. See Geoffrey Hinton's
    'Improving neural networks by preventing co-adaptation of feature
    detectors'.")
   (mask :initform nil :reader mask)))

(defmaker ->dropout)

(defmethod default-size ((lump ->dropout))
  (size (x lump)))

(defun ensure-mask (lump)
  (when (dropout lump)
    (let ((x (nodes (x lump))))
      (unless (and (mask lump)
                   (= (mat-size x)
                      (mat-size (mask lump))))
        (setf (slot-value lump 'mask)
              (make-mat (mat-size x) :ctype flt-ctype)))))
  (mask lump))

(define-cuda-kernel (cuda-dropout-xorwow)
    (void ((x :mat :io) (n int) (mask :mat :io) (dropout-probability float)))
  (let ((i (+ (* block-dim-x block-idx-x) thread-idx-x)))
    (when (< i n)
      (if (< (aref mask i) dropout-probability)
          (progn
            (set (aref x i) 0.0)
            (set (aref mask i) 0.0))
          (set (aref mask i) 1.0)))))

(defun dropout! (x mask dropout-probability &key (n (mat-size x)))
  (declare (type flt dropout-probability)
           (type index n))
  (cond ((use-cuda-p)
         (uniform-random! mask)
         (cuda-dropout-xorwow x n mask dropout-probability
                              :grid-dim (list (ceiling n 256) 1 1)
                              :block-dim (list 256 1 1)))
        (t
         (with-facets ((x* (x 'backing-array :direction :io :type flt-vector))
                       (m* (mask 'backing-array :direction :output
                                 :type flt-vector)))
           (declare (optimize speed #.*no-array-bounds-check*))
           (dotimes (i n)
             (let ((dropped (< (random #.(flt 1)) dropout-probability)))
               (cond (dropped
                      (setf (aref x* i) #.(flt 0))
                      (setf (aref m* i) #.(flt 0)))
                     (t
                      (setf (aref m* i) #.(flt 1))))))))))

(defmethod transfer-lump ((lump ->dropout))
  (let ((x (x lump))
        (dropout (dropout lump)))
    (declare (type (or flt null) dropout))
    (assert (= (size lump) (size x)))
    (assert (not (same-stripes-p lump)))
    (assert (not (same-stripes-p x)))
    ;; Some subclasses (->INPUT, for instance) set X to be the same as
    ;; LUMP.
    (unless (eq x lump)
      (copy! (nodes x) (nodes lump)))
    (when dropout
      (if *in-training-p*
          (let ((mask (ensure-mask lump)))
            (dropout! (nodes lump) mask dropout))
          (scal! (- #.(flt 1) dropout) (nodes lump))))))

(defmethod derive-lump ((lump ->dropout))
  (let* ((x (x lump))
         (dropout (dropout lump))
         (xd (derivatives x))
         (ld (derivatives lump))
         (mask (mask lump)))
    (assert (= (size lump) (size x)))
    (if (not dropout)
        (axpy! 1 ld xd :n (* (size lump) (n-stripes lump)))
        (geem! 1 mask ld 1 xd))))


(defclass-now ->multiply-with-gaussian (lump)
  ((x :initarg :x :reader x)
   (variance
    :type (or null flt)
    :initform nil :initarg :variance :reader variance)
   (multipliers :initform nil :reader multipliers)))

(defmaker ->multiply-with-gaussian)

(defmethod default-size ((lump ->multiply-with-gaussian))
  (size (x lump)))

(defun ensure-multipliers (lump)
  (when (variance lump)
    (let ((x (nodes (x lump))))
      (unless (and (multipliers lump)
                   (= (mat-size x)
                      (mat-size (multipliers lump))))
        (setf (slot-value lump 'multipliers)
              (make-mat (mat-size x) :ctype flt-ctype)))))
  (multipliers lump))

(defun mgn! (l x multipliers variance)
  (gaussian-random! multipliers :mean 1 :stddev (sqrt variance))
  (geem! 1 multipliers x 0 l))

(defmethod transfer-lump ((lump ->multiply-with-gaussian))
  (if *in-training-p*
      (mgn! (nodes lump) (nodes (x lump)) (ensure-multipliers lump)
            (variance lump))
      (copy! (nodes (x lump)) (nodes lump))))

(defmethod derive-lump ((lump ->multiply-with-gaussian))
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
    (unless (and (randoms lump)
                 (= (mat-size x)
                    (mat-size (randoms lump))))
      (setf (slot-value lump 'randoms)
            (make-mat (mat-size x) :ctype flt-ctype))))
  (randoms lump))

(defmethod transfer-lump ((lump ->sample-binary))
  (let ((x (x lump)))
    (assert (= (size lump) (size x)))
    (assert (not (same-stripes-p lump)))
    (assert (not (same-stripes-p x)))
    (unless (eq x lump)
      (copy! (nodes x) (nodes lump)))
    (when *in-training-p*
      (let ((randoms (ensure-randoms lump)))
        (uniform-random! randoms)
        (.<! randoms (nodes lump))))))

(defmethod derive-lump ((lump ->sample-binary))
  (let* ((x (x lump))
         (xd (derivatives x))
         (l (nodes lump))
         (ld (derivatives lump))
         (n (mat-size xd)))
    (assert (= (size lump) (size x)))
    ;; (axpy! 1 ld xd)
    ;; #+nil
    (if (use-cuda-p)
        (cuda-sample-binary-derivative xd n l ld
                                       :grid-dim (list (ceiling n 256) 1 1)
                                       :block-dim (list 256 1 1))
        (with-facets ((xd* (xd 'backing-array :direction :io
                               :type flt-vector))
                      (l* (l 'backing-array :direction :input
                             :type flt-vector))
                      (ld* (ld 'backing-array :direction :input
                               :type flt-vector)))
          (declare (optimize (speed 3) #.*no-array-bounds-check*))
          (loop for stripe of-type index below (n-stripes* lump) do
            (with-stripes ((stripe lump ls le)
                           (stripe x xs xe))
              (loop for li upfrom ls below le
                    for xi upfrom xs below xe
                    do (when (/= #.(flt 0) (aref l* li))
                         (incf (aref xd* li) (aref ld* li))))))))))

(define-cuda-kernel (cuda-sample-binary-derivative)
    (void ((xd :mat :io) (n int) (l :mat :input)  (ld :mat :input)))
  (let ((i (+ (* block-dim-x block-idx-x) thread-idx-x)))
    (when (< i n)
      (when (/= 0.0 (aref l i))
        (set (aref xd i) (+ (aref xd i) (aref ld i)))))))


;;;; ->INPUT

(defclass-now ->input (->dropout data-lump)
  ((running-stats :reader running-stats)
   (update-stats-p :initform nil :initarg :update-stats-p
                   :accessor update-stats-p)
   (normalize-with-stats-p :initform nil :initarg :normalize-with-stats-p
                           :accessor normalize-with-stats-p)
   (normalized-cap :initform nil :initarg :normalized-cap
                   :accessor normalized-cap)))

(defmaker ->input)

(defmethod set-input-done ((lump ->input))
  (when (or (update-stats-p lump) (normalize-with-stats-p lump))
    ;; FIXME: cudaize or remove?
    (with-facet (nodes ((nodes lump) 'backing-array :direction :io
                        :type flt-vector))
      (let ((n-stripes* (n-stripes* lump))
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
                                 x)))))))))))))

(defmethod transfer-lump ((lump ->input))
  (setf (slot-value lump 'x) lump)
  (call-next-method))

;;; Do nothing. In particular, prevent the method for ->DROPOUT from
;;; being called.
(defmethod derive-lump ((lump ->input)))


;;;; ->SUM

(defclass-now ->sum (lump)
  ((x :initarg :x :reader x))
  (:documentation "Sum of all nodes \(per stripe)."))

(defmaker ->sum)

(defmethod default-size ((lump ->sum))
  1)

(define-cuda-kernel (cuda-sum-row)
    (void ((x :mat :input) (m int) (n int) (y :mat :output)))
  (let ((i (+ (* block-dim-x block-idx-x) thread-idx-x)))
    (when (< i m)
      (let ((sum 0.0)
            (row-start (* i n)))
        (do ((j 0 (+ j 1)))
            ((>= j n))
          (set sum (+ sum (aref x (+ row-start j)))))
        (set (aref y i) sum)))))

(defmethod transfer-lump ((lump ->sum))
  (let* ((x (x lump))
         (xn (nodes x))
         (ln (nodes lump)))
    (assert (= (n-stripes lump) (n-stripes x)))
    (if (use-cuda-p)
        (let ((m (n-stripes lump)))
          (cuda-sum-row xn m (size x) ln
                        :grid-dim (list (ceiling m 256) 1 1)
                        :block-dim (list 256 1 1)))
        (with-facets ((x* ((nodes x) 'backing-array :direction :input
                           :type flt-vector))
                      (to* ((nodes lump) 'backing-array :direction :output
                            :type flt-vector)))
          (declare (optimize (speed 3) #.*no-array-bounds-check*))
          (loop for stripe of-type index below (n-stripes* lump) do
            (with-stripes ((stripe x xs xe))
              (setf (aref to* stripe)
                    (let ((sum (flt 0)))
                      (declare (type flt sum))
                      (loop for xi upfrom xs below xe
                            do (incf sum (aref x* xi)))
                      sum))))))))


;;;; ->ERROR

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

(defmethod transfer-lump :around ((lump ->error))
  (call-next-method)
  (when (importance lump)
    (.*! (importance lump) (nodes lump))))

(defmethod derive-lump :around ((lump ->error))
  (if (importance lump)
      (axpy! 1 (importance lump) (derivatives lump))
      (.+! 1 (derivatives lump)))
  (call-next-method))


;;;; BPN

(defclass bpn ()
  ((lumps
    :initform (make-array 0 :element-type 'lump :adjustable t :fill-pointer t)
    :initarg :lumps
    :type (array lump (*)) :reader lumps
    :documentation "Lumps in reverse order")
   (n-stripes
    :initform 1 :type index :initarg :n-stripes
    :reader n-stripes)
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

(defmethod set-n-stripes (n-stripes (bpn bpn))
  (setf (slot-value bpn 'n-stripes) n-stripes)
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
  (or (find name (lumps bpn) :key #'name :test #'name=)
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

(defmacro build-bpn ((&key bpn (class ''bpn) initargs
                      (max-n-stripes 1)) &body lumps)
  "Syntactic sugar to assemble BPNs from lumps. Like LET*, it is a
  sequence of bindings (of symbols to lumps). The names of the lumps
  created default to the symbol of the binding. In case a lump is not
  bound to a symbol (because it was created in a nested expression),
  the local function LUMP finds the lump with the given name in the
  bpn being built. Example:

      (build-bpn ()
        (features (->input :size n-features))
        (biases (->weight :size n-features))
        (weights (->weight :size (* n-hiddens n-features)))
        (activations0 (->activation :weights weights :x (lump 'features)))
        (activations (->+ :args (list biases activations0)))
        (output (->sigmoid :x activations)))"
  (let ((bindings
          (mapcar (lambda (lump)
                    (destructuring-bind (symbol init-form) lump
                      `(,symbol (let ((*next-lump-name* ',symbol))
                                  (,(first init-form)
                                   ,@(rest init-form))))))
                  lumps)))
    `(let* ((*bpn-being-built* (apply #'make-instance ,class
                                      :max-n-stripes ,max-n-stripes
                                      ,initargs))
            ,@(when bpn
                `((,bpn *bpn-being-built*))))
       (flet ((lump (name)
                (find-lump name *bpn-being-built* :errorp t)))
         (declare (ignorable #'lump))
         (let* ,bindings
           (declare (ignorable ,@(mapcar #'first bindings)))))
       *bpn-being-built*)))

(defun ->lump (bpn lump-spec)
  (if (typep lump-spec 'lump)
      lump-spec
      (find-lump lump-spec bpn :errorp t)))

(defun forward-bpn (bpn &key from-lump to-lump end-lump)
  "Propagate the values from the already clamped inputs."
  (declare (optimize (debug 3)))
  (let ((from-lump (if from-lump (->lump bpn from-lump) nil))
        (to-lump (if to-lump (->lump bpn to-lump) nil))
        (seen-from-lump-p (not from-lump)))
    (loop for lump across (lumps bpn)
          until (eq lump end-lump)
          do (when (eq lump from-lump) (setq seen-from-lump-p t))
          do (when seen-from-lump-p (transfer-lump lump))
          until (eq lump to-lump))))

;;; Derivatives of weights are left alone to let them accumulate which
;;; is useful in batches such as when training with conjugate
;;; gradient.
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

(defclass bp-learner ()
  ((bpn :initarg :bpn :reader bpn)
   (first-trained-lump :reader first-trained-lump)
   (monitors :initform () :initarg :monitors :accessor monitors)))

(define-descriptions (learner bp-learner :inheritp t)
  bpn first-trained-lump)

(defmethod describe-object :after ((learner bp-learner) stream)
  (when (slot-boundp learner 'bpn)
    (describe (bpn learner) stream)))

(defmethod map-segments (fn (source bp-learner))
  (map-segments fn (bpn source)))

(defmethod initialize-gradient-source* (optimizer (learner bp-learner)
                                        weights dataset)
  (when (next-method-p)
    (call-next-method))
  (setf (slot-value learner 'first-trained-lump) nil))

(defmethod map-segments (fn (bpn bpn))
  (map nil
       (lambda (lump)
         (when (typep lump '->weight)
           (funcall fn lump)))
       (lumps bpn)))

(defmethod segment-weights ((lump lump))
  (nodes lump))

(defun first-trained-weight-lump (optimizer learner)
  "Much time can be wasted computing derivatives of non-trained weight
  lumps. Return the first one that OPTIMIZER trains."
  (or (slot-value learner 'first-trained-lump)
      (setf (slot-value learner 'first-trained-lump)
            (find-if (lambda (lump)
                       (member lump (segments optimizer)))
                     (lumps (bpn learner))))))

(defmethod cost (bpn)
  "Return the sum of costs for all active stripes. The cost of a
  stripe is the sum of the error nodes. The second value is the number
  of stripes."
  (let ((sum (flt 0)))
    (loop for lump across (lumps bpn) do
      (when (typep lump '->error)
        (with-facet (nodes ((nodes lump) 'backing-array :direction :input
                            :type flt-vector))
          (declare (type flt-vector nodes))
          (incf sum (if (same-stripes-p lump)
                        (* (n-stripes lump) (aref nodes 0))
                        (loop for i below (n-stripes lump)
                              sum (aref nodes i)))))))
    (values sum (n-stripes bpn))))

(defun compute-derivatives (samples optimizer learner)
  (let ((bpn (bpn learner))
        (cost #.(flt 0)))
    (do-executors (samples bpn)
      (let ((*in-training-p* t))
        (set-input samples bpn)
        (forward-bpn bpn)
        (incf cost (cost bpn))
        (backward-bpn bpn :last-lump (first-trained-weight-lump
                                      optimizer learner))
        (apply-monitors (monitors learner) samples bpn)))
    cost))


;;;; Gradient based optimization

(defun add-and-forget-derivatives (bpn gradient-sink multiplier)
  (do-gradient-sink ((lump accumulator) gradient-sink)
    (axpy! multiplier (derivatives lump) accumulator))
  ;; All weight derivatives must be zeroed, even the ones not being
  ;; trained on to avoid overflows.
  (loop for lump across (lumps bpn)
        do (when (typep lump '->weight)
             (fill! 0 (derivatives lump)))))

(defmethod accumulate-gradients* ((learner bp-learner) gradient-sink
                                  batch multiplier valuep)
  (let ((bpn (bpn learner))
        (cost #.(flt 0)))
    (loop for samples in (group batch (max-n-stripes bpn))
          do (incf cost (compute-derivatives samples gradient-sink learner)))
    ;; Derivatives of weights keep accumulating in the loop above,
    ;; they are not zeroed like non-weight derivatives, so it's ok to
    ;; call ADD-AND-FORGET-DERIVATIVES once.
    (add-and-forget-derivatives bpn gradient-sink multiplier)
    cost))


;;;; I/O

(defmethod write-weights ((lump ->weight) stream)
  (write-mat (nodes lump) stream))

(defmethod read-weights ((lump ->weight) stream)
  (read-mat (nodes lump) stream))

(defmethod write-weights ((bpn bpn) stream)
  (map-segments (lambda (weights)
                  (write-weights weights stream))
                bpn))

(defmethod read-weights ((bpn bpn) stream)
  (map-segments (lambda (weights)
                  (read-weights weights stream))
                bpn))


;;;; ->NORMALIZED

(defclass-now ->normalized (lump)
  ((x :initarg :x :reader x :documentation "Input comes from here.")
   (group-size :initarg :group-size :reader group-size)
   (scale
    :initform #.(flt 1)
    :type (or flt flt-vector)
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

(defmethod transfer-lump ((lump ->normalized))
  (let* ((x (x lump))
         (group-size (group-size lump))
         (scale (scale lump)))
    (declare (type index group-size)
             (type (or flt flt-vector) scale))
    (assert (= (size lump) (size x)))
    (assert (= (n-stripes lump) (n-stripes x)))
    (with-facets ((x* ((nodes x) 'backing-array :direction :input
                       :type flt-vector))
                  (to* ((nodes lump) 'backing-array :direction :output
                        :type flt-vector)))
      (loop for stripe of-type index below (n-stripes* lump) do
        (let ((scale (if (typep scale 'flt) scale (mref scale stripe))))
          (declare (type flt scale))
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
                                        (/ (aref x* xj) sum))))))))))))

(defmethod derive-lump ((lump ->normalized))
  (let* ((x (x lump))
         (group-size (group-size lump))
         (scale (scale lump)))
    (declare (type index group-size)
             (type (or flt flt-vector) scale))
    (assert (= (size lump) (size x)))
    (assert (= (n-stripes lump) (n-stripes x)))
    (with-facets ((x* ((nodes x) 'backing-array :direction :input
                       :type flt-vector))
                  (xd* ((derivatives x) 'backing-array :direction :io
                        :type flt-vector))
                  (ld* ((derivatives lump) 'backing-array :direction :input
                        :type flt-vector)))
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
                                                   sum-square)))))))))))))))


;;;; Activation lump

(defclass-now ->activation (lump)
  ((weights :type ->weight :initarg :weights :reader weights)
   (x :initarg :x :reader x :documentation "Input comes from here.")
   (transpose-weights-p :initform nil :initarg :transpose-weights-p
                        :reader transpose-weights-p))
  (:documentation "Perform X*WEIGHTS where X is of size M and WEIGHTS
  is a ->WEIGHT whose single stripe is taken to be of dimensions M x N
  stored in row major order. N is the size of this lump. If
  TRANSPOSE-WEIGHTS-P then WEIGHTS is N x M and X*WEIGHTS' is
  computed."))

(defmaker ->activation)

(defun add-activations (&key name size inputs (add-bias-p t)
                        (bpn *bpn-being-built*))
  (when (or add-bias-p inputs)
    (let ((*bpn-being-built* bpn)
          (activation-lumps ()))
      (when add-bias-p
        (let ((w (->weight :name (list :bias name) :size size)))
          (push w activation-lumps)))
      (dolist (input inputs)
        (let* ((input (->lump bpn input))
               (name (list (name input) name))
               (w (->weight :name name :size (* size (size input))))
               (a (->activation :name (list name :activation)
                                :x input :weights w)))
          (push a activation-lumps)))
      (->+ :name (list name :activation) :args (reverse activation-lumps)))))

(defmethod initialize-instance :after ((lump ->activation) &key
                                       &allow-other-keys)
  (assert (= (* (size lump) (size (x lump)))
             (size (weights lump))))
  (setf (slot-value (weights lump) 'dimensions)
        (if (transpose-weights-p lump)
            (list (size lump) (size (x lump)))
            (list (size (x lump)) (size lump))))
  ;; force reshaping
  (setf (max-n-stripes (weights lump)) (max-n-stripes (weights lump))))

(defmethod default-size ((lump ->activation))
  (/ (size (weights lump))
     (size (x lump))))

(defmethod transfer-lump ((lump ->activation))
  (let* ((x (x lump))
         (weights (weights lump))
         (n-stripes (n-stripes lump))
         (nx (size x))
         (nl (/ (size weights) nx)))
    ;; FIXEXT:
    (assert (not (same-stripes-p x)))
    (if (transpose-weights-p lump)
        (gemm! (flt 1) (nodes x) (nodes weights)
                       (flt 0) (nodes lump)
                       :transpose-b? t :lda nx :ldb nx :ldc nl
                       :m n-stripes :n nl :k nx)
        (gemm! (flt 1) (nodes x) (nodes weights)
                       (flt 0) (nodes lump)
                       :lda nx :ldb nl :ldc nl
                       :m n-stripes :n nl :k nx))))

(defmethod derive-lump ((lump ->activation))
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
        ;; dx += a*w
        (gemm! (flt 1) dl* w* (flt 1) dx*
                       :lda nl :ldb nx :ldc nx
                       :m n-stripes :n nx :k nl)
        ;; dx += a*w'
        (gemm! (flt 1) dl* w* (flt 1) dx*
                       :transpose-b? t :lda nl :ldb nl :ldc nx
                       :m n-stripes :n nx :k nl))
    (if (transpose-weights-p lump)
        ;; dw += a'*x
        (gemm! (flt 1) dl* x* (flt 1) dw*
                       :transpose-a? t
                       :lda nl :ldb nx :ldc nx
                       :m nl :n nx :k n-stripes)
        ;; dw += x'*a
        (gemm! (flt 1) x* dl* (flt 1) dw*
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

(defmethod transfer-lump ((lump ->rep))
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
        (loop for stripe of-type index below (n-stripes* lump) do
          (with-stripes ((stripe x xs)
                         (stripe lump ls))
            (dotimes (i xn)
              (let ((v (aref x* (the! index (+ xs i)))))
                (loop for li of-type index upfrom (+ ls i) by xn
                      repeat n
                      do (setf (aref to* li) v))))))))))

(defmethod derive-lump ((lump ->rep))
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
        (loop for stripe of-type index below (n-stripes* lump) do
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

(defmethod transfer-lump ((lump ->stretch))
  (let ((x (x lump)))
    (assert (= (n-stripes lump) (n-stripes x)))
    (let ((n (n lump)))
      (declare (optimize (speed 3) #.*no-array-bounds-check*)
               (type index n))
      (with-facets ((x* ((nodes x) 'backing-array :direction :input
                         :type flt-vector))
                    (l* ((nodes lump) 'backing-array :direction :output
                         :type flt-vector)))
        (loop for stripe of-type index below (n-stripes* lump) do
          (with-stripes ((stripe x xs xe)
                         (stripe lump ls))
            (let ((li ls))
              (loop for xi upfrom xs below xe
                    do (let ((v (aref x* xi)))
                         (loop repeat n
                               do (setf (aref l* li) v)
                                  (incf li)))))))))))

(defmethod derive-lump ((lump ->stretch))
  (let ((x (x lump)))
    (assert (= (n-stripes lump) (n-stripes x)))
    (let ((n (n lump)))
      (declare (optimize (speed 3) #.*no-array-bounds-check*)
               (type index n))
      (with-facets ((xd* ((derivatives x) 'backing-array :direction :io
                          :type flt-vector))
                    (d* ((derivatives lump) 'backing-array :direction :input
                         :type flt-vector)))
        (loop for stripe of-type index below (n-stripes* lump) do
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
  ((args :initarg :args :reader args)
   (ones :initform nil :reader ones)))

(defmaker ->+)

(defmethod default-size ((lump ->+))
  (size (first (args lump))))

(defun ensure-+-ones (lump)
  (let ((n-stripes* (n-stripes* lump)))
    (unless (and (ones lump)
                 (= n-stripes*
                    (mat-size (ones lump))))
      (setf (slot-value lump 'ones)
            (make-mat n-stripes* :ctype flt-ctype))
      (fill! (flt 1) (ones lump))))
  (ones lump))

(defmethod transfer-lump ((lump ->+))
  (let* ((nodes (nodes lump))
         (n-stripes* (n-stripes* lump))
         (ones nil))
    (fill! (flt 0) nodes)
    (dolist (arg (args lump))
      (cond ((= n-stripes* (n-stripes* arg))
             (axpy! (flt 1) (nodes arg) nodes))
            (t
             (assert (same-stripes-p arg))
             (unless ones
               (setq ones (ensure-+-ones lump)))
             (let ((l* (nodes lump))
                   (arg* (nodes arg)))
               (gemm! (flt 1) ones arg* (flt 1) l*
                              :m n-stripes* :n (size lump) :k 1
                              :lda 1 :ldb (size arg) :ldc (size lump))))))))

(defmethod derive-lump ((lump ->+))
  (let* ((n-stripes* (n-stripes* lump))
         (dl (derivatives lump))
         (ones nil))
    (dolist (arg (args lump))
      (let ((darg (derivatives arg)))
        (cond ((= n-stripes* (n-stripes* arg))
               (axpy! (flt 1) dl darg))
              (t
               (assert (same-stripes-p arg))
               (unless ones
                 (setq ones (ensure-+-ones lump)))
               (gemm! (flt 1) ones dl (flt 1) darg
                              :m 1 :n (size arg) :k n-stripes*
                              :lda n-stripes* :ldb (size lump)
                              :ldc (size arg))))))))


(defclass-now ->* (lump)
  ((x :initarg :x :reader x)
   (y :initarg :y :reader y)))

(defmaker ->*)

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
                  (axpy! y (nodes x) to))
                 (t
                  (geem! 1 (nodes x) (nodes y) 0 to))))
          (t
           (assert nil)))))

(defmethod derive-lump ((lump ->*))
  (let* ((n-stripes* (n-stripes* lump))
         (x (x lump))
         (y (y lump)))
    (cond ((= n-stripes* (n-stripes* x))
           (cond ((numberp y)
                  (axpy! y (derivatives lump) (derivatives x)))
                 (t
                  (geem! 1 (derivatives lump) (nodes y) 1 (derivatives x))
                  (geem! 1 (derivatives lump) (nodes x) 1 (derivatives y)))))
          (t
           (assert nil)))))

(define-cuda-kernel (cuda-add-row)
    (void ((x :mat :input) (m int) (n int) (y :mat :io)))
  (let ((i (+ (* block-dim-x block-idx-x) thread-idx-x)))
    (when (< i m)
      (let ((d (aref x i))
            (row-start (* i n)))
        (do ((j 0 (+ j 1)))
            ((>= j n))
          (set (aref y (+ row-start j))
               (+ d (aref y (+ row-start j)))))))))

(defmethod derive-lump ((lump ->sum))
  (let* ((x (x lump))
         (xd (derivatives x))
         (ld (derivatives lump)))
    (assert (= (n-stripes lump) (n-stripes x)))
    (if (use-cuda-p)
        (let ((m (n-stripes lump))
              (n (size x)))
          (cuda-add-row ld m n xd
                        :grid-dim (list (ceiling m 256) 1 1)
                        :block-dim (list 256 1 1)))
        (with-facets ((xd* ((derivatives x) 'backing-array :direction :io
                            :type flt-vector))
                      (ld* ((derivatives lump) 'backing-array :direction :input
                            :type flt-vector)))
          (declare (optimize (speed 3) #.*no-array-bounds-check*))
          (loop for stripe of-type index below (n-stripes* lump) do
            (let ((d (aref ld* stripe)))
              (with-stripes ((stripe x xs xe))
                (loop for xi upfrom xs below xe
                      do (incf (aref xd* xi) d)))))))))

(defclass-now ->abs (lump)
  ((x :initarg :x :reader x)))

(defmaker ->abs)

(defmethod default-size ((lump ->abs))
  (size (x lump)))

(defmethod transfer-lump ((lump ->abs))
  (let ((x (x lump)))
    (assert (= (size lump) (size x)))
    (with-facets ((x* ((nodes x) 'backing-array :direction :input
                       :type flt-vector))
                  (l* ((nodes lump) 'backing-array :direction :output
                       :type flt-vector)))
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
    (with-facets ((x* ((nodes x) 'backing-array :direction :input
                       :type flt-vector))
                  (xd* ((derivatives x) 'backing-array :direction :io
                        :type flt-vector))
                  (ld* ((derivatives lump) 'backing-array :direction :input
                        :type flt-vector)))
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

(defclass-now ->linear (lump)
  ((x :initarg :x :reader x)
   (y :initarg :y :reader y)))

(defmaker ->linear)

(defmethod default-size ((lump ->linear))
  1)

(defmethod transfer-lump ((lump ->linear))
  (let ((x (x lump))
        (y (y lump)))
    (assert (= 1 (size lump)))
    (assert (= (size x) (size y)))
    (with-facets ((x* ((nodes x) 'backing-array :direction :input
                       :type flt-vector))
                  (y* ((nodes y) 'backing-array :direction :input
                       :type flt-vector))
                  (to* ((nodes lump) 'backing-array :direction :output
                        :type flt-vector)))
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
    (with-facets ((x* ((nodes x) 'backing-array :direction :input
                       :type flt-vector))
                  (xd* ((derivatives x) 'backing-array :direction :io
                        :type flt-vector))
                  (y* ((nodes y) 'backing-array :direction :input
                       :type flt-vector))
                  (yd* ((derivatives y) 'backing-array :direction :io
                        :type flt-vector))
                  (dl* ((derivatives lump) 'backing-array :direction :input
                        :type flt-vector)))
      (declare (optimize (speed 3) #.*no-array-bounds-check*))
      (loop for stripe of-type index below (n-stripes* lump) do
        (let ((d (aref dl* stripe)))
          (with-stripes ((stripe x xs xe)
                         (stripe y ys ye))
            (loop for xi upfrom xs below xe
                  for yi upfrom ys below ye
                  do (incf (aref xd* xi)
                           (* d (aref y* yi)))
                     (incf (aref yd* yi)
                           (* d (aref x* xi))))))))))


(defclass-now ->sin (->dropout lump)
  ())

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

(defmethod transfer-lump ((lump ->sin))
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

(defmethod derive-lump ((lump ->sin))
  (let ((x (x lump)))
    (assert (= (size lump) (size x)))
    (sin-derivative! (nodes x) (derivatives lump) (derivatives x))))


(defclass-now ->sigmoid (->dropout lump)
  ())

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

(defmethod transfer-lump ((lump ->sigmoid))
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

(defmethod derive-lump ((lump ->sigmoid))
  (let ((x (x lump)))
    (assert (= (size lump) (size x)))
    (sigmoid-derivative! (nodes lump) (derivatives lump) (derivatives x))))



(defclass-now ->scaled-tanh (lump)
  ((x :initarg :x :reader x)))

(defmaker ->scaled-tanh)

(defmethod default-size ((lump ->scaled-tanh))
  (size (x lump)))

(defmethod transfer-lump ((lump ->scaled-tanh))
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

(defmethod derive-lump ((lump ->scaled-tanh))
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

(define-cuda-kernel (cuda-rectify)
    (void ((x :mat :input) (y :mat :output) (n int)))
  (let ((i (+ (* block-dim-x block-idx-x) thread-idx-x)))
    (when (< i n)
      (let ((xi (aref x i)))
        (if (< xi 0.0)
            (set (aref y i) 0.0)
            (set (aref y i) xi))))))

(defun rectify! (x y &key (n (mat-size x)))
  (assert (eq (mat-ctype x) (mat-ctype y)))
  (assert (<= n (mat-size x)))
  (assert (<= n (mat-size y)))
  (if (use-cuda-p)
      (cuda-rectify x y n
                    :grid-dim (list (ceiling n 256) 1 1)
                    :block-dim (list 256 1 1))
      (with-facets ((x* (x 'backing-array :direction :input
                           :type flt-vector))
                    (y* (y 'backing-array :direction :output
                           :type flt-vector)))
        (dotimes (i n)
          (setf (aref y* i) (max #.(flt 0) (aref x* i)))))))

(defmethod transfer-lump ((lump ->rectified))
  (let ((x (x lump)))
    (assert (= (size lump) (size x)))
    (rectify! (nodes x) (nodes lump))))

(define-cuda-kernel (cuda-rectify-derivative)
    (void ((xd :mat :io) (l :mat :input) (ld :mat :input) (n int)))
  (let ((i (+ (* block-dim-x block-idx-x) thread-idx-x)))
    (when (< i n)
      (when (< 0.0 (aref l i))
        (set (aref xd i) (+ (aref xd i)
                            (aref ld i)))))))

(defmethod derive-lump ((lump ->rectified))
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
        (with-facets ((xd* ((derivatives x) 'backing-array :direction :io
                            :type flt-vector))
                      (l* ((nodes lump) 'backing-array :direction :input
                           :type flt-vector))
                      (ld* ((derivatives lump) 'backing-array :direction :input
                            :type flt-vector)))
          (declare (optimize (speed 3) #.*no-array-bounds-check*))
          (loop for stripe of-type index below (n-stripes* lump) do
            (with-stripes ((stripe lump ls le)
                           (stripe x xs xe))
              (loop for li upfrom ls below le
                    for xi upfrom xs below xe
                    do (when (plusp (aref l* li))
                         (incf (aref xd* xi) (aref ld* li))))))))))


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

(defmethod transfer-lump ((lump ->split-sign))
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

(defmethod derive-lump ((lump ->split-sign))
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
          (loop for stripe of-type index below (n-stripes* lump) do
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

(defmethod transfer-lump ((lump ->softplus))
  (let ((x (x lump)))
    (assert (= (size lump) (size x)))
    (with-facets ((x* ((nodes x) 'backing-array :direction :input
                       :type flt-vector))
                  (l* ((nodes lump) 'backing-array :direction :output
                       :type flt-vector)))
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
    (with-facets ((x* ((nodes x) 'backing-array :direction :input
                       :type flt-vector))
                  (xd* ((derivatives x) 'backing-array :direction :io
                        :type flt-vector))
                  (l* ((nodes lump) 'backing-array :direction :input
                       :type flt-vector))
                  (ld* ((derivatives lump) 'backing-array :direction :input
                        :type flt-vector)))
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


(defclass-now ->exp (lump)
  ((x :initarg :x :reader x)))

(defmaker ->exp)

(defmethod default-size ((lump ->exp))
  (size (x lump)))

(defmethod transfer-lump ((lump ->exp))
  (let ((x (x lump)))
    (assert (= (size lump) (size x)))
    (with-facets ((x* ((nodes x) 'backing-array :direction :input
                       :type flt-vector))
                  (l* ((nodes lump) 'backing-array :direction :output
                       :type flt-vector)))
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
    (with-facets ((xd* ((derivatives x) 'backing-array :direction :io
                        :type flt-vector))
                  (l* ((nodes lump) 'backing-array :direction :input
                       :type flt-vector))
                  (ld* ((derivatives lump) 'backing-array :direction :input
                        :type flt-vector)))
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

(defclass-now ->rough-exponential (lump)
  ((x :initarg :x :reader x)
   (signal-variance :initarg :signal-variance :reader signal-variance)
   (length-scale :initarg :length-scale :reader length-scale)
   (roughness :initarg :roughness :reader roughness)))

(defmaker ->rough-exponential)

(defmethod default-size ((lump ->rough-exponential))
  (size (x lump)))

(defmethod transfer-lump ((lump ->rough-exponential))
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


(defclass-now ->periodic (lump)
  ((x :initarg :x :reader x)
   (period :initarg :period :reader period)))

(defmaker ->periodic)

(defmethod default-size ((lump ->periodic))
  (size (x lump)))

(defmethod transfer-lump ((lump ->periodic))
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
      (loop for stripe of-type index below (n-stripes* lump) do
        (with-stripes ((stripe lump ls le)
                       (stripe x xs xe)
                       (stripe pe pes pee))
          (loop for li upfrom ls below le
                for xi upfrom xs below xe
                for pei upfrom pes below pee
                do (setf (aref l* li)
                         (sin (* #.(flt pi) (/ (aref x* xi)
                                               (aref pe* pei)))))))))))

(defmethod derive-lump ((lump ->periodic))
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

(defmethod transfer-lump ((lump ->ref))
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
      (loop for stripe of-type index below (n-stripes* lump) do
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

(defmethod derive-lump ((lump ->ref))
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
      (loop for stripe of-type index below (n-stripes* lump) do
        (with-stripes ((stripe lump ls le)
                       (stripe index index-s index-e)
                       (stripe into into-s))
          (loop for li upfrom ls below le
                for index-i upfrom index-s below index-e
                do (let ((into-i (round (aref index* index-i))))
                     (when (<= 0 into-i)
                       (incf (aref intod* (+ into-s into-i))
                             (aref d* li))))))))))


(defclass-now ->sum-squared-error (lump)
  ((x :initarg :x :reader x)
   (y :initarg :y :reader y)))

(defmaker ->sum-squared-error)

(defmethod default-size ((lump ->sum-squared-error))
  1)

(defmethod transfer-lump ((lump ->sum-squared-error))
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
      (loop for stripe of-type index below (n-stripes* lump) do
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

(defmethod transfer-lump ((lump ->squared-error))
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
      (loop for stripe of-type index below (n-stripes* lump) do
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

(define-cuda-kernel (cuda-max)
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
        (set (aref y k) max)))))

(defmethod transfer-lump ((lump ->max))
  (let* ((x (x lump))
         (group-size (group-size lump)))
    (declare (type index group-size))
    (if (use-cuda-p)
        (let ((n (mat-size (nodes lump))))
          (cuda-max group-size (nodes x) n (nodes lump)
                    :grid-dim (list (ceiling n 256) 1 1)
                    :block-dim (list 256 1 1)))
        (with-facets ((x* ((nodes x) 'backing-array :direction :input
                           :type flt-vector))
                      (to* ((nodes lump) 'backing-array :direction :output
                            :type flt-vector)))
          (loop for stripe of-type index below (n-stripes* lump) do
            (with-stripes ((stripe lump ls le)
                           (stripe x xs xe))
              (loop for li upfrom ls below le do
                (setf (aref to* li) most-negative-flt))
              (loop for xi upfrom xs below xe
                    for i upfrom 0
                    do (let ((li (+ ls (floor i group-size))))
                         (setf (aref to* li) (max (aref to* li)
                                                  (aref x* xi)))))))))))

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

(defmethod derive-lump ((lump ->max))
  (let* ((x (x lump))
         (group-size (group-size lump)))
    (declare (type index group-size))
    (if (use-cuda-p)
        (let ((n (mat-size (nodes lump))))
          (cuda-max-derivative group-size (nodes x) n (nodes lump)
                               (derivatives lump) (derivatives x)
                               :grid-dim (list (ceiling n 256) 1 1)
                               :block-dim (list 256 1 1)))
        (with-facets ((x* ((nodes x) 'backing-array :direction :input
                           :type flt-vector))
                      (to* ((nodes lump) 'backing-array :direction :input
                            :type flt-vector))
                      (xd* ((derivatives x) 'backing-array :direction :io
                            :type flt-vector))
                      (d* ((derivatives lump) 'backing-array :direction :input
                           :type flt-vector)))
          (loop for stripe of-type index below (n-stripes* lump) do
            (with-stripes ((stripe lump ls)
                           (stripe x xs xe))
              (loop for xi upfrom xs below xe
                    for i upfrom 0
                    do (let ((li (+ ls (floor i group-size))))
                         (when (= (aref to* li)
                                  (aref x* xi))
                           (incf (aref xd* xi)
                                 (aref d* li)))))))))))


;;;; ->MAX-CHANNEL

(defclass-now ->max-channel (lump)
  ((x :initarg :x :reader x :documentation "Input comes from here.")
   (group-size :initarg :group-size :reader group-size)))

(defmaker ->max-channel)

(defmethod default-size ((lump ->max-channel))
  (size (x lump)))

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

(defmethod transfer-lump ((lump ->max-channel))
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

(defmethod derive-lump ((lump ->max-channel))
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


;;;; ->MIN

(defclass-now ->min (lump)
  ((x :initarg :x :reader x :documentation "Input comes from here.")
   (group-size :initarg :group-size :reader group-size)))

(defmaker ->min)

(defmethod default-size ((lump ->min))
  (/ (size (x lump)) (group-size lump)))

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

(defmethod transfer-lump ((lump ->min))
  (let* ((x (x lump))
         (group-size (group-size lump)))
    (declare (type index group-size))
    (if (use-cuda-p)
        (let ((n (mat-size (nodes lump))))
          (cuda-min group-size (nodes x) n (nodes lump)
                    :grid-dim (list (ceiling n 256) 1 1)
                    :block-dim (list 256 1 1)))
        (with-facets ((x* ((nodes x) 'backing-array :direction :input
                           :type flt-vector))
                      (to* ((nodes lump) 'backing-array :direction :output
                            :type flt-vector)))
          (loop for stripe of-type index below (n-stripes* lump) do
            (with-stripes ((stripe lump ls le)
                           (stripe x xs xe))
              (loop for li upfrom ls below le do
                (setf (aref to* li) most-negative-flt))
              (loop for xi upfrom xs below xe
                    for i upfrom 0
                    do (let ((li (+ ls (floor i group-size))))
                         (setf (aref to* li) (min (aref to* li)
                                                  (aref x* xi)))))))))))

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

(defmethod derive-lump ((lump ->min))
  (let* ((x (x lump))
         (group-size (group-size lump)))
    (declare (type index group-size))
    (if (use-cuda-p)
        (let ((n (mat-size (nodes lump))))
          (cuda-min-derivative group-size (nodes x) n (nodes lump)
                               (derivatives lump) (derivatives x)
                               :grid-dim (list (ceiling n 256) 1 1)
                               :block-dim (list 256 1 1)))
        (with-facets ((x* ((nodes x) 'backing-array :direction :input
                           :type flt-vector))
                      (to* ((nodes lump) 'backing-array :direction :input
                            :type flt-vector))
                      (xd* ((derivatives x) 'backing-array :direction :io
                            :type flt-vector))
                      (d* ((derivatives lump) 'backing-array :direction :input
                           :type flt-vector)))
          (loop for stripe of-type index below (n-stripes* lump) do
            (with-stripes ((stripe lump ls)
                           (stripe x xs xe))
              (loop for xi upfrom xs below xe
                    for i upfrom 0
                    do (let ((li (+ ls (floor i group-size))))
                         (when (= (aref to* li)
                                  (aref x* xi))
                           (incf (aref xd* xi)
                                 (aref d* li)))))))))))


;;;; ->SOFTMAX

(defclass-now ->softmax (->normalized)
  ())

(defmaker ->softmax)

(defmethod default-size ((lump ->softmax))
  (size (x lump)))

(defmethod transfer-lump ((lump ->softmax))
  (let* ((x (x lump))
         (group-size (group-size lump))
         (scale (scale lump)))
    (declare (type index group-size)
             (type flt scale))
    (with-facets ((x* ((nodes x) 'backing-array :direction :input
                       :type flt-vector))
                  (to* ((nodes lump) 'backing-array :direction :output
                        :type flt-vector)))
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
                                  (setf (aref to* lj) s)))))))))))

(defmethod derive-lump ((lump ->softmax))
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
      (loop for stripe of-type index below (n-stripes* lump) do
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


;;;; ->CROSS-ENTROPY-SOFTMAX

(defclass-now ->cross-entropy-softmax (lump)
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
    :initarg :class-weights
    :accessor class-weights
    :documentation "If non-NIL, an FLT-VECTOR of GROUP-SIZE. Useful
    TARGET's distribution is different on the training and test sets.
    Just set w_i to test_frequency_i/training_frequency_i.")
   (normalized-lump :reader normalized-lump))
  (:documentation "A specialized lump that is equivalent to hooking
  ->EXP with NORMALIZED-LUMP and ->CROSS-ENTROPY but is numerically
  stable. See
  <http://groups.google.com/group/comp.ai.neural-nets/msg/a7594ebea01fef04?dmode=source>

  It has two parameters X and TARGET. In the transfer phase it
  computes the EXP of each input node and normalizes them as if by
  NORMALIZED-LUMP. These intermediate values are placed into SOFTMAX.
  The value node K is nodes_k = - target_k * ln(softmax_k). Since the
  sum of this is cross entropy: - sum_k target_k * ln(softmax_k),
  simply plug this lump into an ->ERROR.

  In the derive phase it computes the cross entropy error of the
  normalized input: d(-sum_k{target_k * ln(softmax_k)})/dx_k = sum_j{
  target_j * (softmax_k - KDELjk)} which is equal to softmax_k -
  target_k if target sums to 1."))

(defmaker ->cross-entropy-softmax)

(defun add-cross-entropy-softmax (&key predictions-name expectations-name
                                  (error-name (list predictions-name :error))
                                  (add-expectations t)
                                  size inputs (add-bias-p t)
                                  (bpn *bpn-being-built*)
                                  (add-error-p t))
  (let* ((*bpn-being-built* bpn)
         (activations (add-activations :name predictions-name
                                       :size size :inputs inputs
                                       :add-bias-p add-bias-p))
         (expectations (if add-expectations
                           (->input :name expectations-name :size size)
                           (find-lump expectations-name bpn :errorp t)))
         (predictions (->cross-entropy-softmax :name predictions-name
                                               :group-size size
                                               :x activations
                                               :target expectations)))
    (when add-error-p
      (->error :name error-name :x predictions))
    (values predictions expectations)))

(defmethod initialize-instance :after ((lump ->cross-entropy-softmax)
                                       &key &allow-other-keys)
  (unless (slot-boundp lump 'class-weights)
    (setf (slot-value lump 'class-weights)
          (make-mat (group-size lump) :ctype flt-ctype))
    (fill! (flt 1) (class-weights lump))))

(defmethod default-size ((lump ->cross-entropy-softmax))
  (size (x lump)))

(defun ensure-softmax (lump)
  (unless (and (slot-boundp lump 'softmax)
               (= (mat-size (nodes (x lump)))
                  (mat-size (softmax lump))))
    (when (slot-boundp lump 'softmax)
      (mgl-cube:destroy-cube (softmax lump)))
    (setf (slot-value lump 'softmax)
          (make-instance 'mat
                         :ctype flt-ctype
                         :dimensions (mat-dimensions (nodes (x lump))))))
  (softmax lump))

(define-cuda-kernel (cuda-cross-entropy-softmax)
    (void ((group-size int) (x :mat :input) (n int) (target :mat :input)
           (class-weights :mat :input) (cross-entropy :mat :output)
           (softmax :mat :output)))
  (let ((i (* group-size
              (+ (* block-dim-x block-idx-x) thread-idx-x))))
    (when (<= (+ i group-size) n)
      (let ((max (aref x i)))
        ;; It's more stable numerically to subtract the max from
        ;; elements in the group before exponentiating.
        (do ((a 1 (+ a 1)))
            ((>= a group-size))
          (let ((xe (aref x (+ i a))))
            (when (< max xe)
              (set max xe))))
        (let ((sum 0.0))
          (do ((a 0 (+ a 1)))
              ((>= a group-size))
            (let ((xe (aref x (+ i a))))
              (set sum (+ sum (exp (- xe max))))))
          (do ((a 0 (+ a 1)))
              ((>= a group-size))
            (let* ((ia (+ i a))
                   (xe (aref x ia))
                   (s (/ (exp (- xe max)) sum)))
              (set (aref softmax ia) s)
              (set (aref cross-entropy ia)
                   (- (* (aref class-weights a)
                         (aref target ia)
                         (log s)))))))))))

(defmethod transfer-lump ((lump ->cross-entropy-softmax))
  (let* ((x (x lump))
         (group-size (group-size lump))
         (softmax (ensure-softmax lump))
         (target (target lump))
         (class-weights (class-weights lump)))
    (declare (type index group-size))
    (if (use-cuda-p)
        (let ((n (* (n-stripes lump) (size lump))))
          (cuda-cross-entropy-softmax
           group-size (nodes x) n (nodes target)
           class-weights (nodes lump) softmax
           :grid-dim (list (ceiling (/ n group-size) 256) 1 1)
           :block-dim (list 256 1 1)))
        (with-facets ((x* ((nodes x) 'backing-array :direction :input
                           :type flt-vector))
                      (target* ((nodes target) 'backing-array :direction :input
                                :type flt-vector))
                      (to* ((nodes lump) 'backing-array :direction :output
                            :type flt-vector))
                      (softmax* (softmax 'backing-array :direction :output
                                         :type flt-vector))
                      (class-weights* (class-weights 'backing-array
                                                     :direction :input
                                                     :type flt-vector)))
          (loop for stripe of-type index below (n-stripes* lump) do
            (with-stripes ((stripe lump ls le)
                           (stripe x xs xe)
                           (stripe target ts te))
              (loop for li upfrom ls below le
                    for xi upfrom xs below xe
                    for ti upfrom ts below te
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
                           (loop for lj upfrom li below (+ li group-size)
                                 for xj upfrom xi below (+ xi group-size)
                                 for tj upfrom ti below (+ ti group-size)
                                 for i below group-size
                                 do (let ((s (/ (exp (- (aref x* xj) max))
                                                sum)))
                                      (declare (type positive-flt s))
                                      (setf (aref softmax* lj) s)
                                      (setf (aref to* lj)
                                            (- (* (aref class-weights* i)
                                                  (aref target* tj)
                                                  (the flt
                                                       (log s))))))))))))))))

(define-cuda-kernel (cuda-cross-entropy-softmax-derivative)
    (void ((group-size int) (xd :mat :io) (n int) (d :mat :input)
           (target :mat :input) (class-weights :mat :input)
           (softmax :mat :input)))
  (let ((i (* group-size
              (+ (* block-dim-x block-idx-x) thread-idx-x))))
    (when (<= (+ i group-size) n)
      (do ((a 0 (+ a 1)))
          ((>= a group-size))
        (let ((ia (+ i a)))
          (do ((b 0 (+ b 1)))
              ((>= b group-size))
            (let ((ib (+ i b)))
              (set (aref xd ia)
                   (+ (aref xd ia)
                      (* (aref d ib)
                         (aref class-weights b)
                         (aref target ib)
                         (- (aref softmax ia)
                            (if (= a b)
                                1.0
                                0.0))))))))))))

(defmethod derive-lump ((lump ->cross-entropy-softmax))
  (let* ((x (x lump))
         (group-size (group-size lump))
         (softmax (softmax lump))
         (target (target lump))
         (class-weights (class-weights lump)))
    (declare (type index group-size))
    (if (use-cuda-p)
        (let ((n (* (n-stripes lump) (size lump))))
          (cuda-cross-entropy-softmax-derivative
           group-size (derivatives x) n (derivatives lump) (nodes target)
           class-weights softmax
           :grid-dim (list (ceiling (/ n group-size) 256) 1 1)
           :block-dim (list 256 1 1)))
        (with-facets ((xd* ((derivatives x) 'backing-array :direction :io
                            :type flt-vector))
                      (target* ((nodes target) 'backing-array :direction :input
                                :type flt-vector))
                      (d* ((derivatives lump) 'backing-array :direction :input
                           :type flt-vector))
                      (softmax* (softmax 'backing-array :direction :input
                                         :type flt-vector))
                      (class-weights* (class-weights 'backing-array
                                                     :direction :input
                                                     :type flt-vector)))
          ;; FIXEXT: target derivative not calculated
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
                                        (aref class-weights* i)
                                        (aref target* ti)
                                        (- (aref softmax* lj)
                                           (if (= ti tj)
                                               #.(flt 1)
                                               #.(flt 0)))))))))))))))


(defmethod label-indices ((lump ->cross-entropy-softmax))
  (max-row-positions (softmax lump)))

(defmethod label-index-distributions ((lump ->cross-entropy-softmax))
  (rows-to-arrays (softmax lump)))


;;;; RENORMALIZE-ACTIVATIONS

(define-cuda-kernel (cuda-add-row-norms)
    (void ((x :mat :input) (n-rows int) (n-columns int) (y :mat :io)))
  (let ((i (+ (* block-dim-x block-idx-x) thread-idx-x)))
    (when (<= i n-rows)
      (let ((sum 0.0)
            (row-start (* i n-columns)))
        (do ((j 0 (+ j 1)))
            ((>= j n-columns))
          (set sum (+ sum
                      (expt (aref x (+ row-start j)) 2.0))))
        (set (aref y i) (+ (aref y i) sum))))))

(defun add-row-norms (x y)
  (let ((n-rows (mat-dimension x 0))
        (n-columns (mat-dimension x 1)))
    (declare (type fixnum n-rows n-columns))
    (assert (= n-rows (mat-size y)))
    (if (use-cuda-p)
        (let ((n n-rows))
          (cuda-add-row-norms x n-rows n-columns y
                              :grid-dim (list (ceiling n 256) 1 1)
                              :block-dim (list 256 1 1)))
        (with-facets ((x* (x 'backing-array :direction :input
                             :type flt-vector))
                      (y* (y 'backing-array :direction :io
                             :type flt-vector)))
          (dotimes (row n-rows)
            (let ((i (* row n-columns))
                  (sum #.(flt 0)))
              (declare (optimize (speed 3) #.*no-array-bounds-check*))
              (loop repeat n-columns do
                (incf sum (expt (aref x* i) 2))
                (incf i))
              (incf (aref y* row) sum)))))))

(define-cuda-kernel (cuda-add-column-norms)
    (void ((x :mat :input) (n-rows int) (n-columns int) (y :mat :io)))
  (let ((i (+ (* block-dim-x block-idx-x) thread-idx-x)))
    (when (<= i n-columns)
      (let ((sum 0.0)
            (k i))
        (do ((j 0 (+ j 1)))
            ((>= j n-rows))
          (set sum (+ sum (expt (aref x k) 2.0)))
          (set k (+ k n-columns)))
        (set (aref y i) (+ (aref y i) sum))))))

(defun add-column-norms (x y)
  (let ((n-rows (mat-dimension x 0))
        (n-columns (mat-dimension x 1)))
    (declare (type fixnum n-rows n-columns))
    (assert (= n-columns (mat-size y)))
    (if (use-cuda-p)
        (let ((n n-columns))
          (cuda-add-column-norms x n-rows n-columns y
                                 :grid-dim (list (ceiling n 256) 1 1)
                                 :block-dim (list 256 1 1)))
        (with-facets ((x* (x 'backing-array :direction :input
                             :type flt-vector))
                      (y* (y 'backing-array :direction :io
                             :type flt-vector)))
          (dotimes (column n-columns)
            (let ((i column)
                  (sum #.(flt 0)))
              (declare (optimize (speed 3) #.*no-array-bounds-check*)
                       (type index i))
              (loop repeat n-rows do
                (incf sum (expt (aref x* i) 2))
                (incf i n-columns))
              (setf (aref y* column) sum)))))))

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
        (with-facets ((mat* (mat 'backing-array :direction :io
                                 :type flt-vector))
                      (norms* (norms 'backing-array :direction :input
                                     :type flt-vector)))
          (dotimes (i n-rows)
            (when (< l2-upper-bound (aref norms* i))
              (let ((scale (/ l2-upper-bound (+ (aref norms* i)
                                                (flt 0.0000001))))
                    (k (* i n-columns)))
                (dotimes (j n-columns)
                  (setf (aref mat* k) (* scale (aref mat* k)))
                  (incf k)))))))))

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
        (with-facets ((mat* (mat 'backing-array :direction :io
                                 :type flt-vector))
                      (norms* (norms 'backing-array :direction :input
                                     :type flt-vector)))
          (dotimes (j n-columns)
            (when (< l2-upper-bound (aref norms* j))
              (let ((scale (/ l2-upper-bound (+ (aref norms* j)
                                                (flt 0.0000001))))
                    (k j))
                (dotimes (i n-rows)
                  (setf (aref mat* k) (* scale (aref mat* k)))
                  (incf k n-columns)))))))))

(defun mat-and-row/column-sum-size (mat-and-row/column-list)
  (if mat-and-row/column-list
      (destructuring-bind (mat row/column) (first mat-and-row/column-list)
        (ecase row/column
          ((:row) (mat-dimension mat 0))
          ((:column) (mat-dimension mat 1))))
      0))

(defun renormalize-mats (mat-and-row/column-list l2-upper-bound)
  (let ((n (mat-and-row/column-sum-size mat-and-row/column-list))
        (firstp t))
    (with-thread-cached-mat (sums (list n 1) :ctype flt-ctype)
      (loop for (mat row/column) in mat-and-row/column-list
            do (with-thread-cached-mat
                   (square (mat-dimensions mat) :ctype flt-ctype
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

(defun renormalize-activations (->activations l2-upper-bound)
  "If the l2 norm of the incoming weight vector of the a unit is
  larger than L2-UPPER-BOUND then renormalize it to L2-UPPER-BOUND.
  The list of ->ACTIVATIONS is assumed to be eventually fed to the
  same lump.

  To use it, group the activation lumps into the same GD-OPTIMIZER and
  hang this function on AFTER-UPDATE-HOOK, that latter of which is
  done for you ARRANGE-FOR-RENORMALIZING-ACTIVATIONS.

  See \"Improving neural networks by preventing co-adaptation of
  feature detectors (Hinton, 2012)\",
  <http://arxiv.org/pdf/1207.0580.pdf>."
  (when (and ->activations l2-upper-bound)
    (renormalize-mats
     (loop for lump in ->activations
           collect (let ((weights (etypecase lump
                                    (->activation (weights lump))
                                    (->weight lump))))
                     (list (nodes weights)
                           (if (and (typep lump '->activation)
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
  (push (let ((->activations nil)
              (firstp t))
          (lambda ()
            (when firstp
              (setq ->activations
                    (loop for lump in (segments optimizer)
                          collect (or (find-activation-lump-for-weight lump bpn)
                                      lump)))
              (setq firstp nil))
            (renormalize-activations ->activations l2-upper-bound)))
        (after-update-hook optimizer)))

(defun find-activation-lump-for-weight (->weight bpn)
  (loop for lump across (lumps bpn) do
    (when (and (typep lump '->activation)
               (eq (weights lump) ->weight))
      (return lump))))


;;;; Utilities

(defun monitor-bpn-results (dataset bpn monitors)
  (monitor-model-results (lambda (batch)
                           ;; FIXME: DO-EXECUTORS belongs elsewhere.
                           (do-executors (batch bpn)
                             (set-input batch bpn)
                             (forward-bpn bpn))
                           bpn)
                         dataset bpn monitors))

(defmethod make-classification-accuracy-monitors*
    ((bpn bpn) operation-mode label-index-fn attributes)
  (let ((attributes `(,@attributes :model "bpn")))
    (loop for lump across (lumps bpn)
          nconc (make-classification-accuracy-monitors* lump operation-mode
                                                        label-index-fn
                                                        attributes))))

(defmethod make-cross-entropy-monitors* ((bpn bpn) operation-mode
                                         label-index-distribution-fn attributes)
  (let ((attributes `(,@attributes :model "bpn")))
    (loop for lump across (lumps bpn)
          nconc (make-cross-entropy-monitors* lump operation-mode
                                              label-index-distribution-fn
                                              attributes))))

(defmethod make-cost-monitors* ((bpn bpn) operation-mode attributes)
  (let ((attributes `(,@attributes :model "bpn")))
    (loop for lump across (lumps bpn)
          nconc (make-cost-monitors* lump operation-mode attributes))))
