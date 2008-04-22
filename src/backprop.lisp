;;;; Backpropagation for feed-forward networks.
;;;;
;;;; As usual the design tries to strike a balance between efficiency
;;;; and flexibility. There is no constraint on the network topology
;;;; except that it must be acyclic. Networks can be defined in a
;;;; compact form by describing `lumps' of nodes. All nodes in a lump
;;;; have the same transfer function. Parameters of the transfer
;;;; function are connected to nodes in preceeding lumps.
;;;;
;;;; It turns out that on sizable problems - such as the mnist one -
;;;; performance is memory bound. Doing basically the same thing in
;;;; different order this implementation performs 10 times worse than
;;;; the matlab code of Geoffrey Hinton. The solution is to implement
;;;; batch processing which would allow the memory/cache aware BLAS 3
;;;; routines work their magic.


;;;; To take advantage of BLAS 3 for matrix matrix multiplication
;;;; which is by far the tightest bottleneck weight and input matrices
;;;; have to be stored continuously (with a possible pun using the LDA
;;;; parameter of DGEMM). Locality of reference should be better if it
;;;; is really continuous. It follows that lumps shall have their
;;;; separate value/derivative vectors. Addressing into these shall
;;;; observe a special offset (the index of its `stripe') and
;;;; NODEWISE-LUMPs shall simply loop over their stripes changing the
;;;; effective offset in the process. ACTIVATION-LUMPs can call DGEMM.
;;;;
;;;; ARGLISTs then must refer to a cell in a lump. Looking up that
;;;; value takes the stripe into account.
;;;;
;;;; Weight lumps have only one stripe for values (i.e. they ignore
;;;; the stripe) but possibly many for the derivatives.
;;;; INDICES-TO-CALCULATE are per stripe (they prevent usage of BLAS
;;;; though). All non-weight lumps are supposed to have the same
;;;; number of stripes. This suggests that they cloud be different
;;;; lumps as well but then lookup would be really slow.
;;;;
;;;; Another way is to supply a MAX-N-STRIPES argument at creation
;;;; time that sets up and N-STRIPES to propagation. Reference

(in-package :mgl-bp)

;;;; Construction of argument lists

;;; Share some objects of the arglists to reduce memory footprint.
;;; This hash table is used only when constructing the network.
(defvar *arglist-objects*)

(defclass range ()
  ((start :initarg :start :reader start)
   (end :initarg :end :reader end)))

(defgeneric ref (sequence index)
  (:documentation "Index into SEQUENCE. Like ELT/AREF but works on
more types.")
  (:method ((range range) index)
    (assert (<= 0 index (- (end range) (start range) 1)))
    (+ (start range) index))
  (:method ((seq sequence) index)
    (elt seq index)))

(defgeneric sub (sequence start end)
  (:documentation "Return a subsequence of SEQUENCE. Like SUBSEQ but
works with more types.")
  (:method ((range range) start end)
    (assert (<= 0 start end))
    (let ((old-start (start range))
          (old-end (end range)))
      (assert (< start (- old-end old-start)))
      (assert (<= end (- old-end old-start)))
      (make-instance 'range
                     :start (+ old-start start)
                     :end (+ old-start end))))
  (:method ((seq sequence) start end)
    (subseq seq start end)))

(defgeneric col (sequence column row-size)
  (:documentation "Return a subsequence of SEQUENCE. Like SUBSEQ but
works with more types.")
  (:method ((range range) column row-size)
    (let* ((size (ceiling (- (end range) (+ (start range) column)) row-size))
           (v (make-array size :element-type 'index)))
      (loop for i upfrom (+ (start range) column) below (end range) by row-size
            for j upfrom 0
            do (setf (aref v j) i))
      v))
  (:method ((seq sequence) column row-size)
    (let* ((size (ceiling (- (length seq) column) row-size))
           (v (make-array size :element-type 'index)))
      (loop for i upfrom column below (length seq) by row-size
            for j upfrom 0
            do (setf (elt v j) i))
      v)))

(defun enumerate-range (start end)
  (let* ((length (- end start))
         (v (make-array length :element-type 'index)))
    (loop for i below length
          do (setf (aref v i) (+ start i)))
    v))

(defgeneric resolve (object)
  (:method (object)
    object)
  (:method ((range range))
    (if (boundp '*arglist-objects*)
        (let ((key (list :range (start range) (end range))))
          (or (gethash key *arglist-objects*)
              (setf (gethash key *arglist-objects*)
                    (enumerate-range (start range) (end range)))))
        (enumerate-range (start range) (end range)))))


;;;; Node

(eval-when (:compile-toplevel :load-toplevel)

  (defun transfer-name (name)
    (suffix-symbol name '-%transfer))

  (defun derivate-name (name)
    (suffix-symbol name '-%derivate))

  (defun lump-definer-fn (name var size body)
    `(lambda (,size)
       (values ',name
               (coerce (loop for ,var below ,size
                             collect (mapcar #'resolve (list ,@body)))
                       'vector)))))

(defmacro define-lump-definer (name)
  "Lump definer provides syntactic sugar to compactly define a node
type and arglists for lumps:

  (define-lump-definer ->linear)

  (funcall (->linear (_) _ (1+ _)) 3) => linear, #((0 1) (1 2) (2 3))"
  (with-gensyms (%var %body)
    `(defmacro ,name ((,%var) &body ,%body)
       (with-gensyms (%size)
         (lump-definer-fn ',name ,%var %size ,%body)))))

(defmacro define-node-type (name args
                            (transfer-marker &body transfer)
                            (derivate-marker &body derivate))
  "Define a node type with NAME and of ARGS. A node type has a
transfer and a derivate function of the same ARGS. The transfer
function can access the node array of the network with the NODE macro
and shall return a value of type FLT that is going to be new value of
its node. The derivate function can also use ADD-DERIVATIVE and its
return value is discarded.

A lump definer macro that of NAME is defined that can be used to
conveniently initialize the transfer/derivate function and connections
of a lump. By convention NAME starts with ->."
  (assert (eq ':transfer transfer-marker))
  (assert (eq ':derivate derivate-marker))
  (multiple-value-bind (derivate-decls derivate-body) (split-body derivate)
    `(progn
       (define-lump-definer ,name)
       (defun ,(transfer-name name) (%nodes ,@args)
         (declare (type flt-vector %nodes)
                  (ignorable %nodes))
         ,@transfer)
       (defun ,(derivate-name name)
           (%nodes %derivatives %index ,@args)
         (declare (type flt-vector %nodes %derivatives)
                  (type index %index)
                  (ignorable %nodes))
         ,@derivate-decls
         (let ((%current-derivative (aref %derivatives %index)))
           (unless (zerop %current-derivative)
             ,@derivate-body))))))

(defmacro node (i)
  "To be used within a transfer function."
  `(aref %nodes ,i))

(defmacro add-derivative (i value)
  "To be used within the transfer or derivate function."
  `(progn
     (incf (aref %derivatives ,i) (* %current-derivative ,value))
     (values)))


;;;; Lump

(defclass lump (range)
  ((size :initform (error "SIZE not specified") :initarg :size :reader size)
   (bpn :initarg :bpn :reader bpn)
   (name :type symbol :initarg :name :reader name)
   (indices-to-calculate
    :initform nil :initarg :indices-to-calculate :type (or null index-vector)
    :accessor indices-to-calculate
    :documentation "NIL or a simple vector of array indices into this
lump's range (i.e. in the 0 (1- SIZE) interval). Need not be ordered.
If not NIL the node's value is not calculated and its derivatives are
not propagated unless it is in INDICES-TO-CALCULATE. It has no effect
subsequent lumps: they may use values that have not been recalculated.
The primary use-case is to temporarily mask out an uniteresting part
of the network.")))

(defun lump-node-array (lump)
  "Return an array and start, end indices of the nodes in LUMP."
  (values (nodes (bpn lump)) (start lump) (end lump)))

(defclass nodewise-lump (lump)
  ((arglists
    :type simple-vector :initarg :arglists :reader arglists
    :documentation "This defines the network structure. It is a vector
of argument lists, one for each node in the lump. Elements of an
argument list are node indices or simple vectors of node indices. See
type INDEX and INDEX-VECTOR.")
   (transfer-fn :initarg :transfer-fn :reader transfer-fn)
   (derivate-fn :initarg :derivate-fn :reader derivate-fn))
  (:documentation "Nodes are calculated one by one independently."))

(defmethod initialize-instance :after ((lump nodewise-lump)
                                       &key def &allow-other-keys)
  (when def
    ;; def is a lump definer
    (multiple-value-bind (node-type-name arglists)
        (funcall def (size lump))
      (setf (slot-value lump 'transfer-fn) (transfer-name node-type-name))
      (setf (slot-value lump 'derivate-fn) (derivate-name node-type-name))
      (setf (slot-value lump 'arglists) arglists))))

(defclass data-lump (lump) ())
(defclass input-lump (data-lump) ())
(defclass weight-lump (data-lump) ())
(defclass constant-lump (data-lump)
  ((value :initarg :value :reader value)))
;;; FIXME: better name
(defclass hidden-lump (nodewise-lump) ())
(defclass output-lump (nodewise-lump) ())
;;; FIXME: ERROR-NODE must have only one node. Well, except for
;;; CROSS-ENTROPY-SOFTMAX-LUMP.
(defclass error-node (nodewise-lump)
  ((size :initform 1)
   (importance
    :initform (flt 1) :initarg :importance :accessor importance
    :documentation "Error nodes have their incoming derivative set to
IMPORTANCE no matter what other nodes depend on them.")))

(defclass normalized-lump (lump)
  ((group-size :initarg :group-size :reader group-size)
   (scale
    :initform #.(flt 1) :initarg :scale :accessor scale :type flt
    :documentation "The sum of node values per group after
normalization. Can be changed during training, for instance when
clamping.")
   (normalized-lump :initarg :normalized-lump :reader normalized-lump)))

(defclass cross-entropy-softmax-lump (error-node lump)
  ((group-size :initarg :group-size :reader group-size)
   (input-lump
    :initarg :input-lump :reader input-lump
    :documentation "This is EXP'd and normalized.")
   (target-lump
    :initarg :target-lump :reader target-lump
    :documentation "A lump of the same size as INPUT-LUMP that is the
T in -sum_{k}T_k*ln(I_k) which the the cross entropy error.")
   (normalized-lump :reader normalized-lump))
  (:documentation "A specialized lump that is equivalent to hooking
->EXP with NORMALIZED-LUMP and ->CROSS-ENTROPY but is numerically
stable. See http://groups.google.com/group/comp.ai.neural-nets/msg/a7594ebea01fef04?dmode=source"))

(defun lump-size (lump)
  (- (end lump) (start lump)))

(defmethod print-object ((lump lump) stream)
  (print-unreadable-object (lump stream :type t :identity t)
    (format stream "~S ~S ~S" (name lump) :size (lump-size lump))
    (when (indices-to-calculate lump)
      (format stream "(~S)" (length (indices-to-calculate lump)))))
  lump)

(defmacro do-lump ((var lump) &body body)
  "Iterate over the indices of nodes of LUMP skipping missing ones."
  (with-gensyms (%lump %indices-to-calculate %size)
    `(let* ((,%lump ,lump)
            (,%indices-to-calculate (indices-to-calculate ,%lump)))
       (if ,%indices-to-calculate
           (locally (declare (type index-vector ,%indices-to-calculate))
             (loop for ,var across ,%indices-to-calculate
                   do (progn ,@body)))
           (let ((,%size (lump-size ,%lump)))
             (declare (type index ,%size))
             (loop for ,var fixnum below ,%size
                   do (progn ,@body)))))))

(defgeneric transfer-lump (lump nodes)
  (:method ((lump data-lump) nodes)
    (declare (ignore nodes)))
  (:method ((lump normalized-lump) nodes)
    (declare (type flt-vector nodes))
    (let* ((output-start (start lump))
           (input (normalized-lump lump))
           (input-start (start input))
           (group-size (group-size lump))
           (scale (scale lump)))
      (declare (type index output-start input-start group-size)
               (type flt scale))
      (assert (= (lump-size lump) (lump-size input)))
      (do-lump (index lump)
        (when (zerop (mod index group-size))
          (let* ((input-group-start (+ input-start index))
                 (output-group-start (+ output-start index))
                 (input-group-end (+ input-group-start group-size))
                 (output-group-end (+ output-group-start group-size))
                 (sum #.(flt 0)))
            (declare (type flt sum)
                     (type index input-group-start output-group-start
                           input-group-end output-group-end))
            (locally
                (declare (optimize (speed 3)))
              (loop for i fixnum upfrom input-group-start below input-group-end
                    do (incf sum (aref nodes i)))
              (setq sum (/ sum scale))
              (loop for i upfrom input-group-start below input-group-end
                    for j upfrom output-group-start below output-group-end
                    do (setf (aref nodes j) (/ (aref nodes i) sum)))))))))
  (:method ((lump cross-entropy-softmax-lump) nodes)
    (declare (type flt-vector nodes))
    (let* ((output-start (start lump))
           (input (input-lump lump))
           (input-start (start input))
           (group-size (group-size lump)))
      (declare (type index output-start input-start group-size))
      (assert (= (lump-size lump) (lump-size input)))
      (do-lump (index lump)
        (when (zerop (mod index group-size))
          (let* ((input-group-start (+ input-start index))
                 (output-group-start (+ output-start index))
                 (input-group-end (+ input-group-start group-size))
                 (output-group-end (+ output-group-start group-size))
                 (sum #.(flt 0)))
            (declare (type flt sum)
                     (type index input-group-start output-group-start
                           input-group-end output-group-end))
            (locally
                (declare (optimize (speed 3)))
              (loop for i fixnum upfrom input-group-start below input-group-end
                    do (incf sum (exp (aref nodes i))))
              (loop for i upfrom input-group-start below input-group-end
                    for j upfrom output-group-start below output-group-end
                    do (setf (aref nodes j)
                             (/ (exp (aref nodes i)) sum)))))))))
  (:method ((lump nodewise-lump) nodes)
    (declare (type flt-vector nodes)
             (optimize (speed 3) #.*no-array-bounds-check*))
    (let ((transfer-fn (coerce (transfer-fn lump) 'function))
          (arglists (arglists lump))
          (start (start lump)))
      (declare (type simple-vector arglists)
               (type index start))
      (do-lump (index lump)
        (setf (aref nodes (#.*the* index (+ start index)))
              (apply transfer-fn nodes (aref arglists index)))))))

(defgeneric derivate-lump (lump nodes derivatives)
  (:method ((lump lump) nodes derivatives)
    (declare (ignore nodes derivatives)))
  (:method ((lump error-node) nodes derivatives)
    (setf (aref derivatives (start lump)) (importance lump))
    (call-next-method))
  (:method ((lump normalized-lump) nodes derivatives)
    (declare (type flt-vector nodes derivatives))
    (let* ((output-start (start lump))
           (input (normalized-lump lump))
           (input-start (start input))
           (group-size (group-size lump))
           (scale (scale lump)))
      (declare (type index output-start input-start group-size)
               (type flt scale))
      (assert (= (lump-size lump) (lump-size input)))
      (do-lump (index lump)
        (when (zerop (mod index group-size))
          (let* ((input-group-start (+ input-start index))
                 (output-group-start (+ output-start index))
                 (input-group-end (+ input-group-start group-size))
                 (output-group-end (+ output-group-start group-size))
                 (sum #.(flt 0)))
            (declare (type flt sum)
                     (type index input-group-start output-group-start
                           input-group-end output-group-end))
            (locally
                (declare (optimize (speed 3)))
              (loop for i fixnum upfrom input-group-start below input-group-end
                    do (incf sum (aref nodes i)))
              (setq sum (/ sum scale))
              (loop for i upfrom input-group-start below input-group-end
                    for j upfrom output-group-start below output-group-end
                    do (incf (aref derivatives i)
                             (* (aref derivatives j)
                                (let* ((x (aref nodes i))
                                       (a (- sum x)))
                                  (/ a (+ x (* a a)) scale)))))))))))
  (:method ((lump cross-entropy-softmax-lump) nodes derivatives)
    (let* ((input (input-lump lump))
           (target (target-lump lump))
           (input-start (start input))
           (input-end (end input))
           (importance (importance lump)))
      (assert (= (lump-size input) (lump-size target) (lump-size lump)))
      (loop for input-index upfrom input-start below input-end
            for target-index upfrom (start target)
            for lump-index upfrom (start lump)
            do
            ;; FIXME: target derivative not calculated
            (incf (aref derivatives input-index)
                  (* importance
                     (- (aref nodes lump-index) (aref nodes target-index)))))))
  (:method ((lump nodewise-lump) nodes derivatives)
    (declare (type flt-vector nodes derivatives)
             (optimize (speed 3) #.*no-array-bounds-check*))
    (let ((derivate-fn (coerce (derivate-fn lump) 'function))
          (arglists (arglists lump))
          (start (start lump)))
      (declare (type simple-vector arglists)
               (type index start))
      (do-lump (index lump)
        (apply derivate-fn nodes derivatives (#.*the* index (+ start index))
               (aref arglists index))))))


;;;; BPN

(defclass bpn ()
  ((lumps
    :initform (make-array 0 :element-type 'lump :adjustable t :fill-pointer t)
    :type (array lump (*)) :reader lumps
    :documentation "Lumps in reverse order")
   (nodes
    :type (or flt-vector null) :reader nodes
    :documentation "The values of the nodes. All nodes have values.")
   (derivatives
    :type (or flt-vector null) :reader derivatives
    :documentation "Derivatives at nodes, input node derivative are
not calculated.")))

(defun find-lump (name bpn &key errorp)
  (or (find name (lumps bpn) :key #'name :test #'equal)
      (if errorp
          (error "Cannot find lump ~S." name)
          nil)))

(defun bpn-size (bpn)
  "Total number of nodes in BPN."
  (let ((lumps (lumps bpn)))
    (if (zerop (length lumps))
        0
        (end (last1 lumps)))))

(defgeneric initialize-lump (lump bpn)
  (:documentation "Perform any initialization on LUMP in BPN that is
also initialized.")
  (:method ((lump lump) bpn)
    (declare (ignore bpn)))
  (:method ((lump constant-lump) bpn)
    (multiple-value-bind (array start end)
        (lump-node-array lump)
      (fill array (flt (value lump)) :start start :end end))))

(defun initialize-bpn (bpn)
  "Make sure that node vectors have enough capacity in light of newly
added lumps."
  (let ((size (bpn-size bpn)))
    (unless (and (slot-boundp bpn 'nodes)
                 (= size (length (nodes bpn))))
      (setf (slot-value bpn 'nodes)
            (make-array size :element-type 'flt :initial-element #.(flt 0)))
      (loop for lump across (lumps bpn)
            do (initialize-lump lump bpn)))
    (unless (and (slot-boundp bpn 'derivatives)
                 (= size (length (derivatives bpn))))
      (setf (slot-value bpn 'derivatives)
            (make-array size :element-type 'flt :initial-element #.(flt 0)))))
  bpn)

(defun add-lump (lump bpn)
  (when (slot-boundp lump 'bpn)
    (error "Lump ~S is already added to a bpn." lump))
  (when (find-lump (name lump) bpn)
    (error "Cannot add ~S: ~%
            a lump of same name has already been added to this network." lump))
  (let ((start (bpn-size bpn)))
    (setf (slot-value lump 'start) start
          (slot-value lump 'end) (+ start (size lump))
          (slot-value lump 'bpn) bpn)
    (vector-push-extend lump (slot-value bpn 'lumps)))
  lump)

(defmacro build-bpn ((&key (class ''bpn) initargs (initializep t)) &body lumps)
  (with-gensyms (%bpn)
    (let ((bindings
           (mapcar (lambda (lump)
                     (destructuring-bind (class &rest args) lump
                       (multiple-value-bind (known unknown)
                           (split-plist args '(:name :symbol))
                         (destructuring-bind (&key (symbol (gensym))
                                                   (name (list 'quote symbol)))
                             known
                           `(,symbol
                             (add-lump (make-instance ',class :name ,name
                                        ,@unknown)
                              ,%bpn))))))
                   lumps)))
      `(let ((*arglist-objects* (make-hash-table :test #'equal))
             (,%bpn (make-instance ,class ,@initargs)))
         (flet ((lump (name)
                  (find-lump name ,%bpn :errorp t)))
           (let* ,bindings
             (declare (ignorable ,@(mapcar #'first bindings)))
             ;; prevent warning if LUMP goes unused
             #'lump))
         (if ,initializep
             (initialize-bpn ,%bpn)
             ,%bpn)))))

(defun ->lump (bpn lump-spec)
  (if (typep lump-spec 'lump)
      lump-spec
      (find-lump lump-spec bpn :errorp t)))

(defun forward-bpn (bpn &key from-lump to-lump)
  "Propagate the values from the already clamped inputs."
  (declare (optimize (debug 2)))
  (initialize-bpn bpn)
  (let ((from-lump (if from-lump (->lump bpn from-lump) nil))
        (to-lump (if to-lump (->lump bpn to-lump) nil))
        (nodes (nodes bpn))
        (seen-from-lump-p (not from-lump)))
    (loop for lump across (lumps bpn)
          do (when (eq lump from-lump) (setq seen-from-lump-p t))
          do (when seen-from-lump-p (transfer-lump lump nodes))
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
          do (multiple-value-bind (array start end) (segment-derivatives lump)
               (declare (type flt-vector array)
                        (type index start end)
                        (optimize (speed 3)))
               (fill array #.(flt 0) :start start :end end)))))

(defun backward-bpn (bpn &key (last-lump nil last-lump-p))
  "Accumulate derivatives of weights."
  (initialize-bpn bpn)
  (let* ((nodes (nodes bpn))
         (derivatives (derivatives bpn))
         (lumps (lumps bpn))
         (last-lump (if last-lump-p (->lump bpn last-lump) nil)))
    (apply #'zero-non-weight-derivatives bpn
           (if last-lump-p
               (list :last-lump last-lump)
               nil))
    (loop for i downfrom (1- (length lumps)) downto 0
          for lump = (aref lumps i)
          until (and last-lump-p (eq last-lump lump))
          do (derivate-lump lump nodes derivatives))))


;;;; Train

(defclass base-bp-trainer ()
  ((first-trained-lump :reader first-trained-lump)))

(defclass bp-trainer (base-bp-trainer segmented-trainer) ())

(defclass cg-bp-trainer (base-bp-trainer batch-cg-trainer) ())

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
  (values (nodes (bpn lump)) (start lump) (end lump)))

(defmethod segment-derivatives ((lump lump))
  (values (derivatives (bpn lump)) (start lump) (end lump)))

(defun first-trained-weight-lump (trainer bpn)
  "Much time can be wasted computing derivatives of non-trained weight
lumps. Return the first one that TRAINER trains."
  (or (slot-value trainer 'first-trained-lump)
      (setf (slot-value trainer 'first-trained-lump)
            (find-if (lambda (lump)
                       (member lump (segments trainer)))
                     (lumps bpn)))))

(defun cost (bpn)
  (last1 (nodes bpn)))

(defun compute-cost (sample trainer bpn)
  (set-input sample bpn)
  (forward-bpn bpn)
  (backward-bpn bpn :last-lump (first-trained-weight-lump trainer bpn))
  (cost bpn))

(defmethod compute-batch-cost-and-derive (batch trainer (bpn bpn))
  (let ((cost #.(flt 0)))
    (do-segment-set (segment) (segment-set trainer)
      (with-segment-weights ((array start end) segment #'segment-derivatives)
        (fill array #.(flt 0) :start start :end end)))
    (map nil (lambda (sample)
               (incf cost (compute-cost sample trainer bpn)))
         batch)
    ;; By now the weight derivatives have accumulated, see
    ;; ZERO-NON-WEIGHT-DERIVATIVES.
    (segment-set->weights (segment-set trainer) (accumulator1 trainer)
                          :fn #'segment-derivatives)
    cost))

(defun add-and-forget-derivatives (trainer bpn)
  (let ((derivatives (derivatives bpn)))
    (declare (type flt-vector derivatives))
    (do-segment-gradient-accumulators ((lump acc-start accumulator) trainer)
      (let ((start (start lump)))
        (declare (type index start))
        (map-segment-runs
         (lambda (start1 end1)
           (declare (type index start1 end1)
                    (optimize (speed 3) #.*no-array-bounds-check*))
           (assert (<= (+ acc-start (- end1 start)) (length accumulator)))
           (loop for i upfrom start1 below end1
                 for j upfrom (+ acc-start (- start1 start))
                 do (incf (aref accumulator j) (aref derivatives i))
                 (setf (aref derivatives i) #.(flt 0))))
         lump)))))

(defmethod train-one (sample (trainer bp-trainer) bpn &key)
  (compute-cost sample trainer bpn)
  (add-and-forget-derivatives trainer bpn)
  (call-next-method))


;;;; I/O

(defmethod write-weights ((lump weight-lump) stream)
  (multiple-value-bind (array start end) (lump-node-array lump)
    ;; FIXME: pass START and END to WRITE-DOUBLE-FLOAT-VECTOR
    (mgl-util::write-double-float-vector (subseq array start end) stream)))

(defmethod read-weights ((lump weight-lump) stream)
  (multiple-value-bind (array start end) (lump-node-array lump)
    ;; FIXME: pass START and END to READ-DOUBLE-FLOAT-VECTOR
    (let ((r (make-flt-array (- end start))))
      (mgl-util::read-double-float-vector r stream)
      (replace array r :start1 start))))

(defmethod write-weights ((bpn bpn) stream)
  (map-segments (lambda (weights)
                  (write-weights weights stream))
                bpn))

(defmethod read-weights ((bpn bpn) stream)
  (map-segments (lambda (weights)
                  (read-weights weights stream))
                bpn))

;;;; Activation lump

(defclass activation-lump (lump)
  ((weight-lump :initarg :weight-lump :reader weight-lump)
   (input-lump :initarg :input-lump :reader input-lump)
   (transpose-weights-p :initform nil :initarg :transpose-weights-p
                        :reader transpose-weights-p))
  (:documentation "Perform W*I where I is the INPUT-LUMP of length N
and W is the WEIGHT-LUMP taken to be of dimensions M x N. M is the
size of this lump. If TRANSPOSE-WEIGHTS-P then compute W'*I.

This is equivalent to but much faster than ->LINEAR that's more
flexible."))

(defun displace (array start size)
  (declare (type flt-vector array))
  (make-array size :element-type 'flt
              :displaced-to array :displaced-index-offset start))

(defun y<-a*x (a transposep x y)
  (let ((xs (length x))
        (ys (length y)))
    (assert (= (* xs ys) (length a)))
    (if transposep
        (funcall (intern #.(symbol-name 'dgemv) (find-package 'blas))
                 "N" ys xs 1d0 a ys x 1 0d0 y 1)
        (funcall (intern #.(symbol-name 'dgemv) (find-package 'blas))
                 "T" xs ys 1d0 a xs x 1 0d0 y 1))))

(defun y+=a*x (a transposep x y)
  (let ((xs (length x))
        (ys (length y)))
    (assert (= (* xs ys) (length a)))
    (if transposep
        (funcall (intern #.(symbol-name 'dgemv) (find-package 'blas))
                 "N" ys xs 1d0 a ys x 1 1d0 y 1)
        (funcall (intern #.(symbol-name 'dgemv) (find-package 'blas))
                 "T" xs ys 1d0 a xs x 1 1d0 y 1))))

(defun a+=x*yt (a transposep x y)
  (let ((xs (length x))
        (ys (length y)))
    (assert (= (* xs ys) (length a)))
    (if transposep
        (funcall (intern #.(symbol-name 'dger) (find-package 'blas))
                 ys xs 1d0 y 1 x 1 a ys)
        (funcall (intern #.(symbol-name 'dger) (find-package 'blas))
                 xs ys 1d0 x 1 y 1 a xs))))

(defmethod transfer-lump ((lump activation-lump) nodes)
  (declare (type flt-vector nodes))
  (let* ((weight (weight-lump lump))
         (input (input-lump lump))
         (weight-start (start weight))
         (input-start (start input))
         (output-start (start lump))
         (weight-size (lump-size weight))
         (input-size (lump-size input))
         (output-size (lump-size lump))
         (input-end (+ input-start input-size))
         (output-end (+ output-start output-size)))
    (declare (type index weight-start input-start output-start
                   weight-size input-size output-size
                   input-end output-end))
    (assert (= (* input-size output-size) weight-size))
    (if (use-blas-p (lump-size weight))
        (y<-a*x (displace nodes weight-start weight-size)
                (transpose-weights-p lump)
                (displace nodes input-start input-size)
                (displace nodes output-start output-size))
        (let ((ij weight-start))
          (declare (optimize (speed 3) #.*no-array-bounds-check*))
          (cond ((transpose-weights-p lump)
                 (fill nodes #.(flt 0) :start output-start
                       :end (+ output-start output-size))
                 (loop for i upfrom input-start below input-end
                       do (loop for j upfrom output-start below output-end
                                do (incf (aref nodes j)
                                         (* (aref nodes ij)
                                            (aref nodes i)))
                                (incf ij))))
                (t
                 (loop for j upfrom output-start below output-end
                       do (let ((sum #.(flt 0)))
                            (loop for i upfrom input-start below input-end
                                  do (incf sum (* (aref nodes ij)
                                                  (aref nodes i)))
                                  (incf ij))
                            (setf (aref nodes j) sum)))))))))

(defmethod derivate-lump ((lump activation-lump) nodes derivatives)
  (declare (type flt-vector nodes derivatives))
  (let* ((weight (weight-lump lump))
         (input (input-lump lump))
         (weight-start (start weight))
         (input-start (start input))
         (output-start (start lump))
         (weight-size (lump-size weight))
         (input-size (lump-size input))
         (output-size (lump-size lump))
         (input-end (+ input-start input-size))
         (output-end (+ output-start output-size)))
    (declare (type index weight-start input-start output-start
                   weight-size input-size output-size
                   input-end output-end))
    (assert (= (* input-size output-size) weight-size))
    (cond ((use-blas-p (lump-size weight))
           (unless (typep input 'input-lump)
             (y+=a*x (displace nodes weight-start weight-size)
                     (not (transpose-weights-p lump))
                     (displace derivatives output-start output-size)
                     (displace derivatives input-start input-size)))
           (a+=x*yt (displace derivatives weight-start weight-size)
                    (transpose-weights-p lump)
                    (displace nodes input-start input-size)
                    (displace derivatives output-start output-size)))
          (t
           (let ((ij weight-start))
             (declare (optimize (speed 3) #.*no-array-bounds-check*))
             (cond ((transpose-weights-p lump)
                    (loop for i upfrom input-start below input-end
                          do (loop for j upfrom output-start below output-end
                                   do (let ((dj (aref derivatives j)))
                                        (incf (aref derivatives ij)
                                              (* dj (aref nodes i)))
                                        (incf (aref derivatives i)
                                              (* dj (aref nodes ij)))
                                        (incf ij)))))
                   (t
                    (loop for j upfrom output-start below output-end
                          do (let ((dj (aref derivatives j)))
                               (loop for i upfrom input-start below input-end
                                     do (incf (aref derivatives ij)
                                              (* dj (aref nodes i)))
                                     (incf (aref derivatives i)
                                           (* dj (aref nodes ij)))
                                     (incf ij)))))))))))


;;;; Node type library

(define-node-type ->+ (&rest args)
  (:transfer (let ((sum #.(flt 0)))
               (declare (optimize (speed 3) #.*no-array-bounds-check*))
               (dolist (i args)
                 (declare (type index i))
                 (incf sum (node i)))
               sum))
  (:derivate (declare (optimize (speed 3) #.*no-array-bounds-check*))
             (dolist (i args)
               (declare (type index i))
               (add-derivative i #.(flt 1)))))

;;; "x0*y0+x1*y1+..."
(define-node-type ->linear (x y)
  (:transfer (declare (type index-vector x y))
             (let ((sum #.(flt 0)))
               (declare (optimize (speed 3) #.*no-array-bounds-check*))
               (loop for i across x
                     for j across y
                     do (incf sum (* (node i) (node j))))
               sum))
  (:derivate (declare (type index-vector x y)
                      (optimize (speed 3) #.*no-array-bounds-check*))
             (loop for i across x
                   for j across y
                   do
                   (add-derivative i (node j))
                   (add-derivative j (node i)))))

(define-node-type ->sigmoid (x)
  (:transfer (declare (type index x)
                      (optimize (speed 3) #.*no-array-bounds-check*))
             (sigmoid (node x)))
  (:derivate (declare (type index x)
                      (optimize (speed 3) #.*no-array-bounds-check*))
             (let ((s (sigmoid (node x))))
               (declare (type flt s))
               (add-derivative x (* s (- 1 s))))))

(define-node-type ->exp (x)
  (:transfer (declare (type index x)
                      (optimize (speed 3) #.*no-array-bounds-check*))
             (exp (node x)))
  (:derivate (declare (type index x)
                      (optimize (speed 3) #.*no-array-bounds-check*))
             (add-derivative x (exp (node x)))))

(define-node-type ->sum-squared-error (x y)
  (:transfer (declare (type index-vector x y))
             (let ((sum #.(flt 0)))
               (loop for i across x
                     for j across y
                     do (incf sum (expt (- (node i) (node j)) 2)))
               sum))
  (:derivate (declare (type index-vector x y))
             (loop for i across x
                   for j across y
                   do
                   (add-derivative i (* 2 (- (node i) (node j))))
                   (add-derivative j (* 2 (- (node j) (node i)))))))

(define-node-type ->cross-entropy (x y)
  (:transfer (declare (type index-vector x y))
             (let ((sum #.(flt 0)))
               (loop for i across x
                     for j across y
                     do (decf sum (* (node i) (log (node j)))))
               sum))
  (:derivate (declare (type index-vector x y))
             (loop for i across x
                   for j across y
                   do
                   (add-derivative i (- (log (node j))))
                   (add-derivative j (- (/ (node i) (node j)))))))
