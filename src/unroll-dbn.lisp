;;;; Unrolling RBMs to backprop network

(in-package :mgl-unroll-dbn)

;;;; Extending BPNs with limited missing value support
;;;;
;;;; Missing values in the resulting BPN are a pain. There are three
;;;; different cases that must be handled:
;;;;
;;;; 1) marking nodes in lumps as missing (input, nodewise-lump),
;;;; these nodes will not be transferred, derivated.
;;;;
;;;; 2) some node type such as ->sparse1-linear must deal with missing
;;;; inputs
;;;;
;;;; 3) weights must updated only if they were used
;;;;
;;;; So, let's add an indices-present slot to lumps that's basically
;;;; only used by transfer-fn and derivate-fn to skip nodes and
;;;; ->sparse1-linear to skip elements of the sum.
;;;;
;;;; That leaves us with the question of how to do 3). Again, we have
;;;; choices:
;;;;
;;;; a) weights could carry INDICES-PRESENT set by the user at input.
;;;; However, it implies that the user intimately knows the node types
;;;; used.
;;;;
;;;; b) DERIVATE-LUMP marks (or somehow one can tell) which weights
;;;; have been used. (Note that a zero derivative is not a good choice
;;;; you don't want to update weight that have mostly no effect too
;;;; wildly.) Iteration over the used mask is costly. Same if it's a
;;;; hash.
;;;;
;;;; So, I'm leaning towards a), except that iteration over an
;;;; INDICES-PRESENT vector pointing into a weight matrix is likely to
;;;; be woefully inefficient as consecutive indices tend to be present
;;;; or missing together. But that's not really needed as
;;;; SEGMENTED-GD-TRAINER only needs MAP-SEGMENT-RUNS.
;;;;
;;;; I think we have arrived to a fast but kind of inelegant solution
;;;; that is sufficient to define networks with missing values albeit
;;;; not without extra bookkeeping after each clamping. This is good
;;;; enough to translate DBNs with missing values and since the whole
;;;; concept of missing values masking parts of the network goes
;;;; against the static structure assumption and the implementation is
;;;; not very generic maybe it would be better to implement this as an
;;;; add-on without touching the backprop core.

(defmacro *ndx (x y)
  `(the! index (* ,x ,y)))

(defmacro +ndx (x y)
  `(the! index (+ ,x ,y)))

(defclass sparse-weight-lump (weight-lump)
  ((rows-present
    :type (or null index-vector)
    :initform nil :initarg :rows-present :accessor rows-present)
   (row-size
    :type index :initform (error "ROW-SIZE missing") :initarg :row-size
    :reader row-size)))

#+nil
(defmethod map-segment-runs (fn (lump sparse-weight-lump))
  (let ((rows-present (rows-present lump)))
    (declare (type (or null index-vector) rows-present))
    (multiple-value-bind (- start end) (mgl-bp:lump-node-array lump)
      (declare (type index start end))
      (if rows-present
          (let ((row-size (row-size lump)))
            (declare (type index row-size))
            (loop for row across rows-present do
                  (let ((i (+ start (* row row-size))))
                    (funcall fn i (+ i row-size)))))
          (funcall fn start end)))))

(defclass cloud-activation-lump (activation-lump)
  ((punt-to-activation-lump-p
    :initform t :initarg :punt-to-activation-lump-p
    :accessor punt-to-activation-lump-p
    :documentation "Knob for easy testing."))
  (:documentation "In the name of performance combine ACTIVATION-LUMP
and ->CLOUD-ACTIVATION. The easy cases with no missing values are
handled by the fast ACTIVATION-LUMP and the rest by
->CLOUD-ACTIVATION."))

#+nil
(defun missing-values-in-cloud-activation-p (lump)
  (or (indices-to-calculate lump)
      (indices-to-calculate (input-lump lump))))

#+nil
(defmethod transfer-lump ((lump cloud-activation-lump) nodes)
  (declare (type flt-vector nodes))
  (if (and (punt-to-activation-lump-p lump)
           (not (missing-values-in-cloud-activation-p lump)))
      (call-next-method)
      (let* ((weight (weight-lump lump))
             (input (input-lump lump))
             (weight-start (mgl-bp::start weight))
             (input-start (mgl-bp::start input))
             (output-start (mgl-bp::start lump))
             (weight-size (lump-size weight))
             (input-size (lump-size input))
             (output-size (lump-size lump))
             (input-end (+ input-start input-size))
             (output-end (+ output-start output-size)))
        (declare (type index weight-start input-start output-start
                       weight-size input-size output-size
                       input-end output-end))
        (when (punt-to-activation-lump-p lump)
          (assert (typep weight 'sparse-weight-lump))
          (assert (rows-present weight)))
        (assert (= weight-size (* input-size output-size)))
        (locally
            (declare (optimize (speed 3) #.*no-array-bounds-check*))
          (cond ((transpose-weights-p lump)
                 (when (punt-to-activation-lump-p lump)
                   (assert (indices-to-calculate input)))
                 (assert (not (indices-to-calculate lump)))
                 (fill nodes #.(flt 0) :start output-start :end output-end)
                 (do-lump (row input)
                   (let ((x (aref nodes (+ndx input-start row))))
                     (unless (zerop x)
                       (loop for j upfrom output-start below output-end
                             for weight-index
                             upfrom (+ndx weight-start (*ndx row output-size))
                             do (incf (aref nodes j)
                                      (* x (aref nodes weight-index))))))))
                (t
                 (assert (not (indices-to-calculate input)))
                 (when (punt-to-activation-lump-p lump)
                   (assert (indices-to-calculate lump)))
                 (do-lump (row lump)
                   (let ((sum #.(flt 0)))
                     (declare (type flt sum))
                     (loop for i upfrom input-start below input-end
                           for weight-index
                           upfrom (+ndx weight-start (*ndx row input-size))
                           do (incf sum (* (aref nodes i)
                                           (aref nodes weight-index))))
                     (setf (aref nodes (+ndx output-start row)) sum)))))))))

#+nil
(defmethod derivate-lump ((lump cloud-activation-lump) nodes derivatives)
  (declare (type flt-vector nodes derivatives))
  (if (and (punt-to-activation-lump-p lump)
           (not (missing-values-in-cloud-activation-p lump)))
      (call-next-method)
      (let* ((weight (weight-lump lump))
             (input (input-lump lump))
             (weight-start (mgl-bp::start weight))
             (input-start (mgl-bp::start input))
             (output-start (mgl-bp::start lump))
             (weight-size (lump-size weight))
             (input-size (lump-size input))
             (output-size (lump-size lump))
             (input-end (+ input-start input-size))
             (output-end (+ output-start output-size)))
        (declare (type index weight-start input-start output-start
                       weight-size input-size output-size
                       input-end output-end))
        (when (punt-to-activation-lump-p lump)
          (assert (typep weight 'sparse-weight-lump))
          (assert (rows-present weight)))
        (assert (= weight-size (* input-size output-size)))
        (locally
            (declare (optimize (speed 3) #.*no-array-bounds-check*))
          (cond ((transpose-weights-p lump)
                 (when (punt-to-activation-lump-p lump)
                   (assert (indices-to-calculate input)))
                 (assert (not (indices-to-calculate lump)))
                 (do-lump (row input)
                   (let* ((i (+ndx input-start row))
                          (x (aref nodes i)))
                     (loop for j upfrom output-start below output-end
                           for ij upfrom (+ndx weight-start
                                               (*ndx row output-size))
                           do (let ((dj (aref derivatives j)))
                                (incf (aref derivatives ij)
                                      (* dj x))
                                (incf (aref derivatives i)
                                      (* dj (aref nodes ij))))))))
                (t
                 (assert (not (indices-to-calculate input)))
                 (when (punt-to-activation-lump-p lump)
                   (assert (indices-to-calculate lump)))
                 (do-lump (row lump)
                   (let* ((j (+ndx output-start row))
                          (dj (aref derivatives j)))
                     (loop for i upfrom input-start below input-end
                           for ij upfrom (+ndx weight-start
                                               (*ndx row input-size))
                           do (incf (aref derivatives ij)
                                    (* dj (aref nodes i)))
                           (incf (aref derivatives i)
                                 (* dj (aref nodes ij))))))))))))


;;;; Unrolling

;;; Intermediate representation of lumps during the unrolling.
(defstruct lumpy
  ;; At this depth can this lump be found in the unrolled RBM. Depth 0
  ;; is the middle: the hidden layer of the topmost rbm. Depth 1 is
  ;; the visible of layer of the topmost rbm, depth -1 is the
  ;; reconstruction of the same.
  depth
  ;; The level of the layer of the DBN this lumpy comes from.
  level
  chunk
  (incomings '())
  ;; the lumpy with a non-negative depth from which this was mirrored
  ;; or nil
  original
  ;; This symbol is bound to the added lump when creating the network.
  (symbol (gensym)))

(defstruct incoming
  from-lumpy
  cloud
  rbm-index
  transposep)

(defgeneric chunk-lump-name (level chunk reconstructionp)
  (:documentation "The name of the lump that represents CHUNK or its
reconstruction at LEVEL in the DBN.")
  (:method (level chunk reconstructionp)
    `(,(name chunk) ,level ,@(when reconstructionp '(:reconstruction)))))

(defgeneric chunk-activation-lump-name (level chunk reconstructionp)
  (:documentation "The name of the lump that computes the activations
of CHUNK or its reconstructed version at LEVEL in the DBN.")
  (:method (level chunk reconstructionp)
    (append (chunk-lump-name level chunk reconstructionp) '(:activation))))

(defgeneric cloud-weight-lump-name (rbm-index cloud transposep)
  (:documentation "The name of the lump that represents the weights of
CLOUD or its transpose. CLOUD comes from the rbm at RBM-INDEX in the
DBN.")
  (:method (rbm-index (cloud cloud) transposep)
    `(,(name cloud) :weights ,rbm-index ,@(when transposep '(:transpose)))))

(defgeneric cloud-linear-lump-name (rbm-index cloud transposep)
  (:documentation "The name of the lump that represents part of the
activation of a chunk. CLOUD comes from the rbm at RBM-INDEX in the
DBN. TRANSPOSEP determines from which direction the activation crosses
the cloud.")
  (:method (rbm-index cloud transposep)
    (append (cloud-weight-lump-name rbm-index cloud transposep) '(:linear))))

(defgeneric chunk->bpn-definition (chunk sym name size activation-symbol)
  (:documentation "Return a bpn definition form (that is a list of
lump definition forms) for CHUNK that takes a single activation
parameter given by the symbol ACTIVATION-SYMBOL with NAME and SIZE.
Only called for non-conditioning chunks. Second value is a list of
clamp inits, the third is a list of inits.")
  (:method ((chunk sigmoid-chunk) sym name size activation-symbol)
    `((,sym (->sigmoid :name ',name :x ,activation-symbol))))
  (:method ((chunk gaussian-chunk) sym name size activation-symbol)
    ;; this is identity
    `((,sym (->+ :name ',name :args (list ,activation-symbol)))))
  (:method ((chunk exp-normalized-group-chunk) sym name size activation-symbol)
    (let ((exp-symbol (gensym)))
      (values
       `((,exp-symbol (->exp :x ,activation-symbol))
         (,sym (normalized-lump :name ',name
                :scale ,(scale chunk) :group-size ,(group-size chunk)
                :x ,exp-symbol)))
       `((:from-lump ,name :to-lump ,exp-symbol))))))

(defun lumpy-name (lumpy)
  (chunk-lump-name (lumpy-level lumpy) (lumpy-chunk lumpy)
                   (minusp (lumpy-depth lumpy))))

(defun lumpy-activation-name (lumpy)
  (chunk-activation-lump-name (lumpy-level lumpy) (lumpy-chunk lumpy)
                              (minusp (lumpy-depth lumpy))))

;;; Find CHUNK at DEPTH in LUMPIES.
(defun find-lumpy (depth chunk lumpies)
  (find-if (lambda (lumpy)
             (and (eql depth (lumpy-depth lumpy))
                  (eq chunk (lumpy-chunk lumpy))))
           lumpies))

(defun find-lumpy-by-name (name lumpies)
  (find name lumpies :key #'lumpy-name :test #'equal))

;;; Can the chunk on which LUMPY is based have INDICES-PRESENT?
(defun possibly-sparse-lumpy-p (lumpy)
  (endp (lumpy-incomings (or (lumpy-original lumpy) lumpy))))

(defgeneric incoming->bpn-defintion (from-lumpy to-lumpy rbm-index
                                                cloud transposep)
  (:documentation "Return a list of lump definitions that represent
the flow from FROM-LUMPY through CLOUD. The chunk of FROM-LUMPY may be
either of the end points of CLOUD. As the second value return
initialization specs and the name of the `end' lump as the third.")
  (:method (from-lumpy to-lumpy rbm-index (cloud cloud) transposep)
    (let* ((from-chunk (lumpy-chunk from-lumpy))
           (from-size (chunk-size from-chunk))
           (n-weights (matlisp:number-of-elements (weights cloud)))
           (weight-symbol (gensym))
           (weight-name (cloud-weight-lump-name rbm-index cloud transposep))
           (size (/ n-weights from-size))
           (linear-symbol (gensym))
           (linear-name (cloud-linear-lump-name rbm-index cloud transposep))
           (sparsep (or (possibly-sparse-lumpy-p from-lumpy)
                        (possibly-sparse-lumpy-p to-lumpy)))
           (row-size (chunk-size (hidden-chunk cloud))))
      ;; Regardless of TRANSPOSEP the weight lump contains rows of
      ;; weights for each target node in the hidden chunk just like in
      ;; the RBM.
      (list `((,weight-symbol
               (,(if sparsep
                     'sparse-weight-lump
                     'weight-lump)
                ,@(when sparsep (list :row-size row-size))
                :name ',weight-name
                :size ,n-weights))
              (,linear-symbol
               (cloud-activation-lump
                :name ',linear-name
                :size ,size
                :weights ,weight-symbol
                :x ,(lumpy-symbol from-lumpy)
                :transpose-weights-p ,transposep)))
            `((:from-lump ,(if transposep
                               (lumpy-name to-lumpy)
                               (lumpy-name from-lumpy))
               :to-lump ,weight-name :to :rows)
              (:from-lump ,(lumpy-name to-lumpy) :to-lump ,linear-name))
            `((:weight-name ,weight-name
               :rbm-index ,rbm-index
               :cloud-name ,(name cloud)))
            linear-symbol))))

(defun incoming-list->bpn-definition (to-lumpy incomings)
  (let ((x (loop for incoming in incomings
                 collect (incoming->bpn-defintion
                          (incoming-from-lumpy incoming)
                          to-lumpy
                          (incoming-rbm-index incoming)
                          (incoming-cloud incoming)
                          (incoming-transposep incoming)))))
    (values (mapcan #'first x)
            (mapcan #'second x)
            (mapcan #'third x)
            (mapcar #'fourth x))))

(defun lumpies->bpn-definition (lumpies)
  (let ((defs '())
        (clamps '())
        (inits '()))
    (dolist (lumpy (sort lumpies #'> :key #'lumpy-depth))
      (let* ((chunk (lumpy-chunk lumpy))
             (incomings (lumpy-incomings lumpy))
             (name (lumpy-name lumpy))
             (activation-symbol (gensym))
             (activation-name (lumpy-activation-name lumpy))
             (size (chunk-size chunk)))
        (cond ((typep chunk 'constant-chunk)
               (assert (endp incomings))
               (push `(,(lumpy-symbol lumpy)
                       (constant-lump
                        :name ',name :size ,size
                        :default-value ,(default-value chunk)))
                     defs))
              ((or (typep chunk 'conditioning-chunk)
                   (endp incomings))
               (assert (endp incomings))
               (push `(,(lumpy-symbol lumpy)
                       (input-lump :name ',name :size ,size))
                     defs))
              (t
               (multiple-value-bind (cloud-defs cloud-clamps cloud-inits
                                                linear-symbols)
                   (incoming-list->bpn-definition lumpy incomings)
                 (push-all cloud-defs defs)
                 (push-all cloud-clamps clamps)
                 (push-all cloud-inits inits)
                 (push `(,activation-symbol
                         (->+ :name ',activation-name
                              :args (list ,@linear-symbols)))
                       defs)
                 (push `(:from-lump ,name :to-lump ,activation-name)
                       clamps))
               (multiple-value-bind (chunk-defs chunk-clamps chunk-inits)
                   (chunk->bpn-definition chunk (lumpy-symbol lumpy)
                                          name size activation-symbol)
                 (push-all chunk-defs defs)
                 (push-all chunk-clamps clamps)
                 (push-all chunk-inits inits))))))
    (values (reverse defs)
            (remove-if-not (lambda (clamp)
                             (possibly-sparse-lumpy-p
                              (find-lumpy-by-name (getf clamp :from-lump)
                                                  lumpies)))
                           clamps)
            inits)))

(defun unroll-dbn (dbn &key bottom-up-only)
  "Unroll DBN recursively and turn it into a feed-forward
backpropagation network. A single RBM in DBN of the form VISIBLE <->
HIDDEN is transformed into a VISIBLE -> HIDDEN ->
RECONSTRUCTION-OF-VISIBLE network. While the undirected connection <->
has a common weight matrix for both directions in the backprop network
the weights pertaining to ->'s are distinct but are initialized from
the same <-> (with one being the tranpose of it).

If BOTTOM-UP-ONLY then don't generate the part of the network that
represents the top-down flow, that is, skip the reconstructions.

Return backprop network lump definition forms, as the second value the
`clamps': that can be passed to CLAMP-INDICES-IN-UNROLLED-DBN, and as
the third value `inits': initialization specifications suitable for
INITIALIZE-BPN-FROM-RBM.

Chunks of `touching' layers (e.g. the hidden layer of one and the
visible layer of the one on top) are considered to be the same if they
have the same name. If there is no corresponding chunk in the layer
below or there is no rbm below then the chunk is translated into an
INPUT lump. Desired outputs and error node are not added. The first
element of RMBS is the topmost one (last of the DBN), the one that
goes into the middle of the backprop network."
  (let ((lumpies '())
        (n-rbms (length (rbms dbn))))
    (flet ((ensure-lumpy (depth chunk)
             (or (find-lumpy depth chunk lumpies)
                 (let ((lumpy
                        (make-lumpy :depth depth
                                    :original (if (minusp depth)
                                                  (find-lumpy (abs depth)
                                                              chunk lumpies)
                                                  nil)
                                    :level (- n-rbms (abs depth))
                                    :chunk chunk)))
                   (push lumpy lumpies)
                   lumpy)))
           (add-connection (cloud depth &key from to)
             (assert (> (lumpy-depth from) (lumpy-depth to)))
             (assert (not (member from (lumpy-incomings to)
                                  :key #'incoming-from-lumpy)))
             (push (make-incoming :from-lumpy from
                                  :cloud cloud
                                  :rbm-index (- n-rbms (abs depth) 1)
                                  :transposep (eq (lumpy-chunk from)
                                                  (hidden-chunk cloud)))
                   (lumpy-incomings to))))
      (loop for rbm in (reverse (rbms dbn))
            for depth upfrom 0
            do
            (do-clouds (cloud rbm)
              (let ((hidden-chunk (hidden-chunk cloud))
                    (visible-chunk (visible-chunk cloud)))
                (unless (typep hidden-chunk 'conditioning-chunk)
                  (add-connection cloud depth
                                  :from (ensure-lumpy (1+ depth) visible-chunk)
                                  :to (ensure-lumpy depth hidden-chunk)))
                (unless (or (typep visible-chunk 'conditioning-chunk)
                            bottom-up-only)
                  ;; If the chunk does not need activations then it is
                  ;; in effect not reconstructed, so let's use the
                  ;; non-mirrored version. Thus the quasi-symmetry of
                  ;; the code and the generated network is broken.
                  (let ((hidden-depth
                         (if (typep hidden-chunk 'conditioning-chunk)
                             depth
                             (- depth))))
                    (add-connection cloud depth
                                    :from (ensure-lumpy hidden-depth
                                                        hidden-chunk)
                                    :to (ensure-lumpy (- (1+ depth))
                                                      visible-chunk)))))))
      (lumpies->bpn-definition lumpies))))

(defmacro setf-eq-p (place newvalue)
  (with-gensyms (%newvalue)
    `(let ((,%newvalue ,newvalue))
       (prog1 (eq ,place ,%newvalue)
         (setf ,place ,%newvalue)))))

(defun clamp-indices-in-unrolled-dbn (bpn clamps)
  "In an unrolled DBN, if there are INDICES-PRESENT in play, the
user's clamper is supposed to set INDICES-TO-CALCULATE in the original
chunk and its reconstruction (they may be different). However this
function shall then be called to propagate these settings to other
lumps that are affected such as weight lumps and basically all lumps
involved in the reconstruction of the chunk in question. Return the
first lump that was changed and may need to be recalculated or NIL if
no lump was changed."
  (let ((first-changed-lump nil))
    (loop
     for lump across (lumps bpn)
     do (let* ((name (name lump))
               (clamp (find-if (lambda (clamp)
                                 (equal name (getf clamp :to-lump)))
                               clamps)))
          (when clamp
            (destructuring-bind (&key from-lump to-lump
                                      (from :indices) (to :indices)) clamp
              (declare (ignore to-lump))
              (assert (eq from :indices))
              (let ((from-lump (find-lump from-lump bpn :errorp t)))
                (when (and (not
                            (ecase to
                              ((:indices)
                               (setf-eq-p (indices-to-calculate lump)
                                          (indices-to-calculate from-lump)))
                              ((:rows)
                               (setf-eq-p (rows-present lump)
                                          (indices-to-calculate from-lump)))))
                           (null first-changed-lump))
                  (setq first-changed-lump lump)))))))
    first-changed-lump))

(defun initialize-bpn-from-dbn (bpn dbn inits)
  "Initialize BPN from the weights of DBN according to INITS that was
returned by UNROLL-DBN."
  (let ((rbms (rbms dbn)))
    (dolist (init inits)
      (destructuring-bind (&key weight-name rbm-index cloud-name)
          init
        (let* ((lump (find-lump weight-name bpn :errorp t))
               (cloud (find-cloud cloud-name (elt rbms rbm-index) :errorp t))
               (weights (storage (weights cloud))))
          (declare (type flt-vector weights))
          (multiple-value-bind (nodes start end)
              (segment-weights lump)
            (declare (type flt-vector nodes))
            (unless (= (length weights) (- end start))
              (error "Cannot initialize lump ~S from cloud ~S: size mismatch"
                     lump cloud))
            (replace nodes weights :start1 start :end1 end)))))))
