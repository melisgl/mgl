(in-package :mgl-bp)

(defsection @mgl-bp (:title "Backpropagation Neural Networks")
  (@mgl-bp-overview section)
  (@mgl-bp-extension-api section)
  (@mgl-bpn section))


;;;; Components / Generic backprop

(defsection @mgl-bp-overview (:title "Backprop Overview")
  "Backpropagation Neural Networks are just functions with typically
  lots of parameters called weights. FIXDOC: LOSS FUNCTION, MINIMIZE,
  LAYERS.

  In this implementation, a BPN is assembled from several
  [LUMP][]s (roughly corresponding to layers). Both feed-forward and
  recurrent neural nets are supported (FNN and RNN, respectively).
  BPNs can contain not only LUMPs but other BPNs, too."
  (clump class)
  (nodes generic-function)
  "FIXDOC: set-input")

;;; Everything is defined with DEFCLASS-NOW because DEFMAKER needs the
;;; class to be around at macro expansion time.
(defclass-now clump ()
  ((name :initform (gensym) :initarg :name :reader name))
  (:documentation "A CLUMP is a LUMP or BPN. It represents a
  differentiable function. Arguments come from other clumps wired
  permenantly together. This wiring of CLUMPs is how one builds
  feed-forward nets (FNN) or recurrent neural networks (RNN) that are
  CLUMPs themselves so one can build nets in a hiearchical style if
  desired. Non-composite CLUMPs are called LUMP (note the loss of `C`
  that stands for composite). The various LUMP subtypes correspond to
  different layer types (sigmoid, dropout, rectified linear, tanh,
  etc)."))

(defvar *bpn-being-built* nil)

(defmethod initialize-instance :around ((clump clump) &key &allow-other-keys)
  (call-next-method)
  (if *bpn-being-built*
      ;; This sets MAX-N-STRIPES to that of *BPN-BEING-BUILT*.
      (add-clump clump *bpn-being-built*)
      ;; If we aren't building a bpn, let's ensure that the matrices
      ;; are allocated.
      (setf (max-n-stripes clump) (max-n-stripes clump))))

;;;; FIXME: make these work in fnn and rnn so that they can be used as
;;;; clumps in fnns.

(defgeneric nodes (clump)
  (:documentation "Return the MAT object representing the state of
  CLUMP (that is, the result computed by the most recent FORWARD). For
  ->INPUT lumps, this is where input values are placed. See FIXDOC."))


(defsection @mgl-bp-extension-api (:title "Backprop Extension API")
  (stripedp generic-function)
  (derivatives generic-function)
  (forward generic-function)
  (backward generic-function))

(defgeneric stripedp (clump)
  (:documentation "For efficiency, forward and backprop phases do
  their stuff in batch mode: passing a number of instances through the
  network at the same time. Thus clumps must be able to store values
  of and gradients for each of these instances. However, some clumps
  produce the same result for each instance in a batch. These clumps
  are the weights, the parameters of the network. STRIPEDP returns
  true iff CLUMP does not represent weights (i.e. it's not a
  ->WEIGHT).

  For striped clumps, their NODES and DERIVATIVES are MAT objects with
  a leading dimension (number of rows in the 2d case) equal to the
  number of instances in the batch. Non-striped clumps have no
  restriction on their shape apart from what their usage dictates.")
  (:method ((clump clump))
    t))

(defgeneric forward (clump)
  (:documentation "Compute the values of the function represented by
  CLUMP for all stripes and place the results into NODES of CLUMP."))

(defgeneric backward (clump)
  (:documentation "Compute the partial derivatives of the function
  represented by CLUMP and add them to DERIVATIVES of the
  corresponding argument clumps. The DERIVATIVES of CLUMP contains the
  sum of partial derivatives of all clumps by the corresponding
  output. This function is intended to be called after a FORWARD pass.

  Take the ->SIGMOID clump for example when the network is being
  applied to a batch of two instances `x1` and `x2`. `x1` and `x2` are
  set in the ->INPUT lump X. The sigmoid computes `1/(1+exp(-x))`
  where `X` is its only argument clump.

      f(x) = 1/(1+exp(-x))

  When BACKWARD is called on the sigmoid lump, its DERIVATIVES is a
  2x1 MAT object that contains the partial derivatives of the loss
  function:

      dL(x1)/df
      dL(x2)/df

  Now the BACKWARD method of the sigmoid needs to add `dL(x1)/dx1` and
  `dL(x2)/dx2` to DERIVATIVES of `X`. Now, `dL(x1)/dx1 = dL(x1)/df *
  df(x1)/dx1` and the first term is what we have in DERIVATIVES of the
  sigmoid so it only needs to calculate the second term."))

(defgeneric derivatives (clump)
  (:documentation "Return the MAT object representing the partial
  derivatives of the function CLUMP computes. The returned partial
  derivatives were accumulated by previous BACKWARD calls."))


(defsection @mgl-bpn (:title "BPNs")
  (bpn class)
  (clumps generic-function)
  (find-clump function)
  (add-clump function)
  (->clump function)
  ;; set-input, forward, backward, {read,write}-weights, COST?
  (@mgl-fnn section)
  (@mgl-rnn section))

(defclass bpn (clump)
  ((clumps
    :initform (make-array 0 :element-type 'clump :adjustable t :fill-pointer t)
    :initarg :clumps
    :type (array clump (*)) :reader clumps
    :documentation "Clumps in reverse order")
   (n-stripes
    :initform 1 :type index :initarg :n-stripes
    :reader n-stripes)
   (max-n-stripes
    :initform nil :type index :initarg :max-n-stripes
    :reader max-n-stripes)))

(defmethod initialize-instance :after ((bpn bpn) &key &allow-other-keys)
  (setf (max-n-stripes bpn)
        (cond ((max-n-stripes bpn)
               ;; We do the SETF just make sure clumps have the same
               ;; MAX-N-STRIPES.
               (max-n-stripes bpn))
              ;; Let's inherit MAX-N-STRIPES from the parent if any.
              (*bpn-being-built*
               (max-n-stripes *bpn-being-built*))
              (t 1))))

(defmethod print-object ((bpn bpn) stream)
  (pprint-logical-block (stream ())
    (print-unreadable-object (bpn stream :type t :identity t)
      (unless (zerop (length (clumps bpn)))
        (format stream "~@[~S ~]~S ~S/~S ~:_~S ~S"
                (if (uninterned-symbol-p (name bpn))
                    nil
                    (name bpn))
                :stripes (n-stripes bpn) (max-n-stripes bpn)
                :clumps (length (clumps bpn))))))
  bpn)

(define-descriptions (bpn bpn)
  clumps n-stripes max-n-stripes)

(defmethod set-n-stripes (n-stripes (bpn bpn))
  (setf (slot-value bpn 'n-stripes) n-stripes)
  (loop for clump across (clumps bpn)
        do (setf (n-stripes clump) n-stripes)))

(defmethod set-max-n-stripes (max-n-stripes (bpn bpn))
  (setf (slot-value bpn 'max-n-stripes) max-n-stripes)
  (loop for clump across (clumps bpn)
        do (setf (max-n-stripes clump) max-n-stripes)))

(defun find-clump (name bpn &key (errorp t))
  (or (find name (clumps bpn) :key #'name :test #'name=)
      (if errorp
          (error "Cannot find clump ~S." name)
          nil)))

(defmethod size ((bpn bpn))
  (size (alexandria:last-elt (clumps bpn))))

(defmethod nodes ((bpn bpn))
  (nodes (alexandria:last-elt (clumps bpn))))

(defmethod derivatives ((bpn bpn))
  (derivatives (alexandria:last-elt (clumps bpn))))

(defmethod set-input :around (samples (bpn bpn))
  (setf (n-stripes bpn) (length samples))
  (call-next-method))

(defun add-clump (clump bpn)
  "Add CLUMP to BPN. MAX-N-STRIPES of CLUMP gets set to equal that of
  the previous last, non-weight clump of BPN."
  (when (find-clump (name clump) bpn :errorp nil)
    (error "Cannot add ~S: a clump of same name has already been ~
           added to this network." clump))
  (setf (max-n-stripes clump) (max-n-stripes bpn))
  (vector-push-extend clump (slot-value bpn 'clumps) 1)
  clump)

(defun remove-clump (clump bpn)
  (setf (slot-value bpn 'clumps) (delete clump (clumps bpn)))
  clump)

(defun ->clump (bpn clump-spec)
  (if (typep clump-spec 'clump)
      clump-spec
      (find-clump clump-spec bpn)))

(defgeneric forward-bpn (bpn &key from-clump to-clump end-clump)
  (:documentation "Propagate the values from the already clamped
  inputs."))

(defgeneric backward-bpn (bpn &key last-clump)
  (:documentation "Accumulate derivatives of weights."))

;;; Derivatives of weights are left alone to let them accumulate which
;;; is useful in batches such as when training with conjugate
;;; gradient.
(defgeneric zero-non-weight-derivatives (clump)
  (:method ((bpn bpn))
    (map nil #'zero-non-weight-derivatives (clumps bpn)))
  (:method ((clump clump))
    (when (and (stripedp clump)
               (derivatives clump))
      (fill! 0 (derivatives clump)))))

(defmethod forward ((bpn bpn))
  (forward-bpn bpn))

(defmethod backward ((bpn bpn))
  (backward-bpn bpn))

(defmethod forward-bpn ((bpn bpn) &key from-clump to-clump end-clump)
  (declare (optimize (debug 3)))
  (let ((from-clump (if from-clump (->clump bpn from-clump) nil))
        (to-clump (if to-clump (->clump bpn to-clump) nil))
        (seen-from-clump-p (not from-clump)))
    (loop for clump across (clumps bpn)
          until (eq clump end-clump)
          do (when (eq clump from-clump) (setq seen-from-clump-p t))
          do (when seen-from-clump-p (forward clump))
          until (eq clump to-clump))))

(defmethod backward-bpn ((bpn bpn) &key last-clump)
  (let* ((clumps (clumps bpn))
         (last-clump (if last-clump (->clump bpn last-clump) nil)))
    (loop for i downfrom (1- (length clumps)) downto 0
          for clump = (aref clumps i)
          until (and last-clump (eq last-clump clump))
          do (backward clump))))

(defmethod cost ((bpn bpn))
  (let ((sum 0)
        (sum-importances 0))
    (loop for clump across (clumps bpn) do
      (when (applies-to-p #'cost clump)
        (multiple-value-bind (sum-1 sum-importances-1) (cost clump)
          (incf sum sum-1)
          (incf sum-importances sum-importances-1))))
    (values sum sum-importances)))

(defmethod map-segments (fn (bpn bpn))
  (map nil (lambda (clump)
             (map-segments fn clump))
       (clumps bpn)))

(defmethod write-weights ((bpn bpn) stream)
  (map-segments (lambda (weights)
                  (write-weights weights stream))
                bpn))

(defmethod read-weights ((bpn bpn) stream)
  (map-segments (lambda (weights)
                  (read-weights weights stream))
                bpn))


(defsection @mgl-fnn (:title "Feed-Forward Nets")
  "FNN and RNN have a lot in common (see their common superclass, BPN).
  There is very limited functionality that's specific to FNNs so let's
  get them out of they way before we study a full example."
  (fnn class)
  (build-fnn macro)
  (@mgl-fnn-tutorial section))

(defsection @mgl-fnn-tutorial (:title "FNN Tutorial")
  "Hopefully this example from `example/digit-fnn.lisp` illustrates
  the concepts involved. If it's too dense despite the comments, then
  read up on @MGL-DATASET, @MGL-OPT and come back."
  (digit-fnn.lisp
   (include #.(asdf:system-relative-pathname :mgl "example/digit-fnn.lisp")
            :header-nl "```commonlisp" :footer-nl "```")))

(defclass fnn (bpn)
  ()
  (:documentation "A feed-forward neural net (as opposed to a
  recurrent one, see RNN)."))

;;; LAG needs to be able to look up stuff by name even if several
;;; networks are nested (for example, an LSTM is inside an FNN).
(defvar *names-of-nested-bpns-in-rnn* ())

(defmacro build-fnn ((&key fnn (class ''fnn) initargs
                      max-n-stripes name) &body clumps)
  "Syntactic sugar to assemble FNNs from CLUMPs. Like LET*, it is a
  sequence of bindings (of symbols to CLUMPs). The names of the clumps
  created default to the symbol of the binding. In case a clump is not
  bound to a symbol (because it was created in a nested expression),
  the local function CLUMP can be used to find the clump with the
  given name in the fnn being built. Example:

      (build-fnn ()
        (features (->input :size n-features))
        (biases (->weight :size n-features))
        (weights (->weight :size (* n-hiddens n-features)))
        (activations0 (->mm :weights weights :x (clump 'features)))
        (activations (->+ :args (list biases activations0)))
        (output (->sigmoid :x activations)))"
  (alexandria:with-gensyms (%clump)
    (let ((bindings
            (mapcar (lambda (clump)
                      (destructuring-bind (symbol init-form) clump
                        `(,symbol (let ((,%clump ,(maybe-add-name-to-init
                                                   init-form symbol)))
                                    (when (and ,%clump
                                               (uninterned-symbol-p
                                                (name ,%clump)))
                                      (setf (slot-value ,%clump 'name)
                                            ',symbol))
                                    ,%clump))))
                    clumps)))
      `(let* ((*bpn-being-built* (apply #'make-instance ,class
                                        ,@(when name `(:name ,name))
                                        :max-n-stripes ,max-n-stripes
                                        ,initargs))
              (*names-of-nested-bpns-in-rnn*
                (cons (name *bpn-being-built*) *names-of-nested-bpns-in-rnn*))
              ,@(when fnn
                  `((,fnn *bpn-being-built*))))
         (flet ((clump (name)
                  (find-clump name *bpn-being-built*)))
           (declare (ignorable #'clump))
           (let* ,bindings
             (declare (ignorable ,@(mapcar #'first bindings)))))
         *bpn-being-built*))))

(defun maybe-add-name-to-init (init-form symbol)
  (if (and (symbolp (first init-form))
           (alexandria:starts-with-subseq "->" (symbol-name (first init-form))))
      (append init-form `(:name ',symbol))
      init-form))


(defsection @mgl-rnn (:title "Recurrent Neural Nets")
  (@mgl-rnn-tutorial section)
  (rnn class)
  (unfolder (reader rnn))
  (max-lag (reader rnn))
  (build-rnn macro)
  (lag function)
  (time-step function)
  (set-input (method () (t rnn))))

(defsection @mgl-rnn-tutorial (:title "RNN Tutorial")
  "Hopefully this example from `example/sum-sign-fnn.lisp` illustrates
  the concepts involved. Make sure you are comfortable with
  @MGL-FNN-TUTORIAL before reading this."
  (digit-fnn.lisp
   (include #.(asdf:system-relative-pathname :mgl "example/sum-sign-rnn.lisp")
            :header-nl "```commonlisp" :footer-nl "```")))

;;;; FIXME: don't keep excessively many clumps around?

;;;; FIXME: in forward only operation (i.e. predicting) remove old
;;;; time steps never to be referenced again. For predicting,
;;;; max-time-lag + 1 clumps are enough. The problem is that they are
;;;; not necessarily of the same shape (think IF (= time 0) ...).

;;;; FIXME: MAKE-CLASSIFICATION-ACCURACY-MONITORS* and co don't work
;;;; because they rely on an unimplementable LUMPS.

;;; Weight lumps are shared between clumps corresponding to different
;;; time steps.
(defclass rnn (bpn)
  ((unfolder
    :initarg :unfolder
    :reader unfolder
    :documentation "The UNFOLDER of an RNN is function of no arguments
    that builds and returns a BPN. The unfolder is allowed to create
    networks with arbitrary topology even different ones for different
    [TIME-STEP][]s with the help of LAG, or nested RNNs. Weights of
    the same name are shared between the folds. That is, if a ->WEIGHT
    lump were to be created and a weight lump of the same name already
    exists, then the existing lump will be added to the BPN created by
    UNFOLDER.")
   (max-lag
    :initform 1
    :initarg :max-lag
    :reader max-lag
    :documentation "The networks built by UNFOLDER may contain new
    weights up to time step MAX-LAG. Beyond that point, all weight
    lumps must be reappearances of weight lumps with the same name at
    previous time steps. Most recurrent networks reference only the
    state of lumps at the previous time step (with the function LAG),
    hence the default of 1. But it is possible to have connections to
    arbitrary time steps. The maximum connection lag must be specified
    when creating the RNN.")
   (input-seqs :accessor input-seqs)
   (current-time :initform 0 :accessor current-time)
   (max-time :initform 0 :accessor max-time)
   (weight-lumps :initform () :accessor weight-lumps))
  (:documentation "A recurrent neural net (as opposed to a
  feed-forward one. It is typically built with BUILD-RNN that's no
  more than a shallow convenience macro.

  An RNN takes instances as inputs that are sequences of variable
  length. At each time step, the next unprocessed elements of these
  sequences are set as input until all input sequences in the batch
  run out. To be able to perform backpropagation, all intermediate
  LUMPs must be kept around, so the recursive connections are
  transformed out by
  [unfolding](http://en.wikipedia.org/wiki/Backpropagation_through_time)
  the network. Just how many lumps this means depends on the length of
  the sequences.

  When an RNN is created, `MAX-LAG + 1` BPNs are instantiated so
  that all weights are present and one can start training it."))

(defmethod initialize-instance :after ((rnn rnn) &key &allow-other-keys)
  (loop for i upto (max-lag rnn)
        do (setf (current-time rnn) i)
           (ensure-rnn-bpn rnn)
           (incf (current-time rnn))))

(defmethod print-object ((rnn rnn) stream)
  (pprint-logical-block (stream ())
    (print-unreadable-object (rnn stream :type t :identity t)
      (unless (zerop (length (clumps rnn)))
        (format stream "~@[~S ~]~S ~S/~S ~:_~S ~S ~:_~S ~S"
                (if (uninterned-symbol-p (name rnn))
                    nil
                    (name rnn))
                :stripes (n-stripes rnn) (max-n-stripes rnn)
                :max-lag (max-lag rnn)
                :clumps (length (clumps rnn))))))
  rnn)

(define-descriptions (rnn rnn :inheritp t)
  max-lag)

(defmethod map-segments (fn (rnn rnn))
  (dolist (weights (weight-lumps rnn))
    (funcall fn weights)))

(defmethod set-input (instances (rnn rnn))
  "RNNs operate on batches of instances just like FNNs. But the
  instances here are like datasets: sequences or samplers and they are
  turned into sequences of batches of instances with
  MAP-DATASETS :IMPUTE NIL. The batch of instances at index 2 is
  clamped onto the BPN at time step 2 with SET-INPUT.

  When the input sequences in the batch are not of the same length,
  already exhausted sequences will produce NIL (due to :IMPUTE NIL)
  above. When such a NIL is clamped with SET-INPUT on a BPN of the
  RNN, SET-INPUT must set the IMPORTANCE of the ->ERROR lumps to 0
  else training would operate on the noise left there by previous
  invocations."
  (setf (input-seqs rnn) instances))

;;; Bound by ENSURE-RNN-BPN whenever it calls UNFOLDER in order to let
;;; LAG work anywhere in its dynamic extent.
(defvar *rnn*)

;;; Return the bpn in CLUMPS at index CURRENT-TIME. If necessary
;;; create a new bpn on top of the previous one by calling UNFOLDER.
;;; The previous bpn must already exist.
(defun ensure-rnn-bpn (rnn)
  (let ((clumps (clumps rnn))
        (time (current-time rnn)))
    (assert (<= time (length clumps)) ()
            "Can't create bpn because the previous one doesn't exist.")
    (if (< time (length clumps))
        (aref clumps time)
        (let ((bpn (let ((*rnn* rnn)
                         (*bpn-being-built* rnn))
                     ;; FIXME: Does it work without hiearchical names?
                     ;; Think lump A in fnn F and fnn G. In general,
                     ;; fix naming and clump lookup.
                     (call-with-weights-copied rnn (unfolder rnn)))))
          ;; The set of weights of the RNN is the union of weights of
          ;; its clumps. Remember them to be able to implement
          ;; MAP-SEGMENTS on the RNN.
          (if (< (max-lag rnn) time)
              (check-weights-not-new rnn bpn)
              (map-segments (lambda (weights)
                              (pushnew weights (weight-lumps rnn)))
                            bpn))
          bpn))))

(defun check-weights-not-new (rnn bpn)
  (map-segments (lambda (weights)
                  (assert (find weights (weight-lumps rnn)) ()
                          "After ~S (~S) no new weight lumps can be ~
                          added, but at time step ~S weight ~S has ~
                          no prototype with the same name."
                          :max-lag (max-lag rnn) (length (clumps rnn))
                          (name weights)))
                bpn))

(defun check-rnn (name)
  (assert (boundp '*rnn*) ()
          "~S can only be called from inside BUILD-RNN, or more
          generally from an UNFOLDER function." name))

(defun lag (name &key (lag 1) rnn path)
  "In RNN or if it's NIL the RNN being extended with another
  BPN (called _unfolding_), look up the CLUMP with NAME in the BPN
  that's LAG number of time steps before the BPN being added. This
  function can only be called from UNFOLDER of an RNN which is what
  happens behind the scene in the body of BUILD-RNN.

  FIXDOC: PATH"
  (check-rnn 'lag)
  (let* ((path (or path
                   (and (null rnn)
                        ;; FIXME: The RNN is likely (but not
                        ;; guaranteed) to have an FNN as its
                        ;; replicated child, so we skip it.
                        (rest (reverse *names-of-nested-bpns-in-rnn*)))))
         (rnn (or rnn *rnn*))
         (time (current-time rnn))
         (max-lag (max-lag rnn)))
    (assert (<= lag max-lag) ()
            "Lag ~S is greater than the value of MAX-LAG (~S)." lag max-lag)
    (assert (<= lag time) ()
            "Lag ~S is greater than the current time (~S)." lag time)
    (find-clump name (find-nested-bpn (aref (clumps rnn) (- time lag)) path))))

(defun find-nested-bpn (bpn path)
  (if (endp path)
      bpn
      (let* ((name (first path))
             (nested (find name (clumps bpn) :key #'name :test #'name=)))
        (assert nested () "Can't find nested BPN ~S in ~S." name bpn)
        (find-nested-bpn nested (rest path)))))

(defun time-step ()
  "Return the time step corresponding to the BPN with which an RNN is
  being extended. This is 0 when the RNN is being unfolded for the
  first time. This function can only be called from UNFOLDER of an RNN
  which is what happens behind the scene in the body of BUILD-RNN."
  (cond ((boundp '*rnn*)
         (current-time *rnn*))
        (t
         (error "TIME-STEP can only be called when an RNN is being
                built or forwarded."))))

(defmacro build-rnn ((&key rnn (class ''rnn) name initargs
                      max-n-stripes (max-lag 1)) &body body)
  "Create an RNN with MAX-N-STRIPES and MAX-LAG whose UNFOLDER is BODY
  wrapped in a lambda. Bind symbol given as the RNN argument to the
  RNN object so that BODY can see it."
  (let ((rnn (or rnn (gensym (string '#:rnn))))
        (name (or name `',(gensym))))
    (alexandria:once-only (max-lag)
      `(let ((,rnn (apply #'make-instance ,class
                          :name ,name
                          :max-lag ,max-lag
                          :max-n-stripes ,max-n-stripes
                          :unfolder (lambda ()
                                      (let ((*names-of-nested-bpns-in-rnn* ()))
                                        ,@body))
                          ,initargs)))
         ,rnn))))

(defmethod forward-bpn ((rnn rnn) &key from-clump to-clump end-clump)
  (assert (null from-clump))
  (assert (null to-clump))
  (assert (null end-clump))
  (let ((*rnn* rnn))
    (setf (current-time rnn) 0)
    (map-datasets (lambda (instances)
                    ;; FIXME: only REMOVE-TRAILING-NILS if allowed by
                    ;; some RNN flag
                    (let ((instances (remove-trailing-nils instances))
                          (bpn (ensure-rnn-bpn rnn)))
                      (set-input instances bpn)
                      (forward-bpn bpn))
                    (incf (current-time rnn)))
                  (input-seqs rnn) :impute nil)
    ;; Remember how many clumps were used so that BACKWARD-BPN and
    ;; COST know where to start.
    (setf (max-time rnn) (current-time rnn))))

(defun remove-trailing-nils (seq)
  (let ((last-non-nil-position
          (position nil seq :test-not #'eq :from-end t)))
    (if last-non-nil-position
        (subseq seq 0 (1+ last-non-nil-position))
        ())))

(defmethod backward-bpn ((rnn rnn) &key last-clump)
  (assert (null last-clump))
  (let ((clumps (clumps rnn)))
    (loop for time downfrom (1- (max-time rnn)) downto 0
          do (backward-bpn (aref clumps time)))))

(defmethod cost ((rnn rnn))
  (let ((sum 0)
        (sum-importances 0))
    (loop for bpn across (clumps rnn)
          for i below (max-time rnn)
          do (multiple-value-bind (sum-1 sum-importances-1) (cost bpn)
               (incf sum sum-1)
               (incf sum-importances sum-importances-1)))
    (values sum sum-importances)))


;;;; Train

(defclass bp-learner ()
  ((bpn :initarg :bpn :reader bpn)
   (first-trained-clump :reader first-trained-clump)
   (monitors :initform () :initarg :monitors :accessor monitors)))

(define-descriptions (learner bp-learner :inheritp t)
  bpn first-trained-clump)

(defmethod describe-object :after ((learner bp-learner) stream)
  (when (slot-boundp learner 'bpn)
    (describe (bpn learner) stream)))

(defmethod map-segments (fn (source bp-learner))
  (map-segments fn (bpn source)))

(defmethod initialize-gradient-source* (optimizer (learner bp-learner)
                                        weights dataset)
  (when (next-method-p)
    (call-next-method))
  (setf (slot-value learner 'first-trained-clump) nil))

(defun first-trained-weight-clump (optimizer learner)
  "Much time can be wasted computing derivatives of non-trained weight
  clumps. Return the first one that OPTIMIZER trains."
  ;; FIXME: An RNN has several BPNs that share many of the weights so
  ;; stopping backprop at the first trained weight in a bpn is
  ;; unlikely to work unless it's the very first bpn of the RNN.
  (if (typep (bpn learner) 'rnn)
      nil
      (or (slot-value learner 'first-trained-clump)
          (setf (slot-value learner 'first-trained-clump)
                (find-if (lambda (clump)
                           (member clump (segments optimizer)))
                         (clumps (bpn learner)))))))

(defvar *in-training-p* nil)

(defun compute-derivatives (samples optimizer learner)
  (let ((bpn (bpn learner))
        (cost 0))
    (do-executors (samples bpn)
      (let ((*in-training-p* t))
        (set-input samples bpn)
        (forward bpn)
        (incf cost (cost bpn))
        (zero-non-weight-derivatives bpn)
        (backward-bpn bpn :last-clump (first-trained-weight-clump
                                       optimizer learner))
        (apply-monitors (monitors learner) samples bpn)))
    cost))


;;;; Gradient based optimization

(defun add-and-forget-derivatives (bpn gradient-sink multiplier)
  (do-gradient-sink ((clump accumulator) gradient-sink)
    (axpy! multiplier (derivatives clump) accumulator))
  ;; All weight derivatives must be zeroed, even the ones not being
  ;; trained on to avoid overflows.
  (map-segments (lambda (weights)
                  (fill! 0 (derivatives weights)))
                bpn))

(defmethod accumulate-gradients* ((learner bp-learner) gradient-sink
                                  batch multiplier valuep)
  (let ((bpn (bpn learner))
        (cost 0))
    (loop for samples in (group batch (max-n-stripes bpn))
          do (incf cost (compute-derivatives samples gradient-sink learner)))
    ;; Derivatives of weights keep accumulating in the loop above,
    ;; they are not zeroed like non-weight derivatives, so it's ok to
    ;; call ADD-AND-FORGET-DERIVATIVES once.
    (add-and-forget-derivatives bpn gradient-sink multiplier)
    cost))


;;;; Utilities

(defun monitor-bpn-results (dataset bpn monitors)
  (monitor-model-results (lambda (batch)
                           ;; FIXME: DO-EXECUTORS belongs elsewhere.
                           (do-executors (batch bpn)
                             (set-input batch bpn)
                             (forward bpn))
                           bpn)
                         dataset bpn monitors))

(defmethod make-classification-accuracy-monitors*
    ((bpn bpn) operation-mode label-index-fn attributes)
  (let ((attributes `(,@attributes :model "bpn")))
    (loop for clump across (clumps bpn)
          nconc (make-classification-accuracy-monitors* clump operation-mode
                                                        label-index-fn
                                                        attributes))))

(defmethod make-cross-entropy-monitors* ((bpn bpn) operation-mode
                                         label-index-distribution-fn attributes)
  (let ((attributes `(,@attributes :model "bpn")))
    (loop for clump across (clumps bpn)
          nconc (make-cross-entropy-monitors* clump operation-mode
                                              label-index-distribution-fn
                                              attributes))))
