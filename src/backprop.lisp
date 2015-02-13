(in-package :mgl-bp)

(defsection @mgl-bp (:title "Backpropagation Neural Networks")
  (@mgl-bp-overview section)
  (@mgl-bp-extension-api section)
  (@mgl-bpn section)
  (@mgl-bp-lumps section)
  (@mgl-bp-utilities section))


;;;; Components / Generic backprop

(defsection @mgl-bp-overview (:title "Backprop Overview")
  "Backpropagation Neural Networks are just functions with lots of
  parameters called _weights_ and a layered structure when presented
  as a [computational
  graph](http://en.wikipedia.org/wiki/Automatic_differentiation). The
  network is trained to MINIMIZE some kind of _loss function_ whose
  value the network computes.

  In this implementation, a [BPN][class] is assembled from several
  `LUMP`s (roughly corresponding to layers). Both feed-forward and
  recurrent neural nets are supported (FNN and RNN, respectively).
  `BPN`s can contain not only `LUMP`s but other `BPN`s, too. As we
  see, networks are composite objects and the abstract base class for
  composite and simple parts is called CLUMP."
  (clump class)
  "At this point, you may want to jump ahead to get a feel for how
  things work by reading the @MGL-FNN-TUTORIAL.")

;;; Everything is defined with DEFCLASS-NOW because DEFMAKER needs the
;;; class to be around at macro expansion time.
(defclass-now clump ()
  ((name :initform (gensym) :initarg :name :reader name))
  (:documentation "A CLUMP is a LUMP or a [BPN][class]. It represents
  a differentiable function. Arguments of clumps are given during
  instantiation. Some arguments are clumps themselves so they get
  permenantly wired together like this:

  ```commonlisp
  (->v*m (->input :size 10 :name 'input)
         (->weight :dimensions '(10 20) :name 'weight)
         :name 'activation)
  ```

  The above creates three clumps: the vector-matrix multiplication
  clumps called `ACTIVATION` which has a reference to its operands:
  INPUT and WEIGHT. Note that the example just defines a function, no
  actual computation has taken place, yet.

  This wiring of `CLUMP`s is how one builds feed-forward nets (FNN) or
  recurrent neural networks (RNN) that are `CLUMP`s themselves so one
  can build nets in a hiearchical style if desired. Non-composite
  `CLUMP`s are called LUMP (note the loss of `C` that stands for
  composite). The various LUMP subtypes correspond to different layer
  types (->SIGMOID, ->DROPOUT, ->RELU, ->TANH, etc)."))

(defvar *bpn-being-built* nil)

(defmethod initialize-instance :around ((clump clump) &key &allow-other-keys)
  (call-next-method)
  (if *bpn-being-built*
      ;; This sets MAX-N-STRIPES to that of *BPN-BEING-BUILT*.
      (add-clump clump *bpn-being-built*)
      ;; If we aren't building a bpn, let's ensure that the matrices
      ;; are allocated.
      (setf (max-n-stripes clump) (max-n-stripes clump))))


(defsection @mgl-bp-extension-api (:title "Clump API")
  "These are mostly for extension purposes. About the only thing
  needed from here for normal operation is NODES when clamping inputs
  or extracting predictions."
  (stripedp generic-function)
  (nodes generic-function)
  "`CLUMP`s' `NODES` holds the result computed by the most recent
  FORWARD. For ->INPUT lumps, this is where input values shall be
  placed (see SET-INPUT). Currently, the matrix is always two
  dimensional but this restriction may go away in the future."
  (derivatives generic-function)
  (forward generic-function)
  (backward generic-function)
  "In addition to the above, clumps also have to support SIZE,
  N-STRIPES, MAX-N-STRIPES (and the SETF methods of the latter two)
  which can be accomplished just by inheriting from BPN, FNN, RNN, or
  a LUMP.")

(defgeneric stripedp (clump)
  (:documentation "For efficiency, forward and backprop phases do
  their stuff in batch mode: passing a number of instances through the
  network in batches. Thus clumps must be able to store values of and
  gradients for each of these instances. However, some clumps produce
  the same result for each instance in a batch. These clumps are the
  weights, the parameters of the network. STRIPEDP returns true iff
  CLUMP does not represent weights (i.e. it's not a ->WEIGHT).

  For striped clumps, their NODES and DERIVATIVES are MAT objects with
  a leading dimension (number of rows in the 2d case) equal to the
  number of instances in the batch. Non-striped clumps have no
  restriction on their shape apart from what their usage dictates.")
  (:method ((clump clump))
    t))

(defgeneric derivatives (clump)
  (:documentation "Return the MAT object representing the partial
  derivatives of the function CLUMP computes. The returned partial
  derivatives were accumulated by previous BACKWARD calls.

  This matrix is shaped like the matrix returned by NODES."))

(defgeneric forward (clump)
  (:documentation "Compute the values of the function represented by
  CLUMP for all stripes and place the results into NODES of CLUMP."))

(declaim (special *in-training-p*))

(defmethod forward :after (clump)
  ;; Prepare for backward pass by zeroing non-weight derivatives.
  (when (and *in-training-p*
             (stripedp clump)
             (derivatives clump))
    (fill! 0 (derivatives clump))))

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

(defgeneric map-clumps (fn clump)
  (:method (fn clump)
    (funcall fn clump)))


(defsection @mgl-bpn (:title "BPNs")
  (bpn class)
  (n-stripes (reader bpn))
  (max-n-stripes (reader bpn))
  (clumps (reader bpn))
  (find-clump function)
  (add-clump function)
  (@mgl-bp-training section)
  (@mgl-bp-monitoring section)
  (@mgl-fnn section)
  (@mgl-rnn section))

(defclass bpn (clump)
  ((clumps
    :initform (make-array 0 :element-type 'clump :adjustable t :fill-pointer t)
    :initarg :clumps
    :type (array clump (*)) :reader clumps
    :documentation "A topological sorted adjustable array with a fill
    pointer that holds the clumps that make up the network. Clumps are
    added to it by ADD-CLUMP or, more often, automatically when within
    a BUILD-FNN or BUILD-RNN. Rarely needed, FIND-CLUMP takes care of
    most uses.")
   (n-stripes
    :initform 1 :type index :initarg :n-stripes :reader n-stripes
    :documentation "The current number of instances the network has.
    This is automatically set to the number of instances passed to
    SET-INPUT, so it rarely has to be manipulated directly although it
    can be set. When set N-STRIPES of all CLUMPS get set to the same
    value.")
   (max-n-stripes
    :initform nil :type index :initarg :max-n-stripes
    :reader max-n-stripes
    :documentation "The maximum number of instances the network can
    operate on in parallel. Within BUILD-FNN or BUILD-RNN, it defaults
    to MAX-N-STRIPES of that parent network, else it defaults to 1.
    When set MAX-N-STRIPES of all CLUMPS get set to the same value.")
   ;; The cost calculated by the most recent FORWARD.
   (last-cost :initform (list 0 0) :accessor last-cost))
  (:documentation "Abstract base class for FNN and RNN."))

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
    (print-unreadable-object (bpn stream :type t)
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

(defmethod map-clumps (fn (bpn bpn))
  (call-next-method)
  (map nil (lambda (clump)
             (map-clumps fn clump))
       (clumps bpn)))

(defun find-clump (name bpn &key (errorp t))
  "Find the clump with NAME among CLUMPS of BPN. As always, names are
  compared with EQUAL. If not found, then return NIL or signal and
  error depending on ERRORP."
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
  "Add CLUMP to BPN. MAX-N-STRIPES of CLUMP gets set to that of BPN.
  It is an error to add a clump with a name already used by one of the
  CLUMPS of BPN."
  (when (find-clump (name clump) bpn :errorp nil)
    (error "Cannot add ~S: a clump of same name has already been ~
           added to this network." clump))
  (setf (max-n-stripes clump) (max-n-stripes bpn))
  (vector-push-extend clump (slot-value bpn 'clumps) 1)
  clump)

(defgeneric forward-bpn (bpn &key from-clump to-clump end-clump)
  (:documentation "Propagate the values from the already clamped
  inputs and return total cost of all inputs (i.e. all stripes) and
  the sum of importances. These values are also returned by COST until
  the next forward pass on BPN."))

(defgeneric backward-bpn (bpn &key last-clump)
  (:documentation "Accumulate derivatives of weights."))

;;; Derivatives of weights are left alone to let them accumulate which
;;; is useful in batches such as when training with conjugate
;;; gradient.

(defmethod forward ((bpn bpn))
  (forward-bpn bpn))

(defmethod backward ((bpn bpn))
  (backward-bpn bpn))

(defmethod forward-bpn ((bpn bpn) &key from-clump to-clump end-clump)
  (declare (optimize (debug 3)))
  (let ((seen-from-clump-p (not from-clump))
        (sum-cost 0)
        (sum-importances 0))
    (loop for clump across (clumps bpn)
          until (eq clump end-clump)
          do (when (eq clump from-clump) (setq seen-from-clump-p t))
          do (when seen-from-clump-p
               (forward clump)
               (when (applies-to-p #'cost clump)
                 (multiple-value-bind (sum-cost-1 sum-importances-1)
                     (cost clump)
                   (incf sum-cost sum-cost-1)
                   (incf sum-importances sum-importances-1))))
          until (eq clump to-clump))
    (setf (last-cost bpn) (list sum-cost sum-importances))
    (values sum-cost sum-importances)))

(defmethod backward-bpn ((bpn bpn) &key last-clump)
  (let ((clumps (clumps bpn)))
    (loop for i downfrom (1- (length clumps)) downto 0
          for clump = (aref clumps i)
          until (and last-clump (eq last-clump clump))
          do (backward clump))))

(defmethod cost ((bpn bpn))
  (values-list (last-cost bpn)))

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
        (activations0 (->v*m :weights weights :x (clump 'features)))
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
  (cuda-window-start-time (accessor rnn))
  (*cuda-window-start-time* variable)
  (build-rnn macro)
  (lag function)
  (time-step function)
  (set-input (method () (t rnn)))
  (@mgl-rnn-time-warp section))

(defsection @mgl-rnn-tutorial (:title "RNN Tutorial")
  "Hopefully this example from `example/sum-sign-fnn.lisp` illustrates
  the concepts involved. Make sure you are comfortable with
  @MGL-FNN-TUTORIAL before reading this."
  (sum-sig-rnn.lisp
   (include #.(asdf:system-relative-pathname :mgl "example/sum-sign-rnn.lisp")
            :header-nl "```commonlisp" :footer-nl "```")))

(defvar *cuda-window-start-time* nil
  "The default for CUDA-WINDOW-START-TIME.")

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
   (weight-lumps :initform () :accessor weight-lumps)
   (warp-start
    :initform 1 :initarg :warp-start :reader warp-start
    :documentation "The TIME-STEP from which UNFOLDER will create
    `BPN`s that essentially repeat every WARP-LENGTH steps.")
   (warp-length
    :initform 1 :initarg :warp-length :reader warp-length
    :documentation "An integer such that the BPN UNFOLDER creates at
    time step `I` (where `(<= WARP-START I)`) is identical to the BPN
    created at time step `(+ WARP-START (MOD (- I WARP-START)
    WARP-LENGTH))` except for a shift in its time lagged
    connections.")
   (cuda-window-start-time
    :initform *cuda-window-start-time*
    :initarg :cuda-window-start-time
    :accessor cuda-window-start-time
    :documentation "Due to unfolding, the memory footprint of an RNN
    is almost linear in the number of time steps (i.e. the max
    sequence length). For prediction, this is addressed by
    @MGL-RNN-TIME-WARP. For training, we cannot discard results of
    previous time steps because they are needed for backpropagation,
    but we can at least move them out of GPU memory if they are not
    going to be used for a while and copy them back before they are
    needed. Obviously, this is only relevant if CUDA is being used.

    If CUDA-WINDOW-START-TIME is NIL, then this feature is turned off.
    Else, during training, at CUDA-WINDOW-START-TIME or later time
    steps, matrices belonging to non-weight lumps may be forced out of
    GPU memory and later brought back as neeeded.

    This feature is implemented in terms of
    MGL-MAT:WITH-SYNCING-CUDA-FACETS that uses CUDA host memory (also
    known as _page-locked_ or _pinned memory_) to do asynchronous
    copies concurrently with normal computation. The consequence of
    this is that it is now main memory usage that's unbounded which
    toghether with page-locking makes it a potent weapon to bring a
    machine to a halt. You were warned.")
   (step-monitors
    :initform () :initarg :step-monitors :accessor step-monitors
    :documentation "During training, unfolded `BPN`s corresponding to
    previous time steps may be expensive to get at because they are no
    longer in GPU memory. This consideration also applies to making
    prediction with the additional caveat that with *WARP-TIME* true,
    previous states are discarded so it's not possible to gather
    statistics after FORWARD finished.

    Add monitor objects to this slot and they will be automatically
    applied to the RNN after each step when `FORWARD`ing the RNN
    during training or prediction. To be able to easily switch between
    sets of monitors, in addition to a list of monitors this can be a
    symbol or a function, too. If it's a symbol, then its a designator
    for its SYMBOL-VALUE. If it's a function, then it must have no
    arguments and it's a designator for its return value.")
   ;; KLUDGE: If true, then as a performance hack, at each time step
   ;; drop the instances that belong to input sequences that have run
   ;; out. This results in the number of stripes being different from
   ;; step to step which is a pain so only ->* supports it as of now
   ;; which is enough for LSTMs, but this is hackish enough that it
   ;; will probably never be exported.
   (remove-trailing-nil-instances
    :initform nil
    :initarg :remove-trailing-nil-instances
    :accessor remove-trailing-nil-instances))
  (:documentation "A recurrent neural net (as opposed to a
  feed-forward one. It is typically built with BUILD-RNN that's no
  more than a shallow convenience macro.

  An RNN takes instances as inputs that are sequences of variable
  length. At each time step, the next unprocessed elements of these
  sequences are set as input until all input sequences in the batch
  run out. To be able to perform backpropagation, all intermediate
  `LUMP`s must be kept around, so the recursive connections are
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
;;; LAG and TIME-STEP work anywhere in its dynamic extent.
(defvar *rnn*)

(defun time-step (&key (rnn *rnn*))
  "Return the time step RNN is currently executing or being unfolded for.
  It is 0 when the RNN is being unfolded for the first time."
  (current-time rnn))

(defsection @mgl-rnn-time-warp (:title "Time Warp")
  "The unbounded memory usage of `RNN`s with one BPN allocated per
  time step can become a problem. For training, where the gradients
  often have to be backpropagated from the last time step to the very
  beginning, this is hard to solve but with CUDA-WINDOW-START-TIME the
  limit is no longer GPU memory.

  For prediction on the other hand, one doesn't need to keep old steps
  around indefinitely: they can be discarded when future time steps
  will never reference them again."
  (*warp-time* variable)
  (warped-time function)
  (warp-start (reader rnn))
  (warp-length (reader rnn))
  (step-monitors (accessor rnn)))

(defvar *warp-time* nil
  "Controls whether warping is enabled (see @MGL-RNN-TIME-WARP). Don't
  enable it for training, as it would make backprop impossible.")

(defun warped-time (&key (rnn *rnn*) (time (time-step :rnn rnn)) (lag 0))
  "Return the index of the BPN in CLUMPS of RNN whose task it is to
  execute computation at `(- (TIME-STEP RNN) LAG)`. This is normally
  the same as TIME-STEP (disregarding LAG). That is, CLUMPS can be
  indexed by TIME-STEP to get the BPN. However, when *WARP-TIME* is
  true, execution proceeds in a cycle as the structure of the network
  allows.

  Suppose we have a typical RNN that only ever references the previous
  time step so its MAX-LAG is 1. Its UNFOLDER returns `BPN`s of
  identical structure bar a shift in their time lagged connections
  except for the very first, so WARP-START and WARP-LENGTH are both 1.
  If *WARP-TIME* is NIL, then the mapping from TIME-STEP to the BPN in
  CLUMPS is straightforward:

      time:   |  0 |  1 |  2 |  3 |  4 |  5
      --------+----+----+----+----+----+----
      warped: |  0 |  1 |  2 |  3 |  4 |  5
      --------+----+----+----+----+----+----
      bpn:    | b0 | b1 | b2 | b3 | b4 | b5

  When *WARP-TIME* is true, we reuse the `B1` - `B2` bpns in a loop:

      time:   |  0 |  1 |  2 |  3 |  4 |  5
      --------+----+----+----+----+----+----
      warped: |  0 |  1 |  2 |  1 |  2 |  1
      --------+----+----+----+----+----+----
      bpn:    | b0 | b1 | b2 | b1*| b2 | b1*

  `B1*` is the same BPN as `B1`, but its connections created by LAG go
  through warped time and end up referencing `B2`. This way, memory
  consumption is independent of the number time steps needed to
  process a sequence or make predictions.

  To be able to pull this trick off WARP-START and WARP-LENGTH must be
  specified when the RNN is instantiated. In general, with
  *WARP-TIME* `(+ WARP-START (MAX 2 WARP-LENGTH))` bpns are needed.
  The 2 comes from the fact that with cycle length 1 a bpn would need
  to takes its input from itself which is problematic because it has
  NODES for only one set of values."
  (let ((step (- time lag))
        (warp-length (max 2 (warp-length rnn)))
        (warp-start (warp-start rnn)))
    (if (or (not *warp-time*)
            (< step (+ warp-start warp-length)))
        step
        (+ warp-start (mod (- step warp-start) warp-length)))))

;;; Return the bpn in CLUMPS at index CURRENT-TIME. If necessary
;;; create a new bpn on top of the previous one by calling UNFOLDER.
;;; The previous bpn must already exist.
(defun ensure-rnn-bpn (rnn)
  (let ((clumps (clumps rnn))
        (step (warped-time :rnn rnn)))
    (assert (<= step (length clumps)) ()
            "Can't create bpn because the previous one doesn't exist.")
    (if (< step (length clumps))
        (aref clumps step)
        (let ((bpn (let ((*rnn* rnn)
                         (*bpn-being-built* rnn))
                     ;; FIXME: Does it work without hiearchical names?
                     ;; Think lump A in fnn F and fnn G. In general,
                     ;; fix naming and clump lookup.
                     (call-with-weights-copied rnn (unfolder rnn)))))
          ;; The set of weights of the RNN is the union of weights of
          ;; its clumps. Remember them to be able to implement
          ;; MAP-SEGMENTS on the RNN.
          (if (< (max-lag rnn) (current-time rnn))
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
  that's LAG number of time steps before the BPN being added. If this
  function is called from UNFOLDER of an RNN (which is what happens
  behind the scene in the body of BUILD-RNN), then it returns an
  opaque object representing a lagged connection to a clump, else it
  returns the CLUMP itself.

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
    (let ((ref (make-lagged-clump :path path :name name :lag lag)))
      (cond (*bpn-being-built*
             (resolve-clump rnn ref)
             ref)
            (t
             (resolve-clump rnn ref))))))

(defstruct lagged-clump
  path
  name
  lag)

(defgeneric resolve-clump (rnn ref)
  (:method (rnn (clump clump))
    clump)
  (:method (rnn (lagged lagged-clump))
    (let* ((path (lagged-clump-path lagged))
           (name (lagged-clump-name lagged))
           (lag (lagged-clump-lag lagged))
           (step (warped-time :rnn rnn :lag lag)))
      (find-clump name (find-nested-bpn (aref (clumps rnn) step) path)))))

(defun resolve-clumps (object)
  (if (not (boundp '*rnn*))
      object
      (if (listp object)
          (loop for e in object collect (resolve-clump *rnn* e))
          (resolve-clump *rnn* object))))

(defun find-nested-bpn (bpn path)
  (if (endp path)
      bpn
      (let* ((name (first path))
             (nested (find name (clumps bpn) :key #'name :test #'name=)))
        (assert nested () "Can't find nested BPN ~S in ~S." name bpn)
        (find-nested-bpn nested (rest path)))))

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

(defun resolve-step-monitors (step-monitors)
  (cond ((symbolp step-monitors)
         (symbol-value step-monitors))
        ((functionp step-monitors)
         (funcall step-monitors))
        (t step-monitors)))

(defmethod forward-bpn ((rnn rnn) &key from-clump to-clump end-clump)
  (assert (null from-clump))
  (assert (null to-clump))
  (assert (null end-clump))
  (let ((*rnn* rnn)
        (sum-cost 0)
        (sum-importances 0)
        (step-monitors (resolve-step-monitors (step-monitors rnn))))
    ;; MAX-TIME is not known and NIL also means that we are
    ;; forwarding.
    (setf (max-time rnn) nil)
    (setf (current-time rnn) 0)
    (map-datasets (lambda (instances)
                    (multiple-value-bind (to-cuda to-host)
                        (rnn-forward-cuda-syncs rnn)
                      (with-syncing-cuda-facets (to-cuda to-host)
                        (let ((instances
                                (if (remove-trailing-nil-instances rnn)
                                    (remove-trailing-nils instances)
                                    instances))
                              (bpn (ensure-rnn-bpn rnn)))
                          (set-input instances bpn)
                          (multiple-value-bind (sum-cost-1 sum-importances-1)
                              (forward-bpn bpn)
                            (incf sum-cost sum-cost-1)
                            (incf sum-importances sum-importances-1))
                          (apply-monitors step-monitors instances rnn))))
                    (incf (current-time rnn)))
                  (input-seqs rnn) :impute nil)
    ;; Remember how many clumps were used so that BACKWARD-BPN and
    ;; COST know where to start.
    (setf (max-time rnn) (current-time rnn))
    (setf (last-cost rnn) (list sum-cost sum-importances))
    (values sum-cost sum-importances)))

(defun remove-trailing-nils (seq)
  (let ((last-non-nil-position
          (position nil seq :test-not #'eq :from-end t)))
    (if last-non-nil-position
        (subseq seq 0 (1+ last-non-nil-position))
        ())))

(defmethod backward-bpn ((rnn rnn) &key last-clump)
  (assert (null last-clump))
  (let ((*rnn* rnn)
        (clumps (clumps rnn)))
    (loop for time downfrom (1- (max-time rnn)) downto 0
          do (setf (current-time rnn) time)
             (multiple-value-bind (to-cuda to-host)
                 (rnn-backward-cuda-syncs rnn)
               (with-syncing-cuda-facets (to-cuda to-host)
                 (backward-bpn (aref clumps time)))))))

;;; Return the MATs needed in the next step, and the MATs that aren't
;;; going to be accessed in the forward pass anymore (those earlier
;;; than TIME - MAX-LAG).
(defun rnn-forward-cuda-syncs (rnn)
  (when (and (cuda-window-start-time rnn)
             (not *warp-time*)
             (use-cuda-p))
    (let ((clumps (clumps rnn))
          (time (time-step :rnn rnn))
          (destroy-start-index (cuda-window-start-time rnn)))
      (values (let ((time (1+ time)))
                (if (< time (length clumps))
                    (collect-non-weight-mats-for-cuda-sync (aref clumps time))
                    ()))
              (let ((time (- time (1+ (max-lag rnn)))))
                (if (<= destroy-start-index time)
                    (collect-non-weight-mats-for-cuda-sync (aref clumps time))
                    ()))))))

;;; Return the MATs needed in the next backprop step, and the MATs
;;; that aren't going to be accessed in the backward pass anymore
;;; (simply TIME + 1).
(defun rnn-backward-cuda-syncs (rnn)
  (when (and (cuda-window-start-time rnn)
             (not *warp-time*)
             (use-cuda-p))
    (let ((clumps (clumps rnn))
          (time (time-step :rnn rnn))
          (destroy-start-index (cuda-window-start-time rnn)))
      (values (let ((time (- time 2)))
                (if (<= 0 time)
                    (collect-non-weight-mats-for-cuda-sync (aref clumps time))
                    ()))
              (let ((time (1+ time)))
                (if (and (< time (length clumps))
                         (<= destroy-start-index time))
                    (collect-non-weight-mats-for-cuda-sync (aref clumps time))
                    ()))))))

(defun collect-non-weight-mats-for-cuda-sync (bpn)
  (let ((mats ()))
    (flet ((maybe-collect (mat)
             (when (and mat (cuda-enabled mat))
               (push mat mats))))
      (map-clumps (lambda (clump)
                    (when (and (typep clump 'lump)
                               (not (typep clump '->weight)))
                      (maybe-collect (nodes clump))
                      (maybe-collect (derivatives clump))))
                  bpn))
    mats))

(defmethod cost ((rnn rnn))
  (if (null (max-time rnn))
      ;; RNN is being forwarded, so just return the cost of the
      ;; current time step.
      (cost (aref (clumps rnn) (warped-time :rnn rnn)))
      (values-list (last-cost rnn))))


(defsection @mgl-bp-training (:title "Training")
  "`BPN`s are trained to minimize the loss function they compute.
  Before a BPN is passed to MINIMIZE (as its `GRADIENT-SOURCE`
  argument), it must be wrapped in a BP-LEARNER object. BP-LEARNER has
  [MONITORS][(accessor bp-learner)] slot which is used for example by
  [RESET-OPTIMIZATION-MONITORS][(method () (iterative-optimizer
  t))].

  Without the bells an whistles, the basic shape of training is this:

  ```commonlisp
  (minimize optimizer (make-instance 'bp-learner :bpn bpn)
            :dataset dataset)
  ```"
  (bp-learner class)
  (bpn (reader bp-learner))
  (monitors (reader bp-learner)))

(defclass bp-learner ()
  ((bpn
    :initarg :bpn :reader bpn
    :documentation "The BPN for which this BP-LEARNER provides the
    gradients.")
   (first-trained-clump :reader first-trained-clump)
   (monitors
    :initform () :initarg :monitors :accessor monitors
    :documentation "A list of `MONITOR`s.")))

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

(defvar *in-training-p* nil)

(defun compute-derivatives (samples optimizer learner)
  (declare (ignore optimizer))
  (let ((bpn (bpn learner))
        (cost 0))
    (do-executors (samples bpn)
      (let ((*in-training-p* t))
        (set-input samples bpn)
        (incf cost (forward-bpn bpn))
        (backward-bpn bpn)
        (apply-monitors (monitors learner) samples bpn)))
    cost))


;;;; Gradient based optimization

(defun add-and-forget-derivatives (bpn gradient-sink multiplier)
  (let ((clumps-not-to-be-zeroed ()))
    (do-gradient-sink ((clump accumulator) gradient-sink)
      (if (eq (derivatives clump) accumulator)
          ;; The optimizer is using DERIVATIVES directly as its
          ;; accumulator and will zero it when it sees it.
          (push clump clumps-not-to-be-zeroed)
          (axpy! multiplier (derivatives clump) accumulator)))
    ;; All weight derivatives must be zeroed, even the ones not being
    ;; trained on to avoid overflows.
    (map-segments (lambda (weights)
                    (unless (find weights clumps-not-to-be-zeroed)
                      (fill! 0 (derivatives weights))))
                  bpn)))

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


(defsection @mgl-bp-monitoring (:title "Monitoring")
  (monitor-bpn-results function)
  #+nil
  (make-classification-accuracy-monitors* (method () (bpn t t t)))
  #+nil
  (make-cross-entropy-monitors* (method () (bpn t t t)))
  (make-step-monitor-monitors function)
  (make-step-monitor-monitor-counter generic-function))

(defun monitor-bpn-results (dataset bpn monitors)
  "For every batch (of size MAX-N-STRIPES of BPN) of instances in
  DATASET, set the batch as the next input with SET-INPUT, perform a
  FORWARD pass and apply MONITORS to the BPN (with APPLY-MONITORS).
  Finally, return the counters of MONITORS. This is built on top of
  MONITOR-MODEL-RESULTS."
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

(defun make-step-monitor-monitors
    (rnn &key (counter-values-fn #'counter-raw-values)
     (make-counter #'make-step-monitor-monitor-counter))
  "Return a list of monitors, one for every monitor in STEP-MONITORS
  of RNN. These monitors extract the results from their warp
  counterpairs with COUNTER-VALUES-FN and add them to their own
  counter that's created by MAKE-COUNTER. Wow. Ew. The idea is that
  one does something like this do monitor warped prediction:

  ```commonlisp
  (let ((*warp-time* t))
    (setf (step-monitors rnn)
          (make-cost-monitors rnn :attributes '(:event \"warped pred.\")))
    (monitor-bpn-results dataset rnn
                         ;; Just collect and reset the warp
                         ;; monitors after each batch of
                         ;; instances.
                         (make-step-monitor-monitors rnn)))
  ```"
  (mapcar (lambda (step-monitor)
            (make-instance
             'monitor
             :measurer (lambda (instances result)
                         (declare (ignore instances result))
                         (let ((counter (counter step-monitor)))
                           (multiple-value-prog1
                               (funcall counter-values-fn counter)
                             (reset-counter counter))))
             :counter (funcall make-counter (counter step-monitor))))
          (resolve-step-monitors (step-monitors rnn))))

(defgeneric make-step-monitor-monitor-counter (step-counter)
  (:documentation "In an RNN, STEP-COUNTER aggregates results of all
  the time steps during the processing of instances in the current
  batch. Return a new counter into which results from STEP-COUNTER can
  be accumulated when the processing of the batch is finished. The
  default implementation creates a copy of STEP-COUNTER.")
  (:method (step-counter)
    (copy 'make-step-monitor-monitor-counter step-counter)))
