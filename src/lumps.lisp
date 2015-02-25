(in-package :mgl-bp)

(defsection @mgl-bp-lumps (:title "Lumps")
  (@mgl-bp-lump section)
  (@mgl-bp-inputs section)
  (@mgl-bp-weight-lump section)
  (@mgl-bp-activations section)
  (@mgl-bp-activation-functions section)
  (@mgl-bp-losses section)
  (@mgl-bp-stochasticity section)
  (@mgl-bp-arithmetic section)
  (@mgl-bp-rnn-operations section))


(defsection @mgl-bp-lump (:title "Lump Base Class")
  (lump class)
  (size (reader lump))
  (default-value (reader lump))
  (default-size generic-function)
  (nodes (reader lump))
  (derivatives (reader lump)))

(defgeneric default-size (lump)
  (:method (lump)
    (or (slot-boundp lump 'size)
        (error "Can't compute size for ~S." lump)))
  (:documentation "Return a default for the [SIZE][(reader lump)] of
  LUMP if one is not supplied at instantiation. The value is often
  computed based on the sizes of the inputs. This function is for
  implementing new lump types."))

(defun check-size-and-default-size (lump size)
  (when (and size (/= size (default-size lump)))
    (error "~S was given a SIZE ~S that's different from the ~
    automatically calculated size ~S." (type-of lump) size
    (default-size lump))))

(defclass-now lump (clump)
  ((size
    :type index :initarg :size :reader size
    :documentation "The number of values in a single stripe.")
   (nodes
    :initform nil :type (or mat null) :reader nodes
    :documentation "The values computed by the lump in the forward
    pass are stored here. It is an `N-STRIPES * SIZE` matrix that has
    storage allocated for `MAX-N-STRIPES * SIZE` elements for
    non-weight lumps. ->WEIGHT lumps have no stripes nor restrictions
    on their shape.")
   (derivatives
    :type (or mat null) :reader derivatives
    :documentation "The derivatives computed in the backward pass are
    stored here. This matrix is very much like [NODES][(reader lump)]
    in shape and size.")
   (default-value
    :initform 0 :initarg :default-value :type real
    :reader default-value
    :documentation "Upon creation or resize the lump's nodes get
    filled with this value.")
   (shared-with-clump
    :initform nil
    :initarg :shared-with-clump
    :reader shared-with-clump))
  (:documentation "A LUMP is a simple, layerlike component of a neural
  network. There are many kinds of lumps, each of which performs a
  specific operation or just stores inputs and weights. By convention,
  the names of lumps start with the prefix `->`. Defined as classes,
  they also have a function of the same name as the class to create
  them easily. These maker functions typically have keyword arguments
  corresponding to initargs of the class, with some (mainly the input
  lumps) turned into normal positional arguments. So instead of having
  to do

      (make-instance '->tanh :x some-input :name 'my-tanh)

  one can simply write

      (->tanh some-input :name 'my-tanh)

  Lumps instantiated in any way within a BUILD-FNN or BUILD-RNN are
  automatically added to the network being built.

  A lump has its own NODES and DERIVATIVES matrices allocated for it
  in which the results of the forward and backward passes are stored.
  This is in contrast to a [BPN][class] whose NODES and DERIVATIVES
  are those of its last constituent CLUMP.

  Since lumps almost always live within a BPN, their
  [N-STRIPES][(reader bpn)] and [MAX-N-STRIPES][(reader bpn)] are
  handled automagically behind the scenes."))

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

(defgeneric print-lump-parts (lump stream)
  (:method (lump stream)
    (declare (ignore lump stream))))

(defmethod print-object ((lump lump) stream)
  (pprint-logical-block (stream ())
    (print-unreadable-object (lump stream :type t)
      (format stream "~S ~:_~S ~:_~S" (name lump) :size
              (if (slot-boundp lump 'size)
                  (size lump)
                  :unbound))
      (let ((mgl-cube:*let-input-through-p* t))
        (format stream " ~S/~S ~S ~,5F" (n-stripes lump) (max-n-stripes lump)
                :norm (ignore-errors (nrm2 (nodes lump)))))
      (print-lump-parts lump stream)))
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

(defmethod segment-derivatives ((lump lump))
  (derivatives lump))

(defmethod non-constant-mats ((lump lump))
  (if (derivatives lump)
      (list (nodes lump) (derivatives lump))
      (list (nodes lump))))

;;; Only weights are segments. Nothing to do for other lumps.
(defmethod map-segments (fn (lump lump)))

(defmethod write-state* ((lump lump) stream context))

(defmethod read-state* ((lump lump) stream context))


(defsection @mgl-bp-weight-lump (:title "Weight Lump")
  (->weight class)
  (dimensions (reader ->weight))
  (with-weights-copied macro))

(defclass-now ->weight (lump)
  ((dimensions
    :initarg :dimensions :reader dimensions
    :documentation "NODES and DERIVATIVES of this lump will be
    allocated with these dimensions."))
  (:documentation "A set of optimizable parameters of some kind. When
  a BPN is is trained (see @MGL-BP-TRAINING) the NODES of weight lumps
  will be changed. Weight lumps perform no computation.

  Weights can be created by specifying the total size or the
  dimensions:

  ```cl-transcript
  (dimensions (->weight :size 10 :name 'w))
  => (1 10)
  (dimensions (->weight :dimensions '(5 10) :name 'w))
  => (5 10)
  ```"))

(defmaker (->weight :make-instance-args args)
  (maybe-copy-weight '->weight args))

(defmethod initialize-instance :around ((weight ->weight) &key dimensions size
                                        &allow-other-keys)
  (assert (or size dimensions) () "SIZE or DIMENSIONS must be specified.")
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
  lump normally. If FROM-BPN is NIL, then no weights are copied."
  `(call-with-weights-copied ,from-bpn (lambda () ,@body)))

(defun maybe-copy-weight (class-name args)
  (assert (getf args :name) ()
          "~A lumps must be named explicitly to allow weight ~
          sharing to work." class-name)
  (let* ((name (getf args :name))
         (to-be-copied (find name *lumps-to-copy* :key #'name :test #'name=)))
    (cond (to-be-copied
           (when *bpn-being-built*
             (add-clump to-be-copied *bpn-being-built*))
           to-be-copied)
          (t
           (apply #'make-instance class-name args)))))

(defmethod non-constant-mats ((lump ->weight))
  ())

(defmethod forward ((lump ->weight)))

(defmethod backward ((lump ->weight)))

(defmethod map-segments (fn (lump ->weight))
  (funcall fn lump))

(defmethod write-state* ((lump ->weight) stream context)
  (write-mat (nodes lump) stream))

(defmethod read-state* ((lump ->weight) stream context)
  (read-mat (nodes lump) stream))


;;;; Resolve references to lagged clumps behind the scenes.

;;; silence style warnings
(defgeneric x (object))
(defgeneric y (object))
(defgeneric args (object))

(defmethod x :around (object)
  (resolve-clumps (call-next-method)))

(defmethod y :around (object)
  (resolve-clumps (call-next-method)))

(defmethod args :around (object)
  (resolve-clumps (call-next-method)))

(defmethod name ((ref lagged-clump))
  (name (resolve-clump *rnn* ref)))

(defmethod size ((ref lagged-clump))
  (size (resolve-clump *rnn* ref)))


;;;; Define dropout before ->INPUT that inherits from it.

(defsection @mgl-bp-dropout-lump (:title "Dropout Lump")
  (->dropout class)
  (dropout (accessor ->dropout)))

(defclass-now ->dropout (lump)
  ((x :initarg :x :reader x)
   (dropout
    :type (or null real)
    :initform 0.5 :initarg :dropout :accessor dropout
    :documentation "If non-NIL, then in the forward pass zero out each
    node in this chunk with DROPOUT probability.")
   (mask :initform nil :reader mask))
  (:documentation "The output of this lump is identical to its input,
  except it randomly zeroes out some of them during training which act
  as a very strong regularizer. See Geoffrey Hinton's 'Improving
  neural networks by preventing co-adaptation of feature
  detectors'.

  The SIZE of this lump is the size of its input which is determined
  automatically."))

(defmethod initialize-instance :after ((lump ->dropout)
                                       &key size &allow-other-keys)
  (check-size-and-default-size lump size))

(defmaker (->dropout :unkeyword-args (x)))

(defmethod default-size ((lump ->dropout))
  (size (x lump)))

(defmethod print-lump-parts ((lump ->dropout) stream)
  (when (dropout lump)
    (format stream " ~S ~,2F" :dropout (dropout lump))))

(defmethod non-constant-mats ((lump ->dropout))
  (if (mask lump)
      (cons (mask lump) (call-next-method))
      (call-next-method)))

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
  (if (use-cuda-p x mask)
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


(defsection @mgl-bp-inputs (:title "Inputs")
  (@mgl-bp-input-lump section)
  (@mgl-bp-embedding-lump section))


(defsection @mgl-bp-input-lump (:title "Input Lump")
  (->input class)
  (dropout (accessor ->input)))

(defclass-now ->input (->dropout)
  ((dropout :initform nil :accessor dropout
            :documentation "See [DROPOUT][(ACCESSOR ->DROPOUT)]."))
  (:documentation "A lump that has no input lumps, does not change its
  values in the forward pass (except when [DROPOUT][(ACCESSOR
  ->INPUT)] is non-zero), and does not compute derivatives. _Clamp_
  inputs on NODES of input lumps in SET-INPUT.

  For convenience, ->INPUT can perform dropout itself although it
  defaults to no dropout.

  ```cl-transcript
  (->input :size 10 :name 'some-input)
  ==> #<->INPUT SOME-INPUT :SIZE 10 1/1 :NORM 0.00000>
  ```"))

(defmethod initialize-instance :before ((lump ->input) &key &allow-other-keys)
  (setf (slot-value lump 'x) lump))

(defmaker (->input))

(defmethod forward ((lump ->input))
  ;; Let dropout do its thing.
  (call-next-method))

;;; Do nothing. In particular, prevent the method for ->DROPOUT from
;;; being called.
(defmethod backward ((lump ->input)))


(defsection @mgl-bp-activations (:title "Activations")
  (@mgl-bp-activation-subnet section)
  (@mgl-bp-batch-normalization section))


(defsection @mgl-bp-activation-subnet (:title "Activation Subnet")
  "So we have some inputs. Usually the next step is to multiply the
  input vector with a weight matrix and add biases. This can be done
  directly with ->+, ->V*M and ->WEIGHT, but it's more convenient to
  use activation subnets to reduce the clutter."
  (->activation class)
  (->activation function))

(defclass ->activation (bpn)
  ()
  (:documentation "Activation subnetworks are built by the function
  ->ACTIVATION and they have a number of lumps hidden inside them.
  Ultimately, this subnetwork computes a sum like `sum_i x_i * W_i +
  sum_j y_j .* V_j + biases` where `x_i` are input lumps, `W_i` are
  dense matrices representing connections, while `V_j` are peephole
  connection vectors that are mulitplied in an elementwise manner with
  their corresponding input `y_j`."))

(defun ->activation (inputs &key (name (gensym)) size peepholes (add-bias-p t))
  "Create a subnetwork of class ->ACTIVATION that computes the over
  activation from dense connection from lumps in INPUTS, and
  elementwise connection from lumps in PEEPHOLES. Create new ->WEIGHT
  lumps as necessary. INPUTS and PEEPHOLES can be a single lump or a
  list of lumps. Finally, if ADD-BIAS-P, then add an elementwise bias
  too. SIZE must be specified explicitly, because it is not possible
  to determine it unless there are peephole connections.

  ```cl-transcript
  (->activation (->input :size 10 :name 'input) :name 'h1 :size 4)
  ==> #<->ACTIVATION (H1 :ACTIVATION) :STRIPES 1/1 :CLUMPS 4>
  ```

  This is the basic workhorse of neural networks which takes care of
  the linear transformation whose results and then fed to some
  non-linearity (->SIGMOID, ->TANH, etc).

  The name of the subnetwork clump is `(,NAME :ACTIVATION)`. The bias
  weight lump (if any) is named `(:BIAS ,NAME)`. Dense connection
  weight lumps are named are named after the input and NAME: `(,(NAME
  INPUT) ,NAME)`, while peepholes weight lumps are named `(,(NAME
  INPUT) ,NAME :PEEPHOLE)`. This is useful to know if, for example,
  they are to be initialized differently."
  (check-type size index)
  (let ((inputs (alexandria:ensure-list inputs)))
    (when (or add-bias-p inputs peepholes)
      (build-fnn (:name (list name :activation) :class '->activation)
        ;; To save memory, which is especially critical in a long RNN,
        ;; we make ->V*M and ->* below use the NODES and DERIVATIVES of
        ;; this ->+ lump. In the forward pass they add their results to
        ;; the shared nodes (instead of setting it) and in the backward
        ;; pass all args of ->+ have the same derivative so it works out
        ;; fine.
        (shared-with-clump
         (->+ (if add-bias-p
                  (list (->weight :name (list :bias name) :size size))
                  ())
              :name (list :sum name)
              :size size))
        (ignored
         (progn
           (dolist (input inputs)
             (let* ((name (list (name input) name))
                    (w (->weight :name name :size (* size (size input)))))
               (->v*m input w :name (list name :activation)
                     :shared-with-clump shared-with-clump)))
           (dolist (peephole peepholes)
             (let* ((name (list (name peephole) name :peephole))
                    (w (->weight :name name :size size)))
               (assert (= size (size peephole)) ()
                       "Size of peephole input lump ~S is not ~S."
                       peephole size)
               (->* peephole w :name (list name :activation)
                    :shared-with-clump shared-with-clump)))))))))


(defsection @mgl-bp-batch-normalization (:title "Batch-Normalization")
  "Batch normalization is special in that it has state apart from the
  computed results (NODES) and its derivatives (DERIVATIVES). This
  state is the estimated mean and variance of its inputs and they are
  encapsulated by ->BATCH-NORMALIZATION.

  The actual work is performed by the ->BATCH-NORMALIZED lump that has
  a ->BATCH-NORMALIZATION as its BATCH-NORMALIZATION. The reason for
  this split of responsability is to allow multiple batch
  normalization operations to share the same state.

  We are going to discuss the concepts from the ground up, but feel
  free to skip ahead to the ->BATCH-NORMALIZED-ACTIVATION utility
  function that covers most of the practical use cases."
  (->batch-normalization class)
  (scale (reader ->batch-normalization))
  (shift (reader ->batch-normalization))
  (batch-size (reader ->batch-normalization))
  (variance-adjustment (reader ->batch-normalization))
  (population-decay (reader ->batch-normalization))
  "Now let's move on to how batch normalization is actually
  performed."
  (->batch-normalized class)
  (batch-normalization (reader ->batch-normalized))
  (->batch-normalized-activation function))

(defclass-now ->batch-normalization (->weight)
  ((scale
    :initarg :scale :reader scale
    :documentation "A weight lump of the same size as SHIFT. This is
    $\\gamma$ in the paper.")
   (shift
    :initarg :shift :reader shift
    :documentation "A weight lump of the same size as SCALE. This is
    $\\beta$ in the paper.")
   ;; A list of means of (SIZE X), one for each subbatch, of which
   ;; there are (/ N-STRIPES BATCH-SIZE).
   (batch-mean :initform nil :accessor batch-mean)
   (batch-stddev :initform nil :accessor batch-stddev)
   (batch-size
    :initform nil :initarg :batch-size :reader batch-size
    :documentation "Normally all stripes participate in the batch.
    Lowering the number of stripes may increase the regularization
    effect, but it also makes the computation less efficient. By
    setting BATCH-SIZE to a divisor of N-STRIPES one can decouple the
    concern of efficiency from that of regularization. The default
    value, NIL, is equivalent to N-STRIPES. BATCH-SIZE only affects
    training.

    With the special value :USE-POPULATION, instead of the mean and
    the stddev of the current batch, use the population statistics for
    normalization. This effectively cancels the regularization effect,
    leaving only the faster learning.")
   (variance-adjustment
    :initform 1e-4 :reader variance-adjustment
    :documentation "A small positive real number that's added to the
    sample stddev. This is $\\epsilon$ in the paper.")
   (population-mean :initform nil :accessor population-mean)
   (population-stddev :initform nil :accessor population-stddev)
   (population-decay
    :initform 0.99 :initarg :population-decay :reader population-decay
    :documentation "While training, an exponential moving average of
    batch means and standard deviances (termed _population
    statistics_) is updated. When making predictions, normalization is
    performed using these statistics. These population statistics are
    persisted by SAVE-STATE.")
   (n-steps :initform 0 :accessor n-steps))
  (:documentation "The primary purpose of this class is to hold the
  estimated mean and variance of the inputs to be normalized and allow
  them to be shared between multiple ->BATCH-NORMALIZED lumps that
  carry out the computation. These estimations are saved and loaded by
  SAVE-STATE and LOAD-STATE.

  ```commonlisp
  (->batch-normalization
   (->weight :name '(h1 :scale) :size 10)
   (->weight :name '(h1 :shift) :size 10)
   :name '(h1 :batch-normalization))
  ```"))

(defmaker (->batch-normalization
           :unkeyword-args (scale shift)
           :make-instance-args args)
  (maybe-copy-weight '->batch-normalization (list* :size (size scale) args)))

(defmethod default-size ((lump ->batch-normalization))
  (size (scale lump)))

(defun ensure-batch-mean (lump subbatch-index)
  (let ((length (length (batch-mean lump))))
    (if (< subbatch-index length)
        (setf (elt (batch-mean lump) subbatch-index)
              (adjust! (elt (batch-mean lump) subbatch-index)
                       (list 1 (size lump)) 0))
        (setf (slot-value lump 'batch-mean)
              (append1 (batch-mean lump) (make-mat (list 1 (size lump)))))))
  (elt (batch-mean lump) subbatch-index))

(defun ensure-batch-stddev (lump subbatch-index)
  (let ((length (length (batch-stddev lump))))
    (if (< subbatch-index length)
        (setf (elt (batch-stddev lump) subbatch-index)
              (adjust! (elt (batch-stddev lump) subbatch-index)
                       (list 1 (size lump)) 0))
        (setf (slot-value lump 'batch-stddev)
              (append1 (batch-stddev lump) (make-mat (list 1 (size lump)))))))
  (elt (batch-stddev lump) subbatch-index))

(defun ensure-population-mean (lump)
  (setf (slot-value lump 'population-mean)
        (if (population-mean lump)
            (adjust! (population-mean lump) (list 1 (size lump)) 0)
            (make-mat (list 1 (size lump)))))
  (population-mean lump))

(defun ensure-population-stddev (lump)
  (setf (slot-value lump 'population-stddev)
        (if (population-stddev lump)
            (adjust! (population-stddev lump) (list 1 (size lump)) 0)
            (make-mat (list 1 (size lump))
                      :initial-element (variance-adjustment lump))))
  (population-stddev lump))

(defmethod write-state* ((lump ->batch-normalization) stream context)
  (write-mat (ensure-population-mean lump) stream)
  (write-mat (ensure-population-stddev lump) stream))

(defmethod read-state* ((lump ->batch-normalization) stream context)
  (read-mat (ensure-population-mean lump) stream)
  (read-mat (ensure-population-stddev lump) stream))

(defclass-now ->batch-normalized (lump)
  ((x :initarg :x :reader x)
   (normalization
    :initarg :normalization :reader batch-normalization
    :documentation "The ->BATCH-NORMALIZATION of this lump. May be
    shared between multiple ->BATCH-NORMALIZED lumps."))
  (:documentation "This is an implementation of v1 of the [Batch
  Normalization paper](http://arxiv.org/abs/1502.03167). The output of
  ->BATCH-NORMALIZED is its input normalized so that for all elements
  the mean across stripes is zero and the variance is 1. That is, the
  mean of the batch is subtracted from the inputs and they are
  rescaled by their sample stddev. Actually, after the normalization
  step the values are rescaled and shifted (but this time with learnt
  parameters) in order to keep the representational power of the model
  the same. The primary purpose of this lump is to speed up learning,
  but it also acts as a regularizer. See the paper for the details.

  The primary input of ->BATCH-NORMALIZED is often an ->ACTIVATION and
  its output is fed into an activation function (see
  @MGL-BP-ACTIVATION-FUNCTIONS)."))

(defmethod initialize-instance :after ((lump ->batch-normalized)
                                       &key size &allow-other-keys)
  (check-size-and-default-size lump size))

(defmaker (->batch-normalized :unkeyword-args (x)))

(defmethod print-lump-parts ((lump ->batch-normalized) stream)
  (format stream " ~S ~S" :batch-size (batch-size (batch-normalization lump))))

(defmethod default-size ((lump ->batch-normalized))
  (size (x lump)))

(defmethod forward ((lump ->batch-normalized))
  (let* ((state (batch-normalization lump))
         (x (x lump))
         (population-mean (ensure-population-mean state))
         (population-stddev (ensure-population-stddev state))
         (n-stripes (n-stripes x))
         (batch-size (if (member (batch-size state) '(nil :use-population))
                         n-stripes
                         (batch-size state)))
         (use-population-p (eq (batch-size state) :use-population))
         (nx (nodes x))
         (nl (nodes lump))
         (size (size lump))
         (dimensions (list batch-size size))
         (decay (population-decay state)))
    (assert (= size (size x)))
    (when *in-training-p*
      (assert (zerop (mod n-stripes batch-size)) ()
              "BATCH-SIZE ~S is not a divisor of N-STRIPES ~S." batch-size
              n-stripes))
    (flet ((foo (batch-mean batch-stddev)
             (with-ones (ones (list batch-size 1))
               ;; Calculate BATCH-MEAN and BATCH-STDDEV, update
               ;; POPULATION-MEAN and POPULATION-STDDEV.
               (when *in-training-p*
                 (incf (n-steps state))
                 (sum! nx batch-mean :axis 0)
                 (scal! (/ batch-size) batch-mean)
                 (with-thread-cached-mat (diffs dimensions)
                   ;; Subtract BATCH-MEAN from each row of X and place
                   ;; the result into DIFFS.
                   (copy! nx diffs)
                   (gemm! -1 ones batch-mean 1 diffs)
                   (.square! diffs)
                   (sum! diffs batch-stddev :axis 0)
                   (scal! (/ batch-size) batch-stddev)
                   (.+! (variance-adjustment state) batch-stddev)
                   (.sqrt! batch-stddev))
                 ;; update population statistics
                 (scal! decay population-mean)
                 (axpy! (- 1 decay) batch-mean population-mean)
                 (scal! decay population-stddev)
                 (axpy! (- 1 decay) batch-stddev population-stddev))
               ;; calculate the output
               (multiple-value-bind (mean stddev)
                   (if (and *in-training-p*
                            (or (not use-population-p)
                                ;; KLUDGE: let the population
                                ;; statistics settle
                                (< (n-steps state) (log 0.1 decay))))
                       (values batch-mean batch-stddev)
                       (values population-mean population-stddev))
                 ;; FIXOPT: when training this has been done above
                 (copy! nx nl)
                 (gemm! -1 ones mean 1 nl)
                 (with-thread-cached-mat (stddevs dimensions)
                   (gemm! 1 ones stddev 0 stddevs)
                   (.inv! stddevs)
                   (.*! stddevs nl)))
               (scale-columns! (nodes (scale state)) nl)
               (gemm! 1 ones (nodes (shift state)) 1 nl))))
      (let ((batch-dimensions (list batch-size size)))
        (with-shape-and-displacement (nx)
          (with-shape-and-displacement (nl)
            (loop for stripe-start upfrom 0 below n-stripes by batch-size
                  for subbatch-index upfrom 0
                  do (let ((start (* stripe-start size)))
                       (reshape-and-displace! nx batch-dimensions start)
                       (reshape-and-displace! nl batch-dimensions start)
                       (foo (ensure-batch-mean state subbatch-index)
                            (ensure-batch-stddev state subbatch-index))))))))))

(defmethod backward ((lump ->batch-normalized))
  (let* ((state (batch-normalization lump))
         (x (x lump))
         (nx (nodes x))
         (dl/dy (derivatives lump))
         (dl/dx (derivatives x))
         (gamma (nodes (scale state)))
         (dl/dgamma (derivatives (scale state)))
         (dl/dbeta (derivatives (shift state)))
         (n-stripes (n-stripes lump))
         (batch-size (if (member (batch-size state) '(nil :use-population))
                         n-stripes
                         (batch-size state)))
         (use-population-p (eq (batch-size state) :use-population))
         (size (size lump))
         (dimensions (list batch-size size))
         (row-dimensions (list 1 size))
         (decay (population-decay state)))
    (assert (= size (size x)))
    (assert (zerop (mod n-stripes batch-size)) ()
            "BATCH-SIZE ~S is not a divisor of N-STRIPES ~S." batch-size
            n-stripes)
    (flet ((foo (batch-mean batch-stddev)
             (with-ones (ones (list batch-size 1))
               (with-thread-cached-mats
                   ((x-mu* dimensions)
                    (1/stddev* dimensions :place :scratch-1)
                    (dl/dx^.*1/stddev dimensions :place :scratch-2)
                    (dl/dmu row-dimensions :place :scratch-3)
                    (dl/dvar row-dimensions :place :scratch-4))
                 ;; x-mu* = x - mu*
                 (copy! nx x-mu*)
                 (gemm! -1 ones batch-mean 1 x-mu*)
                 ;; 1/stddev* = 1 ./ stddev
                 (gemm! 1 ones batch-stddev 0 1/stddev*)
                 (.inv! 1/stddev*)
                 ;; dl/dx^.*1/stddev = dl/dy .* gamma* .* 1/stddev*
                 (geem! 1 dl/dy 1/stddev* 0 dl/dx^.*1/stddev)
                 (geerv! 1 dl/dx^.*1/stddev gamma 0 dl/dx^.*1/stddev)
                 ;; dl/dmu = -sum(dl/dx^ .* 1/stddev*) (not complete, see
                 ;; below)
                 (sum! dl/dx^.*1/stddev dl/dmu :axis 0 :alpha -1)
                 ;; dl/dx += dl/dx^ .* 1/stddev*
                 (axpy! 1 dl/dx^.*1/stddev dl/dx)
                 ;; dl/dgamma = sum(dl/dy .* x^)
                 (with-thread-cached-mat (dl/dy.*x^ dimensions
                                                    :place :scratch-5)
                   (geem! 1 x-mu* 1/stddev* 0 dl/dy.*x^)
                   (geem! 1 dl/dy dl/dy.*x^ 0 dl/dy.*x^)
                   (sum! dl/dy.*x^ dl/dgamma :axis 0 :beta 1))
                 (sum! dl/dy dl/dbeta :axis 0 :beta 1)
                 ;; we are done with 1/stddev*, it can be destroyed
                 (.expt! 1/stddev* 3)
                 ;; make 1/stddev* hold (x - mu*) .* (1/stddev*)^3
                 (.*! x-mu* 1/stddev*)
                 ;; make 1/stddev* hold dl/dy .* (x - mu*) .* (1/stddev*)^3
                 (.*! dl/dy 1/stddev*)
                 ;; dl/dvar = sum(-1/2 * gamma* .* dl/dy .* (x - mu*)
                 ;; .* (1/stddev*)^3)
                 (sum! 1/stddev* dl/dvar :axis 0 :alpha -0.5)
                 (.*! gamma dl/dvar)
                 ;; now that dl/dvar is computed, let's add the parts
                 ;; that depend on it to dl/dmu and dl/dx
                 ;;
                 ;; dl/dmu += -2/m * dl/dvar .* sum(x - mu*)
                 ;; (continued from above)
                 (with-thread-cached-mat (sum-x-mu* row-dimensions
                                                    :place :scratch-5)
                   (sum! x-mu* sum-x-mu* :axis 0)
                   (geem! (/ -2 batch-size) dl/dvar sum-x-mu* 1 dl/dmu))
                 ;; dl/dx += dl/dvar .* 2(x-mu*)/m
                 (geerv! (/ 2 batch-size) x-mu* dl/dvar 1 dl/dx)
                 ;; dl/dx += dl/dmu* * 1/m
                 (gemm! (/ batch-size) ones dl/dmu 1 dl/dx)))))
      (let ((batch-dimensions (list batch-size size)))
        (with-shape-and-displacement (nx)
          (with-shape-and-displacement (dl/dy)
            (with-shape-and-displacement (dl/dx)
              (loop for stripe-start upfrom 0 below n-stripes by batch-size
                    for subbatch-index upfrom 0
                    do (let ((start (* stripe-start size)))
                         (reshape-and-displace! nx batch-dimensions start)
                         (reshape-and-displace! dl/dy batch-dimensions start)
                         (reshape-and-displace! dl/dx batch-dimensions start)
                         (if (and *in-training-p*
                                  (or (not use-population-p)
                                      (< (n-steps state) (log 0.1 decay))))
                             (foo (ensure-batch-mean state subbatch-index)
                                  (ensure-batch-stddev
                                   state subbatch-index))
                             (foo (ensure-population-mean state)
                                  (ensure-population-stddev state))))))))))))

(defun ->batch-normalized-activation (inputs &key (name (gensym)) size
                                      peepholes (batch-size :use-population))
  "Creates and wraps an ->ACTIVATION in ->BATCH-NORMALIZED and with
  its BATCH-NORMALIZATION the two weight lumps for the scale and shift
  parameters. `(->BATCH-NORMALIZED-ACTIVATION INPUTS :NAME 'H1 :SIZE
  10)` is equivalent to:

  ```commonlisp
  (->batch-normalized (->activation inputs :name 'h1 :size 10 :add-bias-p nil)
                      :normalization (->batch-normalization
                                      (->weight :name '(h1 :scale) :size 10)
                                      (->weight :name '(h1 :shift) :size 10)
                                      :name '(h1 :batch-normalization))
                      :name '(h1 :batch-normalized-activation))
  ```

  Note how biases are turned off since normalization will cancel them
  anyway (but a shift is added which amounts to the same effect)."
  (->batch-normalized
   (->activation inputs :name name :size size
                 :peepholes peepholes :add-bias-p nil)
   :normalization (->batch-normalization
                   (->weight :name `(,name :scale) :size size)
                   (->weight :name `(,name :shift) :size size)
                   :size size :batch-size batch-size
                   :name `(,name :batch-normalization))
   :name `(,name :batch-normalized-activation)))


(defsection @mgl-bp-embedding-lump (:title "Embedding Lump")
  "This lump is like an input and a simple activation molded together
  in the name of efficiency."
  (->embedding class)
  (weights (reader ->embedding))
  (input-row-indices (reader ->embedding)))

(defclass-now ->embedding (lump)
  ((weights
    :initarg :weights :reader weights
    :documentation "A weight lump whose rows indexed by
    INPUT-ROW-INDICES are copied to the output of this lump.")
   (input-row-indices
    :initarg :input-row-indices :accessor input-row-indices
    :documentation "A sequence of batch size length of row indices. To
    be set in SET-INPUT."))
  (:documentation "Select rows of WEIGHTS, one row for each index in
  INPUT-ROW-INDICES. This lump is equivalent to adding an ->INPUT lump
  with a one hot encoding scheme and a ->V*M lump on top of it, but it
  is more efficient in execution and in memory usage, because it works
  with a sparse representation of the input.

  The SIZE of this lump is the number of columns of WEIGHTS which is
  determined automatically.

  ```cl-transcript
  (->embedding :weights (->weight :name 'embedding-weights
                                  :dimensions '(3 5))
               :name 'embeddings)
  ==> #<->EMBEDDING EMBEDDINGS :SIZE 5 1/1 :NORM 0.00000>
  ```"))

(defmethod initialize-instance :after ((lump ->embedding)
                                       &key size &allow-other-keys)
  (check-size-and-default-size lump size))

(defmaker (->embedding))

(defmethod default-size ((lump ->embedding))
  (mat-dimension (nodes (weights lump)) 1))

(defmethod forward ((lump ->embedding))
  (let* ((weights (nodes (weights lump)))
         (input-row-indices (input-row-indices lump))
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
           input-row-indices))))

(defmethod backward ((lump ->embedding))
  (let* ((wd (derivatives (weights lump)))
         (input-row-indices (input-row-indices lump))
         (ld (derivatives lump)))
    (let ((stripe 0))
      (map nil (lambda (row)
                 (when row
                   (with-shape-and-displacement (wd)
                     (with-shape-and-displacement (ld)
                       (axpy! 1 (reshape-to-row-matrix! ld stripe)
                              (reshape-to-row-matrix! wd row)))))
                 (incf stripe))
           input-row-indices))))


(defsection @mgl-bp-activation-functions (:title "Activation Functions")
  "Now we are moving on to the most important non-linearities to which
  activations are fed."
  (@mgl-bp-sigmoid-lump section)
  (@mgl-bp-tanh-lump section)
  (@mgl-bp-scaled-tanh-lump section)
  (@mgl-bp-relu-lump section)
  (@mgl-bp-max-lump section)
  (@mgl-bp-min-lump section)
  (@mgl-bp-max-channel-lump section))


(defsection @mgl-bp-sigmoid-lump (:title "Sigmoid Lump")
  (->sigmoid class)
  (dropout (accessor ->sigmoid)))

(defclass-now ->sigmoid (->dropout lump)
  ((dropout :initform nil :accessor dropout
            :documentation "See [DROPOUT][(ACCESSOR ->DROPOUT)]."))
  (:documentation "Applies the `1/(1 + e^{-x})` function elementwise
  to its inputs. This is one of the classic non-linearities for neural
  networks.

  For convenience, ->SIGMOID can perform dropout itself although it
  defaults to no dropout.

  ```cl-transcript
  (->sigmoid (->activation (->input :size 10) :size 5) :name 'this)
  ==> #<->SIGMOID THIS :SIZE 5 1/1 :NORM 0.00000>
  ```

  The SIZE of this lump is the size of its input which is determined
  automatically."))

(defmaker (->sigmoid :unkeyword-args (x)))

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

(defun sigmoid! (x y)
  (let ((n (mat-size x)))
    (assert (= n (mat-size y)))
    (if (use-cuda-p x)
        (multiple-value-bind (block-dim grid-dim) (choose-1d-block-and-grid n 4)
          (cuda-sigmoid! x n y :grid-dim grid-dim :block-dim block-dim))
        (lisp-sigmoid! x (mat-displacement x) n y (mat-displacement y)))))

(define-lisp-kernel (lisp-sigmoid!)
    ((x :mat :input) (start-x index) (n index) (y :mat :output) (start-y index))
  (loop for xi of-type index upfrom start-x
          below (the! index (+ start-x n))
        for yi of-type index upfrom start-y
        do (setf (aref y yi)
                 (let ((xe (aref x xi)))
                   (/ (1+ (with-zero-on-underflow (xe) (exp (- xe)))))))))

(define-cuda-kernel (cuda-sigmoid!)
    (void ((x :mat :input) (n int) (y :mat :output)))
  (let ((stride (* block-dim-x grid-dim-x)))
    (do ((i (+ (* block-dim-x block-idx-x) thread-idx-x)
            (+ i stride)))
        ((>= i n))
      (let ((e (aref x i)))
        (set (aref y i) (/ 1.0 (+ 1.0 (exp (- e)))))))))

(defmethod backward ((lump ->sigmoid))
  (let ((x (x lump)))
    (assert (= (size lump) (size x)))
    (sigmoid-derivative! (nodes lump) (derivatives lump) (derivatives x))))

(defun sigmoid-derivative! (l ld xd)
  (let ((n (mat-size l)))
    (assert (= n (mat-size ld)))
    (assert (= n (mat-size xd)))
    (if (use-cuda-p l ld xd)
        (multiple-value-bind (block-dim grid-dim) (choose-1d-block-and-grid n 4)
          (cuda-sigmoid-derivative! l n ld xd :grid-dim grid-dim
                                    :block-dim block-dim))
        (lisp-sigmoid-derivative! l (mat-displacement l) n
                                  ld (mat-displacement ld)
                                  xd (mat-displacement xd)))))

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


(defsection @mgl-bp-tanh-lump (:title "Tanh Lump")
  (->tanh class))

(defclass-now ->tanh (lump)
  ((x :initarg :x :reader x))
  (:documentation "Applies the TANH function to its input in an
  elementwise manner. The SIZE of this lump is the size of its input
  which is determined automatically."))

(defmethod initialize-instance :after ((lump ->tanh)
                                       &key size &allow-other-keys)
  (check-size-and-default-size lump size))

(defmaker (->tanh :unkeyword-args (x)))

(defmethod default-size ((lump ->tanh))
  (size (x lump)))

(defmethod forward ((lump ->tanh))
  (let ((x (x lump)))
    (assert (= (size lump) (size x)))
    (tanh! (nodes x) (nodes lump))))

(defun tanh! (x y)
  (let ((n (mat-size x)))
    (assert (= n (mat-size y)))
    (if (use-cuda-p x y)
        (multiple-value-bind (block-dim grid-dim) (choose-1d-block-and-grid n 4)
          (cuda-tanh! x n y :grid-dim grid-dim :block-dim block-dim))
        (lisp-tanh! x (mat-displacement x) n y (mat-displacement y)))))

(define-lisp-kernel (lisp-tanh!)
    ((x :mat :input) (start-x index) (n index) (y :mat :output) (start-y index))
  (loop for xi of-type index upfrom start-x
          below (the! index (+ start-x n))
        for yi of-type index upfrom start-y
        do (let ((xe (aref x xi)))
             (setf (aref y yi) (tanh xe)))))

(define-cuda-kernel (cuda-tanh!)
    (void ((x :mat :input) (n int) (y :mat :output)))
  (let ((stride (* block-dim-x grid-dim-x)))
    (do ((i (+ (* block-dim-x block-idx-x) thread-idx-x)
            (+ i stride)))
        ((>= i n))
      (let ((xe (aref x i)))
        (set (aref y i) (tanh xe))))))

(defmethod backward ((lump ->tanh))
  (let ((x (x lump)))
    (assert (= (size lump) (size x)))
    (tanh-derivative! (nodes x) (derivatives lump) (derivatives x))))

(defun tanh-derivative! (x ld xd)
  (let ((n (mat-size x)))
    (assert (= n (mat-size ld)))
    (assert (= n (mat-size xd)))
    (if (use-cuda-p x ld xd)
        (multiple-value-bind (block-dim grid-dim) (choose-1d-block-and-grid n 4)
          (cuda-tanh-derivative! x n ld xd :grid-dim grid-dim
                                 :block-dim block-dim))
        (lisp-tanh-derivative! x (mat-displacement x) n
                               ld (mat-displacement ld)
                               xd (mat-displacement xd)))))

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


(defsection @mgl-bp-scaled-tanh-lump (:title "Scaled Tanh Lump")
  (->scaled-tanh class))

(defclass-now ->scaled-tanh (lump)
  ((x :initarg :x :reader x))
  (:documentation "Pretty much like TANH but its input and output is
  scaled in such a way that the variance of its output is close to 1
  if the variance of its input is close to 1 which is a nice property
  to combat vanishing gradients. The actual function is `1.7159 *
  tanh(2/3 * x)`. The SIZE of this lump is the size of its input which
  is determined automatically."))

(defmethod initialize-instance :after ((lump ->scaled-tanh)
                                       &key size &allow-other-keys)
  (check-size-and-default-size lump size))

(defmaker (->scaled-tanh :unkeyword-args (x)))

(defmethod default-size ((lump ->scaled-tanh))
  (size (x lump)))

(defmethod forward ((lump ->scaled-tanh))
  (let ((x (x lump)))
    (assert (= (size lump) (size x)))
    (scaled-tanh! (nodes x) (nodes lump))))

(defun scaled-tanh! (x y)
  (let ((n (mat-size x)))
    (assert (= n (mat-size y)))
    (if (use-cuda-p x y)
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
    (if (use-cuda-p x ld xd)
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


(defsection @mgl-bp-relu-lump (:title "Relu Lump")
  "We are somewhere around year 2007 by now."
  (->relu class))

(defclass-now ->relu (lump)
  ((x :initarg :x :reader x))
  (:documentation "`max(0,x)` activation function. Be careful, relu
  units can get stuck in the off state: if they move to far to
  negative territory it can be very difficult to get out of it. The
  SIZE of this lump is the size of its input which is determined
  automatically."))

(defmethod initialize-instance :after ((lump ->relu)
                                       &key size &allow-other-keys)
  (check-size-and-default-size lump size))

(defmaker (->relu :unkeyword-args (x)))

(defmethod default-size ((lump ->relu))
  (size (x lump)))

(defmethod forward ((lump ->relu))
  (let ((x (x lump)))
    (assert (= (size lump) (size x)))
    (rectify! (nodes x) (nodes lump))))

(defun rectify! (x y &key (n (mat-size x)))
  (assert (eq (mat-ctype x) (mat-ctype y)))
  (assert (<= n (mat-size x)))
  (assert (<= n (mat-size y)))
  (if (use-cuda-p x y)
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

(defmethod backward ((lump ->relu))
  (let* ((x (x lump))
         (xd (derivatives x))
         (ln (nodes lump))
         (ld (derivatives lump))
         (n (mat-size (nodes lump))))
    (assert (= (size lump) (size x)))
    (if (use-cuda-p xd ln ld)
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


(defsection @mgl-bp-max-lump (:title "Max Lump")
  "We are in about year 2011."
  (->max class)
  (group-size (reader ->max)))

(defclass-now ->max (lump)
  ((x :initarg :x :reader x)
   (group-size
    :initarg :group-size :reader group-size
    :documentation "The number of inputs in each group."))
  (:documentation "This is basically maxout without dropout (see
  http://arxiv.org/abs/1302.4389). It groups its inputs by
  [GROUP-SIZE][(READER ->MAX)], and outputs the maximum of each group.
  The SIZE of the output is automatically calculated, it is the size
  of the input divided by [GROUP-SIZE][(READER ->MAX)].

  ```cl-transcript
  (->max (->input :size 120) :group-size 3 :name 'my-max)
  ==> #<->MAX MY-MAX :SIZE 40 1/1 :NORM 0.00000 :GROUP-SIZE 3>
  ```

  The advantage of ->MAX over ->RELU is that flow gradient is never
  stopped so there is no problem of units getting stuck in off
  state."))

(defmethod initialize-instance :after ((lump ->max) &key size &allow-other-keys)
  (check-size-and-default-size lump size))

(defmaker (->max :unkeyword-args (x)))

(defmethod default-size ((lump ->max))
  (/ (size (x lump)) (group-size lump)))

(defmethod print-lump-parts ((lump ->max) stream)
  (when (/= (size lump) (group-size lump))
    (format stream " ~S ~S" :group-size (group-size lump))))

(defmethod forward ((lump ->max))
  (let* ((x (x lump))
         (group-size (group-size lump))
         (n (mat-size (nodes lump)))
         (nx (nodes x))
         (nl (nodes lump)))
    (if (use-cuda-p nx nl)
        (cuda-max group-size nx n nl
                  :grid-dim (list (ceiling n 256) 1 1)
                  :block-dim (list 256 1 1))
        (lisp-max group-size nx n nl))))

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
         (n (mat-size (nodes lump)))
         (nx (nodes x))
         (nl (nodes lump))
         (dx (derivatives x))
         (dl (derivatives lump)))
    (if (use-cuda-p nx nl dl dx)
        (cuda-max-derivative group-size nx n nl dl dx
                             :grid-dim (list (ceiling n 256) 1 1)
                             :block-dim (list 256 1 1))
        (lisp-max-derivative group-size nx n nl dl dx))))

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


(defsection @mgl-bp-min-lump (:title "Min Lump")
  (->min class)
  (group-size (reader ->min)))

(defclass-now ->min (lump)
  ((x :initarg :x :reader x)
   (group-size
    :initarg :group-size :reader group-size
    :documentation "The number of inputs in each group."))
  (:documentation "Same as ->MAX, but it computes the MIN of groups.
  Rarely useful."))

(defmethod initialize-instance :after ((lump ->min) &key size &allow-other-keys)
  (check-size-and-default-size lump size))

(defmaker (->min :unkeyword-args (x)))

(defmethod default-size ((lump ->min))
  (/ (size (x lump)) (group-size lump)))

(defmethod print-lump-parts ((lump ->min) stream)
  (when (/= (size lump) (group-size lump))
    (format stream " ~S ~S" :group-size (group-size lump))))

(defmethod forward ((lump ->min))
  (let* ((x (x lump))
         (group-size (group-size lump))
         (n (mat-size (nodes lump)))
         (nx (nodes x))
         (nl (nodes lump)))
    (if (use-cuda-p nx nl)
        (cuda-min group-size nx n nl
                  :grid-dim (list (ceiling n 256) 1 1)
                  :block-dim (list 256 1 1))
        (lisp-min group-size nx n nl))))

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
         (n (mat-size (nodes lump)))
         (nx (nodes x))
         (nl (nodes lump))
         (dx (derivatives x))
         (dl (derivatives lump)))
    (if (use-cuda-p nx nl dx dl)
        (cuda-min-derivative group-size nl n nl dl dx
                             :grid-dim (list (ceiling n 256) 1 1)
                             :block-dim (list 256 1 1))
        (lisp-min-derivative group-size nx n nl dl dx))))

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


(defsection @mgl-bp-max-channel-lump (:title "Max-Channel Lump")
  (->max-channel class)
  (group-size (reader ->max-channel)))

(defclass-now ->max-channel (lump)
  ((x :initarg :x :reader x :documentation "Input comes from here.")
   (group-size
    :initarg :group-size :reader group-size
    :documentation "The number of inputs in each group."))
  (:documentation "Called LWTA (Local Winner Take All) or
  Channel-Out (see http://arxiv.org/abs/1312.1909) in the literature
  it is basically ->MAX, but instead of producing one output per
  group, it just produces zeros for all unit but the one with the
  maximum value in the group. This allows the next layer to get some
  information about the path along which information flowed. The SIZE
  of this lump is the size of its input which is determined
  automatically."))

(defmethod initialize-instance :after ((lump ->max-channel)
                                       &key size &allow-other-keys)
  (check-size-and-default-size lump size))

(defmaker (->max-channel :unkeyword-args (x)))

(defmethod default-size ((lump ->max-channel))
  (size (x lump)))

(defmethod print-lump-parts ((lump ->max-channel) stream)
  (when (/= (size lump) (group-size lump))
    (format stream " ~S ~S" :group-size (group-size lump))))

(defmethod forward ((lump ->max-channel))
  (let* ((x (x lump))
         (group-size (group-size lump))
         (nx (nodes x))
         (nl (nodes lump)))
    (declare (type index group-size))
    (if (use-cuda-p nx nl)
        (let ((n (/ (mat-size nl) group-size)))
          (cuda-max-channel group-size nx n nl
                            :grid-dim (list (ceiling n 256) 1 1)
                            :block-dim (list 256 1 1)))
        (lisp-max-channel
         group-size nx (mat-displacement nx) (mat-size nx)
         nl (mat-displacement nl)))))

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
         (group-size (group-size lump))
         (nx (nodes x))
         (dx (derivatives x))
         (dl (derivatives lump)))
    (declare (type index group-size))
    (if (use-cuda-p nx dx dl)
        (let ((n (/ (mat-size (nodes lump)) group-size)))
          (cuda-max-channel-derivative group-size nx n dl dx
                                       :grid-dim (list (ceiling n 256) 1 1)
                                       :block-dim (list 256 1 1)))
        (lisp-max-channel-derivative
         group-size nx (mat-displacement nx) (mat-size nx)
         dl (mat-displacement dl) dx (mat-displacement dx)))))

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

(defsection @mgl-bp-losses (:title "Losses")
  "Ultimately, we need to tell the network what to learn which means
  that the loss function to be minimized needs to be constructed as
  part of the network."
  (@mgl-bp-loss-lump section)
  (@mgl-bp-squared-difference-lump section)
  (@mgl-bp-softmax-xe-loss-lump section))


;;;; This must be defined before ->LOSS that inherits from it.

(defsection @mgl-bp-sum-lump (:title "Sum Lump")
  (->sum class))

(defclass-now ->sum (lump)
  ((x :initarg :x :reader x))
  (:documentation "Computes the sum of all nodes of its input per
  stripe. This SIZE of this lump is always 1."))

(defmethod initialize-instance :after ((lump ->sum)
                                       &key size &allow-other-keys)
  (check-size-and-default-size lump size))

(defmaker (->sum :unkeyword-args (x)))

(defmethod default-size ((lump ->sum))
  1)

(defmethod forward ((lump ->sum))
  (sum! (nodes (x lump)) (nodes lump) :axis 1))

(defmethod backward ((lump ->sum))
  (with-ones (ones (list 1 (size (x lump))))
    (gemm! 1 (derivatives lump) ones 1 (derivatives (x lump)))))


(defsection @mgl-bp-loss-lump (:title "Loss Lump")
  (->loss class)
  (importance (accessor ->loss)))

(defclass-now ->loss (->sum)
  ((importance
    :initform nil
    :initarg :importance
    :accessor importance
    :documentation "This is to support weighted instances. That is
    when not all training instances are equally important. If non-NIL,
    a 1d MAT with the importances of stripes of the batch. When
    IMPORTANCE is given (typically in SET-INPUT), then instead of
    adding 1 to the derivatives of all stripes, IMPORTANCE is added
    elemtwise."))
  (:documentation "Calculate the loss for the instances in the batch.
  The main purpose of this lump is to provide a training signal.

  An error lump is usually a leaf in the graph of lumps (i.e. there
  are no other lumps whose input is this one). The special thing about
  error lumps is that 1 (but see IMPORTANCE) is added automatically to
  their derivatives. Error lumps have exactly one node (per stripe)
  whose value is computed as the sum of nodes in their input lump."))

(defmaker (->loss :unkeyword-args (x)))

(defmethod forward :around ((lump ->loss))
  (call-next-method)
  (when (importance lump)
    (.*! (importance lump) (nodes lump))))

(defmethod backward :around ((lump ->loss))
  (if (importance lump)
      (axpy! 1 (importance lump) (derivatives lump))
      (.+! 1 (derivatives lump)))
  (call-next-method))

(defmethod cost ((lump ->loss))
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


(defsection @mgl-bp-squared-difference-lump (:title "Squared Difference Lump")
  "In regression, the squared error loss is most common. The squared
  error loss can be constructed by combining ->SQUARED-DIFFERENCE with
  a ->LOSS."
  (->squared-difference class))

(defclass-now ->squared-difference (lump)
  ((x :initarg :x :reader x)
   (y :initarg :y :reader y))
  (:documentation "This lump takes two input lumps and calculates
  their squared difference `(x - y)^2` in an elementwise manner. The
  SIZE of this lump is automatically determined from the size of its
  inputs. This lump is often fed into ->LOSS that sums the squared
  differences and makes it part of the function to be minimized.

  ```cl-transcript
  (->loss (->squared-difference (->activation (->input :size 100)
                                              :size 10)
                                (->input :name 'target :size 10))
          :name 'squared-error)
  ==> #<->LOSS SQUARED-ERROR :SIZE 1 1/1 :NORM 0.00000>
  ```

  Currently this lump is not CUDAized, but it will copy data from the
  GPU if it needs to."))

(defmethod initialize-instance :after ((lump ->squared-difference)
                                       &key size &allow-other-keys)
  (check-size-and-default-size lump size))

(defmaker (->squared-difference :unkeyword-args (x y)))

(defmethod default-size ((lump ->squared-difference))
  (size (x lump)))

(defmethod forward ((lump ->squared-difference))
  (let ((x (x lump))
        (y (y lump)))
    (assert (= (size x) (size y)))
    (assert (= (n-stripes lump) (n-stripes x) (n-stripes y)))
    (with-facets ((x* ((nodes x) 'backing-array :direction :input))
                  (y* ((nodes y) 'backing-array :direction :input))
                  (to* ((nodes lump) 'backing-array :direction :output)))
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

(defmethod backward ((lump ->squared-difference))
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


(defsection @mgl-bp-softmax-xe-loss-lump
    (:title "Softmax Cross-Entropy Loss Lump")
  (->softmax-xe-loss class)
  (group-size (reader ->softmax-xe-loss))
  (target (accessor ->softmax-xe-loss))
  (ensure-softmax-target-matrix function))

(defclass-now ->softmax-xe-loss (lump)
  ((x :initarg :x :reader x)
   (group-size
    :initarg :group-size :reader group-size
    :documentation "The number of elements in a softmax group. This is
    the number of classes for classification. Often GROUP-SIZE is
    equal to SIZE (it is the default), but in general the only
    constraint is that SIZE is a multiple of GROUP-SIZE.")
   (target
    :initform nil :initarg :target :accessor target
    :documentation "Set in SET-INPUT, this is either a MAT of the same
    size as the input lump `X` or if the target is very sparse, this
    can also be a sequence of batch size length that contains the
    index value pairs of non-zero entries:

        (;; first instance in batch has to non-zero targets
         (;; class 10 has 30% expected probability
          (10 . 0.3)
          ;; class 2 has 70% expected probability
          (2 .  0.7))
         ;; second instance in batch puts 100% on class 7
         7
         ;; more instance in the batch follow
         ...)

    Actually, in the rare case where [GROUP-SIZE][(reader
    ->softmax-xe-loss)] is not SIZE (i.e. there are several softmax
    normalization groups for every example), the length of the above
    target sequence is BATCH-SIZE * N-GROUPS. Indices are always
    relative to the start of the group.

    If [GROUP-SIZE][(reader ->softmax-xe-loss)] is large (for example,
    in neural language models with a huge number of words), using
    sparse targets can make things go much faster, because calculation
    of the derivative is no longer quadratic.

    Giving different weights to training instances is implicitly
    supported. While target values in a group should sum to 1,
    multiplying all target values with a weight `W` is equivalent to
    training that `W` times on the same example.")
   ;; Make sure SET-MAX-N-STRIPES doesn't create DERIVATIVES. We don't
   ;; use it anyway.
   (derivatives :initform nil))
  (:documentation "A specialized lump that computes the softmax of its
  input in the forward pass and backpropagates a cross-entropy loss.
  The advantage of doing these together is numerical stability. The
  total cross-entropy is the sum of cross-entropies per group of
  [GROUP-SIZE][(reader ->softmax-xe-loss)] elements:

      XE(x) = - sum_{i=1,g} t_i * ln(s_i)

  where `g` is the number of classes ([GROUP-SIZE][(reader
  ->softmax-xe-loss)]), `t_i` are the targets (i.e. the true
  probabilities of the class, often all zero but one), `s_i` is the
  output of softmax calculated from input `X`:

      s_i = softmax{x_1, x_2, ..., x_g} = e^x_i / (sum_{j=1,g} e^x_j)

  In other words, in the forward phase this lump takes input `X`,
  computes its elementwise EXP, normalizes each group of
  [GROUP-SIZE][(reader ->softmax-xe-loss)] elements to sum to 1 to get
  the softmax which is the result that goes into NODES. In the
  backward phase, there are two sources of gradients: the lumps that
  use the output of this lump as their input (currently not
  implemented and would result in an error) and an implicit
  cross-entropy loss.

  One can get the cross-entropy calculated in the most recent forward
  pass by calling COST on this lump.

  This is the most common loss function for classification. In fact,
  it is nearly ubiquitous. See the @MGL-FNN-TUTORIAL and the
  @MGL-RNN-TUTORIAL for how this loss and SET-INPUT work together."))

(defmaker (->softmax-xe-loss :unkeyword-args (x)))

(defmethod default-size ((lump ->softmax-xe-loss))
  (size (x lump)))

(defmethod initialize-instance :after ((lump ->softmax-xe-loss)
                                       &key &allow-other-keys)
  (unless (slot-boundp lump 'group-size)
    (setf (slot-value lump 'group-size) (size lump))))

(defmethod print-lump-parts ((lump ->softmax-xe-loss) stream)
  (when (/= (size lump) (group-size lump))
    (format stream " ~S ~S" :group-size (group-size lump))))

(defun ensure-softmax-target-matrix (softmax-xe-loss n)
  "Set TARGET of SOFTMAX-XE-LOSS to a MAT capable of holding the dense
  target values for N stripes."
  (setf (target softmax-xe-loss)
        (if (typep (target softmax-xe-loss) 'mat)
            (adjust! (target softmax-xe-loss) (list n (size softmax-xe-loss)) 0)
            (make-mat (list n (size softmax-xe-loss))
                      :max-size (* (max-n-stripes softmax-xe-loss)
                                   (size softmax-xe-loss))))))

(eval-when (:compile-toplevel :load-toplevel :execute)
  (defparameter *n-softmax-threads* 128))

(defmethod forward ((lump ->softmax-xe-loss))
  (let* ((nx (nodes (x lump)))
         (group-size (group-size lump))
         (softmax (nodes lump))
         (n (* (n-stripes lump) (size lump))))
    (if (use-cuda-p nx softmax)
        (cuda-softmax-xe group-size nx n softmax
                         :grid-dim (list (/ n group-size) 1 1)
                         :block-dim (list *n-softmax-threads* 1 1))
        (lisp-softmax-xe group-size nx n softmax))))

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
          (if (use-cuda-p dx softmax)
              (multiple-value-bind (block-dim grid-dim)
                  (choose-1d-block-and-grid group-size 4)
                (cuda-softmax-xe-derivative/sparse
                 group-start group-size
                 dx softmax (+ group-start target-index) target-value
                 :grid-dim grid-dim :block-dim block-dim))
              (lisp-softmax-xe-derivative/sparse
               group-start group-size dx softmax
               (+ group-start target-index) target-value)))
        (if (use-cuda-p dx softmax)
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


(defsection @mgl-bp-stochasticity (:title "Stochasticity")
  (@mgl-bp-dropout-lump section)
  (@mgl-bp-gaussian-random-lump section)
  (@mgl-bp-sample-binary-lump section))


(defsection @mgl-bp-gaussian-random-lump (:title "Gaussian Random Lump")
  (->gaussian-random class)
  (mean (accessor ->gaussian-random))
  (variance (accessor ->gaussian-random))
  (variance-for-prediction (accessor ->gaussian-random)))

(defclass-now ->gaussian-random (lump)
  ((mean
    :type real :initform 0 :initarg :mean :accessor mean
    :documentation "The mean of the normal distribution.")
   (variance
    :type real :initform 1 :initarg :variance :accessor variance
    :documentation "The variance of the normal distribution.")
   (variance-for-prediction
    :initform 0 :initarg :variance-for-prediction
    :accessor variance-for-prediction
    :documentation "If not NIL, then this value overrides VARIANCE
    when not in training (i.e. when making predictions)."))
  (:documentation "This lump has no input, it produces normally
  distributed independent random numbers with MEAN and VARIANCE (or
  VARIANCE-FOR-PREDICTION). This is useful building block for noise
  based regularization methods.

  ```cl-transcript
  (->gaussian-random :size 10 :name 'normal :mean 1 :variance 2)
  ==> #<->GAUSSIAN-RANDOM NORMAL :SIZE 10 1/1 :NORM 0.00000>
  ```"))

(defmaker (->gaussian-random))

(defmethod forward ((lump ->gaussian-random))
  (gaussian-random! (nodes lump) :mean (mean lump)
                    :stddev (sqrt (if *in-training-p*
                                      (variance lump)
                                      (variance-for-prediction lump)))))

(defmethod backward ((lump ->gaussian-random)))


(defsection @mgl-bp-sample-binary-lump (:title "Binary Sampling Lump")
  (->sample-binary class))

(defclass-now ->sample-binary (lump)
  ((x :initarg :x :reader x)
   (randoms :initform nil :reader randoms))
  (:documentation "Treating values of its input as probabilities,
  sample independent binomials. Turn true into 1 and false into 0. The
  SIZE of this lump is determined automatically from the size of its
  input.

  ```cl-transcript
  (->sample-binary (->input :size 10) :name 'binarized-input)
  ==> #<->SAMPLE-BINARY BINARIZED-INPUT :SIZE 10 1/1 :NORM 0.00000>
  ```"))

(defmethod initialize-instance :after ((lump ->sample-binary)
                                       &key size &allow-other-keys)
  (check-size-and-default-size lump size))

(defmaker (->sample-binary :unkeyword-args (x)))

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


(defsection @mgl-bp-arithmetic (:title "Arithmetic")
  (@mgl-bp-sum-lump section)
  (@mgl-bp-v*m-lump section)
  (@mgl-bp-+-lump section)
  (@mgl-bp-*-lump section)
  (@mgl-bp-abs-lump section)
  (@mgl-bp-exp-lump section)
  (@mgl-bp-normalized-lump section))


(defsection @mgl-bp-v*m-lump (:title "Vector-Matrix Multiplication Lump")
  (->v*m class)
  (weights (reader ->v*m))
  (transpose-weights-p (reader ->v*m)))

(defclass-now ->v*m (lump)
  ((x :initarg :x :reader x)
   (weights
    :type ->weight :initarg :weights :reader weights
    :documentation "A ->WEIGHT lump.")
   (transpose-weights-p
    :initform nil :initarg :transpose-weights-p
    :reader transpose-weights-p
    :documentation "Determines whether the input is multiplied by
    WEIGHTS or its transpose."))
  (:documentation "Perform `X * WEIGHTS` where `X` (the input) is of
  size `M` and WEIGHTS is a ->WEIGHT whose single stripe is taken to
  be of dimensions `M x N` stored in row major order. `N` is the size
  of this lump. If TRANSPOSE-WEIGHTS-P then WEIGHTS is `N x M` and `X
  * WEIGHTS'` is computed."))

(defmaker (->v*m :unkeyword-args (x weights)))

(defmethod initialize-instance :after ((lump ->v*m) &key
                                       &allow-other-keys)
  (assert (= (* (size lump) (size (x lump)))
             (size (weights lump))))
  (setf (slot-value (weights lump) 'dimensions)
        (if (transpose-weights-p lump)
            (list (size lump) (size (x lump)))
            (list (size (x lump)) (size lump))))
  ;; force reshaping
  (setf (max-n-stripes (weights lump)) (max-n-stripes (weights lump))))

(defmethod default-size ((lump ->v*m))
  (/ (size (weights lump))
     (size (x lump))))

(defmethod print-lump-parts ((lump ->v*m) stream)
  (when (transpose-weights-p lump)
    (format stream " ~S ~S" :tranpose t)))

(defmethod forward ((lump ->v*m))
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

(defmethod backward ((lump ->v*m))
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


(defsection @mgl-bp-+-lump (:title "Elementwise Addition Lump")
  (->+ class))

(defclass-now ->+ (lump)
  ((args :initarg :args :reader args))
  (:documentation "Performs elementwise addition on its input lumps.
  The SIZE of this lump is automatically determined from the size of
  its inputs if there is at least one. If one of the inputs is a
  ->WEIGHT lump, then it is added to every stripe.

  ```cl-transcript
  (->+ (list (->input :size 10) (->weight :size 10 :name 'bias))
       :name 'plus)
  ==> #<->+ PLUS :SIZE 10 1/1 :NORM 0.00000>
  ```"))

(defmethod initialize-instance :after ((lump ->+) &key size &allow-other-keys)
  (check-size-and-default-size lump size))

(defmaker (->+ :unkeyword-args (args)))

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


(defsection @mgl-bp-*-lump (:title "Elementwise Multiplication Lump")
  (->* class))

(defclass-now ->* (lump)
  ((x :initarg :x :reader x)
   (y :initarg :y :reader y))
  (:documentation "Performs elementwise multiplication on its two
  input lumps. The SIZE of this lump is automatically determined from
  the size of its inputs. Either input can be a ->WEIGHT lump.

  ```cl-transcript
  (->* (->input :size 10) (->weight :size 10 :name 'scale)
       :name 'mult)
  ==> #<->* MULT :SIZE 10 1/1 :NORM 0.00000>
  ```"))

(defmethod initialize-instance :after ((lump ->*) &key size &allow-other-keys)
  (check-size-and-default-size lump size))

(defmaker (->* :unkeyword-args (x y)))

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
                  ;; KLUDGE: reshaping due to
                  ;; REMOVE-TRAILING-NIL-INSTANCES.
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
                    ;; KLUDGE: reshaping due to
                    ;; REMOVE-TRAILING-NIL-INSTANCES.
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


(defsection @mgl-bp-abs-lump (:title "Abs Lump")
  (->abs class))

(defclass-now ->abs (lump)
  ((x :initarg :x :reader x)))

(defmaker (->abs :unkeyword-args (x)))

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


(defsection @mgl-bp-sine-lump (:title "Sine Lump")
  (->sin class))

(defclass-now ->sin (lump)
  ((x :initarg :x :reader x))
  (:documentation "Applies the SIN function to its input in an
  elementwise manner. The SIZE of this lump is the size of its input
  which is determined automatically."))

(defmethod initialize-instance :after ((lump ->sin)
                                       &key size &allow-other-keys)
  (check-size-and-default-size lump size))

(defmaker (->sin :unkeyword-args (x)))

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
    (if (use-cuda-p x y)
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
    (if (use-cuda-p x ld xd)
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


(defsection @mgl-bp-exp-lump (:title "Exp Lump")
  (->exp class))

;;; FIXDOC
(defclass-now ->exp (lump)
  ((x :initarg :x :reader x)))

(defmaker (->exp :unkeyword-args (x)))

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


(defsection @mgl-bp-normalized-lump (:title "Normalized Lump")
  (->normalized class))

;;; FIXDOC
(defclass-now ->normalized (lump)
  ((x :initarg :x :reader x)
   (group-size :initarg :group-size :reader group-size)
   (scale
    :initform 1
    :type (or real array)
    :initarg :scale :accessor scale
    :documentation "The sum of nodes after normalization. Can be
    changed during training, for instance when clamping. If it is a
    vector then its length must be MAX-N-STRIPES which automatically
    maintained.")))

(defmaker (->normalized :unkeyword-args (x)))

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


(defsection @mgl-bp-rnn-operations (:title "Operations for RNNs")
  (@mgl-bp-lstm-subnet section)
  (@mgl-bp-seq-barrier-lump section))


(defsection @mgl-bp-lstm-subnet (:title "LSTM Subnet")
  (->lstm class)
  (->lstm function))

(defclass ->lstm (bpn)
  ()
  (:documentation "Long-Short Term Memory subnetworks are built by the
  function ->LSTM and they have many lumps hidden inside them. These
  lumps are packaged into a subnetwork to reduce clutter."))

(defun ->lstm (inputs &key name cell-init output-init size
               (activation-fn '->activation) (gate-fn '->sigmoid)
               (input-fn '->tanh) (output-fn '->tanh) (peepholes t))
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

  A notable difference from the paper is that in addition to being a
  single lump, `x_t` (INPUTS) can also be a list of lumps. Whenever
  some activation is to be calculated based on `x_t`, it is going to
  be the sum of individual activations. For example, `W_ix * x_t` is
  really `sum_j W_ijx * inputs_j`.

  If CELL-INIT is non-NIL, then it must be a CLUMP of SIZE form which
  stands for the initial state of the value cell (`c_{-1}`). CELL-INIT
  being NIL is equivalent to the state of all zeros.

  ACTIVATION-FN defaults to ->ACTIVATION, but it can be for example
  ->BATCH-NORMALIZED-ACTIVATION. In general, functions like the
  aforementioned two with signature like (INPUTS &KEY NAME SIZE
  PEEPHOLES) can be passed as ACTIVATION-FN."
  (check-type size index)
  (let* ((inputs (alexandria:ensure-list inputs))
         (input-gate-name `(,name :input))
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
         (funcall gate-fn (funcall activation-fn (add (lagged-output) inputs)
                                   :name input-gate-name :size size
                                   :peepholes (when peepholes
                                                (add (lagged-cell) ())))
                  :name input-gate-name))
        ;; f_t = s(W_fx * x_t + W_fm * m_{t_1} + W_fc .* c_{t-1} + b_f)
        (forget-gate
         (funcall gate-fn (funcall activation-fn (add (lagged-output) inputs)
                                   :name forget-gate-name
                                   :size size
                                   :peepholes (when peepholes
                                                (add (lagged-cell) ())))
                  :name forget-gate-name))
        ;; c_t = f_t .* c_{t-1} + i_t .* g(W_cx * x_t + W_cm * m_{t-1} + b_c)
        (cell
         ;; Save memory by sharing.
         (let ((shared-with-clump (->+ () :name cell-name :size size)))
           (when (lagged-cell)
             (->* forget-gate (lagged-cell)
                  :shared-with-clump shared-with-clump))
           (->* input-gate (funcall input-fn
                                    (funcall activation-fn
                                             (add (lagged-output) inputs)
                                             :name cell-name
                                             :size size))
                :shared-with-clump shared-with-clump)
           shared-with-clump))
        ;; o_t = s(W_ox * x_t + W_om * m_{t-1} + W_oc .* c_t + b_o)
        (output-gate
         (funcall gate-fn (funcall activation-fn (add (lagged-output) inputs)
                                   :name output-gate-name :size size
                                   :peepholes (when peepholes
                                                (list cell)))
                  :name output-gate-name ))
        ;; m_t = o_t .* h(c_t)
        (output
         (->* output-gate (funcall output-fn cell) :name output-name))))))


(defsection @mgl-bp-seq-barrier-lump (:title "Sequence Barrier Lump")
  (->seq-barrier class)
  (seq-elt-fn (reader ->seq-barrier))
  (seq-indices (accessor ->seq-barrier)))

(defclass-now ->seq-barrier (lump)
  ((seq-elt-fn
    :initarg :seq-elt-fn :reader seq-elt-fn
    :documentation "A function of an [INDEX][displaced] argument that
    returns the lump with that index in some sequence.")
   (seq-indices
    :accessor seq-indices
    :documentation "A sequence of length batch size of indices. The
    element at index `I` is the index to be passed to SEQ-ELT-FN to
    find the lump whose stripe `I` is copied to stripe `I` of this
    this lump."))
  (:documentation "In an RNN, processing of stripes (instances in the
  batch) may require different number of time step so the final state
  for stripe 0 is in stripe 0 of some lump L at time step 7, while for
  stripe 1 it is in stripe 1 of sump lump L at time step 42.

  This lump copies the per-stripe states from different lumps into a
  single lump so that further processing can take place (typically
  when the RNN is embedded in another network).

  The SIZE of this lump is automatically set to the size of the lump
  returned by `(FUNCALL SEQ-ELT-FN 0)`."))

(defmethod initialize-instance :after ((lump ->seq-barrier)
                                       &key size &allow-other-keys)
  (check-size-and-default-size lump size))

(defmaker (->seq-barrier))

(defmethod default-size ((lump ->seq-barrier))
  (size (funcall (seq-elt-fn lump) 0)))

(defmethod forward ((lump ->seq-barrier))
  ;; For each row of NODES, there is an input sequence of some length.
  ;; Look up the clump at the end of the sequence and copy its
  ;; corresponding row to NODES.
  (let ((nodes (nodes lump))
        (size (size lump))
        (seq-indices (seq-indices lump))
        (seq-elt-fn (seq-elt-fn lump))
        (stripe 0))
    (map-displacements
     (lambda (nodes)
       (let* ((seq-index (pop seq-indices))
              (end-clump (funcall seq-elt-fn seq-index))
              (end-nodes (nodes end-clump)))
         ;; KLUDGE: With *WARP-TIME* the state this barrier looks up
         ;; is destroyed if an input sequence ends but a sequence in
         ;; at a higher numbered stripe does not. The workaround there
         ;; is to sort inputs by their length. In this situation, the
         ;; assertion would fail because at later time steps N-STRIPES
         ;; is set to exclude the already ended sequences _at the
         ;; highest stripes_.
         (unless *warp-time*
           (assert (< stripe (n-stripes end-clump))))
         (with-shape-and-displacement (end-nodes size (mat-displacement nodes))
           (copy! end-nodes nodes)))
       (incf stripe))
     nodes size)))

(defmethod backward ((lump ->seq-barrier))
  (let ((derivatives (derivatives lump))
        (size (size lump))
        (seq-indices (seq-indices lump))
        (seq-elt-fn (seq-elt-fn lump)))
    (map-displacements
     (lambda (derivatives)
       (let* ((seq-index (pop seq-indices))
              (end-clump (funcall seq-elt-fn seq-index))
              (end-derivatives (derivatives end-clump)))
         (with-shape-and-displacement (end-derivatives size
                                       (mat-displacement derivatives))
           (axpy! 1 derivatives end-derivatives))))
     derivatives size)))


(defsection @mgl-bp-utilities (:title "Utilities")
  (renormalize-activations function)
  (arrange-for-renormalizing-activations function))

;;; FIXME: maybe use ->ACTIVATIONS instead?
(defun renormalize-activations (->v*m-lumps l2-upper-bound)
  "If the l2 norm of the incoming weight vector of the a unit is
  larger than L2-UPPER-BOUND then renormalize it to L2-UPPER-BOUND.
  The list of ->V*M-LUMPS is assumed to be eventually fed to the same
  lump.

  To use it, group the activation clumps into the same GD-OPTIMIZER
  and hang this function on AFTER-UPDATE-HOOK, that latter of which is
  done for you ARRANGE-FOR-RENORMALIZING-ACTIVATIONS.

  See \"Improving neural networks by preventing co-adaptation of
  feature detectors (Hinton, 2012)\",
  <http://arxiv.org/pdf/1207.0580.pdf>."
  (when (and ->v*m-lumps l2-upper-bound)
    (renormalize-mats
     (loop for lump in ->v*m-lumps
           collect (let ((weights (etypecase lump
                                    (->v*m (weights lump))
                                    (->weight lump))))
                     (list (nodes weights)
                           (if (and (typep lump '->v*m)
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
  (push (let ((->v*ms nil)
              (firstp t))
          (lambda ()
            (when firstp
              (setq ->v*ms
                    (loop for lump in (segments optimizer)
                          collect (or (find-activation-lump-for-weight lump bpn)
                                      lump)))
              (setq firstp nil))
            (renormalize-activations ->v*ms l2-upper-bound)))
        (after-update-hook optimizer)))

(defun find-activation-lump-for-weight (->weight bpn)
  ;; FIXME: this iteration is broken for nested bpns.
  (loop for lump across (clumps bpn) do
    (when (and (typep lump '->v*m)
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
    (if (use-cuda-p mat)
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
    (if (use-cuda-p mat)
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
