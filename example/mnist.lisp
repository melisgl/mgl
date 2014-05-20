;;;; Code for the MNIST handwritten digit recognition challange.
;;;;
;;;; See papers by Geoffrey Hinton, for the DBN-to-BPN approach:
;;;;
;;;;   To Recognize Shapes, First Learn to Generate Images,
;;;;   http://www.cs.toronto.edu/~hinton/absps/montrealTR.pdf
;;;;
;;;;   http://www.cs.toronto.edu/~hinton/MatlabForSciencePaper.html
;;;;
;;;; and for the DBN-to-DBM-to-BPN one:
;;;;
;;;;   "Deep Boltzmann Machines",
;;;;   http://www.cs.toronto.edu/~hinton/absps/dbm.pdf
;;;;
;;;; Download the four files from http://yann.lecun.com/exdb/mnist and
;;;; gunzip them. Set *MNIST-DATA-DIR* to point to their directory and
;;;; call TRAIN-MNIST/1 or TRAIN-MNIST/2 for DBN-to-BPN and
;;;; DBN-to-DBM-to-BPN approaches, respectively.
;;;;
;;;;
;;;; DBN-to-BPN (see TRAIN-MNIST/1)
;;;;
;;;; A 784-500-500-2000 deep belief network is trained, then 10
;;;; softmax units are attached to the top layer of the
;;;; backpropagation network converted from the top half of the DBN.
;;;; The new, 2000->10 connections in the backprop network are trained
;;;; for a few batches and finally all weights are trained together.
;;;; Takes less than two days to train on a 2.16GHz Core Duo and
;;;; reaches ~98.86% accuracy.
;;;;
;;;; During DBN training the DBN TEST RMSE is the RMSE of the mean
;;;; field reconstruction of the test images while TRAINING RMSE is of
;;;; the stochastic reconstructions (that is, the hidden layer is
;;;; sampled) during training. Naturally, the stochastic one tends to
;;;; be higher but is cheap to compute.
;;;;
;;;; MGL is marginally faster (~10%) training the first RBM than the
;;;; original matlab code when both are using single-threaded BLAS.
;;;; Since 86% percent of the time is spent in BLAS this probably
;;;; means only that the two BLAS implementations perform similarly.
;;;; However, since the matlab code precomputes inputs for the stacked
;;;; RBMs, this code is a doing a lot more on them making it ~32%
;;;; slower on the 2nd. Probably due to the size of last layer, the
;;;; picture changes again in the 3rd rbm: they are roughly on par.
;;;;
;;;; CG training of the BPN is also of the same speed except for the
;;;; first phase where only the softmax weights are trained. Here,
;;;; this code is about 6 times slower due to the input being clamped
;;;; at the usual place and the BPN forwarded which is avoidable in
;;;; this case, but it's not very important as the vast majority of
;;;; time is spent training the whole BPN.
;;;;
;;;;
;;;; DBN-to-DBM-to-BPN (see TRAIN-MNIST/2)
;;;;
;;;; A 784-500-1000 DBM is pretrained as a DBN then the DBM is
;;;; translated to a BPN. It's pretty much the same training process
;;;; as before except that after the DBN the DBM is trained by PCD
;;;; before unrolling that into a BPN. Accuracy is ~99.09% a bit
;;;; higher than the reported 99.05%.

(in-package :mgl-example-mnist)

(defmacro setq* ((symbol special) value)
  `(progn
     (setq ,symbol ,value)
     (when ,special
       (setf (symbol-value ,special) ,symbol))))


(defparameter *mnist-data-dir*
  (merge-pathnames "mnist-data/" *example-dir*)
  "Set this to the directory where the uncompressed mnist files reside.")

(defparameter *mnist-save-dir*
  (merge-pathnames "mnist-save/" *example-dir*)
  "Set this to the directory where the trained models are saved.")

(defstruct image
  (label nil :type (integer 0 10))
  (array nil :type mat))

(defmethod label ((image image))
  (image-label image))


;;;; Reading the files

(defun read-low-endian-ub32 (stream)
  (+ (ash (read-byte stream) 24)
     (ash (read-byte stream) 16)
     (ash (read-byte stream) 8)
     (read-byte stream)))

(defun read-magic-or-lose (stream magic)
  (unless (eql magic (read-low-endian-ub32 stream))
    (error "Bad magic.")))

(defun read-image-labels (stream)
  (read-magic-or-lose stream 2049)
  (let ((n (read-low-endian-ub32 stream)))
    (loop repeat n collect (read-byte stream))))

(defun read-image-array (stream)
  (let ((a (make-array (* 28 28) :element-type 'mgl-util:flt)))
    (loop for i below (* 28 28) do
      (let ((pixel (/ (read-byte stream) #.(flt 255))))
        (unless (<= 0 pixel 1)
          (error "~S" pixel))
        (setf (aref a i) pixel)))
    (array-to-mat a)))

(defun print-image-array (a)
  (with-facets ((a (a 'backing-array :direction :input)))
    (dotimes (y 28)
      (dotimes (x 28)
        (princ (if (< 0.5 (aref a (+ (* y 28) x))) #\O #\.)))
      (terpri))))

(defun read-image-arrays (stream)
  (read-magic-or-lose stream 2051)
  (let ((n (read-low-endian-ub32 stream)))
    (read-magic-or-lose stream 28)
    (read-magic-or-lose stream 28)
    (coerce (loop repeat n collect (read-image-array stream))
            'vector)))

(defun load-training (&optional (mnist-data-dir *mnist-data-dir*))
  (log-msg "Loading training images~%")
  (prog1
      (map 'vector
           (lambda (label array)
             (make-image :label label :array array))
           (with-open-file (s (merge-pathnames "train-labels-idx1-ubyte"
                                               mnist-data-dir)
                            :element-type 'unsigned-byte)
             (read-image-labels s))
           (with-open-file (s (merge-pathnames "train-images-idx3-ubyte"
                                               mnist-data-dir)
                            :element-type 'unsigned-byte)
             (read-image-arrays s)))
    (log-msg "Loading training images done~%")))

(defun load-test (&optional (mnist-data-dir *mnist-data-dir*))
  (log-msg "Loading test images~%")
  (prog1
      (map 'vector
           (lambda (label array)
             (make-image :label label :array array))
           (with-open-file (s (merge-pathnames "t10k-labels-idx1-ubyte"
                                               mnist-data-dir)
                            :element-type 'unsigned-byte)
             (read-image-labels s))
           (with-open-file (s (merge-pathnames "t10k-images-idx3-ubyte"
                                               mnist-data-dir)
                            :element-type 'unsigned-byte)
             (read-image-arrays s)))
    (log-msg "Loading test images done~%")))

(defvar *training-images*)
(defvar *test-images*)

(defun training-images ()
  (unless (boundp '*training-images*)
    (setq *training-images* (load-training)))
  *training-images*)

(defun test-images ()
  (unless (boundp '*test-images*)
    (setq *test-images* (load-test)))
  *test-images*)


;;;; Sampling, clamping, utilities

(defun make-sampler (images &key (n-epochs 1)
                     (max-n (* n-epochs (length images)))
                     discard-label-p sample-visible-p)
  (make-instance 'counting-function-sampler
                 :max-n-samples max-n
                 :sampler (let ((g (make-random-generator images)))
                            (lambda ()
                              (list (funcall g)
                                    :discard-label-p discard-label-p
                                    :sample-visible-p sample-visible-p)))))

(defun make-tiny-sampler (images &key discard-label-p)
  (make-sampler (subseq images 0 1000) :discard-label-p discard-label-p))

(defun sample-image-array (sample)
  (image-array (first sample)))

(defun clamp-images (samples mat)
  (assert (= (length samples) (mat-dimension mat 0)))
  (map-concat #'copy! samples mat :key #'sample-image-array))

(defun clamp-labels (samples mat)
  (assert (= (length samples) (mat-dimension mat 0)))
  (fill! 0 mat)
  (let ((n-columns (mat-dimension mat 1))
        (displacement (mat-displacement mat)))
    (with-facets ((a (mat 'backing-array :direction :io :type flt-vector)))
      (loop for sample in samples
            for row upfrom 0
            do (destructuring-bind (image &key discard-label-p sample-visible-p)
                   sample
                 (declare (ignore sample-visible-p))
                 (unless discard-label-p
                   (setf (aref a (+ displacement
                                    (* row n-columns)
                                    (image-label image)))
                         #.(flt 1))))))))


;;;; Logging

(defclass mnist-base-trainer (cesc-trainer)
  ((training :initarg :training :reader training)
   (test :initarg :test :reader test)))

(defmethod log-training-period ((trainer mnist-base-trainer) learner)
  (declare (ignore learner))
  (length (training trainer)))

(defmethod log-test-period ((trainer mnist-base-trainer) learner)
  (declare (ignore learner))
  (length (training trainer)))


;;;; DBN

(defclass mnist-dbn (dbn) ())

(defclass mnist-rbm (rbm) ())

(defmethod set-input (samples (rbm mnist-rbm))
  (let ((inputs (find 'inputs (visible-chunks rbm) :key #'name)))
    (when inputs
      (clamp-images samples (nodes inputs)))))

(defclass mnist-rbm-trainer (mnist-base-trainer segmented-gd-trainer)
  ())

(defclass mnist-rbm-cd-learner (rbm-cd-learner)
  ((trainer :initarg :trainer :reader trainer)))

(defmethod n-gibbs ((learner mnist-rbm-cd-learner))
  (let ((x (slot-value learner 'n-gibbs)))
    (if (integerp x)
        x
        (funcall x (trainer learner)))))

(defmethod log-test-error ((trainer mnist-rbm-trainer) learner)
  (call-next-method)
  (let ((rbm (rbm learner)))
    (log-dbn-cesc-accuracy rbm (make-tiny-sampler (training trainer))
                           "training reconstruction")
    (log-dbn-cesc-accuracy rbm (make-tiny-sampler (training trainer)
                                                   :discard-label-p t)
                           "training")
    (map nil (lambda (counter)
               (log-msg "dbn test: ~:_test ~:_~A~%" counter))
         (collect-dbn-mean-field-errors/labeled
          (make-sampler (test trainer)) (dbn rbm) :rbm rbm
          :counters-and-measurers
          (make-dbn-reconstruction-rmse-counters-and-measurers
           (dbn rbm) :rbm rbm)))
    (log-dbn-cesc-accuracy rbm (make-sampler (test trainer) :discard-label-p t)
                           "test")
    (log-msg "---------------------------------------------------~%")))


;;;; DBN training

(defclass mnist-rbm-segment-trainer (batch-gd-trainer)
  ((n-inputs-in-epoch :initarg :n-inputs-in-epoch :reader n-inputs-in-epoch)))

(defmethod learning-rate ((trainer mnist-rbm-segment-trainer))
  (let ((x (slot-value trainer 'learning-rate)))
    (if (numberp x)
        x
        (funcall x trainer))))

(defmethod momentum ((trainer mnist-rbm-segment-trainer))
  (if (< (n-inputs trainer) (* 5 (n-inputs-in-epoch trainer)))
      #.(flt 0.5)
      #.(flt 0.9)))

#+nil
(defmethod lag ((trainer mnist-rbm-trainer))
  (min (flt 0.99)
       (let ((n (/ (n-inputs trainer)
                   (n-inputs-in-epoch trainer))))
         (- 1 (expt 0.5 n)))))

(defun init-mnist-dbn (dbn &key stddev (start-level 0))
  (loop for i upfrom start-level
        for rbm in (subseq (rbms dbn) start-level) do
          (flet ((this (x)
                   (if (listp x)
                       (elt x i)
                       x)))
            (do-clouds (cloud rbm)
              (if (conditioning-cloud-p cloud)
                  (fill! (flt 0) (weights cloud))
                  (progn
                    (log-msg "init: ~A ~A~%" cloud (this stddev))
                    (gaussian-random! (weights cloud)
                                      :stddev (this stddev))))))))

(defun train-mnist-dbn (dbn training test &key n-epochs n-gibbs learning-rate
                        decay visible-sampling (start-level 0))
  (loop
    for rbm in (subseq (rbms dbn) start-level)
    for i upfrom start-level do
      (log-msg "Starting to train level ~S RBM in DBN.~%" i)
      (flet ((this (x)
               (if (and x (listp x))
                   (elt x i)
                   x)))
        (let* ((n (length training))
               (trainer
                 (make-instance 'mnist-rbm-trainer
                                :training training
                                :test test
                                :segmenter
                                (lambda (cloud)
                                  (make-instance 'mnist-rbm-segment-trainer
                                                 :n-inputs-in-epoch n
                                                 :learning-rate
                                                 (this learning-rate)
                                                 :weight-decay
                                                 (if (conditioning-cloud-p
                                                      cloud)
                                                     (flt 0)
                                                     (this decay))
                                                 :batch-size 100)))))
          (train (make-sampler training
                               :n-epochs n-epochs
                               :sample-visible-p (this visible-sampling))
                 trainer
                 (make-instance 'mnist-rbm-cd-learner
                                :trainer trainer
                                :rbm rbm
                                :visible-sampling (this visible-sampling)
                                :n-gibbs (this n-gibbs)))))))


;;;; BPN

(defclass mnist-bpn (bpn) ())

(defmethod set-input (samples (bpn mnist-bpn))
  (let* ((inputs (or (find-lump (chunk-lump-name 'inputs nil) bpn)
                     (find-lump 'inputs bpn)))
         (expectations (find-lump 'expectations bpn :errorp t)))
    (clamp-images samples (nodes inputs))
    (clamp-labels samples (nodes expectations))))

(defun tack-cross-entropy-softmax-error-on (bpn inputs)
  (add-cross-entropy-softmax :predictions-name 'predictions
                             :expectations-name 'expectations
                             :size 10
                             :inputs inputs
                             :bpn bpn))

(defun prediction-weight-p (lump)
  (let ((name (name lump)))
    (and (listp name)
         (= 2 (length name))
         (eq 'predictions (second name)))))

(defun make-bpn (defs chunk-name &key class initargs)
  (let ((bpn-def `(build-bpn (:class ',class
                              :max-n-stripes 1000
                              :initargs ',initargs)
                    ,@defs)))
    (log-msg "bpn def:~%~S~%" bpn-def)
    (let* ((bpn (eval bpn-def))
           (name (chunk-lump-name chunk-name nil))
           (lump (find-lump name bpn)))
      (tack-cross-entropy-softmax-error-on bpn (list lump))
      bpn)))


;;;; BPN training

(defclass mnist-bp-trainer (mnist-base-trainer) ())

(defclass mnist-cg-bp-trainer (mnist-bp-trainer cg-trainer) ())

(defmethod log-test-error ((trainer mnist-bp-trainer) learner)
  (call-next-method)
  (map nil (lambda (counter)
             (log-msg "bpn test: test ~:_~A~%" counter))
       (bpn-cesc-error (make-sampler (test trainer)) (bpn learner)))
  (log-msg "---------------------------------------------------~%"))

(defun init-bpn-lump-weights (name bpn stddev)
  (gaussian-random! (nodes (find-lump name bpn :errorp t)) :stddev stddev))

(defun train-mnist-bpn-cg (bpn training test &key
                           (batch-size (min (length training) 1000))
                           (n-softmax-epochs 5) n-epochs)
  (log-msg "Starting to train the softmax layer of BPN~%")
  (train (make-sampler training :n-epochs n-softmax-epochs)
         (make-instance 'mnist-cg-bp-trainer
                        :training training
                        :test test
                        :cg-args (list :max-n-line-searches 3)
                        :batch-size batch-size
                        :segment-filter #'prediction-weight-p)
         (make-instance 'bp-learner :bpn bpn))
  (log-msg "Starting to train the whole BPN~%")
  (train (make-sampler training :n-epochs (- n-epochs n-softmax-epochs))
         (make-instance 'mnist-cg-bp-trainer
                        :training training
                        :test test
                        :cg-args (list :max-n-line-searches 3)
                        :batch-size batch-size)
         (make-instance 'bp-learner :bpn bpn)))


;;;; Code for the DBN to BPN approach (paper [1]) and also for dropout
;;;; training with the bpn (paper [3]) with :DROPOUT T.

(defclass mnist-dbn/1 (mnist-dbn)
  ()
  (:default-initargs
   :layers (list (list (make-instance 'constant-chunk :name 'c0)
                       (make-instance 'sigmoid-chunk :name 'inputs
                                      :size (* 28 28)))
                 (list (make-instance 'constant-chunk :name 'c1)
                       (make-instance 'sigmoid-chunk :name 'f1
                                      :size 500))
                 (list (make-instance 'constant-chunk :name 'c2)
                       (make-instance 'sigmoid-chunk :name 'f2
                                      :size 500))
                 (list (make-instance 'constant-chunk :name 'c3)
                       (make-instance 'sigmoid-chunk :name 'f3
                                      :size 2000)))
    :rbm-class 'mnist-rbm
    :max-n-stripes 100))

(defun unroll-mnist-dbn/1 (dbn)
  (multiple-value-bind (defs inits) (unroll-dbn dbn :bottom-up-only t)
    (log-msg "bpn inits:~%~S~%" inits)
    (let ((bpn (make-bpn defs 'f3 :class 'mnist-bpn)))
      (initialize-bpn-from-bm bpn dbn inits)
      (map nil (lambda (lump)
                 (when (prediction-weight-p lump)
                   (init-bpn-lump-weights (name lump) bpn 0.1)))
           (lumps bpn))
      bpn)))

(defun train-mnist/1 (&key training test load-dbn-p quick-run-p dropoutp
                      dbn-var bpn-var dbn-filename bpn-filename)
  (let ((dbn nil)
        (bpn nil))
    (cond (load-dbn-p
           (with-experiment ()
             (setq* (dbn dbn-var) (make-instance 'mnist-dbn/1))
             (load-weights dbn-filename dbn)
             (log-msg "Loaded DBN~%")
             (map nil (lambda (counter)
                        (log-msg "dbn test ~:_~A~%" counter))
                  (collect-dbn-mean-field-errors/labeled
                   (make-sampler test) dbn))))
          (t
           (with-experiment ()
             (setq* (dbn dbn-var) (make-instance 'mnist-dbn/1))
             (init-mnist-dbn dbn :stddev 0.1)
             (train-mnist-dbn dbn training test
                              :n-epochs (if quick-run-p 2 50) :n-gibbs 1
                              :start-level 0 :learning-rate (flt 0.1)
                              :decay (flt 0.0002) :visible-sampling nil)
             (unless quick-run-p
               (save-weights dbn-filename dbn)))))
    (with-experiment ()
      (setq* (bpn bpn-var) (unroll-mnist-dbn/1 dbn))
      (cond (dropoutp
             (train-mnist-bpn-gd bpn training test
                                 :n-softmax-epochs (if quick-run-p 1 10)
                                 :n-epochs (if quick-run-p 2 1000)
                                 :learning-rate (flt 1)
                                 :learning-rate-decay (flt 0.998)
                                 :l2-upper-bound nil
                                 :rescale-on-dropout-p t)
             (unless quick-run-p
               (save-weights bpn-filename bpn)))
            (t
             (train-mnist-bpn-cg bpn training test
                                 :n-softmax-epochs (if quick-run-p 1 5)
                                 :n-epochs (if quick-run-p 1 37))
             (unless quick-run-p
               (save-weights bpn-filename bpn)))))
    (values bpn dbn)))


;;;; Code for the DBN to DBM to BPN approach (paper [2]) and also for
;;;; dropout training with the bpn (paper [3]) with :DROPOUT T.

(defclass mnist-rbm/2 (mnist-rbm) ())
(defclass mnist-bpn/2 (mnist-bpn bpn-clamping-cache) ())

(defun clamp-chunk-labels (samples chunk)
  (setf (indices-present chunk)
        (if (and samples (getf (rest (elt samples 0)) :discard-label-p))
            #.(make-array 0 :element-type 'index)
            nil))
  (clamp-labels samples (nodes chunk)))

(defun strip-sample-visible (samples)
  (mapcar (lambda (sample)
            (destructuring-bind (image &key discard-label-p sample-visible-p)
                sample
              (declare (ignore sample-visible-p))
              (list image :discard-label-p discard-label-p)))
          samples))

(defmethod set-input (samples (rbm mnist-rbm/2))
  (call-next-method (strip-sample-visible samples) rbm)
  ;; All samples have the same SAMPLE-VISIBLE-P, look at the first
  ;; only.
  (when (and samples (getf (rest (elt samples 0)) :sample-visible-p))
    (sample-visible rbm))
  (let ((label-chunk (find-chunk 'label rbm)))
    (when label-chunk
      (clamp-chunk-labels samples label-chunk))))

(defclass mnist-dbm (dbm)
  ((layers :initform (list
                      (list (make-instance 'constant-chunk :name 'c0)
                            (make-instance 'sigmoid-chunk :name 'inputs
                                           :size (* 28 28)))
                      (list (make-instance 'constant-chunk :name 'c1)
                            (make-instance 'sigmoid-chunk :name 'f1
                                           :size 500)
                            (make-instance 'softmax-label-chunk* :name 'label
                                           :size 10 :group-size 10))
                      (list (make-instance 'constant-chunk :name 'c2)
                            (make-instance 'sigmoid-chunk :name 'f2
                                           :size 1000))))
   (clouds :initform '(:merge
                       (:chunk1 c0 :chunk2 label :class nil)
                       (:chunk1 inputs :chunk2 label :class nil)))))

(defun make-mnist-dbm ()
  (make-instance 'mnist-dbm :max-n-stripes 100))

(defmethod set-input (samples (dbm mnist-dbm))
  (clamp-images samples (nodes (mgl-bm:find-chunk 'inputs dbm)))
  (when (and samples (getf (rest (elt samples 0)) :sample-visible-p))
    (sample-visible dbm))
  (let ((label-chunk (find-chunk 'label dbm :errorp t)))
    (clamp-chunk-labels samples label-chunk)))

(defclass mnist-dbm-trainer (mnist-base-trainer segmented-gd-trainer)
  ())

(defmethod log-test-error ((trainer mnist-dbm-trainer) learner)
  (call-next-method)
  (let ((dbm (bm learner)))
    (log-dbm-cesc-accuracy dbm (make-tiny-sampler (training trainer))
                           "training reconstruction")
    (log-dbm-cesc-accuracy dbm (make-tiny-sampler (training trainer)
                                                  :discard-label-p t)
                           "training")
    ;; This is too time consuming for little benefit.
    #+nil
    (map nil (lambda (counter)
               (log-msg "dbm test: ~:_test ~:_~A~%" counter))
         (collect-bm-mean-field-errors
          (make-sampler (test trainer)) dbm
          :counters-and-measurers
          (make-dbm-reconstruction-rmse-counters-and-measurers dbm)))
    (log-dbm-cesc-accuracy dbm (make-sampler (test trainer) :discard-label-p t)
                           "test")
    (log-msg "---------------------------------------------------~%")))

(defclass mnist-dbm-segment-trainer (batch-gd-trainer)
  ((n-inputs-in-epoch :initarg :n-inputs-in-epoch :reader n-inputs-in-epoch)))

(defmethod learning-rate ((trainer mnist-dbm-segment-trainer))
  ;; This is adjusted for each batch. Ruslan's code adjusts it per
  ;; epoch.
  (/ (slot-value trainer 'learning-rate)
     (min (flt 10)
          (expt 1.000015
                (* (/ (n-inputs trainer) (n-inputs-in-epoch trainer))
                   600)))))

(defmethod momentum ((trainer mnist-dbm-segment-trainer))
  (if (< (n-inputs trainer) (* 5 (n-inputs-in-epoch trainer)))
      #.(flt 0.5)
      #.(flt 0.9)))

(defmethod initialize-gradient-source (learner dbm (trainer mnist-dbm-trainer))
  (declare (ignore dbm))
  (when (next-method-p)
    (call-next-method))
  (set-input (sample-batch (make-sampler (training trainer) :sample-visible-p t)
                           (n-particles learner))
             (persistent-chains learner))
  (up-dbm (persistent-chains learner)))

(defun train-mnist-dbm (dbm training test &key (n-epochs 500))
  (log-msg "Starting to train DBM.~%")
  (let ((n (length training)))
    (train (make-sampler training :n-epochs n-epochs :sample-visible-p t)
           (make-instance 'mnist-dbm-trainer
                          :training training
                          :test test
                          :segmenter
                          (lambda (cloud)
                            (make-instance 'mnist-dbm-segment-trainer
                                           :n-inputs-in-epoch n
                                           :learning-rate (flt 0.001)
                                           :weight-decay
                                           (if (conditioning-cloud-p cloud)
                                               (flt 0)
                                               (flt 0.0002))
                                           :batch-size 100)))
           (make-instance 'bm-pcd-learner
                          :bm dbm
                          :n-particles 100
                          :visible-sampling t
                          :n-gibbs 5
                          :sparser
                          (lambda (cloud chunk)
                            (when (and (member (name chunk) '(f1 f2))
                                       (not (equal 'label
                                                   (name (chunk1 cloud))))
                                       (not (equal 'label
                                                   (name (chunk2 cloud)))))
                              (make-instance
                               'cheating-sparsity-gradient-source
                               :cloud cloud
                               :chunk chunk
                               :sparsity (flt (if (eq 'f2 (name chunk))
                                                  0.1
                                                  0.2))
                               :cost (flt 0.001)
                               :damping (flt 0.9))))))))

(defun make-mnist-dbn/2 (dbm)
  (mgl-bm:dbm->dbn dbm :dbn-class 'mnist-dbn :rbm-class 'mnist-rbm/2
                   :dbn-initargs '(:max-n-stripes 100)))

(defun unroll-mnist-dbm (dbm &key use-label-weights initargs)
  (multiple-value-bind (defs inits)
      (unroll-dbm dbm :chunks (remove 'label (chunks dbm) :key #'name))
    (log-msg "inits:~%~S~%" inits)
    (let ((bpn (make-bpn defs 'f2 :class 'mnist-bpn/2 :initargs initargs))
          (name (chunk-lump-name 'f2 nil)))
      ;; In the DBM, the label is computed top-down so make sure its
      ;; weights we'll copy to the BPN will be transposed.
      (setf (slot-value (find-lump `((,name predictions) :activation) bpn
                                   :errorp t)
                        'mgl-bp::transpose-weights-p)
            t)
      (initialize-bpn-from-bm bpn dbm
                              (if use-label-weights
                                  (list*
                                   '(:cloud-name (label c2)
                                     :weight-name (:bias predictions))
                                   '(:cloud-name (label f2)
                                     :weight-name ((:chunk f2) predictions))
                                   inits)
                                  inits))
      bpn)))

(defun train-mnist/2 (&key training test load-dbn-p load-dbm-p dropoutp
                      quick-run-p dbn-var dbm-var bpn-var
                      dbn-filename dbm-filename bpn-filename)
  (let ((dbn nil)
        (dbm nil)
        (bpn nil))
    (flet ((train-dbn ()
             (with-experiment ()
               (init-mnist-dbn dbn :stddev '(0.001 0.01) :start-level 0)
               (train-mnist-dbn
                dbn training test
                :start-level 0
                :n-epochs (if quick-run-p 2 100)
                :n-gibbs (list 1
                               (lambda (trainer)
                                 (ceiling (1+ (n-inputs trainer))
                                          (* 20 (length training)))))
                :learning-rate (list (flt 0.05)
                                     (lambda (trainer)
                                       (/ (flt 0.05)
                                          (ceiling (1+ (n-inputs trainer))
                                                   (* 20 (length training))))))
                :decay (flt 0.001)
                :visible-sampling t))
             (unless quick-run-p
               (save-weights dbn-filename dbn)))
           (train-dbm ()
             (with-experiment ()
               (train-mnist-dbm dbm training test
                                :n-epochs (if quick-run-p 1 500)))
             (unless quick-run-p
               (save-weights dbm-filename dbm)))
           (make-dbm ()
             (with-experiment ()
               (setq* (dbm dbm-var) (make-mnist-dbm))))
           (make-dbn ()
             (with-experiment ()
               (setq* (dbn dbn-var) (make-mnist-dbn/2 dbm)))))
      (cond (load-dbm-p
             (make-dbm)
             (load-weights dbm-filename dbm))
            (load-dbn-p
             (make-dbm)
             (make-dbn)
             (load-weights dbn-filename dbn)
             (log-msg "Loaded DBN~%")
             (train-dbm))
            (t
             (make-dbm)
             (make-dbn)
             (train-dbn)
             (train-dbm)))
      (with-experiment ()
        (setq* (bpn bpn-var)
               (unroll-mnist-dbm
                dbm
                :use-label-weights t
                :initargs
                ;; We are playing games with wrapping training examples
                ;; in lists when sampling, so let the map cache know
                ;; what's the key in the cache and what to pass to the
                ;; dbm. We carefully omit labels.
                (list :populate-key #'first
                      :populate-convert-to-dbm-sample-fn
                      (lambda (sample)
                        (list (first sample)
                              :sample-visible-p nil
                              :discard-label-p t))
                      :populate-map-cache-lazily-from-dbm dbm
                      :populate-periodic-fn
                      (make-instance 'periodic-fn :period 1000
                                     :fn (lambda (n)
                                           (log-msg "populated: ~S~%" n))))))
        (unless (populate-map-cache-lazily-from-dbm bpn)
          (log-msg "Populating MAP cache~%")
          (populate-map-cache bpn dbm (concatenate 'vector training test)
                              :if-exists :error))
        (cond (dropoutp
               (train-mnist-bpn-gd bpn training test
                                   :n-softmax-epochs 0
                                   :n-epochs (if quick-run-p 2 1000)
                                   :learning-rate (flt 1)
                                   :learning-rate-decay (flt 0.998)
                                   :l2-upper-bound nil
                                   :rescale-on-dropout-p t)
               (unless quick-run-p
                 (save-weights bpn-filename bpn)))
              (t
               (train-mnist-bpn-cg bpn training test :batch-size 10000
                                   :n-softmax-epochs (if quick-run-p 1 5)
                                   :n-epochs (if quick-run-p 1 100))
               (unless quick-run-p
                 (save-weights bpn-filename bpn))))))))

(defmethod negative-phase (batch (learner bm-pcd-learner)
                           (trainer mnist-dbm-trainer) multiplier)
  (let ((bm (persistent-chains learner)))
    (mgl-bm::check-no-self-connection bm)
    (flet ((foo (chunk)
             (mgl-bm::set-mean (list chunk) bm)
             (sample-chunk chunk)))
      (loop for i below (n-gibbs learner) do
        (sample-chunk (find-chunk 'f1 bm))
        (foo (find-chunk 'inputs bm))
        (foo (find-chunk 'f2 bm))
        (foo (find-chunk 'label bm))
        (mgl-bm::set-mean (list (find-chunk 'f1 bm)) bm))
      (mgl-bm::set-mean* (list (find-chunk 'f2 bm)) bm)
      (mgl-bm::accumulate-negative-phase-statistics
       learner trainer
       (* multiplier (/ (length batch) (n-stripes bm)))))))


;;;; Code for the plain dropout backpropagation network with rectified
;;;; linear units (paper [3])

(defclass mnist-bpn-gd-trainer (mnist-bp-trainer segmented-gd-trainer)
  ())

#+nil
(defclass mnist-bpn-gd-trainer (mnist-bp-trainer svrg-trainer)
  ())

(defclass mnist-bpn-gd-segment-trainer (batch-gd-trainer)
  ((n-inputs-in-epoch :initarg :n-inputs-in-epoch :reader n-inputs-in-epoch)
   (n-epochs-to-reach-final-momentum
    :initarg :n-epochs-to-reach-final-momentum
    :reader n-epochs-to-reach-final-momentum)
   (learning-rate-decay
    :initform (flt 0.998)
    :initarg :learning-rate-decay
    :accessor learning-rate-decay)))

(defmethod learning-rate ((trainer mnist-bpn-gd-segment-trainer))
  (* (expt (learning-rate-decay trainer)
           (/ (n-inputs trainer)
              (n-inputs-in-epoch trainer)))
     (- 1 (momentum trainer))
     (slot-value trainer 'learning-rate)))

(defmethod momentum ((trainer mnist-bpn-gd-segment-trainer))
  (let ((n-epochs-to-reach-final (n-epochs-to-reach-final-momentum trainer))
        (initial (flt 0.5))
        (final (flt 0.99))
        (epoch (/ (n-inputs trainer) (n-inputs-in-epoch trainer))))
    (if (< epoch n-epochs-to-reach-final)
        (let ((weight (/ epoch n-epochs-to-reach-final)))
          (+ (* initial (- 1 weight))
             (* final weight)))
        final)))

#+nil
(defmethod lag ((trainer mnist-bpn-gd-trainer))
  (min (flt 0.99)
       (let ((n (/ (n-inputs trainer)
                   (length (training trainer)))))
         (- 1 (expt 0.5 n)))))

(defun make-grouped-segmenter (name-groups segmenter)
  (let ((group-to-trainer (make-hash-table :test #'equal)))
    (lambda (segment)
      (let* ((name (name segment))
             (group (find-if (lambda (names)
                               (member name names :test #'name=))
                             name-groups)))
        (assert group () "Can't find group for ~S." name)
        (or (gethash group group-to-trainer)
            (setf (gethash group group-to-trainer)
                  (funcall segmenter segment)))))))

(defun train-mnist-bpn-gd (bpn training test &key (n-softmax-epochs 5)
                           (n-epochs 200) l2-upper-bound
                           class-weights learning-rate learning-rate-decay
                           rescale-on-dropout-p)
  (when class-weights
    (setf (class-weights (find-lump 'predictions bpn)) class-weights))
  (setf (max-n-stripes bpn) 100)
  (let ((name-groups '(((:cloud (c0 f1))
                        (:cloud (inputs f1))
                        (:bias f1)
                        (inputs f1))
                       ((:cloud (c1 f2))
                        (:cloud (f1 f2))
                        (:bias f2)
                        (f1 f2))
                       ((:cloud (c2 f3))
                        (:cloud (f2 f3))
                        (:bias f3)
                        (f2 f3))
                       ;; for mnist/3
                       ((:bias predictions)
                        (f3 predictions))
                       ;; for mnist/4
                       ((:bias predictions)
                        (f2 predictions)))))
    (flet ((make-trainer (lump &key softmaxp)
             (declare (ignore lump))
             (let ((trainer (make-instance
                             'mnist-bpn-gd-segment-trainer
                             :n-inputs-in-epoch (length training)
                             :n-epochs-to-reach-final-momentum
                             (min 500
                                  (/ (if softmaxp n-softmax-epochs n-epochs)
                                     2))
                             :learning-rate (flt learning-rate)
                             :learning-rate-decay (flt learning-rate-decay)
                             :batch-size 100)))
               (when l2-upper-bound
                 (arrange-for-renormalizing-activations
                  bpn trainer l2-upper-bound))
               trainer))
           (make-segmenter (fn)
             (if l2-upper-bound
                 (make-grouped-segmenter name-groups fn)
                 fn)))
      (unless (zerop n-softmax-epochs)
        (log-msg "Starting to train the softmax layer of BPN~%")
        (train (make-sampler training :n-epochs n-softmax-epochs)
               (make-instance 'mnist-bpn-gd-trainer
                              :training training
                              :test test
                              :segmenter
                              (make-segmenter
                               (lambda (lump)
                                 (when (prediction-weight-p lump)
                                   (make-trainer lump :softmaxp t)))))
               (make-instance 'bp-learner :bpn bpn)))
      (map nil (lambda (lump)
                 (when (and (typep lump '->dropout)
                            (not (typep lump '->input)))
                   (log-msg "dropout ~A ~A~%" lump 0.5)
                   (if rescale-on-dropout-p
                       (set-dropout-and-rescale-activation-weights
                        lump (flt 0.5) bpn)
                       (setf (slot-value lump 'dropout) (flt 0.5))))
                 (when (and (typep lump '->input)
                            (member (name lump) '((:chunk inputs)
                                                  inputs)
                                    :test #'name=))
                   (log-msg "dropout ~A ~A~%" lump 0.2)
                   (if rescale-on-dropout-p
                       (set-dropout-and-rescale-activation-weights
                        lump (flt 0.2) bpn)
                       (setf (slot-value lump 'dropout) (flt 0.2)))))
           (lumps bpn))
      (unless (zerop n-epochs)
        (mgl-example-util:log-msg "Starting to train the whole BPN~%")
        (train (make-sampler training :n-epochs n-epochs)
               (make-instance 'mnist-bpn-gd-trainer
                              :training training
                              :test test
                              :segmenter (make-segmenter #'make-trainer))
               (make-instance 'bp-learner :bpn bpn))))))

(defun build-rectified-mnist-bpn (&key (n-units-1 1200) (n-units-2 1200)
                                  (n-units-3 1200))
  (build-bpn (:class 'mnist-bpn :max-n-stripes 100)
    (inputs (->input :name 'inputs :size 784))
    (f1-activations (add-activations :name 'f1 :inputs '(inputs)
                                     :size n-units-1))
    (f1* (->rectified :x f1-activations))
    (f1 (->dropout :x f1*))
    (f2-activations (add-activations :name 'f2 :inputs (list f1)
                                     :size n-units-2))
    (f2* (->rectified :x f2-activations))
    (f2 (->dropout :x f2*))
    (f3-activations (add-activations :name 'f3 :inputs (list f2)
                                     :size n-units-3))
    (f3* (->rectified :x f3-activations))
    (f3 (->dropout :x f3*))
    (predictions (tack-cross-entropy-softmax-error-on
                  mgl-bp::*bpn-being-built* (list f3)))))

(defun init-bpn-weights (bpn &key stddev)
  (loop for lump across (lumps bpn) do
    (when (typep lump '->weight)
      (gaussian-random! (nodes lump) :stddev stddev))))

(defun train-mnist/3 (&key training test quick-run-p bpn-var bpn-filename)
  (with-experiment ()
    (let ((bpn nil))
      (setq* (bpn bpn-var) (build-rectified-mnist-bpn))
      (init-bpn-weights bpn :stddev 0.01)
      (train-mnist-bpn-gd bpn training test
                          :n-softmax-epochs 0
                          :n-epochs (if quick-run-p 2 3000)
                          :learning-rate (flt 1)
                          :learning-rate-decay (flt 0.998)
                          :l2-upper-bound (sqrt (flt 15)))
      (unless quick-run-p
        (save-weights bpn-filename bpn)))))


;;;; Maxout

(defun build-maxout-mnist-bpn (&key (n-units-1 1200) (n-units-2 1200)
                               (group-size 5))
  (build-bpn (:class 'mnist-bpn :max-n-stripes 100)
    (inputs (->input :name 'inputs :size 784))
    (f1-activations (add-activations :name 'f1 :inputs '(inputs)
                                     :size n-units-1))
    (f1* (->max :x f1-activations :group-size group-size))
    (f1 (->dropout :x f1*))
    (f2-activations (add-activations :name 'f2 :inputs (list f1)
                                     :size n-units-2))
    (f2* (->max :x f2-activations :group-size group-size))
    (f2 (->dropout :x f2*))
    (predictions (tack-cross-entropy-softmax-error-on
                  mgl-bp::*bpn-being-built* (list f2)))))

(defun train-mnist/4 (&key training test quick-run-p bpn-var bpn-filename)
  (with-experiment ()
    (let ((bpn nil))
      (setq* (bpn bpn-var) (build-maxout-mnist-bpn))
      (init-bpn-weights bpn :stddev 0.01)
      (train-mnist-bpn-gd bpn training test
                          :n-softmax-epochs 0
                          :n-epochs (if quick-run-p 2 3000)
                          :learning-rate (flt 1)
                          :learning-rate-decay (flt 0.998)
                          :l2-upper-bound (sqrt (flt 3.75)))
      (unless quick-run-p
        (save-weights bpn-filename bpn)))))


;;;; Globals for tracking progress of training and filenames

;;;; MNIST/1

(defvar *dbn/1*)
(defvar *bpn/1*)

(defparameter *mnist-1-dbn-filename*
  (merge-pathnames "mnist-1.dbn" *mnist-save-dir*))

(defparameter *mnist-1-bpn-filename*
  (merge-pathnames "mnist-1.bpn" *mnist-save-dir*))

(defparameter *mnist-1-dropout-bpn-filename*
  (merge-pathnames "mnist-1-dropout.bpn" *mnist-save-dir*))

(defvar *dbn/2*)
(defvar *dbm/2*)
(defvar *bpn/2*)

;;;; MNIST/2

(defparameter *mnist-2-dbn-filename*
  (merge-pathnames "mnist-2.dbn" *mnist-save-dir*))

(defparameter *mnist-2-dbm-filename*
  (merge-pathnames "mnist-2.dbm" *mnist-save-dir*))

(defparameter *mnist-2-bpn-filename*
  (merge-pathnames "mnist-2.bpn" *mnist-save-dir*))

(defparameter *mnist-2-dropout-bpn-filename*
  (merge-pathnames "mnist-2-dropout.bpn" *mnist-save-dir*))

;;;; MNIST/3

(defvar *bpn/3*)

(defparameter *mnist-3-bpn-filename*
  (merge-pathnames "mnist-3.bpn" *mnist-save-dir*))

;;;; MNIST/4

(defvar *bpn/4*)

(defparameter *mnist-4-bpn-filename*
  (merge-pathnames "mnist-4.bpn" *mnist-save-dir*))

(defun run-quick-test ()
  (train-mnist/1 :training (training-images) :test (test-images)
                 :quick-run-p t)
  (train-mnist/2 :training (training-images) :test (test-images)
                 :quick-run-p t)
  (train-mnist/2 :training (training-images) :test (test-images) :dropoutp t
                 :quick-run-p t)
  (train-mnist/3 :training (training-images) :test (test-images)
                 :quick-run-p t)
  (train-mnist/4 :training (training-images) :test (test-images)
                 :quick-run-p t))

#|

(run-quick-test)

(progn
  (train-mnist/1 :training (training-images) :test (test-images)
                 :dbn-var '*dbn/1* :bpn-var '*bpn/1*
                 :dbn-filename *mnist-1-dbn-filename*
                 :bpn-filename *mnist-1-bpn-filename*)
  (train-mnist/1 :training (training-images) :test (test-images)
                 :load-dbn-p t :dropoutp t
                 :dbn-var '*dbn/1* :bpn-var '*bpn/1*
                 :dbn-filename *mnist-1-dbn-filename*
                 :bpn-filename *mnist-1-dropout-bpn-filename*)
  (train-mnist/2 :training (training-images) :test (test-images)
                 :dbn-var '*dbn/2* :dbm-var '*dbm/2* :bpn-var '*bpn/2*
                 :dbn-filename *mnist-2-dbn-filename*
                 :dbm-filename *mnist-2-dbm-filename*
                 :bpn-filename *mnist-2-bpn-filename*)
  (train-mnist/2 :training (training-images) :test (test-images)
                 :load-dbm-p t :dropoutp t
                 :dbn-var '*dbn/2* :dbm-var '*dbm/2* :bpn-var '*bpn/2*
                 :dbn-filename *mnist-2-dbn-filename*
                 :dbm-filename *mnist-2-dbm-filename*
                 :bpn-filename *mnist-2-dropout-bpn-filename*)
  (train-mnist/3 :training (training-images) :test (test-images)
                 :bpn-var '*bpn/3* :bpn-filename *mnist-3-bpn-filename*)
  (train-mnist/4 :training (training-images) :test (test-images)
                 :bpn-var '*bpn/4* :bpn-filename *mnist-4-bpn-filename*))

(require :sb-sprof)

(progn
  (sb-sprof:reset)
  (sb-sprof:start-profiling)
  (sleep 10)
  (sb-sprof:stop-profiling)
  (sb-sprof:report :type :graph))

(let ((dgraph (cl-dot:generate-graph-from-roots *dbn/2* (chunks *dbn/2*)
                                                '(:rankdir "BT"))))
  (cl-dot:dot-graph dgraph
                    (asdf-system-relative-pathname "example/mnist-dbn-2.png")
                    :format :png))

(let ((dgraph (cl-dot:generate-graph-from-roots *dbm/2* (chunks *dbm/2*)
                                                '(:rankdir "BT"))))
  (cl-dot:dot-graph dgraph
                    (asdf-system-relative-pathname "example/mnist-dbm-2.png")
                    :format :png))

(let ((dgraph (cl-dot:generate-graph-from-roots *bpn/2* (lumps *bpn/2*))))
  (cl-dot:dot-graph dgraph
                    (asdf-system-relative-pathname "example/mnist-bpn-2.png")
                    :format :png))

(dotimes (i 100)
  (print-image-array (image-array (aref (training-images) i)))
  (print (image-label (aref (training-images) i)))
  (terpri))

|#
