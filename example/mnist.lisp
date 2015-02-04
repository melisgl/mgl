;;;; Code for the MNIST handwritten digit recognition challange.
;;;;
;;;; References:
;;;;
;;;; For the DBN-to-BPN approach:
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
;;;; For dropout:
;;;;
;;;;   "Improving neural networks by preventing co-adaptation of
;;;;   feature detectors"
;;;;   http://arxiv.org/pdf/1207.0580.pdf
;;;;
;;;; Maxout:
;;;;
;;;;  "Maxout Networks"
;;;;  http://arxiv.org/abs/1302.4389
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
;;;; reaches ~98.86% accuracy. With CUDA on a GTX Titan, it takes 30
;;;; minutes with double floats.
;;;;
;;;; If the same model is fine-tuned with gradient descent and
;;;; dropout, then accuracy improves to ~98.96.
;;;;
;;;; During DBN training the DBN TEST RMSE is the RMSE of the mean
;;;; field reconstruction of the test images while TRAINING RMSE is of
;;;; the stochastic reconstructions (that is, the hidden layer is
;;;; sampled) during training. Naturally, the stochastic one tends to
;;;; be higher but is cheap to compute.
;;;;
;;;;
;;;; DBN-to-DBM-to-BPN (see TRAIN-MNIST/2)
;;;;
;;;; A 784-500-1000 DBM is pretrained as a DBN then the DBM is
;;;; translated to a BPN. It's pretty much the same training process
;;;; as before except that after the DBN the DBM is trained by PCD
;;;; before unrolling that into a BPN. Accuracy is ~99.09% a bit
;;;; higher than the reported 99.05%. 
;;;;
;;;; Fine-tuned with dropout, it's ~99.22%.
;;;;
;;;;
;;;; Rectified Dropout BPN (see TRAIN-MNIST/3)
;;;;
;;;; A 784-1200-1200-1200 BPN is trained directly. ~98.91% in 7-fold
;;;; CV. ~98.97% with INPUT-WEIGHT-PENALTY.
;;;;
;;;;
;;;; Maxout BPN (see TRAIN-MNIST/4)
;;;;
;;;; I couldn't quite reproduce the 99.06% claimed in the paper.
;;;; 98.91% in 7-fold CV . 98.97% with INPUT-WEIGHT-PENALTY.
;;;;
;;;;
;;;; Max-channel BPN (see TRAIN-MNIST/5)
;;;;
;;;; 7-fold CV: 99.01%. 99.07% with INPUT-WEIGHT-PENALTY.

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
  (let ((a (make-array (* 28 28))))
    (loop for i below (* 28 28) do
      (let ((pixel (/ (read-byte stream) 255)))
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
  (make-instance 'function-sampler
                 :max-n-samples max-n
                 :generator (let ((g (make-random-generator images)))
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
        (displacement (mat-displacement mat))
        (one (coerce-to-ctype 1 :ctype (mat-ctype mat))))
    (with-facets ((a (mat 'backing-array :direction :io)))
      (loop for sample in samples
            for row upfrom 0
            do (destructuring-bind (image &key discard-label-p sample-visible-p)
                   sample
                 (declare (ignore sample-visible-p))
                 (unless discard-label-p
                   (setf (aref a (+ displacement
                                    (* row n-columns)
                                    (image-label image)))
                         one)))))))

;;; Create a list of indices suitable as TARGET for ->SOFTMAX-XE-LOSS.
(defun label-target-list (samples)
  (loop for sample in samples
        collect (destructuring-bind (image &key discard-label-p
                                     sample-visible-p)
                    sample
                  (declare (ignore sample-visible-p))
                  (if discard-label-p
                      nil
                      (image-label image)))))

(defun make-mnist-label-monitors (model &key attributes)
  (make-label-monitors
   model
   :label-index-fn #'sample-image-label-index
   :label-index-distribution-fn #'sample-image-label-index-distribution
   :attributes attributes))

(defun sample-image-label-index (sample)
  (image-label (car sample)))

(defun sample-image-label-index-distribution (sample)
  (let ((d (make-array 10 :initial-element 0)))
    (setf (aref d (image-label (car sample))) 1)
    d))


;;;; Logging

(defclass mnist-base-optimizer ()
  ((training :initarg :training :reader training)
   (test :initarg :test :reader test)))

(defun log-training-period (optimizer learner)
  (declare (ignore learner))
  (length (training optimizer)))

(defun log-test-period (optimizer learner)
  (declare (ignore learner))
  (length (training optimizer)))


;;;; DBN

(defclass mnist-dbn (dbn) ())

(defclass mnist-rbm (rbm) ())

(defmethod set-input (samples (rbm mnist-rbm))
  (let ((inputs (find 'inputs (visible-chunks rbm) :key #'name)))
    (when inputs
      (clamp-images samples (nodes inputs)))))

(defclass mnist-rbm-optimizer (mnist-base-optimizer segmented-gd-optimizer)
  ())

(defclass mnist-rbm-cd-learner (rbm-cd-learner)
  ((optimizer :initarg :optimizer :reader optimizer)))

(defmethod n-gibbs ((learner mnist-rbm-cd-learner))
  (let ((x (slot-value learner 'n-gibbs)))
    (if (integerp x)
        x
        (funcall x (optimizer learner)))))

(defun log-rbm-test-error (optimizer learner)
  (when (zerop (n-instances optimizer))
    (report-optimization-parameters optimizer learner))
  (log-msg "test at n-instances: ~S~%" (n-instances optimizer))
  (let ((rbm (rbm learner)))
    (log-padded
     (append
      (monitor-rbm-cesc-accuracy rbm (make-tiny-sampler (training optimizer))
                                 '(:event "pred." :dataset "train+"))
      (monitor-rbm-cesc-accuracy rbm (make-tiny-sampler (training optimizer)
                                                        :discard-label-p t)
                                 '(:event "pred." :dataset "train"))
      (monitor-dbn-mean-field-reconstructions
       (make-sampler (test optimizer)) (dbn rbm)
       (make-reconstruction-monitors
        (dbn rbm) :attributes '(:event "pred." :dataset "test+")))
      (monitor-rbm-cesc-accuracy rbm (make-sampler (test optimizer)
                                                   :discard-label-p t)
                                 '(:event "pred." :dataset "test"))))
    (log-mat-room)
    (log-msg "---------------------------------------------------~%")))

(defun monitor-rbm-cesc-accuracy (rbm sampler attributes)
  (if (dbn rbm)
      (monitor-dbn-mean-field-reconstructions
       sampler (dbn rbm)
       (make-mnist-label-monitors (dbn rbm) :attributes attributes)
       :set-visible-p t)
      (monitor-bm-mean-field-reconstructions
       sampler rbm
       (make-mnist-label-monitors rbm :attributes attributes)
       :set-visible-p t)))


;;;; DBN training

(defclass mnist-rbm-segment-optimizer (batch-gd-optimizer)
  ((n-instances-in-epoch
    :initarg :n-instances-in-epoch
    :reader n-instances-in-epoch)))

(defmethod learning-rate ((optimizer mnist-rbm-segment-optimizer))
  (let ((x (slot-value optimizer 'learning-rate)))
    (if (numberp x)
        x
        (funcall x optimizer))))

(defmethod momentum ((optimizer mnist-rbm-segment-optimizer))
  (if (< (n-instances optimizer) (* 5 (n-instances-in-epoch optimizer)))
      0.5
      0.9))

(defun init-mnist-dbn (dbn &key stddev (start-level 0))
  (loop for i upfrom start-level
        for rbm in (subseq (rbms dbn) start-level) do
          (flet ((this (x)
                   (if (listp x)
                       (elt x i)
                       x)))
            (do-clouds (cloud rbm)
              (if (conditioning-cloud-p cloud)
                  (fill! 0 (weights cloud))
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
      (setf (n-rbms dbn) (1+ i))
      (flet ((this (x)
               (if (and x (listp x))
                   (elt x i)
                   x)))
        (let* ((n (length training))
               (optimizer
                 (make-instance 'mnist-rbm-optimizer
                                :training training
                                :test test
                                :segmenter
                                (lambda (cloud)
                                  (make-instance 'mnist-rbm-segment-optimizer
                                                 :n-instances-in-epoch n
                                                 :learning-rate
                                                 (this learning-rate)
                                                 :weight-decay
                                                 (if (conditioning-cloud-p
                                                      cloud)
                                                     0
                                                     (this decay))
                                                 :batch-size 100)))))
          
          (minimize
           (monitor-optimization-periodically
            optimizer
            '((:fn log-rbm-test-error :period log-test-period)
              (:fn reset-optimization-monitors
               :period log-training-period
               :last-eval 0)))
           (make-instance
            'mnist-rbm-cd-learner
            :optimizer optimizer
            :rbm rbm
            :visible-sampling (this visible-sampling)
            :n-gibbs (this n-gibbs)
            :monitors
            (let ((attributes '(:event "train" :dataset "train+")))
              (append (make-reconstruction-monitors rbm :attributes attributes)
                      (make-mnist-label-monitors rbm :attributes attributes))))
           :dataset
           (make-sampler training
                         :n-epochs n-epochs
                         :sample-visible-p (this visible-sampling)))))))


;;;; BPN

(defclass mnist-bpn (bpn) ())

(defmethod set-input (samples (bpn mnist-bpn))
  (let* ((inputs (or (find-clump (chunk-lump-name 'inputs nil) bpn :errorp nil)
                     (find-clump 'inputs bpn)))
         (prediction (find-clump 'prediction bpn)))
    (clamp-images samples (nodes inputs))
    (setf (target prediction) (label-target-list samples))))

(defun make-softmax (inputs)
  (->softmax-xe-loss (->activation inputs :name 'prediction :size 10)
                     :name 'prediction))

(defun prediction-weight-p (lump)
  (let ((name (name lump)))
    (and (listp name)
         (= 2 (length name))
         (eq 'prediction (second name)))))

(defun make-bpn (defs chunk-name &key class initargs)
  (let ((bpn-def `(build-fnn (:class ',class
                              :max-n-stripes 1000
                              :initargs ',initargs)
                    ,@defs)))
    (log-msg "bpn def:~%~S~%" bpn-def)
    (let* ((bpn (eval bpn-def))
           (name (chunk-lump-name chunk-name nil))
           (lump (find-clump name bpn)))
      (let ((mgl-bp::*bpn-being-built* bpn))
        (make-softmax (list lump)))
      bpn)))


;;;; BPN training

(defclass mnist-bp-optimizer (mnist-base-optimizer) ())

(defun log-bpn-test-error (optimizer learner)
  (when (zerop (n-instances optimizer))
    (report-optimization-parameters optimizer learner))
  (log-msg "test at n-instances: ~S~%" (n-instances optimizer))
  (log-padded
   (let ((bpn (bpn learner)))
     (monitor-bpn-results (make-sampler (test optimizer)) bpn
                          (make-mnist-label-monitors
                           bpn :attributes '(:event "pred." :dataset "test")))))
  (log-mat-room)
  (log-msg "---------------------------------------------------~%"))

(defun init-bpn-lump-weights (lump stddev)
  (gaussian-random! (nodes lump) :stddev stddev))

;;;; BPN CG training

(defclass mnist-cg-bp-optimizer (mnist-bp-optimizer cg-optimizer) ())

(defun train-mnist-bpn-cg (bpn training test &key
                           (batch-size (min (length training) 1000))
                           (n-softmax-epochs 5) n-epochs)
  (log-msg "Starting to train the softmax layer of BPN~%")
  (minimize (monitor-optimization-periodically
             (make-instance 'mnist-cg-bp-optimizer
                            :training training
                            :test test
                            :cg-args (list :max-n-line-searches 3)
                            :batch-size batch-size
                            :on-cg-batch-done '(log-cg-batch-done)
                            :segment-filter #'prediction-weight-p)
             '((:fn log-bpn-test-error :period log-test-period)))
            (make-instance 'bp-learner :bpn bpn)
            :dataset
            (make-sampler training :n-epochs n-softmax-epochs))
  (log-msg "Starting to train the whole BPN~%")
  (minimize (monitor-optimization-periodically
             (make-instance 'mnist-cg-bp-optimizer
                            :training training
                            :test test
                            :cg-args (list :max-n-line-searches 3)
                            :batch-size batch-size
                            :on-cg-batch-done '(log-cg-batch-done))
             '((:fn log-bpn-test-error :period log-test-period)))
            (make-instance 'bp-learner :bpn bpn)
            :dataset
            (make-sampler training :n-epochs (- n-epochs n-softmax-epochs))))


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
      (initialize-fnn-from-bm bpn dbn inits)
      (map nil (lambda (lump)
                 (when (prediction-weight-p lump)
                   (init-bpn-lump-weights lump 0.1)))
           (clumps bpn))
      bpn)))

(defun train-mnist/1 (&key training test load-dbn-p quick-run-p dropoutp
                      dbn-var bpn-var dbn-filename bpn-filename)
  (with-example-log ()
    (log-msg "TRAIN-MNIST/1 :DROPOUTP ~S~%" dropoutp)
    (let ((dbn nil)
          (bpn nil))
      (cond (load-dbn-p
             (repeatably ()
               (setq* (dbn dbn-var) (make-instance 'mnist-dbn/1))
               (load-weights dbn-filename dbn)
               (log-msg "Loaded DBN~%")
               (log-padded
                (monitor-dbn-mean-field-reconstructions
                 (make-sampler test) dbn
                 (make-reconstruction-monitors dbn)
                 :set-visible-p t))))
            (t
             (repeatably ()
               (setq* (dbn dbn-var) (make-instance 'mnist-dbn/1))
               (init-mnist-dbn dbn :stddev 0.1)
               (train-mnist-dbn dbn training test
                                :n-epochs (if quick-run-p 2 50) :n-gibbs 1
                                :start-level 0 :learning-rate 0.1
                                :decay 0.0002 :visible-sampling nil)
               (unless quick-run-p
                 (save-weights dbn-filename dbn)))))
      (repeatably ()
        (setq* (bpn bpn-var) (unroll-mnist-dbn/1 dbn))
        (cond (dropoutp
               (train-mnist-bpn-gd bpn training test
                                   :n-softmax-epochs (if quick-run-p 1 10)
                                   :n-epochs (if quick-run-p 2 1000)
                                   :learning-rate 1
                                   :learning-rate-decay 0.998
                                   :l2-upper-bound nil
                                   :set-dropout-p t
                                   :rescale-on-dropout-p t)
               (unless quick-run-p
                 (save-weights bpn-filename bpn)))
              (t
               (train-mnist-bpn-cg bpn training test
                                   :n-softmax-epochs (if quick-run-p 1 5)
                                   :n-epochs (if quick-run-p 2 37))
               (unless quick-run-p
                 (save-weights bpn-filename bpn)))))
      (values bpn dbn))))


;;;; Code for the DBN to DBM to BPN approach (paper [2]) and also for
;;;; dropout training with the bpn (paper [3]) with :DROPOUT T.

(defclass mnist-rbm/2 (mnist-rbm) ())
(defclass mnist-bpn/2 (mnist-bpn fnn-clamping-cache) ())

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
                            (make-instance 'softmax-label-chunk :name 'label
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

(defclass mnist-dbm-optimizer (mnist-base-optimizer segmented-gd-optimizer)
  ())

(defun log-dbm-test-error (optimizer learner)
  (when (zerop (n-instances optimizer))
    (report-optimization-parameters optimizer learner))
  (log-msg "test at n-instances: ~S~%" (n-instances optimizer))
  (let ((dbm (bm learner)))
    (log-padded
     (append
      (monitor-dbm-cesc-accuracy dbm (make-tiny-sampler (training optimizer))
                                 '(:event "pred." :dataset "train+"))
      (monitor-dbm-cesc-accuracy dbm (make-tiny-sampler (training optimizer)
                                                        :discard-label-p t)
                                 '(:event "pred." :dataset "train"))
      ;; This is too time consuming for little benefit.
      #+nil
      (monitor-bm-mean-field-results
       (make-sampler (test optimizer)) dbm
       :monitors (make-dbm-reconstruction-rmse-monitors
                  dbm :attributes '(:event "pred." :dataset "test+")))
      (monitor-dbm-cesc-accuracy dbm (make-sampler (test optimizer)
                                                   :discard-label-p t)
                                 '(:event "pred." :dataset "test")))))
  (log-mat-room)
  (log-msg "---------------------------------------------------~%"))

(defun monitor-dbm-cesc-accuracy (dbm sampler attributes)
  (monitor-bm-mean-field-bottom-up
   sampler dbm (make-mnist-label-monitors dbm :attributes attributes)))

(defclass mnist-dbm-segment-optimizer (batch-gd-optimizer)
  ((n-instances-in-epoch
    :initarg :n-instances-in-epoch
    :reader n-instances-in-epoch)))

(defmethod learning-rate ((optimizer mnist-dbm-segment-optimizer))
  ;; This is adjusted for each batch. Ruslan's code adjusts it per
  ;; epoch.
  (/ (slot-value optimizer 'learning-rate)
     (min 10
          (expt 1.000015
                (* (/ (n-instances optimizer) (n-instances-in-epoch optimizer))
                   600)))))

(defmethod momentum ((optimizer mnist-dbm-segment-optimizer))
  (if (< (n-instances optimizer) (* 5 (n-instances-in-epoch optimizer)))
      0.5
      0.9))

(defmethod initialize-gradient-source* ((optimizer mnist-dbm-optimizer) learner
                                        weights dataset)
  (when (next-method-p)
    (call-next-method))
  (set-input (list-samples (make-sampler (training optimizer)
                                         :sample-visible-p t)
                           (n-particles learner))
             (persistent-chains learner))
  (up-dbm (persistent-chains learner)))

(defun train-mnist-dbm (dbm training test &key (n-epochs 500))
  (log-msg "Starting to train DBM.~%")
  (let ((n (length training)))
    (minimize (monitor-optimization-periodically
               (make-instance 'mnist-dbm-optimizer
                              :training training
                              :test test
                              :segmenter
                              (lambda (cloud)
                                (make-instance 'mnist-dbm-segment-optimizer
                                               :n-instances-in-epoch n
                                               :learning-rate 0.001
                                               :weight-decay
                                               (if (conditioning-cloud-p cloud)
                                                   0
                                                   0.0002)
                                               :batch-size 100)))
               '((:fn log-dbm-test-error :period log-test-period)
                 (:fn reset-optimization-monitors
                  :period log-training-period
                  :last-eval 0)))
              (make-instance
               'bm-pcd-learner
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
                    :sparsity (if (eq 'f2 (name chunk))
                                  0.1
                                  0.2)
                    :cost 0.001
                    :damping 0.9)))
               :monitors
               (let ((attributes '(:event "train" :dataset "train+")))
                 (append
                  (make-reconstruction-monitors dbm :attributes attributes)
                  (make-mnist-label-monitors dbm :attributes attributes))))
              :dataset
              (make-sampler training :n-epochs n-epochs :sample-visible-p t))))

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
      (setf (slot-value (find-clump `((,name prediction) :activation)
                                    (find-clump '(prediction :activation) bpn))
                        'mgl-bp::transpose-weights-p)
            t)
      (initialize-fnn-from-bm bpn dbm inits)
      (when use-label-weights
        (initialize-fnn-from-bm (find-clump '(prediction :activation) bpn) dbm
                                (list
                                 '(:cloud-name (label c2)
                                   :weight-name (:bias prediction))
                                 '(:cloud-name (label f2)
                                   :weight-name ((:chunk f2) prediction)))))
      bpn)))

(defun train-mnist/2 (&key training test load-dbn-p load-dbm-p dropoutp
                      quick-run-p dbn-var dbm-var bpn-var
                      dbn-filename dbm-filename bpn-filename)
  (with-example-log ()
    (log-msg "TRAIN-MNIST/2 :DROPOUTP ~S~%" dropoutp)
    (let ((dbn nil)
          (dbm nil)
          (bpn nil))
      (flet ((train-dbn ()
               (repeatably ()
                 (init-mnist-dbn dbn :stddev '(0.001 0.01) :start-level 0)
                 (train-mnist-dbn
                  dbn training test
                  :start-level 0
                  :n-epochs (if quick-run-p 2 100)
                  :n-gibbs (list 1
                                 (lambda (optimizer)
                                   (ceiling (1+ (n-instances optimizer))
                                            (* 20 (length training)))))
                  :learning-rate (list 0.05
                                       (lambda (optimizer)
                                         (/ 0.05
                                            (ceiling
                                             (1+ (n-instances optimizer))
                                             (* 20 (length training))))))
                  :decay 0.001
                  :visible-sampling t))
               (unless quick-run-p
                 (save-weights dbn-filename dbn)))
             (train-dbm ()
               (repeatably ()
                 (train-mnist-dbm dbm training test
                                  :n-epochs (if quick-run-p 1 500)))
               (unless quick-run-p
                 (save-weights dbm-filename dbm)))
             (make-dbm ()
               (repeatably ()
                 (setq* (dbm dbm-var) (make-mnist-dbm))))
             (make-dbn ()
               (repeatably ()
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
        (repeatably ()
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
                        (make-instance
                         'periodic-fn :period 1000
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
                                     :learning-rate 1
                                     :learning-rate-decay 0.998
                                     :l2-upper-bound nil
                                     :set-dropout-p t
                                     :rescale-on-dropout-p t)
                 (unless quick-run-p
                   (save-weights bpn-filename bpn)))
                (t
                 (train-mnist-bpn-cg bpn training test :batch-size 10000
                                     :n-softmax-epochs (if quick-run-p 1 5)
                                     :n-epochs (if quick-run-p 1 100))
                 (unless quick-run-p
                   (save-weights bpn-filename bpn)))))))))

(defmethod negative-phase (batch (learner bm-pcd-learner)
                           (optimizer mnist-dbm-optimizer) multiplier)
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
       learner optimizer
       (* multiplier (/ (length batch) (n-stripes bm)))))))


;;;; Code for the plain dropout backpropagation network with rectified
;;;; linear units (paper [3])

(defclass mnist-bpn-gd-optimizer (mnist-bp-optimizer segmented-gd-optimizer)
  ())

(defclass mnist-bpn-gd-segment-optimizer (batch-gd-optimizer)
  ((n-instances-in-epoch
    :initarg :n-instances-in-epoch
    :reader n-instances-in-epoch)
   (n-epochs-to-reach-final-momentum
    :initarg :n-epochs-to-reach-final-momentum
    :reader n-epochs-to-reach-final-momentum)
   (learning-rate-decay
    :initform 0.998
    :initarg :learning-rate-decay
    :accessor learning-rate-decay)))

(defmethod learning-rate ((optimizer mnist-bpn-gd-segment-optimizer))
  (* (expt (learning-rate-decay optimizer)
           (/ (n-instances optimizer)
              (n-instances-in-epoch optimizer)))
     (- 1 (momentum optimizer))
     (slot-value optimizer 'learning-rate)))

(defmethod momentum ((optimizer mnist-bpn-gd-segment-optimizer))
  (let ((n-epochs-to-reach-final (n-epochs-to-reach-final-momentum optimizer))
        (initial 0.5)
        (final 0.99)
        (epoch (/ (n-instances optimizer) (n-instances-in-epoch optimizer))))
    (if (< epoch n-epochs-to-reach-final)
        (let ((weight (/ epoch n-epochs-to-reach-final)))
          (+ (* initial (- 1 weight))
             (* final weight)))
        final)))

(defun make-grouped-segmenter (group-name-fn segmenter)
  (let ((group-name-to-optimizer (make-hash-table :test #'equal)))
    (lambda (segment)
      (let ((group-name (funcall group-name-fn segment)))
        (or (gethash group-name group-name-to-optimizer)
            (setf (gethash group-name group-name-to-optimizer)
                  (funcall segmenter segment)))))))

(defun make-dwim-grouped-segmenter (segmenter)
  (make-grouped-segmenter #'weight-lump-target-name segmenter))

(defun weight-lump-target-name (lump)
  (let ((name (name lump)))
    (assert (listp name))
    (assert (= 2 (length name)))
    (if (eq (first name) :cloud)
        (second (second name))
        (second name))))

(defun train-mnist-bpn-gd (bpn training test &key (n-softmax-epochs 5)
                           (n-epochs 200) l2-upper-bound
                           learning-rate learning-rate-decay
                           set-dropout-p
                           input-weight-penalty
                           rescale-on-dropout-p
                           (batch-size 100))
  (setf (max-n-stripes bpn) batch-size)
  (flet ((make-optimizer (lump &key softmaxp)
           (let ((optimizer (make-instance
                             'mnist-bpn-gd-segment-optimizer
                             :n-instances-in-epoch (length training)
                             :n-epochs-to-reach-final-momentum
                             (min 500
                                  (/ (if softmaxp n-softmax-epochs n-epochs)
                                     2))
                             :learning-rate learning-rate
                             :learning-rate-decay learning-rate-decay
                             :weight-penalty (if (and input-weight-penalty
                                                      (member (name lump)
                                                              '((inputs f1))
                                                              :test #'name=))
                                                 input-weight-penalty
                                                 0)
                             :batch-size batch-size)))
             (when l2-upper-bound
               (arrange-for-renormalizing-activations
                bpn optimizer l2-upper-bound))
             #+nil
             (when (member (name lump) '((inputs f1))
                           :test #'name=)
               (push (let ((mask (make-sparse-column-mask (nodes lump) 392)))
                       (.*! mask (nodes lump))
                       (lambda ()
                         (.*! mask (nodes lump))))
                     (after-update-hook optimizer)))
             optimizer))
         (make-segmenter (fn)
           (let ((dwim (make-dwim-grouped-segmenter fn)))
             (lambda (lump)
               (if (and l2-upper-bound
                        (not (and input-weight-penalty
                                  (member (name lump) '((inputs f1) (:bias f1))
                                          :test #'name=))))
                   (funcall dwim lump)
                   (funcall fn lump))))))
    (unless (zerop n-softmax-epochs)
      (log-msg "Starting to train the softmax layer of BPN~%")
      (minimize (monitor-optimization-periodically
                 (make-instance 'mnist-bpn-gd-optimizer
                                :training training
                                :test test
                                :segmenter
                                (make-segmenter
                                 (lambda (lump)
                                   (when (prediction-weight-p lump)
                                     (make-optimizer lump :softmaxp t)))))
                 '((:fn log-bpn-test-error :period log-test-period)
                   (:fn reset-optimization-monitors
                    :period log-training-period
                    :last-eval 0)))
                (make-instance
                 'bp-learner
                 :bpn bpn
                 :monitors (make-mnist-label-monitors
                            bpn :attributes '(:event "train"
                                              :dataset "train+")))
                :dataset
                (make-sampler training :n-epochs n-softmax-epochs)))
    (when set-dropout-p
      (map nil (lambda (lump)
                 (when (and (typep lump '->dropout)
                            (not (typep lump '->input)))
                   (log-msg "dropout ~A ~A~%" lump 0.5)
                   (if rescale-on-dropout-p
                       (set-dropout-and-rescale-activation-weights
                        lump 0.5 bpn)
                       (setf (slot-value lump 'dropout) 0.5)))
                 (when (and (typep lump '->input)
                            (member (name lump) '((:chunk inputs)
                                                  inputs)
                                    :test #'name=))
                   (log-msg "dropout ~A ~A~%" lump 0.2)
                   (if rescale-on-dropout-p
                       (set-dropout-and-rescale-activation-weights
                        lump 0.2 bpn)
                       (setf (slot-value lump 'dropout) 0.2))))
           (clumps bpn)))
    (unless (zerop n-epochs)
      (log-msg "Starting to train the whole BPN~%")
      (minimize (monitor-optimization-periodically
                 (make-instance 'mnist-bpn-gd-optimizer
                                :training training
                                :test test
                                :segmenter
                                (make-segmenter #'make-optimizer))
                 '((:fn log-bpn-test-error :period log-test-period)
                   (:fn reset-optimization-monitors
                    :period log-training-period
                    :last-eval 0)))
                (make-instance
                 'bp-learner
                 :bpn bpn
                 :monitors (make-mnist-label-monitors
                            bpn :attributes '(:event "train"
                                              :dataset "train+")))
                :dataset (make-sampler training :n-epochs n-epochs)))))

(defun build-rectified-mnist-bpn (&key (n-units-1 1200) (n-units-2 1200)
                                  (n-units-3 1200))
  (build-fnn (:class 'mnist-bpn :max-n-stripes 100)
    (inputs (->input :size 784 :dropout 0.2))
    (f1-activations (->activation inputs :name 'f1 :size n-units-1))
    (f1* (->relu f1-activations))
    (f1 (->dropout f1*))
    (f2-activations (->activation f1 :name 'f2 :size n-units-2))
    (f2* (->relu f2-activations))
    (f2 (->dropout f2*))
    (f3-activations (->activation f2 :name 'f3 :size n-units-3))
    (f3* (->relu f3-activations))
    (f3 (->dropout f3*))
    (prediction (make-softmax f3))))

;;; Return a matrix of the same shape as MAT that's zero everywhere,
;;; except in at most N randomly chosen positions in each column where
;;; it's one.
(defun make-sparse-column-mask (mat n)
  (let ((mask (make-mat (mat-dimensions mat) :ctype (mat-ctype mat)))
        (one (coerce-to-ctype 1 :ctype (mat-ctype mat))))
    (destructuring-bind (n-rows n-columns) (mat-dimensions mat)
      (with-facets ((mask* (mask 'backing-array :direction :io)))
        (loop for column below n-columns do
          (loop repeat n
                do (setf (aref mask* (+ (* (random n-rows) n-columns)
                                        column))
                         one)))))
    mask))

(defun init-bpn-weights (bpn &key stddev)
  (map-segments (lambda (weight)
                  (gaussian-random! (nodes weight) :stddev stddev))
                bpn))

(defun train-mnist/3 (&key training test quick-run-p bpn-var bpn-filename)
  (with-example-log ()
    (repeatably ()
      (log-msg "TRAIN-MNIST/3~%")
      (let ((bpn nil))
        (setq* (bpn bpn-var) (build-rectified-mnist-bpn))
        (init-bpn-weights bpn :stddev 0.01)
        (train-mnist-bpn-gd bpn training test
                            :n-softmax-epochs 0
                            :n-epochs (if quick-run-p 2 3000)
                            :learning-rate 1
                            :learning-rate-decay 0.998
                            :l2-upper-bound (sqrt 15)
                            :input-weight-penalty 0.000001
                            :batch-size 96)
        (when (and bpn-filename (not quick-run-p))
          (save-weights bpn-filename bpn))))))


;;;; Maxout

(defun build-maxout-mnist-bpn (&key (n-units-1 1200) (n-units-2 1200)
                               (group-size 5))
  (build-fnn (:class 'mnist-bpn)
    (inputs (->input :size 784 :dropout 0.2))
    (f1-activations (->activation inputs :name 'f1 :size n-units-1))
    (f1* (->max f1-activations :group-size group-size))
    (f1 (->dropout f1*))
    (f2-activations (->activation f1 :name 'f2 :size n-units-2))
    (f2* (->max f2-activations :group-size group-size))
    (f2 (->dropout f2*))
    (prediction (make-softmax f2))))

(defun train-mnist/4 (&key training test quick-run-p bpn-var bpn-filename)
  (with-example-log ()
    (repeatably ()
      (log-msg "TRAIN-MNIST/4~%")
      (let ((bpn nil))
        (setq* (bpn bpn-var) (build-maxout-mnist-bpn))
        (init-bpn-weights bpn :stddev 0.01)
        (train-mnist-bpn-gd bpn training test
                            :n-softmax-epochs 0
                            :n-epochs (if quick-run-p 2 3000)
                            :learning-rate 1
                            :learning-rate-decay 0.998
                            :l2-upper-bound (sqrt 3.75)
                            :input-weight-penalty 0.000001
                            :batch-size 96)
        (when (and bpn-filename (not quick-run-p))
          (save-weights bpn-filename bpn))))))


;;;; Max-channel

(defun build-max-channel-mnist-bpn (&key (n-units-1 1200) (n-units-2 1200)
                                    (group-size 2))
  (build-fnn (:class 'mnist-bpn)
    (inputs (->input :size 784 :dropout 0.2))
    (f1-activations (->activation inputs :name 'f1 :size n-units-1))
    (f1* (->max-channel f1-activations :group-size group-size))
    (f1 (->dropout f1*))
    (f2-activations (->activation f1 :name 'f2 :size n-units-2))
    (f2* (->max-channel f2-activations :group-size group-size))
    (f2 (->dropout f2*))
    (f3-activations (->activation f2 :name 'f3 :size n-units-2))
    (f3* (->max-channel f3-activations :group-size group-size))
    (f3 (->dropout f3*))
    (prediction (make-softmax f3))))

(defun train-mnist/5 (&key training test quick-run-p bpn-var bpn-filename)
  (with-example-log ()
    (repeatably ()
      (log-msg "TRAIN-MNIST/5~%")
      (let ((bpn nil))
        (setq* (bpn bpn-var) (build-max-channel-mnist-bpn))
        (init-bpn-weights bpn :stddev 0.01)
        (train-mnist-bpn-gd bpn training test
                            :n-softmax-epochs 0
                            :n-epochs (if quick-run-p 2 3000)
                            :learning-rate 1
                            :learning-rate-decay 0.998
                            :l2-upper-bound (sqrt 3.75)
                            :input-weight-penalty 0.000001
                            :batch-size 96)
        (when (and bpn-filename (not quick-run-p))
          (save-weights bpn-filename bpn))))))


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
  (train-mnist/1 :training (training-images) :test (test-images) :dropoutp t
                 :quick-run-p t)
  (train-mnist/2 :training (training-images) :test (test-images)
                 :quick-run-p t)
  (train-mnist/2 :training (training-images) :test (test-images) :dropoutp t
                 :quick-run-p t)
  (train-mnist/3 :training (training-images) :test (test-images)
                 :quick-run-p t)
  (train-mnist/4 :training (training-images) :test (test-images)
                 :quick-run-p t)
  (train-mnist/5 :training (training-images) :test (test-images)
                 :quick-run-p t))

#|

(let ((*example-log-file*
        (merge-pathnames "baseline/mnist-quick-test.log" *example-dir*)))
  (ignore-errors (delete-file *example-log-file*))
  (run-quick-test))

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
                 :bpn-var '*bpn/4* :bpn-filename *mnist-4-bpn-filename*)
  (train-mnist/5 :training (training-images) :test (test-images)))

(with-example-log ()
  (mgl-resample:cross-validate (concatenate 'vector (training-images)
                                            (test-images))
                               (lambda (fold test training)
                                 (log-msg "Fold ~S~%" fold)
                                 (train-mnist/1 :training training :test test))
                               :n-folds 7
                               :split-fn (lambda (seq fold n-folds)
                                           (mgl-resample:split-stratified
                                            seq fold n-folds
                                            :key #'image-label))
                               :pass-fold t))

(with-example-log ()
  (let ((*default-mat-ctype* :float))
    (train-mnist/5 :training (training-images) :test (test-images))))

(with-example-log ()
  ;; This only works with doubles for now, because boltzmann
  ;; machines do not yet support single floats.
  (let ((*default-mat-ctype* :double))
    (run-quick-test)))

(progn
  (makunbound '*training-images*)
  (makunbound '*test-images*))

/3 with 1000 epochs

(alexandria:mean '(98.82 99.02 98.89 98.78 98.70 98.89 98.89)) => 98.85571

/4 with 1000 epochs

(alexandria:mean '(98.83 99.02 98.85 98.80 98.67 98.97 98.88)) => 98.86

/5 with 1000 epochs

(alexandria:mean '(98.96 99.12 99.02 98.90 98.88 99.03 99.07)) => 98.99715

/3:

(alexandria:mean '(98.88 99.11 98.84 98.82 98.77 98.90 99.07)) => 98.91286

/3+L1:

(alexandria:mean '(98.81 99.15 98.97 98.85 98.86 99.13 99.04)) => 98.972855

/4:

(alexandria:mean '(98.89 99.13 98.84 98.84 98.71 98.97 99.02)) => 98.91429

/4+L1:

(alexandria:mean '(98.92 99.08 99.08 99.00 98.88 98.82 99.02)) => 98.971436

/5:

(alexandria:mean '(98.92 99.26 99.01 98.91 98.82 99.05 99.13)) => 99.01429

/5+L1:

(alexandria:mean '(99.04 99.28 99.10 98.92 99.01 99.14 99.16)) => 99.09286

2014-09-20 12:02:26: n-inputs: 60000000
2014-09-20 12:02:26: cuda mats: 70307, copies: h->d: 800008, d->h: 2085009
2014-09-20 12:02:26: bpn train: training classification accuracy: 99.81% (60000)
2014-09-20 12:02:26: bpn train: training cross entropy: 5.54801d-3 (60000.0d0)
2014-09-20 12:02:26: n-inputs: 60000000
2014-09-20 12:02:26: cuda mats: 70307, copies: h->d: 800008, d->h: 2085009
2014-09-20 12:02:26: bpn test: test classification accuracy: 99.07% (10000)
2014-09-20 12:02:26: bpn test: test cross entropy: 3.94829d-2 (10000.0d0)
2014-09-20 12:02:26: ---------------------------------------------------


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
