;;;; Code for the MNIST handwritten digit recognition challange.
;;;;
;;;; See papers by Geoffrey Hinton:
;;;;
;;;;   http://www.cs.toronto.edu/~hinton/MatlabForSciencePaper.html
;;;;
;;;;   http://www.cs.toronto.edu/~hinton/absps/montrealTR.pdf
;;;;
;;;; Download the four files from http://yann.lecun.com/exdb/mnist and
;;;; gunzip them. Set *MNIST-DIR* to point to their directory and call
;;;; TRAIN-MNIST.
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
;;;; There is support for gradient descent training for the BPN but
;;;; ultimately its results are worse.

(in-package :mgl-example-mnist-dbm)

(defparameter *mnist-dir*
  (merge-pathnames "mnist-data/" *example-dir*)
  "Set this to the directory where the uncompressed mnist files reside.")

(defstruct image
  (label nil :type (integer 0 10))
  (array nil :type flt-vector))


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
    a))

(defun print-image-array (a)
  (dotimes (y 28)
    (dotimes (x 28)
      (princ (if (< 0.5 (aref a (+ (* y 28) x))) #\O #\.)))
    (terpri)))

#+nil
(dotimes (i 100)
  (print-image-array (image-array (aref *training-images* i)))
  (print (image-label (aref *training-images* i)))
  (terpri))

(defun read-image-arrays (stream)
  (read-magic-or-lose stream 2051)
  (let ((n (read-low-endian-ub32 stream)))
    (read-magic-or-lose stream 28)
    (read-magic-or-lose stream 28)
    (coerce (loop repeat n collect (read-image-array stream))
            'vector)))

(defun load-training (&optional (mnist-dir *mnist-dir*))
  (map 'vector
       (lambda (label array)
         (make-image :label label :array array))
       (with-open-file (s (merge-pathnames "train-labels-idx1-ubyte" mnist-dir)
                        :element-type 'unsigned-byte)
         (read-image-labels s))
       (with-open-file (s (merge-pathnames "train-images-idx3-ubyte" mnist-dir)
                        :element-type 'unsigned-byte)
         (read-image-arrays s))))

(defun load-test (&optional (mnist-dir *mnist-dir*))
  (map 'vector
       (lambda (label array)
         (make-image :label label :array array))
       (with-open-file (s (merge-pathnames "t10k-labels-idx1-ubyte" mnist-dir)
                        :element-type 'unsigned-byte)
         (read-image-labels s))
       (with-open-file (s (merge-pathnames "t10k-images-idx3-ubyte" mnist-dir)
                        :element-type 'unsigned-byte)
         (read-image-arrays s))))

(defvar *training-images*)
(defvar *test-images*)


;;;; Sampling, clamping

(defun make-sampler (images &key max-n)
  (make-instance 'counting-function-sampler
                 :max-n-samples max-n
                 :sampler (make-random-generator images)))

(defun clamp-array (image array start)
  (declare (type flt-vector array)
           (optimize (speed 3)))
  (replace array (image-array image) :start1 start))

(defun clamp-striped-nodes (samples striped)
  (let ((nodes (storage (nodes striped))))
    (loop for sample in samples
          for stripe upfrom 0
          do (with-stripes ((stripe striped start))
               (clamp-array sample nodes start)))))


;;;;

(defclass mnist-logging-trainer (logging-trainer) ())

(defmethod log-training-period ((trainer mnist-logging-trainer) learner)
  10000)

(defmethod log-test-period ((trainer mnist-logging-trainer) learner)
  (length *training-images*))


;;;; DBM

(defclass mnist-dbm (dbm)
  ((layers :initform (list
                      (list (make-instance 'constant-chunk :name 'c0)
                            (make-instance 'sigmoid-chunk :name 'inputs
                                           :size (* 28 28)))
                      (list (make-instance 'constant-chunk :name 'c1)
                            (make-instance 'sigmoid-chunk :name 'f1
                                           :size 500))
                      (list (make-instance 'constant-chunk :name 'c2)
                            (make-instance 'sigmoid-chunk :name 'f2
                                           :size 1000))))))

(defun make-mnist-dbm ()
  (make-instance 'mnist-dbm :max-n-stripes 100))

(defmethod mgl-train:set-input (images (dbm mnist-dbm))
  (clamp-striped-nodes images (mgl-bm:find-chunk 'inputs dbm)))

(defclass mnist-dbm-trainer (mnist-logging-trainer bm-pcd-trainer)
  ((counter :initform (make-instance 'rmse-counter) :reader counter)))

(defmethod log-training-error ((trainer mnist-dbm-trainer) (dbm mnist-dbm))
  (let ((counter (counter trainer))
        (n-inputs (n-inputs trainer)))
    (log-msg "TRAINING RMSE: ~,5F (~D)~%"
             (or (get-error counter) #.(flt 0))
             n-inputs)
    (reset-counter counter)))

(defmethod log-test-error ((trainer mnist-dbm-trainer) (dbm mnist-dbm))
  (let ((sampler (make-sampler *test-images*
                               :max-n 1000
                               #+nil
                               (length *test-images*)))
        (counter (make-instance 'rmse-counter)))
    (let ((max-n-stripes (max-n-stripes dbm)))
      (loop until (finishedp sampler) do
            (let ((samples (sample-batch sampler max-n-stripes)))
              (set-input samples dbm)
              (up-dbm dbm)
              (settle-hidden-mean-field dbm 10 (flt 0.5))
              (settle-visible-mean-field dbm 10 (flt 0.5))
              (multiple-value-call #'add-error
                counter
                (reconstruction-error dbm)))))
    (log-msg "DBM TEST RMSE: ~,5F (~D)~%"
             (get-error counter)
             (n-inputs trainer))))

(defmethod positive-phase :around (batch trainer (dbm mnist-dbm))
  (call-next-method)
  (settle-visible-mean-field dbm 10 (flt 0.5))
  (multiple-value-call #'add-error (counter trainer)
                       (reconstruction-error dbm)))


;;;; DBM training

(defun bias-cloud-p (cloud)
  (or (typep (chunk1 cloud) 'constant-chunk)
      (typep (chunk2 cloud) 'constant-chunk)))

(defclass mnist-dbm-segment-trainer (batch-gd-trainer) ())

(defmethod learning-rate ((trainer mnist-dbm-segment-trainer))
  (* (slot-value trainer 'learning-rate)
     (1+ (/ (n-inputs trainer)
            (length *training-images*)))))

(defun train-mnist-dbm (dbm)
  (log-msg "Starting to train DBM.~%")
  (train (make-sampler *training-images*
                       :max-n (* 50 (length *training-images*)))
         (make-instance 'mnist-dbm-trainer
                        :n-gibbs 5
                        :segmenter
                        (lambda (cloud)
                          (make-instance 'mnist-dbm-segment-trainer
                                         :learning-rate (flt 0.005)
                                         :weight-decay
                                         (if (bias-cloud-p cloud)
                                             (flt 0)
                                             (flt 0.0002))
                                         :batch-size 100)))
         dbm))


;;;; DBN

(defclass mnist-dbn (dbn) ())

(defclass mnist-rbm (rbm) ())

(defun make-mnist-dbn (dbm)
  (mgl-bm:dbm->dbn dbm :dbn-class 'mnist-dbn :rbm-class 'mnist-rbm
                   :dbn-initargs '(:max-n-stripes 100)))

(defmethod mgl-train:set-input (images (rbm mnist-rbm))
  (let ((inputs (find 'inputs (visible-chunks rbm)
                      :key #'name)))
    (when inputs
      (clamp-striped-nodes images inputs))))

(defclass mnist-rbm-trainer (mnist-logging-trainer rbm-cd-trainer)
  ((counter :initform (make-instance 'rmse-counter) :reader counter)))

(defmethod log-training-error ((trainer mnist-rbm-trainer) (rbm mnist-rbm))
  (let ((counter (counter trainer))
        (n-inputs (n-inputs trainer)))
    (log-msg "TRAINING RMSE: ~,5F (~D)~%"
             (or (get-error counter) #.(flt 0))
             n-inputs)
    (reset-counter counter)))

(defmethod log-test-error ((trainer mnist-rbm-trainer) (rbm mnist-rbm))
  (log-msg "DBN TEST RMSE: ~{~,5F~^, ~} (~D)~%"
           (map 'list
                #'get-error
                (dbn-mean-field-errors (make-sampler *test-images*
                                                     :max-n 1000
                                                     #+nil
                                                     (length *test-images*))
                                       (dbn rbm) :rbm rbm))
           (n-inputs trainer)))

(defmethod negative-phase :around (batch trainer (rbm mnist-rbm))
  (call-next-method)
  (multiple-value-call #'add-error (counter trainer)
                       (reconstruction-error rbm)))


;;;; DBN training

(defclass mnist-rbm-segment-trainer (batch-gd-trainer) ())

(defmethod momentum ((trainer mnist-rbm-segment-trainer))
  (if (< (n-inputs trainer) 300000)
      #.(flt 0.5)
      #.(flt 0.9)))

(defun train-mnist-dbn (dbn)
  (dolist (rbm (rbms dbn))
    (do-clouds (cloud rbm)
      (if (bias-cloud-p cloud)
          (fill (storage (weights cloud)) #.(flt 0))
          (map-into (storage (weights cloud))
                    (lambda () (flt (* 0.1 (gaussian-random-1))))))))
  (loop for rbm in (rbms dbn)
        for i upfrom 0 do
        (log-msg "Starting to train level ~S RBM in DBN.~%" i)
        (train (make-sampler *training-images*
                             :max-n (* 50 (length *training-images*)))
               (make-instance 'mnist-rbm-trainer
                              :segmenter
                              (lambda (cloud)
                                (make-instance 'mnist-rbm-segment-trainer
                                               :learning-rate (flt 0.1)
                                               :weight-decay
                                               (if (bias-cloud-p cloud)
                                                   (flt 0)
                                                   (flt 0.0002))
                                               :batch-size 100)))
               rbm)))


;;;; BPN

(defclass mnist-bpn (bpn) ())

(defmethod set-input (images (bpn mnist-bpn))
  (let* ((inputs (find-lump (chunk-lump-name 'inputs nil) bpn :errorp t))
         (expectations (find-lump 'expectations bpn :errorp t))
         (inputs-nodes (storage (nodes inputs)))
         (expectations-nodes (storage (nodes expectations))))
    (loop for image in images
          for stripe upfrom 0
          do (with-stripes ((stripe inputs inputs-start)
                            (stripe expectations expectations-start
                                    expectations-end))
               (clamp-array image inputs-nodes inputs-start)
               (locally (declare (optimize (speed 3)))
                 (fill expectations-nodes #.(flt 0) :start expectations-start
                       :end expectations-end))
               (setf (aref expectations-nodes
                           (+ expectations-start (image-label image)))
                     #.(flt 1))))))

(defun max-position (array start end)
  (position (loop for i upfrom start below end maximizing (aref array i))
            array :start start :end end))

(defun bpn-decode-digit (bpn lump-name)
  (let* ((predictions (find-lump lump-name bpn :errorp t))
         (nodes (storage (if (eq 'expectations lump-name)
                             (nodes predictions)
                             (softmax predictions)))))
    (loop for stripe below (n-stripes predictions)
          collect (with-stripes ((stripe predictions start end))
                    (- (max-position nodes start end)
                       start)))))

(defun classification-error (bpn)
  (values (- (n-stripes bpn)
             (loop for p in (bpn-decode-digit bpn 'predictions)
                   for e in (bpn-decode-digit bpn 'expectations)
                   count (= p e)))
          (n-stripes bpn)))

(defun bpn-error (sampler bpn)
  (let ((counter (make-instance 'error-counter))
        (ce-counter (make-instance 'error-counter))
        (n-stripes (max-n-stripes bpn)))
    (loop until (finishedp sampler) do
          (set-input (sample-batch sampler n-stripes) bpn)
          (forward-bpn bpn)
          (multiple-value-call #'add-error ce-counter (cost bpn))
          (multiple-value-call #'add-error counter (classification-error bpn)))
    (values (get-error counter)
            (get-error ce-counter))))


;;;; BPN training

(defclass mnist-bp-trainer (mnist-logging-trainer bp-trainer)
  ((cross-entropy-counter :initform (make-instance 'error-counter)
                          :reader cross-entropy-counter)
   (counter :initform (make-instance 'error-counter) :reader counter)))

(defclass mnist-cg-bp-trainer (mnist-logging-trainer cg-bp-trainer)
  ((cross-entropy-counter :initform (make-instance 'error-counter)
                          :reader cross-entropy-counter)
   (counter :initform (make-instance 'error-counter) :reader counter)))

(defmethod log-training-error (trainer (bpn mnist-bpn))
  (let ((n-inputs (n-inputs trainer))
        (ce-counter (cross-entropy-counter trainer))
        (counter (counter trainer)))
    (log-msg "CROSS ENTROPY ERROR: ~,5F (~D)~%"
             (or (get-error ce-counter) #.(flt 0))
             n-inputs)
    (log-msg "CLASSIFICATION ACCURACY: ~,2F% (~D)~%"
             (* 100 (- 1 (or (get-error counter) #.(flt 0))))
             n-inputs)
    (reset-counter ce-counter)
    (reset-counter counter)))

(defmethod log-test-error (trainer (bpn mnist-bpn))
  (multiple-value-bind (e ce)
      (bpn-error (make-sampler *test-images* :max-n (length *test-images*)) bpn)
    (log-msg "TEST CROSS ENTROPY ERROR: ~,5F (~D)~%"
             ce (n-inputs trainer))
    (log-msg "TEST CLASSIFICATION ACCURACY: ~,2F% (~D)~%"
             (* 100 (- 1 e)) (n-inputs trainer))))

(defmethod compute-derivatives :around (samples (trainer mnist-cg-bp-trainer)
                                                bpn)
  (let ((ce-counter (cross-entropy-counter trainer))
        (counter (counter trainer)))
    (call-next-method)
    (multiple-value-call #'add-error ce-counter (cost bpn))
    (multiple-value-call #'add-error counter (classification-error bpn))))

(defmethod train-batch :around (batch (trainer mnist-cg-bp-trainer) bpn)
  (let ((ce-counter (cross-entropy-counter trainer))
        (counter (counter trainer)))
    (loop for samples in (group batch (max-n-stripes bpn)) do
          (set-input samples bpn)
          (forward-bpn bpn)
          (multiple-value-call #'add-error ce-counter (cost bpn))
          (multiple-value-call #'add-error counter (classification-error bpn)))
    (multiple-value-bind (best-w best-f
                                 n-line-searches n-succesful-line-searches
                                 n-evaluations)
        (call-next-method)
      (declare (ignore best-w))
      (log-msg "BEST-F: ~S, N-EVALUATIONS: ~S~%" best-f n-evaluations)
      (log-msg "N-LINE-SEARCHES: ~S (succesful ~S)~%"
               n-line-searches n-succesful-line-searches))))

(defun init-weights (name bpn deviation)
  (multiple-value-bind (array start end)
      (segment-weights (find-lump name bpn :errorp t))
    (loop for i upfrom start below end
          do (setf (aref array i) (flt (* deviation (gaussian-random-1)))))))

(defun unroll-mnist-dbm (dbm)
  (multiple-value-bind (defs inits) (unroll-dbm dbm)
    (print inits)
    (terpri)
    (let ((bpn (eval (print
                      `(build-bpn (:class 'mnist-bpn :max-n-stripes 1000)
                         ,@defs
                         ;; Add expectations
                         (expectations (input-lump :size 10))
                         ;; Add a softmax layer. Oh, the pain.
                         (prediction-weights
                          (weight-lump
                           :size (* (size (lump ',(chunk-lump-name 'f3 nil)))
                                    10)))
                         (prediction-biases (weight-lump :size 10))
                         (prediction-activations0
                          (activation-lump :weights prediction-weights
                                           :x (lump
                                               ',(chunk-lump-name 'f3 nil))))
                         (prediction-activations
                          (->+ :args (list prediction-activations0
                                           prediction-biases)))
                         (predictions
                          (cross-entropy-softmax-lump
                           :group-size 10
                           :x prediction-activations
                           :target expectations))
                         (my-error (error-node :x predictions)))))))
      (initialize-bpn-from-bm bpn dbm inits)
      (init-weights 'prediction-weights bpn 0.1)
      (init-weights 'prediction-biases bpn 0.1)
      bpn)))

(defun train-mnist-bpn (bpn &key (use-cg t))
  (flet ((softmax-segmenter (lump)
           (cond ((eq (name lump) 'prediction-weights)
                  (make-instance 'batch-gd-trainer
                                 :learning-rate (flt 0.5)
                                 :weight-decay (flt 0.00005)
                                 :momentum (flt 0.8)
                                 :batch-size 1000))
                 ((eq (name lump) 'prediction-biases)
                  (make-instance 'batch-gd-trainer
                                 :learning-rate (flt 0.15)
                                 :weight-decay (flt 0.00005)
                                 :momentum (flt 0.8)
                                 :batch-size 1000))
                 (t nil))))
    (log-msg "Starting to train the softmax layer of BPN~%")
    (train (make-sampler *training-images*
                         :max-n (* 5 (length *training-images*)))
           (if use-cg
               (make-instance 'mnist-cg-bp-trainer
                              :cg-args (list :max-n-line-searches 3)
                              :batch-size 1000
                              :segment-filter
                              (lambda (lump)
                                (or (eq (name lump) 'prediction-biases)
                                    (eq (name lump) 'prediction-weights))))
               (make-instance 'mnist-bp-trainer
                              :segmenter #'softmax-segmenter))
           bpn)
    (log-msg "Starting to train the whole BPN~%")
    (train (make-sampler *training-images*
                         :max-n (* 37 (length *training-images*)))
           (if use-cg
               (make-instance 'mnist-cg-bp-trainer
                              :cg-args (list :max-n-line-searches 3)
                              :batch-size 1000)
               (make-instance 'mnist-bp-trainer
                              :segmenter
                              (lambda (lump)
                                (or (softmax-segmenter lump)
                                    (make-instance
                                     'batch-gd-trainer
                                     :learning-rate
                                     ;; is this a bias?
                                     (if (< 10000 (size lump))
                                         (flt 0.03)
                                         (flt 0.1))
                                     :momentum (flt 0.8)
                                     :batch-size 1000)))))
           bpn)
    (when use-cg
      (log-msg "Full batches on the whole BPN~%")
      (train (make-sampler *training-images*
                           :max-n (* 1 (length *training-images*)))
             (make-instance 'mnist-cg-bp-trainer
                            :cg-args (list :max-n-line-searches 10)
                            :batch-size (length *training-images*))
             bpn))
    bpn))


;;;;

(defvar *dbn*)
(defvar *dbm*)
(defvar *bpn*)

(defun train-mnist (&key load-dbn-p load-dbm-p)
  (unless (boundp '*training-images*)
    (setq *training-images* (load-training)))
  (unless (boundp '*test-images*)
    (setq *test-images* (load-test)))
  (flet ((train-dbn ()
           (train-mnist-dbn *dbn*)
           (with-open-file (s (merge-pathnames "mnist.dbn" *mnist-dir*)
                            :direction :output
                            :if-does-not-exist :create :if-exists :supersede)
             (mgl-util:write-weights *dbn* s)))
         (train-dbm ()
           (train-mnist-dbm *dbm*)
           (with-open-file (s (merge-pathnames "mnist.dbm" *mnist-dir*)
                            :direction :output
                            :if-does-not-exist :create :if-exists :supersede)
             (mgl-util:write-weights *dbm* s))))
    (cond (load-dbm-p
           (setq *dbm* (make-mnist-dbm))
           (with-open-file (s (merge-pathnames "mnist.dbm" *mnist-dir*))
             (mgl-util:read-weights *dbn* s)))
          (load-dbn-p
           (setq *dbm* (make-mnist-dbm))
           (setq *dbn* (make-mnist-dbn *dbm*))
           (with-open-file (s (merge-pathnames "mnist.dbn" *mnist-dir*))
             (mgl-util:read-weights *dbn* s))
           (log-msg "Loaded DBN~%")
           (log-msg "DBN TEST RMSE: ~{~,5F~^, ~}~%"
                    (map 'list
                         #'get-error
                         (dbn-mean-field-errors
                          (make-sampler *test-images*
                                        :max-n (length *test-images*))
                          *dbn*)))
           (train-dbm))
          (t
           (setq *dbm* (make-mnist-dbm))
           (setq *dbn* (make-mnist-dbn *dbm*))
           (train-dbn)
           (train-dbm)))
    (setq *bpn* (unroll-mnist-dbm *dbn*))
    (train-mnist-bpn *bpn*)
    (with-open-file (s (merge-pathnames "mnist.bpn" *mnist-dir*)
                     :direction :output
                     :if-does-not-exist :create :if-exists :supersede)
      (mgl-util:write-weights *bpn* s))))

#|

(train-mnist)

(train-mnist :load-dbn-p t)

|#
