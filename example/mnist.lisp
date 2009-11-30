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
;;;; gunzip them. Set *MNIST-DIR* to point to their directory and call
;;;; TRAIN-MNIST/1 or TRAIN-MNIST/2 for DBN-to-BPN and
;;;; DBN-to-DBM-to-BPN approaches, respectively.
;;;;
;;;;
;;;; DBN-to-BPN
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
;;;; DBN-to-DBM-to-BPN
;;;;
;;;; A 784-500-1000 DBM is pretrained as a DBN then the DBM is
;;;; translated to a BPN. It's pretty much the same training process.

(in-package :mgl-example-mnist)

(defparameter *mnist-dir*
  (merge-pathnames "mnist-data/" *example-dir*)
  "Set this to the directory where the uncompressed mnist files reside.")

(defstruct image
  (label nil :type (integer 0 10))
  (array nil :type flt-vector))

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
  (log-msg "Loading training images~%")
  (prog1
      (map 'vector
           (lambda (label array)
             (make-image :label label :array array))
           (with-open-file (s (merge-pathnames "train-labels-idx1-ubyte"
                                               mnist-dir)
                            :element-type 'unsigned-byte)
             (read-image-labels s))
           (with-open-file (s (merge-pathnames "train-images-idx3-ubyte"
                                               mnist-dir)
                            :element-type 'unsigned-byte)
             (read-image-arrays s)))
    (log-msg "Loading training images done~%")))

(defun load-test (&optional (mnist-dir *mnist-dir*))
  (log-msg "Loading test images~%")
  (prog1
      (map 'vector
           (lambda (label array)
             (make-image :label label :array array))
           (with-open-file (s (merge-pathnames "t10k-labels-idx1-ubyte"
                                               mnist-dir)
                            :element-type 'unsigned-byte)
             (read-image-labels s))
           (with-open-file (s (merge-pathnames "t10k-images-idx3-ubyte"
                                               mnist-dir)
                            :element-type 'unsigned-byte)
             (read-image-arrays s)))
    (log-msg "Loading test images done~%")))

(defvar *training-images*)
(defvar *test-images*)


;;;; Sampling, clamping, utilities

(defun make-sampler (stories &key max-n omit-label-p sample-visible-p)
  (make-instance 'counting-function-sampler
                 :max-n-samples max-n
                 :sampler (let ((g (make-random-generator stories)))
                            (lambda ()
                              (list (funcall g)
                                    :omit-label-p omit-label-p
                                    :sample-visible-p sample-visible-p)))))

(defun make-training-sampler (&key omit-label-p)
  (make-sampler (subseq *training-images* 0 1000)
                :max-n 1000
                :omit-label-p omit-label-p))

(defun make-test-sampler (&key omit-label-p)
  (make-sampler *test-images*
                :max-n (length *test-images*)
                :omit-label-p omit-label-p))

(defun clamp-array (image array start)
  (declare (type flt-vector array)
           (optimize (speed 3)))
  (replace array (image-array image) :start1 start))

(defun clamp-striped-nodes (images striped)
  (assert (= (length images) (n-stripes striped)))
  (let ((nodes (storage (nodes striped))))
    (loop for image in images
          for stripe upfrom 0
          do (destructuring-bind (image &key omit-label-p sample-visible-p) image
               (declare (ignore omit-label-p sample-visible-p))
               (with-stripes ((stripe striped start))
                 (clamp-array image nodes start))))))

(defun layers->rbms (layers &key (class 'rbm))
  (loop for (v h) on layers
        when h
        collect (make-instance class :visible-chunks v :hidden-chunks h)))


;;;; Logging

(defun ->percent (x)
  (* 100 (- 1 x)))

(defparameter *percent-format* "~,2F%")
(defparameter *list-of-percents-format*
  (format nil "~~{~A~~^, ~~}" *percent-format*))

(defclass mnist-logging-trainer (logging-trainer) ())

(defmethod log-training-period ((trainer mnist-logging-trainer) learner)
  10000)

(defmethod log-test-period ((trainer mnist-logging-trainer) learner)
  (length *training-images*))

(defmethod log-test-error ((trainer mnist-logging-trainer) learner)
  (let ((*print-level* nil))
    (when (zerop (n-inputs trainer))
      (describe learner))
    (describe trainer))
  (log-msg "n-inputs: ~S~%" (n-inputs trainer)))


;;;; DBN

(defclass mnist-dbn (dbn) ())

(defclass mnist-rbm (rbm) ())

(defmethod set-input (images (rbm mnist-rbm))
  (let ((inputs (find 'inputs (visible-chunks rbm)
                      :key #'name)))
    (when inputs
      (clamp-striped-nodes images inputs))))

(defclass mnist-rbm-trainer (mnist-logging-trainer rbm-cd-trainer)
  ((counter
    :initform (make-instance 'rmse-counter :prepend-name "training")
    :reader counter)))

(defmethod n-gibbs ((trainer mnist-rbm-trainer))
  (let ((x (slot-value trainer 'n-gibbs)))
    (if (integerp x)
        x
        (funcall x trainer))))

(defmethod log-training-error ((trainer mnist-rbm-trainer) (rbm mnist-rbm))
  (log-msg "n-inputs: ~D~%" (n-inputs trainer))
  (log-msg "~A~%" (counter trainer))
  (reset-counter (counter trainer)))

(defun collect-dbn-mean-field-errors*
    (sampler dbn &key (rbm (last1 (rbms dbn)))
     (counters-and-measurers
      (make-dbn-reconstruction-rmse-counters-and-measurers dbn :rbm rbm)))
  "Like COLLECT-DBN-MEAN-FIELD-ERRORS but reconstruct the LABEL chunk
even if it's missing in the input."
  (collect-batch-errors (lambda (samples)
                          (set-input samples rbm)
                          (set-hidden-mean rbm)
                          (when (find-chunk 'label rbm)
                            (setf (indices-present (find-chunk 'label rbm)) nil))
                          (down-mean-field dbn :rbm rbm))
                        sampler
                        dbn
                        counters-and-measurers))

(defun log-dbn-classification-accuracy (rbm sampler name)
  (let ((counters-and-measurers
         (make-dbn-reconstruction-misclassification-counters-and-measurers
          (dbn rbm) :rbm rbm)))
    (map nil (lambda (counter)
               (log-msg "~A ~:_~A~%" name counter))
         (collect-dbn-mean-field-errors*
          sampler
          (dbn rbm)
          :counters-and-measurers
          counters-and-measurers))))

(defmethod log-test-error ((trainer mnist-rbm-trainer) (rbm mnist-rbm))
  (call-next-method)
  (log-dbn-classification-accuracy rbm (make-training-sampler)
                                   "dbn training reconstruction")
  (map nil (lambda (counter)
             (log-msg "dbn test ~:_~A~%" counter))
       (collect-dbn-mean-field-errors* (make-test-sampler) (dbn rbm) :rbm rbm))
  (log-dbn-classification-accuracy rbm (make-test-sampler :omit-label-p t)
                                   "dbn test"))

(defmethod negative-phase :around (batch trainer (rbm mnist-rbm))
  (call-next-method)
  (multiple-value-call #'add-error (counter trainer)
                       (reconstruction-rmse
                        (remove-if (lambda (chunk)
                                     (typep chunk 'labeled))
                                   (visible-chunks rbm)))))


;;;; DBN training

(defclass mnist-rbm-segment-trainer (batch-gd-trainer) ())

(defmethod learning-rate ((trainer mnist-rbm-segment-trainer))
  (let ((x (slot-value trainer 'learning-rate)))
    (if (numberp x)
        x
        (funcall x trainer))))

(defmethod momentum ((trainer mnist-rbm-segment-trainer))
  (if (< (n-inputs trainer) (* 5 (length *training-images*)))
      #.(flt 0.5)
      #.(flt 0.9)))

(defun init-mnist-dbn (dbn &key stddev (start-level 0))
  (loop for i upfrom start-level
        for rbm in (subseq (rbms dbn) start-level) do
        (flet ((this (x)
                 (if (listp x)
                     (elt x i)
                     x)))
          (do-clouds (cloud rbm)
            (if (conditioning-cloud-p cloud)
                (fill (storage (weights cloud)) #.(flt 0))
                (progn
                  (log-msg "init: ~A ~A~%" cloud (this stddev))
                  (map-into (storage (weights cloud))
                            (lambda () (flt (* (this stddev)
                                               (gaussian-random-1)))))))))))

(defun train-mnist-dbn (dbn &key n-epochs n-gibbs learning-rate decay
                        visible-sampling (start-level 0))
  (loop for rbm in (subseq (rbms dbn) start-level)
        for i upfrom start-level do
        (log-msg "Starting to train level ~S RBM in DBN.~%" i)
        (flet ((this (x)
                 (if (listp x)
                     (elt x i)
                     x)))
          (train (make-sampler *training-images*
                               :max-n (* n-epochs (length *training-images*))
                               :sample-visible-p (this visible-sampling))
                 (make-instance 'mnist-rbm-trainer
                                :visible-sampling (this visible-sampling)
                                :n-gibbs (this n-gibbs)
                                :segmenter
                                (lambda (cloud)
                                  (make-instance 'mnist-rbm-segment-trainer
                                                 :learning-rate
                                                 (this learning-rate)
                                                 :weight-decay
                                                 (if (conditioning-cloud-p cloud)
                                                     (flt 0)
                                                     (flt (this decay)))
                                                 :batch-size 100)))
                 rbm))))


;;;; BPN

(defclass mnist-bpn (bpn) ())

(defmethod set-input (images (bpn mnist-bpn))
  (let* ((inputs (find-lump (chunk-lump-name 'inputs nil) bpn :errorp t))
         (expectations (find-lump 'expectations bpn :errorp t))
         (inputs-nodes (storage (nodes inputs)))
         (expectations-nodes (storage (nodes expectations))))
    (loop for image in images
          for stripe upfrom 0
          do
          (destructuring-bind (image &key omit-label-p sample-visible-p) image
            (assert omit-label-p)
            (assert (not sample-visible-p))
            (with-stripes ((stripe inputs inputs-start)
                           (stripe expectations expectations-start
                                   expectations-end))
              (clamp-array image inputs-nodes inputs-start)
              (locally (declare (optimize (speed 3)))
                (fill expectations-nodes #.(flt 0) :start expectations-start
                      :end expectations-end))
              (setf (aref expectations-nodes
                          (+ expectations-start (image-label image)))
                    #.(flt 1)))))))

(defun classification-error (bpn)
  (cross-entropy-softmax-max-likelihood-classification-error
   (find-lump 'predictions bpn :errorp t)))

(defun bpn-error (sampler bpn)
  (collect-bpn-errors
   sampler bpn
   :counters-and-measurers
   (list (cons (make-instance 'misclassification-counter)
               (lambda (samples bpn)
                 (declare (ignore samples))
                 (classification-error bpn)))
         (cons (make-instance 'error-counter :name '("cross entropy"))
               (lambda (samples bpn)
                 (declare (ignore samples))
                 (cost bpn))))))

(defun make-bpn (defs chunk-name &key class initargs)
  (let ((bpn-def `(build-bpn (:class ',class
                                     :max-n-stripes 1000
                                     :initargs ',initargs)
                    ,@defs
                    ,@(tack-cross-entropy-softmax-error-on
                       10
                       (chunk-lump-name chunk-name nil)
                       :prefix '||))))
    (log-msg "~%~S~%" bpn-def)
    (eval bpn-def)))


;;;; BPN training

(defclass mnist-cg-bp-trainer (mnist-logging-trainer cg-bp-trainer)
  ((cross-entropy-counter
    :initform (make-instance 'error-counter :name '("training cross entropy"))
    :reader cross-entropy-counter)
   (counter
    :initform (make-instance 'misclassification-counter :prepend-name "training")
    :reader counter)))

(defmethod log-training-error (trainer (bpn mnist-bpn))
  (log-msg "n-inputs: ~S~%" (n-inputs trainer))
  (log-msg "~A~%" (cross-entropy-counter trainer))
  (log-msg "~A~%" (counter trainer))
  (reset-counter (cross-entropy-counter trainer))
  (reset-counter (counter trainer)))

(defmethod log-test-error ((trainer mnist-cg-bp-trainer) (bpn mnist-bpn))
  (call-next-method)
  (map nil (lambda (counter)
             (log-msg "test ~:_~A~%" counter))
       (bpn-error (make-test-sampler :omit-label-p t) bpn)))

(defmethod compute-derivatives :around (samples (trainer mnist-cg-bp-trainer)
                                                bpn)
  (let ((ce-counter (cross-entropy-counter trainer))
        (counter (counter trainer)))
    (call-next-method)
    (multiple-value-call #'add-error ce-counter (cost bpn))
    (multiple-value-call #'add-error counter (classification-error bpn))))

(defmethod train-batch :around (batch (trainer mnist-cg-bp-trainer) bpn)
  (multiple-value-bind (best-w best-f
                               n-line-searches n-succesful-line-searches
                               n-evaluations)
      (call-next-method)
    (declare (ignore best-w))
    (log-msg "best-f: ~,5E, ~:_n-evaluations: ~S~%" best-f n-evaluations)
    (log-msg "n-line-searches: ~S (succesful ~S)~%"
             n-line-searches n-succesful-line-searches)))

(defun init-weights (name bpn deviation)
  (multiple-value-bind (array start end)
      (segment-weights (find-lump name bpn :errorp t))
    (loop for i upfrom start below end
          do (setf (aref array i) (flt (* deviation (gaussian-random-1)))))))

(defun train-mnist-bpn (bpn &key (batch-size 1000))
  (log-msg "Starting to train the softmax layer of BPN~%")
  (train (make-sampler *training-images*
                       :max-n (* 5 (length *training-images*))
                       :omit-label-p t)
         (make-instance 'mnist-cg-bp-trainer
                        :cg-args (list :max-n-line-searches 3)
                        :batch-size batch-size
                        :segment-filter
                        (lambda (lump)
                          (or (eq (name lump) 'prediction-biases)
                              (eq (name lump) 'prediction-weights))))
         bpn)
  (log-msg "Starting to train the whole BPN~%")
  (train (make-sampler *training-images*
                       :max-n (* 95 (length *training-images*))
                       :omit-label-p t)
         (make-instance 'mnist-cg-bp-trainer
                        :cg-args (list :max-n-line-searches 3)
                        :batch-size batch-size)
         bpn)
  (log-msg "Full batches on the whole BPN~%")
  (train (make-sampler *training-images*
                       :max-n (* 1 (length *training-images*))
                       :omit-label-p t)
         (make-instance 'mnist-cg-bp-trainer
                        :cg-args (list :max-n-line-searches 10)
                        :batch-size (length *training-images*))
         bpn))


;;;; Code for the DBN to BPN approach (paper [1])

(defun make-mnist-dbn/1 ()
  (make-instance 'mnist-dbn
                 :rbms (layers->rbms
                        (list (list (make-instance 'constant-chunk :name 'c0)
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
                        :class 'mnist-rbm)
                 :max-n-stripes 100))

(defun unroll-mnist-dbn/1 (dbn)
  (multiple-value-bind (defs inits) (unroll-dbn dbn :bottom-up-only t)
    (log-msg "inits:~%~S~%" inits)
    (let ((bpn (make-bpn defs 'f3 :class 'mnist-bpn)))
      (initialize-bpn-from-bm bpn dbn inits)
      (init-weights 'prediction-weights bpn 0.1)
      (init-weights 'prediction-biases bpn 0.1)
      bpn)))

(defvar *dbn/1*)
(defvar *bpn/1*)

(defparameter *mnist-1-dbn-filename*
  (merge-pathnames "mnist-1.dbn" *mnist-dir*))

(defparameter *mnist-1-bpn-filename*
  (merge-pathnames "mnist-1.bpn" *mnist-dir*))

(defun train-mnist/1 (&key load-dbn-p)
  (unless (boundp '*training-images*)
    (setq *training-images* (load-training)))
  (unless (boundp '*test-images*)
    (setq *test-images* (load-test)))
  (cond (load-dbn-p
         (setq *dbn/1* (make-mnist-dbn/1))
         (load-weights *mnist-1-dbn-filename* *dbn/1*)
         (log-msg "Loaded DBN~%")
         (map nil (lambda (counter)
                    (log-msg "dbn test ~:_~A~%" counter))
              (collect-dbn-mean-field-errors* (make-test-sampler) *dbn/1*)))
        (t
         (setq *dbn/1* (make-mnist-dbn/1))
         (init-mnist-dbn *dbn/1* :stddev 0.1)
         (train-mnist-dbn *dbn/1* :n-epochs 50 :n-gibbs 1
                          :learning-rate (flt 0.1)
                          :decay 0.0002 :visible-sampling nil)
         (save-weights *mnist-1-dbn-filename* *dbn/1*)))
  (setq *bpn/1* (unroll-mnist-dbn/1 *dbn/1*))
  (train-mnist-bpn *bpn/1*)
  (save-weights *mnist-1-bpn-filename* *bpn/1*))


;;;; Code for the DBN to DBM to BPN approach (paper [2])

(defvar *dbn/2*)
(defvar *dbm/2*)
(defvar *bpn/2*)

(defparameter *mnist-2-dbn-filename*
  (merge-pathnames "mnist-2.dbn" *mnist-dir*))

(defparameter *mnist-2-dbm-filename*
  (merge-pathnames "mnist-2.dbm" *mnist-dir*))

(defparameter *mnist-2-bpn-filename*
  (merge-pathnames "mnist-2.bpn" *mnist-dir*))

(defclass mnist-rbm/2 (mnist-rbm) ())
(defclass mnist-bpn/2 (mnist-bpn bpn-clamping-cache) ())

(defun clamp-labels (images chunk)
  (setf (indices-present chunk)
        (if (and images (getf (rest (elt images 0)) :omit-label-p))
            (make-array 0 :element-type 'index)
            nil))
  (let ((nodes (storage (nodes chunk))))
    (loop for image in images
          for stripe upfrom 0
          do (destructuring-bind (image &key omit-label-p sample-visible-p)
                 image
               (declare (ignore sample-visible-p))
               (with-stripes ((stripe chunk start end))
                 (unless omit-label-p
                   (fill nodes (flt 0) :start start :end end)
                   (setf (aref nodes (+ start (image-label image)))
                         #.(flt 1))))))))

(defun strip-sample-visible (samples)
  (mapcar (lambda (sample)
            (destructuring-bind (image &key omit-label-p sample-visible-p)
                sample
              (declare (ignore sample-visible-p))
              (list image :omit-label-p omit-label-p)))
          samples))

(defmethod set-input (images (rbm mnist-rbm/2))
  (call-next-method (strip-sample-visible images) rbm)
  ;; All samples have the same SAMPLE-VISIBLE-P, look at the first
  ;; only.
  (when (and images (getf (rest (elt images 0)) :sample-visible-p))
    (sample-visible rbm))
  (let ((label-chunk (find-chunk 'label rbm)))
    (when label-chunk
      (clamp-labels images label-chunk))))

(defclass softmax-label-chunk* (softmax-label-chunk) ())

;;; Samplers don't return examples, but a list of (SAMPLE &KEY
;;; OMIT-LABEL-P SAMPLE-VISIBLE-P). Work around it.
(defmethod maybe-make-misclassification-measurer ((chunk softmax-label-chunk*))
  (let ((measurer (call-next-method)))
    (when measurer
      (lambda (examples learner)
        (funcall measurer (mapcar #'first examples) learner)))))

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
                       (:chunk1 inputs :chunk2 label :class nil)))
   (training-classification-counters-and-measurers
    :reader training-classification-counters-and-measurers)))

(defmethod initialize-instance :after ((dbm mnist-dbm) &key &allow-other-keys)
  (setf (slot-value dbm 'training-classification-counters-and-measurers)
        (make-bm-reconstruction-misclassification-counters-and-measurers dbm))
  (dolist (counter-and-measurer
            (training-classification-counters-and-measurers dbm))
    (push "training" (slot-value (car counter-and-measurer) 'name))))

(defun make-mnist-dbm ()
  (make-instance 'mnist-dbm :max-n-stripes 100))

(defmethod set-input (images (dbm mnist-dbm))
  (clamp-striped-nodes images (mgl-bm:find-chunk 'inputs dbm))
  (when (and images (getf (rest (elt images 0)) :sample-visible-p))
    (sample-visible dbm))
  (let ((label-chunk (find-chunk 'label dbm :errorp t)))
    (clamp-labels images label-chunk)))

(defclass mnist-dbm-trainer (mnist-logging-trainer bm-pcd-trainer)
  ((counter
    :initform (make-instance 'rmse-counter
                             :prepend-name "dbm training reconstruction")
    :reader counter)))

(defmethod log-training-error ((trainer mnist-dbm-trainer) (dbm mnist-dbm))
  (log-msg "n-inputs: ~S~%"  (n-inputs trainer))
  (log-msg "~A~%" (counter trainer))
  (reset-counter (counter trainer))
  (dolist (counter-and-measurer
            (training-classification-counters-and-measurers dbm))
    (let ((counter (car counter-and-measurer)))
      (log-msg "~A~%" counter)
      (reset-counter counter))))

(defun collect-mnist-dbm-mean-field-errors
    (sampler bm &key
     (counters-and-measurers
      (make-bm-reconstruction-rmse-counters-and-measurers bm)))
  (collect-batch-errors (lambda (samples)
                          (set-input samples bm)
                          (set-hidden-mean bm)
                          (setf (indices-present (find-chunk 'label bm)) nil)
                          (set-visible-mean bm))
                        sampler
                        bm
                        counters-and-measurers))

(defun log-dbm-classification-accuracy (dbm sampler name)
  (let ((counters-and-measurers
         (make-bm-reconstruction-misclassification-counters-and-measurers dbm)))
    (when counters-and-measurers
      (map nil (lambda (counter)
                 (log-msg "dbm ~:_~A ~:_~A~%" name counter))
           (collect-mnist-dbm-mean-field-errors
            sampler dbm :counters-and-measurers counters-and-measurers)))))

(defmethod log-test-error ((trainer mnist-dbm-trainer) (dbm mnist-dbm))
  (call-next-method)
  (save-weights (merge-pathnames (format nil "mnist-2-~A.dbm"
                                         (floor (n-inputs trainer)
                                                (length *training-images*)))
                                 *mnist-dir*)
                dbm)
  (log-dbm-classification-accuracy dbm (make-training-sampler)
                                   "training reconstruction")
  (log-dbm-classification-accuracy dbm (make-training-sampler :omit-label-p t)
                                   "training")
  ;; This is too time consuming.
  #+nil
  (map nil (lambda (counter)
             (log-msg "dbm test ~:_~A~%" counter))
       (collect-mnist-dbm-mean-field-errors (make-test-sampler) dbm))
  (log-dbm-classification-accuracy dbm (make-test-sampler :omit-label-p t)
                                   "test"))

(defmethod positive-phase :around (batch trainer (dbm mnist-dbm))
  (call-next-method)
  (set-visible-mean dbm)
  (apply-counters-and-measurers
   batch
   (training-classification-counters-and-measurers dbm))
  (multiple-value-call #'add-error (counter trainer)
                       (reconstruction-rmse
                        (remove-if (lambda (chunk)
                                     (typep chunk 'labeled))
                                   (visible-chunks dbm)))))

(defclass mnist-dbm-segment-trainer (batch-gd-trainer) ())

(defmethod learning-rate ((trainer mnist-dbm-segment-trainer))
  ;; This is adjusted for each batch. Ruslan's code adjusts it per
  ;; epoch.
  (/ (slot-value trainer 'learning-rate)
     (min (flt 10)
          (expt 1.000015
                (* (/ (n-inputs trainer)
                      (length *training-images*))
                   600)))))

(defmethod momentum ((trainer mnist-dbm-segment-trainer))
  (if (< (n-inputs trainer) (* 5 (length *training-images*)))
      #.(flt 0.5)
      #.(flt 0.9)))

(defmethod initialize-trainer ((trainer mnist-dbm-trainer) (dbm mnist-dbm))
  (call-next-method)
  (set-input (sample-batch (make-sampler *training-images*
                                         :max-n (length *training-images*)
                                         :sample-visible-p t)
                           (n-particles trainer))
             (persistent-chains trainer))
  (up-dbm (persistent-chains trainer)))

(defun train-mnist-dbm (dbm)
  (log-msg "Starting to train DBM.~%")
  (train (make-sampler *training-images*
                       :max-n (* 200 (length *training-images*))
                       :sample-visible-p t)
         (make-instance 'mnist-dbm-trainer
                        :n-particles 100
                        :visible-sampling t
                        :n-gibbs 5
                        :segmenter
                        (lambda (cloud)
                          (when t #+nil (or (equal 'label (name (chunk1 cloud)))
                                            (equal 'label (name (chunk2 cloud))))
                                (make-instance 'mnist-dbm-segment-trainer
                                               :learning-rate (flt 0.001)
                                               :weight-decay
                                               (if (conditioning-cloud-p cloud)
                                                   (flt 0)
                                                   (flt 0.0002))
                                               :batch-size 100)))
                        :sparser
                        (lambda (cloud chunk)
                          (when (and (member (name chunk) '(f1 f2))
                                     (not (equal 'label (name (chunk1 cloud))))
                                     (not (equal 'label (name (chunk2 cloud)))))
                            (make-instance
                             'normal-sparsity-gradient-source
                             :cloud cloud
                             :chunk chunk
                             :sparsity (flt (if (eq 'f2 (name chunk))
                                                0.1
                                                0.2))
                             :cost (flt 0.001)
                             :damping (flt 0.9)))))
         dbm))

(defun make-mnist-dbn/2 (dbm)
  (mgl-bm:dbm->dbn dbm :dbn-class 'mnist-dbn :rbm-class 'mnist-rbm/2
                   :dbn-initargs '(:max-n-stripes 100)))

(defun unroll-mnist-dbm (dbm &key use-label-weights initargs)
  (multiple-value-bind (defs inits)
      (unroll-dbm dbm :chunks (remove 'label (chunks dbm) :key #'name))
    (log-msg "inits:~%~S~%" inits)
    (let ((bpn (make-bpn defs 'f2 :class 'mnist-bpn/2 :initargs initargs)))
      (initialize-bpn-from-bm bpn dbm
                              (if use-label-weights
                                  (list*
                                   '(:cloud-name (label c2)
                                     :weight-name prediction-biases)
                                   '(:cloud-name (label f2)
                                     :weight-name prediction-weights)
                                   inits)
                                  inits))
      bpn)))

(defun train-mnist/2 (&key load-dbn-p load-dbm-p)
  (unless (boundp '*training-images*)
    (setq *training-images* (load-training)))
  (unless (boundp '*test-images*)
    (setq *test-images* (load-test)))
  (flet ((train-dbn ()
           (init-mnist-dbn *dbn/2* :stddev '(0.001 0.01) :start-level 0)
           (train-mnist-dbn
            *dbn/2*
            :start-level 0
            :n-epochs 100
            :n-gibbs (list 1
                           (lambda (trainer)
                             (ceiling (1+ (n-inputs trainer))
                                      (* 20 (length *training-images*)))))
            :learning-rate
            (list (flt 0.05)
                  (lambda (trainer)
                    (/ (flt 0.05)
                       (ceiling (1+ (n-inputs trainer))
                                (* 20 (length *training-images*))))))
            :decay 0.001
            :visible-sampling t)
           (save-weights *mnist-2-dbn-filename* *dbn/2*))
         (train-dbm ()
           (train-mnist-dbm *dbm/2*)
           (save-weights *mnist-2-dbm-filename* *dbm/2*)))
    (cond (load-dbm-p
           (setq *dbm/2* (make-mnist-dbm))
           (load-weights *mnist-2-dbm-filename* *dbm/2*))
          (load-dbn-p
           (setq *dbm/2* (make-mnist-dbm))
           (setq *dbn/2* (make-mnist-dbn/2 *dbm/2*))
           (load-weights *mnist-2-dbn-filename* *dbn/2*)
           (log-msg "Loaded DBN~%")
           (train-dbm))
          (t
           (setq *dbm/2* (make-mnist-dbm))
           (setq *dbn/2* (make-mnist-dbn/2 *dbm/2*))
           (train-dbn)
           (train-dbm)))
    (setq *bpn/2*
          (unroll-mnist-dbm
           *dbm/2*
           :use-label-weights t
           :initargs
           ;; We are playing games with wrapping image objects in
           ;; lists when sampling, so let the map cache know what's
           ;; the key in the cache and what to pass to the dbm. We
           ;; carefully omit labels.
           (list :populate-key #'first
                 :populate-convert-to-dbm-sample-fn (lambda (sample)
                                                      (list (first sample)
                                                            :sample-visible-p nil
                                                            :omit-label-p t))
                 :populate-map-cache-lazily-from-dbm *dbm/2*
                 :populate-periodic-fn
                 (make-instance 'periodic-fn :period 1000
                                :fn (lambda (n)
                                      (log-msg "populated: ~S~%" n))))))
    (unless (populate-map-cache-lazily-from-dbm *bpn/2*)
      (log-msg "Populating MAP cache~%")
      (populate-map-cache *bpn/2* *dbm/2* (concatenate 'vector *training-images*
                                                       *test-images*)
                          :if-exists :error))
    (train-mnist-bpn *bpn/2* :batch-size 10000)
    (save-weights *mnist-2-bpn-filename* *bpn/2*)))

(defmethod negative-phase (batch (trainer mnist-dbm-trainer) bm)
  (mgl-bm::check-no-self-connection bm)
  (flet ((foo (chunk)
           (mgl-bm::set-mean (list chunk) bm)
           (sample-chunk chunk)))
    (loop for i below (n-gibbs trainer) do
          (sample-chunk (find-chunk 'f1 bm))
          (foo (find-chunk 'inputs bm))
          (foo (find-chunk 'f2 bm))
          (foo (find-chunk 'label bm))
          (mgl-bm::set-mean (list (find-chunk 'f1 bm)) bm))
    (mgl-bm::set-mean* (list (find-chunk 'f2 bm)) bm)
    (mgl-bm::accumulate-negative-phase-statistics
     trainer bm
     ;; The number of persistent chains (or fantasy particles), that
     ;; is, N-STRIPES of PERSISTENT-CHAINS is not necessarily the same
     ;; as the batch size. Normalize so that positive and negative
     ;; phase has the same weight.
     :multiplier (/ (length batch)
                    (n-stripes (persistent-chains trainer))))))

#|

(train-mnist/1)

(train-mnist/1 :load-dbn-p t)

(train-mnist/2)

(train-mnist/2 :load-dbn-p t)

(train-mnist/2 :load-dbm-p t)

(require :sb-sprof)

(progn
  (sb-sprof:reset)
  (sb-sprof:start-profiling)
  (sleep 10)
  (sb-sprof:stop-profiling)
  (sb-sprof:report :type :graph))

(describe *dbm/2*)
(map nil #'print (clouds *dbm/2*))

|#
