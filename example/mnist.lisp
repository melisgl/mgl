;;;; Code for the MNIST handwritten digit recognition challange.
;;;;
;;;; See papers by Geoffrey Hinton:
;;;;
;;;;   http://www.cs.toronto.edu/~hinton/MatlabForSciencePaper.html
;;;;
;;;;   http://www.cs.toronto.edu/~hinton/absps/montrealTR.pdf
;;;;
;;;; Download the four files from http://yann.lecun.com/exdb/mnist and
;;;; gunzip them.
;;;;
;;;; A 784-500-500-2000 deep belief network is trained then 10 softmax
;;;; units are attached to the top layer of backpropagation network
;;;; converted from the upward half of the DBN. The new, 2000->10
;;;; connections are trained and finally all weights are trained
;;;; together. Takes six days to run on a 3GHz x86 and reaches 98.96%
;;;; accuracy.

(in-package :mgl-example-mnist)

(defparameter *mnist-dir* "/home/mega/mnist/"
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


;;;; DBN

(defun layers->rbms (layers &key (class 'rbm))
  (loop for (v h) on layers
        when h
        collect (make-instance class :visible-chunks v :hidden-chunks h)))

(defclass mnist-dbn (dbn)
  ((rbms :initform (layers->rbms
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
                    :class 'mnist-rbm))))

(defun make-mnist-dbn ()
  (make-instance 'mnist-dbn))

(defclass mnist-rbm (rbm) ())

(defmethod mgl-train:set-input (sample (rbm mnist-rbm))
  (let* ((dbn (dbn rbm))
         (chunk (find 'inputs (visible-chunks (first (rbms dbn)))
                      :key #'name)))
    (when chunk
      (clamp-array sample (nodes chunk) 0))))

(defclass mnist-rbm-trainer (rbm-trainer)
  ((counter :initform (make-instance 'rmse-counter) :reader counter)))

(defun report-dbn-rmse (rbm trainer)
  (log-msg "DBN TEST RMSE: ~{~,5F~^, ~} (~D)~%"
           (coerce (dbn-rmse (make-sampler *test-images* :max-n 1000)
                             (dbn rbm) :rbm rbm) 'list)
           (n-inputs trainer)))

;;; This prints the rmse of the the training examples after each 100
;;; and the test rmse on each level of the DBN after each 1000.
(defmethod train-one (sample (trainer mnist-rbm-trainer) rbm &key)
  (when (zerop (mod (n-inputs trainer) 10000))
    (report-dbn-rmse rbm trainer))
  (let ((counter (counter trainer)))
    (multiple-value-bind (e n) (mgl-rbm:reconstruction-error rbm)
      (add-error counter e n))
    (call-next-method)
    (let ((n-inputs (n-inputs trainer)))
      (when (zerop (mod n-inputs 1000))
        (log-msg "RMSE: ~,5F (~D, ~D)~%"
                 (or (get-error counter) #.(flt 0))
                 (n-sum-errors counter)
                 n-inputs)
        (reset-counter counter)))))

;;; Make sure we print the test rmse at the end of training each rbm.
(defmethod train :after (sampler (trainer mnist-rbm-trainer) rbm &key)
  (report-dbn-rmse rbm trainer))


;;;; DBN training

(defun bias-cloud-p (cloud)
  (or (typep (visible-chunk cloud) 'constant-chunk)
      (typep (hidden-chunk cloud) 'constant-chunk)))

(defclass mnist-rbm-segment-trainer (batch-gd-trainer) ())

(defmethod momentum ((trainer mnist-rbm-segment-trainer))
  (if (< (n-inputs trainer) 300000)
      #.(flt 0.5)
      #.(flt 0.9)))

(defun train-mnist-dbn ()
  (let ((dbn (make-mnist-dbn)))
    (dolist (rbm (rbms dbn))
      (do-clouds (cloud rbm)
        (if (bias-cloud-p cloud)
            (fill (weights cloud) #.(flt 0))
            (map-into (weights cloud)
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
                 rbm))
    dbn))


;;;; BPN

(defclass mnist-bpn (bpn) ())

(defmethod set-input (image (bpn mnist-bpn))
  (let ((lump (find-lump '(inputs 0) bpn :errorp t)))
    (multiple-value-bind (array start) (lump-node-array lump)
      (clamp-array image array start)))
  (let ((lump (find-lump 'expectations bpn :errorp t)))
    (multiple-value-bind (array start end) (lump-node-array lump)
      (declare (type flt-vector array))
      (fill array #.(flt 0) :start start :end end)
      (setf (aref array (+ start (image-label image))) #.(flt 1)))))

(defun max-position (array start end)
  (position (loop for i upfrom start below end maximizing (aref array i))
            array :start start :end end))

(defun bpn-decode-digit (bpn lump-name)
  (let ((predictions (find-lump lump-name bpn :errorp t)))
    (multiple-value-bind (array start end)
        (lump-node-array predictions)
      (let ((pos (max-position array start end)))
        (values (- pos start) (aref array pos))))))

(defun classification-error (bpn)
  (if (= (bpn-decode-digit bpn 'predictions)
         (bpn-decode-digit bpn 'expectations))
      0
      1))

(defun bpn-error (sampler bpn)
  (let ((sum-errors 0)
        (n-errors 0)
        (sum-ce-errors (flt 0)))
    (loop until (finishedp sampler) do
          (set-input (sample sampler) bpn)
          (forward-bpn bpn)
          (incf sum-errors (classification-error bpn))
          (incf sum-ce-errors (last1 (nodes bpn)))
          (incf n-errors))
    (if (zerop n-errors)
        nil
        (values (/ sum-errors n-errors)
                (/ sum-ce-errors n-errors)))))


;;;; BPN training

(defclass mnist-bp-trainer (bp-trainer)
  ((cross-entropy-counter :initform (make-instance 'error-counter)
                          :reader cross-entropy-counter)
   (counter :initform (make-instance 'error-counter) :reader counter)))

(defclass mnist-cg-bp-trainer (cg-bp-trainer)
  ((cross-entropy-counter :initform (make-instance 'error-counter)
                          :reader cross-entropy-counter)
   (counter :initform (make-instance 'error-counter) :reader counter)))

#+nil
(defmethod mgl-cg:compute-batch-cost-and-derive
    (batch (trainer mnist-cg-bp-trainer) learner)
  (let ((cost (call-next-method)))
    (log-msg "Cost: ~,5F~%" cost)
    cost))
 
(defun report-bpn (bpn trainer &key (max-n (length *test-images*)))
  (multiple-value-bind (e ce)
      (bpn-error (make-sampler *test-images* :max-n max-n) bpn)
    (log-msg "TEST CROSS ENTROPY ERROR: ~,5F (~D)~%"
             ce (n-inputs trainer))
    (log-msg "TEST CLASSIFICATION ACCURACY: ~,2F% (~D)~%"
             (* 100 (- 1 e)) (n-inputs trainer))))

(defmethod train-one (image (trainer mnist-bp-trainer) bpn &key)
  (when (zerop (mod (n-inputs trainer) (length *training-images*)))
    (report-bpn bpn trainer))
  (let ((ce-counter (cross-entropy-counter trainer))
        (counter (counter trainer)))
    (call-next-method)
    (add-error ce-counter (last1 (nodes bpn)) 1)
    (add-error counter (classification-error bpn) 1)
    (let ((n-inputs (n-inputs trainer)))
      (when (zerop (mod n-inputs 1000))
        (log-msg "CROSS ENTROPY ERROR: ~,5F (~D, ~D)~%"
                 (or (get-error ce-counter) #.(flt 0))
                 (n-sum-errors ce-counter)
                 n-inputs)
        (log-msg "CLASSIFICATION ACCURACY: ~,2F% (~D, ~D)~%"
                 (* 100 (- 1 (or (get-error counter) #.(flt 0))))
                 (n-sum-errors counter)
                 n-inputs)
        (reset-counter ce-counter)
        (reset-counter counter)))))

(defmethod train :after (sampler (trainer mnist-bp-trainer) bpn &key)
  (report-bpn bpn trainer))

(defmethod train-batch (batch (trainer mnist-cg-bp-trainer) bpn &key)
  (when (zerop (mod (n-inputs trainer) (length *training-images*)))
    (report-bpn bpn trainer))
  (let ((ce-counter (cross-entropy-counter trainer))
        (counter (counter trainer)))
    (loop for sample in batch do
          (set-input sample bpn)
          (forward-bpn bpn)
          (add-error ce-counter (last1 (nodes bpn)) 1)
          (add-error counter (classification-error bpn) 1))
    (let ((n-inputs (n-inputs trainer)))
      (when (zerop (mod n-inputs 100))
        (log-msg "CROSS ENTROPY ERROR: ~,5F (~D, ~D)~%"
                 (or (get-error ce-counter) #.(flt 0))
                 (n-sum-errors ce-counter)
                 n-inputs)
        (log-msg "CLASSIFICATION ACCURACY: ~,2F% (~D, ~D)~%"
                 (* 100 (- 1 (or (get-error counter) #.(flt 0))))
                 (n-sum-errors counter)
                 n-inputs)
        (reset-counter ce-counter)
        (reset-counter counter))
      (multiple-value-bind (best-w best-f
                                   n-line-searches n-succesful-line-searches
                                   n-evaluations)
          (call-next-method)
        (declare (ignore best-w))
        (log-msg "BEST-F: ~S, N-EVALUATIONS: ~S~%"
                 best-f n-evaluations)
        (log-msg "N-LINE-SEARCHES: ~S (succesful ~S)~%"
                 n-line-searches n-succesful-line-searches)))))

(defmethod train :after (sampler (trainer mnist-cg-bp-trainer) bpn &key)
  (report-bpn bpn trainer))

(defun init-lump (name bpn deviation)
  (multiple-value-bind (array start end)
      (lump-node-array (find-lump name bpn :errorp t))
    (loop for i upfrom start below end
          do (setf (aref array i) (flt (* deviation (gaussian-random-1)))))))

(defun unroll-mnist-dbn (dbn)
  (multiple-value-bind (defs clamps inits) (unroll-dbn dbn :bottom-up-only t)
    (print clamps)
    (print inits)
    (terpri)
    (let ((bpn (eval (print
                      `(build-bpn (:class 'mnist-bpn)
                         ,@defs
                         ;; Add expectations
                         (input-lump :symbol expectations
                                     :size 10)
                         ;; Add a softmax layer. Oh, the pain.
                         (weight-lump :symbol prediction-weights
                                      :size (* (lump-size (lump '(f3 3))) 10))
                         (weight-lump :symbol prediction-biases
                                      :size 10)
                         (activation-lump :symbol prediction-activations0
                                          :weight-lump prediction-weights
                                          :input-lump (lump '(f3 3))
                                          :size 10)
                         (hidden-lump :symbol prediction-activations
                                      :size 10
                                      :def (->+ (_)
                                             (ref prediction-activations0 _)
                                             (ref prediction-biases _)))
                         (cross-entropy-softmax-lump
                          :symbol predictions
                          :size 10 :group-size 10
                          :input-lump prediction-activations
                          :target-lump expectations)
                         ;; just to measure progress
                         (hidden-lump :size 1
                                      :def (->cross-entropy (_)
                                             expectations
                                             predictions)))))))
      (initialize-bpn-from-dbn bpn dbn inits)
      (init-lump 'prediction-weights bpn 0.1)
      (init-lump 'prediction-biases bpn 0.1)
      bpn)))

(defun weight-lump-p (lump)
  (and (typep lump 'weight-lump)
       (member (name lump) '(((inputs f1) :weights 0)
                             ((f1 f2) :weights 1)
                             ((f2 f3) :weights 2)
                             prediction-weights)
               :test #'equal)))

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
                                     (if (< 10000 (mgl-bp:lump-size lump))
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
(defvar *bpn*)

(defun train-mnist (&key load-dbn-p)
  (unless (boundp '*training-images*)
    (setq *training-images* (load-training)))
  (unless (boundp '*test-images*)
    (setq *test-images* (load-test)))
  (cond (load-dbn-p
         (setq *dbn* (make-mnist-dbn))
         (with-open-file (s (merge-pathnames "mnist.dbn" *mnist-dir*))
           (mgl-util:read-weights *dbn* s)))
        (t
         (setq *dbn* (train-mnist-dbn))
         (with-open-file (s (merge-pathnames "mnist.dbn" *mnist-dir*)
                          :direction :output
                          :if-does-not-exist :create :if-exists :supersede)
           (mgl-util:write-weights *dbn* s))))
  (setq *bpn* (unroll-mnist-dbn *dbn*))
  (train-mnist-bpn *bpn*)
  (with-open-file (s (merge-pathnames "mnist.bpn" *mnist-dir*)
                   :direction :output
                   :if-does-not-exist :create :if-exists :supersede)
    (mgl-util:write-weights *bpn* s)))

#|

(setq *mnist-dir* "/home/mega/mnist/")
(train-mnist)

(train-mnist :load-dbn-p t)

|#
