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


;;;; Sampling, clamping, utilities

(defun make-sampler (stories &key max-n omit-label-p sample-visible-p)
  (make-instance 'counting-function-sampler
                 :max-n-samples max-n
                 :sampler (let ((g (make-random-generator stories)))
                            (lambda ()
                              (list (funcall g)
                                    :omit-label-p omit-label-p
                                    :sample-visible-p sample-visible-p)))))

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

(defun bias-cloud-p (cloud)
  (or (typep (chunk1 cloud) 'constant-chunk)
      (typep (chunk2 cloud) 'constant-chunk)))

(defun load-weights (filename obj)
  (with-open-file (stream filename)
    (mgl-util:read-weights obj stream)))

(defun save-weights (filename obj)
  (with-open-file (stream filename :direction :output
                   :if-does-not-exist :create :if-exists :supersede)
    (mgl-util:write-weights obj stream)))

(defun layers->rbms (layers &key (class 'rbm))
  (loop for (v h) on layers
        when h
        collect (make-instance class :visible-chunks v :hidden-chunks h)))


;;;; Logging

(defclass mnist-logging-trainer (logging-trainer) ())

(defmethod log-training-period ((trainer mnist-logging-trainer) learner)
  10000)

(defmethod log-test-period ((trainer mnist-logging-trainer) learner)
  (length *training-images*))


;;;; DBN

(defclass mnist-dbn (dbn) ())

(defclass mnist-rbm (rbm) ())

(defmethod set-input (images (rbm mnist-rbm))
  (let ((inputs (find 'inputs (visible-chunks rbm)
                      :key #'name)))
    (when inputs
      (clamp-striped-nodes images inputs))))

(defclass mnist-rbm-trainer (mnist-logging-trainer rbm-cd-trainer)
  ((counter :initform (make-instance 'rmse-counter) :reader counter)))

(defmethod n-gibbs ((trainer mnist-rbm-trainer))
  (let ((x (slot-value trainer 'n-gibbs)))
    (if (integerp x)
        x
        (funcall x trainer))))

(defmethod log-training-error ((trainer mnist-rbm-trainer) (rbm mnist-rbm))
  (let ((counter (counter trainer))
        (n-inputs (n-inputs trainer)))
    (log-msg "TRAINING RMSE: ~,5F (~D)~%"
             (or (get-error counter) #.(flt 0))
             n-inputs)
    (when (zerop (mod n-inputs 60000))
      (reset-counter counter))))


;;;; Label utilities

(defun count-misclassifications (examples striped
                                 &key (label-fn #'label)
                                 (stripe-label-fn #'stripe-label))
  "Return the number of classification errors and the number of
examples. The length of EXAMPLES must be equal to the number of
stripes in STRIPED. LABEL-FN takes an example and returns its label
that compared by EQL to what STRIPE-LABEL-FN returns for STRIPED and
the index of the stripe."
  (assert (= (length examples) (n-stripes striped)))
  (let ((n 0))
    (loop for example in examples
          for stripe upfrom 0
          do  (unless (= (funcall label-fn example)
                         (funcall stripe-label-fn striped stripe))
                (incf n)))
    (values n (length examples))))

(defclass labeled () ()
  (:documentation "Mixin for chunks/whatever that hold labels."))

(defclass softmax-label-chunk (softmax-chunk labeled) ())

(defgeneric maybe-make-misclassification-measurer (obj)
  (:documentation "Return a function of one parameter that it is
invoked when OBJ has the label(s) computed and it counts
misclassifications.")
  (:method (obj)
    (values nil nil))
  (:method ((labeled labeled))
    (lambda (examples)
      (count-misclassifications examples labeled))))

(defun max-position (array start end)
  (position (loop for i upfrom start below end maximizing (aref array i))
            array :start start :end end))

(defgeneric stripe-label (striped stripe)
  (:documentation "Return the label of STRIPE in STRIPED. Typically
computed by finding the label with the maximum propability.")
  (:method ((chunk softmax-label-chunk) stripe)
    (with-stripes ((stripe chunk start end))
      (- (max-position (storage (nodes chunk)) start end)
         start))))

(defun make-dbn-reconstruction-rmse-counters-and-measurers/no-labels
    (dbn &key (rbm (last1 (rbms dbn))))
  "Like MAKE-DBN-RECONSTRUCTION-RMSE-COUNTERS-AND-MEASURERS but don't
count reconstruction error of labels."
  (loop for i upto (position rbm (rbms dbn))
        collect (let ((i i))
                  (cons (make-instance 'rmse-counter)
                        (lambda (samples)
                          (declare (ignore samples))
                          (reconstruction-rmse
                           (remove-if (lambda (chunk)
                                        (typep chunk 'labeled))
                                      (visible-chunks (elt (rbms dbn) i)))))))))

(defun make-chunk-reconstruction-misclassification-counters-and-measurers
    (chunks)
  (loop for chunk in chunks
        for measurer = (maybe-make-misclassification-measurer chunk)
        when measurer
        collect (cons (make-instance 'error-counter)
                      measurer)))

(defun make-dbn-reconstruction-misclassification-counters-and-measurers
    (dbn &key (rbm (last1 (rbms dbn))))
  "Return a list of counter, measurer conses to keep track of
misclassifications suitable for BM-MEAN-FIELD-ERRORS."
  (make-chunk-reconstruction-misclassification-counters-and-measurers
   (apply #'append
          (mapcar #'chunks
                  (subseq (rbms dbn)
                          0 (1+ (position rbm (rbms dbn))))))))

(defun make-bm-reconstruction-misclassification-counters-and-measurers (bm)
  "Return a list of counter, measurer conses to keep track of
misclassifications suitable for BM-MEAN-FIELD-ERRORS."
  (make-chunk-reconstruction-misclassification-counters-and-measurers
   (chunks bm)))


;;;;;;;;;;

(defgeneric describe-trainer (trainer))

(defmethod describe-trainer ((trainer mgl-gd:gd-trainer))
  (log-msg "  n-inputs=~S, learning-rate=~,2E~%"
           (mgl-gd:n-inputs trainer)
           (mgl-gd:learning-rate trainer))
  (log-msg "  batch-size=~A, momentum=~,2E~%"
           (mgl-gd:batch-size trainer)
           (mgl-gd:momentum trainer))
  (log-msg "  weight-decay=~,2E, weight-penalty=~,2E,~%"
           (mgl-gd:weight-decay trainer)
           (mgl-gd:weight-penalty trainer)))

(defmethod describe-trainer ((trainer mgl-bm::segmented-gd-bm-trainer))
  (log-msg "n-gibbs: ~S~%" (mgl-rbm:n-gibbs trainer))
  (log-msg "visible-sampling: ~S~%" (mgl-rbm:visible-sampling trainer))
  (log-msg "hidden-sampling: ~S~%" (mgl-rbm:hidden-sampling trainer))
  (dolist (trainer (mgl-gd:trainers trainer))
    (mgl-train:do-segment-set (cloud) (mgl-train:segment-set trainer) 
      (log-msg "Cloud: ~S~%" (mgl-rbm:name cloud))
      (log-msg "  Trainer: ~A~%"  (class-name (class-of trainer))))
    (describe-trainer trainer)))


(defmethod log-test-error ((trainer mnist-rbm-trainer) (rbm mnist-rbm))
  (describe-trainer trainer)
  (log-msg "DBN TEST RMSE: ~{~,5F~^, ~} (~D)~%"
           (map 'list
                #'get-error
                (dbn-mean-field-errors (make-sampler *test-images*
                                                     :max-n #+nil
                                                     1000
                                                     (length *test-images*))
                                       (dbn rbm) :rbm rbm))
           (n-inputs trainer))
  (let ((counters-and-measurers
         (make-dbn-reconstruction-misclassification-counters-and-measurers
          (dbn rbm) :rbm rbm)))
    (when counters-and-measurers
      (let ((errors (map 'list
                         #'get-error
                         (dbn-mean-field-errors (make-sampler *test-images*
                                                              :max-n #+nil
                                                              1000
                                                              (length
                                                               *test-images*)
                                                              :omit-label-p t)
                                                (dbn rbm) :rbm rbm
                                                :counters-and-measurers
                                                counters-and-measurers))))
        (log-msg "DBN TEST CLASSIFICATION ACCURACY: ~{~,2F%~^, ~} (~D)~%"
                 (mapcar (lambda (e)
                           (* 100 (- 1 e)))
                         errors)
                 (n-inputs trainer))))))

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
            (if (bias-cloud-p cloud)
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
                                                 (if (bias-cloud-p cloud)
                                                     (flt 0)
                                                     (flt (this decay)))
                                                 :batch-size 100)))
                 rbm))))


;;;; BPN

(defclass mnist-bpn (bpn)
  ;; This slot is an image -> (lump array)* list hash table. When the
  ;; image is clamped in SET-INPUT onto the visible units, the the
  ;; arrays are copied over the nodes of the respective lumps. This is
  ;; not used for the DBN-to-BPN approach, but the DBN-to-DBM-to-BPN
  ;; one stores the marginals of the approximate posteriors (:MAP
  ;; lumps, in the unrolled network) here.
  ((clamping-cache
    :initform (make-hash-table :test #'eq)
    :reader clamping-cache)))

(defun clamp-image-on-bpn (bpn stripe image)
  (let ((cache (clamping-cache bpn)))
    (loop for (lump map-nodes) in (gethash image cache) do
          (with-stripes ((stripe lump lump-start lump-end))
            (declare (type flt-vector map-nodes))
            (assert (= (length map-nodes)
                       (- lump-end lump-start)))
            (replace (storage (nodes lump)) map-nodes :start1 lump-start)))))

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
            (clamp-image-on-bpn bpn stripe image)
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

(defun make-bpn (bm defs inits chunk-name)
  (let ((bpn (eval (print
                    `(build-bpn (:class 'mnist-bpn :max-n-stripes 1000)
                       ,@defs
                       ;; Add expectations
                       (expectations (input-lump :size 10))
                       ;; Add a softmax layer. Oh, the pain.
                       (prediction-weights
                        (weight-lump
                         :size (* (size (lump ',(chunk-lump-name chunk-name
                                                                 nil)))
                                  10)))
                       (prediction-biases (weight-lump :size 10))
                       (prediction-activations0
                        (activation-lump :weights prediction-weights
                                         :x (lump
                                             ',(chunk-lump-name chunk-name
                                                                nil))))
                       (prediction-activations
                        (->+ :args (list prediction-activations0
                                         prediction-biases)))
                       (predictions
                        (cross-entropy-softmax-lump
                         :group-size 10
                         :x prediction-activations
                         :target expectations))
                       (my-error (error-node :x predictions)))))))
    (initialize-bpn-from-bm bpn bm inits)
    (init-weights 'prediction-weights bpn 0.1)
    (init-weights 'prediction-biases bpn 0.1)
    bpn))


;;;; BPN training

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
      (bpn-error (make-sampler *test-images* :max-n (length *test-images*)
                               :omit-label-p t) bpn)
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
    (print inits)
    (terpri)
    (make-bpn dbn defs inits 'f3)))

(defvar *dbn/1*)
(defvar *bpn/1*)

(defun train-mnist/1 (&key load-dbn-p)
  (unless (boundp '*training-images*)
    (setq *training-images* (load-training)))
  (unless (boundp '*test-images*)
    (setq *test-images* (load-test)))
  (cond (load-dbn-p
         (setq *dbn/1* (make-mnist-dbn/1))
         (load-weights (merge-pathnames "mnist-1.dbn" *mnist-dir*) *dbn/1*)
         (log-msg "Loaded DBN~%")
         (log-msg "DBN TEST RMSE: ~{~,5F~^, ~}~%"
                  (map 'list
                       #'get-error
                       (dbn-mean-field-errors
                        (make-sampler *test-images*
                                      :max-n (length *test-images*))
                        *dbn/1*))))
        (t
         (setq *dbn/1* (make-mnist-dbn/1))
         (init-mnist-dbn *dbn/1* :stddev 0.1)
         (train-mnist-dbn *dbn/1* :n-epochs 50 :learning-rate 0.1
                          :decay 0.0002 :visible-sampling nil)
         (save-weights (merge-pathnames "mnist-1.dbn" *mnist-dir*) *dbn/1*)))
  (setq *bpn/1* (unroll-mnist-dbn/1 *dbn/1*))
  (train-mnist-bpn *bpn/1*)
  (save-weights (merge-pathnames "mnist-1.bpn" *mnist-dir*) *bpn/1*))


;;;; Code for the DBN to DBM to BPN approach (paper [2])

(defclass mnist-rbm/2 (mnist-rbm) ())

(defun clamp-labels (images chunk)
  (let ((nodes (storage (nodes chunk))))
    (loop for image in images
          for stripe upfrom 0
          do (destructuring-bind (image &key omit-label-p sample-visible-p)
                 image
               (declare (ignore sample-visible-p))
               (with-stripes ((stripe chunk start end))
                 (cond (omit-label-p
                        (fill nodes #.(flt 0.1) :start start :end end))
                       (t
                        (fill nodes (flt 0) :start start :end end)
                        (setf (aref nodes (+ start (image-label image)))
                              #.(flt 1)))))))))

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
      (lambda (examples)
        (funcall measurer (mapcar #'first examples))))))

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

(defmethod set-input (images (dbm mnist-dbm))
  (clamp-striped-nodes images (mgl-bm:find-chunk 'inputs dbm))
  (when (and images (getf (rest (elt images 0)) :sample-visible-p))
    (sample-visible dbm))
  (let ((label-chunk (find-chunk 'label dbm :errorp t)))
    (clamp-labels images label-chunk)))

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
  (describe-trainer trainer)
  (save-weights (merge-pathnames (format nil "mnist-2-~A.dbm"
                                         (floor (n-inputs trainer)
                                                (length *training-images*)))
                                 *mnist-dir*)
                *dbm/2*)
  (log-msg "DBM TEST RMSE: ~{~,5F~^, ~} (~D)~%"
           (map 'list
                #'get-error
                (bm-mean-field-errors (make-sampler *test-images*
                                                    :max-n #+nil
                                                    1000
                                                    (length *test-images*))
                                      dbm))
           (n-inputs trainer))
  (let ((counters-and-measurers
         (make-bm-reconstruction-misclassification-counters-and-measurers dbm)))
    (when counters-and-measurers
      (let ((errors (map 'list
                         #'get-error
                         (bm-mean-field-errors (make-sampler *test-images*
                                                             :max-n
                                                             10000
                                                             #+nil
                                                             (length
                                                              *training-images*)
                                                             :omit-label-p t)
                                               dbm
                                               :counters-and-measurers
                                               counters-and-measurers))))
        (log-msg "DBM TRAINING CLASSIFICATION ACCURACY: ~{~,2F%~^, ~} (~D)~%"
                 (mapcar (lambda (e)
                           (* 100 (- 1 e)))
                         errors)
                 (n-inputs trainer)))))
  (let ((counters-and-measurers
         (make-bm-reconstruction-misclassification-counters-and-measurers dbm)))
    (when counters-and-measurers
      (let ((errors (map 'list
                         #'get-error
                         (bm-mean-field-errors (make-sampler *test-images*
                                                             :max-n #+nil
                                                             1000
                                                             (length
                                                              *test-images*)
                                                             :omit-label-p t)
                                               dbm
                                               :counters-and-measurers
                                               counters-and-measurers))))
        (log-msg "DBM TEST CLASSIFICATION ACCURACY: ~{~,2F%~^, ~} (~D)~%"
                 (mapcar (lambda (e)
                           (* 100 (- 1 e)))
                         errors)
                 (n-inputs trainer))))))

(defmethod positive-phase :around (batch trainer (dbm mnist-dbm))
  (call-next-method)
  (set-visible-mean dbm)
  (multiple-value-call #'add-error (counter trainer)
                       (reconstruction-rmse
                        (remove-if (lambda (chunk)
                                     (typep chunk 'labeled))
                                   (visible-chunks dbm)))))

(defclass mnist-dbm-segment-trainer (batch-gd-trainer) ())

;; (defun linear (a b x)
;;   (+ a (* b x)))

;; (defun linear-at-points (x1 y1 x2 y2 x)
;;   (let ((b (/ (- y2 y1) (- x2 x1))))
;;     (linear (- (* b (- x1 (/ y1 b)))) b x)))

;; (assert (= 0 (linear-at-points 3 1 5 2 1)))
;; (assert (= 1 (linear-at-points 3 1 5 2 3)))
;; (assert (= 2 (linear-at-points 3 1 5 2 5)))
;; (assert (= 3 (linear-at-points 3 1 5 2 7)))

(defmethod learning-rate ((trainer mnist-dbm-segment-trainer))
  ;; This is adjusted for each batch. Ruslan's code adjusts it per
  ;; epoch.
  ;; FIXME: min 0.0001
  (/ (slot-value trainer 'learning-rate)
     (expt 1.000015 (* (/ (n-inputs trainer)
                          (length *training-images*))
                       600))))

#+nil
(defmethod n-gibbs ((trainer mnist-dbm-trainer))
  (cond ((< (n-inputs trainer) 5000)
         100)
        (t
         10)))

(defmethod momentum ((trainer mnist-dbm-segment-trainer))
  (if (< (n-inputs trainer) (* 5 (length *training-images*)))
      #.(flt 0.5)
      #.(flt 0.9)))

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
                          (when (or t (equal 'label (name (chunk1 cloud)))
                                    (equal 'label (name (chunk2 cloud))))
                            (make-instance 'mnist-dbm-segment-trainer
                                           :learning-rate (flt 0.001)
                                           :weight-decay
                                           (if (bias-cloud-p cloud)
                                               (flt 0)
                                               (flt 0.0002))
                                           :batch-size 100)))
                        :sparser
                        (lambda (cloud chunk)
                          (when (and (member (name chunk) '(f1 f2))
                                     (not (equal 'label (name (chunk1 cloud))))
                                     (not (equal 'label (name (chunk2 cloud)))))
                            (make-instance
                             'sparsity-gradient-source
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

(defun unroll-mnist-dbm (dbm)
  (multiple-value-bind (defs inits) (unroll-dbm dbm :excluded-chunks '(label))
    (prin1 inits)
    (terpri)
    (make-bpn dbm defs inits 'f2)))

(defun collect-map-chunks-and-lumps (bpn dbm)
  (let ((chunks-and-lumps ()))
    (dolist (chunk (hidden-chunks dbm))
      (unless (typep chunk 'conditioning-chunk)
        (let* ((lump-name (chunk-lump-name (name chunk) :map))
               (lump (find-lump lump-name bpn)))
          (when lump
            (push (list chunk lump) chunks-and-lumps)))))
    chunks-and-lumps))

(defun populate-map-cache (bpn dbm images)
  (let ((sampler (make-sampler images :max-n (length images)
                               :sample-visible-p nil
                               :omit-label-p t))
        (cache (clamping-cache bpn))
        (map-chunks-and-lumps (collect-map-chunks-and-lumps bpn dbm)))
    (let ((fn (make-instance 'periodic-fn :period 1000
                             :fn (lambda (n)
                                   (log-msg "populated: ~S~%" n))))
          (n 0))
      (do-batches-for-learner (samples (sampler dbm))
        (call-periodic-fn n fn n)
        (set-input samples dbm)
        (set-hidden-mean dbm)
        (loop for image in samples
              for stripe upfrom 0 do
              (let ((x ()))
                (loop for (chunk lump) in map-chunks-and-lumps
                      do
                      (with-stripes ((stripe chunk chunk-start chunk-end))
                        (let ((xxx (make-flt-array
                                    (- chunk-end chunk-start))))
                          (replace xxx (storage (nodes chunk))
                                   :start2 chunk-start :end2 chunk-end)
                          (push (list lump xxx) x))))
                (setf (gethash image cache) x)))
        (incf n (length samples)))
      (call-periodic-fn n fn n))))

(defvar *dbn/2*)
(defvar *dbm/2*)
(defvar *bpn/2*)

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
           (save-weights (merge-pathnames "mnist-2.dbn" *mnist-dir*)
                         *dbn/2*))
         (train-dbm ()
           (train-mnist-dbm *dbm/2*)
           (save-weights (merge-pathnames "mnist-2.dbm" *mnist-dir*)
                         *dbm/2*)))
    (cond (load-dbm-p
           (setq *dbm/2* (make-mnist-dbm))
           (load-weights (merge-pathnames "mnist-2.dbm" *mnist-dir*) *dbm/2*))
          (load-dbn-p
           (setq *dbm/2* (make-mnist-dbm))
           (setq *dbn/2* (make-mnist-dbn/2 *dbm/2*))
           (load-weights (merge-pathnames "mnist-2.dbn" *mnist-dir*) *dbn/2*)
           (log-msg "Loaded DBN~%")
           (log-msg "DBN TEST RMSE: ~{~,5F~^, ~}~%"
                    (map 'list
                         #'get-error
                         (dbn-mean-field-errors
                          (make-sampler *test-images*
                                        :max-n (length *test-images*))
                          *dbn/2*)))
           (train-dbm))
          (t
           (setq *dbm/2* (make-mnist-dbm))
           (setq *dbn/2* (make-mnist-dbn/2 *dbm/2*))
           (train-dbn)
           (train-dbm)))
    (setq *bpn/2* (unroll-mnist-dbm *dbm/2*))
    (log-msg "Populating MAP cache~%")
    (populate-map-cache *bpn/2* *dbm/2* (concatenate 'vector *training-images*
                                                     *test-images*))
    (train-mnist-bpn *bpn/2* :batch-size 10000)
    (save-weights (merge-pathnames "mnist-2.bpn" *mnist-dir*)
                  *bpn/2*)))

#|

(train-mnist/1)

(train-mnist/1 :load-dbn-p t)

(train-mnist/2)

(train-mnist/2 :load-dbn-p t)

(train-mnist/2 :load-dbm-p t)

(defmethod negative-phase (batch (trainer bm-pcd-trainer) bm)
  (check-no-self-connection bm)
  (loop repeat (n-gibbs trainer) do
        (dolist (chunk (elt (layers bm) 0))
          (sample-chunk chunk))
        (dolist (chunk (elt (layers bm) 2))
          (sample-chunk chunk))
        (dolist (chunk (elt (layers bm) 1))
          (set-mean (list chunk) bm)
          (sample-chunk chunk))
        (dolist (chunk (elt (layers bm) 0))
          (set-mean (list chunk) bm))
        (dolist (chunk (elt (layers bm) 2))
          (set-mean (list chunk) bm)))
  (accumulate-negative-phase-statistics
   trainer bm
   ;; The number of persistent chains (or fantasy particles), that is,
   ;; N-STRIPES of PERSISTENT-CHAINS is not necessarily the same as
   ;; the batch size. Normalize so that positive and negative phase
   ;; has the same weight.
   :multiplier (/ (length batch)
                  (n-stripes (persistent-chains trainer)))))

(require :sb-sprof)

(progn
  (sb-sprof:reset)
  (sb-sprof:start-profiling)
  (sleep 10)
  (sb-sprof:stop-profiling)
  (sb-sprof:report :type :graph))

|#
