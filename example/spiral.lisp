(in-package :mgl-example-spiral)

;;;; Sampling, clamping

(defun sample-spiral ()
  (random (flt (* 4 pi))))

(defun make-sampler (n)
  (make-instance 'counting-function-sampler
                 :max-n-samples n
                 :sampler #'sample-spiral))

(defun clamp-array (x array start)
  (setf (aref array (+ start 0)) x
        (aref array (+ start 1)) (sin x)
        (aref array (+ start 2)) (cos x)))

(defun clamp-rbm (sample rbm)
  (let ((chunk (find 'inputs (visible-chunks rbm) :key #'name)))
    (when chunk
      (clamp-array sample (nodes chunk) 0))))

(defun clamp-bpn (sample bpn)
  (multiple-value-bind (array start)
      (lump-node-array (find-lump '(inputs 0) bpn))
    (clamp-array sample array start)))

(defclass spiral-rbm (rbm) ())

(defmethod mgl-train:set-input (sample (rbm spiral-rbm))
  (clamp-rbm sample rbm))

(defclass spiral-bpn (bpn) ())

(defmethod mgl-train:set-input (sample (bpn spiral-bpn))
  (clamp-bpn sample bpn))


;;;; Progress reporting

(defclass spiral-rbm-trainer (rbm-trainer)
  ((counter :initform (make-instance 'rmse-counter) :reader counter)))

(defclass spiral-bp-trainer (bp-trainer)
  ((counter :initform (make-instance 'rmse-counter) :reader counter)))

(defun report-dbn-rmse (rbm trainer)
  (format *trace-output* "DBN RMSE: ~{~,5F~^, ~} (~D)~%"
          (coerce (dbn-rmse (make-sampler 1000) (dbn rbm) :rbm rbm) 'list)
          (n-inputs trainer)))

;;; This prints the rmse of the the training examples after each 100
;;; and the test rmse on each level of the DBN after each 1000.
(defmethod train-one (sample (trainer spiral-rbm-trainer) rbm &key)
  (when (zerop (mod (n-inputs trainer) 1000))
    (report-dbn-rmse rbm trainer))
  (let ((counter (counter trainer)))
    (multiple-value-bind (e n) (mgl-rbm:reconstruction-error rbm)
      (add-error counter e n))
    (call-next-method)
    (let ((n-inputs (n-inputs trainer)))
      (when (zerop (mod n-inputs 100))
        (format *trace-output* "RMSE: ~,5F (~D, ~D)~%"
                (or (get-error counter) #.(flt 0))
                (n-sum-errors counter)
                n-inputs)
        (force-output *trace-output*)
        (reset-counter counter)))))

;;; Make sure we print the test rmse at the end of training each rbm.
(defmethod train :after (sampler (trainer spiral-rbm-trainer) rbm &key)
  (report-dbn-rmse rbm trainer))

(defun bpn-rmse (sampler bpn)
  (let ((sum-errors #.(flt 0))
        (n-errors 0))
    (loop until (finishedp sampler) do
          (set-input (sample sampler) bpn)
          (forward-bpn bpn)
          (incf sum-errors (last1 (nodes bpn)))
          (incf n-errors 3))
    (if (zerop n-errors)
        nil
        (sqrt (/ sum-errors n-errors)))))

(defun report-bpn-rmse (bpn trainer)
  (format *trace-output* "BPN RMSE: ~,5F (~D)~%"
          (bpn-rmse (make-sampler 1000) bpn) (n-inputs trainer)))

(defmethod train-one (sample (trainer spiral-bp-trainer) bpn &key)
  (when (zerop (mod (n-inputs trainer) 1000))
    (report-bpn-rmse bpn trainer))
  (let ((counter (counter trainer)))
    (call-next-method)
    (add-error counter (last1 (nodes bpn)) 3)
    (let ((n-inputs (n-inputs trainer)))
      (when (zerop (mod n-inputs 100))
        (format *trace-output* "RMSE: ~,5F (~D, ~D)~%"
                (or (get-error counter) #.(flt 0))
                (n-sum-errors counter)
                n-inputs)
        (force-output *trace-output*)
        (reset-counter counter)))))

(defmethod train :after (sampler (trainer spiral-bp-trainer) bpn &key)
  (report-bpn-rmse bpn trainer))


;;;; Training

(defun layers->rbms (layers &key (class 'rbm))
  (loop for (v h) on layers
        when h
        collect (make-instance class :visible-chunks v :hidden-chunks h)))

(defclass spiral-dbn (dbn)
  ((rbms
    :initform (layers->rbms
               (list
                (list (make-instance 'constant-chunk :name 'c0)
                      (make-instance 'gaussian-chunk :name 'inputs :size 3))
                (list (make-instance 'constant-chunk :name 'c1)
                      (make-instance 'sigmoid-chunk :name 'f1 :size 10))
                (list (make-instance 'constant-chunk :name 'c2)
                      (make-instance 'gaussian-chunk :name 'f2 :size 1)))
               :class 'spiral-rbm))))

(defun make-spiral-dbn ()
  (make-instance 'spiral-dbn))

(defun train-spiral-dbn ()
  (let ((dbn (make-spiral-dbn)))
    (dolist (rbm (rbms dbn))
      (train (make-sampler 50000)
             (make-instance 'spiral-rbm-trainer
                            :segmenter
                            (repeatedly (make-instance 'batch-gd-trainer
                                                       :batch-size 100)))
             rbm))
    dbn))

(defclass spiral-bpn (bpn) ())

(defun train-spiral-bpn (dbn)
  (multiple-value-bind (defs clamps inits) (unroll-dbn dbn)
    (declare (ignore clamps))
    (print inits)
    (let ((bpn (eval (print
                      `(build-bpn (:class 'spiral-bpn)
                         ,@defs
                         (error-node
                          :def (->sum-squared-error (_)
                                 (lump '(inputs 0))
                                 (lump '(inputs 0 :reconstruction)))))))))
      (initialize-bpn-from-dbn bpn dbn inits)
      (train (make-sampler 50000)
             (make-instance 'spiral-bp-trainer
                            :segmenter
                            (repeatedly
                              (make-instance 'batch-gd-trainer
                                             :learning-rate (flt 0.01)
                                             :batch-size 100)))
             bpn))))

#|

(defparameter *spiral-dbn* (time (train-spiral-dbn)))

(defparameter *spiral-bpn* (time (train-spiral-bpn *spiral-dbn*)))

|#
