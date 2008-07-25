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

(defun clamp-striped-nodes (samples striped)
  (let ((nodes (storage (nodes striped))))
    (loop for sample in samples
          for stripe upfrom 0
          do (with-stripes ((stripe striped start))
               (clamp-array sample nodes start)))))

(defclass spiral-rbm (rbm) ())

(defmethod mgl-train:set-input (samples (rbm spiral-rbm))
  (let ((chunk (find 'inputs (visible-chunks rbm) :key #'name)))
    (when chunk
      (clamp-striped-nodes samples chunk))))

(defclass spiral-bpn (bpn) ())

(defmethod mgl-train:set-input (samples (bpn spiral-bpn))
  (clamp-striped-nodes samples (find-lump '(inputs 0) bpn)))


;;;; Progress reporting

(defclass spiral-rbm-trainer (rbm-trainer)
  ((counter :initform (make-instance 'rmse-counter) :reader counter)))

(defclass spiral-bp-trainer (bp-trainer)
  ((counter :initform (make-instance 'rmse-counter) :reader counter)))

(defun report-dbn-rmse (rbm trainer)
  (log-msg "DBN RMSE: ~{~,5F~^, ~} (~D)~%"
           (coerce (dbn-rmse (make-sampler 1000) (dbn rbm) :rbm rbm) 'list)
           (n-inputs trainer)))

;;; This prints the rmse of the the training examples after each 100
;;; and the test rmse on each level of the DBN after each 1000.
(defmethod train-batch :around (batch (trainer spiral-rbm-trainer) rbm)
  (when (zerop (mod (n-inputs trainer) 1000))
    (report-dbn-rmse rbm trainer))
  (let ((counter (counter trainer)))
    (multiple-value-bind (e n) (mgl-rbm:reconstruction-error rbm)
      (add-error counter e n))
    (call-next-method)
    (let ((n-inputs (n-inputs trainer)))
      (when (zerop (mod n-inputs 100))
        (log-msg "RMSE: ~,5F (~D, ~D)~%"
                 (or (get-error counter) #.(flt 0))
                 (n-sum-errors counter)
                 n-inputs)
        (force-output *trace-output*)
        (reset-counter counter)))))

;;; Make sure we print the test rmse at the end of training each rbm.
(defmethod train :after (sampler (trainer spiral-rbm-trainer) rbm)
  (report-dbn-rmse rbm trainer))

(defun bpn-rmse (sampler bpn)
  (let ((counter (make-instance 'rmse-counter))
        (n-stripes (max-n-stripes bpn)))
    (while (not (finishedp sampler))
      (set-input (sample-batch sampler n-stripes) bpn)
      (forward-bpn bpn)
      (multiple-value-bind (e n) (cost bpn)
        (add-error counter e (* n 3))))
    (values (get-error counter) counter)))

(defun report-bpn-rmse (bpn trainer)
  (log-msg "BPN RMSE: ~,5F (~D)~%"
           (bpn-rmse (make-sampler 1000) bpn) (n-inputs trainer)))

(defmethod train-batch :around (batch (trainer spiral-bp-trainer) bpn)
  (when (zerop (mod (n-inputs trainer) 1000))
    (report-bpn-rmse bpn trainer))
  (let ((counter (counter trainer)))
    (call-next-method)
    (multiple-value-bind (e n) (cost bpn)
      (add-error counter e (* n 3)))
    (let ((n-inputs (n-inputs trainer)))
      (when (zerop (mod n-inputs 100))
        (log-msg "RMSE: ~,5F (~D, ~D)~%"
                 (or (get-error counter) #.(flt 0))
                 (n-sum-errors counter)
                 n-inputs)
        (reset-counter counter)))))

(defmethod train :after (sampler (trainer spiral-bp-trainer) bpn)
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

(defun make-spiral-dbn (&key (max-n-stripes 1))
  (make-instance 'spiral-dbn :max-n-stripes max-n-stripes))

(defun train-spiral-dbn (&key (max-n-stripes 1))
  (let ((dbn (make-spiral-dbn :max-n-stripes max-n-stripes)))
    (dolist (rbm (rbms dbn))
      (train (make-sampler 50000)
             (make-instance 'spiral-rbm-trainer
                            :segmenter
                            (repeatedly (make-instance 'batch-gd-trainer
                                                       :batch-size 100)))
             rbm))
    dbn))

(defclass spiral-bpn (bpn) ())

(defun train-spiral-bpn (dbn &key (max-n-stripes 1))
  (multiple-value-bind (defs clamps inits) (unroll-dbn dbn)
    (declare (ignore clamps))
    (print inits)
    (let ((bpn (eval (print
                      `(build-bpn (:class 'spiral-bpn
                                          :max-n-stripes ,max-n-stripes)
                         ,@defs
                         (my-error
                          (error-node
                           :x (make-instance
                               '->sum-squared-error
                               :x (lump '(inputs 0))
                               :y (lump '(inputs 0 :reconstruction))))))))))
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

(defparameter *spiral-dbn* (time (train-spiral-dbn :max-n-stripes 100)))

(defparameter *spiral-bpn* (time (train-spiral-bpn *spiral-dbn*
                                                   :max-n-stripes 100)))

|#
