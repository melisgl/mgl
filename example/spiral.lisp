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
  (clamp-striped-nodes samples (find-lump (chunk-lump-name 'inputs nil) bpn)))


;;;; Progress reporting

(defclass spiral-base-trainer (base-trainer) ())

(defmethod log-training-period ((trainer spiral-base-trainer) learner)
  1000)

(defmethod log-test-period ((trainer spiral-base-trainer) learner)
  10000)


;;;; DBN training

(defclass spiral-rbm-trainer (spiral-base-trainer rbm-cd-trainer) ())

(defmethod initialize-trainer ((trainer spiral-rbm-trainer) (rbm spiral-rbm))
  (call-next-method)
  (setf (slot-value trainer 'training-counters-and-measurers)
        (prepend-name-to-counters
         "dbn train: training"
         (make-bm-reconstruction-rmse-counters-and-measurers rbm))))

(defmethod log-test-error ((trainer spiral-rbm-trainer) (rbm spiral-rbm))
  (call-next-method)
  (map nil (lambda (counter)
             (log-msg "dbn test ~:_~A~%" counter))
       (collect-dbn-mean-field-errors (make-sampler 1000) (dbn rbm) :rbm rbm)))


;;;; BPN training

(defun bpn-rmse (bpn)
  ;; rmse is over 3 nodes, cost returns the number of stripes
  (multiple-value-bind (e n) (cost bpn)
    (values e (* n 3))))

(defun bpn-error (sampler bpn)
  (collect-bpn-errors sampler bpn
                      :counters-and-measurers
                      (list (cons (make-instance 'rmse-counter)
                                  (lambda (samples bpn)
                                    (declare (ignore samples))
                                    (bpn-rmse bpn))))))

(defclass spiral-bp-trainer (spiral-base-trainer bp-trainer) ())

(defun make-bpn-counters-and-measurers ()
  (list (cons (make-instance 'rmse-counter)
              (lambda (samples bpn)
                (declare (ignore samples))
                (bpn-rmse bpn)))))

(defmethod initialize-trainer ((trainer spiral-bp-trainer) (bpn spiral-bpn))
  (call-next-method)
  (setf (slot-value trainer 'training-counters-and-measurers)
        (prepend-name-to-counters "bpn train: training"
                                  (make-bpn-counters-and-measurers))))

(defmethod log-test-error ((trainer spiral-bp-trainer) (bpn spiral-bpn))
  (call-next-method)
  (map nil (lambda (counter)
             (log-msg "bpn test: test ~:_~A~%" counter))
       (bpn-error (make-sampler 1000) bpn)))


;;;; Training

(defun layers->rbms (layers &key (class 'rbm))
  (loop for (v h) on layers
        when h
        collect (make-instance class :visible-chunks v :hidden-chunks h)))

(defclass spiral-dbn (dbn)
  ()
  (:default-initargs
   :layers (list (list (make-instance 'constant-chunk :name 'c0)
                       (make-instance 'gaussian-chunk :name 'inputs :size 3))
                 (list (make-instance 'constant-chunk :name 'c1)
                       (make-instance 'sigmoid-chunk :name 'f1 :size 10))
                 (list (make-instance 'constant-chunk :name 'c2)
                       (make-instance 'gaussian-chunk :name 'f2 :size 1)))
    :rbm-class 'spiral-rbm))

(defun make-spiral-dbn (&key (max-n-stripes 1))
  (make-instance 'spiral-dbn :max-n-stripes max-n-stripes))

(defun train-spiral-dbn (&key (max-n-stripes 1))
  (let ((dbn (make-spiral-dbn :max-n-stripes max-n-stripes)))
    (dolist (rbm (rbms dbn))
      (train (make-sampler 50000)
             (make-instance 'spiral-rbm-trainer
                            :segmenter
                            (repeatedly (make-instance 'batch-gd-trainer
                                                       :momentum (flt 0.9)
                                                       :batch-size 100)))
             rbm))
    dbn))

(defun train-spiral-bpn (dbn &key (max-n-stripes 1))
  (multiple-value-bind (defs inits) (unroll-dbn dbn)
    (log-msg "inits:~%~S~%" inits)
    (let ((bpn-def `(build-bpn (:class 'spiral-bpn
                                       :max-n-stripes ,max-n-stripes)
                      ,@defs
                      (my-error
                       (error-node
                        :x (make-instance
                            '->sum-squared-error
                            :x (lump ',(chunk-lump-name 'inputs nil))
                            :y (lump ',(chunk-lump-name
                                        'inputs
                                        :reconstruction))))))))
      (log-msg "bpn def:~%~S~%" bpn-def)
      (let ((bpn (eval bpn-def)))
        (initialize-bpn-from-bm bpn dbn inits)
        (train (make-sampler 50000)
               (make-instance 'spiral-bp-trainer
                              :segmenter
                              (repeatedly
                                (make-instance 'batch-gd-trainer
                                               :learning-rate (flt 0.01)
                                               :momentum (flt 0.9)
                                               :batch-size 100)))
               bpn)))))

#|

(defparameter *spiral-dbn* (time (train-spiral-dbn :max-n-stripes 100)))

(defparameter *spiral-bpn* (time (train-spiral-bpn *spiral-dbn*
                                                   :max-n-stripes 100)))

|#
