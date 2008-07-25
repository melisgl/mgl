(in-package :mgl-test)

#+nil
(defun test-cloud-activation (&key transposep cg)
  (let* ((net (build-bpn (:class 'test-bpn)
                (input-lump :symbol inputs :size 2)
                (weight-lump :symbol biases :size 2)
                (hidden-lump :symbol biased-input :size 2
                             :def (->+ (_) (ref inputs _)
                                       (ref biases _)))
                (weight-lump :symbol weights :size (* 3 2))
                (activation-lump :symbol activations :size 3
                                 :transpose-weights-p transposep
                                 :input-lump biased-input
                                 :weight-lump weights)
                (mgl-unroll-dbn::cloud-activation-lump
                 :symbol activations2 :size 3
                 :transpose-weights-p transposep
                 :input-lump biased-input
                 :weight-lump weights
                 :punt-to-activation-lump-p nil)
                (input-lump :symbol expectations :size 3)
                (error-node :def (->sum-squared-error (_)
                                   expectations
                                   activations))
                (error-node :def (->sum-squared-error (_)
                                   expectations
                                   activations2))))
         (nodes (nodes net))
         (input-start (lump-start (find-lump 'inputs net)))
         (a1-s (lump-start (find-lump 'activations net)))
         (a1-e (lump-end (find-lump 'activations net)))
         (a2-s (lump-start (find-lump 'activations2 net)))
         (a2-e (lump-end (find-lump 'activations2 net)))
         (derivatives (derivatives net))
         (expectations-start (lump-start (find-lump 'expectations net))))
    (init-lump 'biases net 0.1)
    (init-lump 'weights net 0.1)
    (flet ((sample ()
             (list (random (flt 1)) (random (flt 1))))
           (clamper (s bpn1)
             (assert (eq bpn1 net))
             (assert (every #'=
                            (subseq nodes a1-s a1-e)
                            (subseq nodes a2-s a2-e)))
             (assert (every #'=
                            (subseq derivatives a1-s a1-e)
                            (subseq derivatives a2-s a2-e)))
             (destructuring-bind (x y) s
               (setf (aref nodes input-start) x)
               (setf (aref nodes (1+ input-start)) y)
               (setf (aref nodes expectations-start)
                     (+ (* 3 (+ x 2))
                        (* 2 (+ y -1))))
               (setf (aref nodes (1+ expectations-start))
                     (+ (* -3 (+ x 2))
                        (* -2 (+ y -1))))
               (setf (aref nodes (+ 2 expectations-start))
                     (+ (* -0.8 (+ x 2))
                        (* 1.3 (+ y -1)))))))
      (setf (clamper net) #'clamper)
      (mgl-train:train
       (make-instance 'mgl-train:counting-function-sampler
                      :max-n-samples (if cg 50000 100000)
                      :sampler #'sample)
       (if cg
           (make-instance 'test-cg-bp-trainer
                          :cg-args '(:max-n-line-searches 3)
                          :batch-size 1000)
           (make-instance 'test-bp-trainer
                          :segmenter
                          (repeatedly
                            (make-instance 'batch-gd-trainer
                                           :learning-rate (flt 0.01)
                                           :batch-size 1000))))
       net)
      (bpn-error (make-instance 'counting-function-sampler
                                :max-n-samples 1000
                                :sampler #'sample)
                 net))))

(defun test-unroll ()
  (let ((rbm (make-instance 'rbm
                            :visible-chunks (list
                                             (make-instance 'constant-chunk
                                                            :name 'constant)
                                             (make-instance 'gaussian-chunk
                                                            :name 'inputs
                                                            :size 10))
                            :hidden-chunks (list
                                            (make-instance 'constant-chunk
                                                           :name 'constant)
                                            (make-instance 'sigmoid-chunk
                                                           :name 'features
                                                           :size 2)))))
    (unroll-dbn (make-instance 'dbn :rbms (list rbm)))))

(defun test-unroll-dbn ()
  (test-unroll)
  #+nil
  (dolist (transposep '(nil t))
    (dolist (cg '(nil t))
      (assert (> 0.5 (test-cloud-activation :transposep transposep :cg cg))))))
