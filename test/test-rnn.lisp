(in-package :mgl-test)

(defun make-sum-sign-instance ()
  (let ((length (1+ (random 10)))
        (sum 0))
    (loop for i below length
          collect (let ((x (1- (* 2 (random 2)))))
                    (incf sum x)
                    (when (< x 0)
                      (setq sum x))
                    (list x (1+ (sign (flt sum))) (/ length))))))

;;; We have a batch of sum-sign instances from MAKE-SUM-SIGN-INSTANCE
;;; for the RNN. This functions clamps elements belonging to the same
;;; time step (i.e. at the same index) in the sum-sign instances.
(defun clamp-sum-sign (step-instances fnn)
  (let ((inputs (nodes (find-clump 'inputs fnn))))
    (setf (target (find-clump 'prediction fnn))
          (loop for stripe upfrom 0
                for step-instance in step-instances
                collect (when step-instance
                          (destructuring-bind (input expectation importance)
                              step-instance
                            (setf (mref inputs stripe 0) input)
                            (list (cons (round expectation) importance))))))))

(defun name-in-p (name seq)
  (find name seq :test #'name=))

(defclass test-batch-gd-optimizer (batch-gd-optimizer)
  ())

(defmethod learning-rate ((optimizer test-batch-gd-optimizer))
  (let ((x (slot-value optimizer 'learning-rate)))
    (if (numberp x)
        x
        (funcall x optimizer))))

(defun test-sum-sign-rnn (&key (max-n-stripes 100) (n-hiddens 1) use-lstm-p)
  (let ((net (build-rnn ()
               (build-fnn (:class 'test-fnn
                           :initargs '(:clamper clamp-sum-sign))
                 (inputs (->input :size 1))
                 (h (if use-lstm-p
                        (->lstm inputs :name 'h :size n-hiddens
                                :input-fn '->identity
                                :output-fn '->identity)
                        (->activation (if (zerop (time-step))
                                          inputs
                                          (list inputs (lag '(h :activation))))
                                      :name 'h :size n-hiddens)))
                 (prediction (->softmax-xe-loss
                              (->activation h :name 'prediction :size 3)))))))
    (setf (max-n-stripes net) max-n-stripes)
    (map-segments (lambda (lump)
                    (let ((name (name lump))
                          (nodes (nodes lump)))
                      (print (name lump))
                      (cond ((find :bias name)
                             (fill! 0 nodes))
                            ((find :peephole name)
                             (fill! 0 nodes))
                            ((eq 'h (first name))
                             (fill! 0 nodes))
                            (t
                             (orthogonal-random! (nodes lump) :scale 1.01)))))
                  net)
    #+nil
    (progn
      (let ((dgraph (cl-dot:generate-graph-from-roots
                     (aref (mgl-bp::bpns net) 0)
                     (lumps (aref (mgl-bp::bpns net) 0)))))
        (cl-dot:dot-graph dgraph
                          (asdf-system-relative-pathname "bpn0.png")
                          :format :png))
      (let ((dgraph (cl-dot:generate-graph-from-roots
                     (aref (mgl-bp::bpns net) 1)
                     (lumps (aref (mgl-bp::bpns net) 1)))))
        (cl-dot:dot-graph dgraph
                          (asdf-system-relative-pathname "bpn1.png")
                          :format :png)))
    (log-msg "test cost: ~S~%"
             (bpn-error (make-instance 'function-sampler
                                       :max-n-samples 1000
                                       :generator #'make-sum-sign-instance)
                        net :counter-type 'cross-entropy-counter))
    (minimize (monitor-training-periodically
               (make-instance
                'test-bp-optimizer
                :segmenter
                (lambda (lump)
                  (cond ((name-in-p
                          (name lump)
                          '((:bias (h :input))
                            (inputs (h :input))
                            (:bias (h :forget))
                            (inputs (h :forget))
                            (:bias (h :output))
                            (inputs (h :output))
                            (:bias prediction)
                            (h prediction)))
                         (make-instance 'test-batch-gd-optimizer
                                        :learning-rate 1
                                        :momentum 0.9
                                        :momentum-type :nesterov
                                        :batch-size 100))
                        (t
                         (make-instance 'test-batch-gd-optimizer
                                        :learning-rate 0.1
                                        :momentum 0.9
                                        :momentum-type :nesterov
                                        :batch-size 100))))))
              (make-bp-learner net :counter-type 'cross-entropy-counter)
              :dataset
              (make-instance 'function-sampler
                             :max-n-samples 10000
                             :generator #'make-sum-sign-instance))
    (map-segments (lambda (weights)
                    (format t "~S ~S~%" (name weights) (nodes weights)))
                  net)
    (bpn-error (make-instance 'function-sampler
                              :max-n-samples 1000
                              :generator #'make-sum-sign-instance)
               net :counter-type 'cross-entropy-counter)))

(defun test-rnn ()
  (declare (optimize (debug 3)))
  (do-cuda ()
    (dolist (*default-mat-ctype* '(:float :double))
      (dolist (max-n-stripes '(10 100))
        (dolist (use-lstm-p '(nil t))
          (log-msg "cuda: ~S, max-n-stripes: ~S, use-lstm-p: ~S~%"
                   (boundp 'cl-cuda.api.context:*cuda-context*)
                   max-n-stripes use-lstm-p)
          (assert (> 0.01
                     (test-sum-sign-rnn :max-n-stripes max-n-stripes
                                        :use-lstm-p use-lstm-p))))))))
