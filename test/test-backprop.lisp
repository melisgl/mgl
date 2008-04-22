(in-package :mgl-test)

(defclass test-bp-trainer (bp-trainer)
  ((counter :initform (make-instance 'error-counter) :reader counter)))

(defclass test-cg-bp-trainer (cg-bp-trainer)
  ((counter :initform (make-instance 'error-counter) :reader counter)))

(defclass test-bpn (bpn)
  ((clamper :initarg :clamper :accessor clamper)))

(defmethod set-input (sample (bpn test-bpn))
  (funcall (clamper bpn) sample bpn))

(defun cost (bpn)
  (last1 (nodes bpn)))

(defmethod train-one :around (sample (trainer test-bp-trainer) bpn &key)
  (let ((counter (counter trainer)))
    (call-next-method)
    (add-error counter (cost bpn) 1)
    (let ((n-inputs (n-inputs trainer)))
      (when (zerop (mod n-inputs 10000))
        (log-msg "RMSE: ~,5F (~D, ~D)~%"
                 (or (get-error counter) #.(flt 0))
                 (n-sum-errors counter)
                 n-inputs)
        (reset-counter counter)))))

(defmethod train-batch :around (batch (trainer test-cg-bp-trainer) bpn &key)
  (let ((counter (counter trainer)))
    (call-next-method)
    (loop for sample in batch do
          (set-input sample bpn)
          (forward-bpn bpn)
          (add-error counter (cost bpn) 1))
    (let ((n-inputs (n-inputs trainer)))
      (when (zerop (mod n-inputs 10000))
        (log-msg "RMSE: ~,5F (~D, ~D)~%"
                 (or (get-error counter) #.(flt 0))
                 (n-sum-errors counter)
                 n-inputs)
        (reset-counter counter)))))

(defun bpn-error (sampler bpn)
  (let ((counter (make-instance 'error-counter)))
    (loop repeat 1000 do
          (assert (not (finishedp sampler)))
          (set-input (sample sampler) bpn)
          (forward-bpn bpn)
          (add-error counter (cost bpn) 1))
    (get-error counter)))

(defun test-simple (&key cg)
  (let* ((net (build-bpn (:class 'test-bpn)
                (weight-lump :symbol weights :size 1)
                (input-lump :symbol inputs :size 1)
                (output-lump :symbol my-linear :size 1
                             :def (->linear (_) inputs weights))
                (error-node :def (->sum-squared-error (_) inputs my-linear))))
         (nodes (nodes net))
         (derivatives (derivatives net)))
    (setf (aref nodes 0) #.(flt 2))
    (setf (aref nodes 1) #.(flt 3))
    (forward-bpn net)
    (format t "~S~%" nodes)
    (assert (every #'~= nodes '(2 3 6 9)))
    (backward-bpn net)
    (format t "~S~%" derivatives)
    (assert (every #'~= derivatives (list 18 (- 12 6) 6 1)))
    (flet ((sample ()
             (flt (random 5.0)))
           (clamper (x bpn1)
             (assert (eq bpn1 net))
             (setf (aref nodes 1) x)))
      (setf (clamper net) #'clamper)
      (train (make-instance 'counting-function-sampler
                            :max-n-samples 10000
                            :sampler #'sample)
             (if cg
                 (make-instance 'test-cg-bp-trainer
                                :batch-size 10)
                 (make-instance 'test-bp-trainer
                                :segmenter
                                (repeatedly
                                  (make-instance 'batch-gd-trainer
                                                 :learning-rate (flt 0.01)
                                                 :batch-size 1))))
             net)
      (bpn-error (make-instance 'counting-function-sampler
                                :max-n-samples 1000
                                :sampler #'sample)
                 net))))

(defun test-softmax (&key cg)
  (let* ((net (build-bpn (:class 'test-bpn)
                (input-lump :symbol inputs :size 12)
                (weight-lump :symbol weights :size (* 12 12))
                (hidden-lump :symbol linear :size 12
                             :def (->linear (_)
                                    inputs
                                    (sub weights (* _ 12) (* (1+ _) 12))))
                (hidden-lump :symbol exp :size 12
                             :def (->exp (_) (ref linear _)))
                (normalized-lump :symbol norm :size 12 :group-size 3
                                 :normalized-lump exp)
                (input-lump :symbol desires :size 12)
                (error-node :def (->sum-squared-error (_) norm desires)))))
    (flet ((sample ()
             (loop repeat 4 collect (loop repeat 3 collect (random (flt 5)))))
           (clamper (s bpn1)
             (assert (eq bpn1 net))
             (let* ((desires (make-array 12 :initial-element (flt 0)))
                    (inputs
                     (loop for i below 4
                           append
                           (let* ((s (elt s i))
                                  (max (loop for x in s maximizing x))
                                  (pos (position max s :test #'=)))
                             (setf (aref desires (+ (* i 3) pos)) (flt 1))
                             s))))
               (multiple-value-bind (nodes start end)
                   (lump-node-array (find-lump 'inputs net))
                 (loop for i upfrom start below end
                       for j upfrom 0
                       do (setf (aref nodes i) (elt inputs j))))
               (multiple-value-bind (nodes start end)
                   (lump-node-array (find-lump 'desires net))
                 (loop for i upfrom start below end
                       for j upfrom 0
                       do (setf (aref nodes i) (aref desires j)))))))
      (setf (clamper net) #'clamper)
      (train (make-instance 'counting-function-sampler
                            :max-n-samples 100000
                            :sampler #'sample)
             (if cg
                 (make-instance 'test-cg-bp-trainer
                                :cg-args '(:max-n-line-searches 3)
                                :batch-size 100)
                 (make-instance 'test-bp-trainer
                                :segmenter
                                (repeatedly
                                  ;; Small learning rate and huge decay
                                  ;; because the network is constructed in
                                  ;; such a way that pushes all weights up
                                  ;; towards infinity.
                                  (make-instance 'batch-gd-trainer
                                                 :learning-rate (flt 0.0001)
                                                 :weight-decay (flt 0.1)
                                                 :batch-size 10))))
             net)
      (bpn-error (make-instance 'counting-function-sampler
                                :max-n-samples 1000
                                :sampler #'sample)
                 net))))

(defun init-lump (name bpn deviation)
  (multiple-value-bind (array start end)
      (lump-node-array (find-lump name bpn :errorp t))
    (loop for i upfrom start below end
          do (setf (aref array i) (flt (* deviation (gaussian-random-1)))))))

(defun test-cross-entropy (&key cg)
  (let* ((net (build-bpn (:class 'test-bpn)
                (input-lump :symbol inputs :size 3)
                (weight-lump :symbol weights :size (* 3 3))
                (hidden-lump :symbol linear :size 3
                             :def (->linear (_)
                                    inputs
                                    (sub weights (* _ 3) (* (1+ _) 3))))
                (input-lump :symbol desires :size 3)
                (cross-entropy-softmax-lump :symbol output
                                            :size 3 :group-size 3
                                            :input-lump linear
                                            :target-lump desires)
                ;; Stick cross entropy here just to print its value.
                ;; Otherwise it has no effect.
                (hidden-lump :size 1
                             :def (->cross-entropy (_) desires output)))))
    (flet ((sample ()
             (loop repeat 3 collect (random (flt 5))))
           (clamper (s bpn1)
             (assert (eq bpn1 net))
             (let* ((desires (make-array 3 :initial-element (flt 0)))
                    (inputs (let* ((max (loop for x in s maximizing x))
                                   (pos (position max s :test #'=)))
                              (setf (aref desires pos) (flt 1))
                              s)))
               (multiple-value-bind (nodes start end)
                   (lump-node-array (find-lump 'inputs net))
                 (loop for i upfrom start below end
                       for j upfrom 0
                       do (setf (aref nodes i) (elt inputs j))))
               (multiple-value-bind (nodes start end)
                   (lump-node-array (find-lump 'desires net))
                 (loop for i upfrom start below end
                       for j upfrom 0
                       do (setf (aref nodes i) (aref desires j)))))))
      (setf (clamper net) #'clamper)
      (init-lump 'weights net 0.01)
      (train (make-instance 'counting-function-sampler
                            :max-n-samples 100000
                            :sampler #'sample)
             (if cg
                 (make-instance 'test-cg-bp-trainer
                                :cg-args (list :max-n-line-searches 3)
                                :batch-size 1000)
                 (make-instance 'test-bp-trainer
                                :segmenter
                                (repeatedly
                                  (make-instance 'batch-gd-trainer
                                                 :learning-rate (flt 0.1)
                                                 :batch-size 100))))
             net)
      (bpn-error (make-instance 'counting-function-sampler
                                :max-n-samples 1000
                                :sampler #'sample)
                 net))))

(defun lump-start (lump)
  (nth-value 1 (segment-weights lump)))

(defun lump-end (lump)
  (nth-value 2 (segment-weights lump)))

(defun test-activation (&key transposep cg)
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
                (input-lump :symbol expectations :size 3)
                (error-node :def (->sum-squared-error (_)
                                   expectations
                                   activations))))
         (nodes (nodes net))
         (input-start (lump-start (find-lump 'inputs net)))
         (expectations-start (lump-start (find-lump 'expectations net))))
    ;; Set weights
    (multiple-value-bind (nodes bias-start)
        (lump-node-array (find-lump 'biases net))
      (setf (aref nodes (+ bias-start 0)) (flt 2))
      (setf (aref nodes (+ bias-start 1)) (flt 3)))
    (multiple-value-bind (nodes weight-start weight-end)
        (lump-node-array (find-lump 'weights net))
      (if transposep
          (replace nodes (mapcar #'flt '(0 2 4 1 3 5))
                   :start1 weight-start :end1 weight-end)
          (replace nodes (mapcar #'flt '(0 1 2 3 4 5))
                   :start1 weight-start :end1 weight-end)))
    ;; Clamp input
    (setf (aref nodes input-start) (flt -1))
    (setf (aref nodes (1+ input-start)) (flt -4))
    ;; Check foward pass
    (forward-bpn net)
    (assert (every #'= (nodes net)
                   (if transposep
                       #(-1 -4 2 3 1 -1 0 2 4 1 3 5 -1 -1 -1 0 0 0 3)
                       #(-1 -4 2 3 1 -1 0 1 2 3 4 5 -1 -1 -1 0 0 0 3))))
    ;; Check backward pass
    (backward-bpn net)
    (assert
     (every #'= (derivatives net)
            (if transposep
                #(-12 -18 -12 -18 -12 -18 -2 -2 -2 2 2 2 -2 -2 -2 2 2 2 1) 
                #(-12 -18 -12 -18 -12 -18 -2 2 -2 2 -2 2 -2 -2 -2 2 2 2 1))))
    ;; Train. Error typically goes down to 0.0025 - 0.005
    (init-lump 'biases net 0.01)
    (init-lump 'weights net 0.01)
    (flet ((sample ()
             (list (random (flt 1)) (random (flt 1))))
           (clamper (s bpn1)
             (assert (eq bpn1 net))
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
      (train (make-instance 'counting-function-sampler
                            :max-n-samples (if cg 20000 100000)
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
                                                 :batch-size 10))))
             net)
      (bpn-error (make-instance 'counting-function-sampler
                                :max-n-samples 1000
                                :sampler #'sample)
                 net))))

(defun test-bp ()
  (dolist (use-blas '(nil t))
    (let ((*use-blas* use-blas))
      (assert (> 0.01 (test-simple :cg nil)))
      (assert (> 0.01 (test-simple :cg t)))
      (assert (> 0.6 (test-softmax :cg nil)))
      (assert (> 0.1 (test-cross-entropy :cg nil)))
      (assert (> 0.1 (test-cross-entropy :cg t)))
      (dolist (transposep '(nil t))
        (dolist (cg '(nil t))
          (assert (> 0.01 (test-activation :transposep transposep :cg cg))))))))
