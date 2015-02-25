(in-package :mgl-test)

(defun test-derivatives (lump inputs &key (delta 0.0000001))
  (let ((base-value 10)
        (mgl-bp::*in-training-p* t)
        (incoming-derivatives
          (gaussian-random! (make-mat (mat-dimensions (nodes lump))))))
    (dolist (input inputs)
      (fill! base-value (derivatives input))
      (gaussian-random! (nodes input)))
    (forward lump)
    (copy! incoming-derivatives (mgl-bp::derivatives lump))
    (backward lump)
    (let* ((results (nodes lump))
           (n-results (mat-size results))
           (original-results (copy-mat results)))
      (dolist (input inputs)
        (let* ((nodes (nodes input))
               (n (mat-size nodes))
               (derivatives (mgl-bp::derivatives input)))
          (dotimes (i n)
            (let ((original-value (row-major-mref nodes i)))
              (setf (row-major-mref nodes i)
                    (+ (row-major-mref nodes i) delta))
              (forward lump)
              (assert (plusp (nrm2 nodes)))
              (let ((derivative 0))
                (dotimes (j n-results)
                  (incf derivative
                        (* (row-major-mref incoming-derivatives j)
                           (/ (- (row-major-mref results j)
                                 (row-major-mref original-results j))
                              delta))))
                #+nil
                (format t "~S ~S ~S ~S~%" (name input) i
                        (- (row-major-mref derivatives i) 10) derivative)
                (assert (~= (- (row-major-mref derivatives i) 10) derivative)))
              (setf (row-major-mref nodes i) original-value))))))))

(defun test-lump-derivatives ()
  ;; :FLOAT is too imprecise for this to be a good test
  (dolist (*default-mat-ctype* '(:double))
    (dolist (*cuda-enabled* '(nil t))
      (with-cuda* ()
        (let* ((inputs (list (->input :size 10)))
               (lump (apply #'->sigmoid inputs)))
          (dolist (lump (cons lump inputs))
            (setf (max-n-stripes lump) 7)
            (setf (n-stripes lump) 7))
          (test-derivatives lump inputs))
        ;; ->BATCH-NORMALIZED with default BATCH-SIZE
        (let* ((inputs (list (->input :size 5)
                             (->weight :name 'scale :size 5)
                             (->weight :name 'shift :size 5)))
               (lump (->batch-normalized
                      (first inputs)
                      :normalization (->batch-normalization
                                      (second inputs) (third inputs)
                                      :name 'bhn :batch-size 7))))
          (dolist (lump (cons lump inputs))
            (setf (max-n-stripes lump) 7)
            (setf (n-stripes lump) 7))
          (test-derivatives lump inputs))
        ;; ->BATCH-NORMALIZED with BATCH-SIZE 7 out of 21
        (let* ((inputs (list (->input :size 5)
                             (->weight :name 'scale :size 5)
                             (->weight :name 'shift :size 5)))
               (lump (->batch-normalized
                      (first inputs)
                      :normalization (->batch-normalization
                                      (second inputs) (third inputs)
                                      :name 'bhn :batch-size 7))))
          (dolist (lump (cons lump inputs))
            (setf (max-n-stripes lump) 21)
            (setf (n-stripes lump) 21))
          (test-derivatives lump inputs))))))


(defclass test-bp-optimizer (segmented-gd-optimizer)
  ())

(defclass test-cg-bp-optimizer (cg-optimizer)
  ())

(defclass test-fnn (fnn)
  ((clamper :initarg :clamper :accessor clamper)))

(defmethod set-input (samples (bpn test-fnn))
  (funcall (clamper bpn) samples bpn))

(defun make-bpn-cost-monitors (&key (counter-type 'basic-counter))
  (list
   (make-instance 'monitor
                  :measurer (lambda (instances bpn)
                              (declare (ignore instances))
                              (cost bpn))
                  :counter (make-instance
                            counter-type
                            :attributes '(:dataset "train" :type "cost")))))

(defun make-bp-learner (bpn &key (counter-type 'basic-counter))
  (make-instance 'bp-learner
                 :bpn bpn
                 :monitors (make-bpn-cost-monitors :counter-type counter-type)))

(defun monitor-training-periodically (optimizer)
  (monitor-optimization-periodically
   optimizer
   '((:fn reset-optimization-monitors :period 100 :last-eval 0))))

;; (test-simple :cg nil :max-n-stripes 10)

(defun bpn-error (sampler bpn &key (counter-type 'basic-counter))
  (let ((counter (make-instance counter-type))
        (n-stripes (max-n-stripes bpn)))
    (while (not (finishedp sampler))
      (set-input (list-samples sampler n-stripes) bpn)
      (forward bpn)
      (multiple-value-call #'add-to-counter counter (cost bpn)))
    (values (counter-values counter) counter)))

(defun bpn-nodes (bpn stripe)
  (apply #'concatenate 'list
         (map 'list (lambda (lump)
                      (let ((size (size lump)))
                        (with-facets ((nodes ((nodes lump) 'backing-array
                                              :direction :input)))
                          (subseq nodes
                                  (* stripe size) (* (1+ stripe) size)))))
              (clumps bpn))))

(defun bpn-derivatives (bpn stripe)
  (apply #'concatenate 'list
         (map 'list (lambda (lump)
                      (let ((size (size lump)))
                        (with-facets ((derivatives ((derivatives lump)
                                                    'backing-array
                                                    :direction :input)))
                          (subseq derivatives
                                  (* stripe size) (* (1+ stripe) size)))))
              (clumps bpn))))

(defun test-simple (&key cg (max-n-stripes 1))
  (let* ((net (build-fnn (:class 'test-fnn :max-n-stripes max-n-stripes)
                (weights (->weight :size 1))
                (inputs (->input :size 1))
                (prod (->* inputs weights))
                (sd (->squared-difference inputs prod))
                (my-error (->loss sd))))
         (nodes (map 'list #'nodes (clumps net))))
    (setf (n-stripes net) 1)
    (with-facets ((nodes-0 ((elt nodes 0) 'backing-array :direction :output))
                  (nodes-1 ((elt nodes 1) 'backing-array :direction :output)))
      (setf (aref nodes-0 0) (coerce-to-ctype 2))
      (setf (aref nodes-1 0) (coerce-to-ctype 3)))
    (forward net)
    (assert (every #'~= (bpn-nodes net 0) '(2 3 6 9 9)))
    (backward net)
    (assert (every #'~= (bpn-derivatives net 0) (list 18 6 6 1 1)))
    (flet ((sample ()
             (random (coerce-to-ctype 5.0)))
           (clamper (samples bpn1)
             (assert (eq bpn1 net))
             (let* ((inputs (find-clump 'inputs bpn1))
                    (nodes (nodes inputs)))
               (with-facets ((nodes (nodes 'backing-array :direction :output)))
                 (loop for sample in samples
                       for stripe upfrom 0
                       do (with-stripes ((stripe inputs start))
                            (setf (aref nodes start) sample)))))))
      (setf (clamper net) #'clamper)
      (minimize (if cg
                    (make-instance 'test-cg-bp-optimizer
                                   :batch-size 100
                                   :on-cg-batch-done '(log-cg-batch-done))
                    (monitor-training-periodically
                     (make-instance 'test-bp-optimizer
                                    :segmenter
                                    (repeatedly
                                      (make-instance 'sgd-optimizer
                                                     :learning-rate .01
                                                     :momentum 0.9
                                                     :batch-size 10)))))
                (make-bp-learner net)
                :dataset
                (make-instance 'function-sampler
                               :max-n-samples 1000
                               :generator #'sample))
      (bpn-error (make-instance 'function-sampler
                                :max-n-samples 1000
                                :generator #'sample)
                 net))))

(defun test-softmax (&key cg (max-n-stripes 1))
  (let* ((net (build-fnn (:class 'test-fnn :max-n-stripes max-n-stripes)
                (inputs (->input :size 12))
                (weights (->weight :size (* 12 12)))
                (linear (->v*m inputs weights))
                (exp (->exp linear))
                (norm (->normalized exp :group-size 3))
                (expectations (->input :size 12))
                (sd (->squared-difference norm expectations))
                (my-error (->loss sd)))))
    (let* ((inputs (find-clump 'inputs net))
           (inputs-nodes (nodes inputs))
           (expectations (find-clump 'expectations net))
           (expectations-nodes (nodes expectations)))
      (flet ((sample ()
               (loop repeat 4
                     collect (loop repeat 3
                                   collect (random (coerce-to-ctype 5.0)))))
             (clamper (samples bpn1)
               (assert (eq bpn1 net))
               (with-facets ((inputs-nodes (inputs-nodes 'backing-array
                                                         :direction :output))
                             (expectations-nodes
                              (expectations-nodes 'backing-array
                                                  :direction :output)))
                 (loop
                   for sample in samples
                   for stripe upfrom 0
                   do (with-stripes ((stripe inputs input-start input-end)
                                     (stripe expectations
                                             expectations-start
                                             expectations-end))
                        (loop
                          for s in sample
                          for si upfrom 0 do
                            (let* ((expectations
                                     (make-array 12 :initial-element
                                                 (coerce-to-ctype 0.0)))
                                   (inputs
                                     (loop
                                       for i below 4
                                       append
                                       (let* ((s (elt sample i))
                                              (max (loop for x in s
                                                         maximizing x))
                                              (pos (position max s :test #'=)))
                                         (setf (aref expectations
                                                     (+ (* i 3) pos))
                                               (coerce-to-ctype 1))
                                         s))))
                              (loop for i upfrom input-start below input-end
                                    for j upfrom 0
                                    do (setf (aref inputs-nodes i)
                                             (elt inputs j)))
                              (loop for i upfrom expectations-start
                                      below expectations-end
                                    for j upfrom 0
                                    do (setf (aref expectations-nodes i)
                                             (aref expectations j))))))))))
        (setf (clamper net) #'clamper)
        (minimize (if cg
                      (make-instance 'test-cg-bp-optimizer
                                     :cg-args '(:max-n-line-searches 3)
                                     :batch-size 100
                                     :on-cg-batch-done '(log-cg-batch-done))
                      (monitor-training-periodically
                       (make-instance 'test-bp-optimizer
                                      :segmenter
                                      (repeatedly
                                        ;; Small learning rate and
                                        ;; huge decay because the
                                        ;; network is constructed in
                                        ;; such a way that pushes all
                                        ;; weights up towards
                                        ;; infinity.
                                        (make-instance 'sgd-optimizer
                                                       :learning-rate 0.01
                                                       :momentum 0.9
                                                       :weight-decay 0
                                                       :batch-size 10)))))
                  (make-bp-learner net)
                  :dataset
                  (make-instance 'function-sampler
                                 :max-n-samples 100000
                                 :generator #'sample))
        (bpn-error (make-instance 'function-sampler
                                  :max-n-samples 1000
                                  :generator #'sample)
                   net)))))

(defun init-lump (name bpn deviation)
  (gaussian-random! (segment-weights (find-clump name bpn))
                    :stddev deviation))

(defun test-cross-entropy (&key cg (max-n-stripes 1))
  (let* ((net (build-fnn (:class 'test-fnn :max-n-stripes max-n-stripes)
                (input (->input :size 3))
                (weight (->weight :size (* 3 3)))
                (linear (->v*m input weight))
                (output (->softmax-xe-loss linear)))))
    (let ((input (nodes (find-clump 'input net)))
          (output (find-clump 'output net)))
      (flet ((sample ()
               (loop repeat 3 collect (random 5.0)))
             (clamper (samples bpn1)
               (assert (eq bpn1 net))
               (let ((target (ensure-softmax-target-matrix
                              output (length samples))))
                 (fill! 0 target)
                 (loop for sample in samples
                       for stripe upfrom 0
                       do (let ((pos (max-position sample 0 (length sample))))
                            (setf (mref target stripe pos) 1))
                          (loop for x in sample
                                for j upfrom 0
                                do (setf (mref input stripe j) x))))))
        (setf (clamper net) #'clamper)
        (init-lump 'weight net 0.01)
        (minimize (if cg
                      (make-instance 'test-cg-bp-optimizer
                                     :cg-args (list :max-n-line-searches 3)
                                     :batch-size 1000
                                     :on-cg-batch-done '(log-cg-batch-done))
                      (monitor-training-periodically
                       (make-instance 'test-bp-optimizer
                                      :segmenter
                                      (repeatedly
                                        (make-instance 'sgd-optimizer
                                                       :learning-rate 0.1
                                                       :momentum 0.9
                                                       :batch-size 100)))))
                  (make-bp-learner net)
                  :dataset
                  (make-instance 'function-sampler
                                 :max-n-samples 100000
                                 :generator #'sample))
        (bpn-error (make-instance 'function-sampler
                                  :max-n-samples 1000
                                  :generator #'sample)
                   net)))))

(defun lump-start (lump)
  (nth-value 1 (segment-weights lump)))

(defun lump-end (lump)
  (nth-value 2 (segment-weights lump)))

(defun test-activation (&key transposep cg (max-n-stripes 1))
  (let* ((net (build-fnn (:class 'test-fnn :max-n-stripes max-n-stripes)
                (inputs (->input :size 2))
                (biases (->weight :size 2))
                (biased-input (->+ (list inputs biases) :size 2))
                (weights (->weight :size (* 3 2)))
                (activations (->v*m biased-input weights :size 3
                                   :transpose-weights-p transposep))
                (expectations (->input :size 3))
                (sd (->squared-difference expectations activations))
                (my-error (->loss sd)))))
    ;; Set weights
    (let ((nodes (segment-weights (find-clump 'biases net))))
      (setf (row-major-mref nodes 0) 2)
      (setf (row-major-mref nodes 1) 3))
    (let ((inputs (segment-weights (find-clump 'weights net))))
      (with-shape-and-displacement (inputs (mat-size inputs))
        (replace! inputs (if transposep
                             (mapcar #'flt '(0 1 2 3 4 5))
                             (mapcar #'flt '(0 2 4 1 3 5))))))
    ;; Clamp input
    (let ((nodes (segment-weights (find-clump 'inputs net))))
      (setf (row-major-mref nodes 0) -1)
      (setf (row-major-mref nodes 1) -4))
    ;; Check foward pass
    (forward net)
    (assert (every #'= (bpn-nodes net 0)
                   (if transposep
                       #(-1 -4 2 3 1 -1 0 1 2 3 4 5 -1 -1 -1 0 0 0 1 1 1 3)
                       #(-1 -4 2 3 1 -1 0 2 4 1 3 5 -1 -1 -1 0 0 0 1 1 1 3))))
    ;; Check backward pass
    (backward net)
    (assert
     (every #'= (bpn-derivatives net 0)
            (if transposep
                #(-12 -18 -12 -18 -12 -18 -2 2 -2 2 -2 2 -2 -2 -2 2 2 2 1 1)
                #(-12 -18 -12 -18 -12 -18 -2 -2 -2 2 2 2 -2 -2 -2 2 2 2 1 1))))
    ;; Train. Error typically goes down to 0.0025 - 0.005
    (init-lump 'biases net 0.01)
    (init-lump 'weights net 0.01)
    (let* ((inputs (find-clump 'inputs net))
           (input-nodes (nodes inputs))
           (expectations (find-clump 'expectations net))
           (expectations-nodes (nodes expectations)))
      (flet ((sample ()
               (list (random (coerce-to-ctype 1.0))
                     (random (coerce-to-ctype 1.0))))
             (clamper (samples bpn1)
               (assert (eq bpn1 net))
               (with-facets ((input-nodes
                              (input-nodes 'backing-array :direction :output))
                             (expectations-nodes
                              (expectations-nodes 'backing-array
                                                  :direction :output)))
                 (loop for sample in samples
                       for stripe upfrom 0
                       do (with-stripes ((stripe inputs input-start)
                                         (stripe expectations
                                                 expectations-start))
                            (destructuring-bind (x y) sample
                              (setf (aref input-nodes input-start) x)
                              (setf (aref input-nodes (1+ input-start)) y)
                              (setf (aref expectations-nodes expectations-start)
                                    (+ (* 3 (+ x 2))
                                       (* 2 (+ y -1))))
                              (setf (aref expectations-nodes
                                          (1+ expectations-start))
                                    (+ (* -3 (+ x 2))
                                       (* -2 (+ y -1))))
                              (setf (aref expectations-nodes
                                          (+ 2 expectations-start))
                                    (+ (* -0.8 (+ x 2))
                                       (* 1.3 (+ y -1))))))))))
        (setf (clamper net) #'clamper)
        (minimize (if cg
                      (make-instance 'test-cg-bp-optimizer
                                     :cg-args '(:max-n-line-searches 3)
                                     :batch-size 1000
                                     :on-cg-batch-done '(log-cg-batch-done))
                      (monitor-training-periodically
                       (make-instance 'test-bp-optimizer
                                      :segmenter
                                      (repeatedly
                                        (make-instance 'sgd-optimizer
                                                       :learning-rate 0.01
                                                       :momentum 0.9
                                                       :batch-size 10)))))
                  (make-bp-learner net)
                  :dataset
                  (make-instance 'function-sampler
                                 :max-n-samples (if cg 30000 100000)
                                 :generator #'sample))
        (bpn-error (make-instance 'function-sampler
                                  :max-n-samples 1000
                                  :generator #'sample)
                   net)))))

(defun test-bp ()
  (declare (optimize (debug 3)))
  (test-lump-derivatives)
  (do-cuda ()
    (dolist (*default-mat-ctype* '(:float :double))
      (dolist (max-n-stripes '(10 100))
        (dolist (cg '(nil t))
          (log-msg "cuda: ~S, ~S, max-n-stripes: ~S, cg: ~S~%"
                   (cuda-available-p) *default-mat-ctype* max-n-stripes cg)
          ;; CG can easily run into numerical problems on these.
          (unless cg
            (assert (> 0.01 (test-simple :cg cg :max-n-stripes max-n-stripes)))
            (assert (> 0.3 (test-softmax :cg cg :max-n-stripes max-n-stripes))))
          (dolist (transposep '(nil t))
            (assert (> 0.01 (test-activation :transposep transposep :cg cg
                                             :max-n-stripes max-n-stripes))))
          (assert (> 0.1 (test-cross-entropy
                          :cg cg :max-n-stripes max-n-stripes))))))))
