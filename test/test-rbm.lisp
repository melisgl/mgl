(in-package :mgl-test)

(defclass test-trainer ()
  ((counter :initform (make-instance 'rmse-counter) :reader counter)))

(defclass test-cd-trainer (test-trainer rbm-trainer) ())

(defclass test-pcd-trainer (test-trainer rbm-pcd-trainer) ())

(defmethod initialize-trainer ((trainer test-pcd-trainer) rbm)
  (call-next-method)
  (setf (max-n-stripes (mgl-rbm::persistent-chains trainer))
        (batch-size (elt (trainers trainer) 0)))
  (log-msg "Persistent chain n-stripes: ~S~%"
           (n-stripes (mgl-rbm::persistent-chains trainer))))

(defclass test-rbm (rbm)
  ((clamper :initarg :clamper :reader clamper)))

(defmethod set-input (sample (rbm test-rbm))
  (funcall (clamper rbm) sample rbm))

(defun get-squared-error (rbm)
  (set-hidden-mean rbm)
  (set-visible-mean rbm)
  (reconstruction-error rbm))

(defun rbm-rmse (sampler rbm)
  (let ((counter (make-instance 'rmse-counter))
        (n-stripes (max-n-stripes rbm)))
    (while (not (finishedp sampler))
      (set-input (sample-batch sampler n-stripes) rbm)
      (multiple-value-bind (e n) (get-squared-error rbm)
        (add-error counter e n)))
    (get-error counter)))

(defmethod train-batch :around (batch (trainer test-trainer) rbm)
  (call-next-method)
  (let ((counter (counter trainer)))
    (mgl-rbm:inputs->nodes rbm)
    (multiple-value-bind (e n) (get-squared-error rbm)
      (add-error counter e n))
    (let ((n-inputs (n-inputs trainer)))
      (when (/= (floor n-inputs 1000)
                (floor (- n-inputs (length batch)) 1000))
        (log-msg "RMSE: ~,5F (~D, ~D)~%"
                 (or (get-error counter) #.(flt 0))
                 (n-sum-errors counter)
                 n-inputs)
        (reset-counter counter)))))

(defun test-do-chunk ()
  (let ((chunk (make-instance 'sigmoid-chunk :size 5
                              :indices-present
                              (make-array 3 :element-type 'index
                                          :initial-contents '(1 3 4))))
        (r '()))
    (mgl-rbm::do-chunk (i chunk)
      (push i r))
    (assert (equal (reverse r) '(1 3 4)))))

(defun test-rbm/single (&key (visible-type 'sigmoid-chunk)
                        (hidden-type 'sigmoid-chunk)
                        sampler (test-sampler sampler)
                        hidden-bias-p
                        visible-bias-p
                        rank
                        (max-n-samples 10000)
                        (max-n-test-samples 1000)
                        (batch-size 50)
                        (max-n-stripes 10)
                        (trainer-class 'test-cd-trainer)
                        (learning-rate (flt 0.1)))
  (flet ((clamp (samples rbm)
           (let ((chunk (find 'inputs (visible-chunks rbm) :key #'name)))
             (loop for sample in samples
                   for stripe upfrom 0
                   do (with-stripes ((stripe chunk start))
                        (setf (aref (storage (nodes chunk)) (+ start 0))
                              sample))))))
    (let ((rbm (make-instance
                'test-rbm
                :visible-chunks `(,@(when hidden-bias-p
                                      (list
                                       (make-instance 'constant-chunk
                                                      :name 'constant)))
                                  ,(make-instance visible-type
                                                  :name 'inputs :size 1))
                :hidden-chunks `(,@(when visible-bias-p
                                     (list
                                      (make-instance 'constant-chunk
                                                     :name 'constant)))
                                 ,(make-instance hidden-type
                                                 :name 'features :size 1))
                :clouds (if rank
                            `((:class factored-cloud
                               :rank ,rank
                               :visible-chunk inputs
                               :hidden-chunk features))
                            nil)
                :max-n-stripes max-n-stripes
                :clamper #'clamp)))
      (train (make-instance 'counting-function-sampler
                            :max-n-samples max-n-samples
                            :sampler sampler)
             (make-instance trainer-class
                            :segmenter
                            (repeatedly
                              (make-instance 'batch-gd-trainer
                                             :learning-rate (flt learning-rate)
                                             :batch-size batch-size)))
             rbm)
      (rbm-rmse (make-instance 'counting-function-sampler
                               :max-n-samples max-n-test-samples
                               :sampler test-sampler)
                rbm))))

(defun test-rbm/identity-and-xor (&key missingp randomp rank)
  (let ((chunk (make-instance 'sigmoid-chunk :name 'inputs :size 2)))
    (flet ((sample ()
             (select-random-element (list #.(flt 0) #.(flt 1))))
           (clamp (samples rbm)
             (declare (ignore rbm))
             (let ((nodes (storage (nodes chunk))))
               (loop for x in samples
                     for stripe upfrom 0
                     do (with-stripes ((stripe chunk start))
                          (if randomp
                              (if (try-chance 0.5)
                                  (setf (aref nodes (+ start 0))
                                        (if (try-chance 0.7)
                                            x
                                            (- 1 x))
                                        (aref nodes (+ start 1)) (- 1 x))
                                  (setf (aref nodes (+ start 0)) x
                                        (aref nodes (+ start 1))
                                        (if (try-chance 0.7)
                                            (- 1 x)
                                            x)))
                              (setf (aref nodes (+ start 0)) x
                                    (aref nodes (+ start 1)) (- 1 x)))))
               (when missingp
                 (assert (= 1 (length samples)))
                 (let ((vip (make-array 0 :adjustable t :fill-pointer t)))
                   (when (try-chance 0.5)
                     (vector-push-extend 0 vip))
                   (when (try-chance 0.5)
                     (vector-push-extend 1 vip))
                   (setf (indices-present chunk)
                         (make-array (length vip) :element-type 'index
                                     :initial-contents vip)))))))
      (let ((rbm (make-instance
                  'test-rbm
                  :visible-chunks (list
                                   (make-instance 'constant-chunk
                                                  :name 'constant)
                                   chunk)
                  :hidden-chunks (list
                                  (make-instance 'constant-chunk
                                                 :name 'constant)
                                  (make-instance 'sigmoid-chunk :name 'features
                                                 :size 1))
                  :clouds (if rank
                              `((:class factored-cloud
                                 :rank ,rank
                                 :visible-chunk inputs
                                 :hidden-chunk features))
                              nil)
                  :clamper #'clamp)))
        (train (make-instance 'counting-function-sampler
                              :max-n-samples 100000
                              :sampler #'sample)
               (make-instance 'test-cd-trainer
                              :segmenter
                              (lambda (chunk)
                                (make-instance
                                 (if missingp
                                     'per-weight-batch-gd-trainer
                                     'batch-gd-trainer)
                                 :learning-rate
                                 (if (member (name chunk)
                                             '(((inputs features) :a)
                                               ((inputs features) :b))
                                             :test #'equal)
                                     (flt 2)
                                     (flt 0.1))
                                 :batch-size 100)))
               rbm)
        (setq randomp nil)
        (rbm-rmse (make-instance 'counting-function-sampler
                                 :max-n-samples 10000
                                 :sampler #'sample)
                  rbm)))))

(defun test-rbm/identity/softmax (&key (hidden-type 'sigmoid-chunk))
  (flet ((sample ()
           (random 5))
         (clamp (samples rbm)
           (let ((chunk (find 'inputs (visible-chunks rbm) :key #'name)))
             (matlisp:fill-matrix (nodes chunk) (flt 0))
             (loop for sample in samples
                   for stripe upfrom 0
                   do (with-stripes ((stripe chunk start))
                        (setf (aref (storage (nodes chunk)) (+ start sample))
                              #.(flt 1)))))))
    (let ((rbm (make-instance
                'test-rbm
                :visible-chunks (list
                                 (make-instance 'constant-chunk
                                                :name 'constant)
                                 (make-instance 'softmax-chunk :name 'inputs
                                                :size 5 :group-size 5))
                :hidden-chunks (list
                                (make-instance 'constant-chunk :name 'constant)
                                (make-instance hidden-type :name 'features
                                               :size 1))
                :clamper #'clamp)))
      (train (make-instance 'counting-function-sampler
                            :max-n-samples 10000
                            :sampler #'sample)
             (make-instance 'test-cd-trainer
                            :segmenter
                            (repeatedly
                              (make-instance 'per-weight-batch-gd-trainer
                                             :batch-size 10)))
             rbm)
      (rbm-rmse (make-instance 'counting-function-sampler
                               :max-n-samples 10000
                               :sampler #'sample)
                rbm))))

(defun rate-code (value min max n)
  (assert (<= min value max))
  (floor (- value min) (/ (- max min) n)))

(defun test-rbm/sine ()
  (flet ((sample ()
           (random (* 2 pi)))
         (clamp (samples rbm)
           (let* ((chunk (second (visible-chunks rbm)))
                  (nodes (storage (nodes chunk))))
             (loop for x in samples
                   for stripe upfrom 0
                   do (with-stripes ((stripe chunk start))
                        (loop for i below 10
                              do (setf (aref nodes (+ start i)) #.(flt 0)))
                        (setf (aref nodes
                                    (+ start (rate-code x 0 (* 2 pi) 5) 0))
                              #.(flt 1))
                        (setf (aref nodes
                                    (+ start 5 (rate-code (sin x) -1 1 5)))
                              #.(flt 1)))))))
    (let ((rbm (make-instance
                'test-rbm
                :visible-chunks (list
                                 (make-instance 'constant-chunk
                                                :name 'constant)
                                 (make-instance 'softmax-chunk :name 'inputs
                                                :size 10 :group-size 5))
                :hidden-chunks (list
                                (make-instance 'constant-chunk :name 'constant)
                                (make-instance 'gaussian-chunk :name 'features
                                               :size 2))
                :clamper #'clamp)))
      (train (make-instance 'counting-function-sampler
                            :max-n-samples 100000
                            :sampler #'sample)
             (make-instance 'test-cd-trainer
                            :segmenter
                            (repeatedly
                              (make-instance 'per-weight-batch-gd-trainer
                                             :batch-size 10)))
             rbm)
      (rbm-rmse (make-instance 'counting-function-sampler
                               :max-n-samples 10000
                               :sampler #'sample)
                rbm))))

(defun test-factored-cloud ()
  (let* ((n-visible 2)
         (n-hidden 3)
         (rank 1)
         (visible-chunk (make-instance 'sigmoid-chunk
                                       :name 'inputs
                                       :size n-visible))
         (hidden-chunk (make-instance 'sigmoid-chunk
                                      :name 'features
                                      :size n-hidden))
         (rbm (make-instance 'rbm
                             :visible-chunks (list visible-chunk)
                             :hidden-chunks (list hidden-chunk)
                             :default-clouds nil
                             :clouds `((:class factored-cloud
                                        :visible-chunk inputs
                                        :hidden-chunk features
                                        :rank ,rank))))
         (cloud (find-cloud '(inputs features) rbm))
         (a (weights (cloud-a cloud)))
         (b (weights (cloud-b cloud)))
         (v (nodes visible-chunk))
         (h (nodes hidden-chunk))
         (trainer (make-instance 'rbm-trainer
                                 :segmenter
                                 (repeatedly
                                   (make-instance 'batch-gd-trainer)))))
    (assert (= n-hidden (matlisp:nrows (weights (cloud-a cloud)))))
    (assert (= rank (matlisp:ncols (weights (cloud-a cloud)))))
    (assert (= rank (matlisp:nrows (weights (cloud-b cloud)))))
    (assert (= n-visible (matlisp:ncols (weights (cloud-b cloud)))))
    (initialize-trainer trainer rbm)
    (setf (matlisp:matrix-ref a 0 0) (flt 1))
    (setf (matlisp:matrix-ref a 1 0) (flt 3))
    (setf (matlisp:matrix-ref a 2 0) (flt 7))
    (setf (matlisp:matrix-ref b 0 0) (flt -5))
    (setf (matlisp:matrix-ref b 0 1) (flt 11))
    (setf (matlisp:matrix-ref v 0 0) (flt 1/13))
    (setf (matlisp:matrix-ref v 1 0) (flt 1/2))
    (setf (matlisp:matrix-ref h 0 0) (flt 1/4))
    (setf (matlisp:matrix-ref h 1 0) (flt 1/5))
    (setf (matlisp:matrix-ref h 2 0) (flt 1/7))
    (mgl-rbm::accumulate-negative-phase-statistics trainer rbm)
    (assert (= 2 (length (trainers trainer))))
    (let ((db (reshape2 (accumulator (elt (trainers trainer) 0))
                        rank n-visible))
          (da (reshape2 (accumulator (elt (trainers trainer) 1))
                        n-hidden rank)))
      (dotimes (c rank)
        (dotimes (j n-visible)
          (let ((x (matlisp:matrix-ref db c j))
                (y (* (loop for i below n-hidden
                            sum (* (matlisp:matrix-ref a i c)
                                   (matlisp:matrix-ref h i)))
                      (matlisp:matrix-ref v j))))
            (unless (~= x y)
              (error "db_{~A,~A}: ~S /= ~S~%" c j x y)))))
      (dotimes (i n-hidden)
        (dotimes (c rank)
          (let ((x (matlisp:matrix-ref da i c))
                (y (* (loop for j below n-visible
                            sum (* (matlisp:matrix-ref b c j)
                                   (matlisp:matrix-ref v j)))
                      (matlisp:matrix-ref h i))))
            (unless (~= x y)
              (error "da_{~A,~A}: ~S /= ~S~%" i c x y))))))))

(defun test-rbm-examples ()
  ;; Constant one is easily solved with a single large weight.
  (assert (> 0.01 (test-rbm/single :sampler (constantly (flt 1))
                                   :max-n-stripes 7)))
  (assert (> 0.0001 (test-rbm/single :sampler (constantly (flt 1))
                                     :max-n-stripes 7
                                     :rank 1)))
  (assert (> 0.0001 (test-rbm/single :sampler (constantly (flt 1))
                                     :max-n-stripes 7
                                     :rank 3)))
  ;; For constant zero we need to add a bias to either layer.
  (assert (> 0.01
             (test-rbm/single :sampler (constantly (flt 0)) :visible-bias-p t)))
  (assert (> 0.01
             (test-rbm/single :sampler (constantly (flt 0)) :hidden-bias-p t)))
  ;; identity
  (assert (> 0.01
             (test-rbm/single :sampler (repeatedly
                                         (select-random-element
                                          (list #.(flt 0) #.(flt 1))))
                              :visible-bias-p t
                              :hidden-bias-p t
                              :max-n-samples 100000)))
  (assert (> 0.01 (test-rbm/identity-and-xor)))
  (assert (> 0.01 (test-rbm/identity-and-xor :rank 1)))
  (assert (> 0.25 (test-rbm/identity-and-xor :missingp t)))
  #+nil
  (assert (> 0.25 (test-rbm/identity-and-xor :missingp t :rank 1)))
  (assert (> 0.25 (test-rbm/identity-and-xor :randomp t)))
  (assert (> 0.35 (test-rbm/identity-and-xor :missingp t :randomp t)))
  (assert (> 0.2 (test-rbm/single :sampler (constantly (flt 1))
                                  :visible-type 'gaussian-chunk
                                  :max-n-samples 10000)))
  (assert (> 0.4 (test-rbm/identity/softmax)))
  (assert (> 0.2 (test-rbm/identity/softmax :hidden-type 'gaussian-chunk)))
  (assert (> 0.1 (test-rbm/sine))))

(defun compare-objects (x y)
  (let ((class (class-of x))
        (different-slots ()))
    (if (not (eq class (class-of y)))
        (error "~A ~A" class (class-of y))
        (dolist (slot (sb-mop:class-slots class))
          (let ((slot-name (sb-mop:slot-definition-name slot)))
            (if (eql (slot-boundp x slot-name)
                     (slot-boundp y slot-name))
                (when (slot-boundp x slot-name)
                  (unless (eql (slot-value x slot-name)
                               (slot-value y slot-name))
                    (push (list slot-name
                                (slot-value x slot-name)
                                (slot-value y slot-name))
                          different-slots)))
                (push (list slot-name
                            (if (slot-boundp x slot-name)
                                (slot-value x slot-name)
                                :unbound)
                            (if (slot-boundp y slot-name)
                                (slot-value y slot-name)
                                :unbound))
                      different-slots)))))
    (nreverse different-slots)))

(defun test-copy-pcd-chunk ()
  (let ((chunk (make-instance 'sigmoid-chunk
                              :name 'this-chunk
                              :size 10)))
    (assert (equal (mapcar #'first (compare-objects chunk (copy 'pcd chunk)))
                   '(nodes inputs)))))

(defun test-copy-pcd-full-cloud ()
  (let* ((chunk1 (make-instance 'constant-chunk
                                :name 'constant-chunk
                                :size 1))
         (chunk2 (make-instance 'sigmoid-chunk
                                :name 'sigmoid-chunk
                                :size 10))
         (cloud (make-instance 'full-cloud
                               :name 'this-cloud
                               :visible-chunk chunk1
                               :hidden-chunk chunk2)))
    (assert (equal (mapcar #'first (compare-objects cloud (copy 'pcd cloud)))
                   '(visible-chunk hidden-chunk)))))

(defun test-copy-pcd-rbm ()
  (let ((rbm (make-instance
              'rbm
              :visible-chunks (list (make-instance 'constant-chunk
                                                   :name 'constant)
                                    (make-instance 'sigmoid-chunk
                                                   :name 'inputs
                                                   :size 1))
              :hidden-chunks (list (make-instance 'constant-chunk
                                                  :name 'constant)
                                   (make-instance 'sigmoid-chunk
                                                  :name 'features
                                                  :size 1))
              :max-n-stripes 3)))
    (assert (equal (mapcar #'first (compare-objects rbm (copy 'pcd rbm)))
                   '(visible-chunks hidden-chunks clouds max-n-stripes)))
    (dolist (cloud (clouds rbm))
      (assert (member (visible-chunk cloud) (visible-chunks rbm)))
      (assert (member (hidden-chunk cloud) (hidden-chunks rbm))))))

(defun test-copy-pcd ()
  (test-copy-pcd-chunk)
  (test-copy-pcd-full-cloud)
  (test-copy-pcd-rbm))

(defun test-rbm-examples/pcd ()
  ;; Constant one is easily solved with a single large weight.
  (assert (> 0.01 (test-rbm/single :sampler (constantly (flt 1))
                                   :max-n-stripes 7
                                   :trainer-class 'test-pcd-trainer)))
  (assert (> 0.0001 (test-rbm/single :sampler (constantly (flt 1))
                                     :max-n-stripes 7
                                     :rank 1
                                     :trainer-class 'test-pcd-trainer)))
  (assert (> 0.0001 (test-rbm/single :sampler (constantly (flt 1))
                                     :max-n-stripes 7
                                     :rank 3
                                     :trainer-class 'test-pcd-trainer)))
  ;; For constant zero we need to add a bias to either layer.
  (assert (> 0.01
             (test-rbm/single :sampler (constantly (flt 0)) :visible-bias-p t
                              :trainer-class 'test-pcd-trainer)))
  (assert (> 0.01
             (test-rbm/single :sampler (constantly (flt 0)) :hidden-bias-p t
                              :trainer-class 'test-pcd-trainer)))
  ;; identity
  (assert (> 0.01
             (test-rbm/single :sampler (repeatedly
                                         (select-random-element
                                          (list #.(flt 0) #.(flt 1))))
                              :visible-bias-p t
                              :hidden-bias-p t
                              :trainer-class 'test-pcd-trainer
                              :learning-rate (flt 0.01)
                              :max-n-samples 1000000))))

(defun test-rbm ()
  (test-do-chunk)
  (let ((mgl-util:*use-blas* nil))
    (test-factored-cloud)
    (test-rbm-examples))
  (let ((mgl-util:*use-blas* t))
    (test-factored-cloud)
    (test-rbm-examples))
  (test-copy-pcd)
  (test-rbm-examples/pcd))
