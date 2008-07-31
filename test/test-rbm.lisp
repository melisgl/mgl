(in-package :mgl-test)

(defclass test-trainer (rbm-trainer)
  ((counter :initform (make-instance 'rmse-counter) :reader counter)))

(defclass test-rbm (rbm)
  ((clamper :initarg :clamper :reader clamper)))

(defmethod set-input (sample (rbm test-rbm))
  (funcall (clamper rbm) sample rbm))

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
                        common-rank
                        (max-n-samples 10000)
                        (max-n-test-samples 1000)
                        (batch-size 50)
                        (max-n-stripes 10))
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
                :clouds (if common-rank
                            `((:class factored-cloud
                               :common-rank ,common-rank
                               :visible-chunk inputs
                               :hidden-chunk features))
                            nil)
                :max-n-stripes max-n-stripes
                :clamper #'clamp)))
      (train (make-instance 'counting-function-sampler
                            :max-n-samples max-n-samples
                            :sampler sampler)
             (make-instance 'test-trainer
                            :segmenter
                            (repeatedly
                              (make-instance 'batch-gd-trainer
                                             :batch-size batch-size)))
             rbm)
      (rbm-rmse (make-instance 'counting-function-sampler
                               :max-n-samples max-n-test-samples
                               :sampler test-sampler)
                rbm))))

(defun test-rbm/identity-and-xor (&key missingp randomp common-rank)
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
                  :clouds (if common-rank
                              `((:class factored-cloud
                                 :common-rank ,common-rank
                                 :visible-chunk inputs
                                 :hidden-chunk features))
                              nil)
                  :clamper #'clamp)))
        (train (make-instance 'counting-function-sampler
                              :max-n-samples 100000
                              :sampler #'sample)
               (make-instance 'test-trainer
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
                                 :batch-size 10)))
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
             (make-instance 'test-trainer
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
             (make-instance 'test-trainer
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
         (common-rank 1)
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
                                        :common-rank ,common-rank))))
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
    (assert (= common-rank (matlisp:ncols (weights (cloud-a cloud)))))
    (assert (= common-rank (matlisp:nrows (weights (cloud-b cloud)))))
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
    (print (trainers trainer))
    (assert (= 2 (length (trainers trainer))))
    (let ((db (reshape2 (accumulator1 (elt (trainers trainer) 0))
                        common-rank n-visible))
          (da (reshape2 (accumulator1 (elt (trainers trainer) 1))
                        n-hidden common-rank)))
      (print da)
      (print db)
      (dotimes (c common-rank)
        (dotimes (j n-visible)
          (let ((x (matlisp:matrix-ref db c j))
                (y (* (loop for i below n-hidden
                            sum (* (matlisp:matrix-ref a i c)
                                   (matlisp:matrix-ref h i)))
                      (matlisp:matrix-ref v j))))
            (unless (~= x y)
              (error "db_{~A,~A}: ~S /= ~S~%" c j x y)))))
      (dotimes (i n-hidden)
        (dotimes (c common-rank)
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
                                    :common-rank 1)))
  (assert (> 0.0001 (test-rbm/single :sampler (constantly (flt 1))
                                    :max-n-stripes 7
                                    :common-rank 3)))
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
  (assert (> 0.01 (test-rbm/identity-and-xor :common-rank 1)))
  (assert (> 0.25 (test-rbm/identity-and-xor :missingp t)))
  #+nil
  (assert (> 0.25 (test-rbm/identity-and-xor :missingp t
                                             :common-rank 1)))
  (assert (> 0.25 (test-rbm/identity-and-xor :randomp t)))
  (assert (> 0.35 (test-rbm/identity-and-xor :missingp t :randomp t)))
  (assert (> 0.2 (test-rbm/single :sampler (constantly (flt 1))
                                  :visible-type 'gaussian-chunk
                                  :max-n-samples 10000)))
  (assert (> 0.4 (test-rbm/identity/softmax)))
  (assert (> 0.2 (test-rbm/identity/softmax :hidden-type 'gaussian-chunk)))
  (assert (> 0.1 (test-rbm/sine))))

(defun test-rbm ()
  (test-do-chunk)
  (let ((mgl-util:*use-blas* nil))
    (test-factored-cloud)
    (test-rbm-examples))
  (let ((mgl-util:*use-blas* t))
    (test-factored-cloud)
    (test-rbm-examples)))
