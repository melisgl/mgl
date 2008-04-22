(in-package :mgl-test)

(defclass test-trainer (rbm-trainer)
  ((counter :initform (make-instance 'rmse-counter) :reader counter)))

(defclass test-rbm (rbm)
  ((clamper :initarg :clamper :reader clamper)))

(defmethod set-input (sample (rbm test-rbm))
  (funcall (clamper rbm) sample rbm))

(defun rbm-rmse (sampler rbm)
  (let ((counter (make-instance 'rmse-counter)))
    (loop repeat 1000 do
          (assert (not (finishedp sampler)))
          (set-input (sample sampler) rbm)
          (multiple-value-bind (e n) (get-squared-error rbm)
            (add-error counter e n)))
    (get-error counter)))

(defmethod train-one :around (sample (trainer test-trainer) rbm &key)
  (call-next-method)
  (let ((counter (counter trainer)))
    (multiple-value-bind (e n) (get-squared-error rbm)
      (add-error counter e n))
    (let ((n-inputs (n-inputs trainer)))
      (when (zerop (mod n-inputs 1000))
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
                        (max-n-samples 10000)
                        (max-n-test-samples 1000)
                        (batch-size 50))
  (flet ((clamp (sample rbm)
           (let ((chunk (find 'inputs (visible-chunks rbm) :key #'name)))
             (setf (aref (samples chunk) 0) sample))))
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
                                 ,(make-instance hidden-type :size 1))
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

(defun test-rbm/identity-and-xor (&key missingp randomp)
  (let ((chunk (make-instance 'sigmoid-chunk :name 'inputs :size 2)))
    (flet ((sample ()
             (select-random-element (list #.(flt 0) #.(flt 1))))
           (clamp (x rbm)
             (declare (ignore rbm))
             (if randomp
                 (if (try-chance 0.5)
                     (setf (aref (samples chunk) 0) (if (try-chance 0.7)
                                                        x
                                                        (- 1 x))
                           (aref (samples chunk) 1) (- 1 x))
                     (setf (aref (samples chunk) 0) x
                           (aref (samples chunk) 1) (if (try-chance 0.7)
                                                        (- 1 x)
                                                        x)))
                 (setf (aref (samples chunk) 0) x
                       (aref (samples chunk) 1) (- 1 x)))
             (when missingp
               (let ((vip (make-array 0 :adjustable t :fill-pointer t)))
                 (when (try-chance 0.5)
                   (vector-push-extend 0 vip))
                 (when (try-chance 0.5)
                   (vector-push-extend 1 vip))
                 (setf (indices-present chunk)
                       (make-array (length vip) :element-type 'index
                                   :initial-contents vip))))))
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
                  :clamper #'clamp)))
        (train (make-instance 'counting-function-sampler
                              :max-n-samples 100000
                              :sampler #'sample)
               (make-instance 'test-trainer
                              :segmenter
                              (repeatedly
                                (make-instance
                                 (if missingp
                                     'per-weight-batch-gd-trainer
                                     'batch-gd-trainer)
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
         (clamp (sample rbm)
           (let ((chunk (find 'inputs (visible-chunks rbm) :key #'name)))
             (fill (samples chunk) (flt 0))
             (setf (aref (samples chunk) sample) #.(flt 1)))))
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
         (clamp (x rbm)
           (let ((chunk (second (visible-chunks rbm))))
             (loop for i below 10
                   do (setf (aref (samples chunk) i) #.(flt 0)))
             (setf (aref (samples chunk) (rate-code x 0 (* 2 pi) 5))
                   #.(flt 1))
             (setf (aref (samples chunk) (+ 5 (rate-code (sin x) -1 1 5)))
                   #.(flt 1)))))
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

(defun test-rbm ()
  (test-do-chunk)
  ;; Constant one is easily solved with a single large weight.
  (assert (> 0.01 (test-rbm/single :sampler (constantly (flt 1)))))
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
  (assert (> 0.25 (test-rbm/identity-and-xor :missingp t)))
  (assert (> 0.25 (test-rbm/identity-and-xor :randomp t)))
  (assert (> 0.35 (test-rbm/identity-and-xor :missingp t :randomp t)))
  ;; fixme:
  (assert (> 0.1 (test-rbm/single :sampler (constantly (flt 1))
                                  :visible-type 'gaussian-chunk
                                  :max-n-samples 10000)))
  (assert (> 0.4 (test-rbm/identity/softmax)))
  (assert (> 0.1 (test-rbm/identity/softmax :hidden-type 'gaussian-chunk)))
  (assert (> 0.1 (test-rbm/sine))))
