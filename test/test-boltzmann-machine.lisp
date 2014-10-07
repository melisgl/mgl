(in-package :mgl-test)

(defclass test-optimizer (segmented-gd-optimizer)
  ((counter :initform (make-instance 'rmse-counter) :reader counter)))

(defclass test-cd-learner (rbm-cd-learner) ())

(defclass test-pcd-learner (bm-pcd-learner) ())

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
      (set-input (list-samples sampler n-stripes) rbm)
      (multiple-value-bind (e n) (get-squared-error rbm)
        (add-error counter e n)))
    (get-error counter)))

(defmethod mgl-opt:accumulate-gradients* :around
    (learner (optimizer test-optimizer) batch multiplier valuep)
  (call-next-method)
  (let ((counter (counter optimizer))
        (rbm (bm learner)))
    (inputs->nodes rbm)
    (multiple-value-bind (e n) (get-squared-error rbm)
      (add-error counter e n))
    (let ((n-instances (n-instances optimizer)))
      (when (/= (floor n-instances 1000)
                (floor (- n-instances (length batch)) 1000))
        (log-msg "RMSE: ~,5F (~D, ~D)~%"
                 (or (get-error counter) #.(flt 0))
                 (n-sum-errors counter)
                 n-instances)
        (reset-counter counter)))))

(defun test-do-chunk ()
  (let ((chunk (make-instance 'sigmoid-chunk :size 5
                              :indices-present
                              (make-array 3 :element-type 'index
                                          :initial-contents '(1 3 4))))
        (r '()))
    (mgl-bm::do-chunk (i chunk)
      (push i r))
    (assert (equal (reverse r) '(1 3 4)))))

(defun test-rbm/single (&key (visible-type 'sigmoid-chunk)
                        (hidden-type 'sigmoid-chunk)
                        generator (test-generator generator)
                        hidden-bias-p
                        visible-bias-p
                        rank
                        (max-n-samples 10000)
                        (max-n-test-samples 1000)
                        (batch-size 50)
                        (max-n-stripes 10)
                        (learner-class 'test-cd-learner)
                        (learning-rate (flt 0.1))
                        (features-size 1)
                        normal-sparsity
                        cheating-sparsity)
  (flet ((clamp (samples rbm)
           (let ((chunk (find 'inputs (visible-chunks rbm) :key #'name)))
             (with-facets ((nodes ((nodes chunk) 'backing-array
                                   :direction :output
                                   :type flt-vector)))
               (loop for sample in samples
                     for stripe upfrom 0
                     do (with-stripes ((stripe chunk start))
                          (setf (aref nodes (+ start 0)) sample)))))))
    (let ((rbm (make-instance
                'test-rbm
                :visible-chunks `(,@(when hidden-bias-p
                                      (list
                                       (make-instance 'constant-chunk
                                                      :name 'constant1)))
                                  ,(make-instance visible-type
                                                  :name 'inputs :size 1))
                :hidden-chunks `(,@(when visible-bias-p
                                     (list
                                      (make-instance 'constant-chunk
                                                     :name 'constant2)))
                                 ,(make-instance hidden-type
                                                 :name 'features
                                                 :size features-size))
                :clouds (if rank
                            `(:merge
                              (:class factored-cloud
                                      :rank ,rank
                                      :chunk1 inputs
                                      :chunk2 features))
                            '(:merge))
                :max-n-stripes max-n-stripes
                :clamper #'clamp)))
      (minimize (make-instance
                 'test-optimizer
                 :segmenter
                 (repeatedly
                   (make-instance 'batch-gd-optimizer
                                  :learning-rate (flt learning-rate)
                                  :momentum (flt 0.9)
                                  :batch-size batch-size)))
                (if (subtypep learner-class 'bm-pcd-learner)
                    (make-instance learner-class
                                   :bm rbm
                                   :n-particles batch-size)
                    (make-instance learner-class
                                   :bm rbm
                                   :sparser
                                   (lambda (cloud chunk)
                                     (when (and (eq (name chunk) 'features)
                                                (or normal-sparsity
                                                    cheating-sparsity))
                                       (make-instance
                                        (if normal-sparsity
                                            'normal-sparsity-gradient-source
                                            'cheating-sparsity-gradient-source)
                                        :cloud cloud
                                        :chunk chunk
                                        :sparsity (flt (or normal-sparsity
                                                           cheating-sparsity))
                                        :cost (flt 0.1)
                                        :damping (flt 0.9))))))
                :dataset (make-instance 'function-sampler
                                        :max-n-samples max-n-samples
                                        :generator generator))
      (rbm-rmse (make-instance 'function-sampler
                               :max-n-samples max-n-test-samples
                               :generator test-generator)
                rbm))))

(defun test-rbm/identity-and-xor (&key missingp randomp rank)
  (let ((chunk (make-instance 'sigmoid-chunk :name 'inputs :size 2)))
    (flet ((sample ()
             (select-random-element (list #.(flt 0) #.(flt 1))))
           (clamp (samples rbm)
             (declare (ignore rbm))
             (with-facets ((nodes ((nodes chunk) 'backing-array
                                   :direction :output :type flt-vector)))
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
                                                  :name 'constant1)
                                   chunk)
                  :hidden-chunks (list
                                  (make-instance 'constant-chunk
                                                 :name 'constant2)
                                  (make-instance 'sigmoid-chunk :name 'features
                                                 :size 1))
                  :clouds (if rank
                              `(:merge
                                (:class factored-cloud
                                        :rank ,rank
                                        :chunk1 inputs
                                        :chunk2 features))
                              '(:merge))
                  :clamper #'clamp)))
        (minimize (make-instance 'test-optimizer
                                 :segmenter
                                 (lambda (chunk)
                                   (make-instance
                                    (if missingp
                                        'per-weight-batch-gd-optimizer
                                        'batch-gd-optimizer)
                                    :learning-rate
                                    (if (member (name chunk)
                                                '(((inputs features) :a)
                                                  ((inputs features) :b))
                                                :test #'equal)
                                        (flt 2)
                                        (flt 0.1))
                                    :momentum (flt 0.9)
                                    :batch-size 100)))
                  (make-instance 'test-cd-learner :rbm rbm)
                  :dataset
                  (make-instance 'function-sampler
                                 :max-n-samples 100000
                                 :generator #'sample))
        (setq randomp nil)
        (rbm-rmse (make-instance 'function-sampler
                                 :max-n-samples 10000
                                 :generator #'sample)
                  rbm)))))

(defun test-rbm/identity/softmax (&key (hidden-type 'sigmoid-chunk))
  (flet ((sample ()
           (random 5))
         (clamp (samples rbm)
           (let ((chunk (find 'inputs (visible-chunks rbm) :key #'name)))
             (fill! (flt 0) (nodes chunk))
             (with-facets ((nodes ((nodes chunk) 'backing-array
                                   :direction :io :type flt-vector)))
               (loop for sample in samples
                     for stripe upfrom 0
                     do (with-stripes ((stripe chunk start))
                          (setf (aref nodes (+ start sample))
                                #.(flt 1))))))))
    (let ((rbm (make-instance
                'test-rbm
                :visible-chunks (list
                                 (make-instance 'constant-chunk
                                                :name 'constant1)
                                 (make-instance 'softmax-chunk :name 'inputs
                                                :size 5 :group-size 5))
                :hidden-chunks (list
                                (make-instance 'constant-chunk :name 'constant2)
                                (make-instance hidden-type :name 'features
                                               :size 1))
                :clamper #'clamp)))
      (minimize (make-instance 'test-optimizer
                               :segmenter
                               (repeatedly
                                 (make-instance 'per-weight-batch-gd-optimizer
                                                :momentum (flt 0.9)
                                                :batch-size 10)))
                (make-instance 'test-cd-learner :rbm rbm)
                :dataset
                (make-instance 'function-sampler
                               :max-n-samples 30000
                               :generator #'sample))
      (rbm-rmse (make-instance 'function-sampler
                               :max-n-samples 10000
                               :generator #'sample)
                rbm))))

(defun rate-code (value min max n)
  (assert (<= min value max))
  (floor (- value min) (/ (- max min) n)))

(defun test-rbm/sine ()
  (flet ((sample ()
           (random (* 2 pi)))
         (clamp (samples rbm)
           (let* ((chunk (second (visible-chunks rbm))))
             (with-facets ((nodes* ((nodes chunk) 'backing-array
                                    :direction :output :type flt-vector)))
               (loop for x in samples
                     for stripe upfrom 0
                     do (with-stripes ((stripe chunk start))
                          (loop for i below 10
                                do (setf (aref nodes* (+ start i)) #.(flt 0)))
                          (setf (aref nodes*
                                      (+ start (rate-code x 0 (* 2 pi) 5) 0))
                                #.(flt 1))
                          (setf (aref nodes*
                                      (+ start 5 (rate-code (sin x) -1 1 5)))
                                #.(flt 1))))))))
    (let ((rbm (make-instance
                'test-rbm
                :visible-chunks (list
                                 (make-instance 'constant-chunk
                                                :name 'constant1)
                                 (make-instance 'softmax-chunk :name 'inputs
                                                :size 10 :group-size 5))
                :hidden-chunks (list
                                (make-instance 'constant-chunk :name 'constant2)
                                (make-instance 'gaussian-chunk :name 'features
                                               :size 2))
                :clamper #'clamp)))
      (minimize (make-instance 'test-optimizer
                               :segmenter
                               (repeatedly
                                 (make-instance 'per-weight-batch-gd-optimizer
                                                :momentum (flt 0.9)
                                                :batch-size 10)))
                (make-instance 'test-cd-learner :rbm rbm)
                :dataset
                (make-instance 'function-sampler
                               :max-n-samples 100000
                               :generator #'sample))
      (rbm-rmse (make-instance 'function-sampler
                               :max-n-samples 10000
                               :generator #'sample)
                rbm))))

(defun test-rbm-examples ()
  ;; Constant one is easily solved with a single large weight.
  ;; qwe
  (assert (> 0.01 (test-rbm/single :generator (constantly (flt 1))
                                   :max-n-stripes 1)))
  (assert (> 0.0001 (test-rbm/single :generator (constantly (flt 1))
                                     :max-n-stripes 7
                                     :rank 1)))
  (assert (> 0.0001 (test-rbm/single :generator (constantly (flt 1))
                                     :max-n-stripes 7
                                     :rank 3)))
  ;; For constant zero we need to add a bias to either layer.
  (assert (> 0.01
             (test-rbm/single :generator (constantly (flt 0))
                              :visible-bias-p t)))
  (assert (> 0.01
             (test-rbm/single :generator (constantly (flt 0))
                              :hidden-bias-p t)))
  ;; identity
  (assert (> 0.01
             (test-rbm/single :generator (repeatedly
                                         (select-random-element
                                          (list #.(flt 0) #.(flt 1))))
                              :visible-bias-p t
                              :hidden-bias-p t
                              :max-n-samples 100000)))
  (assert (> 0.01 (test-rbm/identity-and-xor)))
  (assert (> 0.05 (test-rbm/identity-and-xor :rank 1)))
  (assert (> 0.25 (test-rbm/identity-and-xor :missingp t)))
  #+nil
  (assert (> 0.25 (test-rbm/identity-and-xor :missingp t :rank 1)))
  (assert (> 0.25 (test-rbm/identity-and-xor :randomp t)))
  (assert (> 0.4 (test-rbm/identity-and-xor :missingp t :randomp t)))
  (assert (> 0.2 (test-rbm/single :generator (constantly (flt 1))
                                  :visible-type 'gaussian-chunk
                                  :max-n-samples 10000)))
  (assert (> 0.4 (test-rbm/identity/softmax)))
  (assert (> 0.2 (test-rbm/identity/softmax :hidden-type 'gaussian-chunk)))
  (assert (> 0.2 (test-rbm/sine))))

(defun compare-objects (x y)
  (let ((class (class-of x))
        (different-slots ()))
    (if (not (eq class (class-of y)))
        (error "~A ~A" class (class-of y))
        (dolist (slot (c2mop:class-slots class))
          (let ((slot-name (c2mop:slot-definition-name slot)))
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

(defun set-= (list1 list2 &key key (test #'eql) (test-not nil test-not-p))
  (if test-not-p
      (and (null (set-difference list1 list2
                                 :key key :test-not test-not))
           (null (set-difference list2 list1
                                 :key key :test-not test-not)))
      (and (null (set-difference list1 list2
                                 :key key :test test))
           (null (set-difference list2 list1
                                 :key key :test test)))))

(defun names= (list1 list2)
  (set-= list1 list2 :test #'equal))

(defun test-copy-chunk (context)
  (let ((chunk (make-instance 'sigmoid-chunk
                              :name 'this-chunk
                              :size 10)))
    (assert (names= (mapcar #'first
                            (compare-objects chunk (copy context chunk)))
                    '(nodes mgl-bm::means mgl-bm::old-nodes inputs)))))

(defun test-copy-full-cloud (context)
  (let* ((chunk1 (make-instance 'constant-chunk
                                :name 'constant-chunk
                                :size 1))
         (chunk2 (make-instance 'sigmoid-chunk
                                :name 'sigmoid-chunk
                                :size 10))
         (cloud (make-instance 'full-cloud
                               :name 'this-cloud
                               :chunk1 chunk1
                               :chunk2 chunk2)))
    ;; These start with NIL and are lazily set up.
    (setf (slot-value cloud 'mgl-bm::cached-activations1) t)
    (setf (slot-value cloud 'mgl-bm::cached-activations2) t)
    (assert (names= (mapcar #'first
                            (compare-objects cloud (copy context cloud)))
                    '(chunk1 chunk2
                      mgl-bm::cached-version1
                      mgl-bm::cached-version2
                      mgl-bm::cached-activations1
                      mgl-bm::cached-activations2)))))

(defun test-copy-pcd-rbm ()
  (let ((rbm (make-instance
              'rbm
              :visible-chunks (list (make-instance 'constant-chunk
                                                   :name 'constant1)
                                    (make-instance 'sigmoid-chunk
                                                   :name 'inputs
                                                   :size 1))
              :hidden-chunks (list (make-instance 'constant-chunk
                                                  :name 'constant2)
                                   (make-instance 'sigmoid-chunk
                                                  :name 'features
                                                  :size 1))
              :max-n-stripes 3)))
    (assert (names= (mapcar #'first (compare-objects rbm (copy 'pcd rbm)))
                    '(chunks visible-chunks hidden-chunks
                      mgl-bm::visible-and-conditioning-chunks
                      mgl-bm::hidden-and-conditioning-chunks
                      mgl-bm::conditioning-chunks
                      clouds max-n-stripes)))
    (dolist (cloud (clouds rbm))
      (assert (member (chunk1 cloud) (visible-chunks rbm)))
      (assert (member (chunk2 cloud) (hidden-chunks rbm))))))

(defun test-copy-pcd ()
  (test-copy-chunk 'pcd)
  (test-copy-full-cloud 'pcd)
  (test-copy-pcd-rbm))

(defun test-rbm-examples/pcd ()
  ;; Constant one is easily solved with a single large weight.
  (assert (> 0.01 (test-rbm/single :generator (constantly (flt 1))
                                   :max-n-stripes 7
                                   :learner-class 'test-pcd-learner)))
  (assert (> 0.0001 (test-rbm/single :generator (constantly (flt 1))
                                     :max-n-stripes 7
                                     :rank 1
                                     :learner-class 'test-pcd-learner)))
  (assert (> 0.0001 (test-rbm/single :generator (constantly (flt 1))
                                     :max-n-stripes 7
                                     :rank 3
                                     :learner-class 'test-pcd-learner)))
  ;; For constant zero we need to add a bias to either layer.
  (assert (> 0.01
             (test-rbm/single :generator (constantly (flt 0)) :visible-bias-p t
                              :learner-class 'test-pcd-learner)))
  (assert (> 0.01
             (test-rbm/single :generator (constantly (flt 0)) :hidden-bias-p t
                              :learner-class 'test-pcd-learner)))
  ;; identity
  (assert (> 0.1
             (test-rbm/single :generator (repeatedly
                                         (select-random-element
                                          (list #.(flt 0) #.(flt 1))))
                              :visible-bias-p t
                              :hidden-bias-p t
                              :learner-class 'test-pcd-learner
                              :learning-rate (flt 0.01)
                              :max-n-samples 1000000))))

;;; Sparsity
#+nil
(test-rbm/single :generator (constantly (flt 1))
                 :max-n-stripes 1
                 :max-n-samples 100000
                 :features-size 20
                 :normal-sparsity (flt 0.7))

(defun test-rbm ()
  (test-do-chunk)
  (test-rbm-examples)
  (test-copy-pcd)
  (test-rbm-examples/pcd))

(defun test-make-dbm ()
  (let ((dbm (make-instance 'dbm
                            :layers (list (list (make-instance 'sigmoid-chunk
                                                               :name 'inputs
                                                               :size 2))
                                          (list (make-instance 'sigmoid-chunk
                                                               :name 'features1
                                                               :size 1))))))
    (assert (names= (mapcar #'name (visible-chunks dbm))
                    '(inputs)))
    (assert (names= (mapcar #'name (hidden-chunks dbm))
                    '(features1)))
    (assert (names= (mapcar #'name (clouds dbm))
                    '((inputs features1)))))
  (let ((dbm (make-instance 'dbm
                            :layers (list
                                     (list (make-instance 'sigmoid-chunk
                                                          :name 'inputs
                                                          :size 2)
                                           (make-instance 'constant-chunk
                                                          :name 'constant0))
                                     (list (make-instance 'sigmoid-chunk
                                                          :name 'features1
                                                          :size 1)
                                           (make-instance 'constant-chunk
                                                          :name 'constant1))
                                     (list (make-instance 'sigmoid-chunk
                                                          :name 'features2
                                                          :size 1)
                                           (make-instance 'constant-chunk
                                                          :name 'constant2))))))
    (assert (names= (mapcar #'name (visible-chunks dbm))
                    '(inputs constant0)))
    (assert (names= (mapcar #'name (hidden-chunks dbm))
                    '(features1 constant1 features2 constant2)))
    (assert (names= (mapcar #'name (clouds dbm))
                    '((inputs features1) (inputs constant1)
                      (constant0 features1)
                      (features1 features2) (features1 constant2)
                      (constant1 features2))))
    (let ((dbn (dbm->dbn dbm)))
      (destructuring-bind (rbm0 rbm1) (rbms dbn)
        (assert (names= (mapcar #'name (visible-chunks rbm0))
                        '(inputs constant0)))
        (assert (names= (mapcar #'name (hidden-chunks rbm0))
                        '(features1 constant1)))
        (assert (names= (mapcar #'name (clouds rbm0))
                        '((inputs features1) (inputs constant1)
                          (constant0 features1))))
        (assert (every (lambda (cloud)
                         (and (= (flt 1) (mgl-bm::scale1 cloud))
                              (= (flt 2) (mgl-bm::scale2 cloud))))
                       (clouds rbm0)))
        (assert (equal (hidden-chunks rbm0)
                       (visible-chunks rbm1)))
        (assert (names= (mapcar #'name (hidden-chunks rbm1))
                        '(features2 constant2)))
        (assert (names= (mapcar #'name (clouds rbm1))
                        '((features1 features2) (features1 constant2)
                          (constant1 features2))))
        (assert (every (lambda (cloud)
                         (and (= (flt 2) (mgl-bm::scale1 cloud))
                              (= (flt 1) (mgl-bm::scale2 cloud))))
                       (clouds rbm1)))))))

(defun test-dbm-up/down ()
  (let* ((dbm-inputs (make-instance 'sigmoid-chunk
                                    :name 'inputs
                                    :size 2))
         (dbm-constant1 (make-instance 'constant-chunk
                                       :name 'constant1
                                       :size 1))
         (dbm-features1 (make-instance 'sigmoid-chunk
                                       :name 'features1
                                       :size 1))
         (dbm-features2 (make-instance 'sigmoid-chunk
                                       :name 'features2
                                       :size 2))
         (dbm (make-instance 'dbm
                             :layers (list (list dbm-inputs)
                                           (list dbm-constant1
                                                 dbm-features1)
                                           (list dbm-features2))))
         (dbn (dbm->dbn dbm))
         (dbn-inputs (find-chunk 'inputs dbn))
         (dbn-features1 (find-chunk 'features1 dbn))
         (dbn-features2 (find-chunk 'features2 dbn)))
    (labels ((~=v (x y)
               (when (eq x y)
                 (error "DBM chunks ~A and DBN chunk ~A are the same." x y))
               (with-facets ((x* ((nodes x) 'backing-array :direction :input
                                  :type flt-vector))
                             (y* ((nodes y) 'backing-array :direction :input
                                  :type flt-vector)))
                 (unless (every #'~= x* y*)
                   (error "~A and ~A are different." x y))))
             (check ()
               (map nil #'~=v
                    (list dbm-inputs dbm-features1 dbm-features2)
                    (list dbn-inputs dbn-features1 dbn-features2))))
      (replace! (weights (find-cloud '(inputs constant1) dbm))
                '(0.35d0 1.09d0))
      (replace! (weights (find-cloud '(inputs features1) dbm))
                '(0.3d0 -0.8d0))
      (replace! (weights (find-cloud '(constant1 features2) dbm))
                '(-0.63d0 -1.75d0))
      (replace! (weights (find-cloud '(features1 features2) dbm))
                '(-2.3d0 -0.1d0))
      (replace! (nodes dbm-inputs)
                '((0.7d0) (0.2d0)))
      (replace! (nodes dbn-inputs)
                '((0.7d0) (0.2d0)))
      (up-dbm dbm)
      (map nil #'set-hidden-mean (rbms dbn))
      (check)
      (down-dbm dbm)
      (down-mean-field dbn)
      (check))))

(defun test-copy-dbm->dbn ()
  (test-copy-chunk 'dbm->dbn)
  (test-copy-full-cloud 'dbm->dbn))

(defun test-dbm ()
  (test-make-dbm)
  (test-dbm-up/down)
  (test-copy-dbm->dbn))

(defun test-bm ()
  (do-cuda ()
    (test-rbm)
    (test-dbm)))
