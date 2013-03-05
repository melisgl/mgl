(in-package :mgl-example-gp)

(defclass test-bpn-gp (trivial-cached-executor-mixin bpn-gp)
  ())

(defmethod find-one-executor (sample (bpn-gp test-bpn-gp))
  (let ((key (sample-to-executor-cache-key sample bpn-gp)))
    (or (lookup-executor-cache key bpn-gp)
        (insert-into-executor-cache
         key bpn-gp
         (destructuring-bind (n-x1 n-x2) key
           (build-simple-bpn-gp n-x1 n-x2 :weights-from bpn-gp))))))

(defmethod set-input (samples (bpn test-bpn-gp))
  (let* ((df (find-lump 'distance-field bpn :errorp t))
         (df* (nodes df))
         (selfp (find-lump 'selfp bpn :errorp t))
         (selfp* (nodes selfp)))
    (loop for sample in samples
          for stripe upfrom 0
          do (destructuring-bind (&key ((:x1 x1-vec)) ((:x2 x2-vec))
                                  ((:y1 y1-vec)) &allow-other-keys)
                 sample
               (declare (ignore y1-vec))
               (let ((n1 (length x1-vec))
                     (n2 (length x2-vec)))
                 (with-stripes ((stripe df dfs dfe)
                                (stripe selfp selfps selfpe))
                   (declare (ignore dfe selfpe))
                   (let ((dfi dfs)
                         (selfpi selfps))
                     (dotimes (i n1)
                       (let ((a (aref x1-vec i)))
                         (dotimes (j n2)
                           (let((b (aref x2-vec j)))
                             (setf (aref df* dfi) (flt (abs (- a b))))
                             (setf (aref selfp* selfpi)
                                   (if (eq a b)
                                       (flt 0)
                                       (flt -1)))
                             (incf dfi)
                             (incf selfpi))))))))))))

(defclass test-bpn-gp-base-trainer (base-trainer)
  ())

(defmethod log-training-period ((trainer test-bpn-gp-base-trainer) learner)
  100)

(defmethod log-test-period ((trainer test-bpn-gp-base-trainer) learner)
  100)

(defun make-test-bpn-gp-counters-and-measurers ()
  (cons (cons (make-instance 'error-counter :name "average neg log likelihood")
              (lambda (samples bpn)
                (declare (ignore samples))
                (cost bpn)))
        (loop for name in '(means-bias signal-variance length-scale
                            roughness noise-variance bias-variance)
              collect (cons (make-instance 'error-counter
                                           :name (string-downcase
                                                  (symbol-name name)))
                            (let ((name name))
                              (lambda (samples bpn)
                                (declare (ignore samples))
                                (values (to-scalar (nodes (find-lump name bpn)))
                                        1)))))))

(defmethod initialize-trainer ((trainer test-bpn-gp-base-trainer) (bpn bpn-gp))
  (call-next-method)
  (setf (slot-value trainer 'training-counters-and-measurers)
        (prepend-name-to-counters
         "bpn gp: training"
         (make-test-bpn-gp-counters-and-measurers))))

(defmethod log-test-error ((trainer test-bpn-gp-base-trainer) (bpn-gp bpn-gp))
  (call-next-method)
  (let ((counter (make-instance 'rmse-counter :name "test rmse")))
    (loop repeat 100 do
      (multiple-value-bind (inputs outputs) (make-input-output 5)
        (let ((gp (update-gp bpn-gp inputs outputs)))
          (loop repeat 10
                do (let* ((x (random-in-test-domain))
                          (y (test-target x))
                          (y* (to-scalar (gp-means gp (vector x)))))
                     (add-error counter (expt (- y y*) 2) 1))))))
    (log-msg "bpn gp: ~A~%" counter)))

(defclass test-bpn-gp-gd-trainer (test-bpn-gp-base-trainer bp-trainer)
  ())

(defun fill-lump (name bpn value)
  (multiple-value-bind (array start end)
      (segment-weights (find-lump name bpn :errorp t))
    (loop for i upfrom start below end
          do (setf (aref array i) (flt value)))))

(defun build-simple-bpn-gp (n-x1 n-x2 &key weights-from)
  (with-weights-copied (weights-from)
    (let* ((n-cov (* n-x1 n-x2))
           (bpn-gp
             (build-bpn (:class 'test-bpn-gp
                         :initargs (list
                                    :mean-lump-name 'means
                                    :covariance-lump-name 'covariances))
               (x1 (->input :size n-x1))
               (x2 (->input :size n-x2))
               ;; Non-zero means the corresponding element in x1 and
               ;; x2 are identical.
               (selfp (->input :size n-cov))
               ;; The matrix of distances between x1 an x2.
               (distance-field (->input :size n-cov))
               ;;
               (means-bias (->weight :size 1))
               (means-bias* (->rep :x means-bias :n n-x1))
               (means (->+ :args (list means-bias*)))
               ;; parameters of the gaussian kernel
               (signal-variance (->weight :size 1))
               (length-scale (->weight :size 1))
               (roughness (->weight :size 1))
               ;; repped versions
               (signal-variance* (->rep :x signal-variance :n n-cov))
               (length-scale* (->rep :x length-scale :n n-cov))
               (roughness* (->rep :x roughness :n n-cov))
               ;;
               (noise-variance (->weight :size 1))
               (bias-variance (->weight :size 1))
               (abs-noise-variance (->abs :x noise-variance))
               (abs-bias-variance (->abs :x bias-variance))
               ;;
               (covariances-1 (->rough-exponential
                               :x distance-field
                               :signal-variance signal-variance*
                               :length-scale length-scale*
                               :roughness roughness*))
               (covariances-2 (->ref :index selfp
                                     :into abs-noise-variance
                                     :drop-negative-index-p t))
               (covariances-3 (->rep :x abs-bias-variance :n n-cov))
               (covariances (->+ :args (list covariances-1
                                             covariances-2
                                             covariances-3)))
               (gp (->gp :means means :covariances covariances))
               (error (->error :x gp)))))
      (setf (max-n-stripes bpn-gp) 10)
      (unless weights-from
        (fill-lump 'means-bias bpn-gp 0)
        (fill-lump 'signal-variance bpn-gp 0.1)
        (fill-lump 'length-scale bpn-gp 0.1)
        (fill-lump 'roughness bpn-gp 2)
        (fill-lump 'noise-variance bpn-gp 0.01)
        (fill-lump 'bias-variance bpn-gp 0.01))
      bpn-gp)))

(defun random-in-test-domain ()
  (random 1d0))

(defun test-target (x)
  x)

(defun make-input-output (n)
  (let* ((inputs (coerce (loop repeat n collect (random-in-test-domain))
                         'vector))
         (outputs (map 'vector #'test-target inputs)))
    (values inputs outputs)))

(defun test-simple-bpn-gp ()
  (let* ((bpn-gp (build-simple-bpn-gp 1 1))
         (sampler (make-instance
                   'counting-function-sampler
                   :max-n-samples 50000
                   :sampler (lambda ()
                              (multiple-value-bind (inputs outputs)
                                  (make-input-output 5)
                                (list :x1 inputs :x2 inputs :y1 outputs)))))
         (trainer (make-instance
                   'test-bpn-gp-gd-trainer
                   :segmenter
                   (lambda (lump)
                     (let ((learning-rate
                             (case (name lump)
                               ((means-bias)
                                (flt 0.1))
                               ((length-scale)
                                (flt 0.1))
                               ((signal-variance)
                                (flt 0.1))
                               ((bias-variance)
                                (flt 0.1))
                               ;; it's easy to hit numerical stability
                               ;; problems when training these guys
                               ((noise-variance roughness)
                                nil))))
                       (when learning-rate
                         (make-instance 'batch-gd-trainer
                                        :learning-rate (flt learning-rate)
                                        :momentum (flt 0.9)
                                        :batch-size 100)))))))
    (train sampler trainer bpn-gp)))

#|

;;; Train the covariance function with backprop.
(test-simple-bpn-gp)

;;; Plotting.
(let* ((prior (make-instance
               'prior-gp
               :mean-fn (constantly 5)
               :covariance-fn (lambda (x1 x2)
                                (+ (* 5 (exp (- (expt (/ (- x1 x2) 10) 2))))
                                   (if (= x1 x2)
                                       1
                                       0)))))
       (inputs (vector 1.7798 -17.324))
       (outputs (vector 2 9))
       (posterior (update-gp prior inputs outputs))
       (x (coerce (loop for i upfrom -500 below 500
                        collect (* i (flt 0.1)))
                  'vector)))
  ;; Plot samples from the prior.
  (mgl-gnuplot:with-session ()
    (format mgl-gnuplot:*command-stream*
            "set title 'samples'~%")
    (mgl-gnuplot:plot* (loop repeat 3
                             collect (gp-samples-as-plot-data
                                      prior x
                                      :options "title 'samples' with lines"))))
  ;; Plot posterior confidence.
  (mgl-gnuplot:with-session ()
    (format mgl-gnuplot:*command-stream*
            "set title '95% confidence interval'~%")
    (mgl-gnuplot:plot*
     (list*
      (mgl-gnuplot:data (aops:stack 1 (aops:reshape inputs '(2 1))
                                    (aops:reshape outputs '(2 1)))
                        "title 'observations' with points")
      (gp-confidences-as-plot-data posterior x)))))

|#
