(in-package :mgl-example-gp)

(defclass test-fnn-gp (parameterized-executor-cache-mixin fnn-gp)
  ())

(defmethod make-executor-with-parameters (parameters (fnn-gp test-fnn-gp))
  (destructuring-bind (n-x1 n-x2) parameters
    (build-simple-fnn-gp n-x1 n-x2 :weights-from fnn-gp)))

(defmethod set-input (samples (bpn test-fnn-gp))
  (let ((df (find-clump 'distance-field bpn))
        (selfp (find-clump 'selfp bpn)))
    (with-facets ((df* ((nodes df) 'backing-array :direction :output
                        :type flt-vector))
                  (selfp* ((nodes selfp) 'backing-array :direction :output
                           :type flt-vector)))
      (loop for sample in samples
            for stripe upfrom 0
            do (destructuring-bind (&key ((:x1 x1-vec)) ((:x2 x2-vec))
                                    ((:y1 y1-vec)) &allow-other-keys)
                   sample
                 (declare (ignore y1-vec))
                 (let ((n1 (mat-size x1-vec))
                       (n2 (mat-size x2-vec)))
                   (with-facets ((x1-vec (x1-vec 'backing-array
                                                 :direction :input
                                                 :type flt-vector))
                                 (x2-vec (x2-vec 'backing-array
                                                 :direction :input
                                                 :type flt-vector)))
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
                                       ;; FIXME: identity is not
                                       ;; numeric
                                       (if (= a b)
                                           (flt 0)
                                           (flt -1)))
                                 (incf dfi)
                                 (incf selfpi))))))))))))))

(defclass test-fnn-gp-base-optimizer ()
  ())

(defun log-training-period (optimizer learner)
  (declare (ignore optimizer learner))
  100)

(defun log-test-period (optimizer learner)
  (declare (ignore optimizer learner))
  100)

(defun make-test-fnn-gp-monitors (&key attributes)
  (cons (make-instance
         'monitor
         :measurer (lambda (samples bpn)
                     (declare (ignore samples))
                     (cost bpn))
         :counter (make-instance
                   'basic-counter
                   :prepend-attributes `(,@attributes
                                         :type "average neg log likelihood")))
        (loop for name in '(means-bias signal-variance length-scale
                            roughness noise-variance bias-variance)
              collect (make-instance
                       'monitor
                       :measurer (let ((name name))
                                   (lambda (samples bpn)
                                     (declare (ignore samples))
                                     (values (mat-as-scalar
                                              (nodes (find-clump name bpn)))
                                             1)))
                       :counter (make-instance
                                 'basic-counter
                                 :prepend-attributes
                                 `(,@attributes
                                   :var ,(string-downcase
                                          (symbol-name name))))))))

(defun log-bpn-test-error (optimizer learner)
  (declare (ignore optimizer))
  (let ((counter (make-instance 'rmse-counter
                                :prepend-attributes '(:event "pred."
                                                      :dataset "test"))))
    (loop repeat 100 do
      (multiple-value-bind (inputs outputs) (make-input-output 5)
        (let ((gp (update-gp (bpn learner) inputs outputs)))
          (loop repeat 10
                do (let* ((x (random-in-test-domain))
                          (y (test-target x))
                          (y* (mat-as-scalar (gp-means gp (scalar-as-mat x)))))
                     (add-to-counter counter (expt (- y y*) 2) 1))))))
    (log-msg "~A~%" counter))
  (log-mat-room)
  (log-msg "---------------------------------------------------~%"))

(defclass test-fnn-gp-gd-optimizer (test-fnn-gp-base-optimizer
                                    segmented-gd-optimizer)
  ())

(defun fill-lump (name bpn value)
  (let ((mat (segment-weights (find-clump name bpn))))
    (fill! value mat)))

(defun build-simple-fnn-gp (n-x1 n-x2 &key weights-from)
  (with-weights-copied (weights-from)
    (let* ((n-cov (* n-x1 n-x2))
           (fnn-gp
             (build-fnn (:class 'test-fnn-gp
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
               (means-bias* (->rep means-bias n-x1))
               (means (->+ (list means-bias*)))
               ;; parameters of the gaussian kernel
               (signal-variance (->weight :size 1))
               (length-scale (->weight :size 1))
               (roughness (->weight :size 1))
               ;; repped versions
               (signal-variance* (->rep signal-variance n-cov))
               (length-scale* (->rep length-scale n-cov))
               (roughness* (->rep roughness n-cov))
               ;;
               (noise-variance (->weight :size 1))
               (bias-variance (->weight :size 1))
               (abs-noise-variance (->abs noise-variance))
               (abs-bias-variance (->abs bias-variance))
               ;;
               (covariances-1 (->rough-exponential
                               distance-field
                               :signal-variance signal-variance*
                               :length-scale length-scale*
                               :roughness roughness*))
               (covariances-2 (->ref :index selfp
                                     :into abs-noise-variance
                                     :drop-negative-index-p t))
               (covariances-3 (->rep abs-bias-variance n-cov))
               (covariances (->+ (list covariances-1
                                       covariances-2
                                       covariances-3)))
               (gp (->gp :means means :covariances covariances))
               (error (->loss gp)))))
      (setf (max-n-stripes fnn-gp) 10)
      (unless weights-from
        (fill-lump 'means-bias fnn-gp 0)
        (fill-lump 'signal-variance fnn-gp 0.1)
        (fill-lump 'length-scale fnn-gp 0.1)
        (fill-lump 'roughness fnn-gp 2)
        (fill-lump 'noise-variance fnn-gp 0.01)
        (fill-lump 'bias-variance fnn-gp 0.01))
      fnn-gp)))

(defun random-in-test-domain ()
  (random 1d0))

(defun test-target (x)
  x)

(defun make-input-output (n)
  (let* ((inputs (coerce (loop repeat n collect (random-in-test-domain))
                         'vector))
         (outputs (map 'vector #'test-target inputs)))
    (values (array-to-mat inputs :ctype flt-ctype)
            (array-to-mat outputs :ctype flt-ctype))))

(defun test-simple-fnn-gp ()
  (let* ((fnn-gp (build-simple-fnn-gp 1 1))
         (sampler (make-instance
                   'function-sampler
                   :max-n-samples 50000
                   :generator (lambda ()
                                (multiple-value-bind (inputs outputs)
                                    (make-input-output 5)
                                  (list :x1 inputs :x2 inputs :y1 outputs)))))
         (optimizer (make-instance
                     'test-fnn-gp-gd-optimizer
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
                           (make-instance 'sgd-optimizer
                                          :learning-rate (flt learning-rate)
                                          :momentum (flt 0.9)
                                          :batch-size 100)))))))
    (monitor-optimization-periodically
     optimizer '((:fn log-bpn-test-error
                  :period log-test-period)
                 (:fn reset-optimization-monitors
                  :period log-training-period
                  :last-eval 0)))
    (minimize optimizer
              (make-instance 'bp-learner
                             :bpn fnn-gp
                             :monitors (make-test-fnn-gp-monitors
                                        :attributes '(:event "train"
                                                      :dataset "train+")))
              :dataset sampler)))

#|

;;; Train the covariance function with backprop.
(with-cuda* ()
  (let ((*random-state* (sb-ext:seed-random-state 983274)))
    (test-simple-fnn-gp)))

;;; Plotting.
(let* ((prior (make-instance
               'prior-gp
               :mean-fn (constantly 5)
               :covariance-fn (lambda (x1 x2)
                                (+ (* 5 (exp (- (expt (/ (- x1 x2) 10) 2))))
                                   (if (= x1 x2)
                                       1
                                       0)))))
       (inputs (array-to-mat #(1.7798 -17.324)))
       (outputs (array-to-mat #(2 9)))
       (posterior (update-gp prior inputs outputs))
       (x (array-to-mat (coerce (loop for i upfrom -500 below 500
                                      collect (* i (flt 0.1)))
                                'vector))))
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
      (mgl-gnuplot:data (mat-to-array
                         (stack 1 (list (reshape! inputs '(2 1))
                                        (reshape! outputs '(2 1)))))
                        "title 'observations' with points")
      (gp-confidences-as-plot-data posterior x)))))

|#
