(in-package :mgl-gp)

;;;; Gaussian processes

;;; Gaussian processes are defined by the mean and covariance
;;; functions.
(defclass gp ()
  ())

(defgeneric gp-means (gp x)
  (:documentation "Returns the vector of means for the vector of
  inputs X. X is a vector of arbitrary objects."))

(defgeneric gp-covariances* (gp x1 x2)
  (:documentation "Returns the matrix of covariances between X1 and
  X2. X1 and X2 are vectors of arbitrary objects. Noise is assumed to
  be included in the covariance function."))

(defgeneric gp-means-and-covariances* (gp x1 x2)
  (:documentation "Returns two values: the means and the covariances
  as matrices.")
  (:method (gp x1 x2)
    (values (gp-means gp x1)
            (gp-covariances* gp x1 x2))))

(defun gp-covariances (gp x1 &optional (x2 x1))
  (gp-covariances* gp x1 x2))

(defun gp-means-and-covariances (gp x1 &optional (x2 x1))
  (gp-means-and-covariances* gp x1 x2))

(defun update-gp (gp inputs outputs &key means covariances)
  "Update GP with the evidence embodied by INPUTS and the
  corresponding OUTPUTS. Return a new POSTERIOR-GP. If MEANS and
  COVARIANCES are given, then GP-MEANS-AND-COVARIANCES is not called."
  (multiple-value-bind (means covariances)
      (if (and means covariances)
          (values means covariances)
          (gp-means-and-covariances gp inputs inputs))
    (update-gp* gp inputs outputs means covariances)))

(defun sample-gp (gp inputs &key means covariances)
  "Return a sample from the multivariate normal distribution defined
  by GP at INPUTS as a column vector."
  (multiple-value-bind (means covariances)
      (if (and means covariances)
          (values means covariances)
          (gp-means-and-covariances gp inputs))
    (values (mv-gaussian-random :means means :covariances covariances)
            means covariances)))


;;;; PRIOR-GP

;;; A gp defined by two lisp functions.
(defclass prior-gp (gp)
  ((mean-fn :initarg :mean-fn :reader mean-fn)
   (covariance-fn :initarg :covariance-fn :reader covariance-fn))
  (:documentation "A GP whose mean and covariance are defined by two
  lisp functions. Can be updated, but it's not trainable."))

(defmethod gp-means ((gp prior-gp) x)
  ;; mean_i = (funcall mean-fn x_i)
  (let* ((n (mat-size x))
         (mean-fn (mean-fn gp))
         (means (make-mat n :ctype flt-ctype)))
    (with-facets ((means (means 'backing-array :direction :output
                                :type flt-vector))
                  (x (x 'backing-array :direction :input
                        :type flt-vector)))
      (dotimes (i n)
        (setf (aref means i)
              (flt (funcall mean-fn (aref x i))))))
    means))

(defmethod gp-covariances* ((gp prior-gp) x1 x2)
  ;; cov_ij = (funcall covariance-fn x_i x_j)
  (let* ((n1 (mat-size x1))
         (n2 (mat-size x2))
         (covariance-fn (covariance-fn gp))
         (covariances (make-mat (list n1 n2) :ctype flt-ctype)))
    (with-facets ((covariances (covariances 'array :direction :output))
                  (x1 (x1 'backing-array :direction :input :type flt-vector))
                  (x2 (x2 'backing-array :direction :input :type flt-vector)))
      (dotimes (row n1)
        (dotimes (col n2)
          (setf (aref covariances row col)
                (flt (funcall covariance-fn (aref x1 row) (aref x2 col)))))))
    covariances))


;;;; POSTERIOR-GP

;;; A gp that takes some evidence into account (see UPDATE-GP). Can't
;;; be trained.
(defclass posterior-gp (gp)
  ((prior-gp :initarg :prior-gp :reader prior-gp)
   (inverted-covariances :initarg :inverted-covariances
                         :reader inverted-covariances)
   (evidence-inputs :initarg :evidence-inputs :reader evidence-inputs)
   (evidence-outputs :initarg :evidence-outputs :reader evidence-outputs)
   (centered-evidence-outputs
    :initarg :centered-evidence-outputs
    :reader centered-evidence-outputs)))

(defun update-gp* (gp inputs outputs means covariances)
  (if (zerop (mat-size inputs))
      gp
      (let* ((n (mat-size inputs))
             (centered-outputs (make-mat (list n 1) :ctype flt-ctype)))
        (copy! outputs centered-outputs)
        (axpy! -1 means centered-outputs)
        (make-instance 'posterior-gp
                       :prior-gp gp
                       :inverted-covariances (invert covariances)
                       :evidence-inputs inputs
                       :evidence-outputs outputs
                       :centered-evidence-outputs centered-outputs))))

(defun posterior-gp-means-and-covariances (gp x1 x2
                                           &key compute-covariances-p)
  (let* ((inverted-covariances (inverted-covariances gp))
         (evidence-inputs (evidence-inputs gp))
         (centered-evidence-outputs (centered-evidence-outputs gp))
         (prior (prior-gp gp)))
    (multiple-value-bind (mean-1 cov-1-e)
        (gp-means-and-covariances prior x1 evidence-inputs)
      (let* ((a (m* cov-1-e inverted-covariances))
             (output (m* a centered-evidence-outputs)))
        (axpy! 1 mean-1 output)
        (values
         output
         (when compute-covariances-p
           (let ((cov-1-2 (gp-covariances prior x1 x2)))
             (if (eq x1 x2)
                 ;; optimization: cov = cov-1-2 - a * cov-1-e^t
                 (gemm! -1d0 a cov-1-e 1d0 cov-1-2 :transpose-b? t)
                 ;; general case: cov = cov-1-2 - a * cov-e-2
                 (let ((cov-e-2 (gp-covariances prior evidence-inputs x2)))
                   (gemm! -1d0 a cov-e-2 1d0 cov-1-2)))
             cov-1-2)))))))

(defmethod gp-means ((gp posterior-gp) x)
  (values (posterior-gp-means-and-covariances gp x x)))

(defmethod gp-covariances* ((gp posterior-gp) x1 x2)
  (nth-value 1 (posterior-gp-means-and-covariances
                gp x1 x2 :compute-covariances-p t)))

(defmethod gp-means-and-covariances* ((gp posterior-gp) x1 x2)
  (posterior-gp-means-and-covariances gp x1 x2 :compute-covariances-p t))


;;; See [3] for explanations of parameters.
(defun gaussian-kernel (x1 x2 &key signal-variance (bias-variance 0)
                                length-scale (roughness 2))
  (+ (* signal-variance
        (exp (* -0.5d0
                (expt (/ (abs (- x1 x2))
                         length-scale)
                      roughness))))
     bias-variance))


;;;; BPN-GP

;;; A gp with BPN based mean and covariance. Can be trained and
;;; updated.
;;;
;;; It is a bpn with two of its lumps standing for mean and
;;; covariance. SET-INPUT is expected to take lists of elements of the
;;; form (&KEY X1 X2 Y1 &ALLOW-OTHER-KEYS) where X1 and X2 are inputs
;;; as in GP-COVARIANCES, and Y1 is the output for X1. The mean lump
;;; is supposed to calculate the means for X1, and the covariance lump
;;; the X1/X2 covariances.
;;;
;;; Y1 goes unused during mean/covariance calculation (hence
;;; prediction, too), it's there for training the gp.
;;;
;;; During training X1 must be EQ to X2.
(defclass bpn-gp (bpn gp)
  ((mean-lump-name :initarg :mean-lump-name
                   :reader mean-lump-name)
   (covariance-lump-name :initarg :covariance-lump-name
                         :reader covariance-lump-name)))

;;; For subclasses that use TRIVIAL-CACHED-EXECUTOR-MIXIN or such.
(defmethod sample-to-executor-cache-key (sample (bpn-gp bpn-gp))
  (destructuring-bind (&key x1 x2 &allow-other-keys) sample
    (list (mat-size x1) (mat-size x2))))

(defun make-vector-from-lump-stripe (lump stripe)
  (copy-row (nodes lump) stripe))

(defun make-matrix-from-lump-stripe (lump n-rows n-cols stripe)
  (reshape! (copy-row (nodes lump) stripe) (list n-rows n-cols)))

(defun extract-means (lump stripe)
  (make-vector-from-lump-stripe lump stripe))

(defun extract-covariances (lump stripe n-rows n-cols)
  (make-matrix-from-lump-stripe lump n-rows n-cols stripe))

(defmethod gp-means-and-covariances* ((bpn bpn-gp) x1 x2)
  (let ((samples (list (list :x1 x1 :x2 x2))))
    (do-executors (samples bpn)
      (let ((gp-lump (find-gp-lump bpn)))
        (set-input samples bpn)
        (forward-bpn bpn
                     ;; Skip the heavy duty part that's only necessary
                     ;; for training anyway.
                     :end-lump gp-lump)
        (return-from gp-means-and-covariances*
          (values (extract-means (means gp-lump) 0)
                  (extract-covariances (covariances gp-lump) 0
                                       (mat-size x1) (mat-size x2))))))))

(defmethod gp-means ((bpn bpn-gp) x)
  (nth-value 0 (gp-means-and-covariances bpn x)))

(defmethod gp-covariances* ((bpn bpn-gp) x1 x2)
  (nth-value 1 (gp-means-and-covariances bpn x1 x2)))


;;;; BPN-GP training

(defmethod set-input :after (samples (bpn bpn-gp))
  (let ((gp-lump (find-gp-lump bpn)))
    (when gp-lump
      (setf (samples gp-lump) samples))))

;;; The after method on SET-INPUT stores SAMPLES here. The MEANS and
;;; COVARIANCES lumps are supposed to compute the means and
;;; covariances of the inputs. The output is the log-likelihood of
;;; evidence.
(deflump ->gp (lump)
  ((means :initarg :means :reader means)
   (covariances :initarg :covariances :reader covariances)
   (samples :accessor samples)
   ;; one gp per stripe
   (posterior-gps :accessor posterior-gps)))

(defmethod default-size ((lump ->gp))
  1)

(defmethod transfer-lump ((lump ->gp))
  ;; be kind to gc in case gps are huge
  (setf (posterior-gps lump) ())
  (setf (posterior-gps lump)
        (let ((means (means lump))
              (covariances (covariances lump)))
          (loop for stripe upfrom 0
                for sample in (samples lump)
                collect
                (destructuring-bind (&key x1 x2 y1 &allow-other-keys) sample
                  (assert (eq x1 x2))
                  ;; We get away with pass NIL as the gp, because
                  ;; means and covariances are passed in directly and
                  ;; we are not going to call
                  ;; GP-MEANS-AND-COVARIANCES.
                  (update-gp nil x1 y1
                             :means (extract-means means stripe)
                             :covariances (extract-covariances
                                           covariances stripe
                                           (mat-size x1) (mat-size x1)))))))
  (with-facets ((nodes ((nodes lump) 'backing-array :direction :output
                        :type flt-vector)))
    (loop for stripe of-type index below (mgl-bp::n-stripes* lump)
          for gp in (posterior-gps lump)
          do (let* ((inverted-covariances (inverted-covariances gp))
                    (centered-outputs (centered-evidence-outputs gp))
                    (n (mat-size centered-outputs)))
               (setf (aref nodes stripe)
                     (flt
                      (* 0.5
                         (+ (- (logdet inverted-covariances))
                            (mat-as-scalar
                             (mm* (list centered-outputs :transpose? t)
                                  inverted-covariances
                                  centered-outputs))
                            (* n (log (* 2 pi)))))))))))

(defmethod derive-lump ((lump ->gp))
  (let* ((means (means lump))
         (means-d (derivatives means))
         (covariances (covariances lump))
         (cov-d (derivatives covariances)))
    (with-facets ((derivatives ((derivatives lump) 'backing-array
                                :direction :input :type flt-vector)))
      (loop for stripe of-type index below (mgl-bp::n-stripes* lump)
            for gp in (posterior-gps lump)
            do (let* ((d (aref derivatives stripe))
                      (inverted-covariances (inverted-covariances gp))
                      (centered-outputs (centered-evidence-outputs gp))
                      (dmean (m* centered-outputs inverted-covariances
                                 :transpose-a? t))
                      (2dcov (m- (mm* inverted-covariances
                                      centered-outputs
                                      (list centered-outputs :transpose? t)
                                      inverted-covariances)
                                 inverted-covariances)))
                 (with-shape-and-displacement (means-d)
                   (reshape-to-row-matrix! means-d stripe)
                   (axpy! (- d) dmean means-d))
                 (with-shape-and-displacement (cov-d)
                   (reshape-to-row-matrix! cov-d stripe)
                   (axpy! (- (* d 0.5)) 2dcov cov-d)))))))

(defun find-gp-lump (bpn)
  (find-if (lambda (lump)
             (typep lump '->gp))
           (lumps bpn)))


;;;; Utilities for plotting

(defun gp-confidences-as-plot-data
    (gp inputs &key means covariances
     (levels-and-options '((0 "title 'mean' with lines")
                           (-1.96 "title 'mean - 1.96 * stddev' with lines")
                           (1.96
                            "title 'mean + 1.96 * stddev' with lines"))))
  "Return a list of MGL-GNUPLOT:DATA-MAPPINGs, one for each level in
  LEVELS-AND-OPTIONS (a list of (LEVEL OPTIONS)). Each mapping
  contains INPUTS in its first column, and MEANS + LEVEL*VARIANCES in
  the second."
  (multiple-value-bind (means covariances)
      (if (and means covariances)
          (values means covariances)
          (gp-means-and-covariances gp inputs))
    (loop for (level options) in levels-and-options
          collect (mgl-gnuplot:data*
                   (gp-data-matrix-for-level inputs means covariances level)
                   options))))

(defun gp-samples-as-plot-data (gp inputs &key means covariances options)
  "Returns a matrix that contains INPUTS in its first column, and a
  sample taken with SAMPLE-GP in its second."
  (mgl-gnuplot:data*
   (gp-data-matrix inputs (sample-gp gp inputs :means means
                                     :covariances covariances))
   options))

(defun gp-data-matrix (inputs outputs)
  (aops:stack* 'flt 1 (as-column-vector inputs) (as-column-vector outputs)))

(defun gp-data-matrix-for-level (inputs means covariances level)
  (gp-data-matrix inputs
                  (clnu:e+ (as-column-vector means)
                           (clnu:e* level
                                    (clnu:esqrt
                                     (as-column-vector
                                      (clnu:diagonal-vector covariances)))))))
