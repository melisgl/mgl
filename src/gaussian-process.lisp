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


;;;; FNN-GP

;;; A gp with FNN based mean and covariance. Can be trained and
;;; updated.
;;;
;;; It is a fnn with two of its lumps standing for mean and
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
(defclass fnn-gp (fnn gp)
  ((mean-lump-name :initarg :mean-lump-name
                   :reader mean-lump-name)
   (covariance-lump-name :initarg :covariance-lump-name
                         :reader covariance-lump-name)))

;;; For subclasses that use TRIVIAL-CACHED-EXECUTOR-MIXIN or such.
(defmethod instance-to-executor-parameters (sample (fnn-gp fnn-gp))
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

(defmethod gp-means-and-covariances* ((fnn fnn-gp) x1 x2)
  (let ((samples (list (list :x1 x1 :x2 x2))))
    (do-executors (samples fnn)
      (let ((gp-lump (find-gp-lump fnn)))
        (set-input samples fnn)
        (mgl-bp::forward-bpn fnn
                             ;; Skip the heavy duty part that's only necessary
                             ;; for training anyway.
                             :end-clump gp-lump)
        (return-from gp-means-and-covariances*
          (values (extract-means (means gp-lump) 0)
                  (extract-covariances (covariances gp-lump) 0
                                       (mat-size x1) (mat-size x2))))))))

(defmethod gp-means ((fnn fnn-gp) x)
  (nth-value 0 (gp-means-and-covariances fnn x)))

(defmethod gp-covariances* ((fnn fnn-gp) x1 x2)
  (nth-value 1 (gp-means-and-covariances fnn x1 x2)))


;;;; FNN-GP training

(defmethod set-input :after (samples (fnn fnn-gp))
  (let ((gp-lump (find-gp-lump fnn)))
    (when gp-lump
      (setf (samples gp-lump) samples))))

;;; The after method on SET-INPUT stores SAMPLES here. The MEANS and
;;; COVARIANCES lumps are supposed to compute the means and
;;; covariances of the inputs. The output is the log-likelihood of
;;; evidence.
(defclass-now ->gp (lump)
  ((means :initarg :means :reader means)
   (covariances :initarg :covariances :reader covariances)
   (samples :accessor samples)
   ;; one gp per stripe
   (posterior-gps :accessor posterior-gps)))

(defmaker (->gp))

(defmethod default-size ((lump ->gp))
  1)

(defmethod forward ((lump ->gp))
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
    (loop for stripe of-type index below (n-stripes lump)
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

(defmethod backward ((lump ->gp))
  (let* ((means (means lump))
         (means-d (derivatives means))
         (covariances (covariances lump))
         (cov-d (derivatives covariances)))
    (with-facets ((derivatives ((derivatives lump) 'backing-array
                                :direction :input :type flt-vector)))
      (loop for stripe of-type index below (n-stripes lump)
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

(defun find-gp-lump (fnn)
  (find-if (lambda (lump)
             (typep lump '->gp))
           (clumps fnn)))


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
  (mat-to-array
   (with-shape-and-displacement (inputs)
     (with-shape-and-displacement (outputs)
       (reshape! inputs (list (mat-size inputs) 1))
       (reshape! outputs (list (mat-size outputs) 1))
       (stack 1 (list inputs outputs))))))

(defun gp-data-matrix-for-level (inputs means covariances level)
  (gp-data-matrix inputs
                  (m+ means
                      (array-to-mat
                       (clnu:e* level
                                (clnu:esqrt
                                 (as-column-vector
                                  (clnu:diagonal-vector
                                   (mat-to-array covariances)))))))))


(defclass-now ->ref (lump)
  ((index :initarg :index :reader index)
   (into :initarg :into :reader into)
   (drop-negative-index-p
    :initform nil
    :initarg :drop-negative-index-p
    :reader drop-negative-index-p)))

(defmaker (->ref))

(defmethod default-size ((lump ->ref))
  (size (index lump)))

(defmethod forward ((lump ->ref))
  (let* ((index (index lump))
         (into (into lump))
         (n (size into))
         (drop-negative-index-p (drop-negative-index-p lump)))
    (with-facets ((l* ((nodes lump) 'backing-array :direction :output
                       :type flt-vector))
                  (index* ((nodes index) 'backing-array :direction :input
                           :type flt-vector))
                  (into* ((nodes into) 'backing-array :direction :input
                          :type flt-vector)))
      (assert (= (size lump) (size index)))
      (loop for stripe of-type index below (n-stripes lump) do
        (with-stripes ((stripe lump ls le)
                       (stripe index index-s index-e)
                       (stripe into into-s))
          (loop for li upfrom ls below le
                for index-i upfrom index-s below index-e
                do (let ((into-i (round (aref index* index-i))))
                     (assert (and (or drop-negative-index-p (<= 0 into-i))
                                  (< into-i n)))
                     (when (<= 0 into-i)
                       (setf (aref l* li)
                             (aref into* (+ into-s into-i)))))))))))

(defmethod backward ((lump ->ref))
  (let ((index (index lump))
        (into (into lump)))
    (assert (= (size lump) (size index)))
    (assert (typep index '->input))
    (with-facets ((d* ((derivatives lump) 'backing-array :direction :input
                       :type flt-vector))
                  (index* ((nodes index) 'backing-array :direction :input
                           :type flt-vector))
                  (intod* ((derivatives into) 'backing-array :direction :io
                           :type flt-vector)))
      (loop for stripe of-type index below (n-stripes lump) do
        (with-stripes ((stripe lump ls le)
                       (stripe index index-s index-e)
                       (stripe into into-s))
          (loop for li upfrom ls below le
                for index-i upfrom index-s below index-e
                do (let ((into-i (round (aref index* index-i))))
                     (when (<= 0 into-i)
                       (incf (aref intod* (+ into-s into-i))
                             (aref d* li))))))))))


(defclass-now ->rep (lump)
  ((x :initarg :x :reader x)
   (n :initarg :n :reader n)))

(defmaker (->rep :unkeyword-args (x n)))

(defmethod default-size ((lump ->rep))
  (* (n lump) (size (x lump))))

(defmethod forward ((lump ->rep))
  (let ((x (x lump)))
    ;; (assert (= (n-stripes lump) (n-stripes x)))
    (let ((n (n lump))
          (xn (size x)))
      (declare (optimize (speed 3) #.*no-array-bounds-check*)
               (type index n xn))
      (with-facets ((x* ((nodes x) 'backing-array :direction :input
                         :type flt-vector))
                    (to* ((nodes lump) 'backing-array :direction :output
                          :type flt-vector)))
        (loop for stripe of-type index below (n-stripes lump) do
          (with-stripes ((stripe x xs)
                         (stripe lump ls))
            (dotimes (i xn)
              (let ((v (aref x* (the! index (+ xs i)))))
                (loop for li of-type index upfrom (+ ls i) by xn
                      repeat n
                      do (setf (aref to* li) v))))))))))

(defmethod backward ((lump ->rep))
  (let ((x (x lump)))
    ;; (assert (= (n-stripes lump) (n-stripes x)))
    (let ((n (n lump))
          (xn (size x)))
      (declare (optimize (speed 3) #.*no-array-bounds-check*)
               (type index n xn))
      (with-facets ((xd* ((derivatives x) 'backing-array :direction :io
                          :type flt-vector))
                    (d* ((derivatives lump) 'backing-array :direction :input
                         :type flt-vector)))
        (loop for stripe of-type index below (n-stripes lump) do
          (with-stripes ((stripe x xs)
                         (stripe lump ls))
            (dotimes (i xn)
              (let ((sum (flt 0)))
                (loop for li of-type index upfrom (+ ls i) by xn
                      repeat n
                      do (incf sum (aref d* li)))
                (incf (aref xd* (+ xs i)) sum)))))))))


(defclass-now ->stretch (lump)
  ((x :initarg :x :reader x)
   (n :initarg :n :reader n)))

(defmaker (->stretch :unkeyword-args (x n)))

(defmethod default-size ((lump ->stretch))
  (* (n lump) (size (x lump))))

(defmethod forward ((lump ->stretch))
  (let ((x (x lump)))
    (assert (= (n-stripes lump) (n-stripes x)))
    (let ((n (n lump)))
      (declare (optimize (speed 3) #.*no-array-bounds-check*)
               (type index n))
      (with-facets ((x* ((nodes x) 'backing-array :direction :input
                         :type flt-vector))
                    (l* ((nodes lump) 'backing-array :direction :output
                         :type flt-vector)))
        (loop for stripe of-type index below (n-stripes lump) do
          (with-stripes ((stripe x xs xe)
                         (stripe lump ls))
            (let ((li ls))
              (loop for xi upfrom xs below xe
                    do (let ((v (aref x* xi)))
                         (loop repeat n
                               do (setf (aref l* li) v)
                                  (incf li)))))))))))

(defmethod backward ((lump ->stretch))
  (let ((x (x lump)))
    (assert (= (n-stripes lump) (n-stripes x)))
    (let ((n (n lump)))
      (declare (optimize (speed 3) #.*no-array-bounds-check*)
               (type index n))
      (with-facets ((xd* ((derivatives x) 'backing-array :direction :io
                          :type flt-vector))
                    (d* ((derivatives lump) 'backing-array :direction :input
                         :type flt-vector)))
        (loop for stripe of-type index below (n-stripes lump) do
          (with-stripes ((stripe x xs xe)
                         (stripe lump ls))
            (let ((li ls))
              (loop for xi upfrom xs below xe
                    do (let ((sum (flt 0)))
                         (loop repeat n
                               do (incf sum (aref d* li))
                                  (incf li))
                         (incf (aref xd* xi) sum))))))))))


(declaim (inline rough-exponential))
(defun rough-exponential (x &key signal-variance length-scale (roughness 2))
  (+ (* (abs signal-variance)
        (exp (* #.(flt -0.5)
                (if (zerop x)
                    #.(flt 0)
                    (expt (abs (/ x length-scale))
                          roughness)))))))

(declaim (inline derive-rough-exponential))
(defun derive-rough-exponential (x &key signal-variance length-scale
                                     (roughness 2))
  ;; d/dx(s^2*exp(-0.5*abs(x/l)^r)+b^2)
  (let* ((a0 (abs (/ x length-scale)))
         (a1 (if (zerop x) (flt 0) (expt a0 roughness)))
         (a2 (exp (* -0.5 a1)))
         (a3 (* #.(flt 0.5) roughness (abs signal-variance) a2)))
    (values
     ;; d/dx
     (if (zerop x)
         (flt 0)
         (- (/ (* a3 a1) x)))
     ;; d/dv
     (* (sign signal-variance) a2)
     ;; d/dl
     (/ (* a3 a1) length-scale)
     ;; d/r
     (if (zerop x)
         (flt 0)
         (* #.(flt -0.25) (abs signal-variance) a2 a1 (* 2 (log a0)))))))

(defclass-now ->rough-exponential (lump)
  ((x :initarg :x :reader x)
   (signal-variance :initarg :signal-variance :reader signal-variance)
   (length-scale :initarg :length-scale :reader length-scale)
   (roughness :initarg :roughness :reader roughness)))

(defmaker (->rough-exponential :unkeyword-args (x)))

(defmethod default-size ((lump ->rough-exponential))
  (size (x lump)))

(defmethod forward ((lump ->rough-exponential))
  (let ((x (x lump))
        (sv (signal-variance lump))
        (lsc (length-scale lump))
        (r (roughness lump)))
    (assert (= (size lump) (size x)))
    (with-facets ((l* ((nodes lump) 'backing-array :direction :output
                       :type flt-vector))
                  (x* ((nodes x) 'backing-array :direction :input
                       :type flt-vector))
                  (sv* ((nodes sv) 'backing-array :direction :input
                        :type flt-vector))
                  (lsc* ((nodes lsc) 'backing-array :direction :input
                         :type flt-vector))
                  (r* ((nodes r) 'backing-array :direction :input
                       :type flt-vector)))
      (declare (optimize (speed 3) #.*no-array-bounds-check*))
      (loop for stripe of-type index below (n-stripes lump) do
        (with-stripes ((stripe lump ls le)
                       (stripe x xs xe)
                       (stripe sv svs sve)
                       (stripe lsc lscs lsce)
                       (stripe r rs re))
          (loop for li upfrom ls below le
                for xi upfrom xs below xe
                for svi upfrom svs below sve
                for lsci upfrom lscs below lsce
                for ri upfrom rs below re
                do (setf (aref l* li)
                         (rough-exponential (aref x* xi)
                                            :signal-variance (aref sv* svi)
                                            :length-scale (aref lsc* lsci)
                                            :roughness (aref r* ri)))))))))

(defmethod backward ((lump ->rough-exponential))
  (let ((x (x lump))
        (sv (signal-variance lump))
        (lsc (length-scale lump))
        (r (roughness lump)))
    (assert (= (size lump) (size x)))
    (with-facets ((x* ((nodes x) 'backing-array :direction :input
                       :type flt-vector))
                  (sv* ((nodes sv) 'backing-array :direction :input
                        :type flt-vector))
                  (lsc* ((nodes lsc) 'backing-array :direction :input
                         :type flt-vector))
                  (r* ((nodes r) 'backing-array :direction :input
                       :type flt-vector))
                  (ld* ((derivatives lump) 'backing-array :direction :input
                        :type flt-vector))
                  (xd* ((derivatives x) 'backing-array :direction :io
                        :type flt-vector))
                  (svd* ((derivatives sv) 'backing-array :direction :io
                         :type flt-vector))
                  (lscd* ((derivatives lsc) 'backing-array :direction :io
                          :type flt-vector))
                  (rd* ((derivatives r) 'backing-array :direction :io
                        :type flt-vector)))
      (declare (optimize (speed 3) #.*no-array-bounds-check*))
      (loop for stripe of-type index below (n-stripes lump) do
        (with-stripes ((stripe lump ls le)
                       (stripe x xs xe)
                       (stripe sv svs sve)
                       (stripe lsc lscs lsce)
                       (stripe r rs re))
          (loop for li upfrom ls below le
                for xi upfrom xs below xe
                for svi upfrom svs below sve
                for lsci upfrom lscs below lsce
                for ri upfrom rs below re
                do (let ((d (aref ld* li)))
                     (multiple-value-bind (dx dsv dlsc dr)
                         (derive-rough-exponential
                          (aref x* xi)
                          :signal-variance (aref sv* svi)
                          :length-scale (aref lsc* lsci)
                          :roughness (aref r* ri))
                       (incf (aref xd* xi) (* d dx))
                       (incf (aref svd* svi) (* d dsv))
                       (incf (aref lscd* lsci) (* d dlsc))
                       (incf (aref rd* ri) (* d dr))))))))))


(defclass-now ->periodic (lump)
  ((x :initarg :x :reader x)
   (period :initarg :period :reader period)))

(defmaker (->periodic :unkeyword-args (x)))

(defmethod default-size ((lump ->periodic))
  (size (x lump)))

(defmethod forward ((lump ->periodic))
  (let ((x (x lump))
        (pe (period lump)))
    (assert (= (size lump) (size x)))
    (with-facets ((l* ((nodes lump) 'backing-array :direction :output
                       :type flt-vector))
                  (x* ((nodes x) 'backing-array :direction :input
                       :type flt-vector))
                  (pe* ((nodes pe) 'backing-array :direction :input
                        :type flt-vector)))
      ;; (declare (optimize (speed 3) #.*no-array-bounds-check*))
      (loop for stripe of-type index below (n-stripes lump) do
        (with-stripes ((stripe lump ls le)
                       (stripe x xs xe)
                       (stripe pe pes pee))
          (loop for li upfrom ls below le
                for xi upfrom xs below xe
                for pei upfrom pes below pee
                do (setf (aref l* li)
                         (sin (* #.(flt pi) (/ (aref x* xi)
                                               (aref pe* pei)))))))))))

(defmethod backward ((lump ->periodic))
  (let ((x (x lump))
        (pe (period lump)))
    (assert (= (size lump) (size x)))
    (with-facets ((ld* ((derivatives lump) 'backing-array :direction :input
                        :type flt-vector))
                  (x* ((nodes x) 'backing-array :direction :input
                       :type flt-vector))
                  (xd* ((derivatives x) 'backing-array :direction :io
                        :type flt-vector))
                  (pe* ((nodes pe) 'backing-array :direction :input
                        :type flt-vector))
                  (ped* ((derivatives pe) 'backing-array :direction :io
                         :type flt-vector)))
      ;; (declare (optimize (speed 3) #.*no-array-bounds-check*))
      (loop for stripe of-type index below (n-stripes lump) do
        (with-stripes ((stripe lump ls le)
                       (stripe x xs xe)
                       (stripe pe pes pee))
          (loop for li upfrom ls below le
                for xi upfrom xs below xe
                for pei upfrom pes below pee
                do (let* ((xv (aref x* xi))
                          (pev (aref pe* pei))
                          (d (aref ld* li))
                          (a (cos (/ (* #.(flt pi) xv)
                                     pev))))
                     (incf (aref xd* xi)
                           (* d (/ (* #.(flt pi) a)
                                   pev)))
                     (incf (aref ped* pei)
                           (* d (- (/ (* #.(flt pi) xv a)
                                      (expt pev 2))))))))))))
