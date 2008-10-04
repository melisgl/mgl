(in-package :mgl-example-util)

(defun time->string (&optional (time (get-universal-time)))
  (destructuring-bind (second minute hour date month year)
      (subseq (multiple-value-list (decode-universal-time time)) 0 6)
    (format nil "~4,'0D-~2,'0D-~2,'0D ~2,'0D:~2,'0D:~2,'0D"
            year month date hour minute second)))

(defparameter *example-dir*
  (merge-pathnames (make-pathname :directory '(:relative "example"))
                   (make-pathname :name nil :type nil
                                  :defaults (asdf:component-pathname
                                             (asdf:find-system :mgl-example)))))

(defparameter *log-file*
  (merge-pathnames "mgl-example.log" *example-dir*))

(defun log-msg (format &rest args)
  (format *trace-output* "~&~A: " (time->string))
  (apply #'format *trace-output* format args)
  (with-open-file (s *log-file* :direction :output
                   :if-exists :append :if-does-not-exist :create)
    (format s "~A: " (time->string))
    (apply #'format s format args)))

(defun nshuffle-vector (vector)
  "Shuffle a vector in place using Fisher-Yates algorithm."
  (loop for idx downfrom (1- (length vector)) to 1
        for other = (random (1+ idx))
        do (unless (= idx other)
             (rotatef (aref vector idx) (aref vector other))))
  vector)

(defun make-random-generator (vector)
  "Return a function that returns elements of VECTOR in random order
without end. When there are no more elements, start over with a
different random order."
  (let ((vector (copy-seq (coerce vector 'vector)))
        (l (length vector))
        (n 0))
    (lambda ()
      (when (zerop n)
        (setq vector (nshuffle-vector vector)))
      (prog1
          (aref vector n)
        (setf n (mod (1+ n) l))))))

;;; See "Accurate Methods for the Statistics of Surprise and
;;; Coincidence" by Ted Dunning
;;; (http://citeseer.ist.psu.edu/29096.html)
(defun log-likelihood-ratio (k1 n1 k2 n2)
  ;; Add a few positive and negative occurrences as a prior.
  (let ((k1 (+ 1 k1))
        (k2 (+ 1 k2))
        (n1 (+ 2 n1))
        (n2 (+ 2 n2)))
    (flet ((l (p k n)
             (+ (* k (log p))
                (* (- n k) (log (- 1 p))))))
      (let ((p1 (/ k1 n1))
            (p2 (/ k2 n2))
            (p (/ (+ k1 k2) (+ n1 n2))))
        (+ (- (l p k1 n1))
           (- (l p k2 n2))
           (l p1 k1 n1)
           (l p2 k2 n2))))))


;;;; PERIODIC-FN

(defclass periodic-fn ()
  ((period :initarg :period :reader period)
   (fn :initarg :fn :reader fn)
   (last-eval :initform nil :initarg :last-eval :accessor last-eval)))

(defun call-periodic-fn (n fn &rest args)
  (let ((period (period fn)))
    (when (typep period '(or symbol function))
      (setq period (apply period args)))
    (when (or (null (last-eval fn))
              (and (/= (floor n period)
                       (floor (last-eval fn) period))))
      (setf (last-eval fn) n)
      (apply (fn fn) args))))

(defun call-periodic-fn! (n fn &rest args)
  (when (or (null (last-eval fn))
            (and (/= n (last-eval fn))))
    (setf (last-eval fn) n)
    (apply (fn fn) args)))


;;;; Logging trainer

(defgeneric log-training-error (trainer learner))
(defgeneric log-test-error (trainer learner))
(defgeneric log-training-period (trainer learner))
(defgeneric log-test-period (trainer learner))

(defclass logging-trainer ()
  ((log-training-fn :initform (make-instance 'periodic-fn
                                             :period 'log-training-period
                                             :fn 'log-training-error)
                    :reader log-training-fn)
   (log-test-fn :initform (make-instance 'periodic-fn
                                         :period 'log-test-period
                                         :fn 'log-test-error)
                :reader log-test-fn)))

(defmethod mgl-train:train-batch :around
    (samples (trainer logging-trainer) learner)
  (multiple-value-prog1 (call-next-method)
    (call-periodic-fn (mgl-train:n-inputs trainer) (log-training-fn trainer)
                      trainer learner)
    (call-periodic-fn (mgl-train:n-inputs trainer) (log-test-fn trainer)
                      trainer learner)))

(defmethod mgl-train:train :around (sampler (trainer logging-trainer) learner)
  (setf (last-eval (log-training-fn trainer))
        (mgl-train:n-inputs trainer))
  (call-periodic-fn! (mgl-train:n-inputs trainer)
                     (log-test-fn trainer) trainer learner)
  (multiple-value-prog1 (call-next-method)
    (call-periodic-fn! (mgl-train:n-inputs trainer) (log-training-fn trainer)
                       trainer learner)
    (call-periodic-fn! (mgl-train:n-inputs trainer) (log-test-fn trainer)
                       trainer learner)))
