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
