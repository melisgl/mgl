;;;; Utilities

(in-package :mgl-util)

(defvar *use-blas* 10000
  "Use BLAS routines if available. If it is NIL then BLAS is never
used. If it is a real number then BLAS is only used when the problem
size exceeds that number. In all other cases BLAS is used whenever
possible.")

(let* ((a (make-array 4 :element-type 'double-float))
       (b (make-array 2 :element-type 'double-float
                      :displaced-to a :displaced-index-offset 1)))
  (defun blas-supports-displaced-arrays-p ()
    (if (find-package 'blas)
        (handler-case
            (progn
              (replace a '(0d0 1d0 2d0 3d0))
              (funcall (intern #.(symbol-name 'dscal) (find-package 'blas))
                       2 2d0 b 1)
              (assert (= (aref a 0) 0d0))
              (assert (= (aref a 1) 2d0))
              (assert (= (aref a 2) 4d0))
              (assert (= (aref a 3) 3d0))
              t)
          (error (c)
            (declare (ignore c))
            nil))
        nil)))

(eval-when (:load-toplevel :execute)
  (when *use-blas*
    (cond ((not (find-package 'blas))
           (warn "~S is ~S but there is no BLAS package. ~
                  It may be loaded any time to speed up things."
                 '*use-blas* *use-blas*))
          ((not (blas-supports-displaced-arrays-p))
           (warn "Blas does not support displaced arrays. ~
                  Setting *USE-BLAS* to NIL. See README.")
           (setq *use-blas* nil)))))

(defun use-blas-p (problem-size)
  (let ((x *use-blas*))
    (and x
         (or (not (realp x))
             (< x problem-size))
         (find-package 'blas))))


;;;; Macrology

(defmacro with-gensyms (vars &body body)
  `(let ,(mapcar #'(lambda (v) `(,v (gensym ,(symbol-name v))))
                 vars)
    ,@body))

(defun split-body (body)
  "Return a list of declarations and the rest of BODY."
  (let ((pos (position-if-not (lambda (form)
                                (and (listp form)
                                     (eq (first form) 'declare)))
                              body)))
    (if pos
        (values (subseq body 0 pos)
                (subseq body pos))
        (values body nil))))

(defun suffix-symbol (symbol &rest suffixes)
  (intern (format nil "~A~{~A~}" (symbol-name symbol)
                  (mapcar #'string suffixes))
          (symbol-package symbol)))

(defmacro special-case (test &body body)
  "Let the compiler compile BODY for the case when TEST is true and
also when it's false. The purpose is to allow different constraints to
propagate to the two branches allowing more them to be more
optimized."
  `(if ,test
       (progn ,@body)
       (progn ,@body)))


;;;; Misc

(eval-when (:compile-toplevel :load-toplevel)
  (deftype flt () 'double-float)
  (deftype flt-vector () '(simple-array flt (*)))
  (declaim (inline flt))
  (defun flt (x)
    (coerce x 'flt))
  (deftype index () '(integer 0 #.(1- array-total-size-limit)))
  (deftype index-vector () '(simple-array index (*)))
  (defparameter *no-array-bounds-check*
    #+sbcl '(sb-c::insert-array-bounds-checks 0)
    #-sbcl '())
  (defparameter *the*
    #+sbcl 'sb-ext:truly-the
    #-sbcl 'the))

(defun make-flt-array (dimensions)
  (make-array dimensions :element-type 'flt :initial-element #.(flt 0)))

(defun gaussian-random-1 ()
  "Return a single float of zero mean and unit variance."
  (loop
   (let* ((x1 (1- (* 2.0 (random 1.0))))
          (x2 (1- (* 2.0 (random 1.0))))
          (w (+ (* x1 x1) (* x2 x2))))
     (declare (type single-float x1 x2)
              (type (single-float 0.0) w)
              (optimize (speed 3)))
     (when (< w 1.0)
       ;; Now we have two random numbers but return only one.
       (return
         (* x2
            (#.*the* single-float (sqrt (/ (* -2.0 (log w)) w)))))))))

(defun select-random-element (seq)
  (elt seq (random (length seq))))

(defun split-plist (list keys)
  (let ((known ())
        (unknown ()))
    (loop for (key value) on list by #'cddr
          do (cond ((find key keys)
                    (push key known)
                    (push value known))
                   (t
                    (push key unknown)
                    (push value unknown))))
    (values (reverse known) (reverse unknown))))

(defun last1 (seq)
  (if (listp seq)
      (first (last seq))
      (aref seq (1- (length seq)))))

(defmacro push-all (list place)
  (with-gensyms (e)
    `(dolist (,e ,list)
       (push ,e ,place))))

(defmacro repeatedly (&body body)
  "Like CONSTANTLY but evaluates BODY it for each time."
  (with-gensyms (args)
    `(lambda (&rest ,args)
       (declare (ignore ,args))
       ,@body)))

(declaim (inline sigmoid))
(defun sigmoid (x)
  (/ (1+ (exp (- x)))))

(declaim (inline try-chance))
(defun try-chance (chance)
  (< (random #.(flt 1)) (flt chance)))

(declaim (inline binarize-randomly))
(defun binarize-randomly (x)
  "Return 1 with X probability and 0 otherwise."
  (if (try-chance x)
      #.(flt 1)
      #.(flt 0)))

#+nil
(defun random-index (seq)
  (let ((x (random #.(mgl-util:flt 1))))
    (do* ((i 0 (1+ i))
          (sum #.(mgl-util:flt 0) (+ sum (aref seq i))))
         ((or (<= x sum) (= i (1- (length seq))))
          i))))


;;;; float vector I/O

(deftype single-float-vector () '(simple-array single-float (*)))
(deftype double-float-vector () '(simple-array double-float (*)))

#+sbcl
(progn

(defun sync->fd (fd-stream)
  (force-output fd-stream)
  (let ((fd (sb-impl::fd-stream-fd fd-stream)))
    (sb-unix:unix-lseek fd (file-position fd-stream) sb-unix:l_set)))

(defun sync<-fd (fd-stream)
  (let ((fd (sb-impl::fd-stream-fd fd-stream)))
    (file-position fd-stream
                   (sb-unix:unix-lseek fd 0 sb-unix:l_incr))))

(defun write-single-float-vector (array fd-stream)
  (declare (type single-float-vector array))
  (sync->fd fd-stream)
  (let ((fd (sb-impl::fd-stream-fd fd-stream)))
    (sb-unix:unix-write fd
                        (sb-sys:vector-sap array)
                        0
                        (* 4 (length array))))
  (sync<-fd fd-stream))

(defun read-single-float-vector (array fd-stream)
  (declare (type single-float-vector array))
  (sync->fd fd-stream)
  (let* ((l (* 4 (length array)))
         (l2 (sb-unix:unix-read (sb-impl::fd-stream-fd fd-stream)
                                (sb-sys:vector-sap array)
                                l)))
    (sync<-fd fd-stream)
    (unless (= l l2)
      (error "Read only ~S bytes out of ~S~%" l2 l))))

(defun write-double-float-vector (array fd-stream)
  (declare (type double-float-vector array))
  (sync->fd fd-stream)
  (let ((fd (sb-impl::fd-stream-fd fd-stream)))
    (sb-unix:unix-write fd
                        (sb-sys:vector-sap array)
                        0
                        (* 8 (length array))))
  (sync<-fd fd-stream))

(defun read-double-float-vector (array fd-stream)
  (declare (type double-float-vector array))
  (sync->fd fd-stream)
  (let* ((l (* 8 (length array)))
         (l2 (sb-unix:unix-read (sb-impl::fd-stream-fd fd-stream)
                                (sb-sys:vector-sap array)
                                l)))
    (sync<-fd fd-stream)
    (unless (= l l2)
      (error "Read only ~S bytes out of ~S~%" l2 l))))
)

#+allegro
(progn

(defun write-single-float-vector (array stream)
  (declare (type single-float-vector array)
           (type excl:simple-stream stream))
  (excl:write-vector array stream))

(defun read-single-float-vector (array stream)
  (declare (type single-float-vector array)
           (type excl:simple-stream stream))
  (let* ((l (* 4 (length array)))
         (l2 (excl:read-vector array stream)))
    (unless (= l l2)
      (error "Read only ~S bytes out of ~S~%" l2 l))))

(defun write-double-float-vector (array stream)
  (declare (type double-float-vector array)
           (type excl:simple-stream stream))
  (excl:write-vector array stream))

(defun read-double-float-vector (array stream)
  (declare (type double-float-vector array)
           (type excl:simple-stream stream))
  (let* ((l (* 8 (length array)))
         (l2 (excl:read-vector array stream)))
    (unless (= l l2)
      (error "Read only ~S bytes out of ~S~%" l2 l))))

)


;;;; Weight I/O

(defgeneric write-weights (object stream)
  (:documentation "Write the weights of OBJECT to STREAM."))

(defgeneric read-weights (object stream)
  (:documentation "Read the weights of OBJECT from STREAM."))
