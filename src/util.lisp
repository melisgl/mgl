;;;; BLAS support

(in-package :mgl-util)

(defvar *use-blas* 10000
  "Use BLAS routines if available. If it is NIL then BLAS is never
used \(not quite true as some code does not care about this setting).
If it is a real number then BLAS is only used when the problem size
exceeds that number. In all other cases BLAS is used whenever
possible.")

(defun cost-of-copy (mat)
  (matlisp:number-of-elements mat))

(defun cost-of-fill (mat)
  (matlisp:number-of-elements mat))

(defun cost-of-gemm (a b job)
  (* (if (member job '(:nt :nn))
         (matlisp:ncols a)
         (matlisp:nrows a))
     (matlisp:number-of-elements b)))

(defun use-blas-p (cost)
  (let ((x *use-blas*))
    (and x
         (or (not (realp x))
             (< x cost))
         (find-package 'blas))))

(declaim (inline storage))
(defun storage (matlisp-matrix)
  (the flt-vector (values (matlisp::store matlisp-matrix))))

(defgeneric reshape2 (mat m n)
  (:method ((mat matlisp:real-matrix) m n)
    (assert (<= (* m n) (length (storage mat))))
    (make-instance 'matlisp:real-matrix
                   :nrows m :ncols n :store (storage mat))))

(defgeneric set-ncols (mat ncols)
  (:method ((mat matlisp:real-matrix) ncols)
    (assert (<= 0 ncols (/ (length (storage mat))
                           (matlisp:nrows mat))))
    (setf (matlisp:ncols mat) ncols)
    (setf (matlisp:number-of-elements mat) (* ncols (matlisp:nrows mat)))))



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
  (deftype positive-flt () '(double-float #.least-positive-double-float))
  (deftype flt-vector () '(simple-array flt (*)))
  (declaim (inline flt))
  (defun flt (x)
    (coerce x 'flt))
  (deftype index () '(integer 0 #.(1- array-total-size-limit)))
  (deftype index-vector () '(simple-array index (*)))
  (defparameter *no-array-bounds-check*
    #+sbcl '(sb-c::insert-array-bounds-checks 0)
    #-sbcl '()))

(defmacro the! (&rest args)
  `(#+sbcl sb-ext:truly-the
    #+cmu ext:truly-the
    #-(or sbcl cmu ) the
    ,@args))

(defun make-flt-array (dimensions)
  (make-array dimensions :element-type 'flt :initial-element #.(flt 0)))

(defun gaussian-random-1 ()
  "Return a single float of zero mean and unit variance."
  (loop
   (let* ((x1 (1- (* #.(flt 2) (random #.(flt 1)))))
          (x2 (1- (* #.(flt 2) (random #.(flt 1)))))
          (w (+ (* x1 x1) (* x2 x2))))
     (declare (type flt x1 x2)
              (type (double-float 0d0) w)
              (optimize (speed 3)))
     (when (< w 1.0)
       ;; Now we have two random numbers but return only one. The
       ;; other would be X1 times the same.
       (return
         (* x2
            (the! double-float (sqrt (/ (* -2.0 (log w)) w)))))))))

(defun select-random-element (seq)
  (elt seq (random (length seq))))

(defmacro while (test &body body)
  `(loop while ,test do (progn ,@body)))

(defun last1 (seq)
  (if (listp seq)
      (first (last seq))
      (aref seq (1- (length seq)))))

(defun append1 (list obj)
  (append list (list obj)))

(defmacro push-all (list place)
  (with-gensyms (e)
    `(dolist (,e ,list)
       (push ,e ,place))))

(defun group (seq n)
  (let ((l (length seq)))
    (loop for i below l by n
          collect (subseq seq i (min l (+ i n))))))

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


;;;; Stripes

(defgeneric max-n-stripes (learner)
  (:documentation "The number of examples with which the learner is
capable of dealing simultaneously."))

(defgeneric set-max-n-stripes (max-n-stripes object)
  (:documentation "Allocate the necessary stuff to allow for N-STRIPES
number of examples to be worked with simultaneously."))

(defsetf max-n-stripes (object) (store)
  `(set-max-n-stripes ,store ,object))

(defgeneric n-stripes (learner)
  (:documentation "The number of examples with which the learner is
currently dealing."))

(defgeneric set-n-stripes (n-stripes object)
  (:documentation "Set the number of stripes \(out of MAX-N-STRIPES)
that are in use."))

(defsetf n-stripes (object) (store)
  `(set-n-stripes ,store ,object))

(defgeneric stripe-start (stripe obj))
(defgeneric stripe-end (stripe obj))

(eval-when (:compile-toplevel :load-toplevel :execute)
  (defun stripe-binding (stripe obj start &optional end)
    (with-gensyms (%stripe %obj)
      `((,%stripe ,stripe)
        (,%obj ,obj)
        (,start (the index (stripe-start ,%stripe ,%obj)))
        ,@(when end `((,end (the index (stripe-end ,%stripe ,%obj)))))))))

(defmacro with-stripes (specs &body body)
  `(let* ,(mapcan (lambda (spec) (apply #'stripe-binding spec))
                  specs)
     ,@body))


;;;; Various accessor type generic functions share by packages.

(defgeneric name (object))
(defgeneric size (object))
(defgeneric nodes (object))
(defgeneric default-value (object))
(defgeneric group-size (object))
(defgeneric batch-size (object))


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
