(in-package :mgl-util)

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
propagate to the two branches allowing them to be more optimized."
  `(if ,test
       (progn ,@body)
       (progn ,@body)))


;;;; Types

(eval-when (:compile-toplevel :load-toplevel)
  (deftype flt () 'double-float)
  (deftype positive-flt () '(double-float #.least-positive-double-float))
  (defconstant least-negative-flt least-negative-double-float)
  (defconstant most-postitive-flt most-positive-double-float)
  (deftype flt-vector () '(simple-array flt (*)))
  (declaim (inline flt))
  (defun flt (x)
    (coerce x 'flt))
  (deftype index () '(integer 0 #.(1- array-total-size-limit)))
  (deftype index-vector () '(simple-array index (*))))

(defun make-flt-array (dimensions)
  (make-array dimensions :element-type 'flt :initial-element #.(flt 0)))

(defun flt-vector (&rest args)
  (make-array (length args) :element-type 'flt :initial-contents args))


;;;; Declarations

(eval-when (:compile-toplevel :load-toplevel)
  (defparameter *no-array-bounds-check*
    #+sbcl '(sb-c::insert-array-bounds-checks 0)
    ;; (SAFETY 0) is too coarse, avoid warnings by using the
    ;; relatively uncontroversial (SPEED 3) instead of ().
    #-sbcl '(speed 3)))

(defmacro the! (&rest args)
  `(#+sbcl sb-ext:truly-the
    #+cmu ext:truly-the
    #-(or sbcl cmu) the
    ,@args))


;;;; Pathnames

(defparameter *mgl-dir*
  (make-pathname :name nil :type nil
                 :defaults (asdf:component-pathname (asdf:find-system :mgl))))

(defun asdf-system-relative-pathname (pathname)
  (namestring (merge-pathnames pathname *mgl-dir*)))


;;;; Misc

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

(defun subseq* (sequence start &optional end)
  (setq start (max 0 start))
  (when end
    (setq end (min (length sequence) end)))
  (subseq sequence start end))

(defun max-position (seq start end)
  (position (loop for i upfrom start below end maximizing (elt seq i))
            seq :start start :end end))

(defun hash-table->alist (hash-table)
  (let ((r ()))
    (maphash (lambda (key value)
               (push (cons key value) r))
             hash-table)
    r))

(defun alist->hash-table (alist &rest args)
  (let ((h (apply #'make-hash-table args)))
    (loop for (key . value) in alist
          do (setf (gethash key h) value))
    h))

(defun hash-table->vector (hash-table)
  (let ((v (make-array (hash-table-count hash-table)))
        (i 0))
    (maphash (lambda (key value)
               (setf (aref v i) (cons key value))
               (incf i))
             hash-table)
    v))

(defun reverse-hash-table (hash-table &key (test #'eql))
  "Return a hash table that maps from the values of HASH-TABLE back to
its keys. HASH-TABLE had better be a bijection."
  (let ((r (make-hash-table :test test)))
    (maphash (lambda (key value)
               (setf (gethash value r) key))
             hash-table)
    r))

(defmacro repeatedly (&body body)
  "Like CONSTANTLY but evaluates BODY it for each time."
  (with-gensyms (args)
    `(lambda (&rest ,args)
       (declare (ignore ,args))
       ,@body)))

(defun nshuffle-vector (vector)
  "Shuffle a vector in place using Fisher-Yates algorithm."
  (loop for idx downfrom (1- (length vector)) to 1
        for other = (random (1+ idx))
        do (unless (= idx other)
             (rotatef (aref vector idx) (aref vector other))))
  vector)

(defun shuffle-vector (vector)
  (nshuffle-vector (copy-seq vector)))

(defun make-seq-generator (vector)
  "Return a function that returns elements of VECTOR in order without
end. When there are no more elements, start over."
  (let ((vector (copy-seq (coerce vector 'vector)))
        (l (length vector))
        (n 0))
    (lambda ()
      (prog1
          (aref vector n)
        (setf n (mod (1+ n) l))))))

(defun make-random-generator (seq)
  "Return a function that returns elements of VECTOR in random order
without end. When there are no more elements, start over with a
different random order."
  (let* ((vector (copy-seq (coerce seq 'vector)))
         (l (length vector))
         (n 0))
    (lambda ()
      (when (zerop n)
        (setq vector (nshuffle-vector vector)))
      (prog1
          (aref vector n)
        (setf n (mod (1+ n) l))))))

(defun make-n-gram-mappee (function n)
  "Make a function of a single argument that's suitable for the
function arguments to a mapper function. It calls FUNCTION with every
N element."
  (let ((previous-values '()))
    (lambda (x)
      (push x previous-values)
      (when (< n (length previous-values))
        (setf previous-values (subseq previous-values 0 n)))
      (when (= n (length previous-values))
        (funcall function (reverse previous-values))))))

(defun break-seq (fractions seq)
  "Split SEQ into a number of subsequences. FRACTIONS is either a
positive integer or a list of non-negative real numbers. If FRACTIONS
is a positive integer then return a list of that many subsequences of
equal size \(bar rounding errors), else split SEQ into subsequences,
where the length of subsequence I is proportional to element I of
FRACTIONS:

  (BREAK-SEQ '(2 3) '(0 1 2 3 4 5 6 7 8 9))
    => ((0 1 2 3) (4 5 6 7 8 9))"
  (let ((length (length seq)))
    (if (numberp fractions)
        (let ((fraction-size (/ length fractions)))
          (loop for fraction below fractions
                collect (subseq seq
                                (floor (* fraction fraction-size))
                                (floor (* (1+ fraction) fraction-size)))))
        (let ((sum-fractions (loop for x in fractions sum x))
              (n-fractions (length fractions)))
          (loop with sum = 0
                for fraction in fractions
                for i upfrom 0
                collect (subseq seq
                                (floor (* (/ sum sum-fractions)
                                          length))
                                ;; We want to partition SEQ: elements
                                ;; must not be lost or duplicated. Use
                                ;; INCF, because float precision in an
                                ;; expression and in a variable may be
                                ;; different.
                                (if (= i (1- n-fractions))
                                    length
                                    (floor (* (/ (incf sum fraction)
                                                 sum-fractions)
                                              length)))))))))

(defun collect-distinct (seq &key (key #'identity) (test #'eql))
  (let ((result ()))
    (map nil
         (lambda (x)
           (pushnew (funcall key x) result :test test))
         seq)
    (nreverse result)))

(defun stratified-split (fractions seq &key (key #'identity) (test #'eql)
                                         randomizep)
  "Similar to BREAK-SEQ, but also makes sure that keys are equally
distributed among the paritions. It can be useful for classification
tasks to partition the data set while keeping the distribution of
classes the same."
  (let ((keys (collect-distinct seq :key key :test test)))
    (if (zerop (length keys))
        ()
        (let ((per-key-splits
                (loop for k in keys
                      collect
                      (let ((elements
                              (coerce
                               (remove-if-not (lambda (x)
                                                (funcall test k
                                                         (funcall key x)))
                                              seq)
                               'vector)))
                        (break-seq fractions
                                   (if randomizep
                                       (nshuffle-vector elements)
                                       elements))))))
          (loop for i below (length (elt per-key-splits 0))
                collect (apply #'concatenate
                               (if (listp seq)
                                   'list
                                   `(vector ,(array-element-type seq)))
                               (mapcar (lambda (splits)
                                         (elt splits i))
                                       per-key-splits)))))))


;;;; Periodic functions

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


;;;; Math

;;; Beat Allegro's underflow errors into submission with a club. The
;;; values must be known to be FLT for this to work.
#+allegro
(defmacro with-zero-on-underflow (&body body)
  `(locally (declare (optimize (safety 0) (speed 3)))
     ,@body))

#-allegro
(defmacro with-zero-on-underflow (&body body)
  `(locally ,@body))

(declaim (inline sech))
(defun sech (x)
  (declare (type flt x))
  (/ (cosh x)))

(declaim (inline sigmoid))
(defun sigmoid (x)
  (declare (type flt x))
  (/ (1+ (with-zero-on-underflow (exp (- x))))))

;;; From Yann Lecun's Efficient backprop.
(declaim (inline scaled-tanh))
(defun scaled-tanh (x)
  (declare (type flt x))
  (* #.(flt 1.7159) (tanh (* #.(flt 2/3) x))))

(declaim (inline try-chance))
(defun try-chance (chance)
  (< (random #.(flt 1)) (flt chance)))

(declaim (inline binarize-randomly))
(defun binarize-randomly (x)
  "Return 1 with X probability and 0 otherwise."
  (if (try-chance x)
      #.(flt 1)
      #.(flt 0)))

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

;; Knuth's slow poisson sampler.
(defun poisson-random (mean)
  (let ((l (exp (- mean)))
        (k 1)
        (p (random #.(flt 1))))
    (while (<= l p)
      (incf k)
      (setq p (* p (random #.(flt 1)))))
    (1- k)))

(defun select-random-element (seq)
  (elt seq (random (length seq))))

(defun binomial-log-likelihood-ratio (k1 n1 k2 n2)
  "See \"Accurate Methods for the Statistics of Surprise and
Coincidence\" by Ted Dunning \(http://citeseer.ist.psu.edu/29096.html).

All classes must have non-zero counts, that is, K1, N1-K1, K2, N2-K2
are positive integers. To ensure this - and also as kind of prior -
add a small number such as 1 to K1, K2 and 2 to N1, N2 before
calling."
  (flet ((log-l (p k n)
           (+ (* k (log p))
              (* (- n k) (log (- 1 p))))))
    (let ((p1 (/ k1 n1))
          (p2 (/ k2 n2))
          (p (/ (+ k1 k2) (+ n1 n2))))
      (* 2
         (+ (- (log-l p k1 n1))
            (- (log-l p k2 n2))
            (log-l p1 k1 n1)
            (log-l p2 k2 n2))))))

(defun multinomial-log-likelihood-ratio (k1 k2)
  "See \"Accurate Methods for the Statistics of Surprise and
Coincidence\" by Ted Dunning \(http://citeseer.ist.psu.edu/29096.html).

K1 is the number of outcomes in each class. K2 is the same in a
possibly different process.

All elements in K1 and K2 are positive integers. To ensure this - and
also as kind of prior - add a small number such as 1 each element in
K1 and K2 before calling."
  (flet ((log-l (p k)
           (let ((sum 0))
             (map nil
                  (lambda (p-i k-i)
                    (incf sum (* k-i (log p-i))))
                  p k)
             sum))
         (normalize (k)
           (let ((sum (loop for k-i across k sum k-i)))
             (map 'vector
                  (lambda (x)
                    (/ x sum))
                  k)))
         (sum (x y)
           (map 'vector #'+ x y)))
    (let ((p1 (normalize k1))
          (p2 (normalize k2))
          (p (normalize (sum k1 k2))))
      (* 2
         (+ (- (log-l p k1))
            (- (log-l p k2))
            (log-l p1 k1)
            (log-l p2 k2))))))


;;;; BLAS support

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
         (matlisp:nrows a)
         (matlisp:ncols a))
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
    (if (and (= (matlisp:nrows mat) m)
             (= (matlisp:ncols mat) n))
        mat
        (make-instance 'matlisp:real-matrix
                       :nrows m :ncols n :store (storage mat)))))

(defgeneric set-ncols (mat ncols)
  (:method ((mat matlisp:real-matrix) ncols)
    (assert (<= 0 ncols (/ (length (storage mat))
                           (matlisp:nrows mat))))
    (setf (matlisp:ncols mat) ncols)
    (setf (matlisp:number-of-elements mat) (* ncols (matlisp:nrows mat)))))

(defgeneric sum-elements (mat))

(defmethod sum-elements ((a matlisp:real-matrix))
  (let ((sum 0d0)
        (store-a (storage a)))
    (declare (type double-float sum))
    (loop for i of-type fixnum below (matlisp:number-of-elements a)
          do (incf sum (aref store-a i)))
    sum))


;;;; Float vector I/O

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


;;;; Printing

(defun print-table (list &key (stream t))
  (unless (endp list)
    (format stream "~&")
    (let* ((n-columns (length (first list)))
           (column-widths (loop for column below n-columns
                                collect
                                (loop for row in list
                                      maximizing
                                      (length
                                       (princ-to-string (elt row column)))))))
      (loop for row in list
            do (loop for i below n-columns
                     for column in row
                     for width in column-widths
                     do (let ((s (princ-to-string column)))
                          (loop repeat (- width (length s))
                                do (format stream " "))
                          (format stream "~A" s)
                          (when (< (1+ i) n-columns)
                            (format stream " | "))))
            (terpri stream)))))


;;;; DESCRIBE customization

(defmacro with-safe-printing (&body body)
  `(multiple-value-bind (v e)
       (ignore-errors (progn ,@body))
     (if e
         "#<error printing object>"
         v)))

(defun format-description (description stream)
  (pprint-newline :mandatory stream)
  (destructuring-bind (name value &optional (format "~S"))
      description
    (format stream "~A = ~? " name format (list value))))

(defun pprint-descriptions (class descriptions stream)
  (pprint-newline :mandatory stream)
  (pprint-indent :block 2 stream)
  (pprint-logical-block (stream ())
    (format stream "~A description:" class)
    (pprint-indent :block 2 stream)
    (map nil (lambda (description)
               (format-description description stream))
         descriptions))
  (pprint-indent :block 0 stream)
  (pprint-newline :mandatory stream))

(defun ->description (object description)
  (if (symbolp description)
      `(list ',description
        (with-safe-printing (,description ,object)))
      `(list ',(first description)
        (with-safe-printing ,(second description))
        ,@(cddr description))))

(defmacro define-descriptions ((object class &key inheritp)
                               &body descriptions)
  (let ((%stream (gensym)))
    `(defmethod describe-object ((,object ,class) ,%stream)
       (pprint-logical-block (,%stream ())
         (if (and (next-method-p) ,inheritp)
             (call-next-method)
             (print-unreadable-object (,object ,%stream :type t :identity t)))
         (pprint-descriptions ',class
                              (list ,@(mapcar (lambda (description)
                                                (->description object
                                                               description))
                                              descriptions))
                              ,%stream)))))
