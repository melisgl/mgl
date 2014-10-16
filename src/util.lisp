(in-package :mgl-util)



;;;; Macrology

(defmacro special-case (test &body body)
  "Let the compiler compile BODY for the case when TEST is true and
  also when it's false. The purpose is to allow different constraints
  to propagate to the two branches allowing them to be more
  optimized."
  `(if ,test
       (progn ,@body)
       (progn ,@body)))

(defmacro apply-key (key object)
  (alexandria:once-only (key object)
    `(if ,key (funcall ,key ,object) ,object)))


;;;; Types

#+nil
(eval-when (:compile-toplevel :load-toplevel)
  (deftype flt () 'single-float)
  (defconstant flt-ctype :float)
  (deftype positive-flt () '(single-float #.least-positive-single-float))
  (defconstant most-negative-flt most-negative-single-float)
  (defconstant least-negative-flt least-negative-single-float)
  (defconstant least-positive-flt least-positive-single-float)
  (defconstant most-positive-flt most-positive-single-float))

(eval-when (:compile-toplevel :load-toplevel)
  (deftype flt () 'double-float)
  (defconstant flt-ctype :double)
  (deftype positive-flt () '(double-float #.least-positive-double-float))
  (defconstant most-negative-flt most-negative-double-float)
  (defconstant least-negative-flt least-negative-double-float)
  (defconstant least-positive-flt least-positive-double-float)
  (defconstant most-positive-flt most-positive-double-float))

(eval-when (:compile-toplevel :load-toplevel)
  (deftype flt-vector () '(simple-array flt (*)))
  (declaim (inline flt))
  (defun flt (x)
    (coerce x 'flt))
  (deftype index () '(integer 0 #.(1- array-total-size-limit)))
  (deftype index-vector () '(simple-array index (*))))

(defun flt-vector (&rest args)
  (make-array (length args) :element-type 'flt :initial-contents args))

(defparameter *no-array-bounds-check*
  #+sbcl '(sb-c::insert-array-bounds-checks 0)
  ;; (SAFETY 0) is too coarse, avoid warnings by using the
  ;; relatively uncontroversial (SPEED 3) instead of ().
  #-sbcl '(speed 3))

;;; A version of THE that's trusted by the compiler.
(defmacro the! (&rest args)
  `(#+sbcl sb-ext:truly-the
    #+cmu ext:truly-the
    #-(or sbcl cmu) the
    ,@args))

;;; Beat Allegro's underflow errors into submission with a club. The
;;; values must be known to be FLT for this to work.
#+allegro
(defmacro with-zero-on-underflow ((prototype) &body body)
  (alexandria:with-gensyms (trap-underflow)
    `(catch ',trap-underflow
       (handler-bind ((floating-point-underflow
                        #'(lambda (c)
                            (declare (ignore c))
                            (throw ',trap-underflow (float 0 ,prototype)))))
         ,@body))))

#-allegro
(defmacro with-zero-on-underflow ((prototype) &body body)
  (declare (ignore prototype))
  `(locally ,@body))


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
  (alexandria:with-gensyms (e)
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

(defun hash-table->vector (hash-table)
  (let ((v (make-array (hash-table-count hash-table)))
        (i 0))
    (maphash (lambda (key value)
               (setf (aref v i) (cons key value))
               (incf i))
             hash-table)
    v))

(defmacro repeatedly (&body body)
  "Like CONSTANTLY but evaluates BODY it for each time."
  (alexandria:with-gensyms (args)
    `(lambda (&rest ,args)
       (declare (ignore ,args))
       ,@body)))

(defun make-sequence-generator (seq)
  "Return a function that returns elements of SEQ in order without
  end. When there are no more elements, start over."
  (let* ((vector (copy-seq (coerce seq 'vector)))
         (l (length vector))
         (n 0))
    (lambda ()
      (prog1
          (aref vector n)
        (setf n (mod (1+ n) l))))))

(defun make-random-generator (seq &key (reorder #'mgl-resample:shuffle))
  "Return a function that returns elements of VECTOR in random order
  without end. When there are no more elements, start over with a
  different random order."
  (let* ((vector (funcall reorder (copy-seq (coerce seq 'vector))))
         (l (length vector))
         (n 0))
    (lambda ()
      (when (zerop n)
        (setq vector (funcall reorder vector)))
      (prog1
          (aref vector n)
        (setf n (mod (1+ n) l))))))

(defun make-n-gram-mappee (function n)
  "Make a function of a single argument that's suitable for the
  function arguments to a mapper function. It calls FUNCTION with
  every N element."
  (let ((previous-values '()))
    (lambda (x)
      (push x previous-values)
      (when (< n (length previous-values))
        (setf previous-values (subseq previous-values 0 n)))
      (when (= n (length previous-values))
        (funcall function (reverse previous-values))))))

(defun applies-to-p (generic-function &rest args)
  (find nil (compute-applicable-methods generic-function args)
        :key #'swank-mop:method-qualifiers))


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

(declaim (inline sign))
(defun sign (x)
  (declare (type flt x))
  (cond ((plusp x) #.(flt 1))
        ((minusp x) #.(flt -1))
        (t #.(flt 0))))

(declaim (inline sech))
(defun sech (x)
  (declare (type flt x))
  (/ (cosh x)))

(declaim (inline sigmoid))
(defun sigmoid (x)
  (declare (type flt x))
  (/ (1+ (with-zero-on-underflow (x) (exp (- x))))))

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
  (flt (mgl-mat::gaussian-random-1)))

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
  Coincidence\" by Ted Dunning
  (http://citeseer.ist.psu.edu/29096.html).

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
  Coincidence\" by Ted Dunning
  \(http://citeseer.ist.psu.edu/29096.html).

  K1 is the number of outcomes in each class. K2 is the same in a
  possibly different process.

  All elements in K1 and K2 are positive integers. To ensure this -
  and also as kind of prior - add a small number such as 1 each
  element in K1 and K2 before calling."
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


;;;; Running mean and variance.
;;;;
;;;; See Knuth TAOCP vol 2, 3rd edition, page 232.

(defclass running-stat ()
  ((n :initform 0)
   (mean :initform 0)
   (m2 :initform 0)))

(defun clear-running-stat (stat)
  (with-slots (n mean m2) stat
    (setf n 0
          mean 0
          m2 0)))

(defun add-to-running-stat (x stat &key (weight 1))
  (with-slots (n mean m2) stat
    (incf n weight)
    (let ((delta (* weight (- x mean))))
      (incf mean (/ delta n))
      (incf m2 (* delta (- x mean))))))

(defun running-stat-variance (stat)
  (with-slots (n mean m2) stat
    (if (<= n 1)
        0
        (/ m2 (1- n)))))

(defun running-stat-mean (stat)
  (slot-value stat 'mean))

(defmethod print-object ((stat running-stat) stream)
  (pprint-logical-block (stream ())
    (print-unreadable-object (stat stream :type t)
      (format stream ":mean ~,5F :variance ~,5F"
              (running-stat-mean stat)
              (running-stat-variance stat))))
  stat)


;;;; Array utilities

(defun as-column-vector (a)
  (aops:reshape a (list (array-total-size a) 1)))

(defun rows-to-arrays (mat)
  (let ((arrays ()))
    (map-displacements (lambda (mat)
                         (push (mat-to-array mat) arrays))
                       mat (mat-dimension mat 1))
    (nreverse arrays)))

(defun max-row-positions (mat)
  "Find the colums with the maximum in each row of the 2d MAT and
  return them as a list."
  (let ((displacement (mat-displacement mat))
        (n-rows (mat-dimension mat 0))
        (n-columns (mat-dimension mat 1)))
    (with-facets ((m (mat 'backing-array :direction :input)))
      (loop for row below n-rows
            collect (let ((start (+ displacement (* row n-columns))))
                      (- (max-position m start (+ start n-columns))
                         start))))))


;;;; Printing

(defun print-table (list &key (stream t) (empty-value nil empty-value-p)
                    (repeat-marker nil repeat-marker-p) (compactp t)
                    (new-line-prefix ""))
  (unless (endp list)
    (flet ((convert (x)
             (if (and empty-value-p (eq x empty-value))
                 ""
                 (princ-to-string x))))
      (let* ((n-columns (length (first list)))
             (column-widths (loop for column below n-columns
                                  collect
                                  (loop for row in list
                                        maximizing
                                        (if (eq row :horizontal-break)
                                            0
                                            (length
                                             (convert (elt row column)))))))
             (previous-row nil))
        (loop
          for row-index upfrom 0
          for row in list
          do (unless (zerop row-index)
               (format stream "~A" new-line-prefix))
             (cond ((eq row :horizontal-break)
                    (loop for i below n-columns
                          for width in column-widths
                          do (loop repeat width do (format stream "-"))
                             (when (< (1+ i) n-columns)
                               (format stream (if compactp "+" "-+-")))))
                   (t
                    (loop for i below n-columns
                          for column in row
                          for width in column-widths
                          do (let* ((s (convert column))
                                    (s (if (and repeat-marker-p
                                                (not (eq previous-row
                                                         :horizontal-break))
                                                (< i (length previous-row))
                                                (string= (convert
                                                          (elt previous-row i))
                                                         s))
                                           repeat-marker
                                           s)))
                               (loop repeat (- width (length s))
                                     do (format stream " "))
                               (format stream "~A" s)
                               (when (< (1+ i) n-columns)
                                 (format stream (if compactp "|" " | ")))))))
             (terpri stream)
             (setq previous-row row))))))


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


;;;; Experiments

(defvar *experiment-random-seed* 1234)

(defun call-repeatably (fn &key (seed *experiment-random-seed*))
  (with-cuda* (:random-seed seed)
    (let ((*random-state*
            #+sbcl (sb-ext:seed-random-state seed)
            #+allegro (make-random-state t seed)
            #-(or sbcl allegro) *random-state))
      (funcall fn))))

(defmacro repeatably ((&key (seed *experiment-random-seed*)) &body body)
  `(call-repeatably (lambda () ,@body) :seed ,seed))
