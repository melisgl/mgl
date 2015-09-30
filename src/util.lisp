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

(defun group (seq group-size &key (start 0) (end (length seq)))
  (if (= start end)
      ()
      (loop for i upfrom start below end by group-size
            collect (subseq seq i (min end (+ i group-size))))))

(defun sort-groups! (seq pred group-size &key key (start 0) (end (length seq)))
  (let ((v (map 'vector (lambda (e) (cons e (apply-key key e))) seq)))
    (loop for i upfrom start below end by group-size
          do (let* ((length (min group-size (- end i)))
                    (a (make-array length :displaced-to v
                                   :displaced-index-offset i)))
               (replace seq (sort a pred :key #'cdr) :start1 i)))
    (map (if (listp seq) 'list 'vector) #'car v)))

(defun shuffle-groups (seq group-size &key (start 0) (end (length seq)))
  (apply #'concatenate (if (listp seq) 'list 'vector)
         `(,(subseq seq 0 start)
           ,@(mgl-resample:shuffle!
              (group seq group-size :start start :end end))
           ,(subseq seq end))))

(defun subseq* (sequence start &optional end)
  (setq start (max 0 start))
  (when end
    (setq end (min (length sequence) end)))
  (subseq sequence start end))

(defun max-position (seq start end)
  (declare (type index start end))
  (cond ((typep seq '(simple-array single-float (*)))
         (let ((max most-negative-single-float)
               (pos nil))
           (declare (optimize speed)
                    (type (simple-array single-float (*)) seq))
           (loop for i of-type index upfrom start below end
                 do (let ((x (aref seq i)))
                      (when (< max x)
                        (setq max x)
                        (setq pos i))))
           pos))
        ((typep seq '(simple-array double-float (*)))
         (let ((max most-negative-double-float)
               (pos nil))
           (declare (optimize speed)
                    (type (simple-array double-float (*)) seq))
           (loop for i of-type index upfrom start below end
                 do (let ((x (aref seq i)))
                      (when (< max x)
                        (setq max x)
                        (setq pos i))))
           pos))
        (t
         (position (loop for i upfrom start below end
                         maximizing (elt seq i))
                   seq :start start :end end))))

(defun insert-into-sorted-vector
    (item vector pred &key key (max-length (array-total-size vector)))
  "Insert ITEM into VECTOR while keeping it sorted by PRED. Extend the
  vector if needed while respecting MAX-LENGTH. "
  (declare (type (array * (*)) vector)
           (type index max-length)
           (optimize speed))
  (let* ((key (if key (coerce key 'function) nil))
         (pred (coerce pred 'function))
         (len (length vector))
         (item-key (apply-key key item)))
    ;; Pick off the common case quickly where ITEM won't be collected.
    (unless (and (= len max-length)
                 (funcall pred (apply-key key (aref vector (1- len))) item-key))
      (let ((pos (1+ (or (position item-key vector
                                   :key key :test-not pred :from-end t)
                         -1))))
        (when (< pos max-length)
          (when (< len max-length)
            (vector-push-extend nil vector))
          (replace vector vector :start1 (1+ pos) :start2 pos :end2 len)
          (setf (aref vector pos) item))))
    vector))

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
  (let* ((vector (coerce seq 'vector))
         (l (length vector))
         (n 0))
    (lambda ()
      (prog1
          (aref vector n)
        (setf n (mod (1+ n) l))))))

(defun make-random-generator (seq &key (reorder #'mgl-resample:shuffle!))
  "Return a function that returns elements of VECTOR in random order
  without end. When there are no more elements, start over with a
  different random order."
  (let* ((vector (copy-seq (coerce seq 'vector)))
         (l (length vector))
         (n 0))
    (lambda ()
      (when (zerop n)
        (setq vector (funcall reorder vector)))
      (prog1
          (aref vector n)
        (setf n (mod (1+ n) l))))))

(defun make-sorted-group-generator (generator pred group-size &key key
                                    randomize-size)
  (assert (plusp group-size))
  (let ((group ()))
    (lambda ()
      (unless group
        (setq group (sort (loop repeat group-size collect (funcall generator))
                          pred :key key))
        (when randomize-size
          (setq group (shuffle-groups group randomize-size))))
      (pop group))))

(defun applies-to-p (generic-function &rest args)
  (find nil (compute-applicable-methods generic-function args)
        :key #'swank-mop:method-qualifiers))

(defun uninterned-symbol-p (object)
  (and (symbolp object)
       (null (symbol-package object))))


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

(defun half-life-to-decay (half-life)
  "b^h=0.5, b=0.5^(1/h)"
  (expt 0.5d0 (/ half-life)))

(defun half-life-to-decay-rate (half-life)
  (- 1 (expt 0.5d0 (/ half-life))))

(defun decay-to-half-life (decay)
  (log 0.5 decay))

(defun decay-rate-to-half-life (decay-rate)
  (log 0.5 (- 1 decay-rate)))

(defun cross-entropy-to-perplexity (cross-entropy)
  (exp cross-entropy))

(defun perplexity-to-cross-entropy (perplexity)
  (log perplexity))

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
  (http://citeseer.ist.psu.edu/29096.html).

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


;;;; Permutations

(defun permute (seq permutation)
  (let ((vector (coerce seq 'vector)))
    (map (if (listp seq) 'list 'vector)
         (lambda (index)
           (aref vector index))
         permutation)))

(defun invert-permutation (permutation)
  (let* ((n (length permutation))
         (p (make-array n :element-type 'fixnum)))
    (dotimes (i n)
      (setf (aref p (aref permutation i)) i))
    p))

;;; Return a vector of indices. (ELT SEQ (AREF PERMUTATION I)) is
;;; sorted by PRED.
;;;
;;;     (let ((seq '(3 1 2)))
;;;       (permute seq (sorting-permutation seq #'<)))
;;;     => (1 2 3)
(defun sorting-permutation (seq pred &key (key #'identity))
  (sort (coerce (alexandria:iota (length seq))
                'vector)
        (lambda (a b)
          (funcall pred (funcall key (elt seq a))
                   (funcall key (elt seq b))))))


;;;; Array utilities

(defun as-column-vector (a)
  (aops:reshape a (list (array-total-size a) 1)))

(defun rows-to-arrays (mat)
  (let ((arrays ()))
    (map-displacements (lambda (mat)
                         (push (mat-to-array mat) arrays))
                       mat (mat-dimension mat 1))
    (nreverse arrays)))

(defun max-row-positions (mat &key start end)
  "Find the colums with the maximum in each row of the 2d MAT and
  return them as a list."
  (let* ((displacement (mat-displacement mat))
         (n-rows (mat-dimension mat 0))
         (n-columns (mat-dimension mat 1))
         (start (or start 0))
         (end (or end n-columns)))
    (with-facets ((m (mat 'backing-array :direction :input)))
      (loop for row below n-rows
            collect (let ((row-start (+ displacement (* row n-columns))))
                      (- (max-position m (+ row-start start) (+ row-start end))
                         (+ row-start start)))))))


;;;; Classes

(defmacro defclass-now (name direct-superclasses direct-slots &rest options)
  `(eval-when (:compile-toplevel :load-toplevel :execute)
     (defclass ,name ,direct-superclasses ,direct-slots ,@options)))

(defmacro defmaker ((name &key unkeyword-args extra-keyword-args
                     (make-instance-args (gensym)))
                    &body body)
  (destructuring-bind (name maker-name)
      (if (listp name) name (list name name))
    (let ((args (append (make-instance-args (find-class name))
                        extra-keyword-args)))
      `(defun ,maker-name (,@unkeyword-args
                           &key ,@(remove-unkeyword-args args unkeyword-args))
         (let ((,make-instance-args
                 (append ,@(mapcan
                            (lambda (arg)
                              (destructuring-bind
                                  (var initform &optional indicator) arg
                                (declare (ignore initform))
                                (let ((keyword
                                        (alexandria:make-keyword var)))
                                  (cond ((find var unkeyword-args)
                                         `((list ,keyword ,var)))
                                        (indicator
                                         `((if ,indicator
                                               (list ,keyword ,var)
                                               ())))
                                        (t
                                         `((list ,keyword ,var)))))))
                            args))))
           ,(if body
                `(locally ,@body)
                `(apply #'make-instance ',name ,make-instance-args)))))))

(defun remove-unkeyword-args (args unkeyword-args)
  (assert (every (lambda (unkeyword-arg)
                   (find unkeyword-arg args :key #'first))
                 unkeyword-args))
  (remove-if (lambda (arg)
               (destructuring-bind (arg initform &optional indicator) arg
                 (cond ((find arg unkeyword-args)
                        (assert indicator ()
                                "Cannot unkeywordify argument ~S because ~
                                it has an initform: ~S."
                                arg initform)
                        t)
                       (t nil))))
             args))

;;; Adapted from SWANK::EXTRA-KEYWORDS/SLOTS, it doesn't collect every
;;; initarg like SWANK::EXTRA-KEYWORDS/MAKE-INSTANCE does.
(defun make-instance-args (class)
  (closer-mop:ensure-finalized class)
  (multiple-value-bind (slots allow-other-keys-p)
      (if (closer-mop:class-finalized-p class)
          (values (closer-mop:class-slots class) nil)
          (values (closer-mop:class-direct-slots class) t))
    (values
     (mapcan (lambda (slot)
               (mapcar
                (lambda (initarg)
                  (if (swank-mop:slot-definition-initfunction slot)
                      (list (intern (symbol-name initarg))
                            (swank-mop:slot-definition-initform slot))
                      (list (intern (symbol-name initarg))
                            nil
                            (gensym (symbol-name initarg)))))
                (closer-mop:slot-definition-initargs slot)))
             slots)
     allow-other-keys-p)))


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
  (flet ((foo ()
           (let ((*random-state*
                   #+sbcl (sb-ext:seed-random-state seed)
                   #+allegro (make-random-state t seed)
                   #-(or sbcl allegro) *random-state)
                 (*cuda-default-random-seed* seed))
             (funcall fn))))
    (if (use-cuda-p)
        (with-curand-state ((mgl-mat::make-xorwow-state/simple
                             seed *cuda-default-n-random-states*))
          (foo))
        (foo))))

(defmacro repeatably ((&key (seed *experiment-random-seed*)) &body body)
  `(call-repeatably (lambda () ,@body) :seed ,seed))
