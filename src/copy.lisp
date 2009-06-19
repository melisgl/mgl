;;;; Generic, customizable copy mechanism.

(in-package :mgl-util)

(defvar *objects-copied*)

(defmacro with-copying (&body body)
  `(flet ((foo () ,@body))
     (if (boundp '*objects-copied*)
         (foo)
         (let ((*objects-copied* (make-hash-table)))
           (foo)))))

(defgeneric copy-object-extra-initargs (context original-object)
  (:documentation "Return a list of ")
  (:method (context original-object)
    '()))

(defgeneric copy-object-slot (context original-object slot-name value)
  (:documentation "Return the value of the slot in the copied object
and T, or NIL as the second value if the slot need not be initialized.
The default implementation of COPY-FOR-PCD for classes calls
COPY-SLOT-FOR-PCD.")
  (:method (context original-object slot-name value)
    (values (copy context value) t)))

(defmacro define-slot-not-to-be-copied (context class slot-name)
  `(defmethod copy-object-slot ((context (eql ,context))
                                (original ,class)
                                (slot-name (eql ',slot-name))
                                value)
     (values nil nil)))

(defmacro define-slots-not-to-be-copied (context class &body slot-names)
  (let ((%context (gensym)))
    `(let ((,%context ,context))
       ,@(loop for slot-name in slot-names
               collect `(define-slot-not-to-be-copied ,%context ,class
                         ,slot-name)))))

(defmacro define-slot-to-be-shallow-copied (context class slot-name)
  `(defmethod copy-object-slot ((context (eql ,context))
                                (original ,class)
                                (slot-name (eql ',slot-name))
                                value)
     (values value t)))

(defmacro define-slots-to-be-shallow-copied (context class &body slot-names)
  (let ((%context (gensym)))
    `(let ((,%context ,context))
       ,@(loop for slot-name in slot-names
               collect `(define-slot-to-be-shallow-copied ,%context ,class
                         ,slot-name)))))

(defgeneric copy (context object)
  (:documentation "Make a deepish copy of OBJECT suitable for use as
the rbm with the persistent chains in PCD learning, sharing weights
and parameters but not node values. Return the copied object, with
node values initialized, MAX-N-STRIPES, N-STRIPES set to the number of
persistent chains.

INDICES-PRESENT must also be set up, and probably left unchanged for
the duration of training. When using INDICES-PRESENT a large number of
persistent chains can be required so CD learning may be a better bet.
The same consideration applies to non-constant conditioning chunks.")
  (:method :around (context object)
           (with-copying
             (multiple-value-bind (copy existp)
                 (gethash object *objects-copied*)
               (if existp
                   copy
                   (setf (gethash object *objects-copied*)
                         (call-next-method))))))
  (:method (context object)
    object)
  (:method (context (cons cons))
    ;; With the *OBJECTS-COPIED* hash table, this is suitable for
    ;; circular lists, too.
    (cons (copy context (car cons))
          (copy context (cdr cons))))
  (:method (context (object standard-object))
    (let ((class (class-of object))
          (initargs ())
          (inits ()))
      (dolist (slot (c2mop:class-slots class))
        (let ((slot-name (c2mop:slot-definition-name slot)))
          (when (slot-boundp object slot-name)
            (multiple-value-bind (new-slot-value initializep)
                (copy-object-slot context object slot-name
                                  (slot-value object slot-name))
              (when initializep
                (let ((initarg (first (c2mop:slot-definition-initargs slot))))
                  (if initarg
                      (push-all (list new-slot-value initarg) initargs)
                      (push (list slot-name new-slot-value) inits))))))))
      (let ((instance
             (apply #'make-instance class
                    (append (copy-seq
                             (copy-object-extra-initargs context object))
                            initargs))))
        (loop for (slot-name value) in inits
              do (setf (slot-value instance slot-name) value))
        instance))))
