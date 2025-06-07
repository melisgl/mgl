(in-package :mgl-core)

(defsection @mgl-counter (:title "Counters")
  (add-to-counter generic-function)
  (counter-values generic-function)
  (counter-raw-values generic-function)
  (reset-counter generic-function)
  (@mgl-attributes section)
  (@mgl-counter-classes section))

(defgeneric add-to-counter (counter &rest args)
  (:documentation "Add ARGS to COUNTER in some way. See specialized
  methods for type specific documentation. The kind of arguments to be
  supported is the what the measurer functions (see @MGL-MEASURER)
  intended to be paired with the counter return as multiple values."))

(defgeneric counter-values (counter)
  (:documentation "Return any number of values representing the state
  of COUNTER. See specialized methods for type specific
  documentation."))

(defgeneric counter-raw-values (counter)
  (:documentation "Return any number of values representing the state
  of COUNTER in such a way that passing the returned values as
  arguments ADD-TO-COUNTER on a fresh instance of the same type
  recreates the original state."))

(defgeneric reset-counter (counter)
  (:documentation "Restore state of COUNTER to what it was just after
  creation."))


(defsection @mgl-attributes (:title "Attributes")
  (attributed class)
  (attributes (accessor attributed))
  (name (method (attributed)))
  (with-padded-attribute-printing macro)
  (log-padded function))

(defclass attributed ()
  ((attributes
    :initform ()
    :initarg :attributes
    :accessor attributes
    :documentation "A plist of attribute keys and values."))
  (:documentation "This is a utility class that all counters subclass.
  The ATTRIBUTES plist can hold basically anything. Currently the
  attributes are only used when printing and they can be specified by
  the user. The monitor maker functions such as those in
  @MGL-CLASSIFICATION-MONITOR also add attributes of their own to the
  counters they create.

  With the :PREPEND-ATTRIBUTES initarg when can easily add new
  attributes without clobbering the those in the :INITFORM, (:TYPE
  \"rmse\") in this case.

      (princ (make-instance 'rmse-counter
                            :prepend-attributes '(:event \"pred.\"
                                                  :dataset \"test\")))
      ;; pred. test rmse: 0.000e+0 (0)
      => #<RMSE-COUNTER pred. test rmse: 0.000e+0 (0)>"))

(defvar *attribute-print-widths* ())

(defun attribute-print-width (attribute)
  (cdr (assoc attribute *attribute-print-widths*)))

(defun get-all (plist indicator)
  (loop for (indicator-2 value) on plist by #'cddr
        when (eq indicator-2 indicator)
          collect value))

(defun plists-to-column-widths (plists)
  (let ((widths (make-hash-table)))
    (dolist (plist plists)
      (loop for (indicator value) on plist by #'cddr
            do (setf (gethash indicator widths)
                     (max (gethash indicator widths 0)
                          (length (format nil "~{~A~^, ~}"
                                          (get-all plist indicator)))))))
    (alexandria:hash-table-alist widths)))

(defmacro with-padded-attribute-printing ((attributeds) &body body)
  "Note the width of values for each attribute key which is the number
  of characters in the value's PRINC-TO-STRING'ed representation. In
  BODY, if attributes with they same key are printed they are forced
  to be at least this wide. This allows for nice, table-like output:

      (let ((attributeds
              (list (make-instance 'basic-counter
                                   :attributes '(:a 1 :b 23 :c 456))
                    (make-instance 'basic-counter
                                   :attributes '(:a 123 :b 45 :c 6)))))
        (with-padded-attribute-printing (attributeds)
          (map nil (lambda (attributed)
                     (format t \"~A~%\" attributed))
               attributeds)))
      ;; 1   23 456: 0.000e+0 (0)
      ;; 123 45 6  : 0.000e+0 (0)"
  `(let ((*attribute-print-widths*
           (append
            (plists-to-column-widths (mapcar #'attributes ,attributeds))
            *attribute-print-widths*)))
     ,@body))

(defun pad-to-width (value width)
  (let ((value (princ-to-string value)))
    (concatenate 'string value
                 (make-string (max 0 (- width (length value)))
                              :initial-element #\Space))))

(defmethod name ((attributed attributed))
  "Return a string assembled from the values of the ATTRIBUTES of
  ATTRIBUTED. If there are multiple entries with the same key, then
  they are printed near together.
 
  Values may be padded according to an enclosing
  WITH-PADDED-ATTRIBUTE-PRINTING."
  (let ((attributes (attributes attributed))
        (indicators-seen ()))
    (format nil "~{~A~^ ~:_~}"
            (loop for (indicator value*) on attributes by #'cddr
                  when (not (member indicator indicators-seen))
                    collect (let ((width (attribute-print-width indicator))
                                  (value
                                    (format nil "~{~A~^, ~}"
                                            (get-all attributes indicator))))
                              (if width
                                  (pad-to-width value width)
                                  value))
                  do (push indicator indicators-seen)))))

(defmethod initialize-instance :after ((attributed attributed)
                                       &key prepend-attributes
                                       &allow-other-keys)
  (when prepend-attributes
    (setf (slot-value attributed 'attributes)
          (append prepend-attributes (slot-value attributed 'attributes)))))

(defun print-name (attributed stream)
  (let ((name (name attributed)))
    (when (plusp (length name))
      (format stream "~A: ~:_" name))))

(defmacro maybe-print-unreadable-object ((object stream &key type identity)
                                         &body body)
  `(flet ((foo ()
            ,@body))
     (if *print-escape*
         (print-unreadable-object (,object ,stream :type ,type
                                   :identity ,identity)
           (foo))
         (foo))))

(defmethod print-object ((attributed attributed) stream)
  (maybe-print-unreadable-object (attributed stream :type t)
    (print-name attributed stream))
  attributed)

(defun log-padded (attributeds)
  "Log (see LOG-MSG) ATTRIBUTEDS non-escaped (as in PRINC or ~A) with
  the output being as table-like as possible."
  (with-padded-attribute-printing (attributeds)
    (map nil (lambda (attributed)
               (log-msg "~A~%" attributed))
         attributeds)))


(defsection @mgl-counter-classes (:title "Counter classes")
  "In addition to the really basic ones here, also see
  @MGL-CLASSIFICATION-COUNTER."
  (basic-counter class)
  (rmse-counter class)
  (concat-counter class)
  (concatenation-type (reader concat-counter)))

(defclass basic-counter (attributed)
  ((numerator :initform 0 :reader numerator*)
   (denominator :initform 0 :reader denominator*))
  (:documentation "A simple counter whose ADD-TO-COUNTER takes two
  additional parameters: an increment to the internal sums of called
  the NUMERATOR and DENOMINATOR. COUNTER-VALUES returns two
  values:

  - NUMERATOR divided by DENOMINATOR (or 0 if DENOMINATOR is 0) and

  - DENOMINATOR

  Here is an example the compute the mean of 5 things received in two
  batches:

       (let ((counter (make-instance 'basic-counter)))
         (add-to-counter counter 6.5 3)
         (add-to-counter counter 3.5 2)
         counter)
       => #<BASIC-COUNTER 2.00000e+0 (5)>"))

(defmethod print-object ((counter basic-counter) stream)
  (maybe-print-unreadable-object (counter stream :type t)
    (print-name counter stream)
    (multiple-value-bind (e c) (counter-values counter)
      (if e
          (format stream "~,3E" e)
          (format stream "~A" e))
      (if (integerp c)
          (format stream " (~D)" c)
          (format stream " (~,2F)" c)))))

(defmethod add-to-counter ((counter basic-counter) &rest args)
  (destructuring-bind (numerator denominator) args
    (incf (slot-value counter 'numerator) numerator)
    (incf (slot-value counter 'denominator) denominator)))

(defmethod counter-values ((counter basic-counter))
  (with-slots (numerator denominator) counter
    (values (if (zerop denominator)
                0
                (/ numerator denominator))
            denominator)))

(defmethod counter-raw-values ((counter basic-counter))
  (with-slots (numerator denominator) counter
    (values numerator denominator)))

(defmethod reset-counter ((counter basic-counter))
  (with-slots (numerator denominator) counter
    (setf numerator 0)
    (setf denominator 0)))


(defclass rmse-counter (basic-counter)
  ((attributes :initform '(:type "rmse")))
  (:documentation "A BASIC-COUNTER with whose nominator accumulates
  the square of some statistics. It has the attribute :TYPE \"rmse\".
  COUNTER-VALUES returns the square root of what BASIC-COUNTER's
  COUNTER-VALUES would return.

      (let ((counter (make-instance 'rmse-counter)))
        (add-to-counter counter (+ (* 3 3) (* 4 4)) 2)
        counter)
      => #<RMSE-COUNTER rmse: 3.53553e+0 (2)>"))

(defmethod counter-values ((counter rmse-counter))
  (multiple-value-bind (e n) (call-next-method)
    (values (sqrt e) n)))


(defclass concat-counter (attributed)
  ((concatenation :initform () :initarg :concatenation :accessor concatenation)
   (concatenation-type
    :initform 'list :initarg :concatenation-type :reader concatenation-type
    :documentation "A type designator suitable as the RESULT-TYPE
    argument to CONCATENATE."))
  (:documentation "A counter that simply concatenates
  sequences.

  ```cl-transcript
  (let ((counter (make-instance 'concat-counter)))
    (add-to-counter counter '(1 2 3) #(4 5))
    (add-to-counter counter '(6 7))
    (counter-values counter))
  => (1 2 3 4 5 6 7)
  ```"))

(defmethod add-to-counter ((counter concat-counter) &rest args)
  (dolist (seq args)
    (setf (concatenation counter)
          (concatenate (concatenation-type counter)
                       (concatenation counter)
                       seq))))

(defmethod counter-values ((counter concat-counter))
  (concatenation counter))

(defmethod counter-raw-values ((counter concat-counter))
  (concatenation counter))

(defmethod reset-counter ((counter concat-counter))
  (setf (concatenation counter) ()))
