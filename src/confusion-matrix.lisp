(in-package :mgl-util)

(defclass confusion-matrix ()
  ((counts :initform (make-hash-table :test #'equal) :reader counts))
  (:documentation "A confusion matrix keeps count of classification
results. The correct class is called `target' and the output of the
classifier is called `prediction'. Classes are compared with EQUAL."))

(defgeneric sort-confusion-classes (matrix classes)
  (:documentation "Return a list of CLASSES sorted for presentation
purposes.")
  (:method ((matrix confusion-matrix) classes)
    (sort (copy-seq classes) #'string<=
          :key (lambda (class)
                 (confusion-class-name matrix class)))))

(defgeneric confusion-class-name (matrix class)
  (:documentation "Name of CLASS for presentation purposes.")
  (:method ((matrix confusion-matrix) class)
    (princ-to-string class)))

(defgeneric confusion-count (matrix target prediction)
  (:method ((matrix confusion-matrix) target prediction)
    (gethash (cons target prediction) (counts matrix) 0)))

(defgeneric (setf confusion-count) (count matrix target prediction)
  (:method (count (matrix confusion-matrix) target prediction)
    (setf (gethash (cons target prediction) (counts matrix)) count)))

(defgeneric map-confusion-matrix (fn matrix)
  (:documentation "Call FN with TARGET, PREDICTION, COUNT paramaters
for each cell in the confusion matrix. Cells with a zero count may be
ommitted.")
  (:method (fn (matrix confusion-matrix))
    (maphash (lambda (key value)
               (funcall fn (car key) (cdr key) value))
             (counts matrix))))

(defgeneric confusion-matrix-classes (matrix)
  (:documentation "A list of all classes. The default is to collect
classes from the counts. This can be overridden if, for instance, some
classes are not present in the results.")
  (:method ((matrix confusion-matrix))
    (let ((all-classes ()))
      (map-confusion-matrix (lambda (target prediction count)
                              (declare (ignore count))
                              (pushnew target all-classes :test #'equal)
                              (pushnew prediction all-classes :test #'equal))
                            matrix)
      all-classes)))

(defun confusion-matrix-accuracy (matrix &key filter)
  "Return the overall accuracy of the results in MATRIX. It's computed
as the number of correctly classified cases (hits) divided by the name
of cases. Return the number of hits and the number of cases as the
second and third value. If FILTER function is given, then call it with
the target and the prediction of the cell. Disregard cell for which
FILTER returns NIL.

Precision and recall can be easily computed by giving the right
filter, although those are provided in separate convenience
functions."
  (let ((n-hits 0)
        (total 0))
    (map-confusion-matrix (lambda (target prediction count)
                            (when (or (null filter)
                                      (funcall filter target prediction))
                              (when (eql target prediction)
                                (incf n-hits count))
                              (incf total count)))
                          matrix)
    (if (zerop total)
        (values nil 0 0)
        (values (/ n-hits total) n-hits total))))

(defun confusion-matrix-precision (matrix prediction)
  "Return the accuracy over the cases when the classifier said
PREDICTION."
  (confusion-matrix-accuracy matrix
                             :filter (lambda (target prediction2)
                                       (declare (ignore target))
                                       (eql prediction prediction2))))

(defun confusion-matrix-recall (matrix target)
  "Return the accuracy over the cases when the correct class is
TARGET."
  (confusion-matrix-accuracy matrix
                             :filter (lambda (target2 prediction)
                                       (declare (ignore prediction))
                                       (eql target target2))))

(defun add-confusion-matrix (matrix result-matrix)
  "Add MATRIX into RESULT-MATRIX."
  (map-confusion-matrix (lambda (target prediction count)
                          (incf (confusion-count result-matrix target prediction)
                                count))
                        matrix)
  result-matrix)

(defmethod print-object ((matrix confusion-matrix) stream)
  (flet ((->% (x)
           (if x
               (format nil "~,2F%" (* 100 x))
               nil)))
    (let ((all-classes
           (sort-confusion-classes matrix
                                   (confusion-matrix-classes matrix))))
      (print-unreadable-object (matrix stream :type t)
        (print-table
         `(("" ,@(mapcar (lambda (class)
                           (confusion-class-name matrix class))
                         all-classes)
            "Recall")
           ,@(loop
              for target in all-classes
              collect (append (cons (confusion-class-name matrix target)
                                    (loop for prediction in all-classes
                                          collect (confusion-count
                                                   matrix target prediction)))
                              (list (->% (confusion-matrix-recall matrix
                                                                  target)))))
           ("Precision"
            ,@(loop for prediction in all-classes
                    collect (->% (confusion-matrix-precision matrix prediction)))
            ""))
         :stream stream)
        (format stream "Accuracy: ~A" (->% (confusion-matrix-accuracy matrix)))))
    matrix))
