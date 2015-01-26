(in-package :mgl-common)

(defsection @mgl-common ()
  (name generic-function)
  (name= function)
  (size generic-function)
  (nodes generic-function)
  (default-value generic-function)
  (group-size generic-function)
  (batch-size generic-function)
  (weights generic-function))

(defgeneric name (object))
(setf (fdefinition 'name=) #'equal)
(defgeneric size (object))
(defgeneric nodes (object)
  (:documentation "Returns a MAT object representing the state or
  result of OBJECT. The first dimension of the returned matrix is
  equal to the number of stripes."))
(defgeneric default-value (object))
(defgeneric group-size (object))
(defgeneric batch-size (object))
(defgeneric weights (object))
