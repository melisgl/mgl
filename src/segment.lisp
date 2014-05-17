;;;; Segments of learners
;;;;
;;;; The weights of a learner can be stored in a multitude of ways.
;;;; Trainers need to access, iterate over these weights reasonably
;;;; fast. Random access doesn't seem to be necessary.
;;;;
;;;; The following implementation requires that weights are stored in
;;;; FLT-VECTORs. MAP-SEGMENT provides iteration over segments, and
;;;; SEGMENT-WEIGHTS returns the array and the start, end indices
;;;; associated with it.

(in-package :mgl-train)

(defgeneric map-segments (fn segmentable)
  (:documentation "Apply FN to each segment of LEARNER.")
  (:method (fn (segment-list list))
    (mapc fn segment-list)))

(defgeneric segment-weights (segment)
  (:documentation "Return the weight array and start, end indices of
SEGMENT."))

(defmacro with-segment-weights (((weights start end) segment) &body body)
  `(multiple-value-bind (,weights ,start ,end) (segment-weights ,segment)
     (declare #+nil
              (type flt-vector ,weights)
              (type index ,start ,end))
     ,@body))

(defun segment-size (segment)
  (with-segment-weights ((weights start end) segment)
    (declare (ignore weights))
    (- end start)))

(defgeneric map-segment-runs (fn segment)
  (:documentation "Call FN with start and end of intervals of
consecutive indices that are not missing in SEGMENT. Called by
trainers that support partial updates.")
  (:method (fn segment)
    (with-segment-weights ((array start end) segment)
      (declare (ignore array))
      (funcall fn start end))))

(defgeneric segments (object)
  (:documentation "A list of segments associated with OBJECT. Trainers
must implement this. It is also defined on SEGMENT-SETs."))

(defun list-segments (segmentable)
  (let ((segments ()))
    (map-segments (lambda (segment)
                    (push segment segments))
                  segmentable)
    (reverse segments)))


(defclass segment-set ()
  ((segments :initform (error "Must specify segment list.")
             :initarg :segments :reader segments)
   (start-indices :reader start-indices)
   (size :reader segment-set-size))
  (:documentation "It's like a concatenation of segments."))

(defmethod print-object ((set segment-set) stream)
  (pprint-logical-block (stream ())
    (print-unreadable-object (set stream :type t :identity t)
      (format stream "~A" (segments set))))
  set)

(defmethod initialize-instance :after ((segment-set segment-set)
                                       &key &allow-other-keys)
  (let ((n 0)
        (start-indices '()))
    (dolist (segment (segments segment-set))
      (push n start-indices)
      (incf n (segment-size segment)))
    (setf (slot-value segment-set 'start-indices) (reverse start-indices)
          (slot-value segment-set 'size) n)))

(defmacro do-segment-set ((segment &key start-in-segment-set) segment-set
                          &body body)
  "Iterate over SEGMENTS in SEGMENT-SET ...."
  (with-gensyms (%segment-set %start-index)
    `(let* ((,%segment-set ,segment-set))
       (loop for ,segment in (segments ,%segment-set)
             ,@(when start-in-segment-set
                 (list 'for %start-index 'in
                       (list 'start-indices %segment-set)))
             do
             (let (,@(when start-in-segment-set
                       (list (list start-in-segment-set %start-index))))
               ,@(when start-in-segment-set
                   `((declare (type index ,start-in-segment-set))))
               ,@body)))))

(defun segment-set<-weights (segment-set weights)
  "Copy the values of WEIGHTS to SEGMENT-SET."
  (declare (type flt-vector weights)
           (optimize (speed 3)))
  (do-segment-set (segment :start-in-segment-set start-in-segment-set)
                  segment-set
    (with-segment-weights ((array start end) segment)
      (mgl-mat:with-facets ((array (array 'mgl-mat:backing-array
                                          :direction :output :type flt-vector)))
        (replace array weights :start1 start :end1 end
                 :start2 start-in-segment-set)))))

(defun segment-set->weights (segment-set weights)
  "Copy the values from SEGMENT-SET to WEIGHTS."
  (declare (type flt-vector weights)
           (optimize (speed 3)))
  (do-segment-set (segment :start-in-segment-set start-in-segment-set)
                  segment-set
    (with-segment-weights ((array start end) segment)
      (mgl-mat:with-facets ((array (array 'mgl-mat:backing-array
                                          ;; :IO because it may be
                                          ;; subseq of it.
                                          :direction :io
                                          :type flt-vector)))
        (replace weights array :start1 start-in-segment-set
                 :start2 start :end2 end)))))

(defun map-segment-set-and-mat (fn x y)
  (let ((fn (coerce fn 'function)))
    (multiple-value-bind (segment-set mat mat-first-p)
        (if (typep x 'mat)
            (values y x t)
            (values x y nil))
      (declare (optimize (speed 3)))
      (do-segment-set (segment :start-in-segment-set start-in-segment-set)
                      segment-set
        (with-segment-weights ((m start end) segment)
          (let ((n (- end start)))
            (with-shape-and-displacement (m n start)
              (with-shape-and-displacement (mat n start-in-segment-set)
                (if mat-first-p
                    (funcall fn mat m)
                    (funcall fn m mat))))))))))

(defun segment-set<-mat (segment-set mat)
  "Copy the values of MAT to SEGMENT-SET."
  (map-segment-set-and-mat #'copy! mat segment-set))

(defun segment-set->mat (segment-set mat)
  "Copy the values of SEGMENT-SET to MAT."
  (map-segment-set-and-mat #'copy! segment-set mat))
