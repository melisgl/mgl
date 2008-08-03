(in-package :mgl-gd)

;;;; Generic gradient based optimization interface.

(defgeneric map-segment-gradient-accumulators (fn trainer)
  (:documentation "Call FN of lambda list (SEGMENT ACC-START
ACCUMULATOR &OPTIONAL ACCUMULATOR2) on each segment trained by
TRAINER."))

(defmacro do-segment-gradient-accumulators
    (((segment acc-start accumulator &optional (accumulator2 (gensym)))
      trainer)
     &body body)
  `(map-segment-gradient-accumulators
    (lambda (,segment ,acc-start ,accumulator &optional ,accumulator2)
      (declare (type index ,acc-start)
               (type matlisp:real-matrix ,accumulator)
               (type (or matlisp:real-matrix null) ,accumulator2)
               (ignorable ,accumulator2))
      ,@body)
    ,trainer))

(defgeneric maybe-update-weights (trainer n-new-inputs)
  (:documentation "Update the weights being trained. N-NEW-INPUTS have
been seen since the last time this was called."))

(defun find-segment-gradient-accumulator (segment trainer)
  (do-segment-gradient-accumulators ((segment2 start accumulator accumulator2)
                                     trainer)
    (when (eq segment2 segment)
      (return-from find-segment-gradient-accumulator
        (values start accumulator accumulator2)))))

(defmacro with-segment-gradient-accumulator (((start accumulator
                                                     &optional accumulator2)
                                              (segment trainer))
                                             &body body)
  `(multiple-value-bind (,start ,accumulator accumulator2)
       (find-segment-gradient-accumulator ,segment ,trainer)
     (declare (type (or index null) ,start)
              (type (or matlisp:real-matrix null) ,accumulator ,accumulator2))
     ,@body))


;;;; Gradient descent

(defclass gd-trainer ()
  ((n-inputs :initform 0 :initarg :n-inputs :accessor n-inputs)
   (segment-set
    :reader segment-set
    :documentation "The set of segments that are to be trained. The
ACCUMULATOR1, ACCUMULATOR2, WEIGHT-DELTAS, etc vectors are indexed by
SEGMENT-SET indices.")
   (weight-deltas :type matlisp:real-matrix :accessor weight-deltas)
   (use-accumulator2
    :initform nil :initarg :use-accumulator2 :reader use-accumulator2
    :documentation "Controls whether ACCUMULATOR2 shall be set up by
INITIALIZE-GD-TRAINER.")
   (accumulator1
    :type matlisp:real-matrix :accessor accumulator1
    :documentation "One of two FLT vectors that are accessed directly
by the client and are used to store the sum of the computed gradient.")
   (accumulator2
    :type (or null matlisp:real-matrix) :accessor accumulator2
    :documentation "Another accumulator that, if present, is simply
added to ACCUMULATOR1. Accumulating the positive and negative
gradients in different accumulators in a longish batch helps with
numerical aaccuracy problems.")
   (learning-rate
    :initform #.(flt 0.1) :initarg :learning-rate :accessor learning-rate
    :documentation "This is normally divided by the number of inputs
in the batch or the number of uses the weight in question have seen.")
   (momentum :initform #.(flt 0.9) :initarg :momentum :accessor momentum)
   (weight-decay
    :initform #.(flt 0) :initarg :weight-decay :accessor weight-decay
    :documentation "WEIGHT-DECAY * WEIGHT is added to the gradient to
penalize large weights. It's as if the function whose minima is sought
had sum_i{0.5 * WEIGHT-DECAY * WEIGHT_i^2} added to it.")
   (weight-penalty
    :initform #.(flt 0) :initarg :weight-penalty :accessor weight-penalty
    :documentation "WEIGHT-PENALTY is added to the gradient pushing
the weight towards negative infinity. It's as if the function whose
minima is sought had sum_i{WEIGHT-PENALTY*WEIGHT_i} added to it.
Putting it on feature biases consitutes a sparsity constraint on the
features.")
   (batch-size
    :initarg :batch-size :accessor batch-size
    :documentation "Normally, after having gone through BATCH-SIZE
number of inputs weights are updated. See subclasses for more correct
descriptions."))
  (:documentation "This is the common base class of gradient descent
based trainers with momentum and weight decay."))

(defclass batch-gd-trainer (gd-trainer)
  ((n-inputs-in-batch
    :initform 0 :initarg :n-inputs-in-batch :accessor n-inputs-in-batch
    :documentation "In-batch counter of inputs."))
  (:documentation "Updates all weights simultaneously after chewing
through BATCH-SIZE inputs. PER-WEIGHT-BATCH-GD-TRAINER may be a better
choice when some weights can go unused for instance due to missing
input values."))

(defclass normalized-batch-gd-trainer (batch-gd-trainer)
  ((n-weight-uses-in-batch
    :accessor n-weight-uses-in-batch
    :documentation "Number of uses of the weight in its current batch."))
  (:documentation "Like BATCH-GD-TRAINER but keeps count of how many
times each weight was used in the batch and divides the accumulated
gradient by this count instead of dividing by N-INPUTS-IN-BATCH. This
only makes a difference if there are missing values in the learner
that's being trained. The main feature that distuinguishes this class
from PER-WEIGHT-BATCH-GD-TRAINER is that batches end at same time for
all weights."))

(defclass per-weight-batch-gd-trainer (gd-trainer)
  ((n-weight-uses-in-batch
    :accessor n-weight-uses-in-batch
    :documentation "Number of uses of the weight in its current batch."))
  (:documentation "This is much like BATCH-GD-TRAINER but it is more
clever about when to update weights. Basically every weight has its
own batch independent from the batches of others. It has desirable
properties. One can for example put two neural networks together
without adding any connections between them and the learning will
produce results equivalent to separated case. Also, adding inputs with
only missing values does not change anything."))

(defmethod initialize-trainer ((trainer gd-trainer) segmentable)
  (setf (slot-value trainer 'segment-set)
        (make-instance 'segment-set :segments (list-segments segmentable)))
  (let ((n-weights (segment-set-size (segment-set trainer))))
    (setf (accumulator1 trainer) (matlisp:make-real-matrix n-weights 1))
    (matlisp:fill-matrix (accumulator1 trainer) #.(flt 0))
    (cond ((use-accumulator2 trainer)
           (setf (accumulator2 trainer) (matlisp:make-real-matrix n-weights 1))
           (matlisp:fill-matrix (accumulator2 trainer) #.(flt 0)))
          (t
           (setf (accumulator2 trainer) nil)))
    (setf (weight-deltas trainer) (matlisp:make-real-matrix n-weights 1))))

(defun set-up-n-weight-uses (trainer)
  (let ((n-weights (segment-set-size (segment-set trainer))))
    (setf (n-weight-uses-in-batch trainer)
          (make-array n-weights :element-type 'index :initial-element 0))))

(defmethod initialize-trainer ((trainer normalized-batch-gd-trainer) segmentable)
  (call-next-method)
  (set-up-n-weight-uses trainer))

(defmethod initialize-trainer ((trainer per-weight-batch-gd-trainer) segmentable)
  (call-next-method)
  (set-up-n-weight-uses trainer))

(defmethod segments ((trainer gd-trainer))
  (segments (segment-set trainer)))

(defmethod map-segment-gradient-accumulators (fn (trainer gd-trainer))
  (let ((segment-set (segment-set trainer))
        (accumulator1 (accumulator1 trainer))
        (accumulator2 (accumulator2 trainer)))
    (do-segment-set (segment :start-in-segment-set start) segment-set
      (funcall fn segment start accumulator1 accumulator2))))

;;; delta_w' += m * delta_w + df/dw
;;;
;;; w' -= learning_rate * delta_w'
;;;
;;; This is the same as:
;;;
;;; delta_w' += m * delta_w + learning_rate * df/dw
;;;
;;; w' -= delta_w'
;;;
;;; Decrement WEIGHTS by
;;;
;;;   (+ (/ (+ ACCUMULATOR1 ACCUMULATOR2)
;;;        N-INPUTS)
;;;     (* WEIGHT-DECAY WEIGHTS))
(defmethod maybe-update-weights ((trainer batch-gd-trainer) n-new-inputs)
  (when (<= (batch-size trainer)
            (incf (n-inputs-in-batch trainer) n-new-inputs))
    (let ((accumulator1 (storage (accumulator1 trainer)))
          (accumulator2 (if (accumulator2 trainer)
                            (storage (accumulator2 trainer))
                            nil))
          (weight-deltas (storage (weight-deltas trainer)))
          (learning-rate (learning-rate trainer))
          (n-inputs (flt (n-inputs-in-batch trainer)))
          (momentum (momentum trainer))
          (weight-decay (weight-decay trainer))
          (weight-penalty (weight-penalty trainer)))
      (declare (type flt-vector accumulator1 weight-deltas)
               (type (or null flt-vector) accumulator2)
               (type flt learning-rate n-inputs momentum
                     weight-decay weight-penalty)
               (optimize (speed 3) #.*no-array-bounds-check*))
      (do-segment-set (segment :start-in-segment-set start-in-segment-set)
          (segment-set trainer)
        (with-segment-weights ((weights start end) segment)
          ;; Maybe this could be done with Matlisp, but it's not a
          ;; hotspot.
          (do ((i start-in-segment-set (the! index (1+ i)))
               (j start (1+ j)))
              ((<= end j))
            (let ((delta (+ (* momentum (aref weight-deltas i))
                            ;; Normally we'd multiply this by LEARNING-RATE
                            ;; here, but doing it when updating the weights
                            ;; plays nicer with changing learning rates.
                            (/ (if accumulator2
                                   (+ (aref accumulator1 i)
                                      (aref accumulator2 i))
                                   (aref accumulator1 i))
                               n-inputs)
                            (* weight-decay (aref weights j))
                            weight-penalty)))
              (setf (aref weight-deltas i) delta)
              (decf (aref weights j) (* learning-rate delta)))))
        (setf (n-inputs-in-batch trainer) 0)
        (fill accumulator1 #.(flt 0))
        (when accumulator2
          (fill accumulator2 #.(flt 0))))))
  (incf (n-inputs trainer) n-new-inputs))

(defmethod maybe-update-weights ((trainer normalized-batch-gd-trainer)
                                 n-new-inputs)
  (declare (type index n-new-inputs))
  (let ((accumulator1 (storage (accumulator1 trainer)))
        (accumulator2 (if (accumulator2 trainer)
                          (storage (accumulator2 trainer))
                          nil))
        (n-weight-uses-in-batch (n-weight-uses-in-batch trainer))
        (weight-deltas (storage (weight-deltas trainer)))
        (learning-rate (learning-rate trainer))
        (momentum (momentum trainer))
        (weight-decay (weight-decay trainer))
        (weight-penalty (weight-penalty trainer))
        (batch-size (batch-size trainer)))
    (declare (type flt-vector accumulator1 weight-deltas)
             (type (or null flt-vector) accumulator2)
             (type index-vector n-weight-uses-in-batch)
             (type flt learning-rate momentum weight-decay weight-penalty)
             (type index batch-size))
    (do-segment-set (segment :start-in-segment-set start-in-segment-set)
        (segment-set trainer)
      (with-segment-weights ((weights weights-start weights-end) segment)
        (declare (ignore weights weights-end))
        (map-segment-runs
         (lambda (start end)
           (declare (type index start end)
                    (optimize (speed 3) #.*no-array-bounds-check*))
           (do ((i (the! index
                         (+ start-in-segment-set (- start weights-start)))
                   (the! index (1+ i)))
                (j start (1+ j)))
               ((<= end j))
             (setf (aref n-weight-uses-in-batch i)
                   (the! index
                         (+ n-new-inputs
                            (the! index
                                  (aref n-weight-uses-in-batch i)))))))
         segment)))
    (when (<= batch-size (the index (incf (n-inputs-in-batch trainer)
                                          n-new-inputs)))
      (setf (n-inputs-in-batch trainer) 0)
      (do-segment-set (segment :start-in-segment-set start-in-segment-set)
          (segment-set trainer)
        (with-segment-weights ((weights start end) segment)
          (declare (optimize (speed 3) #.*no-array-bounds-check*))
          (do ((i start-in-segment-set (the! index (1+ i)))
               (j start (1+ j)))
              ((<= end j))
            (let ((delta (+ (* momentum (aref weight-deltas i))
                            (* (if (zerop (aref n-weight-uses-in-batch i))
                                   #.(flt 0)
                                   (/ (flt (aref n-weight-uses-in-batch i))))
                               (if accumulator2
                                   (+ (aref accumulator1 i)
                                      (aref accumulator2 i))
                                   (aref accumulator1 i)))
                            (* weight-decay (aref weights j))
                            weight-penalty)))
              (setf (aref weight-deltas i) delta)
              (decf (aref weights j) (* learning-rate delta))
              (setf (aref n-weight-uses-in-batch i) 0
                    (aref accumulator1 i) #.(flt 0))
              (when accumulator2
                (setf (aref accumulator2 i) #.(flt 0)))))))))
  (incf (n-inputs trainer) n-new-inputs))

(defmethod maybe-update-weights ((trainer per-weight-batch-gd-trainer)
                                 n-new-inputs)
  (assert (= 1 n-new-inputs))
  (let ((accumulator1 (storage (accumulator1 trainer)))
        (accumulator2 (if (accumulator2 trainer)
                          (storage (accumulator2 trainer))
                          nil))
        (n-weight-uses-in-batch (n-weight-uses-in-batch trainer))
        (weight-deltas (storage (weight-deltas trainer)))
        (learning-rate (learning-rate trainer))
        (momentum (momentum trainer))
        (weight-decay (weight-decay trainer))
        (weight-penalty (weight-penalty trainer))
        (batch-size (batch-size trainer)))
    (declare (type flt-vector accumulator1 weight-deltas)
             (type (or null flt-vector) accumulator2)
             (type index-vector n-weight-uses-in-batch)
             (type flt learning-rate momentum weight-decay weight-penalty)
             (type index batch-size)
             (optimize (speed 3) #.*no-array-bounds-check*))
    (do-segment-set (segment :start-in-segment-set start-in-segment-set)
        (segment-set trainer)
      (with-segment-weights ((weights weights-start weights-end) segment)
        (declare (ignore weights-end))
        (map-segment-runs
         (lambda (start end)
           (declare (type index start end))
           (do ((i (the! index
                         (+ start-in-segment-set (- start weights-start)))
                   (the! index (1+ i)))
                (j start (1+ j)))
               ((<= end j))
             (when (<= batch-size
                       (setf (aref n-weight-uses-in-batch i)
                             (1+ (the! index
                                       (aref n-weight-uses-in-batch i)))))
               (let ((delta (+ (* momentum (aref weight-deltas i))
                               (/ (if accumulator2
                                      (+ (aref accumulator1 i)
                                         (aref accumulator2 i))
                                      (aref accumulator1 i))
                                  (aref n-weight-uses-in-batch i))
                               (* weight-decay (aref weights j))
                               weight-penalty)))
                 (setf (aref weight-deltas i) delta)
                 (decf (aref weights j) (* learning-rate delta))
                 (setf (aref n-weight-uses-in-batch i) 0
                       (aref accumulator1 i) #.(flt 0))
                 (when accumulator2
                   (setf (aref accumulator2 i) #.(flt 0)))))))
         segment))))
  (incf (n-inputs trainer)))

(defmethod n-inputs-until-update ((trainer batch-gd-trainer))
  ;; BATCH-SIZE may be setf'ed to a value lower than N-INPUTS-IN-BATCH
  (max 0 (- (batch-size trainer)
            (n-inputs-in-batch trainer))))

(defmethod n-inputs-until-update ((trainer per-weight-batch-gd-trainer))
  ;; Weight updates are async, don't overpromise.
  1)


;;;; Trainer

(defclass segmented-gd-trainer ()
  ((n-inputs :initform 0 :initarg :n-inputs :accessor n-inputs)
   (segmenter
    :initarg :segmenter :accessor segmenter
    :documentation "A function that maps a segment to a trainer or
NIL. Several segments may be mapped to the same trainer. Used to
initialize SEGMENT-TRAINERS before training.")
   (trainers :type list :reader trainers)
   (segments :type list :reader segments))
  (:documentation "A trainer that delegates training of segments to
other trainers. Useful to delegate training of different segments to
different trainers or simply to not train all segments."))

(defmethod initialize-trainer ((trainer segmented-gd-trainer) learner)
  (let ((segmenter (segmenter trainer))
        (trainer-segments (make-hash-table :test 'eq)))
    (map-segments (lambda (segment)
                    (let ((trainer (funcall segmenter segment)))
                      (when trainer
                        (unless (gethash trainer trainer-segments)
                          (setf (gethash trainer trainer-segments)
                                nil))
                        (push segment (gethash trainer trainer-segments)))))
                  learner)
    (let ((trainers ()))
      (maphash (lambda (trainer segments)
                 (initialize-trainer trainer segments)
                 (push trainer trainers)
                 (values))
               trainer-segments)
      (setf (slot-value trainer 'trainers) trainers)
      ;; The child trainer may not use all the segments assigned to it
      ;; so let's ask it.
      (setf (slot-value trainer 'segments)
            (apply #'append (mapcar #'segments trainers))))))

(defmethod maybe-update-weights ((segmented-gd-trainer segmented-gd-trainer)
                                 n-new-inputs)
  (dolist (trainer (trainers segmented-gd-trainer))
    (maybe-update-weights trainer n-new-inputs))
  (incf (n-inputs segmented-gd-trainer) n-new-inputs))

(defmethod n-inputs-until-update ((trainer segmented-gd-trainer))
  (loop for child-trainer in (trainers trainer)
        minimizing (n-inputs-until-update child-trainer)))

(defmethod map-segment-gradient-accumulators (fn (trainer segmented-gd-trainer))
  (dolist (trainer (trainers trainer))
    (map-segment-gradient-accumulators fn trainer)))
