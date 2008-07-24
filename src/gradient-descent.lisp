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
               (type flt-vector ,accumulator)
               (type (or flt-vector null) ,accumulator2)
               (ignorable ,accumulator2))
      ,@body)
    ,trainer))

(defgeneric maybe-update-weights (trainer n-new-inputs)
  (:documentation "Update the weights being trained. N-NEW-INPUTS have
been seen since the last time this was called."))


;;;; Gradient descent

(defclass gd-trainer ()
  ((n-inputs :initform 0 :initarg :n-inputs :accessor n-inputs)
   (segment-set
    :reader segment-set
    :documentation "The set of segments that are to be trained. The
ACCUMULATOR1, ACCUMULATOR2, WEIGHT-DELTAS, etc vectors are indexed by
SEGMENT-SET indices.")
   (weight-deltas :accessor weight-deltas)
   (use-accumulator2
    :initform nil :initarg :use-accumulator2 :reader use-accumulator2
    :documentation "Controls whether ACCUMULATOR2 shall be set up by
INITIALIZE-GD-TRAINER.")
   (accumulator1
    :type flt-vector :accessor accumulator1
    :documentation "One of two FLT vectors that are accessed directly
by the client and are used to store the sum of the computed gradient.")
   (accumulator2
    :type (or null flt-vector) :accessor accumulator2
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
    :documentation "WEIGHT-DECAY * WEIGHT is subtracted from the
gradient to penalize large weights."))
  (:documentation "This is the common base class of gradient descent
based trainers with momentum and weight decay."))

(defclass batch-gd-trainer (gd-trainer)
  ((batch-size
    :initarg :batch-size :accessor batch-size
    :documentation "After having gone through BATCH-SIZE number of
inputs weights are updated.")
   (n-inputs-in-batch
    :initform 0 :initarg :n-inputs-in-batch :accessor n-inputs-in-batch
    :documentation "In-batch counter of inputs."))
  (:documentation "Updates all weights simultaneously after chewing
through BATCH-SIZE inputs. PER-WEIGHT-BATCH-GD-TRAINER may be a better
choice when some weights can go unused for instance due to missing
input values."))

(defclass per-weight-batch-gd-trainer (gd-trainer)
  ((batch-size
    :initarg :batch-size :accessor batch-size
    :documentation "After BATCH-SIZE number of `uses' of a weight it
is updated. Normally there is one use per input, but it might be less
when there are missing values or more with weight sharing.")
   (n-weight-uses-in-batch
    :accessor n-weight-uses-in-batch
    :documentation "Number of uses of the weight in its current batch."))
  (:documentation "This is much like BATCH-GD-TRAINER but it is more
clever about when to update weights. Basically every weight has its
own batch independent from the batches of others. It has desirable
properties. One can for example put two neural networks together
without addding any connections between them and the learning will
produce results equivalent to separated case. Also, adding inputs with
only missing values does not change anything."))

(defmethod initialize-trainer ((trainer gd-trainer) segmentable)
  (setf (slot-value trainer 'segment-set)
        (make-instance 'segment-set :segments (list-segments segmentable)))
  (let ((n-weights (segment-set-size (segment-set trainer))))
    (setf (accumulator1 trainer) (make-flt-array n-weights))
    (fill (accumulator1 trainer) #.(flt 0))
    (cond ((use-accumulator2 trainer)
           (setf (accumulator2 trainer) (make-flt-array n-weights))
           (fill (accumulator2 trainer) #.(flt 0)))
          (t
           (setf (accumulator2 trainer) nil)))
    (setf (weight-deltas trainer) (make-flt-array n-weights))))

(defmethod initialize-trainer ((trainer per-weight-batch-gd-trainer)
                               segmentable)
  (call-next-method)
  (let ((n-weights (segment-set-size (segment-set trainer))))
    (setf (n-weight-uses-in-batch trainer)
          (make-array n-weights :element-type 'index :initial-element 0))))

(defmethod supports-partial-updates-p ((trainer per-weight-batch-gd-trainer))
  t)

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
    (let ((accumulator1 (accumulator1 trainer))
          (accumulator2 (accumulator2 trainer))
          (weight-deltas (weight-deltas trainer))
          (learning-rate (learning-rate trainer))
          (n-inputs (flt (n-inputs-in-batch trainer)))
          (momentum (momentum trainer))
          (weight-decay (weight-decay trainer)))
      (declare (type flt-vector accumulator1 weight-deltas)
               (type (or null flt-vector) accumulator2)
               (type flt learning-rate n-inputs momentum weight-decay)
               (optimize (speed 3) #.*no-array-bounds-check*))
      (do-segment-set (segment :start-in-segment-set start-in-segment-set)
          (segment-set trainer)
        (with-segment-weights ((weights start end) segment)
          (do ((i start-in-segment-set (#.*the* index (1+ i)))
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
                            (* weight-decay (aref weights j)))))
              (setf (aref weight-deltas i) delta)
              (decf (aref weights j) (* learning-rate delta)))))
        (setf (n-inputs-in-batch trainer) 0)
        (fill accumulator1 #.(flt 0))
        (when accumulator2
          (fill accumulator2 #.(flt 0))))))
  (incf (n-inputs trainer) n-new-inputs))

(defmethod maybe-update-weights ((trainer per-weight-batch-gd-trainer)
                                 n-new-inputs)
  (let ((accumulator1 (accumulator1 trainer))
        (accumulator2 (accumulator2 trainer))
        (n-weight-uses-in-batch (n-weight-uses-in-batch trainer))
        (weight-deltas (weight-deltas trainer))
        (learning-rate (learning-rate trainer))
        (momentum (momentum trainer))
        (weight-decay (weight-decay trainer))
        (batch-size (batch-size trainer)))
    (declare (type flt-vector accumulator1 weight-deltas)
             (type (or null flt-vector) accumulator2)
             (type index-vector n-weight-uses-in-batch)
             (type flt learning-rate momentum weight-decay)
             (type index batch-size)
             (optimize (speed 3) #.*no-array-bounds-check*))
    (do-segment-set (segment :start-in-segment-set start-in-segment-set)
        (segment-set trainer)
      (with-segment-weights ((weights weights-start weights-end) segment)
        (declare (ignore weights-end))
        (map-segment-runs
         (lambda (start end)
           (declare (type index start end))
           (do ((i (#.*the* index
                            (+ start-in-segment-set (- start weights-start)))
                   (#.*the* index (1+ i)))
                (j start (1+ j)))
               ((<= end j))
             (when (<= batch-size
                       (setf (aref n-weight-uses-in-batch i)
                             (1+ (#.*the* index
                                          (aref n-weight-uses-in-batch i)))))
               (let ((delta (+ (* momentum (aref weight-deltas i))
                               (/ (if accumulator2
                                      (+ (aref accumulator1 i)
                                         (aref accumulator2 i))
                                      (aref accumulator1 i))
                                  (aref n-weight-uses-in-batch i))
                               (* weight-decay (aref weights j)))))
                 (setf (aref weight-deltas i) delta)
                 (decf (aref weights j) (* learning-rate delta))
                 (setf (aref n-weight-uses-in-batch i) 0
                       (aref accumulator1 i) #.(flt 0))
                 (when accumulator2
                   (setf (aref accumulator2 i) #.(flt 0)))))))
         segment))))
  (incf (n-inputs trainer) n-new-inputs))

(defmethod train-one (sample (trainer gd-trainer) learner &key)
  (declare (ignore sample learner))
  (maybe-update-weights trainer 1))

(defmethod train-batch (batch (trainer gd-trainer) learner &key)
  (declare (ignore learner))
  (maybe-update-weights trainer (length batch)))


;;;; Trainer

(defclass segmented-trainer ()
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

(defmethod initialize-trainer ((trainer segmented-trainer) learner)
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

(defmethod maybe-update-weights ((segmented-trainer segmented-trainer)
                                 n-new-inputs)
  (dolist (trainer (trainers segmented-trainer))
    (maybe-update-weights trainer n-new-inputs))
  (incf (n-inputs segmented-trainer) n-new-inputs))

(defmethod train-one (sample (trainer segmented-trainer) learner &key)
  (declare (ignore sample learner))
  (maybe-update-weights trainer 1))

(defmethod train-batch (batch (trainer segmented-trainer) learner &key)
  (declare (ignore learner))
  (maybe-update-weights trainer (length batch)))

(defmethod map-segment-gradient-accumulators (fn (trainer segmented-trainer))
  (dolist (trainer (trainers trainer))
    (map-segment-gradient-accumulators fn trainer)))
