;;;; Gradient based optimization
;;;;
;;;; The two important concepts are gradient source and gradient sink.
;;;; A `sink' is a trainer that drives the training process, relying
;;;; on the `source' to calculate the gradients.

(in-package :mgl-gd)

;;;; Abstract interface for implementing gradient sinks

(defclass gradient-sink ()
  ((gradient-source :initarg :gradient-source :reader gradient-source))
  (:documentation "Base class of all gradient descent trainers."))

(defgeneric initialize-gradient-sink (sink source segmentable)
  (:method :before ((sink gradient-sink) source segmentable)
    (setf (slot-value sink 'gradient-source) source))
  (:documentation "Called automatically before training starts, this
function sets up SINK to be suitable for SOURCE. It typically creates
accumulator arrays in the sink for the gradients."))

(defgeneric n-inputs-until-update (sink)
  (:documentation "Return the largest number of inputs guaranteed not
to cause a change in the learner being trained."))

(defgeneric maybe-update-weights (sink n-new-inputs)
  (:documentation "Update the weights of the learner being trained.
N-NEW-INPUTS have been seen since the last time this was called."))


;;;; Abstract interface for implementing gradient sources

(defgeneric segmentable (source)
  (:method (source)
    source)
  (:documentation "Return the segmentable model of SOURCE. By default
it is assumed that SOURCE itself is a segmentable, so only non-trivial
sources need to define this."))

(defgeneric initialize-gradient-source (source segmentable sink)
  (:documentation "Called automatically before training starts, this
function sets up SOURCE to be suitable for its own
SEGMENTABLE (returned by the function SEGMENTABLE)."))

(defgeneric accumulate-gradients (batch source sink multiplier)
  (:documentation "For each example in BATCH calculate the derivatives
and add them (multiplied by MULTIPLIER) to the corresponding
accumulator (in the sense of FIND-SINK-ACCUMULATOR) of SINK."))

(defvar *accumulating-interesting-gradients* nil)


;;;; Implementation of training based on the above abstract interfaces

(defmethod train (sampler (sink gradient-sink) source)
  (while (not (finishedp sampler))
    (let ((batch (sample-batch sampler (n-inputs-until-update sink))))
      (train-batch batch sink source))))

(defmethod train :around (sampler (sink gradient-sink) source)
  (let ((segmentable (segmentable source)))
    (initialize-gradient-sink sink source segmentable)
    (initialize-gradient-source source segmentable sink))
  (call-next-method))

(defmethod train-batch (batch (sink gradient-sink) learner)
  (let ((*accumulating-interesting-gradients* t))
    (accumulate-gradients batch (gradient-source sink) sink #.(flt 1)))
  (maybe-update-weights sink (length batch)))


;;;; Interface to gradient sinks for gradient sources

(defgeneric map-gradient-sink (fn sink)
  (:documentation "Call FN of lambda list (SEGMENT ACCUMULATOR
ACC-START) on each segment and their corresponding accumulator array
plus start index in SINK."))

(defmacro do-gradient-sink (((segment accumulator acc-start) sink)
                            &body body)
  `(map-gradient-sink
    (lambda (,segment ,accumulator ,acc-start)
      (declare #+nil
               (type flt-vector ,accumulator)
               (type index ,acc-start))
      ,@body)
    ,sink))

(defgeneric find-sink-accumulator (segment source sink)
  (:documentation "Return the accumulator and start index belonging to
SEGMENT of SOURCE in SINK or NIL if it is not found.")
  (:method (segment source sink)
    (declare (ignore source))
    (do-gradient-sink ((segment2 accumulator start) sink)
      (when (eq segment2 segment)
        (return-from find-sink-accumulator
          (values accumulator start))))))

(defmacro with-sink-accumulator (((accumulator start) (segment source sink))
                                 &body body)
  `(multiple-value-bind (,accumulator ,start)
       (find-sink-accumulator ,segment ,source ,sink)
     (declare (type (or index null) ,start)
              #+nil
              (type (or flt-vector null) ,accumulator))
     ,@body))


;;;; Abstract gradient descent base class

(defclass gd-trainer (gradient-sink)
  ((n-inputs :initform 0 :initarg :n-inputs :accessor n-inputs)
   (segment-set
    :reader segment-set
    :documentation "The set of segments that are to be trained. The
ACCUMULATOR, WEIGHT-DELTAS, etc vectors are indexed by SEGMENT-SET
indices.")
   (weight-deltas :type mat :accessor weight-deltas)
   (accumulator
    :type mat :accessor accumulator
    :documentation "An FLT vector that is accessed directly by the
client and are used to store the sum of the computed gradient.")
   (learning-rate
    :initform #.(flt 0.1) :initarg :learning-rate :accessor learning-rate
    :documentation "This is normally divided by the number of inputs
in the batch or the number of uses the weight in question has seen.")
   (momentum
    :initform #.(flt 0) :initarg :momentum :accessor momentum)
   (momentum-type
    :initform :normal :initarg :momentum-type :accessor momentum-type
    :type '(member :normal :nesterov))
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
   (after-update-hook
    :type 'list
    :initform () :initarg :after-update-hook :accessor after-update-hook
    :documentation "A list of functions with no arguments called after
weights are updated.")
   (batch-size
    :initarg :batch-size :accessor batch-size
    :documentation "Normally, after having gone through BATCH-SIZE
number of inputs weights are updated. See subclasses for more correct
descriptions."))
  (:documentation "Gradient descent trainer with momentum, weight
decay, weight penalty. Batch size and all other parameters can be
changed during training. One may even want to subclass this trainer,
define a method for BATCH-SIZE make it a function of N-INPUTS.

Depending on BATCH-SIZE, this may be stochastic or non-stochastic
gradient descent."))

(defmethod print-object ((trainer gd-trainer) stream)
  (pprint-logical-block (stream ())
    (print-unreadable-object (trainer stream :type t :identity t)
      (format stream "~S" (ignore-errors (segment-set trainer)))))
  trainer)

(define-descriptions (trainer gd-trainer)
  n-inputs segment-set
  (learning-rate (learning-rate trainer) "~,5E")
  (momentum (momentum trainer) "~,5E")
  momentum-type
  (weight-decay (weight-decay trainer) "~,5E")
  (weight-penalty (weight-penalty trainer) "~,5E")
  (n-after-upate-hook (length (after-update-hook trainer)) "~S")
  batch-size)

(defmethod initialize-gradient-sink ((trainer gd-trainer) source segmentable)
  (when (next-method-p)
    (call-next-method))
  (setf (slot-value trainer 'segment-set)
        (make-instance 'segment-set :segments (list-segments segmentable)))
  (let ((n-weights (segment-set-size (segment-set trainer))))
    (setf (accumulator trainer) (make-mat n-weights :ctype flt-ctype))
    (setf (weight-deltas trainer) (make-mat n-weights :ctype flt-ctype))))

(defmethod segments ((trainer gd-trainer))
  (segments (segment-set trainer)))

(defmethod map-gradient-sink (fn (trainer gd-trainer))
  (let ((segment-set (segment-set trainer))
        (accumulator (accumulator trainer)))
    (do-segment-set (segment :start-in-segment-set start) segment-set
      (funcall fn segment accumulator start))))


;;;; BATCH-GD-TRAINER

(defclass batch-gd-trainer (gd-trainer)
  ((n-inputs-in-batch
    :initform 0 :initarg :n-inputs-in-batch :accessor n-inputs-in-batch
    :documentation "In-batch counter of inputs.")
   (before-update-hook
    :type list :initform () :initarg :before-update-hook
    :accessor before-update-hook
    :documentation "A list of functions of no parameters. Each
function is called just before UPDATE-WEIGHTS takes place. Convenient
to hang some additional gradient accumulating code on."))
  (:documentation "Updates all weights simultaneously after chewing
through BATCH-SIZE inputs. PER-WEIGHT-BATCH-GD-TRAINER may be a better
choice when some weights can go unused for instance due to missing
input values."))

(defmethod n-inputs-until-update ((trainer batch-gd-trainer))
  ;; BATCH-SIZE may be setf'ed to a value lower than N-INPUTS-IN-BATCH
  (max 0 (- (batch-size trainer)
            (n-inputs-in-batch trainer))))

(defmethod maybe-update-weights ((trainer batch-gd-trainer) n-new-inputs)
  (when (<= (batch-size trainer)
            (incf (n-inputs-in-batch trainer) n-new-inputs))
    (ecase (momentum-type trainer)
      ((:normal)
       (update-all-weights/normal trainer))
      ((:nesterov)
       (update-all-weights/nesterov trainer))))
  (incf (n-inputs trainer) n-new-inputs))

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
;;;   (+ (/ ACCUMULATOR N-INPUTS)
;;;      (* WEIGHT-DECAY WEIGHTS))
;;;
;;; plus momentum, weight-penalty.
(defun update-all-weights/nesterov (trainer)
  (map nil #'funcall (before-update-hook trainer))
  (let ((accumulator (accumulator trainer))
        (weight-deltas (weight-deltas trainer))
        (learning-rate (learning-rate trainer))
        (n-inputs (flt (n-inputs-in-batch trainer)))
        (momentum (momentum trainer))
        (weight-decay (weight-decay trainer))
        (weight-penalty (weight-penalty trainer)))
    (declare (type flt learning-rate n-inputs momentum
                   weight-decay #+nil weight-penalty))
    (mgl-mat:scal! momentum weight-deltas)
    (mgl-mat:axpy! (/ n-inputs) accumulator weight-deltas)
    ;; FIXME: WEIGHT-PENALTY
    #+nil
    (unless (zerop weight-penalty)
      (locally (declare (optimize (speed 3) #.*no-array-bounds-check*))
        (dotimes (i (length weight-deltas))
          (decf (aref weight-deltas i) weight-penalty))))
    (with-shape-and-displacement (weight-deltas)
      (with-shape-and-displacement (accumulator)
        (do-segment-set (segment :start-in-segment-set start-in-segment-set)
                        (segment-set trainer)
          (let ((weights (segment-weights segment)))
            (reshape-and-displace! weight-deltas (mat-size weights)
                                   start-in-segment-set)
            (reshape-and-displace! accumulator (mat-size weights)
                                   start-in-segment-set)
            (unless (zerop weight-decay)
              (mgl-mat:axpy! weight-decay weights weight-deltas)
              (mgl-mat:scal! (- 1 (* learning-rate weight-decay)) weights))
            #+nil
            (format t "~A: norm: ~S~%" (name segment)
                    (mgl-bp::norm weight-deltas))
            (mgl-mat:axpy! (- (/ learning-rate n-inputs)) accumulator weights)
            (mgl-mat:axpy! (- (* learning-rate momentum)) weight-deltas
                           weights)
            (unless (zerop weight-penalty)
              (add-sign! (* learning-rate weight-penalty) weights 0 accumulator)
              (axpy! 1 accumulator weights))))))
    (fill! (flt 0) accumulator)
    (setf (n-inputs-in-batch trainer) 0))
  (map nil #'funcall (after-update-hook trainer)))

(defun update-all-weights/normal (trainer)
  (map nil #'funcall (before-update-hook trainer))
  (let ((accumulator (accumulator trainer))
        (weight-deltas (weight-deltas trainer))
        (learning-rate (learning-rate trainer))
        (n-inputs (flt (n-inputs-in-batch trainer)))
        (momentum (momentum trainer))
        (weight-decay (weight-decay trainer))
        (weight-penalty (weight-penalty trainer)))
    (declare (type flt learning-rate n-inputs momentum
                   weight-decay #+nil weight-penalty))
    (mgl-mat:scal! momentum weight-deltas)
    (mgl-mat:axpy! (flt (/ n-inputs)) accumulator weight-deltas)
    (with-shape-and-displacement (weight-deltas)
      (do-segment-set (segment :start-in-segment-set start-in-segment-set)
                      (segment-set trainer)
        (let ((weights (segment-weights segment)))
          (reshape-and-displace! weight-deltas (mat-size weights)
                                 start-in-segment-set)
          (unless (zerop weight-penalty)
            (add-sign! weight-penalty weights 1 weight-deltas))
          (unless (zerop weight-decay)
            (axpy! weight-decay weights weight-deltas))
          (mgl-mat:axpy! (- learning-rate) weight-deltas weights))))
    (fill! (flt 0) accumulator)
    (setf (n-inputs-in-batch trainer) 0))
  (map nil #'funcall (after-update-hook trainer)))


;;;; NORMALIZED-BATCH-GD-TRAINER

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

(defun set-up-n-weight-uses (trainer)
  (let ((n-weights (segment-set-size (segment-set trainer))))
    (setf (n-weight-uses-in-batch trainer)
          (make-array n-weights :element-type 'index :initial-element 0))))

(defmethod initialize-gradient-sink ((trainer normalized-batch-gd-trainer)
                                     source segmentable)
  (call-next-method)
  (set-up-n-weight-uses trainer))

(defmethod n-inputs-until-update ((trainer normalized-batch-gd-trainer))
  ;; Weights are updated as in BATCH-GD-TRAINER but we need to collect
  ;; weight usage statistics after each example.
  1)

(defmethod maybe-update-weights ((trainer normalized-batch-gd-trainer)
                                 n-new-inputs)
  (declare (type index n-new-inputs))
  (let ((accumulator (accumulator trainer))
        (n-weight-uses-in-batch (n-weight-uses-in-batch trainer))
        (weight-deltas (weight-deltas trainer))
        (learning-rate (learning-rate trainer))
        (momentum (momentum trainer))
        (weight-decay (weight-decay trainer))
        (weight-penalty (weight-penalty trainer))
        (batch-size (batch-size trainer)))
    (declare (type index-vector n-weight-uses-in-batch)
             (type flt learning-rate momentum weight-decay weight-penalty)
             (type index batch-size))
    (with-facets ((accumulator (accumulator 'array :direction :io
                                            :type flt-vector))
                  (weight-deltas (weight-deltas 'array :direction :io
                                                :type flt-vector)))
      (do-segment-set (segment :start-in-segment-set start-in-segment-set)
                      (segment-set trainer)
        (map-segment-runs
         (lambda (start end)
           (declare (type index start end)
                    (optimize (speed 3) #.*no-array-bounds-check*))
           (do ((i (the! index (+ start-in-segment-set start))
                   (the! index (1+ i)))
                (j start (1+ j)))
               ((<= end j))
             (setf (aref n-weight-uses-in-batch i)
                   (the! index
                         (+ n-new-inputs
                            (the! index
                                  (aref n-weight-uses-in-batch i)))))))
         segment))
      (when (<= batch-size (the index (incf (n-inputs-in-batch trainer)
                                            n-new-inputs)))
        (setf (n-inputs-in-batch trainer) 0)
        (do-segment-set (segment :start-in-segment-set start-in-segment-set)
                        (segment-set trainer)
          (let* ((weights (segment-weights segment))
                 (start (mat-displacement weights))
                 (end (+ start (the index (mat-size weights)))))
            (declare (optimize (speed 3) #.*no-array-bounds-check*)
                     (type index start end))
            (with-facets ((weights (weights 'backing-array :direction :io
                                            :type flt-vector)))
              (do ((i start-in-segment-set (the! index (1+ i)))
                   (j start (1+ j)))
                  ((<= end j))
                (let ((delta (+ (* momentum (aref weight-deltas i))
                                (* (if (zerop (aref n-weight-uses-in-batch i))
                                       #.(flt 0)
                                       (/ (flt
                                           (aref n-weight-uses-in-batch i))))
                                   (aref accumulator i))
                                (* weight-decay (aref weights j))
                                weight-penalty)))
                  (setf (aref weight-deltas i) delta)
                  (decf (aref weights j) (* learning-rate delta))
                  (setf (aref n-weight-uses-in-batch i) 0
                        (aref accumulator i) #.(flt 0)))))))
        (map nil #'funcall (after-update-hook trainer)))))
  (incf (n-inputs trainer) n-new-inputs))


;;;; PER-WEIGHT-BATCH-GD-TRAINER

(defclass per-weight-batch-gd-trainer (gd-trainer)
  ((n-weight-uses-in-batch
    :accessor n-weight-uses-in-batch
    :documentation "Number of uses of the weight in its current batch."))
  (:documentation "This is much like BATCH-GD-TRAINER but it is more
clever about when to update weights. Basically every weight has its
own batch independent from the batches of others. It has desirable
properties. One can for example put two neural networks together
without adding any connections between them and the learning will
produce results equivalent to the separated case. Also, adding inputs
with only missing values does not change anything."))

(defmethod initialize-gradient-sink ((trainer per-weight-batch-gd-trainer)
                                     source segmentable)
  (call-next-method)
  (set-up-n-weight-uses trainer))

(defmethod n-inputs-until-update ((trainer per-weight-batch-gd-trainer))
  ;; Weight updates are async, don't overpromise.
  1)

(defmethod maybe-update-weights ((trainer per-weight-batch-gd-trainer)
                                 n-new-inputs)
  (assert (= 1 n-new-inputs))
  (let ((accumulator (accumulator trainer))
        (n-weight-uses-in-batch (n-weight-uses-in-batch trainer))
        (weight-deltas (weight-deltas trainer))
        (learning-rate (learning-rate trainer))
        (momentum (momentum trainer))
        (weight-decay (weight-decay trainer))
        (weight-penalty (weight-penalty trainer))
        (batch-size (batch-size trainer)))
    (declare (type index-vector n-weight-uses-in-batch)
             (type flt learning-rate momentum weight-decay weight-penalty)
             (type index batch-size))
    (with-facets ((accumulator (accumulator 'array :direction :io
                                            :type flt-vector))
                  (weight-deltas (weight-deltas 'array :direction :io
                                                :type flt-vector)))
      (declare (optimize (speed 3) #.*no-array-bounds-check*))
      (do-segment-set (segment :start-in-segment-set start-in-segment-set)
                      (segment-set trainer)
        (let ((weights (segment-weights segment)))
          (with-facets ((weights (weights 'array :direction :io
                                          :type flt-vector)))
            (map-segment-runs
             (lambda (start end)
               (declare (type index start end))
               (do ((i (the! index (+ start-in-segment-set start))
                       (the! index (1+ i)))
                    (j start (1+ j)))
                   ((<= end j))
                 (when (<= batch-size
                           (setf (aref n-weight-uses-in-batch i)
                                 (1+ (the! index
                                           (aref n-weight-uses-in-batch i)))))
                   (let ((delta (+ (* momentum (aref weight-deltas i))
                                   (/ (aref accumulator i)
                                      (aref n-weight-uses-in-batch i))
                                   (* weight-decay (aref weights j))
                                   weight-penalty)))
                     (setf (aref weight-deltas i) delta)
                     (decf (aref weights j) (* learning-rate delta))
                     (setf (aref n-weight-uses-in-batch i) 0
                           (aref accumulator i) #.(flt 0))))))
             segment)))))
    (map nil #'funcall (after-update-hook trainer)))
  (incf (n-inputs trainer)))


;;;; SEGMENTED-GD-TRAINER

(defclass segmented-gd-trainer (gradient-sink)
  ((n-inputs :initform 0 :initarg :n-inputs :accessor n-inputs)
   (segmenter
    :initarg :segmenter :accessor segmenter
    :documentation "When this trainer is initialized it loops over the
segment of the learner with MAP-SEGMENTS. SEGMENTER is a function that
is called with each segment and returns a trainer or NIL. Several
segments may be mapped to the same trainer. After the segment->trainer
mappings are collected, each trainer is initialized by
INITIALIZE-GRADIENT-SINK with the list segments mapped to it.")
   (trainers :type list :reader trainers)
   (segments :type list :reader segments))
  (:documentation "A trainer that delegates training of segments to
other trainers. Useful to delegate training of different segments to
different trainers (capable of working with segmantables) or simply to
not train all segments."))

(define-descriptions (trainer segmented-gd-trainer)
  n-inputs trainers segments)

(defmethod describe-object :after ((trainer segmented-gd-trainer) stream)
  (when (slot-boundp trainer 'trainers)
    (terpri stream)
    (dolist (trainer (trainers trainer))
      (describe trainer stream))))

(defmethod initialize-gradient-sink ((trainer segmented-gd-trainer) source
                                     segmentable)
  (when (next-method-p)
    (call-next-method))
  (let ((segmenter (segmenter trainer))
        (trainer-segments (make-hash-table :test 'eq)))
    (map-segments (lambda (segment)
                    (let ((trainer (funcall segmenter segment)))
                      (when trainer
                        (unless (gethash trainer trainer-segments)
                          (setf (gethash trainer trainer-segments)
                                nil))
                        (push segment (gethash trainer trainer-segments)))))
                  segmentable)
    (let ((trainers ()))
      (maphash (lambda (trainer segments)
                 (initialize-gradient-sink trainer segments segments)
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
  (if (trainers trainer)
      (loop for child-trainer in (trainers trainer)
            minimizing (n-inputs-until-update child-trainer))
      nil))

(defmethod map-gradient-sink (fn (trainer segmented-gd-trainer))
  (dolist (trainer (trainers trainer))
    (map-gradient-sink fn trainer)))
