(in-package :mgl-rbm)

(defclass dbn ()
  ((rbms :type list :initarg :rbms :reader rbms)
   (max-n-stripes :initform 1 :initarg :max-n-stripes :reader max-n-stripes))
  (:documentation "Deep Belief Network: a stack of RBMs."))

(defmethod n-stripes ((dbn dbn))
  (n-stripes (first (rbms dbn))))

(defmethod set-n-stripes ((dbn dbn) n-stripes)
  (dolist (rbm (rbms dbn))
    (setf (n-stripes rbm) n-stripes)))

(defmethod set-max-n-stripes (max-n-stripes (dbn dbn))
  (dolist (rbm (rbms dbn))
    (setf (max-n-stripes rbm) max-n-stripes)))

(defmethod initialize-instance :after ((dbn dbn) &key &allow-other-keys)
  (dolist (rbm (rbms dbn))
    (setf (slot-value rbm 'dbn) dbn))
  ;; make sure rbms have the same MAX-N-STRIPES
  (setf (max-n-stripes dbn) (max-n-stripes dbn)))

(defun add-rbm (rbm dbn)
  (setf (slot-value rbm 'dbn) dbn
        (slot-value dbn 'rbms) (append1 (rbms dbn) rbm))
  (setf (max-n-stripes rbm) (max-n-stripes dbn)))

(defun previous-rbm (dbn rbm)
  (let ((pos (position rbm (rbms dbn))))
    (if (and pos (plusp pos))
        (elt (rbms dbn) (1- pos))
        nil)))

(defmethod set-input :around (samples (rbm rbm))
  ;; Do SET-INPUT on the previous rbm (if any) and propagate its mean
  ;; to this one.
  (when (dbn rbm)
    (let ((prev (previous-rbm (dbn rbm) rbm)))
      (when prev
        (set-input samples prev)
        (set-hidden-mean prev))))
  (setf (n-stripes rbm) (length samples))
  (unwind-protect
       ;; Do any clamping specific to this RBM.
       (call-next-method)
    ;; Then remember the inputs.
    (nodes->inputs rbm)))

(defmethod set-input (samples (dbn dbn))
  (set-input samples (last1 (rbms dbn))))

(defun not-before (list obj)
  (let ((pos (position obj list)))
    (if pos
        (subseq list pos)
        list)))

(defun down-mean-field (dbn &key (rbm (last1 (rbms dbn))))
  "Propagate the means down from the means of RBM."
  (mapc #'set-visible-mean
        (not-before (reverse (rbms dbn)) rbm)))

(defun dbn-rmse (sampler dbn &key (rbm (last1 (rbms dbn))))
  (let* ((n (1+ (position rbm (rbms dbn))))
         (sum-errors (make-array n :initial-element #.(flt 0)))
         (n-errors (make-array n :initial-element 0))
         (max-n-stripes (max-n-stripes dbn)))
    (loop until (finishedp sampler) do
          (set-input (sample-batch sampler max-n-stripes) rbm)
          (set-hidden-mean rbm)
          (down-mean-field dbn :rbm rbm)
          (loop for i below n
                for rbm in (rbms dbn) do
                (multiple-value-bind (e n) (reconstruction-error rbm)
                  (incf (aref sum-errors i) e)
                  (incf (aref n-errors i) n))))
    (map 'vector
         (lambda (sum-errors n-errors)
           (if (zerop n-errors)
               nil
               (sqrt (/ sum-errors n-errors))))
         sum-errors n-errors)))

(defmethod write-weights ((dbn dbn) stream)
  (dolist (rbm (rbms dbn))
    (write-weights rbm stream)))

(defmethod read-weights ((dbn dbn) stream)
  (dolist (rbm (rbms dbn))
    (read-weights rbm stream)))
