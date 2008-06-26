(in-package :mgl-rbm)

(defclass dbn ()
  ((rbms :type list :initarg :rbms :accessor rbms)))

(defun add-rbm (rbm dbn)
  (setf (slot-value dbn 'rbms)
        (append (rbms dbn) (list rbm))))

(defun up-mean-field (dbn &key (to-rbm (last1 (rbms dbn))) exclude-to)
  "Propagate the means upwards from the bottom rbm up to TO-RBM
\(including it unless EXCLUDE-TO)."
  (loop for (rbm next-rbm) on (rbms dbn)
        while (or (not exclude-to) (not (eq rbm to-rbm)))
        do (set-hidden-mean rbm)
        (when next-rbm
          (nodes->inputs next-rbm))
        while (not (eq rbm to-rbm))))

(defun not-before (list obj)
  (let ((pos (position obj list)))
    (if pos
        (subseq list pos)
        list)))

(defun down-mean-field (dbn &key from-rbm)
  "Propagate the means down from the means of the top rbm or FROM-RBM."
  (mapc #'set-visible-mean
        (not-before (reverse (rbms dbn)) from-rbm)))

(defun dbn-rmse (sampler dbn &key (rbm (last1 (rbms dbn))))
  (let* ((n (1+ (position rbm (rbms dbn))))
         (sum-errors (make-array n :initial-element #.(flt 0)))
         (n-errors (make-array n :initial-element 0)))
    (loop until (finishedp sampler) do
          (set-input (sample sampler) rbm)
          (set-hidden-mean rbm)
          (down-mean-field dbn :from-rbm rbm)
          (loop for i below n
                for rbm in (rbms dbn) do
                (multiple-value-bind (e n) (layer-error (visible-chunks rbm))
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
