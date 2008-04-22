(in-package :mgl-rbm)

(defclass dbn ()
  ((rbms :type list :initarg :rbms :accessor rbms)))

(defun add-rbm (rbm dbn)
  (setf (slot-value dbn 'rbms)
        (append (rbms dbn) (list rbm))))

;;; Copy the visible means over the visible samples in those chunks
;;; that come from RBM-BELOW or all if RBM-BELOW is NIL.
(defun copy-visible-means-to-samples (rbm rbm-below)
  (let ((chunks-below (if rbm-below
                          (hidden-chunks rbm-below)
                          nil)))
    (map nil (lambda (chunk)
               (when (or (null rbm-below) (find chunk chunks-below))
                 (locally (declare (optimize (speed 3)))
                   (replace (the flt-vector (samples chunk))
                            (the flt-vector (means chunk))))))
         (visible-chunks rbm))))

(defun up-mean-field (dbn &key (to-rbm (last1 (rbms dbn)))
                      (from-samples/means :samples)
                      exclude-to
                      (to-samples/means :samples))
  "Propagate the means upwards from the bottom rbm up to including
TO-RBM (if specified). Start from the SAMPLES of MEANS of the bottom
rbm according to FROM-SAMPLES/MEANS. In the rest of the rbms passed
along the way (always including TO-RBM) place the result in the
visible layer according to TO-SAMPLES/MEANS."
  (let ((rbms (let ((rbms (rbms dbn)))
                (subseq rbms 0 (+ (if exclude-to 0 1)
                                  (position to-rbm rbms))))))
    (when rbms
      (destructuring-bind (bottom &rest bottom->top) rbms
        (set-hidden-mean bottom from-samples/means)
        (loop for rbm-below = bottom then rbm
              for rbm in bottom->top
              do (when (eq :samples to-samples/means)
                   (copy-visible-means-to-samples rbm rbm-below))
              (set-hidden-mean rbm :means))))
    (when (and exclude-to
               (eq :samples to-samples/means)
               ;; Punt if we did nothing and the data is in the
               ;; samples.
               (not (and (eq :samples from-samples/means)
                         (endp rbms))))
      (copy-visible-means-to-samples to-rbm (last1 rbms)))))

(defun down-mean-field (dbn &key from-rbm (from-samples/means :means))
  "Propagate the means down from the means of the top rbm or FROM-RBM."
  (destructuring-bind (top &rest top->bottom)
      (let ((rbms (reverse (rbms dbn))))
        (if from-rbm
            (subseq rbms (position from-rbm rbms))
            rbms))
    (set-visible-mean top from-samples/means)
    (loop for rbm in top->bottom
          do (set-visible-mean rbm :means))))

(defun reconstruct-mean-field (dbn &key to-rbm (from-samples/means :samples))
  (up-mean-field dbn :to-rbm to-rbm :from-samples/means from-samples/means
                 :to-samples/means :means)
  (down-mean-field dbn :from-rbm to-rbm :from-samples/means :means))

(defun dbn-rmse (sampler dbn rbm)
  (let* ((n (1+ (position rbm (rbms dbn))))
         (sum-errors (make-array n :initial-element #.(flt 0)))
         (n-errors (make-array n :initial-element 0)))
    (loop until (finishedp sampler) do
          (set-input (sample sampler) rbm)
          (set-hidden-mean rbm :samples)
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
