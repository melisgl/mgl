(in-package :mgl-example-util)

(defun time->string (&optional (time (get-universal-time)))
  (destructuring-bind (second minute hour date month year)
      (subseq (multiple-value-list (decode-universal-time time)) 0 6)
    (format nil "~4,'0D-~2,'0D-~2,'0D ~2,'0D:~2,'0D:~2,'0D"
            year month date hour minute second)))

(defparameter *example-dir*
  (merge-pathnames (make-pathname :directory '(:relative "example"))
                   (make-pathname :name nil :type nil
                                  :defaults (asdf:component-pathname
                                             (asdf:find-system :mgl-example)))))

(defparameter *log-file*
  (merge-pathnames "mgl-example.log" *example-dir*))

(defun log-msg (format &rest args)
  (format *trace-output* "~&~A: " (time->string))
  (apply #'format *trace-output* format args)
  (with-open-file (s *log-file* :direction :output
                   :if-exists :append :if-does-not-exist :create)
    (format s "~A: " (time->string))
    (apply #'format s format args)))


;;;; Logging trainer

(defgeneric log-training-error (trainer learner))
(defgeneric log-test-error (trainer learner))
(defgeneric log-training-period (trainer learner))
(defgeneric log-test-period (trainer learner))

(defclass logging-trainer ()
  ((log-training-fn :initform (make-instance 'periodic-fn
                                             :period 'log-training-period
                                             :fn 'log-training-error)
                    :reader log-training-fn)
   (log-test-fn :initform (make-instance 'periodic-fn
                                         :period 'log-test-period
                                         :fn 'log-test-error)
                :reader log-test-fn)))

(defmethod train-batch :after (samples (trainer logging-trainer) learner)
  (call-periodic-fn (n-inputs trainer)
                    (log-training-fn trainer)
                    trainer learner)
  (call-periodic-fn (n-inputs trainer)
                    (log-test-fn trainer)
                    trainer learner))

(defmethod train :before (sampler (trainer logging-trainer) learner)
  (setf (last-eval (log-training-fn trainer))
        (n-inputs trainer))
  (call-periodic-fn! (n-inputs trainer)
                     (log-test-fn trainer) trainer learner))

(defmethod train :after (sampler (trainer logging-trainer) learner)
  (call-periodic-fn! (n-inputs trainer)
                     (log-training-fn trainer)
                     trainer learner)
  (call-periodic-fn! (n-inputs trainer)
                     (log-test-fn trainer)
                     trainer learner))


;;;; Misc

(defun tack-cross-entropy-softmax-error-on (n-classes lump-name &key
                                            (prefix '||))
  (flet ((foo (symbol)
           (intern (format nil "~A~A" prefix (symbol-name symbol))
                   (symbol-package prefix))))
    `((,(foo 'expectations) (input-lump :size ,n-classes))
      (,(foo 'prediction-weights) (weight-lump
                                   :size (* (size (lump ',lump-name))
                                            ,n-classes)))
      (,(foo 'prediction-biases) (weight-lump :size ,n-classes))
      (,(foo 'prediction-activations0)
       (activation-lump :weights ,(foo 'prediction-weights)
        :x (lump ',lump-name)
        :transpose-weights-p t))
      (,(foo 'prediction-activations)
       (->+ :args (list ,(foo 'prediction-activations0)
                        ,(foo 'prediction-biases))))
      (,(foo 'predictions)
       (cross-entropy-softmax-lump
        :group-size ,n-classes
        :x ,(foo 'prediction-activations)
        :target ,(foo 'expectations)))
      (,(foo 'ce-error) (error-node :x ,(foo 'predictions))))))

(defun load-weights (filename obj)
  (with-open-file (stream filename)
    (mgl-util:read-weights obj stream)))

(defun save-weights (filename obj)
  (with-open-file (stream filename :direction :output
                   :if-does-not-exist :create :if-exists :supersede)
    (mgl-util:write-weights obj stream)))

