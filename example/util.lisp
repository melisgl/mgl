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
  (pprint-logical-block (*trace-output* nil)
    (format *trace-output* "~A: " (time->string))
    (pprint-logical-block (*trace-output* nil)
      (apply #'format *trace-output* format args)))
  (with-open-file (s *log-file* :direction :output
                   :if-exists :append :if-does-not-exist :create)
    (pprint-logical-block (s nil)
      (format s "~A: " (time->string))
      (pprint-logical-block (s nil)
        (apply #'format s format args)))))


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


;;;; BASE-TRAINER

(defclass base-trainer (logging-trainer)
  ((training-counters-and-measurers :reader training-counters-and-measurers)))

(defmethod log-test-error ((trainer base-trainer) learner)
  (let ((*print-level* nil))
    (when (zerop (n-inputs trainer))
      (describe learner))
    (describe trainer))
  (log-msg "n-inputs: ~S~%" (n-inputs trainer)))

(defmethod log-training-error ((trainer base-trainer) learner)
  (log-msg "n-inputs: ~S~%"  (n-inputs trainer))
  (dolist (counter-and-measurer (training-counters-and-measurers trainer))
    (let ((counter (car counter-and-measurer)))
      (log-msg "~A~%" counter)
      (reset-counter counter))))

(defmethod negative-phase :around (batch (trainer base-trainer) (bm bm))
  (call-next-method)
  (apply-counters-and-measurers (training-counters-and-measurers trainer)
                                batch bm))

(defmethod positive-phase :around (batch (trainer base-trainer) (dbm dbm))
  (call-next-method)
  (set-visible-mean dbm)
  (apply-counters-and-measurers (training-counters-and-measurers trainer)
                                batch dbm))

(defmethod compute-derivatives :around (samples (trainer base-trainer) (bpn bpn))
  (call-next-method)
  (apply-counters-and-measurers (training-counters-and-measurers trainer)
                                samples bpn))

(defmethod train-batch :around (batch (trainer base-trainer) (bpn bpn))
  (if (typep trainer 'cg-bp-trainer)
      (let ((result (multiple-value-list (call-next-method))))
        (destructuring-bind (best-w best-f n-line-searches
                                    n-succesful-line-searches n-evaluations)
            result
          (declare (ignore best-w))
          (log-msg "best-f: ~,5E, ~:_n-evaluations: ~S~%" best-f n-evaluations)
          (log-msg "n-line-searches: ~S (succesful ~S)~%"
                   n-line-searches n-succesful-line-searches))
        result)
      (call-next-method)))

(defun prepend-name-to-counters (name counters-and-measurers)
  (dolist (counter-and-measurer counters-and-measurers counters-and-measurers)
    (push name (slot-value (car counter-and-measurer) 'name))))


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

(defun maximally-likely-node (striped stripe &key (nodes (nodes striped)))
  (with-stripes ((stripe striped start end))
    (- (max-position (storage nodes) start end)
       start)))

(defun maximally-likely-in-cross-entropy-softmax-lump (lump stripe)
  (values (maximally-likely-node lump stripe :nodes (softmax lump))
          (maximally-likely-node (target lump) stripe)))

(defun cross-entropy-softmax-max-likelihood-classification-error (lump)
  "A measurer that return the number of misclassifications."
  (values (loop for stripe below (n-stripes lump)
                count (multiple-value-bind (prediction target)
                          (maximally-likely-in-cross-entropy-softmax-lump
                           lump stripe)
                        (/= prediction target)))
          (n-stripes lump)))
