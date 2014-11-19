(cl:defpackage :mgl-example-sum-sign-rnn
  (:use #:common-lisp #:mgl))

(in-package :mgl-example-sum-sign-rnn)

;;; There is a single input at each time step...
(defparameter *n-inputs* 1)
;;; and we want to learn the rule that outputs the sign of the sum of
;;; inputs so far in the sequence.
(defparameter *n-outputs* 3)

;;; Generate a training example that's a sequence of random length
;;; between 1 and 11. Elements of the sequence are lists of two
;;; elements:
;;;
;;; 1. The input for the network (a single random number).
;;;
;;; 2. The sign of the sum of inputs so far encoded as 0, 1, 2 (for
;;;    negative, zero and positive values). To add a twist, the sum is
;;;    reset whenever a negative input is seen.
(defun make-sum-sign-instance ()
  (let ((length (1+ (random 10)))
        (sum 0))
    (loop for i below length
          collect (let ((x (1- (* 2 (random 2)))))
                    (incf sum x)
                    (when (< x 0)
                      (setq sum x))
                    (list x (cond ((minusp sum) 0)
                                  ((zerop sum) 1)
                                  (t 2)))))))

;;; Build an RNN with a single lstm hidden layer and softmax output.
;;; For each time step, a SUM-SIGN-FNN will be instantiated.
(defun make-sum-sign-rnn (&key (n-hiddens 1))
  (build-rnn ()
    (build-fnn (:class 'sum-sign-fnn)
      (input (->input :size 1))
      (h (->lstm :name 'h :inputs (list input) :n-cells n-hiddens))
      (prediction (->softmax-xe-loss
                   :x (->activation :name 'prediction
                                    :size *n-outputs*
                                    :inputs (list h)))))))

;;; We define this class to be able to specialize how inputs are
;;; translated by adding a SET-INPUT method later.
(defclass sum-sign-fnn (fnn)
  ())

;;; We have a batch of instances from MAKE-SUM-SIGN-INSTANCE for the
;;; RNN. This function is invoked with elements of these instances
;;; belonging to the same time step (i.e. at the same index) and sets
;;; the input and target up.
(defmethod set-input (instances (fnn sum-sign-fnn))
  (let ((input-nodes (nodes (find-clump 'input fnn))))
    (setf (target (find-clump 'prediction fnn))
          (loop for stripe upfrom 0
                for instance in instances
                collect
                ;; Sequences in the batch are not of equal length. The
                ;; RNN sends a NIL our way if a sequence has run out.
                (when instance
                  (destructuring-bind (input target) instance
                    (setf (mref input-nodes stripe 0) input)
                    target))))))

;;; Train the network by minimizing the loss (cross-entropy here) with
;;; the Adam optimizer.
(defun train-sum-sign-rnn ()
  (let ((rnn (make-sum-sign-rnn)))
    (setf (max-n-stripes rnn) 50)
    ;; Initialize the weights in the usual sqrt(1 / fan-in) style.
    (map-segments (lambda (weights)
                    (let* ((fan-in (mat-dimension (nodes weights) 0))
                           (limit (sqrt (/ 6 fan-in))))
                      (uniform-random! (nodes weights)
                                       :limit (* 2 limit))
                      (.+! (- limit) (nodes weights))))
                  rnn)
    (minimize (monitor-optimization-periodically
               (make-instance 'adam-optimizer
                              :learning-rate 0.2
                              :batch-size 100)
               '((:fn log-test-error :period 30000)
                 (:fn reset-optimization-monitors :period 3000)))
              (make-instance 'bp-learner
                             :bpn rnn
                             :monitors (make-bpn-cost-monitors))
              :dataset (make-sampler 30000))))

;;; Return a sampler object that produces MAX-N-SAMPLES number of
;;; random inputs.
(defun make-sampler (max-n-samples)
  (make-instance 'function-sampler :max-n-samples max-n-samples
                 :generator #'make-sum-sign-instance))

;;; Log the test error. Also, describe the optimizer and the bpn at
;;; the beginning of training. Called periodically during training
;;; (see above).
(defun log-test-error (optimizer learner)
  (when (zerop (n-instances optimizer))
    (describe optimizer)
    (describe (bpn learner)))
  (log-padded (monitor-bpn-results (make-sampler 1000) (bpn learner)
                                   (make-bpn-cost-monitors :dataset "pred."))))

;;; Return a list of monitor objects (yes, only one here) that will
;;; measure the cost and accumulate it in a counter.
(defun make-bpn-cost-monitors (&key (dataset "train"))
  (list (make-instance 'monitor
                       :measurer (lambda (instances bpn)
                                   (declare (ignore instances))
                                   (cost bpn))
                       :counter (make-instance 'basic-counter
                                               :attributes `(:dataset ,dataset
                                                             :type "cost")))))

#|

;;; Transcript follows:
(train-sum-sign-rnn)
.. 2015-01-19 22:27:01: training at n-instances: 0
.. 2015-01-19 22:27:01: train cost: 0.000e+0 (0)
.. #<ADAM-OPTIMIZER {101F9578A3}>
..  GD-OPTIMIZER description:
..    N-INSTANCES = 0
..    SEGMENT-SET = #<SEGMENT-SET
..                    (#<->WEIGHT (H #) :SIZE 1 1/1 :norm 0.50959>
..                     #<->WEIGHT (H #) :SIZE 1 1/1 :norm 1.48846>
..                     #<->WEIGHT (#1=# #2=# :PEEPHOLE) :SIZE
..                       1 1/1 :norm 2.07815>
..                     #<->WEIGHT (H #2#) :SIZE 1 1/1 :norm 2.31284>
..                     #<->WEIGHT (#1# #3=# :PEEPHOLE) :SIZE
..                       1 1/1 :norm 0.67760>
..                     #<->WEIGHT (H #3#) :SIZE 1 1/1 :norm 0.11992>
..                     #<->WEIGHT (H PREDICTION) :SIZE
..                       3 1/1 :norm 2.43519>
..                     #<->WEIGHT (:BIAS PREDICTION) :SIZE
..                       3 1/1 :norm 1.64322>
..                     #<->WEIGHT (#1# #4=# :PEEPHOLE) :SIZE
..                       1 1/1 :norm 1.23665>
..                     #<->WEIGHT (INPUT #4#) :SIZE 1 1/1 :norm 1.24288>
..                     #<->WEIGHT (:BIAS #4#) :SIZE 1 1/1 :norm 0.43173>
..                     #<->WEIGHT (INPUT #1#) :SIZE 1 1/1 :norm 1.03770>
..                     #<->WEIGHT (:BIAS #1#) :SIZE 1 1/1 :norm 0.28466>
..                     #<->WEIGHT (INPUT #5=#) :SIZE 1 1/1 :norm 2.43322>
..                     #<->WEIGHT (:BIAS #5#) :SIZE 1 1/1 :norm 1.11231>
..                     #<->WEIGHT (INPUT #6=#) :SIZE 1 1/1 :norm 0.22344>
..                     #<->WEIGHT (:BIAS #6#) :SIZE 1 1/1 :norm 1.44019>)
..                    {101F9580F3}>
..    LEARNING-RATE = 2.00000e-1
..    MOMENTUM = 0.00000e+0
..    MOMENTUM-TYPE = :NORMAL
..    WEIGHT-DECAY = 0.00000e+0
..    WEIGHT-PENALTY = 0.00000e+0
..    N-AFTER-UPATE-HOOK = 0
..    BATCH-SIZE = 100
..  
..  BATCH-GD-OPTIMIZER description:
..    N-BEFORE-UPATE-HOOK = 0
..  
..  ADAM-OPTIMIZER description:
..    MEAN-UPDATE-RATE = 1.00000e-1
..    VARIANCE-UPDATE-RATE = 1.00000e-3
..    VARIANCE-ADJUSTMENT = 1.00000e-8
..  #<RNN {101F915923}>
..   BPN description:
..     CLUMPS = #(#<SUM-SIGN-FNN :STRIPES 1/50 :CLUMPS 4 {101F915F63}>
..                #<SUM-SIGN-FNN :STRIPES 1/50 :CLUMPS 4 {101F938893}>)
..     N-STRIPES = 1
..     MAX-N-STRIPES = 50
..   
..   RNN description:
..     MAX-LAG = 1
..   2015-01-19 22:27:01: pred. cost: 1.088d+0 (5472.00)
.. 2015-01-19 22:27:01: training at n-instances: 3000
.. 2015-01-19 22:27:01: train cost: 4.186d-1 (16481.00)
.. 2015-01-19 22:27:02: training at n-instances: 6000
.. 2015-01-19 22:27:02: train cost: 5.619d-2 (16505.00)
.. 2015-01-19 22:27:02: training at n-instances: 9000
.. 2015-01-19 22:27:02: train cost: 2.512d-2 (16643.00)
.. 2015-01-19 22:27:03: training at n-instances: 12000
.. 2015-01-19 22:27:03: train cost: 1.585d-2 (16946.00)
.. 2015-01-19 22:27:03: training at n-instances: 15000
.. 2015-01-19 22:27:03: train cost: 1.135d-2 (16240.00)
.. 2015-01-19 22:27:04: training at n-instances: 18000
.. 2015-01-19 22:27:04: train cost: 8.704d-3 (16622.00)
.. 2015-01-19 22:27:04: training at n-instances: 21000
.. 2015-01-19 22:27:04: train cost: 6.912d-3 (16690.00)
.. 2015-01-19 22:27:05: training at n-instances: 24000
.. 2015-01-19 22:27:05: train cost: 5.627d-3 (16528.00)
.. 2015-01-19 22:27:05: training at n-instances: 27000
.. 2015-01-19 22:27:05: train cost: 4.735d-3 (16427.00)
.. 2015-01-19 22:27:06: training at n-instances: 30000
.. 2015-01-19 22:27:06: train cost: 4.006d-3 (16453.00)
.. 2015-01-19 22:27:06: pred. cost: 3.698d-3 (5347.00)
..
==> (#<->WEIGHT (H (H :OUTPUT)) :SIZE 1 1/1 :norm 1.57197>
-->  #<->WEIGHT (H (H :CELL)) :SIZE 1 1/1 :norm 0.98579>
-->  #<->WEIGHT ((H :CELL) (H :FORGET) :PEEPHOLE) :SIZE 1 1/1 :norm 0.25004>
-->  #<->WEIGHT (H (H :FORGET)) :SIZE 1 1/1 :norm 5.06433>
-->  #<->WEIGHT ((H :CELL) (H :INPUT) :PEEPHOLE) :SIZE 1 1/1 :norm 4.45586>
-->  #<->WEIGHT (H (H :INPUT)) :SIZE 1 1/1 :norm 3.14358>
-->  #<->WEIGHT (H PREDICTION) :SIZE 3 1/1 :norm 21.68755>
-->  #<->WEIGHT (:BIAS PREDICTION) :SIZE 3 1/1 :norm 4.38377>
-->  #<->WEIGHT ((H :CELL) (H :OUTPUT) :PEEPHOLE) :SIZE 1 1/1 :norm 3.22046>
-->  #<->WEIGHT (INPUT (H :OUTPUT)) :SIZE 1 1/1 :norm 3.07891>
-->  #<->WEIGHT (:BIAS (H :OUTPUT)) :SIZE 1 1/1 :norm 5.64394>
-->  #<->WEIGHT (INPUT (H :CELL)) :SIZE 1 1/1 :norm 4.79663>
-->  #<->WEIGHT (:BIAS (H :CELL)) :SIZE 1 1/1 :norm 0.42101>
-->  #<->WEIGHT (INPUT (H :FORGET)) :SIZE 1 1/1 :norm 3.62612>
-->  #<->WEIGHT (:BIAS (H :FORGET)) :SIZE 1 1/1 :norm 4.56776>
-->  #<->WEIGHT (INPUT (H :INPUT)) :SIZE 1 1/1 :norm 1.44047>
-->  #<->WEIGHT (:BIAS (H :INPUT)) :SIZE 1 1/1 :norm 4.70072>)

|#
