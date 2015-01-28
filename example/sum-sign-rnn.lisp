(cl:defpackage :mgl-example-sum-sign-rnn
  (:use #:common-lisp #:mgl))

(in-package :mgl-example-sum-sign-rnn)

;;; There is a single input at each time step...
(defparameter *n-inputs* 1)
;;; and we want to learn the rule that outputs the sign of the sum of
;;; inputs so far in the sequence.
(defparameter *n-outputs* 3)

;;; Generate a training example that's a sequence of random length
;;; between 1 and LENGTH. Elements of the sequence are lists of two
;;; elements:
;;;
;;; 1. The input for the network (a single random number).
;;;
;;; 2. The sign of the sum of inputs so far encoded as 0, 1, 2 (for
;;;    negative, zero and positive values). To add a twist, the sum is
;;;    reset whenever a negative input is seen.
(defun make-sum-sign-instance (&key (length 10))
  (let ((length (max 1 (random length)))
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
      (h (->lstm input :name 'h :size n-hiddens))
      (prediction (->softmax-xe-loss (->activation h :name 'prediction
                                                   :size *n-outputs*))))))

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
                             :monitors (make-cost-monitors
                                        rnn :attributes '(:event "train")))
              :dataset (make-sampler 30000))))

;;; Return a sampler object that produces MAX-N-SAMPLES number of
;;; random inputs.
(defun make-sampler (max-n-samples &key (length 10))
  (make-instance 'function-sampler :max-n-samples max-n-samples
                 :generator (lambda ()
                              (make-sum-sign-instance :length length))))

;;; Log the test error. Also, describe the optimizer and the bpn at
;;; the beginning of training. Called periodically during training
;;; (see above).
(defun log-test-error (optimizer learner)
  (when (zerop (n-instances optimizer))
    (describe optimizer)
    (describe (bpn learner)))
  (let ((rnn (bpn learner)))
    (log-padded
     (append
      (monitor-bpn-results (make-sampler 1000) rnn
                           (make-cost-monitors rnn
                                               :attributes '(:event "pred.")))
      ;; Same result in a different way: monitor predictions for
      ;; sequences up to length 20, but don't unfold the RNN
      ;; unnecessarily to save memory.
      (let ((*warp-time* t))
        (setf (warp-monitors rnn)
              (make-cost-monitors rnn :attributes '(:event "warped pred.")))
        (monitor-bpn-results (make-sampler 1000 :length 20) rnn
                             ;; Just collect and reset the warp
                             ;; monitors after each batch of
                             ;; instances.
                             (make-warp-monitor-monitors rnn)))))
    ;; Verify that no further unfoldings took place.
    (assert (<= (length (clumps rnn)) 10))))

#|

;;; Transcript follows:
(train-sum-sign-rnn)
.. 2015-01-28 14:12:54: training at n-instances: 0
.. 2015-01-28 14:12:54: train cost: 0.000e+0 (0)
.. #<ADAM-OPTIMIZER {101F2B4143}>
..  GD-OPTIMIZER description:
..    N-INSTANCES = 0
..    SEGMENT-SET = #<SEGMENT-SET
..                    (#<->WEIGHT (H #) :SIZE 1 1/1 :norm 2.00342>
..                     #<->WEIGHT (H #) :SIZE 1 1/1 :norm 1.75262>
..                     #<->WEIGHT (#1=# #2=# :PEEPHOLE) :SIZE
..                       1 1/1 :norm 0.92961>
..                     #<->WEIGHT (H #2#) :SIZE 1 1/1 :norm 0.20846>
..                     #<->WEIGHT (#1# #3=# :PEEPHOLE) :SIZE
..                       1 1/1 :norm 0.73093>
..                     #<->WEIGHT (H #3#) :SIZE 1 1/1 :norm 1.78671>
..                     #<->WEIGHT (H PREDICTION) :SIZE
..                       3 1/1 :norm 1.45207>
..                     #<->WEIGHT (:BIAS PREDICTION) :SIZE
..                       3 1/1 :norm 3.23801>
..                     #<->WEIGHT (#1# #4=# :PEEPHOLE) :SIZE
..                       1 1/1 :norm 1.01936>
..                     #<->WEIGHT (INPUT #4#) :SIZE 1 1/1 :norm 2.36555>
..                     #<->WEIGHT (:BIAS #4#) :SIZE 1 1/1 :norm 2.29635>
..                     #<->WEIGHT (INPUT #1#) :SIZE 1 1/1 :norm 1.32443>
..                     #<->WEIGHT (:BIAS #1#) :SIZE 1 1/1 :norm 1.30479>
..                     #<->WEIGHT (INPUT #5=#) :SIZE 1 1/1 :norm 1.89177>
..                     #<->WEIGHT (:BIAS #5#) :SIZE 1 1/1 :norm 1.50471>
..                     #<->WEIGHT (INPUT #6=#) :SIZE 1 1/1 :norm 1.97235>
..                     #<->WEIGHT (:BIAS #6#) :SIZE 1 1/1 :norm 1.77381>)
..                    {101F2B4AD3}>
..    LEARNING-RATE = 2.00000e-1
..    MOMENTUM = NONE
..    MOMENTUM-TYPE = :NONE
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
..  #<RNN {101F278433}>
..   BPN description:
..     CLUMPS = #(#<SUM-SIGN-FNN :STRIPES 1/50 :CLUMPS 4>
..                #<SUM-SIGN-FNN :STRIPES 1/50 :CLUMPS 4>)
..     N-STRIPES = 1
..     MAX-N-STRIPES = 50
..   
..   RNN description:
..     MAX-LAG = 1
..   2015-01-28 14:12:55: pred.        cost: 2.133d+0 (4647.00)
.. 2015-01-28 14:12:55: warped pred. cost: 2.028d+0 (9516.00)
.. 2015-01-28 14:12:55: training at n-instances: 3000
.. 2015-01-28 14:12:55: train cost: 1.155d+0 (13643.00)
.. 2015-01-28 14:12:56: training at n-instances: 6000
.. 2015-01-28 14:12:56: train cost: 4.123d-1 (13765.00)
.. 2015-01-28 14:12:57: training at n-instances: 9000
.. 2015-01-28 14:12:57: train cost: 8.208d-2 (13654.00)
.. 2015-01-28 14:12:57: training at n-instances: 12000
.. 2015-01-28 14:12:57: train cost: 2.889d-2 (13752.00)
.. 2015-01-28 14:12:58: training at n-instances: 15000
.. 2015-01-28 14:12:58: train cost: 1.781d-2 (13935.00)
.. 2015-01-28 14:12:58: training at n-instances: 18000
.. 2015-01-28 14:12:58: train cost: 1.271d-2 (13911.00)
.. 2015-01-28 14:12:59: training at n-instances: 21000
.. 2015-01-28 14:12:59: train cost: 9.828d-3 (13833.00)
.. 2015-01-28 14:13:00: training at n-instances: 24000
.. 2015-01-28 14:13:00: train cost: 7.782d-3 (13482.00)
.. 2015-01-28 14:13:00: training at n-instances: 27000
.. 2015-01-28 14:13:00: train cost: 6.457d-3 (14243.00)
.. 2015-01-28 14:13:01: training at n-instances: 30000
.. 2015-01-28 14:13:01: train cost: 5.451d-3 (13834.00)
.. 2015-01-28 14:13:01: pred.        cost: 5.024d-3 (4556.00)
.. 2015-01-28 14:13:01: warped pred. cost: 4.932d-3 (9569.00)
..
==> (#<->WEIGHT (H (H :OUTPUT)) :SIZE 1 1/1 :norm 3.25778>
-->  #<->WEIGHT (H (H :CELL)) :SIZE 1 1/1 :norm 2.64093>
-->  #<->WEIGHT ((H :CELL) (H :FORGET) :PEEPHOLE) :SIZE 1 1/1 :norm 0.24070>
-->  #<->WEIGHT (H (H :FORGET)) :SIZE 1 1/1 :norm 5.09629>
-->  #<->WEIGHT ((H :CELL) (H :INPUT) :PEEPHOLE) :SIZE 1 1/1 :norm 4.29833>
-->  #<->WEIGHT (H (H :INPUT)) :SIZE 1 1/1 :norm 3.12919>
-->  #<->WEIGHT (H PREDICTION) :SIZE 3 1/1 :norm 21.61549>
-->  #<->WEIGHT (:BIAS PREDICTION) :SIZE 3 1/1 :norm 4.45618>
-->  #<->WEIGHT ((H :CELL) (H :OUTPUT) :PEEPHOLE) :SIZE 1 1/1 :norm 0.77321>
-->  #<->WEIGHT (INPUT (H :OUTPUT)) :SIZE 1 1/1 :norm 1.74186>
-->  #<->WEIGHT (:BIAS (H :OUTPUT)) :SIZE 1 1/1 :norm 8.51236>
-->  #<->WEIGHT (INPUT (H :CELL)) :SIZE 1 1/1 :norm 4.18734>
-->  #<->WEIGHT (:BIAS (H :CELL)) :SIZE 1 1/1 :norm 1.02118>
-->  #<->WEIGHT (INPUT (H :FORGET)) :SIZE 1 1/1 :norm 4.29466>
-->  #<->WEIGHT (:BIAS (H :FORGET)) :SIZE 1 1/1 :norm 6.57931>
-->  #<->WEIGHT (INPUT (H :INPUT)) :SIZE 1 1/1 :norm 6.37012>
-->  #<->WEIGHT (:BIAS (H :INPUT)) :SIZE 1 1/1 :norm 3.06961>)

|#
