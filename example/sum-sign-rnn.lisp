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
  (log-padded
   (monitor-bpn-results (make-sampler 1000) (bpn learner)
                        (make-cost-monitors (bpn learner)
                                            :attributes '(:event "pred.")))))

#|

;;; Transcript follows:
(train-sum-sign-rnn)
.. 2015-01-21 14:39:42: training at n-instances: 0
.. 2015-01-21 14:39:42: train cost: 0.000e+0 (0)
.. #<ADAM-OPTIMIZER {1010825793}>
..  GD-OPTIMIZER description:
..    N-INSTANCES = 0
..    SEGMENT-SET = #<SEGMENT-SET
..                    (#<->WEIGHT (H #) :SIZE 1 1/1 :norm 1.84894>
..                     #<->WEIGHT (H #) :SIZE 1 1/1 :norm 1.26455>
..                     #<->WEIGHT (#1=# #2=# :PEEPHOLE) :SIZE
..                       1 1/1 :norm 1.79465>
..                     #<->WEIGHT (H #2#) :SIZE 1 1/1 :norm 1.85332>
..                     #<->WEIGHT (#1# #3=# :PEEPHOLE) :SIZE
..                       1 1/1 :norm 1.24396>
..                     #<->WEIGHT (H #3#) :SIZE 1 1/1 :norm 1.12703>
..                     #<->WEIGHT (H PREDICTION) :SIZE
..                       3 1/1 :norm 2.04857>
..                     #<->WEIGHT (:BIAS PREDICTION) :SIZE
..                       3 1/1 :norm 2.85912>
..                     #<->WEIGHT (#1# #4=# :PEEPHOLE) :SIZE
..                       1 1/1 :norm 0.94985>
..                     #<->WEIGHT (INPUT #4#) :SIZE 1 1/1 :norm 0.04429>
..                     #<->WEIGHT (:BIAS #4#) :SIZE 1 1/1 :norm 2.06559>
..                     #<->WEIGHT (INPUT #1#) :SIZE 1 1/1 :norm 2.02872>
..                     #<->WEIGHT (:BIAS #1#) :SIZE 1 1/1 :norm 1.39504>
..                     #<->WEIGHT (INPUT #5=#) :SIZE 1 1/1 :norm 2.00406>
..                     #<->WEIGHT (:BIAS #5#) :SIZE 1 1/1 :norm 2.27676>
..                     #<->WEIGHT (INPUT #6=#) :SIZE 1 1/1 :norm 1.35271>
..                     #<->WEIGHT (:BIAS #6#) :SIZE 1 1/1 :norm 1.67755>)
..                    {1010826493}>
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
..  #<RNN {10107ED8E3}>
..   BPN description:
..     CLUMPS = #(#<SUM-SIGN-FNN :STRIPES 1/50 :CLUMPS 4 {10107EDBB3}>
..                #<SUM-SIGN-FNN :STRIPES 1/50 :CLUMPS 4 {1010807733}>)
..     N-STRIPES = 1
..     MAX-N-STRIPES = 50
..   
..   RNN description:
..     MAX-LAG = 1
..   2015-01-21 14:39:43: pred. cost: 2.150d+0 (5668.00)
.. 2015-01-21 14:39:43: training at n-instances: 3000
.. 2015-01-21 14:39:43: train cost: 4.892d-1 (16419.00)
.. 2015-01-21 14:39:44: training at n-instances: 6000
.. 2015-01-21 14:39:44: train cost: 8.402d-2 (16331.00)
.. 2015-01-21 14:39:44: training at n-instances: 9000
.. 2015-01-21 14:39:44: train cost: 4.018d-2 (16680.00)
.. 2015-01-21 14:39:45: training at n-instances: 12000
.. 2015-01-21 14:39:45: train cost: 2.576d-2 (16349.00)
.. 2015-01-21 14:39:45: training at n-instances: 15000
.. 2015-01-21 14:39:45: train cost: 1.840d-2 (16629.00)
.. 2015-01-21 14:39:45: training at n-instances: 18000
.. 2015-01-21 14:39:45: train cost: 1.396d-2 (16476.00)
.. 2015-01-21 14:39:46: training at n-instances: 21000
.. 2015-01-21 14:39:46: train cost: 1.106d-2 (16435.00)
.. 2015-01-21 14:39:46: training at n-instances: 24000
.. 2015-01-21 14:39:46: train cost: 9.131d-3 (16566.00)
.. 2015-01-21 14:39:47: training at n-instances: 27000
.. 2015-01-21 14:39:47: train cost: 7.606d-3 (16394.00)
.. 2015-01-21 14:39:47: training at n-instances: 30000
.. 2015-01-21 14:39:47: train cost: 6.466d-3 (16455.00)
.. 2015-01-21 14:39:47: pred. cost: 5.997d-3 (5530.00)
..
==> (#<->WEIGHT (H (H :OUTPUT)) :SIZE 1 1/1 :norm 0.98670>
-->  #<->WEIGHT (H (H :CELL)) :SIZE 1 1/1 :norm 0.73203>
-->  #<->WEIGHT ((H :CELL) (H :FORGET) :PEEPHOLE) :SIZE 1 1/1 :norm 0.57356>
-->  #<->WEIGHT (H (H :FORGET)) :SIZE 1 1/1 :norm 1.74084>
-->  #<->WEIGHT ((H :CELL) (H :INPUT) :PEEPHOLE) :SIZE 1 1/1 :norm 0.08929>
-->  #<->WEIGHT (H (H :INPUT)) :SIZE 1 1/1 :norm 0.26631>
-->  #<->WEIGHT (H PREDICTION) :SIZE 3 1/1 :norm 19.76594>
-->  #<->WEIGHT (:BIAS PREDICTION) :SIZE 3 1/1 :norm 4.22242>
-->  #<->WEIGHT ((H :CELL) (H :OUTPUT) :PEEPHOLE) :SIZE 1 1/1 :norm 0.57314>
-->  #<->WEIGHT (INPUT (H :OUTPUT)) :SIZE 1 1/1 :norm 0.87375>
-->  #<->WEIGHT (:BIAS (H :OUTPUT)) :SIZE 1 1/1 :norm 7.38429>
-->  #<->WEIGHT (INPUT (H :CELL)) :SIZE 1 1/1 :norm 3.63191>
-->  #<->WEIGHT (:BIAS (H :CELL)) :SIZE 1 1/1 :norm 0.04245>
-->  #<->WEIGHT (INPUT (H :FORGET)) :SIZE 1 1/1 :norm 3.92401>
-->  #<->WEIGHT (:BIAS (H :FORGET)) :SIZE 1 1/1 :norm 1.97384>
-->  #<->WEIGHT (INPUT (H :INPUT)) :SIZE 1 1/1 :norm 0.60136>
-->  #<->WEIGHT (:BIAS (H :INPUT)) :SIZE 1 1/1 :norm 6.22344>)

|#
