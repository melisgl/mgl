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
                             :monitors (make-cost-monitors rnn))
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
                           (make-cost-monitors
                            rnn :attributes '(:event "pred.")))
      ;; Same result in a different way: monitor predictions for
      ;; sequences up to length 20, but don't unfold the RNN
      ;; unnecessarily to save memory.
      (let ((*warp-time* t))
        (monitor-bpn-results (make-sampler 1000 :length 20) rnn
                             ;; Just collect and reset the warp
                             ;; monitors after each batch of
                             ;; instances.
                             (make-cost-monitors
                              rnn :attributes '(:event "warped pred."))))))
    ;; Verify that no further unfoldings took place.
    (assert (<= (length (clumps rnn)) 10))))

#|

;;; Transcript follows:
(let (;; Backprop nets do not need double float. Using single floats
      ;; is faster and needs less memory.
      (*default-mat-ctype* :float)
      ;; Enable moving data in and out of GPU memory so that the RNN
      ;; can work with sequences so long that the unfolded network
      ;; wouldn't otherwise fit in the GPU.
      (*cuda-window-start-time* 1))
  ;; Seed the random number generators.
  (repeatably ()
    ;; Enable CUDA if available.
    (with-cuda* ()
      (train-sum-sign-rnn))))
.. 2015-02-08 12:09:14: training at n-instances: 0
.. 2015-02-08 12:09:14: cost: 0.000e+0 (0)
.. #<ADAM-OPTIMIZER {103192AEF3}>
..  GD-OPTIMIZER description:
..    N-INSTANCES = 0
..    SEGMENT-SET = #<SEGMENT-SET
..                    (#<->WEIGHT (H #) :SIZE 1 1/1 :norm 1.73685>
..                     #<->WEIGHT (H #) :SIZE 1 1/1 :norm 0.31893>
..                     #<->WEIGHT (#1=# #2=# :PEEPHOLE) :SIZE
..                       1 1/1 :norm 1.81610>
..                     #<->WEIGHT (H #2#) :SIZE 1 1/1 :norm 0.21965>
..                     #<->WEIGHT (#1# #3=# :PEEPHOLE) :SIZE
..                       1 1/1 :norm 1.74939>
..                     #<->WEIGHT (H #3#) :SIZE 1 1/1 :norm 0.40377>
..                     #<->WEIGHT (H PREDICTION) :SIZE
..                       3 1/1 :norm 2.15898>
..                     #<->WEIGHT (:BIAS PREDICTION) :SIZE
..                       3 1/1 :norm 2.94470>
..                     #<->WEIGHT (#1# #4=# :PEEPHOLE) :SIZE
..                       1 1/1 :norm 0.97601>
..                     #<->WEIGHT (INPUT #4#) :SIZE 1 1/1 :norm 0.65261>
..                     #<->WEIGHT (:BIAS #4#) :SIZE 1 1/1 :norm 0.37653>
..                     #<->WEIGHT (INPUT #1#) :SIZE 1 1/1 :norm 0.92334>
..                     #<->WEIGHT (:BIAS #1#) :SIZE 1 1/1 :norm 0.01609>
..                     #<->WEIGHT (INPUT #5=#) :SIZE 1 1/1 :norm 1.09995>
..                     #<->WEIGHT (:BIAS #5#) :SIZE 1 1/1 :norm 1.41244>
..                     #<->WEIGHT (INPUT #6=#) :SIZE 1 1/1 :norm 0.40475>
..                     #<->WEIGHT (:BIAS #6#) :SIZE 1 1/1 :norm 1.75358>)
..                    {10319D3653}>
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
..  #<RNN {10318C6C53}>
..   BPN description:
..     CLUMPS = #(#<SUM-SIGN-FNN :STRIPES 1/50 :CLUMPS 4>
..                #<SUM-SIGN-FNN :STRIPES 1/50 :CLUMPS 4>)
..     N-STRIPES = 1
..     MAX-N-STRIPES = 50
..   
..   RNN description:
..     MAX-LAG = 1
..   2015-02-08 12:09:15: pred.        cost: 1.223e+0 (4455.00)
.. 2015-02-08 12:09:15: warped pred. cost: 1.228e+0 (9476.00)
.. 2015-02-08 12:09:15: Foreign memory usage:
.. foreign arrays: 166 (used bytes: 40,400)
.. CUDA memory usage:
.. device arrays: 114 (used bytes: 220,892, pooled bytes: 19,200)
.. host arrays: 162 (used bytes: 39,600)
.. host->device copies: 6,164, device->host copies: 4,490
.. 2015-02-08 12:09:17: training at n-instances: 3000
.. 2015-02-08 12:09:17: cost: 4.986e-1 (13726.00)
.. 2015-02-08 12:09:19: training at n-instances: 6000
.. 2015-02-08 12:09:19: cost: 5.769e-2 (13890.00)
.. 2015-02-08 12:09:20: training at n-instances: 9000
.. 2015-02-08 12:09:20: cost: 2.168e-2 (13872.00)
.. 2015-02-08 12:09:22: training at n-instances: 12000
.. 2015-02-08 12:09:22: cost: 1.348e-2 (13953.00)
.. 2015-02-08 12:09:24: training at n-instances: 15000
.. 2015-02-08 12:09:24: cost: 9.696e-3 (13948.00)
.. 2015-02-08 12:09:26: training at n-instances: 18000
.. 2015-02-08 12:09:26: cost: 7.382e-3 (13849.00)
.. 2015-02-08 12:09:28: training at n-instances: 21000
.. 2015-02-08 12:09:28: cost: 5.861e-3 (13758.00)
.. 2015-02-08 12:09:30: training at n-instances: 24000
.. 2015-02-08 12:09:30: cost: 4.849e-3 (13908.00)
.. 2015-02-08 12:09:32: training at n-instances: 27000
.. 2015-02-08 12:09:32: cost: 4.013e-3 (13570.00)
.. 2015-02-08 12:09:34: training at n-instances: 30000
.. 2015-02-08 12:09:34: cost: 3.459e-3 (13721.00)
.. 2015-02-08 12:09:34: pred.        cost: 3.284e-3 (4593.00)
.. 2015-02-08 12:09:34: warped pred. cost: 3.285e-3 (9717.00)
.. 2015-02-08 12:09:34: Foreign memory usage:
.. foreign arrays: 220 (used bytes: 53,600)
.. CUDA memory usage:
.. device arrays: 148 (used bytes: 224,428, pooled bytes: 19,200)
.. host arrays: 216 (used bytes: 52,800)
.. host->device copies: 465,818, device->host copies: 371,990
..
==> (#<->WEIGHT (H (H :OUTPUT)) :SIZE 1 1/1 :norm 0.75226>
-->  #<->WEIGHT (H (H :CELL)) :SIZE 1 1/1 :norm 0.04934>
-->  #<->WEIGHT ((H :CELL) (H :FORGET) :PEEPHOLE) :SIZE 1 1/1 :norm 0.14970>
-->  #<->WEIGHT (H (H :FORGET)) :SIZE 1 1/1 :norm 0.67698>
-->  #<->WEIGHT ((H :CELL) (H :INPUT) :PEEPHOLE) :SIZE 1 1/1 :norm 1.40400>
-->  #<->WEIGHT (H (H :INPUT)) :SIZE 1 1/1 :norm 0.33328>
-->  #<->WEIGHT (H PREDICTION) :SIZE 3 1/1 :norm 21.41458>
-->  #<->WEIGHT (:BIAS PREDICTION) :SIZE 3 1/1 :norm 4.97210>
-->  #<->WEIGHT ((H :CELL) (H :OUTPUT) :PEEPHOLE) :SIZE 1 1/1 :norm 0.65959>
-->  #<->WEIGHT (INPUT (H :OUTPUT)) :SIZE 1 1/1 :norm 0.77142>
-->  #<->WEIGHT (:BIAS (H :OUTPUT)) :SIZE 1 1/1 :norm 6.28021>
-->  #<->WEIGHT (INPUT (H :CELL)) :SIZE 1 1/1 :norm 4.96948>
-->  #<->WEIGHT (:BIAS (H :CELL)) :SIZE 1 1/1 :norm 0.33460>
-->  #<->WEIGHT (INPUT (H :FORGET)) :SIZE 1 1/1 :norm 3.11331>
-->  #<->WEIGHT (:BIAS (H :FORGET)) :SIZE 1 1/1 :norm 1.13583>
-->  #<->WEIGHT (INPUT (H :INPUT)) :SIZE 1 1/1 :norm 0.15849>
-->  #<->WEIGHT (:BIAS (H :INPUT)) :SIZE 1 1/1 :norm 4.95984>)

|#
