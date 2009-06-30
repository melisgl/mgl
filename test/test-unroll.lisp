(in-package :mgl-test)

(defun test-unroll-dbn ()
  (let ((rbm (make-instance 'rbm
                            :visible-chunks (list
                                             (make-instance 'constant-chunk
                                                            :name 'constant1)
                                             (make-instance 'gaussian-chunk
                                                            :name 'inputs
                                                            :size 10))
                            :hidden-chunks (list
                                            (make-instance 'constant-chunk
                                                           :name 'constant2)
                                            (make-instance 'sigmoid-chunk
                                                           :name 'features
                                                           :size 2)))))
    (unroll-dbn (make-instance 'dbn :rbms (list rbm)))))

(defun test-unroll-dbm ()
  (let ((dbm (make-instance 'dbm
                            :layers (list
                                     (list (make-instance 'sigmoid-chunk
                                                          :name 'inputs
                                                          :size 2)
                                           (make-instance 'constant-chunk
                                                          :name 'constant0))
                                     (list (make-instance 'sigmoid-chunk
                                                          :name 'features1
                                                          :size 1)
                                           (make-instance 'constant-chunk
                                                          :name 'constant1))
                                     (list (make-instance 'sigmoid-chunk
                                                          :name 'features2
                                                          :size 1)
                                           (make-instance 'constant-chunk
                                                          :name 'constant2))))))
    (unroll-dbm dbm)))

(defun test-unroll ()
  (test-unroll-dbn)
  (test-unroll-dbm))
