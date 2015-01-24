(in-package :mgl-test)

(defun test-unroll-dbn ()
  (let* ((rbm (make-instance 'rbm
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
                                                            :size 2))))
         (dbn (make-instance 'dbn :rbms (list rbm))))
    (multiple-value-call #'create-from-unrolled dbn (unroll-dbn dbn))))

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
    (multiple-value-call #'create-from-unrolled dbm (unroll-dbm dbm))))

(defun create-from-unrolled (bm defs inits)
  (let ((bpn (eval `(build-fnn ()
                      ,@defs))))
    (initialize-fnn-from-bm bpn bm inits)))

(defun test-unroll ()
  (do-cuda ()
    (test-unroll-dbn)
    (test-unroll-dbm)))
