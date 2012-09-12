(in-package :mgl-bm)

(defclass dbn ()
  ((rbms :type list :initarg :rbms :reader rbms)
   (max-n-stripes :initform 1 :initarg :max-n-stripes :reader max-n-stripes))
  (:documentation "Deep Belief Network: a stack of RBMs. DBNs with
multiple hidden layers are not Boltzmann Machines. The chunks in the
hidden layer of a constituent RBM and the chunk in the visible layer
of the RBM one on top of it must be EQ for the DBN to consider them
the same. Naming them the same is not enough, in fact, all chunks must
have unique names under EQUAL as usual.

Similarly to DBMs, DBNs can be constructed using the :LAYERS initarg.
When using this feature, a number of RBMs are instantiated. Often one
wants to create a DBN that consists of some RBM subclass, this is what
the :RBM-CLASS initarg is for."))

(defmethod n-stripes ((dbn dbn))
  (n-stripes (first (rbms dbn))))

(defmethod set-n-stripes (n-stripes (dbn dbn))
  (dolist (rbm (rbms dbn))
    (setf (n-stripes rbm) n-stripes)))

(defmethod set-max-n-stripes (max-n-stripes (dbn dbn))
  (dolist (rbm (rbms dbn))
    (setf (max-n-stripes rbm) max-n-stripes)))

(defun check-no-name-clashes (rbms)
  (let ((name-clashes (name-clashes
                       (append (apply #'append (mapcar #'visible-chunks rbms))
                               (apply #'append (mapcar #'hidden-chunks rbms))))))
    (when name-clashes
      (error "Name conflict between chunks: ~S" name-clashes)))
  (let ((name-clashes (name-clashes (apply #'append (mapcar #'clouds rbms)))))
    (when name-clashes
      (error "Name conflict between clouds: ~S" name-clashes))))

(defmethod initialize-instance :around ((dbn dbn) &key (layers () layersp)
                                        clouds-up-to-layers (rbm-class 'rbm)
                                        &allow-other-keys)
  (when layersp
    (setf (slot-value dbn 'rbms)
          (loop for (layer1 layer2) on layers
                for cloud-spec in (or clouds-up-to-layers
                                      (make-list (1- (length layers))
                                                 :initial-element
                                                 '(:merge)))
                while layer2
                collect (make-instance rbm-class
                                       :visible-chunks layer1
                                       :hidden-chunks layer2
                                       :clouds cloud-spec))))
  (call-next-method))

(defmethod initialize-instance :after ((dbn dbn) &key &allow-other-keys)
  (check-no-name-clashes (rbms dbn))
  (dolist (rbm (rbms dbn))
    (setf (slot-value rbm 'dbn) dbn))
  ;; make sure rbms have the same MAX-N-STRIPES
  (setf (max-n-stripes dbn) (max-n-stripes dbn)))

(defmethod chunks ((dbn dbn))
  (apply #'append (mapcar #'chunks (rbms dbn))))

(defmethod visible-chunks ((dbn dbn))
  (set-difference (apply #'append (mapcar #'visible-chunks (rbms dbn)))
                  (hidden-chunks dbn)))

(defmethod hidden-chunks ((dbn dbn))
  (apply #'append (mapcar #'hidden-chunks (rbms dbn))))

(defmethod clouds ((dbn dbn))
  (apply #'append (mapcar #'clouds (rbms dbn))))

(defmethod find-chunk (name (dbn dbn) &key errorp)
  (dolist (rbm (rbms dbn))
    (let ((chunk (find-chunk name rbm)))
      (when chunk
        (return-from find-chunk chunk))))
  (when errorp
    (error "Can't find chunk named ~A in ~A." name dbn))
  nil)

(defmethod find-cloud (name (dbn dbn) &key errorp)
  (dolist (rbm (rbms dbn))
    (let ((cloud (find-cloud name rbm)))
      (when cloud
        (return-from find-cloud cloud))))
  (when errorp
    (error "Can't find cloud named ~A in ~A." name dbn))
  nil)

(defun add-rbm (rbm dbn)
  (check-no-name-clashes (cons rbm (rbms dbn)))
  (setf (slot-value rbm 'dbn) dbn
        (slot-value dbn 'rbms) (append1 (rbms dbn) rbm))
  (setf (max-n-stripes rbm) (max-n-stripes dbn)))

(defun previous-rbm (dbn rbm)
  (let ((pos (position rbm (rbms dbn))))
    (if (and pos (plusp pos))
        (elt (rbms dbn) (1- pos))
        nil)))

(defmethod set-input :before (samples (rbm rbm))
  ;; Do SET-INPUT on the previous rbm (if any) and propagate its mean
  ;; to this one.
  (when (dbn rbm)
    (let ((prev (previous-rbm (dbn rbm) rbm)))
      (when prev
        (set-input samples prev)
        (set-hidden-mean prev)))))

(defmethod set-input (samples (dbn dbn))
  (set-input samples (last1 (rbms dbn))))

(defun not-before (list obj)
  (let ((pos (position obj list)))
    (if pos
        (subseq list pos)
        list)))

(defun down-mean-field (dbn &key (rbm (last1 (rbms dbn))))
  "Propagate the means down from the means of RBM."
  (mapc #'set-visible-mean
        (not-before (reverse (rbms dbn)) rbm)))

(defun make-dbn-reconstruction-rmse-counters-and-measurers
    (dbn &key (rbm (last1 (rbms dbn))) chunk-filter)
  "Return a list of counter, measurer conses to keep track of
reconstruction rmse suitable for COLLECT-BM-MEAN-FIELD-ERRORS."
  (loop for i upto (position rbm (rbms dbn))
        collect (let ((i i))
                  (cons (make-instance 'rmse-counter
                                       :prepend-name (format nil "level ~A" i))
                        (lambda (samples dbn)
                          (declare (ignore samples))
                          (reconstruction-rmse
                           (remove-if* chunk-filter
                                       (visible-chunks
                                        (elt (rbms dbn) i)))))))))

(defun collect-dbn-mean-field-errors
    (sampler dbn &key (rbm (last1 (rbms dbn)))
     (counters-and-measurers
      (make-dbn-reconstruction-rmse-counters-and-measurers dbn :rbm rbm)))
  "Run the mean field up to RBM then down to the bottom and collect
the errors with COLLECT-BATCH-ERRORS. By default, return the rmse at
each level in the DBN."
  (collect-batch-errors (lambda (samples)
                          (set-input samples rbm)
                          (set-hidden-mean rbm)
                          (down-mean-field dbn :rbm rbm))
                        sampler dbn counters-and-measurers))

(defun make-dbn-reconstruction-misclassification-counters-and-measurers
    (dbn &key (rbm (last1 (rbms dbn))) chunk-filter)
  "Return a list of counter, measurer conses to keep track of
misclassifications suitable for BM-MEAN-FIELD-ERRORS."
  (make-chunk-reconstruction-misclassification-counters-and-measurers
   (apply #'append
          (mapcar #'visible-chunks
                  (subseq (rbms dbn)
                          0 (1+ (position rbm (rbms dbn))))))
   :chunk-filter chunk-filter))

(defun make-dbn-reconstruction-cross-entropy-counters-and-measurers
    (dbn &key (rbm (last1 (rbms dbn))) chunk-filter)
  "Return a list of counter, measurer conses to keep track of
misclassifications suitable for BM-MEAN-FIELD-ERRORS."
  (make-chunk-reconstruction-cross-entropy-counters-and-measurers
   (apply #'append
          (mapcar #'visible-chunks
                  (subseq (rbms dbn)
                          0 (1+ (position rbm (rbms dbn))))))
   :chunk-filter chunk-filter))

(defun collect-dbn-mean-field-errors/labeled
    (sampler dbn &key (rbm (last1 (rbms dbn)))
     (counters-and-measurers
      (append
       (make-dbn-reconstruction-misclassification-counters-and-measurers
        dbn :rbm rbm)
       (make-dbn-reconstruction-cross-entropy-counters-and-measurers
        dbn :rbm rbm))))
  "Like COLLECT-DBN-MEAN-FIELD-ERRORS but reconstruct labeled chunks
even if it's missing in the input."
  (collect-batch-errors (lambda (samples)
                          (set-input samples rbm)
                          (set-hidden-mean rbm)
                          (mark-labels-present dbn)
                          (down-mean-field dbn :rbm rbm))
                        sampler dbn counters-and-measurers))

(defmethod write-weights ((dbn dbn) stream)
  (dolist (rbm (rbms dbn))
    (write-weights rbm stream)))

(defmethod read-weights ((dbn dbn) stream)
  (dolist (rbm (rbms dbn))
    (read-weights rbm stream)))
