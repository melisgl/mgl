(in-package :mgl-bm)

(defclass dbn ()
  ((rbms :type list :initarg :rbms :reader rbms)
   (max-n-stripes :initform 1 :initarg :max-n-stripes :reader max-n-stripes))
  (:documentation "Deep Belief Network: a stack of RBMs. DBNs with
multiple hidden layers are not Boltzmann Machines. The chunks in the
hidden layer of a constituent RBM and the chunk in the visible layer
of the RBM one on top of it must be EQ for the DBN to consider them
the same. Naming them the same is not enough, in fact, all chunks must
have unique names under EQUAL as usual."))

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

(defmethod initialize-instance :before ((dbn dbn) &key rbms &allow-other-keys)
  (check-no-name-clashes rbms))

(defmethod initialize-instance :after ((dbn dbn) &key &allow-other-keys)
  (dolist (rbm (rbms dbn))
    (setf (slot-value rbm 'dbn) dbn))
  ;; make sure rbms have the same MAX-N-STRIPES
  (setf (max-n-stripes dbn) (max-n-stripes dbn)))

(defmethod chunks ((dbn dbn))
  (apply #'append (mapcar #'chunks (rbms dbn))))

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

(defun dbn-mean-field-errors
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
                        sampler
                        dbn
                        counters-and-measurers)
  (map 'list #'car counters-and-measurers))

(defun make-dbn-reconstruction-rmse-counters-and-measurers
    (dbn &key (rbm (last1 (rbms dbn))))
  "Return a list of counter, measurer conses to keep track of
reconstruction rmse suitable for BM-MEAN-FIELD-ERRORS."
  (loop for i upto (position rbm (rbms dbn))
        collect (let ((i i))
                  (cons (make-instance 'rmse-counter)
                        (lambda (samples)
                          (declare (ignore samples))
                          (reconstruction-error (elt (rbms dbn) i)))))))

(defun make-dbn-reconstruction-rmse-counters-and-measurers/no-labels
    (dbn &key (rbm (last1 (rbms dbn))))
  "Like MAKE-DBN-RECONSTRUCTION-RMSE-COUNTERS-AND-MEASURERS but don't
count reconstruction error of labels (that is, those chunks that
inherit from LABELED)."
  (loop for i upto (position rbm (rbms dbn))
        collect (let ((i i))
                  (cons (make-instance 'rmse-counter)
                        (lambda (samples)
                          (declare (ignore samples))
                          (reconstruction-rmse
                           (remove-if (lambda (chunk)
                                        (typep chunk 'labeled))
                                      (visible-chunks (elt (rbms dbn) i)))))))))

(defun make-dbn-reconstruction-misclassification-counters-and-measurers
    (dbn &key (rbm (last1 (rbms dbn))))
  "Return a list of counter, measurer conses to keep track of
misclassifications suitable for BM-MEAN-FIELD-ERRORS."
  (make-chunk-reconstruction-misclassification-counters-and-measurers
   (apply #'append
          (mapcar #'chunks
                  (subseq (rbms dbn)
                          0 (1+ (position rbm (rbms dbn))))))))

(defmethod write-weights ((dbn dbn) stream)
  (dolist (rbm (rbms dbn))
    (write-weights rbm stream)))

(defmethod read-weights ((dbn dbn) stream)
  (dolist (rbm (rbms dbn))
    (read-weights rbm stream)))
