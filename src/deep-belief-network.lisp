(in-package :mgl-bm)

(defclass dbn ()
  ((rbms :type list :initarg :rbms :reader rbms)
   (inactive-rbms :type list :initform () :reader inactive-rbms)
   (max-n-stripes :initform 1 :initarg :max-n-stripes :reader max-n-stripes))
  (:documentation "Deep Belief Network: a stack of RBMs. DBNs with
  multiple hidden layers are not Boltzmann Machines. The chunks in the
  hidden layer of a constituent RBM and the chunk in the visible layer
  of the RBM one on top of it must be EQ for the DBN to consider them
  the same. Naming them the same is not enough, in fact, all chunks
  must have unique names under EQUAL as usual.

  Similarly to DBMs, DBNs can be constructed using the :LAYERS
  initarg. When using this feature, a number of RBMs are instantiated.
  Often one wants to create a DBN that consists of some RBM subclass,
  this is what the :RBM-CLASS initarg is for."))

(defun all-rbms (dbn)
  (append (rbms dbn) (inactive-rbms dbn)))

(defmethod n-stripes ((dbn dbn))
  (n-stripes (or (first (rbms dbn))
                 (first (inactive-rbms dbn)))))

(defmethod set-n-stripes (n-stripes (dbn dbn))
  (dolist (rbm (rbms dbn))
    (setf (n-stripes rbm) n-stripes)))

(defmethod set-max-n-stripes (max-n-stripes (dbn dbn))
  (setf (slot-value dbn 'max-n-stripes) max-n-stripes)
  (dolist (rbm (all-rbms dbn))
    (setf (max-n-stripes rbm) max-n-stripes)))

(defun check-no-name-clashes (rbms)
  (let ((name-clashes
          (name-clashes
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

(defun n-rbms (dbn)
  (length (rbms dbn)))

(defun set-n-rbms (dbn n-rbms)
  (when(/= n-rbms (n-rbms dbn))
    (let ((all-rbms (all-rbms dbn)))
      (setf (slot-value dbn 'rbms) (subseq all-rbms 0 n-rbms))
      (setf (slot-value dbn 'inactive-rbms) (subseq all-rbms n-rbms))))
  n-rbms)

(defsetf n-rbms (dbn) (n-rbms)
  `(set-n-rbms ,dbn ,n-rbms))

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
  (check-no-name-clashes (cons rbm (all-rbms dbn)))
  (setf (slot-value rbm 'dbn) dbn)
  (if (inactive-rbms dbn)
      (setf (slot-value dbn 'inactive-rbms) (append1 (inactive-rbms dbn) rbm))
      (setf (slot-value dbn 'rbms) (append1 (rbms dbn) rbm)))
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

(defun down-mean-field (dbn)
  "Propagate the means down from the means of DBN."
  (mapc #'set-visible-mean (reverse (rbms dbn))))

(defmethod make-reconstruction-monitors* ((dbn dbn) operation-mode attributes)
  (loop for i upfrom 0
        for rbm in (rbms dbn)
        nconc (make-reconstruction-monitors
               rbm :attributes `(,@attributes
                                 :model ,(format nil "dbn l~A" i)))))

(defmethod make-classification-accuracy-monitors* ((dbn dbn) operation-mode
                                                   label-index-fn attributes)
  (loop for i upfrom 0
        for rbm in (rbms dbn)
        nconc (make-classification-accuracy-monitors
               rbm :label-index-fn label-index-fn
               :attributes `(,@attributes :model ,(format nil "dbn l~A" i)))))

(defmethod make-cross-entropy-monitors* ((dbn dbn) operation-mode
                                         label-index-distribution-fn attributes)
  (loop for i upfrom 0
        for rbm in (rbms dbn)
        nconc (make-cross-entropy-monitors
               rbm :label-index-distribution-fn label-index-distribution-fn
               :attributes `(,@attributes :model ,(format nil "dbn l~A" i)))))

(defun monitor-dbn-mean-field-bottom-up (dataset dbn monitors)
  "Run the mean field up to RBM then down to the bottom and collect
  the errors with COLLECT-BATCH-ERRORS. By default, return the rmse at
  each level in the DBN."
  (let ((rbm (last1 (rbms dbn))))
    (monitor-model-results (lambda (batch)
                             (set-input batch rbm)
                             (set-hidden-mean rbm)
                             dbn)
                           dataset dbn monitors)))

(defun monitor-dbn-mean-field-reconstructions (dataset dbn monitors &key
                                               set-visible-p)
  "Run the mean field up to RBM then down to the bottom and collect
  the errors with COLLECT-BATCH-ERRORS. By default, return the rmse at
  each level in the DBN."
  (let ((rbm (last1 (rbms dbn))))
    (monitor-model-results (lambda (batch)
                             (set-input batch rbm)
                             (set-hidden-mean rbm)
                             (when set-visible-p
                               (loop for rbm in (rbms dbn)
                                     do (mark-everything-present rbm)))
                             (down-mean-field dbn)
                             dbn)
                           dataset dbn monitors)))

(defmethod write-state* ((dbn dbn) stream context)
  (dolist (rbm (rbms dbn))
    (write-state* rbm stream context)))

(defmethod read-state* ((dbn dbn) stream context)
  (dolist (rbm (rbms dbn))
    (read-state* rbm stream context)))
