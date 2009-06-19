(in-package :mgl-bm)

(defclass dbn ()
  ((rbms :type list :initarg :rbms :reader rbms)
   (max-n-stripes :initform 1 :initarg :max-n-stripes :reader max-n-stripes))
  (:documentation "Deep Belief Network: a stack of RBMs. DBNs with
multiple hidden layers are not Boltzmann Machines."))

(defmethod n-stripes ((dbn dbn))
  (n-stripes (first (rbms dbn))))

(defmethod set-n-stripes (n-stripes (dbn dbn))
  (dolist (rbm (rbms dbn))
    (setf (n-stripes rbm) n-stripes)))

(defmethod set-max-n-stripes (max-n-stripes (dbn dbn))
  (dolist (rbm (rbms dbn))
    (setf (max-n-stripes rbm) max-n-stripes)))

(defun check-no-name-clashes (rbms)
  (unless (unique-names-p
           (append (apply #'append (mapcar #'visible-chunks rbms))
                   (apply #'append (mapcar #'hidden-chunks rbms))))
    (error "Name conflict between chunks: ~S" rbms))
  (unless (unique-names-p (apply #'append (mapcar #'clouds rbms)))
    (error "Name conflict between clouds: ~S" rbms)))

(defmethod initialize-instance :before ((dbn dbn) &key rbms &allow-other-keys)
  (check-no-name-clashes rbms))

(defmethod initialize-instance :after ((dbn dbn) &key &allow-other-keys)
  (dolist (rbm (rbms dbn))
    (setf (slot-value rbm 'dbn) dbn))
  ;; make sure rbms have the same MAX-N-STRIPES
  (setf (max-n-stripes dbn) (max-n-stripes dbn)))

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

(defmethod set-input :around (samples (rbm rbm))
  ;; Do SET-INPUT on the previous rbm (if any) and propagate its mean
  ;; to this one.
  (when (dbn rbm)
    (let ((prev (previous-rbm (dbn rbm) rbm)))
      (when prev
        (set-input samples prev)
        (set-hidden-mean prev))))
  (call-next-method))

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

(defun make-reconstruction-rmse-counters-and-measurers
    (dbn &key (rbm (last1 (rbms dbn))))
  (loop for i upto (position rbm (rbms dbn))
        collect (let ((i i))
                  (cons (make-instance 'rmse-counter)
                        (lambda (dbn samples)
                          (declare (ignore samples))
                          (reconstruction-error (elt (rbms dbn) i)))))))

(defun dbn-mean-field-errors
    (sampler dbn &key (rbm (last1 (rbms dbn)))
     (counters-and-measurers
      (make-reconstruction-rmse-counters-and-measurers dbn :rbm rbm)))
  "Sample from SAMPLER until it runs out. Set the samples as inputs
and run the mean field up to RBM then down to the bottom.
COUNTERS-AND-MEASURERS is a sequence of conses of a counter and
function. The function takes two parameters: the DBN and a sequence of
samples and is called after each mean field reconstruction. Measurers
return two values: the cumulative error and the counter, suitable as
the second and third argument to ADD-ERROR. Finally, return the
counters. By default, return the rmse at each level in the DBN."
  (let ((max-n-stripes (max-n-stripes dbn)))
    (loop until (finishedp sampler) do
          (let ((samples (sample-batch sampler max-n-stripes)))
            (set-input samples rbm)
            (set-hidden-mean rbm)
            (down-mean-field dbn :rbm rbm)
            (map nil
                 (lambda (counter-and-measurer)
                   (assert (consp counter-and-measurer))
                   (multiple-value-call
                       #'add-error (car counter-and-measurer)
                       (funcall (cdr counter-and-measurer)
                                dbn samples)))
                 counters-and-measurers))))
  (map 'list #'car counters-and-measurers))

(defmethod write-weights ((dbn dbn) stream)
  (dolist (rbm (rbms dbn))
    (write-weights rbm stream)))

(defmethod read-weights ((dbn dbn) stream)
  (dolist (rbm (rbms dbn))
    (read-weights rbm stream)))
