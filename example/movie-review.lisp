;;;; Movie review sentiment classification.
;;;;
;;;; Data set `polarity review 2.0' from
;;;; http://www.cs.cornell.edu/people/pabo/movie-review-data/
;;;;
;;;; * Bo Pang and Lillian Lee, A Sentimental Education: Sentiment
;;;;   Analysis Using Subjectivity Summarization Based on Minimum
;;;;   Cuts, Proceedings of ACL 2004.
;;;;
;;;; Their best result is 87.15% for an SVM with unigram bag-of-words
;;;; which is reproduced with linear kernel, log-c=8, n-words=8000,
;;;; most frequent, normalized-binary.
;;;;
;;;; However, this one claims 90.45%:
;;;;
;;;; * Customizing Sentiment classifiers to new domains: a case study,
;;;;   Aue, Gamon
;;;;   (http://research.microsoft.com/~anthaue/new_domain_sentiment.pdf)
;;;;
;;;; In personal communications, Gamon elaborates: "the kernel is
;;;; linear, we used vanilla settings on the algorithm (SMO, Platt 99)
;;;; without parameter tuning. The 90.45% accuracy is based on
;;;; uni/bi/trigram features, using the top 20k features (based on
;;;; Dunning's log likelihood ratio as established on the training
;;;; set) in 5 fold crossvalidation" and "with unigrams accuracy tops
;;;; out at 88% (using top 10k features based on log likelihood ratio)".
;;;;
;;;; ... but I fail to reproduce anything near that with or without
;;;; LLR. The best I get is 88.45 with: linear svm, log-c=8,
;;;; n-words=20000, llr, normalized-binary, n-grams=3. With unigrams
;;;; the best is 86.85 with linear svm, log-c=8, n-words=10000, llr,
;;;; normalized-binary, n-grams=1.
;;;;
;;;; The RBM implementation achieves about the same score on 10 fold
;;;; cross validation with LLR feature selection on uni-, bi- and
;;;; trigrams using only DBN without fine tuning. The greatest problem
;;;; is overfitting.

(in-package :mgl-example-movie-review)

(defparameter *mr-dir*
  (merge-pathnames "review-polarity-data/" *example-dir*)
  "Set this to the directory where the unpacked data files reside.")
(defparameter *n-words* 4000)
(defparameter *n-grams* 3)

(defvar *negative-stories*)
(defvar *positive-stories*)
(defvar *training-stories*)
(defvar *test-stories*)
(defvar *words*)
(defvar *dbn*)
(defvar *bpn*)


;;;;

(defstruct story
  (label nil :type (member -1 1))
  (id nil :type (integer 0 1000))
  (file-id nil :type unsigned-byte)
  string
  array)

(defun slurp-file (file)
  (with-open-file (in file :direction :input
                   #+sbcl :external-format #+sbcl :latin-1)
    (let ((v (make-array (file-length in) :element-type 'base-char)))
      (assert (= (length v) (read-sequence v in)))
      v)))

(defparameter *filename-scanner*
  (cl-ppcre:create-scanner "cv([0-9]*)_([0-9]*)"))

(defun read-folder (name label)
  (let ((r (coerce
            (loop for filename in (directory
                                   (merge-pathnames (make-pathname :name :wild
                                                                   :type "txt")
                                                    name))
                  collect
                  (let ((scans (nth-value 1 (cl-ppcre:scan-to-strings
                                             *filename-scanner*
                                             (pathname-name filename)))))
                    (make-story :label label
                                :id (read-from-string (aref scans 0))
                                :file-id (read-from-string (aref scans 1))
                                :string (slurp-file filename))))
            'vector)))
    (assert (= 1000 (length r)))
    r))

(defun load-stories (&optional (data-dir *mr-dir*))
  (values (read-folder (merge-pathnames "txt_sentoken/neg/" data-dir) -1)
          (read-folder (merge-pathnames "txt_sentoken/pos/" data-dir) 1)))


;;;; Tokenization. MAP-WORDS and its helpers. Has stemming, special
;;;; handling of numbers and percentages to collapse similar ones into
;;;; the same word.

(defun ->base-string (string)
  (let ((s (make-array (length string) :element-type 'base-char)))
    (replace s string)))

(defun normalize-word (word)
  (->base-string (string-downcase word)))

(defparameter *default-word-scanner*
  (cl-ppcre:create-scanner "\\b([^\\s]+)\\b"))

(defun map-words (function string &key
                  (word-scanner *default-word-scanner*)
                  (n-grams *n-grams*))
  "Apply FUNCTION to all stemmed words in STREAM."
  (let ((function (coerce function 'function)))
    (loop for i upfrom 1 upto n-grams
          do (let ((function (mgl-util:make-n-gram-mappee function i)))
               (cl-ppcre:do-register-groups (word) (word-scanner string)
                 (let ((word (normalize-word word)))
                   (map nil function (list word))))))))

(defun test-map-words ()
  (let ((words ()))
    (map-words (lambda (word)
                 (push word words))
               "However , this movie doesn't suck simply . Why ? (0/4)"
               :n-grams 3)
    (assert (equal (reverse words)
                   '(("however") ("this")  ("movie") ("doesn't") ("suck")
                     ("simply") ("why") ("0/4")
                     ("however" "this")
                     ("this" "movie")
                     ("movie" "doesn't")
                     ("doesn't" "suck")
                     ("suck" "simply")
                     ("simply" "why")
                     ("why" "0/4")
                     ("however" "this" "movie")
                     ("this" "movie" "doesn't")
                     ("movie" "doesn't" "suck")
                     ("doesn't" "suck" "simply")
                     ("suck" "simply" "why")
                     ("simply" "why" "0/4"))))))

(test-map-words)

(defun map-story-words (fn story)
  (map-words fn (story-string story)))


;;;; Sampling, clamping

(defun make-sampler (stories &key max-n omit-label-p)
  (make-instance 'counting-function-sampler
                 :max-n-samples max-n
                 :sampler (if omit-label-p
                              (let ((g (make-random-generator stories)))
                                (lambda ()
                                  (list (funcall g) :omit-label-p t)))
                              (make-random-generator stories))))

(defun clamp-story (story array start end)
  (declare (type flt-vector array))
  (locally (declare (optimize (speed 3)))
    (fill array #.(flt 0) :start start :end end))
  (loop for (i . v) across (story-array story)
        do
        (let ((i (+ start i)))
          (assert (< i end))
          (setf (aref array i) v))))

(defun clamp-label (label array start end)
  (declare (type flt-vector array))
  (fill array #.(flt 0) :start start :end end)
  (setf (aref array (+ start (if (minusp label) 0 1)))
        #.(flt 1)))

(defun max-position (array start end)
  (position (loop for i upfrom start below end maximizing (aref array i))
            array :start start :end end))

(defun decode-label (array start end)
  (declare (type flt-vector array))
  (ecase (- (max-position array start end) start)
    ((0) -1)
    ((1) 1)))

(defun encode-all (stories mapper words &key (kind :binary))
  (map nil (lambda (story)
             (setf (story-array story)
                   (mgl-util:encode/bag-of-words story
                                                 mapper
                                                 (lambda (word)
                                                   (gethash word words))
                                                 :kind kind)))
       stories))

(defun index-features (scored-features n &key (start 0))
  (mgl-example-util:log-msg "Total number of features: ~S~%"
                            (hash-table-count scored-features))
  (let* ((features->indices (mgl-util:index-scored-features scored-features n
                                                            :start start))
         (last-index (+ (1- (hash-table-count features->indices))
                        start))
         (last-feature (gethash last-index
                                (reverse-hash-table features->indices))))
    (mgl-example-util:log-msg "Cut off score: ~S~%"
                              (gethash last-feature scored-features))
    features->indices))

(defun most-frequent-features (documents mapper n &key (start 0))
  (index-features (count-features documents mapper)
                  n :start start))

(defun best-llr-features (documents mapper n &key (start 0))
  (index-features (mgl-util:compute-feature-llrs documents mapper #'story-label)
                  n :start start))


;;;; Common

(defclass mr-logging-trainer (logging-trainer) ())

(defmethod log-training-period ((trainer mr-logging-trainer) learner)
  (floor (length *training-stories*) 4))

(defmethod log-test-period ((trainer mr-logging-trainer) learner)
  (length *training-stories*))


;;;; DBN

(defun all-chunks (chunks)
  (let ((position (position :visible-only chunks)))
    (if position
        (append (subseq chunks 0 position) (subseq chunks (1+ position)))
        chunks)))

(defun hidden-chunks-only (chunks)
  (let ((position (position :visible-only chunks)))
    (subseq chunks 0 position)))

(defun layers->rbms (layers &key (class 'rbm))
  (loop for (v h) on layers
        when h
        collect (make-instance class
                               :visible-chunks (all-chunks v)
                               :hidden-chunks (hidden-chunks-only h)
                               :clouds (if (find 'inputs (all-chunks v)
                                                 :key #'name)
                                           nil
                                           #+nil
                                           '((:class factored-cloud
                                              :visible-chunk inputs
                                              :hidden-chunk f1
                                              :rank 10))
                                           nil))))

(defclass mr-dbn (dbn)
  ((rbms :initform (layers->rbms
                    (list (list (make-instance 'constant-chunk :name 'c0)
                                (make-instance 'softmax-chunk :name 'input-label
                                               :size 2 :group-size 2)
                                (make-instance 'constrained-poisson-chunk
                                               :name 'inputs
                                               ;; each input has its
                                               ;; own scale
                                               :scale (make-flt-array 0)
                                               :group-size *n-words*
                                               :size *n-words*))
                          (list (make-instance 'constant-chunk :name 'c1)
                                (make-instance 'sigmoid-chunk :name 'f1
                                               :size 100))
                          #+nil
                          (list (make-instance 'constant-chunk :name 'c2)
                                (make-instance 'sigmoid-chunk :name 'f2
                                               :size 100))
                          #+nil
                          (list (make-instance 'constant-chunk :name 'c3)
                                (make-instance 'sigmoid-chunk :name 'f3
                                               :size 400)))
                    :class 'mr-rbm))))

(defun make-mr-dbn (&key (max-n-stripes 1))
  (make-instance 'mr-dbn :max-n-stripes max-n-stripes))

(defclass mr-rbm (rbm) ())

(defun story-size (story)
  (loop for x across (story-array story) summing (cdr x)))

(defun find-dbn-input-labels (rbm)
  (let ((chunks-and-rbms ()))
    (dolist (rbm1 (rbms (dbn rbm)))
      (let ((chunk (find 'input-label (visible-chunks rbm1) :key #'name)))
        (when chunk
          (push (cons chunk rbm1) chunks-and-rbms)))
      (when (eq rbm1 rbm)
        (return)))
    chunks-and-rbms))

(defmacro do-dbn-input-labels (((label-chunk label-rbm) rbm) &body body)
  `(loop for (,label-chunk . ,label-rbm) in (find-dbn-input-labels ,rbm)
    do (progn ,@body)))

(defmethod mgl-train:set-input (samples (rbm mr-rbm))
  (let ((chunk (find 'inputs (visible-chunks rbm) :key #'name)))
    (when chunk
      (let ((nodes (storage (nodes chunk)))
            (scale (if (and (typep chunk 'exp-normalized-group-chunk)
                            (typep (scale chunk) 'flt-vector))
                       (scale chunk)
                       nil)))
        (loop for sample in samples
              for stripe upfrom 0
              do (destructuring-bind (story &key omit-label-p)
                     (if (listp sample) sample (list sample))
                   (declare (ignore omit-label-p))
                   (when scale
                     (setf (aref scale stripe) (story-size story)))
                   (with-stripes ((stripe chunk start end))
                     (clamp-story story nodes start end)))))))
  (let ((chunk (find 'input-label (visible-chunks rbm) :key #'name)))
    (when chunk
      (let ((nodes (storage (nodes chunk))))
        (loop for sample in samples
              for stripe upfrom 0
              do (destructuring-bind (sample &key omit-label-p)
                     (if (listp sample) sample (list sample))
                   (with-stripes ((stripe chunk start end))
                     (fill nodes #.(mgl-util:flt 0) :start start :end end)
                     (unless omit-label-p
                       (ecase (story-label sample)
                         ((-1) (setf (aref nodes (+ start 0)) (flt 1)))
                         ((1) (setf (aref nodes (+ start 1)) (flt 1))))))))))))

(defclass mr-rbm-trainer (mr-logging-trainer rbm-cd-trainer)
  ((counter :initform (make-instance 'rmse-counter) :reader counter)))

(defmethod initialize-trainer ((trainer mr-rbm-trainer) rbm)
  (call-next-method)
  (when (typep trainer 'bm-pcd-trainer)
    (setf (max-n-stripes (persistent-chains trainer))
          100
          #+nil
          (batch-size (elt (trainers trainer) 0)))
    (describe (persistent-chains trainer))
    (let ((inputs (find 'inputs (visible-chunks rbm) :key #'name)))
      (when inputs
        (let ((features (find 'f1 (hidden-chunks rbm) :key #'name)))
          (fill (storage (nodes features)) (flt 0.01)))
        (fill (scale inputs) (flt 128))))
    (format *trace-output* "n-stripes: ~S~%"
            (n-stripes (persistent-chains trainer)))))

(defun dbn-error (sampler rbm)
  (let* ((dbn (dbn rbm))
         (label-chunks-and-rbms (find-dbn-input-labels rbm))
         (sum-errors (make-array (length label-chunks-and-rbms)
                                 :initial-element 0))
         (n-errors 0)
         (max-n-stripes (max-n-stripes dbn)))
    (loop until (finishedp sampler) do
          (let ((samples (sample-batch sampler max-n-stripes)))
            (set-input samples rbm)
            (set-hidden-mean rbm)
            ;; this could be done until only LABEL-RBM
            (down-mean-field dbn :rbm rbm)
            (loop for (label-chunk . label-rbm) in label-chunks-and-rbms
                  for i upfrom 0
                  do
                  (let ((nodes (storage (nodes label-chunk))))
                    (loop
                     for sample in samples
                     for stripe upfrom 0
                     do (with-stripes ((stripe label-chunk start end))
                          (unless (= (story-label (if (listp sample)
                                                      (first sample)
                                                      sample))
                                     (decode-label nodes start end))
                            (incf (aref sum-errors i)))))))
            (incf n-errors (length samples))))
    (if (zerop n-errors)
        nil
        (map 'list (lambda (x)
                     (/ x n-errors))
             sum-errors))))

(defmethod log-training-error ((trainer mr-rbm-trainer) (rbm mr-rbm))
  (let ((counter (counter trainer))
        (n-inputs (n-inputs trainer)))
    (log-msg "TRAINING RMSE: ~,3E (~D)~%"
             (or (get-error counter) #.(flt 0))
             n-inputs)
    (reset-counter counter)))

(defmethod log-test-error ((trainer mr-rbm-trainer) (rbm mr-rbm))
  (let ((dbn (dbn rbm)))
    (log-msg "DBN TEST RMSE: ~{~,3E~^, ~} (~D)~%"
             (map 'list
                  #'get-error
                  (dbn-mean-field-errors
                   (make-sampler *test-stories*
                                 :omit-label-p t
                                 :max-n (length *test-stories*))
                   dbn :rbm rbm))
             (n-inputs trainer))
    (when (find-dbn-input-labels rbm)
      (let ((errors (dbn-error (make-sampler *test-stories*
                                             :omit-label-p t
                                             :max-n (length *test-stories*))
                               rbm)))
        (log-msg "DBN TEST CLASSIFICATION ACCURACY: ~{~,2F%~^, ~} (~D)~%"
                 (mapcar (lambda (e)
                           (* 100 (- 1 e)))
                         errors)
                 (n-inputs trainer))))))

(defmethod negative-phase :around (batch trainer (rbm mr-rbm))
  (call-next-method)
  (multiple-value-call #'add-error (counter trainer)
                       (reconstruction-error rbm)))


;;;; DBN training

(defun bias-cloud-p (cloud)
  (or (typep (chunk1 cloud) 'constant-chunk)
      (typep (chunk2 cloud) 'constant-chunk)))

(defclass mr-rbm-segment-trainer (batch-gd-trainer)
  ((mgl-gd:use-accumulator2 :initform t)))

(defun train-mr-dbn ()
  (let ((dbn (make-mr-dbn :max-n-stripes 100)))
    (loop for rbm in (rbms dbn)
          for i upfrom 0 do
          (log-msg "Starting to train level ~S RBM in DBN.~%" i)
          (train (make-sampler *training-stories*
                               :max-n (* (if (zerop i) 20 20)
                                         (length *training-stories*)))
                 (make-instance
                  'mr-rbm-trainer
                  :n-gibbs 1
                  :segmenter
                  (lambda (cloud)
                    (print cloud)
                    (multiple-value-bind (decay learning-rate)
                        (cond ((equal (name cloud) '((inputs f1) :a))
                               (values 0 0.0001))
                              ((equal (name cloud) '((inputs f1) :b))
                               (values 0.002 0.01))
                              ((bias-cloud-p cloud)
                               (values 0 0.01))
                              (t
                               (values 0.002 0.01)))
                      #+nil
                      (when (find 'input-label (name cloud))
                        (setq learning-rate (* 0.1 (flt learning-rate))))
                      (format t "~S ~S~%" learning-rate decay)
                      (make-instance 'mr-rbm-segment-trainer
                                     :learning-rate (flt learning-rate)
                                     :weight-decay (flt decay)
                                     :batch-size 10))))
                 rbm))
    dbn))


;;;; BPN

(defclass mr-bpn (bpn) ())

(defmethod set-input (stories (bpn mr-bpn))
  (let* ((inputs (find-lump (chunk-lump-name 'inputs nil) bpn :errorp t))
         (label (find-lump (chunk-lump-name 'input-label nil) bpn :errorp nil))
         (expectations (find-lump 'expectations bpn :errorp t))
         (inputs* (storage (nodes inputs)))
         (expectations* (storage (nodes expectations))))
    (when label
      (matlisp:fill-matrix (nodes label) (flt 0)))
    (loop for story in stories
          for stripe upfrom 0
          do (with-stripes ((stripe inputs inputs-start inputs-end)
                            (stripe expectations expectations-start
                                    expectations-end))
               (clamp-story story inputs* inputs-start inputs-end)
               (clamp-label (story-label story)
                            expectations*
                            expectations-start expectations-end)))))

(defun bpn-decode-labels (bpn lump-name)
  (let* ((predictions (find-lump lump-name bpn :errorp t))
         (nodes (storage (if (eq 'expectations lump-name)
                             (nodes predictions)
                             (softmax predictions)))))
    (loop for stripe below (n-stripes predictions)
          collect (with-stripes ((stripe predictions start end))
                    (decode-label nodes start end)))))

(defun classification-error (bpn)
  (values (- (n-stripes bpn)
             (loop for p in (bpn-decode-labels bpn 'predictions)
                   for e in (bpn-decode-labels bpn 'expectations)
                   count (= p e)))
          (n-stripes bpn)))

(defun bpn-error (sampler bpn)
  (let ((counter (make-instance 'error-counter))
        (ce-counter (make-instance 'error-counter))
        (n-stripes (max-n-stripes bpn)))
    (loop until (finishedp sampler) do
          (set-input (sample-batch sampler n-stripes) bpn)
          (forward-bpn bpn)
          (multiple-value-call #'add-error ce-counter (cost bpn))
          (multiple-value-call #'add-error counter (classification-error bpn)))
    (values (get-error counter)
            (get-error ce-counter))))


;;;; BPN training

(defclass mr-cg-bp-trainer (mr-logging-trainer cg-bp-trainer)
  ((cross-entropy-counter :initform (make-instance 'error-counter)
                          :reader cross-entropy-counter)
   (counter :initform (make-instance 'error-counter) :reader counter)))

(defclass mr-bp-trainer (mr-logging-trainer bp-trainer)
  ((cross-entropy-counter :initform (make-instance 'error-counter)
                          :reader cross-entropy-counter)
   (counter :initform (make-instance 'error-counter) :reader counter)))

(defmethod log-training-error (trainer (bpn mr-bpn))
  (let ((n-inputs (n-inputs trainer))
        (ce-counter (cross-entropy-counter trainer))
        (counter (counter trainer)))
    (log-msg "CROSS ENTROPY ERROR: ~,5F (~D)~%"
             (or (get-error ce-counter) #.(flt 0))
             n-inputs)
    (log-msg "CLASSIFICATION ACCURACY: ~,2F% (~D)~%"
             (* 100 (- 1 (or (get-error counter) #.(flt 0))))
             n-inputs)
    (reset-counter ce-counter)
    (reset-counter counter)))

(defmethod log-test-error (trainer (bpn mr-bpn))
  (multiple-value-bind (e ce)
      (bpn-error (make-sampler *test-stories*
                               :max-n (length *test-stories*))
                 bpn)
    (log-msg "TEST CROSS ENTROPY ERROR: ~,5F (~D)~%"
             ce (n-inputs trainer))
    (log-msg "TEST CLASSIFICATION ACCURACY: ~,2F% (~D)~%"
             (* 100 (- 1 e)) (n-inputs trainer))))

(defmethod compute-derivatives :around (samples (trainer mr-cg-bp-trainer) bpn)
  (let ((ce-counter (cross-entropy-counter trainer))
        (counter (counter trainer)))
    (call-next-method)
    (multiple-value-call #'add-error ce-counter (cost bpn))
    (multiple-value-call #'add-error counter (classification-error bpn))))

(defmethod train-batch :around (batch (trainer mr-cg-bp-trainer) bpn)
  (let ((ce-counter (cross-entropy-counter trainer))
        (counter (counter trainer)))
    (loop for samples in (group batch (max-n-stripes bpn)) do
          (set-input samples bpn)
          (forward-bpn bpn)
          (multiple-value-call #'add-error ce-counter (cost bpn))
          (multiple-value-call #'add-error counter (classification-error bpn)))
    (multiple-value-bind (best-w best-f
                                 n-line-searches n-succesful-line-searches
                                 n-evaluations)
        (call-next-method)
      (declare (ignore best-w))
      (log-msg "BEST-F: ~S, N-EVALUATIONS: ~S~%" best-f n-evaluations)
      (log-msg "N-LINE-SEARCHES: ~S (succesful ~S)~%"
               n-line-searches n-succesful-line-searches))))

(defun init-lump (name bpn deviation)
  (multiple-value-bind (array start end)
      (segment-weights (find-lump name bpn :errorp t))
    (loop for i upfrom start below end
          do (setf (aref array i) (flt (* deviation (gaussian-random-1)))))))

(defun unroll-mr-dbn (dbn &key (name (chunk-lump-name 'f3 nil)) (n-classes 2))
  (multiple-value-bind (defs clamps inits) (unroll-dbn dbn :bottom-up-only t)
    (print clamps)
    (print inits)
    (terpri)
    (let ((bpn (eval (print
                      `(build-bpn (:class 'mr-bpn)
                         ,@defs
                         ;; Add expectations
                         (expectations (input-lump :size ,n-classes))
                         ;; Add a softmax layer. Oh, the pain.
                         (prediction-weights (weight-lump
                                              :size (* (size (lump ',name))
                                                       ,n-classes)))
                         (prediction-biases (weight-lump :size ,n-classes))
                         (prediction-activations0
                          (activation-lump :weights prediction-weights
                                           :x (lump ',name)))
                         (prediction-activations
                          (->+ :args (list prediction-activations0
                                           prediction-biases)))
                         (predictions
                          (cross-entropy-softmax-lump
                           :group-size ,n-classes
                           :x prediction-activations
                           :target expectations))
                         (my-error (error-node :x predictions)))))))
      (initialize-bpn-from-dbn bpn dbn inits)
      (init-lump 'prediction-weights bpn 0.1)
      (init-lump 'prediction-biases bpn 0)
      bpn)))

(defun train-mr-bpn (bpn &key (n-softmax-batches 5) (n-whole-batches 0))
  (setf (max-n-stripes bpn) 1000)
  (log-msg "Starting to train the softmax layer of BPN~%")
  (train (make-sampler *training-stories*
                       :max-n (* n-softmax-batches (length *training-stories*)))
         (make-instance 'mr-cg-bp-trainer
                        :cg-args (list :max-n-line-searches 10)
                        :batch-size (length *training-stories*)
                        :segment-filter
                        (lambda (lump)
                          (or (eq (name lump) 'prediction-biases)
                              (eq (name lump) 'prediction-weights))))
         bpn)
  (log-msg "Starting to train the whole BPN~%")
  (train (make-sampler *training-stories*
                       :max-n (* n-whole-batches (length *training-stories*)))
         (make-instance 'mr-cg-bp-trainer
                        :cg-args (list :max-n-line-searches 3)
                        :batch-size (length *training-stories*))
         bpn)
  bpn)

(defun train-mr-bpn-gd (bpn)
  (flet ((softmax-segmenter (lump)
           (cond ((eq (name lump) 'prediction-weights)
                  (make-instance 'batch-gd-trainer
                                 :learning-rate (flt 0.0005)
                                 :weight-decay (flt 0.002)
                                 :momentum (flt 0.8)
                                 :batch-size 10))
                 ((eq (name lump) 'prediction-biases)
                  (make-instance 'batch-gd-trainer
                                 :learning-rate (flt 0.00015)
                                 :weight-decay (flt 0.002)
                                 :momentum (flt 0.8)
                                 :batch-size 10))
                 (t nil))))
    (log-msg "Starting to train the softmax layer of BPN~%")
    (train (make-sampler *training-stories*
                         :max-n (* 30 (length *training-stories*)))
           (make-instance 'mr-bp-trainer
                          :segmenter #'softmax-segmenter)
           bpn)
    (log-msg "Starting to train the whole BPN~%")
    (train (make-sampler *training-stories*
                         :max-n (* 30 (length *training-stories*)))
           (make-instance 'mr-bp-trainer
                          :segmenter
                          (lambda (lump)
                            (if (or (eq (name lump) 'prediction-biases)
                                    (and (listp (name lump))
                                         (member 'constant (name lump))))
                                (make-instance
                                 'batch-gd-trainer
                                 :learning-rate (flt 0.0002)
                                 :weight-decay (flt 0.002)
                                 :momentum (flt 0.8)
                                 :batch-size 10)
                                (make-instance
                                 'batch-gd-trainer
                                 :learning-rate (flt 0.0002)
                                 :weight-decay (flt 0.002)
                                 :momentum (flt 0.8)
                                 :batch-size 10))))
           bpn)
    bpn))


;;;;

(defun split-fold (fold n-folds &key (negative-stories *negative-stories*)
                   (positive-stories *positive-stories*))
  (assert (<= 0 fold (1- n-folds)))
  (flet ((in-fold-p (story)
           #+nil
           (= fold (floor (story-id story) (/ 1000 n-folds)))
           (= fold (mod (story-id story) n-folds))))
    (values
     (concatenate 'vector (remove-if #'in-fold-p negative-stories)
                  (remove-if #'in-fold-p positive-stories))
     (concatenate 'vector (remove-if-not #'in-fold-p negative-stories)
                  (remove-if-not #'in-fold-p positive-stories)))))

(defun funcall* (fn &rest args)
  (apply (read-from-string fn) args))

(defun make-problem (stories)
  (funcall* "libsvm:make-problem"
            (map 'vector #'story-label stories)
            (map 'vector #'story-array stories)))

(defun test-svm (&key (training-stories *training-stories*)
                 (test-stories *test-stories*))
  (loop
   for log-c upfrom 7 upto 9 do
   (let ((model (funcall* "libsvm:train"
                          (make-problem training-stories)
                          (funcall* "libsvm:make-parameter"
                                    :kernel-type :linear
                                    :shrinking nil
                                    :c (expt 2 log-c)))))
     (log-msg "SVM accuracy=~S (log(c)=~S)~%"
              (flt
               (/ (loop for story across test-stories
                        count (= (story-label story)
                                 (the (member -1d0 1d0)
                                   (funcall* "libsvm:predict"
                                             model (story-array story)))))
                  (length *test-stories*)))
              log-c))))

(defun train-all (&key (fold 0) (n-folds 5) unrollp)
  (setq *dbn* nil *words* nil *bpn* nil)
  (log-msg "Training fold ~S~%" fold)
  (unless (and (boundp '*negative-stories*) (boundp '*positive-stories*))
    (multiple-value-setq (*negative-stories* *positive-stories*)
      (load-stories)))
  (multiple-value-setq (*training-stories* *test-stories*)
    (split-fold fold n-folds))
  ;;(gethash 2 (mgl-util:reverse-hash-table *words*))
  (when (find-package '#:libsvm)
    (setq *words*
          (best-llr-features *training-stories* #'map-story-words *n-words*
                             :start 1)
          #+nil
          (most-frequent-features *training-stories* #'map-story-words *n-words*
                                  :start 1))
    (encode-all *training-stories* #'map-story-words *words*
                :kind :normalized-binary)
    (encode-all *test-stories* #'map-story-words *words*
                :kind :normalized-binary)
    (test-svm))
  (setq *words*
        (best-llr-features *training-stories* #'map-story-words *n-words*)
        #+nil
        (most-frequent-features *training-stories* #'map-story-words *n-words*))
  (encode-all *training-stories* #'map-story-words *words* :kind :binary)
  (encode-all *test-stories* #'map-story-words *words* :kind :binary)
  (setq *dbn* (train-mr-dbn))
  (when unrollp
    (setq *bpn* (unroll-mr-dbn *dbn* :name (chunk-lump-name 'f1 nil)))
    (train-mr-bpn *bpn*)))

#|

(loop for i upfrom 0 below 10
      do (train-all :fold i :n-folds 10 :unrollp t))

|#
