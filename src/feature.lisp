(in-package :mgl-util)

(defun count-features (documents mapper)
  "Return scored features as an EQUAL hash table whose keys are
features of DOCUMENTS and values are counts of occurrences of
features. MAPPER takes a function and a document and calls function
with features of the document."
  (let ((features (make-hash-table :test #'equal)))
    (map nil (lambda (document)
               (funcall mapper
                        (lambda (feature)
                          (incf (gethash feature features 0)))
                        document))
         documents)
    features))

(defun document-features (document mapper)
  (let ((features (make-hash-table :test #'equal)))
    (funcall mapper
             (lambda (feature)
               (setf (gethash feature features) t))
             document)
    features))

(defun all-document-classes (documents class-fn)
  (let ((r ()))
    (map nil (lambda (document)
               (pushnew (funcall class-fn document) r))
         documents)
    r))

(defun compute-feature-llrs (documents mapper class-fn
                             &key (classes
                                   (all-document-classes documents class-fn)))
  "Return scored features as an EQUAL hash table whose keys are
features of DOCUMENTS and values are their log likelihood ratios.
MAPPER takes a function and a document and calls function with
features of the document."
  (when (< (length classes) 2)
    (error "LLR feature selection needs at least 2 classes."))
  (flet ((document-class-index (document)
           (let ((class (funcall class-fn document)))
             (or (position class classes)
                 (error "Unexpected class ~S" class)))))
    (let ((all (make-hash-table :test #'equal)))
      (map nil (lambda (document)
                 (let ((index (document-class-index document)))
                   (maphash (lambda (feature -)
                              (incf
                               (first
                                (or (gethash feature all)
                                    (setf (gethash feature all)
                                          (make-list (1+ (length classes))
                                                     :initial-element 0)))))
                              (incf (elt (gethash feature all) (1+ index))))
                            (document-features document mapper))))
           documents)
      (let ((class-counts
             (loop for class in classes
                   collect (count class documents
                                  :key (lambda (document)
                                         (funcall class-fn document)))))
            (total (length documents)))
        (assert (= total (loop for x in class-counts sum x)))
        (maphash (lambda (feature counts)
                   (destructuring-bind (count &rest feature-class-counts)
                       counts
                     (assert (= count (loop for x in feature-class-counts
                                            sum x)))
                     (let ((k1 (map 'vector (lambda (x)
                                              (+ 0.01 x))
                                    feature-class-counts))
                           (k2 (map 'vector
                                    (lambda (x y)
                                      (+ 0.01 (- x y)))
                                    class-counts
                                    feature-class-counts)))
                       (setf (gethash feature all)
                             (multinomial-log-likelihood-ratio k1 k2)))))
                 all)
        all))))

(defun index-scored-features (feature-scores n &key (start 0))
  "Take scored features as a feature -> score hash table \(returned by
COUNT-FEATURES or COMPUTE-FEATURE-LLR, for instance) and return a
feature -> index hash table that maps the first N \(or less) features
with the highest scores to distinct dense indices starting from
START."
  (let ((sorted (stable-sort (hash-table->vector feature-scores)
                             #'> :key #'cdr)))
    (flet ((vector->hash-table (v)
             (let ((h (make-hash-table :test #'equal)))
               (loop for x across v
                     for i upfrom start
                     do (setf (gethash (car x) h) i))
               h)))
      (vector->hash-table (subseq* sorted 0 n)))))

(defun read-indexed-features (stream)
  (with-standard-io-syntax
    (alist->hash-table (read stream) :test #'equal)))

(defun write-indexed-features (features->indices stream)
  (with-standard-io-syntax
    (prin1 (hash-table->alist features->indices) stream)))


;;;; Encoding

(defun encode/bag-of-words (document mapper feature->index &key (kind :binary))
  "Return a sparse vector that represents the encoded DOCUMENT. Get
the features of DOCUMENT from MAPPER, convert each feature to an index
by FEATURE->INDEX. FEATURE->INDEX may return NIL if the feature is not
used. The result is a vector of index/value conses. Indexes are unique
within the vector and are in increasing order. Depending on KIND value
is calculated differently: for :FREQUENCY it is the number of times
the corresponding feature was found in DOCUMENT, for :BINARY it is
always 1. :NORMALIZED-FREQUENCY and :NORMALIZED-FREQUENCY are like the
unnormalized counterparts except that as the final step values in the
assembled sparse vector are normalized to sum to 1."
  (assert (member kind '(:binary :frequency
                         :normalized-binary :normalized-frequency)))
  (let ((v (make-array 20 :adjustable t :fill-pointer 0)))
    (funcall mapper
             (lambda (feature)
               (let ((index (funcall feature->index feature)))
                 (when index
                   (let ((pos (position index v :key #'car)))
                     (if pos
                         (incf (cdr (aref v pos)))
                         (vector-push-extend (cons index #.(flt 1)) v))))))
             document)
    (when (member kind '(:binary :normalized-binary))
      (loop for x across v
            do (setf (cdr x) #.(flt 1))))
    (when (member kind '(:normalized-binary :normalized-frequency))
      (let ((sum (loop for x across v summing (cdr x))))
        (map-into v (lambda (x)
                      (cons (car x)
                            (/ (cdr x) sum)))
                  v)))
    (sort v #'< :key #'car)))
