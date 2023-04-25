(in-package :mgl-core)

(named-readtables:in-readtable pythonic-string-reader:pythonic-string-syntax)

(defsection @mgl-features (:title "Features")
  (@mgl-feature-selection section)
  (@mgl-feature-encoding section))


(defsection @mgl-feature-selection (:title "Feature Selection")
  "The following _scoring functions_ all return an EQUAL hash table
  that maps features to scores."
  (count-features function)
  (feature-llrs function)
  (feature-disambiguities function))

(defun count-features (documents mapper &key (key #'identity))
  """Return scored features as an EQUAL hash table whose keys are
  features of DOCUMENTS and values are counts of occurrences of
  features. MAPPER takes a function and a document and calls function
  with features of the document.

  ```cl-transcript
  (sort (alexandria:hash-table-alist
         (count-features '(("hello" "world")
                           ("this" "is" "our" "world"))
                         (lambda (fn document)
                           (map nil fn document))))
        #'string< :key #'car)
  => (("hello" . 1) ("is" . 1) ("our" . 1) ("this" . 1) ("world" . 2))
  ```"""
  (let ((features (make-hash-table :test #'equal)))
    (flet ((foo (document)
             (let ((document (funcall key document)))
               (funcall mapper
                        (lambda (feature)
                          (incf (gethash feature features 0)))
                        document))))
      (if (typep documents 'sequence)
          (map nil #'foo documents)
          (funcall documents #'foo)))
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

(defun feature-llrs (documents mapper class-fn
                             &key (classes
                                   (all-document-classes documents class-fn)))
  """Return scored features as an EQUAL hash table whose keys are
  features of DOCUMENTS and values are their log likelihood ratios.
  MAPPER takes a function and a document and calls function with
  features of the document.

  ```cl-transcript
  (sort (alexandria:hash-table-alist
         (feature-llrs '((:a "hello" "world")
                         (:b "this" "is" "our" "world"))
                       (lambda (fn document)
                         (map nil fn (rest document)))
                       #'first))
        #'string< :key #'car)
  => (("hello" . 2.6032386) ("is" . 2.6032386) ("our" . 2.6032386)
      ("this" . 2.6032386) ("world" . 4.8428774e-8))
  ```"""
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

(defun feature-disambiguities (documents mapper class-fn
                               &key (classes
                                     (all-document-classes documents class-fn)))
  "Return scored features as an EQUAL hash table whose keys are
  features of DOCUMENTS and values are their _disambiguities_. MAPPER
  takes a function and a document and calls function with features of
  the document.

  From the paper 'Using Ambiguity Measure Feature Selection Algorithm
  for Support Vector Machine Classifier'."
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
                     (setf (gethash feature all)
                           (/ (apply #'max feature-class-counts)
                              (+ 10 count)))))
                 all)
        all))))


(defsection @mgl-feature-encoding (:title "Feature Encoding")
  "Features can rarely be fed directly to algorithms as is, they need
  to be transformed in some way. Suppose we have a simple language
  model that takes a single word as input and predicts the next word.
  However, both input and output is to be encoded as float vectors of
  length 1000. What we do is find the top 1000 words by some
  measure (see @MGL-FEATURE-SELECTION) and associate these words with
  the integers in \\[0..999] (this is `ENCODE`ing). By using for
  example [one-hot](http://en.wikipedia.org/wiki/One-hot) encoding, we
  translate a word into a float vector when passing in the input. When
  the model outputs the probability distribution of the next word, we
  find the index of the max and find the word associated with it (this
  is `DECODE`ing)"
  (encode generic-function)
  (decode generic-function)
  (encoder/decoder class)
  (make-indexer function)
  "Also see MGL-NLP::@MGL-NLP-BAG-OF-WORDS.")

(defgeneric encode (encoder decoded)
  (:documentation "Encode DECODED with ENCODER. This interface is
  generic enough to be almost meaningless. See ENCODER/DECODER for a
  simple, MGL-NLP:BAG-OF-WORDS-ENCODER for a slightly more involved
  example.

  If ENCODER is a function designator, then it's simply `FUNCALL`ed
  with DECODED.")
  (:method ((encoder symbol) decoded)
    (funcall encoder decoded))
  (:method ((encoder function) decoded)
    (funcall encoder decoded)))

(defgeneric decode (decoder encoded)
  (:documentation "Decode ENCODED with ENCODER. For an DECODER /
  ENCODER pair, `(DECODE DECODER (ENCODE ENCODER OBJECT))` must be
  equal in some sense to `OBJECT`.

  If DECODER is a function designator, then it's simply `FUNCALL`ed
  with ENCODED.")
  (:method ((decoder symbol) encoded)
    (funcall decoder encoded))
  (:method ((decoder function) encoded)
    (funcall decoder encoded)))

;;; Not exported currently.
(defun index-scored-features (scored-features n &key (start 0))
  "Take scored features as a feature -> score hash table (returned by
  COUNT-FEATURES or COMPUTE-FEATURE-LLR, for instance) and return a
  feature -> index hash table that maps the first N (or less) features
  with the highest scores to distinct dense indices starting from
  START."
  (let ((sorted (stable-sort (hash-table->vector scored-features)
                             #'> :key #'cdr)))
    (flet ((vector->hash-table (v)
             (let ((h (make-hash-table :test #'equal)))
               (loop for x across v
                     for i upfrom start
                     do (setf (gethash (car x) h) i))
               h)))
      (vector->hash-table (subseq* sorted 0 n)))))

(defclass encoder/decoder ()
  ((encodings :initarg :encodings :reader encodings)
   (decodings :initarg :decodings :reader decodings))
  (:documentation """Implements O(1) ENCODE and DECODE by having an
  internal decoded-to-encoded and an encoded-to-decoded EQUAL hash
  table. ENCODER/DECODER objects can be saved and loaded (see
  @MGL-PERSISTENCE) as long as the elements in the hash tables have
  read/write consitency.

  ```cl-transcript
  (let ((indexer
          (make-indexer
           (alexandria:alist-hash-table '(("I" . 3) ("me" . 2) ("mine" . 1)))
           2)))
    (values (encode indexer "I")
            (encode indexer "me")
            (encode indexer "mine")
            (decode indexer 0)
            (decode indexer 1)
            (decode indexer 2)))
  => 0
  => 1
  => NIL
  => "I"
  => "me"
  => NIL
  ```"""))

(defmethod encode ((indexer encoder/decoder) object)
  (gethash object (encodings indexer)))

(defmethod decode ((indexer encoder/decoder) index)
  (gethash index (decodings indexer)))

(defmethod write-state* ((indexer encoder/decoder) stream context)
  (with-standard-io-syntax
    (format stream "~S~%"
            (cons (hash-table-test (encodings indexer))
                  (alexandria:hash-table-alist (encodings indexer))))))

(defmethod read-state* ((indexer encoder/decoder) stream context)
  (destructuring-bind (test . alist) (with-standard-io-syntax
                                       (let ((*read-eval* nil))
                                         (read stream)))
    (setf (slot-value indexer 'encodings)
          (alexandria:alist-hash-table alist :test test))
    (setf (slot-value indexer 'decodings)
          (reverse-map (encodings indexer)))))

(defun make-indexer (scored-features n &key (start 0) (class 'encoder/decoder))
  "Take the top N features from SCORED-FEATURES (see
  @MGL-FEATURE-SELECTION), assign indices to them starting from START.
  Return an ENCODER/DECODER (or another CLASS) that converts between
  objects and indices."
  (let ((encodings (index-scored-features scored-features n :start start)))
    (make-instance class :encodings encodings
                   :decodings (reverse-map encodings))))

(defun reverse-map (hash-table)
  (let ((result (make-hash-table :test #'equal)))
    (maphash (lambda (key value)
               (setf (gethash value result) key))
             hash-table)
    result))
