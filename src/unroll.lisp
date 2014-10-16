;;;; Unrolling RBMs to backprop network

(in-package :mgl-unroll)

;;; Intermediate representation of lumps during the unrolling.
(defstruct lumpy
  ;; At this depth can this lump be found in the unrolled RBM. Depth 0
  ;; is the middle: the hidden layer of the topmost rbm. Depth 1 is
  ;; the visible of layer of the topmost rbm, depth -1 is the
  ;; reconstruction of the same.
  depth
  chunk
  (incomings '())
  ;; The lumpy with a non-negative depth from which this was mirrored
  ;; or nil. In a DBN being unrolled, the lumpy for the reconstruction
  ;; of a chunk has the lumpy of the chunk as its ORIGINAL. In a DBM
  ;; being unrolled, the reconstruction is really the marginals of the
  ;; approximate posterior in the paper and it is to be clamped.
  original
  ;; :RECONSTRUCTION is the reconstruction in a DBN, :MAP is the
  ;; marginals of the approximate posterior in a DBM.
  (kind nil :type (member nil :reconstruction :map))
  ;; This symbol is bound to the added lump when creating the network.
  (symbol (gensym)))

(defstruct incoming
  from-lumpy
  cloud
  transposep)

(defun chunk-lump-name (chunk-name kind)
  "The name of the lump that represents CHUNK."
  `(:chunk ,chunk-name ,@(when kind (list kind))))

(defun chunk-activation-lump-name (chunk-name kind)
  "The name of the lump that computes the activations of CHUNK."
  (append (chunk-lump-name chunk-name kind) '(:activation)))

(defun cloud-weight-lump-name (cloud-name transposep)
  "The name of the lump that represents the weights of CLOUD or its
  transpose. CLOUD comes from the rbm in the DBN."
  `(:cloud ,cloud-name ,@(when transposep '(:transpose))))

(defun cloud-linear-lump-name (cloud-name transposep)
  "The name of the lump that represents part of the activation of a
  chunk. CLOUD comes from the rbm in the DBN. TRANSPOSEP determines
  from which direction the activation crosses the cloud."
  (append (cloud-weight-lump-name cloud-name transposep) '(:linear)))

(defgeneric chunk->bpn-definition (chunk sym name size activation-symbol)
  (:documentation "Return a bpn definition form (that is a list of
  lump definition forms) for CHUNK that takes a single activation
  parameter given by the symbol ACTIVATION-SYMBOL with NAME and SIZE.
  Only called for non-conditioning chunks. Second value is a list of
  clamp inits, the third is a list of inits.")
  (:method ((chunk sigmoid-chunk) sym name size activation-symbol)
    `((,sym (->sigmoid :name ',name :x ,activation-symbol))))
  (:method ((chunk gaussian-chunk) sym name size activation-symbol)
    ;; this is identity
    `((,sym (->+ :name ',name :args (list ,activation-symbol)))))
  (:method ((chunk relu-chunk) sym name size activation-symbol)
    `((,sym (->rectified :name ',name :x ,activation-symbol))))
  (:method ((chunk exp-normalized-group-chunk) sym name size activation-symbol)
    (let ((exp-symbol (gensym)))
      (values
       `((,exp-symbol (->exp :x ,activation-symbol))
         (,sym (->normalized :name ',name
                :scale ,(scale chunk) :group-size ,(group-size chunk)
                :x ,exp-symbol)))
       `((:from-lump ,name :to-lump ,exp-symbol))))))

(defun lumpy-name (lumpy)
  (chunk-lump-name (name (lumpy-chunk lumpy)) (lumpy-kind lumpy)))

(defun lumpy-activation-name (lumpy)
  (chunk-activation-lump-name (name (lumpy-chunk lumpy)) (lumpy-kind lumpy)))

;;; Find CHUNK at DEPTH in LUMPIES.
(defun find-lumpy (lumpies &key (depth nil depthp) chunk kind)
  (find-if (lambda (lumpy)
             (and (or (not depthp) (eql depth (lumpy-depth lumpy)))
                  (eq chunk (lumpy-chunk lumpy))
                  (eq kind (lumpy-kind lumpy))))
           lumpies))

(defun find-lumpy-by-name (name lumpies)
  (find name lumpies :key #'lumpy-name :test #'equal))

(defgeneric incoming->bpn-defintion (from-lumpy to-lumpy cloud transposep)
  (:documentation "Return a list of four elemenets. The first is a
  list of lump definitions that represent the flow from FROM-LUMPY
  through CLOUD. The chunk of FROM-LUMPY may be either of the end
  points of CLOUD. The third values is the cloud inits. The third is
  name of the `end' lump.")
  (:method (from-lumpy to-lumpy (cloud full-cloud) transposep)
    (let* ((from-chunk (lumpy-chunk from-lumpy))
           (from-size (size from-chunk))
           (n-weights (mat-size (weights cloud)))
           (weight-symbol (gensym))
           (weight-name (cloud-weight-lump-name (name cloud) transposep))
           (size (/ n-weights from-size))
           (linear-symbol (gensym))
           (linear-name (cloud-linear-lump-name (name cloud) transposep)))
      (list `((,weight-symbol
               (->weight
                :name ',weight-name
                :size ,n-weights))
              (,linear-symbol
               (->activation
                :name ',linear-name
                :size ,size
                :weights ,weight-symbol
                :x ,(lumpy-symbol from-lumpy)
                :transpose-weights-p ,transposep)))
            `((:from-lump ,(if transposep
                               (lumpy-name to-lumpy)
                               (lumpy-name from-lumpy))
               :to-lump ,weight-name :to :rows)
              (:from-lump ,(lumpy-name to-lumpy) :to-lump ,linear-name))
            `((:cloud-name ,(name cloud)
               :weight-name ,weight-name))
            linear-symbol)))
  (:method (from-lumpy to-lumpy (cloud factored-cloud) transposep)
    ;; We have two clouds: B goes from visible to shared and A from
    ;; shared to hidden. So it's Hidden = AxBxVisible or Visible =
    ;; B'*A'*Hidden.
    (let* ((cloud-a (cloud-a cloud))
           (cloud-b (cloud-b cloud))
           (weight-b-symbol (gensym))
           (weight-a-symbol (gensym))
           (shared-symbol (gensym))
           (linear-symbol (gensym))
           (weight-base-name (cloud-weight-lump-name (name cloud) transposep))
           (weight-b-name (list weight-base-name :b))
           (weight-a-name (list weight-base-name :a))
           (shared-name (list
                         (cloud-linear-lump-name (name cloud) transposep)
                         :shared))
           (linear-name (cloud-linear-lump-name (name cloud) transposep)))
      (list `((,weight-b-symbol (->weight
                                 :name ',weight-b-name
                                 :size ,(mat-size (weights cloud-b))))
              (,weight-a-symbol (->weight
                                 :name ',weight-a-name
                                 :size ,(mat-size (weights cloud-a))))
              (,shared-symbol (->activation
                               :name ',shared-name
                               :size ,(rank cloud)
                               :weights ,(if transposep
                                             weight-a-symbol
                                             weight-b-symbol)
                               :x ,(lumpy-symbol from-lumpy)
                               :transpose-weights-p ,transposep))
              (,linear-symbol (->activation
                               :name ',linear-name
                               :size ,(size (lumpy-chunk to-lumpy))
                               :weights ,(if transposep
                                             weight-b-symbol
                                             weight-a-symbol)
                               :x ,shared-symbol
                               :transpose-weights-p ,transposep)))
            ;; cloud inits
            `((:cloud-name ,(name cloud)
               :weight-b-name ,weight-b-name
               :weight-a-name ,weight-a-name))
            linear-symbol))))

(defun incoming-list->bpn-definition (to-lumpy incomings)
  (let ((x (loop for incoming in incomings
                 collect (incoming->bpn-defintion
                          (incoming-from-lumpy incoming)
                          to-lumpy
                          (incoming-cloud incoming)
                          (incoming-transposep incoming)))))
    (values (mapcan #'first x)
            (mapcan #'second x)
            (mapcan #'third x)
            (mapcar #'fourth x))))

(defun lumpies->bpn-definition (lumpies)
  (let ((lumpies (stable-sort lumpies #'> :key #'lumpy-depth))
        (defs '())
        (clamps '())
        (inits '()))
    (dolist (lumpy lumpies)
      (let* ((chunk (lumpy-chunk lumpy))
             (incomings (lumpy-incomings lumpy))
             (name (lumpy-name lumpy))
             (activation-symbol (gensym))
             (activation-name (lumpy-activation-name lumpy))
             (size (size chunk)))
        (cond ((typep chunk 'constant-chunk)
               (assert (endp incomings))
               (push `(,(lumpy-symbol lumpy)
                       (->constant
                        :name ',name :size ,size
                        :default-value ,(default-value chunk)))
                     defs))
              ((or (typep chunk 'conditioning-chunk)
                   (endp incomings))
               (assert (endp incomings))
               (push `(,(lumpy-symbol lumpy)
                       (->input :name ',name :size ,size))
                     defs))
              (t
               (multiple-value-bind (cloud-defs cloud-clamps cloud-inits
                                     linear-symbols)
                   (incoming-list->bpn-definition lumpy incomings)
                 (push-all cloud-defs defs)
                 (push-all cloud-clamps clamps)
                 (push-all cloud-inits inits)
                 (push `(,activation-symbol
                         (->+ :name ',activation-name
                              :args (list ,@linear-symbols)))
                       defs)
                 (push `(:from-lump ,name :to-lump ,activation-name)
                       clamps))
               (multiple-value-bind (chunk-defs chunk-clamps chunk-inits)
                   (chunk->bpn-definition chunk (lumpy-symbol lumpy)
                                          name size activation-symbol)
                 (push-all chunk-defs defs)
                 (push-all chunk-clamps clamps)
                 (push-all chunk-inits inits))))))
    (values (reverse defs) inits)))

(defun add-connection (cloud &key from to)
  (assert (> (lumpy-depth from) (lumpy-depth to)))
  (assert (not (member from (lumpy-incomings to)
                       :key #'incoming-from-lumpy)))
  (assert (not (typep (lumpy-chunk to) 'conditioning-chunk)))
  (push (make-incoming :from-lumpy from
                       :cloud cloud
                       :transposep (eq (lumpy-chunk from)
                                       (chunk2 cloud)))
        (lumpy-incomings to)))

(defun ensure-lumpy (lumpies &key depth chunk kind)
  (or (find-lumpy lumpies :depth depth :chunk chunk :kind kind)
      (make-lumpy :depth depth
                  :chunk chunk
                  :kind kind
                  :original (if kind
                                (find-lumpy lumpies
                                            :chunk chunk
                                            :kind nil)
                                nil))))

(defun unroll-dbn (dbn &key bottom-up-only)
  "Unroll DBN recursively and turn it into a feed-forward
  backpropagation network. A single RBM in DBN of the form VISIBLE <->
  HIDDEN is transformed into a VISIBLE -> HIDDEN ->
  RECONSTRUCTION-OF-VISIBLE network. While the undirected connection
  <-> has a common weight matrix for both directions, in the backprop
  network the weights pertaining to ->'s are distinct but are
  initialized from the same <-> (with one being the tranpose of it).

  If BOTTOM-UP-ONLY then don't generate the part of the network that
  represents the top-down flow, that is, skip the reconstructions.

  Return backprop network lump definition forms, as the second value
  `inits': initialization specifications suitable for
  INITIALIZE-BPN-FROM-BM.

  If there is no corresponding chunk in the layer below or there is no
  rbm below then the chunk is translated into an INPUT lump. Desired
  outputs and error node are not added. The first element of RMBS is
  the topmost one (last of the DBN), the one that goes into the middle
  of the backprop network."
  (let ((lumpies '()))
    (flet ((ensure-lumpy (depth chunk)
             (let ((lumpy (ensure-lumpy lumpies
                                        :depth depth :chunk chunk
                                        :kind (if (minusp depth)
                                                  :reconstruction
                                                  nil))))
               (pushnew lumpy lumpies)
               lumpy)))
      (loop for rbm in (reverse (rbms dbn))
            for depth upfrom 0
            do
            (do-clouds (cloud rbm)
              (let ((visible-chunk
                     (cloud-chunk-among-chunks cloud (visible-chunks rbm)))
                    (hidden-chunk
                     (cloud-chunk-among-chunks cloud (hidden-chunks rbm))))
                (unless (typep hidden-chunk 'conditioning-chunk)
                  (add-connection cloud
                                  :from (ensure-lumpy (1+ depth) visible-chunk)
                                  :to (ensure-lumpy depth hidden-chunk)))
                (unless (or (typep visible-chunk 'conditioning-chunk)
                            bottom-up-only)
                  ;; If the chunk does not need activations then it is
                  ;; in effect not reconstructed, so let's use the
                  ;; non-mirrored version. Thus the quasi-symmetry of
                  ;; the code and the generated network is broken.
                  (let ((hidden-depth
                         (if (typep hidden-chunk 'conditioning-chunk)
                             depth
                             (- depth))))
                    (add-connection cloud
                                    :from (ensure-lumpy hidden-depth
                                                        hidden-chunk)
                                    :to (ensure-lumpy (- (1+ depth))
                                                      visible-chunk)))))))
      (lumpies->bpn-definition lumpies))))

(defun unroll-dbm (dbm &key (chunks (chunks dbm)) (map chunks)
                   (reconstruction ()))
  (let ((lumpies '()))
    (flet ((ensure-lumpy (depth chunk &optional kind)
             (let ((lumpy (ensure-lumpy lumpies
                                        :depth depth :chunk chunk :kind kind)))
               (pushnew lumpy lumpies)
               lumpy)))
      (loop for (lower-layer higher-layer) on (layers dbm)
            while higher-layer
            for clouds in (rest (clouds-up-to-layers dbm))
            for lower-depth downfrom (1- (length (layers dbm)))
            do
            (dolist (cloud clouds)
              (let ((lower-chunk (cloud-chunk-among-chunks cloud lower-layer))
                    (higher-chunk (cloud-chunk-among-chunks cloud higher-layer))
                    (higher-depth (1- lower-depth)))
                (when (and lower-chunk higher-chunk)
                  ;; Normal and :MAP lumpies.
                  (when (and (member lower-chunk chunks)
                             (member higher-chunk chunks))
                    (unless (typep higher-chunk 'conditioning-chunk)
                      (add-connection
                       cloud
                       :from (ensure-lumpy lower-depth lower-chunk)
                       :to (ensure-lumpy higher-depth higher-chunk)))
                    (when (and (member lower-chunk map)
                               (not (typep lower-chunk 'conditioning-chunk)))
                      ;; Add the marginals of the approximate
                      ;; posterior as an input.
                      (let ((lower-lumpy
                              (ensure-lumpy lower-depth lower-chunk)))
                        ;; If it has no connections from below then
                        ;; it's an input, so don't add the :MAP
                        ;; connection.
                        (when (lumpy-incomings lower-lumpy)
                          (add-connection
                           cloud
                           :from (ensure-lumpy (1+ lower-depth) higher-chunk
                                               :map)
                           :to lower-lumpy)))))
                  ;; :RECONSTRUCTION lumpies (higher -> lower).
                  (when (and (or (member higher-chunk reconstruction)
                                 (zerop higher-depth))
                             (member lower-chunk reconstruction)
                             (not (typep lower-chunk 'conditioning-chunk)))
                    (add-connection
                     cloud
                     :from (ensure-lumpy (- higher-depth) higher-chunk
                                         (if (zerop higher-depth)
                                             nil
                                             :reconstruction))
                     :to (ensure-lumpy (- lower-depth) lower-chunk
                                       :reconstruction)))))))
      (lumpies->bpn-definition lumpies))))

(defgeneric initialize-from-cloud (bpn cloud args)
  (:method (bpn (cloud full-cloud) args)
    (destructuring-bind (&key weight-name) args
      (let* ((lump (find-lump weight-name bpn :errorp t))
             (weights (weights cloud))
             (nodes (nodes lump)))
        (unless (= (mat-size weights) (mat-size nodes))
          (error "Cannot initialize lump ~S from cloud ~S: size mismatch"
                 lump cloud))
        (copy! (weights cloud) nodes))))
  (:method (bpn (cloud factored-cloud) args)
    (destructuring-bind (&key weight-b-name weight-a-name) args
      (initialize-from-cloud bpn (cloud-b cloud)
                             (list :weight-name weight-b-name))
      (initialize-from-cloud bpn (cloud-a cloud)
                             (list :weight-name weight-a-name)))))

(defun initialize-bpn-from-bm (bpn bm inits)
  "Initialize BPN from the weights of BM according to cloud INITS that
  was returned by UNROLL-DBN or UNROLL-DBM."
  (dolist (init inits)
    (multiple-value-bind (known unknown)
        (split-plist init '(:cloud-name))
      (destructuring-bind (&key cloud-name) known
        (let ((cloud (find-cloud cloud-name bm :errorp t)))
          (initialize-from-cloud bpn cloud unknown))))))


;;;; BPN setup

(defun set-dropout-and-rescale-activation-weights (lump dropout bpn)
  "Set the dropout of LUMP to DROPOUT. Find the activation lump to
  which LUMP is fed and rescale its weights to compensate. There must
  be exactly on such activation lump or this function will fail."
  (assert (null (dropout lump)))
  (setf (slot-value lump 'dropout) dropout)
  (let ((scale (/ (- (flt 1) dropout)))
        (activation-lumps
          (loop for lump-1 across (lumps bpn)
                when (and (typep lump-1 '->activation)
                          (eq lump (mgl-bp::x lump-1))
                          (typep (mgl-bp::weights lump-1) '->weight))
                  collect lump-1)))
    (assert (/= 0 (length activation-lumps)) ()
            "Can't rescale activation weights for LUMP there is none.")
    (assert (= 1 (length activation-lumps)) ()
            "Can't rescale activation weights for LUMP there are too many.")
    (let ((lump-1 (first activation-lumps)))
      (scal! scale (nodes (mgl-bp::weights lump-1))))))


;;;; SET-INPUT support for BPN converted from a DBM with MAP lumps

(defun collect-map-chunks-and-lumps (bpn dbm)
  "Return a list of chunk, lump sublists. Elements are MAP lumps in
  BPN and the corresponding chunk in DBM."
  (let ((chunks-and-lumps ()))
    (dolist (chunk (set-difference (hidden-chunks dbm)
                                   (conditioning-chunks dbm)))
      (let* ((lump-name (chunk-lump-name (name chunk) :map))
             (lump (find-lump lump-name bpn)))
        (when lump
          (push (list chunk lump) chunks-and-lumps))))
    chunks-and-lumps))

(defclass bpn-clamping-cache ()
  ((clamping-cache
    :initform (make-hash-table)
    :reader clamping-cache)
   (populate-key
    :initform #'identity
    :initarg :populate-key
    :reader populate-key)
   (populate-convert-to-dbm-sample-fn
    :initform #'identity
    :initarg :populate-convert-to-dbm-sample-fn
    :reader populate-convert-to-dbm-sample-fn)
   (populate-map-cache-lazily-from-dbm
    :initform nil
    :type (or null dbm)
    :initarg :populate-map-cache-lazily-from-dbm
    :reader populate-map-cache-lazily-from-dbm)
   (populate-periodic-fn
    :initform nil
    :initarg :populate-periodic-fn
    :reader populate-periodic-fn))
  (:documentation "This slot is a sample -> (lump array)* list hash
  table. Inherit from this and set input will clamp the arrays to the
  respective lumps for the right sample."))

(defun populate-map-cache (bpn dbm samples &key (key (populate-key bpn))
                           (convert-to-dbm-sample-fn
                            (populate-convert-to-dbm-sample-fn bpn))
                           (if-exists :skip)
                           (periodic-fn (populate-periodic-fn bpn)))
  "Populate the CLAMPING-CACHE of the MAP lumps of BPN unrolled from
  DBM. The values for the MAP lumps are taken from mean field of the
  correspending chunk of the DBM. What happens when the cache already
  has an entry for a sample is determined by IF-EXISTS: if :SKIP, the
  default, the cache is unchanged; if :SUPERSEDE, the cache entry is
  replaced by the calculated contents; if :APPEND, the new (lump
  array) entries are appended to the existing ones; if :ERROR, an
  error is signalled."
  (let ((sampler (make-sequence-sampler samples
                                        :max-n-samples (length samples)))
        (cache (clamping-cache bpn))
        (map-chunks-and-lumps (collect-map-chunks-and-lumps bpn dbm)))
    (do-batches-for-model (samples (sampler dbm))
      (when periodic-fn
        (call-periodic-fn (hash-table-count cache) periodic-fn
                          (hash-table-count cache)))
      (cond ((eq if-exists :skip)
             (setq samples
                   (remove-if (lambda (sample)
                                (gethash (funcall key sample) cache))
                              samples)))
            ((eq if-exists :error)
             (when (some (lambda (sample)
                           (gethash (funcall key sample) cache))
                         samples)
               (error "Cache entry already exists."))))
      (when samples
        (set-input (map 'list convert-to-dbm-sample-fn samples) dbm)
        (set-hidden-mean dbm)
        (loop for sample in samples
              for stripe upfrom 0 do
                (let ((k (funcall key sample))
                      (x ()))
                  (loop for (chunk lump) in map-chunks-and-lumps
                        do
                           (with-stripes ((stripe chunk chunk-start chunk-end))
                             (let* ((n (- chunk-end chunk-start))
                                    (xxx (make-mat n :ctype flt-ctype))
                                    (nodes (nodes chunk)))
                               (with-shape-and-displacement (nodes n
                                                             chunk-start)
                                 (copy! nodes xxx))
                               (push (list lump xxx) x))))
                  (setf (gethash k cache)
                        (ecase if-exists
                          ((:skip :supersede) x)
                          ((:append) (append (gethash k cache) x))))))))
    (when periodic-fn
      (call-periodic-fn (hash-table-count cache) periodic-fn
                        (hash-table-count cache)))))

(defun clamp-cached-entry-on-bpn (bpn stripe sample &key
                                  (key (populate-key bpn)))
  (let ((cache (clamping-cache bpn))
        (k (funcall key sample)))
    (unless (nth-value 1 (gethash k cache))
      (error "No clamping cache entry for ~S" k))
    (loop for (lump map-nodes) in (gethash k cache) do
      (with-stripes ((stripe lump lump-start lump-end))
        (let ((n (- lump-end lump-start))
              (nodes (nodes lump)))
          (assert (= (mat-size map-nodes) n))
          (with-shape-and-displacement (nodes n lump-start)
            (copy! map-nodes nodes)))))))

(defmethod set-input :before (samples (bpn bpn-clamping-cache))
  (when (populate-map-cache-lazily-from-dbm bpn)
    (populate-map-cache bpn (populate-map-cache-lazily-from-dbm bpn) samples
                        :if-exists :skip))
  (loop for stripe upfrom 0
        for sample in samples
        do (clamp-cached-entry-on-bpn bpn stripe sample)))
