(in-package :mgl-visuals)

(defgeneric graph-node-type (x y)
  (:method (x (chunk chunk))
    (cl-ppcre:regex-replace-all "-CHUNK"
                                (string (class-name (class-of chunk)))
                                ""))
  (:method (x (lump lump))
    (cl-ppcre:regex-replace-all "-LUMP|->"
                                (string (class-name (class-of lump)))
                                "")))

(defgeneric graph-node-fields (x y)
  (:method-combination append :most-specific-last))

(defmethod graph-node-fields append (x y)
  (list (if (= 1 (size y))
            (graph-node-type x y)
            (format nil "~A (~A)" (graph-node-type x y) (size y)))
        (prin1-to-string (name y))))

(defun cl-dot-escape (string)
  (cl-ppcre:regex-replace-all "([^a-zA-Z0-9])" string "\\\\\\1"))

(defun graph-node-label (x y)
  (format nil "{~{~A~^|~}}" (mapcar #'cl-dot-escape (graph-node-fields x y))))


;;;; RBM, DBN, DBM support

(defmethod graph-node-fields append (x (chunk constant-chunk))
  (unless (= (flt 1) (default-value chunk))
    (list (format nil "VALUE: ~,5E" (default-value chunk)))))

(defmethod graph-node-fields append (x (chunk mgl-bm::normalized-group-chunk))
  `(,(format nil "GROUP-SIZE: ~S" (group-size chunk))
    ,@(unless (= (flt 1) (scale chunk))
        (list (format nil "SCALE: ~,5E" (scale chunk))))))

(defmethod cl-dot:graph-object-node (x (chunk chunk))
  (make-instance 'cl-dot:node
                 :attributes `(:label ,(graph-node-label x chunk)
                               :shape :record
                               :style :filled
                               :color :black
                               :fillcolor ,(cond
                                             ((typep chunk 'constant-chunk)
                                              "#555555")
                                             ((typep chunk 'conditioning-chunk)
                                              "#888888")
                                             ((member chunk (visible-chunks x)) 
                                              "#AAAAAA")
                                             (t
                                              "#DDDDDD")))))

(defmethod cl-dot:graph-object-points-to (x (chunk chunk))
  (let ((neighbours ()))
    (flet ((add (cloud chunk dir label)
             (declare (ignore cloud))
             (push (make-instance 'cl-dot:attributed
                                  :object chunk
                                  :attributes `(:dir ,dir
                                                ,@(when label
                                                    (list :label label))))
                   neighbours)))
      (dolist (cloud (clouds x) neighbours)
        (when (eq chunk (chunk1 cloud))
          (add cloud (chunk2 cloud)
               (cond ((typep (chunk1 cloud) 'conditioning-chunk)
                      :forward)
                     ((typep (chunk2 cloud) 'conditioning-chunk)
                      :back)
                     (t :none))
               (if (or (/= (flt 1) (mgl-bm::scale1 cloud))
                       (/= (flt 1) (mgl-bm::scale2 cloud)))
                   (format nil "~A/~A" (float (mgl-bm::scale1 cloud) 0.0)
                           (float (mgl-bm::scale2 cloud) 0.0))
                   nil)))))))


;;;; BPN

(defmethod graph-node-fields append (x (lump constant-lump))
  (unless (= (flt 1) (default-value lump))
    (list (format nil "VALUE: ~,5E" (default-value lump)))))

(defmethod graph-node-fields append (x (lump normalized-lump))
  `(,(format nil "GROUP-SIZE: ~S" (group-size lump))
    ,@(unless (= (flt 1) (scale lump))
        (list (format nil "SCALE: ~,5E" (scale lump))))))

(defmethod cl-dot:graph-object-node ((bpn bpn) (lump lump))
  (make-instance 'cl-dot:node
                 :attributes `(:label ,(graph-node-label bpn lump)
                               :shape :record
                               :style :filled
                               :color :black
                               :fillcolor ,(cond
                                             ((typep lump 'constant-lump)
                                              "#555555")
                                             ((typep lump 'weight-lump)
                                              "#888888")
                                             ((typep lump 'input-lump)
                                              "#AAAAAA")
                                             (t
                                              "#DDDDDD")))))

(defmethod cl-dot:graph-object-points-to ((bpn bpn) (lump normalized-lump))
  (list (make-instance 'cl-dot:attributed
                       :object (mgl-bp::x lump)
                       :attributes '(:dir :back))))

(defmethod graph-node-fields append (x (lump activation-lump))
  (when (transpose-weights-p lump)
    (list (format nil "TRANSPOSE: ~S" (transpose-weights-p lump)))))

(defmethod cl-dot:graph-object-points-to ((bpn bpn) (lump activation-lump))
  (list (make-instance 'cl-dot:attributed
                       :object (mgl-bp::x lump)
                       :attributes '(:dir :back))
        (make-instance 'cl-dot:attributed
                       :object (mgl-bp::weights lump)
                       :attributes '(:dir :back
                                     :label "weights"))))

(defmethod cl-dot:graph-object-points-to ((bpn bpn) (lump ->+))
  (mapcar (lambda (lump)
            (make-instance 'cl-dot:attributed
                           :object lump
                           :attributes '(:dir :back)))
          (mgl-bp::args lump)))

(defmethod cl-dot:graph-object-points-to ((bpn bpn) (lump ->sum))
  (list (make-instance 'cl-dot:attributed
                       :object (mgl-bp::x lump)
                       :attributes '(:dir :back))))

(defmethod cl-dot:graph-object-points-to ((bpn bpn) (lump ->linear))
  (list (make-instance 'cl-dot:attributed
                       :object (mgl-bp::x lump)
                       :attributes '(:dir :back))
        (make-instance 'cl-dot:attributed
                       :object (mgl-bp::y lump)
                       :attributes '(:dir :back))))

(defmethod cl-dot:graph-object-points-to ((bpn bpn) (lump ->sigmoid))
  (list (make-instance 'cl-dot:attributed
                       :object (mgl-bp::x lump)
                       :attributes '(:dir :back))))

(defmethod cl-dot:graph-object-points-to ((bpn bpn) (lump ->exp))
  (list (make-instance 'cl-dot:attributed
                       :object (mgl-bp::x lump)
                       :attributes '(:dir :back))))

(defmethod cl-dot:graph-object-points-to ((bpn bpn) (lump ->sum-squared-error))
  (list (make-instance 'cl-dot:attributed
                       :object (mgl-bp::x lump)
                       :attributes '(:dir :back))
        (make-instance 'cl-dot:attributed
                       :object (mgl-bp::y lump)
                       :attributes '(:dir :back))))

(defmethod graph-node-fields append (x (lump cross-entropy-softmax-lump))
  (list (format nil "GROUP-SIZE: ~S" (group-size lump))))

(defmethod cl-dot:graph-object-points-to ((bpn bpn)
                                          (lump cross-entropy-softmax-lump))
  (list (make-instance 'cl-dot:attributed
                       :object (mgl-bp::x lump)
                       :attributes '(:dir :back))
        (make-instance 'cl-dot:attributed
                       :object (target lump)
                       :attributes '(:dir :back
                                     :label "target"))))

#|

;;; Example usage:
(let* ((dbm (make-instance
             'dbm
             :layers (list
                      (list (make-instance 'constant-chunk :name 'c0)
                            (make-instance 'sigmoid-chunk :name 'inputs
                                           :size (* 28 28)))
                      (list (make-instance 'constant-chunk :name 'c1)
                            (make-instance 'sigmoid-chunk :name 'f1
                                           :size 500)
                            (make-instance 'softmax-label-chunk :name 'label
                                           :size 10 :group-size 10))
                      (list (make-instance 'constant-chunk :name 'c2)
                            (make-instance 'sigmoid-chunk :name 'f2
                                           :size 1000)))
             :clouds '(:merge
                       (:chunk1 c0 :chunk2 label :class nil)
                       (:chunk1 inputs :chunk2 label :class nil))))
       (dbn (dbm->dbn dbm))
       (bpn (eval `(build-bpn () ,@(unroll-dbm dbm)))))
  (let ((dgraph (cl-dot:generate-graph-from-roots dbn (chunks dbn)
                                                  '(:rankdir "BT"))))
    (cl-dot:dot-graph dgraph
                      (asdf-system-relative-pathname "src/visuals/test-dbn.png")
                      :format :png))
  (let ((dgraph (cl-dot:generate-graph-from-roots dbm (chunks dbm)
                                                  '(:rankdir "BT"))))
    (cl-dot:dot-graph dgraph
                      (asdf-system-relative-pathname "src/visuals/test-dbm.png")
                      :format :png))
  (let ((dgraph (cl-dot:generate-graph-from-roots bpn (lumps bpn)
                                                  '(:rankdir "BT"))))
    (cl-dot:dot-graph dgraph
                      (asdf-system-relative-pathname "src/visuals/test-bpn.png")
                      :format :png)))

|#
