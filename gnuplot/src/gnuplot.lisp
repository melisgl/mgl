(in-package :mgl-gnuplot)

;;;; Data model

(defclass command ()
  ()
  (:documentation "A gnuplot command. There are separate classes for
  plot/splot, etc."))

(defclass plot ()
  ((kind :type (member :2d :3d) :initform :2d :initarg :kind :reader kind)
   (mappings :initform () :initarg :mappings :accessor mappings)))

(defclass mapping ()
  ((options :type (or list string) :initform nil :initarg :options
            :reader options))
  (:documentation "A plot can contain multiple mappings. For example
  in

    plot sin(x) title 'sine', \"datafile\" with lines

  there is a function and a file mapping. Subclasses of MAPPING
  represent the function or the data and the options (e.g. 'title',
  'with' above)."))

(defclass function-mapping (mapping)
  ((expression :initarg :expression :reader function-expression)))

(defclass data-mapping (mapping)
  ((data :initarg :data :reader mapping-data))
  (:documentation "Consider plot '-', sin(x). For gnuplot '-'
  indicates inline data that will be read from the command stream
  after all mappings are read. The DATA slot of this class holds an
  object that can be serialized by WRITE-DATA."))

(defclass file-mapping (mapping)
  ((filename :initarg :filename :reader filename))
  (:documentation "When the data comes from a file. As in plot
  'filename' title 'something'."))


;;;; Serialization interface

(defgeneric write-command (command stream)
  (:documentation "Write the gnuplot COMMAND to STREAM. Commands
  represented by the classes PLOT, SET-COMMAND, etc."))

(defgeneric write-mapping (mapping stream)
  (:documentation "Write the gnuplot MAPPING to STREAM."))

(defgeneric write-data (data stream)
  (:documentation "Serialize DATA to STREAM in gnuplot format."))


;;;; Serialization

(defmethod write-command ((command string) stream)
  (write-line command stream))

(defun kind-to-string (kind)
  (ecase kind
    ((:2d) "plot")
    ((:3d) "splot")))

(defmethod write-command ((plot plot) stream)
  (write-string (kind-to-string (kind plot)) stream)
  (let ((firstp t))
    (dolist (mapping (mappings plot))
      (write-string (if firstp " " ", ") stream)
      (write-mapping mapping stream)
      (setf firstp nil)))
  (terpri stream)
  ;; Now write the inline data sections one by one.
  (dolist (mapping (mappings plot))
    (when (typep mapping 'data-mapping)
      (write-data mapping stream)
      ;; Write the end of inline data section marker.
      (write-line "e" stream))))

(defun write-options (options stream)
  (labels ((foo (x)
             (if (listp x)
                 (map nil #'foo x)
                 (format stream " ~A" x))))
    (foo options)))

(defmethod write-mapping ((mapping mapping) stream)
  (write-options (options mapping) stream))

;;;; Function mapping serialization

(defmethod write-mapping ((mapping function-mapping) stream)
  (format stream "~A" (function-expression mapping))
  (call-next-method))

(defmethod write-data ((mapping function-mapping) stream)
  nil)

;;;; File mapping serialization

(defmethod write-mapping ((mapping file-mapping) stream)
  (format stream "~S" (namestring (filename mapping))))

(defmethod write-data ((mapping file-mapping) stream)
  nil)

;;;; Data mapping serialization

(defmethod write-mapping ((mapping data-mapping) stream)
  (write-mapping (mapping-data mapping) stream)
  (call-next-method))

(defmethod write-data ((mapping data-mapping) stream)
  (write-data (mapping-data mapping) stream))

;;;; Raw array serialization

(defmethod write-mapping ((array array) stream)
  (format stream "'-'"))

(defmethod write-data ((array array) stream)
  (let* ((dimensions (array-dimensions array))
         (n-cols (cond ((= 1 (length dimensions))
                        1)
                       ((= 2 (length dimensions))
                        (elt dimensions 1))
                       (t
                        (error "Don't know how to write array of dimensions ~S."
                               dimensions)))))
    (dotimes (i (array-total-size array))
      (format stream " ~,F" (row-major-aref array i))
      (when (zerop (mod (1+ i) n-cols))
        (terpri stream)))
    t))


;;;; List serialization

(defmethod write-mapping ((list list) stream)
  (format stream "'-'"))

(defmethod write-data ((list list) stream)
  (dolist (x list)
    (dolist (e (alexandria:ensure-list x))
      (format stream " ~,F" e))
    (terpri stream))
  t)


(defvar *command-stream* nil
  "The default stream to which commands and inline data are written by
  WRITE-COMMAND.")

(defmacro with-command-stream ((stream) &body body)
  "Binds *COMMAND-STREAM* to STREAM routing all command output to
  STREAM by default."
  `(let ((*command-stream* ,stream))
     ,@body))



;;;; Invoking gnuplot

(defvar *gnuplot-binary* "/usr/bin/gnuplot")

(defun start-session (&key (binary *gnuplot-binary*) display geometry
                      (persistp t) (output *standard-output*)
                      (error *error-output*))
  (external-program:process-input-stream
   (pipe-to-gnuplot :stream :binary binary :display display :geometry geometry
                    :persistp persistp :output output :error error)))

(defun end-session (stream)
  (close stream))

(defmacro with-session ((&key display geometry (persistp t)
                         (output '*standard-output*) (error '*error-output*))
                        &body body)
  "Start gnuplot, bind STREAM and *COMMAND-STREAM* to its standard
  input. The stream is closed when BODY exits."
  `(with-command-stream ((start-session :display ,display :geometry ,geometry
                                        :persistp ,persistp :output ,output
                                        :error ,error))
     (unwind-protect
          (progn ,@body)
       (end-session *command-stream*))))

(defun pipe-to-gnuplot (input &key (binary *gnuplot-binary*) display geometry
                        persistp output error)
  (flet ((foo (input)
           (external-program:start binary
                                   (append (cond (display
                                                  (list "-display" display))
                                                 ;; workaround for gnuplot bug
                                                 (geometry
                                                  (list "-display" ":0.0"))
                                                 (t
                                                  ()))
                                           (if geometry
                                               (list "-geometry" geometry)
                                               ())
                                           (if persistp '("-p") ()))
                                   :input input :output output :error error)))
    (if (stringp input)
        (with-input-from-string (stream input)
          (foo stream))
        (foo input))))


;;;; Convenience

(defun command (command)
  (write-command command *command-stream*))

(defun plot* (mappings)
  (command (make-instance 'plot
                          :kind :2d
                          :mappings mappings)))

(defun splot* (mappings)
  (command (make-instance 'plot
                          :kind :3d
                          :mappings mappings)))

(defun fn* (expression options)
  (make-instance 'function-mapping :expression expression :options options))

(defun data* (data options)
  (make-instance 'data-mapping :data data :options options))

(defun file* (filename options)
  (make-instance 'file-mapping :filename filename :options options))

(defmacro plot (() &body mappings)
  `(plot* (list ,@mappings)))

(defmacro splot (() &body mappings)
  `(splot* (list ,@mappings)))

(defmacro fn (expression &rest options)
  `(fn* ,expression (list ,@options)))

(defmacro data (data &rest options)
  `(data* ,data (list ,@options)))

(defmacro file (filename &rest options)
  `(file* ,filename (list ,@options)))

#|

(with-session (:geometry "2000x2000")
  (let ((*command-stream* (make-broadcast-stream *command-stream*
                                                 *standard-output*)))
    (command
     (make-instance 'plot
                    :mappings (list
                               (make-instance 'function-mapping
                                              :options "title 'sine'"
                                              :expression "sin(x)")
                               (make-instance
                                'data-mapping
                                :data
                                '((1 2)
                                  (2 3)
                                  (3 -1))
                                #+nil
                                (make-array '(3 2)
                                            :initial-contents '((1 2)
                                                                (2 3)
                                                                (3 -1)))))))))

(with-session ()
  (let ((*command-stream* (make-broadcast-stream *command-stream*
                                                 *standard-output*)))
    (command "set xrange [1:5]")
    (command "set yrange [-3:3]")
    (command "set view map")
    (plot ()
      (data '((1 2) (2 3) (3 -1)) "title 'stuff'")
      (fn "sin(x) title 'sine'")
      (fn "cos(x)" "title 'cosine'")
      #+nil
      (file "/tmp/xxx" "title 'cosine'"))))

(with-session ()
  (let ((*command-stream* (make-broadcast-stream *command-stream*
                                                 *standard-output*)))
    (command "set view map")
    (splot ()
      (data '((0 0 1) (0 1 2) (1 0 -1) (1 1 0))
            "title 'stuff' with image")
      (fn "sin(x)+cos(y)"))))

|#
