(cl:defpackage :mgl-gnuplot
  (:use #:common-lisp)
  ;; convenience
  (:export
   #:command
   #:plot*
   #:splot*
   #:fn*
   #:data*
   #:file*
   #:plot
   #:splot
   #:fn
   #:data
   #:file)
  ;; I/O
  (:export
   #:*command-stream*
   #:with-command-stream)
  ;; interactive use
  (:export
   #:*gnuplot-binary
   #:start-session
   #:end-session
   #:with-session)
  ;; data model
  (:export
   #:command
   #:plot
   #:mapping
   #:function-mapping
   #:data-mapping
   #:file-mapping)
  ;; serialization interface
  (:export
   #:write-command
   #:write-mapping
   #:write-data)
  (:documentation "Minimalistic, interactive or batch mode gnuplot
interface that supports multiplots and inline data."))
