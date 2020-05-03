(in-package :mgl)

;;;; Register in PAX World

(defun pax-sections ()
  (list @mgl-manual))
(defun pax-pages ()
  `((:objects
     (, @mgl-manual)
     :source-uri-fn ,(make-github-source-uri-fn
                      :mgl
                      "https://github.com/melisgl/mgl"))))
(register-doc-in-pax-world :mgl (pax-sections) (pax-pages))

#+nil
(progn
  (update-asdf-system-readmes @mgl-manual :mgl)
  (update-asdf-system-html-docs @mgl-manual :mgl
                                :pages (pax-pages)))
