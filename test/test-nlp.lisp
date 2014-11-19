(in-package :mgl-test)

(defun test-make-n-gram-mappee ()
  (assert (equal (let ((r ()))
                   (map nil (make-n-gram-mappee (lambda (x)
                                                  (push x r))
                                                3)
                        '(0 1 2 3))
                   (reverse r))
                 '((0 1 2) (1 2 3)))))

(defun test-blue ()
  (assert (equal (multiple-value-list (mgl:bleu '(((1 2 3 4) (1 2 3 4))
                                                  ((a b) (1 2)))
                                                :candidate-key #'first
                                                :reference-key #'second))
                 '(0.8408964 1.0 (2/3 3/4 1 1)))))

;;; This can be used to compare to multi-bleu.perl.
#+nil
(defun multi-bleu-perl (corpus &key candidate-key reference-key)
  (let ((candidate-file
          (cl-fad:with-output-to-temporary-file (stream)
            (map nil (lambda (sentence)
                       (format stream "~A~%" (funcall candidate-key sentence)))
                 corpus)))
        (reference-file
          (cl-fad:with-output-to-temporary-file (stream)
            (map nil (lambda (sentence)
                       (format stream "~A~%" (funcall reference-key sentence)))
                 corpus))))
    (unwind-protect
         (with-output-to-string (*standard-output*)
           (external-program:run "/usr/bin/perl"
                                 (list "multi-bleu.perl"
                                       (namestring reference-file))
                                 :input (namestring candidate-file)
                                 :output *standard-output*
                                 :error *error-output*))
      (ignore-errors (delete-file candidate-file))
      (ignore-errors (delete-file reference-file)))))

#+nil
(multi-bleu-perl '(("a b c d" "a b c d"))
                 :candidate-key #'first :reference-key #'second)

(defun test-nlp ()
  (test-make-n-gram-mappee)
  (test-blue))
