;;; -*- Mode: lisp; Syntax: ansi-common-lisp; Package: :matlisp; Base: 10 -*-
;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;
;;; Copyright (c) 2000 The Regents of the University of California.
;;; All rights reserved. 
;;; 
;;; Permission is hereby granted, without written agreement and without
;;; license or royalty fees, to use, copy, modify, and distribute this
;;; software and its documentation for any purpose, provided that the
;;; above copyright notice and the following two paragraphs appear in all
;;; copies of this software.
;;; 
;;; IN NO EVENT SHALL THE UNIVERSITY OF CALIFORNIA BE LIABLE TO ANY PARTY
;;; FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES
;;; ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF
;;; THE UNIVERSITY OF CALIFORNIA HAS BEEN ADVISED OF THE POSSIBILITY OF
;;; SUCH DAMAGE.
;;;
;;; THE UNIVERSITY OF CALIFORNIA SPECIFICALLY DISCLAIMS ANY WARRANTIES,
;;; INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
;;; MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE
;;; PROVIDED HEREUNDER IS ON AN "AS IS" BASIS, AND THE UNIVERSITY OF
;;; CALIFORNIA HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES,
;;; ENHANCEMENTS, OR MODIFICATIONS.
;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;
;;; Originally written by Raymond Toy.
;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;
;;; $Id: print.lisp,v 1.8 2003/05/31 23:05:39 rtoy Exp $
;;;
;;; $Log: print.lisp,v $
;;; Revision 1.8  2003/05/31 23:05:39  rtoy
;;; Indent more typically.
;;;
;;; Revision 1.7  2001/06/22 12:52:41  rtoy
;;; Use ALLOCATE-REAL-STORE and ALLOCATE-COMPLEX-STORE to allocate space
;;; instead of using the error-prone make-array.
;;;
;;; Revision 1.6  2001/04/28 13:06:46  rtoy
;;; This is not Fortran.  Instead of printing *'s when the number won't
;;; fit in the given field, print them out!
;;;
;;; Revision 1.5  2001/02/21 19:33:34  simsek
;;; o Added the formatting hack *matrix-indent*.
;;;
;;; Revision 1.4  2000/07/11 18:02:03  simsek
;;; o Added credits
;;;
;;; Revision 1.3  2000/07/11 02:11:56  simsek
;;; o Added support for Allegro CL
;;;
;;; Revision 1.2  2000/05/08 17:19:18  rtoy
;;; Changes to the STANDARD-MATRIX class:
;;; o The slots N, M, and NXM have changed names.
;;; o The accessors of these slots have changed:
;;;      NROWS, NCOLS, NUMBER-OF-ELEMENTS
;;;   The old names aren't available anymore.
;;; o The initargs of these slots have changed:
;;;      :nrows, :ncols, :nels
;;;
;;; Revision 1.1  2000/04/14 00:11:12  simsek
;;; o This file is adapted from obsolete files 'matrix-float.lisp'
;;;   'matrix-complex.lisp' and 'matrix-extra.lisp'
;;; o Initial revision.
;;;
;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;;; Routines for printing a matrix nicely.

(in-package "MATLISP")

#+nil (export '(*print-matrix*
	  print-element))

(defvar *print-matrix*
  5
  "Maximum number of columns and/or rows to print.  Set this to NIL to
  print no elements (same as *PRINT-ARRAY* set to NIL).  Set this to T
  to print all elements of the matrix.

  This is useful for preventing printing of huge matrices by accident.")

(defun set-print-limits-for-matrix (n m)
  (declare (type fixnum n m))
  (if (eq *print-matrix* t)
      (values n m)
    (if (eq *print-matrix* nil)
	(values 0 0)
      (if (and (integerp *print-matrix*)
	       (> *print-matrix* 0))
	  (values (min n *print-matrix*)
		  (min m *print-matrix*))
	(error "Cannot set the print limits for matrix.
Required that *PRINT-MATRIX* be T,NIL or a positive INTEGER,
but got *PRINT-MATRIX* of type ~a"
	       (type-of *print-matrix*))))))
      

(defgeneric print-element (matrix
			    element
			    stream)
  (:documentation "
 Syntax
 ======
 (PRINT-ELEMENT matrix element stream)

 Purpose
 =======
 This generic function is specialized to MATRIX to
 print ELEMENT to STREAM.  Called by PRINT-MATRIX
 to format a matrix to STREAM.
"))

(defmethod print-element ((matrix standard-matrix)
			  element
			  stream)
  (format stream "~a" element))

(defmethod print-element ((matrix real-matrix)
			  element
			  stream)
  (format stream "~11,5,,,,,'Eg" element))

(defmethod print-element ((matrix complex-matrix)
			  element
			  stream)
  
  (let ((realpart (realpart element))
	(imagpart (imagpart element)))

    (if (zerop imagpart)
	(format stream "      ~11,4,,,,,'Eg      " realpart)
      (format stream "#C(~11,4,,,,,'Ee ~11,4,,,,,'Ee)" 
	      realpart
	      imagpart))))

(defvar *matrix-indent* 0
  "Determines how many spaces will be printed before each row 
   of a matrix (default 0)")

(defun print-matrix (matrix stream)
  (with-slots (number-of-rows number-of-cols)
      matrix
    (multiple-value-bind (max-n max-m)
	(set-print-limits-for-matrix number-of-rows number-of-cols)
      (declare (type fixnum max-n max-m))
      (format stream " ~d x ~d" number-of-rows number-of-cols)

      (decf max-n)
      (decf max-m)  
      (flet ((print-row (i)
	       (when (minusp i)
		 (return-from print-row))
	       (format stream "~%   ")
		  
	       (dotimes (k *matrix-indent*)
		 (format stream " "))
	       (dotimes (j max-m)
		 (declare (type fixnum j))
		 (print-element matrix 
				(matrix-ref matrix i j)
				stream)
		 (format stream " "))
	       (if (< max-m (1- number-of-cols))
		   (progn
		     (format stream "... ")
		     (print-element matrix 
				    (matrix-ref matrix i (1- number-of-cols))
				    stream)
		     (format stream " "))
		   (if (< max-m number-of-cols)
		       (progn
			 (print-element matrix 
					(matrix-ref matrix i (1- number-of-cols))
					stream)
			 (format stream " "))))))
	   
	(dotimes (i max-n)
	  (declare (type fixnum i))
	  (print-row i))
	   
	(if (< max-n (1- number-of-rows))
	    (progn
	      (format stream "~%     :")
	      (print-row (1- number-of-rows)))
	    (if (< max-n number-of-rows)
		(print-row (1- number-of-rows))))))))


(defmethod print-object ((matrix standard-matrix) stream)
  (print-unreadable-object (matrix stream :type t :identity (not *print-matrix*))
    (when *print-matrix*
      (print-matrix matrix stream))))

#+nil
(defmethod print-object ((matrix standard-matrix) stream)
  (format stream "#<~a" (type-of matrix))
  (if *print-matrix*
      (print-matrix matrix stream)
    (format stream "{~x}" (kernel:get-lisp-obj-address matrix)))
  (format stream " >~%"))

#+nil
(defmethod print-object ((matrix real-matrix) stream)
  (format stream "#<~a" (type-of matrix))
  (if *print-matrix*
      (print-matrix matrix stream)
    (format stream "{~x}" (kernel:get-lisp-obj-address matrix)))
  (format stream " >~%"))

#+nil
(defmethod print-object ((matrix complex-matrix) stream)
  (format stream "#<~a" (type-of matrix))
  (if *print-matrix*
      (print-matrix matrix stream)
    (format stream "{~x}" (kernel:get-lisp-obj-address matrix)))
  (format stream " >~%"))
