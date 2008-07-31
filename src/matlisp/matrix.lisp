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
;;; Originally written by Raymond Toy
;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;
;;; $Id: matrix.lisp,v 1.14 2004/05/24 16:34:22 rtoy Exp $
;;;
;;; $Log: matrix.lisp,v $
;;; Revision 1.14  2004/05/24 16:34:22  rtoy
;;; More SBCL support from Robert Sedgewick.  The previous SBCL support
;;; was incomplete.
;;;
;;; Revision 1.13  2003/05/31 22:20:26  rtoy
;;; o Add some support for CMUCL with Gerd's PCL so we can inline
;;;   accessors and such for the matrix classes.
;;; o Only use one, system-independent, standard-matrix class.
;;; o Try to declare the types of the slots of the matrix classes
;;; o FORTRAN-MATRIX-INDEXING changed to use fixnum arithmetic.
;;;
;;; Revision 1.12  2003/03/09 14:26:30  rtoy
;;; Forgot one more :type 'fixnum bug.  From Gerd Moellmann.
;;;
;;; Revision 1.11  2003/02/19 21:59:52  rtoy
;;; Correct the slot type declarations.
;;;
;;; Revision 1.10  2001/10/29 18:00:28  rtoy
;;; Updates from M. Koerber to support QR routines with column pivoting:
;;;
;;; o Add an integer4 type and allocate-integer4-store routine.
;;; o Add the necessary Fortran routines
;;; o Add Lisp interface to the Fortran routines
;;; o Update geqr for the new routines.
;;;
;;; Revision 1.9  2001/06/22 12:51:49  rtoy
;;; o Added ALLOCATE-REAL-STORE and ALLOCATE-COMPLEX-STORE functions to
;;;   allocate appropriately sized arrays for holding real and complex
;;;   matrix elements.
;;; o Use it to allocate space.
;;;
;;; Revision 1.8  2000/10/04 15:56:50  simsek
;;; o Fixed bug in (MAKE-COMPLEX-MATRIX n)
;;;
;;; Revision 1.7  2000/07/11 18:02:03  simsek
;;; o Added credits
;;;
;;; Revision 1.6  2000/07/11 02:11:56  simsek
;;; o Added support for Allegro CL
;;;
;;; Revision 1.5  2000/05/11 18:28:10  rtoy
;;; After the great standard-matrix renaming, row-vector-p and
;;; col-vector-p were swapped.
;;;
;;; Revision 1.4  2000/05/11 18:02:55  rtoy
;;; o After the great standard-matrix renaming, I forgot a few initargs
;;;   that needed to be changed
;;; o MAKE-REAL-MATRIX and MAKE-COMPLEX-MATRIX didn't properly handle
;;;   things like #(1 2 3 4) and #((1 2 3 4)).  Make them accept these.
;;;
;;; Revision 1.3  2000/05/08 17:19:18  rtoy
;;; Changes to the STANDARD-MATRIX class:
;;; o The slots N, M, and NXM have changed names.
;;; o The accessors of these slots have changed:
;;;      NROWS, NCOLS, NUMBER-OF-ELEMENTS
;;;   The old names aren't available anymore.
;;; o The initargs of these slots have changed:
;;;      :nrows, :ncols, :nels
;;;
;;; Revision 1.2  2000/05/05 21:35:16  simsek
;;; o Fixed row-vector-p and col-vector-p
;;;
;;; Revision 1.1  2000/04/14 00:11:12  simsek
;;; o This file is adapted from obsolete files 'matrix-float.lisp'
;;;   'matrix-complex.lisp' and 'matrix-extra.lisp'
;;; o Initial revision.
;;;
;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;;; Definitions of STANDARD-MATRIX, REAL-MATRIX, COMPLEX-MATRIX.

(in-package "MATLISP")

#+nil (export '(real-matrix
	  complex-matrix
	  standard-matrix
	  real-matrix-element-type
	  real-matrix-store-type
	  complex-matrix-element-type
	  complex-matrix-store-type
	  #|
	  n
	  m
	  nxm
	  |#
	  nrows
	  ncols
	  number-of-elements
	  row-vector-p
	  col-vector-p
	  row-or-col-vector-p
	  square-matrix-p
	  size
	  fortran-matrix-indexing
	  fortran-complex-matrix-indexing
	  complex-coerce
	  fill-matrix
	  make-real-matrix-dim
	  make-real-matrix
	  make-complex-matrix-dim
	  make-complex-matrix))

(eval-when (load eval compile)
(deftype integer4-matrix-element-type ()
  '(signed-byte 32))

(deftype real-matrix-element-type () 
  "The type of the elements stored in a REAL-MATRIX"
  'double-float)

(deftype real-matrix-store-type (size) 
  "The type of the storage structure for a REAL-MATRIX"
  `(simple-array double-float ,size))

(deftype complex-matrix-element-type () 
  "The type of the elements stored in a COMPLEX-MATRIX"
  'double-float)

(deftype complex-matrix-store-type (size) 
  "The type of the storage structure for a COMPLEX-MATRIX"
  `(simple-array double-float ,size))
)

(declaim (ftype (function (standard-matrix) fixnum)
		n
		m
		nxm
		store-size)
	 (ftype (function (real-matrix) fixnum)
		n
		m
		nxm
		store-size)
	 (ftype (function (complex-matrix) fixnum)
		n
		m
		nxm
		store-size)
	 (ftype (function (real-matrix) (simple-array double-float (*)))
		 store)
	 (ftype (function (complex-matrix) (simple-array double-float (*)))
		 store))


#|
(defgeneric n (matrix)
  (:documentation 
"
  Syntax
  ======
  (N matrix)

  Purpose
  =======
  Returns the number of rows of MATRIX.
"))

(defgeneric m (matrix)
  (:documentation
"
  Syntax
  ======
  (M matrix)

  Purpose
  =======
  Returns the number of columns of MATRIX.
"))

(defgeneric nxm (matrix)
  (:documentation
"
  Syntax
  ======
  (NxM matrix)

  Purpose
  =======
  Returns the number of elements of MATRIX;
  which is number of rows * number of columns. 
"))
|#

(defgeneric store-size (matrix)
  (:documentation
"
  Syntax
  ======
  (STORE-SIZE matrix)

  Purpose
  ======= 
  Total number of elements needed to store the matrix.  (Usually
  the same as (NxM matrix), but not necessarily so!
"))

(defgeneric store (matrix)
  (:documentation
"
  Syntax
  ======
  (STORE matrix)

  Purpose
  =======
The actual storage for the matrix.  It is typically a one dimensional
array but not necessarily so.  The float and complex matrices do use
1-D arrays.  The complex matrix actually stores the real and imaginary
parts in successive elements of the matrix because Fortran stores them
that way.
"))

#+(and (or cmu sbcl) gerds-pcl)
(declaim (ext:slots (slot-boundp real-matrix complex-matrix)
		    (inline standard-matrix real-matrix complex-matrix)))

(defclass standard-matrix ()
  ((number-of-rows
    :initarg :nrows
    :initform 0
    :accessor nrows
    :type fixnum
    :documentation "Number of rows in the matrix")
   (number-of-cols
    :initarg :ncols
    :initform 0
    :accessor ncols
    :type fixnum
    :documentation "Number of columns in the matrix")
   (number-of-elements
    :initarg :nels
    :initform 0
    :accessor number-of-elements
    :type fixnum
    :documentation "Total number of elements in the matrix (nrows * ncols)")
   (store-size
    :initarg :store-size
    :initform 0
    :accessor store-size
    :type fixnum
    :documentation "Total number of elements needed to store the matrix.  (Usually
the same as nels, but not necessarily so!")
   (store
    :initarg :store
    :accessor store
    :documentation "The actual storage for the matrix.  It is typically a one dimensional
array but not necessarily so.  The float and complex matrices do use
1-D arrays.  The complex matrix actually stores the real and imaginary
parts in successive elements of the matrix because Fortran stores them
that way."))
  (:documentation "Basic matrix class."))


#+(and nil :allegro)
(defclass standard-matrix ()
  ((number-of-rows
    :initarg :nrows
    :initform 0
    :accessor nrows
    :documentation "Number of rows in the matrix")
   (number-of-cols
    :initarg :ncols
    :initform 0
    :accessor ncols
    :documentation "Number of columns in the matrix")
   (number-of-elements
    :initarg :nels
    :initform 0
    :accessor number-of-elements
    :documentation "Total number of elements in the matrix (nrows * ncols)")
   (store-size
    :initarg :store-size
    :initform 0
    :accessor store-size
    :documentation "Total number of elements needed to store the matrix.  (Usually
the same as nels, but not necessarily so!")
   (store
    :initarg :store
    :accessor store
    :documentation "The actual storage for the matrix.  It is typically a one dimensional
array but not necessarily so.  The float and complex matrices do use
1-D arrays.  The complex matrix actually stores the real and imaginary
parts in successive elements of the matrix because Fortran stores them
that way."))
  (:documentation "Basic matrix class."))

(defclass real-matrix (standard-matrix)
  ((store
    :type (simple-array real-matrix-element-type (*))))
  (:documentation "A class of matrices with real elements."))

(defclass complex-matrix (standard-matrix)
  ((store
    :type (simple-array complex-matrix-element-type (*))))
  (:documentation "A class of matrices with complex elements."))

(defmethod initialize-instance :after ((matrix standard-matrix) &rest initargs)
  (declare (ignore initargs))
  (let* ((n (nrows matrix))
	 (m (ncols matrix))
	 (nxm (* n m)))
    (declare (type fixnum n m nxm))
    (setf (number-of-elements matrix) nxm)
    (setf (store-size matrix) nxm)))

(defmethod make-load-form ((matrix standard-matrix) &optional env)
  "MAKE-LOAD-FORM allows us to determine a load time value for
   matrices, for example #.(make-matrix ...)"
  (make-load-form-saving-slots matrix :environment env))

(defgeneric row-vector-p (matrix)
  (:documentation "
  Syntax
  ======
  (ROW-VECTOR-P x)

  Purpose
  =======
  Return T if X is a row vector (number of columns is 1)"))

(defgeneric col-vector-p (matrix)
  (:documentation "
  Syntax
  ======
  (COL-VECTOR-P x)
 
  Purpose
  =======
  Return T if X is a column vector (number of rows is 1)"))

(defgeneric row-or-col-vector-p (matrix)
  (:documentation "
  Syntax
  ======
  (ROW-OR-COL-VECTOR-P x)

  Purpose
  =======
  Return T if X is either a row or a column vector"))

(defgeneric square-matrix-p (matrix)
  (:documentation "
  Syntax
  ======
  (SQUARE-MATRIX-P x)

  Purpose
  =======
  Return T if X is square matrix"))

(defgeneric size (matrix)
  (:documentation "
  Syntax
  ======
  (SIZE x)

  Purpose
  =======
  Return the number of rows and columns of the matrix X as a list"))

(declaim (inline row-vector-p))
(defmethod row-vector-p ((matrix standard-matrix))
  (= (nrows matrix) 1))

(declaim (inline col-vector-p))
(defmethod col-vector-p ((matrix standard-matrix))
  (= (ncols matrix) 1))

(declaim (inline row-or-col-vector-p))
(defmethod row-or-col-vector-p ((matrix standard-matrix))
  (or (row-vector-p matrix) (col-vector-p matrix)))

(declaim (inline square-matrix-p))
(defmethod square-matrix-p ((matrix standard-matrix))
  (= (nrows matrix) (ncols matrix)))

(defmethod size ((matrix standard-matrix))
  (list (nrows matrix) (ncols matrix)))

;; For compatibility with Fortran, matrices are stored in column major
;; order instead of row major order.  Also, we store the matrix as a
;; one-dimensional array instead of a two-dimensional array.  This
;; makes it easy to interface to LAPACK routines.
;;
;; furthermore, this next macro should really be left as a macro
;; to avoid integer to pointer coercions, since FORTRAN-MATRIX-INDEXING
;; will be called too many times.

#+nil
(defmacro fortran-matrix-indexing (i j l)
  `(let ((i ,i)
	 (j ,j)
	 (l ,l))
     (declare (optimize (speed 3) (safety 0))
	      (type fixnum i j l))
     (let* ((q (* j l))
	    (p (+ i q)))
       (declare (type fixnum q p))
       p)))

(declaim (inline fortran-matrix-indexing))
(defun fortran-matrix-indexing (row col nrows)
  (declare (type (and fixnum (integer 0)) row col nrows))
  (the fixnum (+ row (the fixnum (* col nrows)))))

;; For matrices with complex-valued elements, we store the array as a
;; double-length double-precision floating-point vector, as Fortran
;; does too.  The first element is the real part; the second, the
;; imaginary part.

#+nil
(defmacro fortran-complex-matrix-indexing (i j l)
  `(let ((i ,i)
	 (j ,j)
	 (l ,l))
     (declare (optimize (speed 3) (safety 0))
	      (type fixnum i j l))
     (let* ((q (* j l))
	    (p (+ i q))
	    (r (* 2 p)))
       (declare (type fixnum q p r))
       r)))

(declaim (inline fortran-complex-matrix-indexing))
(defun fortran-complex-matrix-indexing (row col nrows)
  (declare (type (and fixnum (integer 0)) row col nrows))
  (the fixnum (* 2 (the fixnum (+ row (the fixnum (* col nrows)))))))



;;; coerce is broken in CMUCL.  Here is a function 
;;; that implements coerce correctly for what we want.

(declaim (inline complex-coerce)
	 (ftype (function (number) (complex complex-matrix-element-type)) 
		complex-coerce))

(defun complex-coerce (val)
  "
 Syntax
 ======
 (COMPLEX-COERCE number)

 Purpose
 =======
 Coerce NUMBER to a complex number.
"
  (declare (type number val))
  (typecase val
    ((complex complex-matrix-element-type) val)
    (complex (complex (coerce (realpart val) 'complex-matrix-element-type)
		      (coerce (imagpart val) 'complex-matrix-element-type)))
    (t (complex (coerce val 'complex-matrix-element-type) 0.0d0))))

(defgeneric fill-matrix (matrix fill-element)
  (:documentation 
   "
   Syntax
   ======
   (FILL-MATRIX matrix fill-element)
  
   Purpose
   =======
   Fill MATRIX with FILL-ELEMENT.
"))

(defmethod fill-matrix ((matrix real-matrix) (fill real))
  (copy! fill matrix))

(defmethod fill-matrix ((matrix real-matrix) (fill complex))
  (error "cannot fill a real matrix with a complex number,
don't know how to coerce COMPLEX to REAL"))

(defmethod fill-matrix ((matrix complex-matrix) (fill number))
  (copy! fill matrix))

(defmethod fill-matrix ((matrix t) (fill t))
  (error "arguments MATRIX and FILL to FILL-MATRIX must be a
matrix and a number"))

;; Allocate an array suitable for the store part of a real matrix.

(declaim (inline allocate-integer4-store))
(defun allocate-integer4-store (size &optional (initial-element 0))
  "(ALLOCATE-INTEGER-STORE SIZE [INITIAL-ELEMENT]).  Allocates
integer storage.  Default INITIAL-ELEMENT = 0."
  (make-array size
	      :element-type 'integer4-matrix-element-type
	      :initial-element initial-element))

(declaim (inline allocate-real-store))
(defun allocate-real-store (size &optional (initial-element 0))
  (make-array size :element-type 'real-matrix-element-type
	      :initial-element (coerce initial-element 'real-matrix-element-type)))

(declaim (inline allocate-complex-store))
(defun allocate-complex-store (size)
  (make-array (* 2 size) :element-type 'complex-matrix-element-type
	      :initial-element (coerce 0 'complex-matrix-element-type)))

(defun make-real-matrix-dim (n m &optional (fill 0.0d0))
  "
  Syntax
  ======
  (MAKE-REAL-MATRIX-DIM n m [fill-element])

  Purpose
  =======
  Creates an NxM REAL-MATRIX with initial contents FILL-ELEMENT,
  the default 0.0d0

  See MAKE-REAL-MATRIX.
"
  (declare (type fixnum n m))

  (let ((casted-fill
	 (typecase fill
	   (real-matrix-element-type fill)
	   (real (coerce fill 'real-matrix-element-type))
	   (t (error "argument FILL-ELEMENT to MAKE-REAL-MATRIX-DIM must be a REAL")))))

    (declare (type real-matrix-element-type casted-fill))
    (make-instance 'real-matrix :nrows n :ncols m
		   :store (allocate-real-store (* n m) casted-fill))))


;;; Make a matrix from a 2-D Lisp array
(defun make-real-matrix-array (array)
  " 
  Syntax
  ======
  (MAKE-REAL-MATRIX-ARRAY array)

  Purpose
  =======
  Creates a REAL-MATRIX with the same contents as ARRAY.
"
  (let* ((n (array-dimension array 0))
	 (m (array-dimension array 1))
	 (size (* n m))
	 (store (allocate-real-store size)))
    (declare (type fixnum n m size)
	     (type (real-matrix-store-type (*)) store))
    (dotimes (i n)
      (declare (type fixnum i))
      (dotimes (j m)
	(declare (type fixnum j))
	(setf (aref store (fortran-matrix-indexing i j n))
	      (coerce (aref array i j) 'real-matrix-element-type))))
    (make-instance 'real-matrix :nrows n :ncols m :store store)))

(defun make-real-matrix-seq-of-seq (seq)
  (let* ((n (length seq))
	 (m (length (elt seq 0)))
	 (size (* n m))
	 (store (allocate-real-store size)))
    (declare (type fixnum n m size)
	     (type (real-matrix-store-type (*)) store))
    (dotimes (i n)
      (declare (type fixnum i))	     
      (let ((this-row (elt seq i)))
	(unless (= (length this-row) m)
	  (error "Number of columns is not the same for all rows!"))
	(dotimes (j m)
          (declare (type fixnum j))
	  (setf (aref store (fortran-matrix-indexing i j n)) 
		(coerce (elt this-row j) 'real-matrix-element-type)))))
    (make-instance 'real-matrix :nrows n :ncols m :store store)))

(defun make-real-matrix-seq (seq)
  (let* ((n (length seq))
	 (store (allocate-real-store n)))
    (declare (type fixnum n))
    (dotimes (k n)
      (declare (type fixnum k))
      (setf (aref store k) (coerce (elt seq k) 'real-matrix-element-type)))
    (make-instance 'real-matrix :nrows n :ncols 1 :store store)))
  
(defun make-real-matrix-sequence (seq)
  (cond ((or (listp seq) (vectorp seq))
	 (let ((peek (elt seq 0)))
	   (cond ((or (listp peek) (vectorp peek))
		  ;; We have a seq of seqs
		  (make-real-matrix-seq-of-seq seq))
		 (t
		  ;; Assume a simple sequence
		  (make-real-matrix-seq seq)))))
	((arrayp seq)
	 (make-real-matrix-array seq))))

(defun make-real-matrix (&rest args)
  "
 Syntax
 ======
 (MAKE-REAL-MATRIX {arg}*)

 Purpose
 =======
 Create a REAL-MATRIX.

 Examples
 ========

 (make-real-matrix n)
        square NxN matrix
 (make-real-matrix n m)
        NxM matrix
 (make-real-matrix '((1 2 3) (4 5 6)))
        2x3 matrix:

              1 2 3
              4 5 6

 (make-real-matrix #((1 2 3) (4 5 6)))
        2x3 matrix:

              1 2 3
              4 5 6

 (make-real-matrix #((1 2 3) #(4 5 6)))
        2x3 matrix:

              1 2 3
              4 5 6

 (make-real-matrix #2a((1 2 3) (4 5 6)))
        2x3 matrix:

              1 2 3
              4 5 6
 (make-real-matrix #(1 2 3 4))
        4x1 matrix (column vector)

          1
          2
          3
          4

 (make-real-matrix #((1 2 3 4))
        1x4 matrix (row vector)

          1 2 3 4
"

  (let ((nargs (length args)))
    (case nargs
      (1
       (let ((arg (first args)))
	 (typecase arg
	   (integer
	    (assert (plusp arg) nil
		    "matrix dimension must be positive, not ~A" arg)
	    (make-real-matrix-dim arg arg))
	   (sequence
	    (make-real-matrix-sequence arg))
	   ((array * (* *))
	    (make-real-matrix-array arg))
	   (t (error "don't know how to make matrix from ~a" arg)))))
      (2
       (destructuring-bind (n m)
	   args
	 (assert (and (typep n '(integer 1))
		      (typep n '(integer 1)))
		 nil
		 "cannot make a ~A x ~A matrix" n m)
	 (make-real-matrix-dim n m)))
      (t
       (error "require 1 or 2 arguments to make a matrix")))))



(defun make-complex-matrix-dim (n m &optional (fill #c(0.0d0 0.0d0)))
  "
  Syntax
  ======
  (MAKE-COMPLEX-MATRIX-DIM n m [fill-element])

  Purpose
  =======
  Creates an NxM COMPLEX-MATRIX with initial contents FILL-ELEMENT,
  the default #c(0.0d0 0.0d0)

  See MAKE-COMPLEX-MATRIX.
"
  (declare (type fixnum n m))
  (let* ((size (* n m))
	 (store (allocate-complex-store size))
	 (matrix (make-instance 'complex-matrix :nrows n :ncols m :store store)))
    
    (fill-matrix matrix fill)
    matrix))

(defun make-complex-matrix-array (array)
  " 
  Syntax
  ======
  (MAKE-COMPLEX-MATRIX-ARRAY array)

  Purpose
  =======
  Creates a COMPLEX-MATRIX with the same contents as ARRAY.
"
  (let* ((n (array-dimension array 0))
	 (m (array-dimension array 1))
	 (size (* n m))
	 (store (allocate-complex-store size)))
    (declare (type fixnum n m size)
	     (type (complex-matrix-store-type (*)) store))
    (dotimes (i n)
      (declare (type fixnum i))
      (dotimes (j m)
	(declare (type fixnum j))
	(let* ((val (complex-coerce (aref array i j)))
	       (realpart (realpart val))
	       (imagpart (imagpart val))
	       (index (fortran-complex-matrix-indexing i j n)))
	    (declare (type complex-matrix-element-type realpart imagpart)
		     (type (complex complex-matrix-element-type) val)
		     (type fixnum index))
	  (setf (aref store index) realpart)
	  (setf (aref store (1+ index)) imagpart))))
    
    (make-instance 'complex-matrix :nrows n :ncols m :store store)))


(defun make-complex-matrix-seq-of-seq (seq)
  (let* ((n (length seq))
	 (m (length (elt seq 0)))
	 (size (* n m))
	 (store (allocate-complex-store size)))
    (declare (type fixnum n m size)
	     (type (complex-matrix-store-type (*)) store))
    
    (dotimes (i n)
      (declare (type fixnum i))
      (let ((this-row (elt seq i)))
	(unless (= (length this-row) m)
	  (error "Number of columns is not the same for all rows!"))
	(dotimes (j m)
	  (declare (type fixnum j))
	  (let* ((val (complex-coerce (elt this-row j)))
		 (realpart (realpart val))
		 (imagpart (imagpart val))
		 (index (fortran-complex-matrix-indexing i j n)))
	    (declare (type complex-matrix-element-type realpart imagpart)
		     (type (complex complex-matrix-element-type) val)
		     (type fixnum index))
	    (setf (aref store index) realpart)
	    (setf (aref store (1+ index)) imagpart)))))
    
    (make-instance 'complex-matrix :nrows n :ncols m :store store)))


(defun make-complex-matrix-seq (seq)
  (let* ((n (length seq))
	 (store (allocate-complex-store n)))
    (declare (type fixnum n)
	     (type (complex-matrix-store-type (*)) store))
    
    (dotimes (k n)
      (declare (type fixnum k))
      (let* ((val (complex-coerce (elt seq k)))
	     (realpart (realpart val))
	     (imagpart (imagpart val))
	     (index (* 2 k)))
	(declare (type complex-matrix-element-type realpart imagpart)
		 (type (complex complex-matrix-element-type) val)
		 (type fixnum index))
	(setf (aref store index) realpart)
	(setf (aref store (1+ index)) imagpart)))
    
    (make-instance 'complex-matrix :nrows n :ncols 1 :store store)))


(defun make-complex-matrix-sequence (seq)
  (cond ((or (listp seq) (vectorp seq))
	 (let ((peek (elt seq 0)))
	   (cond ((or (listp peek) (vectorp peek))
		  ;; We have a seq of seqs
		  (make-complex-matrix-seq-of-seq seq))
		 (t
		  ;; Assume a simple sequence
		  (make-complex-matrix-seq seq)))))
	((arrayp seq)
	 (make-complex-matrix-array seq))))


(defun make-complex-matrix (&rest args)
  "
 Syntax
 ======
 (MAKE-FLOAT-MATRIX {arg}*)

 Purpose
 =======
 Create a FLOAT-MATRIX.

 Examples
 ========

 (make-complex-matrix n)
        square NxN matrix
 (make-complex-matrix n m)
        NxM matrix
 (make-complex-matrix '((1 2 3) (4 5 6)))
        2x3 matrix:

              1 2 3
              4 5 6

 (make-complex-matrix #((1 2 3) (4 5 6)))
        2x3 matrix:

              1 2 3
              4 5 6

 (make-complex-matrix #((1 2 3) #(4 5 6)))
        2x3 matrix:

              1 2 3
              4 5 6

 (make-complex-matrix #2a((1 2 3) (4 5 6)))
        2x3 matrix:

              1 2 3
              4 5 6

"
  (let ((nargs (length args)))
    (case nargs
      (1
       (let ((arg (first args)))
	 (typecase arg
	   (integer
	    (assert (plusp arg) nil
		    "matrix dimension must be positive, not ~A" arg)
	    (make-complex-matrix-dim arg arg))
	   (sequence
	    (make-complex-matrix-sequence arg))
	   ((array * (* *))
	    (make-complex-matrix-array arg))
	   (t (error "don't know how to make matrix from ~a" arg)))))
      (2
       (destructuring-bind (n m)
	   args
	 (assert (and (typep n '(integer 1))
		      (typep n '(integer 1)))
		 nil
		 "cannot make a ~A x ~A matrix" n m)
	 (make-complex-matrix-dim n m)))
      (t
       (error "require 1 or 2 arguments to make a matrix")))))
