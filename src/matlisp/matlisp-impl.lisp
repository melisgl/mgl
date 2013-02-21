;;;; Lisp implementation of the necessary Matlisp machinery. Based on
;;;; and ripped from Matlisp code originally written by Raymond Toy.

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

(in-package :matlisp)

(eval-when (:load-toplevel :execute)
  (write-line
   (if (find-package 'blas)
       "Matlisp is already loaded: redefining with slow, Lisp implementation."
       "Loading Lisp implementation of a fraction Matlisp. ~
        Load Matlisp at any time to speed things up.")))

(defgeneric copy (matrix))

(defmethod copy ((matrix real-matrix))
  (let* ((n (nrows matrix))
	 (m (ncols matrix))
	 (result (make-real-matrix-dim n m)))
    (copy! matrix result)))

(defgeneric copy! (matrix new-matrix))

(defmethod copy! ((x real-matrix) (y real-matrix))
  (let* ((nxm-x (number-of-elements x))
	 (nxm-y (number-of-elements y))
	 (nxm (min nxm-x nxm-y))
         (x* (store x))
         (y* (store y)))
    (declare (type fixnum nxm-x nxm-y nxm)
             (type (simple-array real-matrix-element-type (*)) x* y*)
             (optimize (speed 3)))
    (dotimes (i nxm y)
      (setf (aref y* i) (aref x* i)))))

(defmethod copy! ((x double-float) (y real-matrix))
  (let ((nxm-y (number-of-elements y))
        (y* (store y)))
    (declare (type (simple-array real-matrix-element-type (*)) y*)
             (optimize (speed 3)))
    (fill y* x :end nxm-y)
    y))

(defgeneric gemm! (alpha a b beta c &optional job))

(defmethod gemm! :before ((alpha number)
			  (a standard-matrix)
			  (b standard-matrix)
			  (beta number)
			  (c standard-matrix)
			  &optional (job :NN))
  (let ((n-a (nrows a))
	(m-a (ncols a))
	(n-b (nrows b))
	(m-b (ncols b))
	(n-c (nrows c))
	(m-c (ncols c)))
    (declare (type fixnum n-a m-a n-b m-b n-c m-c))

    (case job
      (:nn t)
      (:tn (rotatef n-a m-a))
      (:nt (rotatef n-b m-b))
      (:tt (rotatef n-a m-a) (rotatef n-b m-b))
      (t (error "argument JOB to GEMM! is not recognized")))

    (if (not (and (= m-a n-b)
		  (= n-a n-c)
		  (= m-b m-c)))
	(error "dimensions of A,B,C given to GEMM! do not match"))))

(defmethod gemm! ((alpha double-float)
		  (a real-matrix)
		  (b real-matrix)
		  (beta double-float)
		  (c real-matrix)
		  &optional (job :nn))
  (let* ((a-nrows (nrows a))
         (b-nrows (nrows b))
         (c-nrows (nrows c))
         (c-ncols (ncols c))
         (transpose-a (member job '(:tn tn :tt tt)))
         (transpose-b (member job '(:nt nt :tt tt)))
         (k (if transpose-a
                (nrows a)
                (ncols a)))
         (a* (store a))
         (b* (store b))
         (c* (store c)))
    (declare (type fixnum a-nrows b-nrows c-ncols c-nrows k)
             (type (simple-array real-matrix-element-type (*)) a* b* c*))
    (dotimes (c-col c-ncols)
      (loop for ci upfrom (* c-nrows c-col)
            for c-row below c-nrows
            do
            (let ((sum 0d0)
                  (ai (if transpose-a (* c-row a-nrows) c-row))
                  (bi (if transpose-b c-col (* c-col b-nrows)))
                  (ainc (if transpose-a 1 a-nrows))
                  (binc (if transpose-b b-nrows 1)))
              (declare (type double-float sum)
                       (type fixnum ai bi ainc binc)
                       #+nil (optimize (speed 3)))
              (loop repeat k
                    do (incf sum (* (aref a* ai) (aref b* bi)))
                    (incf ai ainc)
                    (incf bi binc))
              (setf (aref c* ci)
                    (+ (* beta (aref c* ci))
                       (* alpha sum)))))))
  c)

(defgeneric m+! (a b))

(defmethod m+! :before ((a standard-matrix) (b standard-matrix))
  (let ((n-a (nrows a))
	(m-a (ncols a))
	(n-b (nrows b))
	(m-b (ncols b)))
    (declare (type fixnum n-a m-a n-b m-b))
    (unless (and (= n-a n-b)
		 (= m-a m-b))
      (error "Cannot add a ~d x ~d matrix and a ~d x ~d matrix"
	     n-a m-a
	     n-b m-b))))

(defmethod m+! ((a standard-matrix) (b standard-matrix))
  (let* ((nxm-a (number-of-elements a))
	 (nxm-b (number-of-elements b))
	 (nxm (min nxm-a nxm-b))
         (a* (store a))
         (b* (store b)))
    (declare (type fixnum nxm-a nxm-b nxm)
             (type (simple-array real-matrix-element-type (*)) a* b*)
             (optimize (speed 3)))
    (dotimes (i nxm b)
      (incf (aref b* i) (aref a* i)))))

(defmethod scal! ((a double-float) (x real-matrix))
  (let* ((nx (number-of-elements x))
         (x* (store x)))
    (declare (type fixnum nx)
             (type (simple-array real-matrix-element-type (*)) x*)
             (optimize (speed 3)))
    (dotimes (i nx x)
      (setf (aref x* i) (* a (aref x* i))))))

(defun ones (n &optional (m n))
  (unless (and (typep m '(integer 1))
	       (typep n '(integer 1)))
    (error
     "the number of rows (~d) and columns (~d) must be positive integers" n m))
  (make-real-matrix-dim n m 1.0d0))

(defgeneric sum (matrix))

(defmethod sum ((a real-matrix))
  (if (row-or-col-vector-p a)
      (let ((result 0.0d0)
	    (store (store a))
	    (nxm (number-of-elements a)))
	(declare (type fixnum nxm)
		 (type real-matrix-element-type result)
		 (type (real-matrix-store-type (*)) store))
	(dotimes (i nxm)
	 (declare (type fixnum i))
           (incf result (aref store i)))
	result)

    (let* ((n (nrows a))
	   (m (ncols a))
	   (result (make-real-matrix-dim n 1))
	   (store-a (store a))
	   (store-result (store result)))
      (declare (type fixnum n m)
	       (type (real-matrix-store-type (*)) store-a store-result))
      (dotimes (i n)
	(declare (type fixnum i))
	(setf (aref store-result i)
	      (let ((val 0.0d0))
		(declare (type real-matrix-element-type val))
		(dotimes (j m val)
		  (declare (type fixnum j))
		  (incf val (aref store-a (fortran-matrix-indexing  i j n)))))))
      result)))

(defmethod m.*! ((a real-matrix) (b real-matrix))
  (let* ((nxm (number-of-elements b))
	 (a-store (store a))
	 (b-store (store b)))
    (declare (type fixnum nxm))

    (dotimes (k nxm b)
      (declare (type fixnum k))
      (setf (aref b-store k) (* (aref a-store k) (aref b-store k))))))

(defmethod m.*! ((a standard-matrix) (b number))
  (scal! b a))

(defmethod m.*! ((a number) (b standard-matrix))
  (scal! a b))
