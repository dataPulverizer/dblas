/**
*
*  @title: Level 2 BLAS algorithms for D
*  @author: Chibisi Chima-Okereke
*  @date: 2017-02-24
*  @description Level 2 BLAS algorithms for D translated from GSL library.
*  
*/

module dblas.l2;
import std.stdio : writeln;

import dblas.l1;
//import std.math: abs, fabs, sqrt;
import std.complex: Complex, complex, conj;
import std.algorithm.comparison : min, max;
import std.traits : isFloatingPoint;

/* Enums ... */
/* Order Enum */
enum CBLAS_ORDER {
	CblasRowMajor = 101,
	CblasColMajor = 102
}

alias CBLAS_ORDER.CblasRowMajor CblasRowMajor;
alias CBLAS_ORDER.CblasColMajor CblasColMajor;

alias CBLAS_ORDER CBLAS_LAYOUT;

/* Transpose Enum */
enum CBLAS_TRANSPOSE {
	CblasNoTrans = 111,
	CblasTrans = 112,
    CblasConjTrans = 113
}

alias CBLAS_TRANSPOSE.CblasNoTrans CblasNoTrans;
alias CBLAS_TRANSPOSE.CblasTrans CblasTrans;
alias CBLAS_TRANSPOSE.CblasConjTrans CblasConjTrans;

/* Upper/Lower Enum */
enum CBLAS_UPLO {
	CblasUpper = 121,
	CblasLower = 122
}

alias CBLAS_UPLO.CblasUpper CblasUpper;
alias CBLAS_UPLO.CblasLower CblasLower;

/* Diag Enum */
enum CBLAS_DIAG {
	CblasNonUnit = 131,
	CblasUnit = 132
}

alias CBLAS_DIAG.CblasNonUnit CblasNonUnit;
alias CBLAS_DIAG.CblasUnit CblasUnit;

/* Left/Right Enum */
enum CBLAS_SIDE {
	CblasLeft = 141,
	CblasRight = 142
}

alias CBLAS_SIDE.CblasLeft CblasLeft;
alias CBLAS_SIDE.CblasRight CblasRight;


/* Transformation of matrix to row and column major band matrix */
//T[] row_major(T)(T[] mat, int n, int ku, int m, int kl)
T[] row_major(T)(T[] mat, int m, int n, int kl, int ku)
{
	T[] a;
	int ldm = n, lda = ku + kl + 1, k;
	a.length = lda*m;
	static if(isComplex!T)
		a[] = complex(0, 0);
	else
		a[] = 0;
	for(int i = 0; i < m; ++i){
		k = kl - i;
		for(int j = max(0, i - kl); j < min(n, i + ku + 1); ++j){
			a[(k + j) + i*lda] = mat[j + i*ldm];
		}
	}
	return a;
}

//T[] col_major(T)(T[] mat, int n, int ku, int m, int kl)
T[] col_major(T)(T[] mat, int m, int n, int kl, int ku)
{
	T[] a;
	int ldm = m, lda = ku + kl + 1, k;
	a.length = lda*n;
	static if(isComplex!T)
		a[] = complex(0, 0);
	else
		a[] = 0;
	for(int j = 0; j < n; ++j){
		k = ku - j;
		for(int i = max(0, j - ku); i < min(m, j + kl + 1); ++i){
			a[(k + i) + j*lda] = mat[i + j*ldm];
		}
	}
	return a;
}


/*T OFFSET(T)(in T N, in T incX)
{
	return incX > 0 ?  0 : ((N - 1) * -incX);
}*/

T TRCOUNT(T)(T N, T i){
	return ((i + 1)*(2*N - i))/2;
}

T TPUP(T)(T N, T i, T j){
	return TRCOUNT(N, i - 1) + j - i;
}

T TPLO(T)(T N, T i, T j){
	return (i*(i + 1))/2 + j;
}

/**
*  @title gbmv blas function: Computes a matrix-vector product using a general band
*                             matrix
*  @description The gbmv routines perform a matrix-vector operation defined as
*
*               y := alpha*A*x + beta*y
*               or
*               y := alpha*A'*x + beta*y
*               or
*               y := alpha*conj(A')*x + beta*y
*
*               where alpha and beta are scalars, x and y are vectors,
*               A is an m-by-n band matrix, with kl sub-diagonals and ku
*               super-diagonals.
*
*         Input Parameters:
*
*  @param layout: Specifies whether two-dimensional array storage is row-major
*                 (CblasRowMajor) or column-major (CblasColMajor).
*
*  @param trans:  Specifies the operation:
*                 If trans = CblasNoTrans, then y := alpha*A*x + beta*y
*                 If trans = CblasTrans , then y := alpha*A'*x + beta*y
*                 If trans = CblasConjTrans , then y := alpha *conjg(A')*x + beta*y
*
*         m:      Specifies the number of rows of matrix A, m >= 0
*         n:      Specifies the number of columns of the matrix A.
*                 The value of n >= 0.
*         kl:     Specifies the number of sub-diagonals of the matrix A.
*                 The value of kl must satisfy kl >= 0.
*         ku:     Specifies the number of super-diagonals of the matrix A.
*                 The value of ku must satisfy ku >= 0.
*         alpha:  Specifies the scala alpha
*         a:      Array, size lda*n.
*			      layout = CblasColMajor : Before entry, the leading (kl + ku + 1) by n
*			      part of the array a must contain the matrix of coefficients. This matrix must
*			      be supplied column-by-column, with the leading diagonal of the matrix in
*			      row (ku) of the array, the first super-diagonal starting at position 1 in row
*			      (ku - 1) , the first sub-diagonal starting at position 0 in row (ku + 1) ,
*			      and so on. Elements in the array a that ` not correspond to elements in
*			      the band matrix (such as the top left ku by ku triangle) are not referenced.
*			      The following program segment transfers a band matrix from conventional
*			      full matrix storage (matrix, with leading dimension ldm) to band storage (a,
*			      with leading dimension lda):
*			      for (j = 0; j < n; j++) {
*			      k = ku - j;
*			      for (i = max(0, j-ku); i < min(m, j+kl+1); i++) {
*			      a[(k+i) + j*lda] = matrix[i + j*ldm];
*			      }
*			      }
*			      layout = CblasRowMajor : Before entry, the leading (kl + ku + 1) by m
*			      part of the array a must contain the matrix of coefficients. This matrix must
*			      be supplied row-by-row, with the leading diagonal of the matrix in column
*			      (kl) of the array, the first super-diagonal starting at position 0 in column
*			      (kl + 1) , the first sub-diagonal starting at position 1 in row (kl - 1) ,
*			      and so on. Elements in the array a that do not correspond to elements in
*			      the band matrix (such as the top left kl by kl triangle) are not referenced.
*			      The following program segment transfers a band matrix from row-major full
*			      matrix storage (matrix, with leading dimension ldm) to band storage (a,
*			      with leading dimension lda):
*			      for (i = 0; i < m; i++) {
*			      	k = kl - i;
*			      	for (j = max(0, i-kl); j < min(n, i+ku+1); j++) {
*			      		a[(k+j) + i*lda] = matrix[j + i*ldm];
*			      	}
*			      }
*
*  @param lda:    Specifies the leading dimension of a as declared in the calling
*			      (sub)program. The value of lda must be at least (kl + ku + 1) .
*  @param x:      Array, size at least (1 + (n - 1)*abs(incX)) when
*			      trans = CblasNoTrans , and at least (1 + (m - 1)*abs(incX))
*			      otherwise. Before entry, the array x must contain the vector x.
*  @param incX:   Specifies the increment for the elements of x. incX must not be zero.
*			      beta Specifies the scalar beta. When beta is equal to zero, then y need not be
*			      set on input.
*  @param y:      Array, size at least (1 +(m - 1)*abs(incY)) when
*			      trans = CblasNoTrans and at least (1 +(n - 1)*abs(incY)) otherwise.
*			      Before entry, the incremented array y must contain the vector y.
*  @param incY:   Specifies the increment for the elements of y.
*			      The value of incY must not be zero.
*
*                 Output Parameters:
*
*  @param y:      Updated vector y.
*  @example
*
*/
void gbmv(N, X)(in CBLAS_LAYOUT layout, in CBLAS_TRANSPOSE transA, in N m, in N n, in N kl, in N ku, 
	      in X alpha, in X* a, in N lda, in X* x, in N incX, in X beta, X* y, in N incY)
{
	N i, j, lenX, lenY, L, U, iy, jx;

	CBLAS_TRANSPOSE trans = (transA != CblasConjTrans) ? transA : CblasTrans;
	
	if(m == 0 || n == 0)
		return;
	if(alpha == 0 && beta == 1)
		return;

	if(trans == CblasNoTrans)
	{
		lenX = n;
		lenY = m;
		L = kl;
		U = ku;
	}else{
		lenX = m;
		lenY = n;
		L = ku;
		U = kl;
	}


	if(beta == 0)
	{
		iy = OFFSET(lenY, incY);
		for(i = 0; i < lenY; ++i)
		{
			y[iy] = 0;
			iy += incY;
		}
	}else if(beta != 1){
		iy = OFFSET(lenY, incY);
		for(i = 0; i < lenY; ++i)
		{
			y[iy] *= beta;
			iy += incY;
		}
	}

	if(alpha == 0)
		return;

	if((layout == CblasRowMajor && trans == CblasNoTrans)
		|| (layout == CblasColMajor && trans == CblasTrans))
	{
		iy = OFFSET(lenY, incY);
		for(i = 0; i < lenY; ++i)
		{
			X temp = 0.0;
			const N j_min = (i > L ? i - L : 0);
			const N j_max = min(lenX, i + U + 1);
			jx = OFFSET(lenX, incX) + j_min*incX;
			for(j = j_min; j < j_max; ++j)
			{
				temp += x[jx] * a[(L - i + j) + i * lda];
                jx += incX;
			}
			y[iy] += alpha * temp;
            iy += incY;
		}
	} else if((layout == CblasRowMajor && trans == CblasTrans)
		|| (layout == CblasColMajor && trans == CblasNoTrans)){
		jx = OFFSET(lenX, incX);
		for (j = 0; j < lenX; ++j) {
	        const X temp = alpha * x[jx];
	        if (temp != 0.0) {
	            const N i_min = (j > U ? j - U : 0);
	            const N i_max = min(lenY, j + L + 1);
	            iy = OFFSET(lenY, incY) + i_min * incY;
	            for (i = i_min; i < i_max; i++) {
	                y[iy] += temp * a[lda * j + (U + i - j)];
	                iy += incY;
	            }
	        }
	        jx += incX;
	    }
	} else {
		assert(0, "Unrecognised operation");
	}
}


void gbmv(N, X: Complex!V, V = typeof(X.re))(in CBLAS_LAYOUT layout, in CBLAS_TRANSPOSE transA, in N m,
	        in N n, in N kl, in N ku, in X alpha, in X* a, in N lda, in X* x, in N incX, in X beta, 
	        X* y, in N incY)
{
	N i, j, lenX, lenY, L, U, iy, jx;

	CBLAS_TRANSPOSE trans = transA;

	X zero = complex(0, 0);
	X one = complex(1, 0);
	
	if(m == 0 || n == 0)
		return;
	if(alpha == zero && beta == one)
		return;

	if(trans == CblasNoTrans)
	{
		lenX = n;
		lenY = m;
		L = kl;
		U = ku;
	}else{
		lenX = m;
		lenY = n;
		L = ku;
		U = kl;
	}


	if(beta == zero)
	{
		iy = OFFSET(lenY, incY);
		for(i = 0; i < lenY; ++i)
		{
			y[iy] = zero;
			iy += incY;
		}
	}else if(beta != one){
		iy = OFFSET(lenY, incY);
		for(i = 0; i < lenY; ++i)
		{
			y[iy] *= beta;
			iy += incY;
		}
	}

	if(alpha == zero)
		return;

	if((layout == CblasRowMajor && trans == CblasNoTrans)
		|| (layout == CblasColMajor && trans == CblasTrans))
	{
		iy = OFFSET(lenY, incY);
		for(i = 0; i < lenY; ++i)
		{
			X temp = zero;
			const N j_min = (i > L ? i - L : 0);
			const N j_max = min(lenX, i + U + 1);
			jx = OFFSET(lenX, incX) + j_min*incX;
			for(j = j_min; j < j_max; ++j)
			{
				temp += x[jx] * a[(L - i + j) + i * lda];
                jx += incX;
			}
			y[iy] += alpha * temp;
            iy += incY;
		}
	} else if((layout == CblasRowMajor && trans == CblasTrans)
		|| (layout == CblasColMajor && trans == CblasNoTrans)){
		jx = OFFSET(lenX, incX);
		for (j = 0; j < lenX; ++j) {
	        const X temp = alpha * x[jx];
	        if (temp != zero) {
	            const N i_min = (j > U ? j - U : 0);
	            const N i_max = min(lenY, j + L + 1);
	            iy = OFFSET(lenY, incY) + i_min * incY;
	            for (i = i_min; i < i_max; i++) {
	                y[iy] += temp * a[lda * j + (U + i - j)];
	                iy += incY;
	            }
	        }
	        jx += incX;
	    }
	} else if(layout == CblasColMajor && trans == CblasConjTrans)
	{
		iy = OFFSET(lenY, incY);
		for(i = 0; i < lenY; ++i)
		{
			X temp = zero;
			const N j_min = (i > L ? i - L : 0);
			const N j_max = min(lenX, i + U + 1);
			jx = OFFSET(lenX, incX) + j_min*incX;
			for(j = j_min; j < j_max; ++j)
			{
				temp += x[jx] * conj(a[(L - i + j) + i * lda]);
                jx += incX;
			}
			y[iy] += alpha * temp;
            iy += incY;
		}
	} else if(layout == CblasRowMajor && trans == CblasConjTrans){
		jx = OFFSET(lenX, incX);
		for (j = 0; j < lenX; ++j) {
	        const X temp = alpha * x[jx];
	        if (temp != zero) {
	            const N i_min = (j > U ? j - U : 0);
	            const N i_max = min(lenY, j + L + 1);
	            iy = OFFSET(lenY, incY) + i_min * incY;
	            for (i = i_min; i < i_max; i++) {
	                y[iy] += temp * conj(a[lda * j + (U + i - j)]);
	                iy += incY;
	            }
	        }
	        jx += incX;
	    }
	} else {
		assert(0, "Unrecognised operation");
	}
}


/**
*  @title gemv blas function: Computes a matrix-vector product 
*                             using a general matrix
*  @description The gemv routines perform a matrix-vector operation defined as
*
*               y := alpha*A*x + beta*y ,
*               or
*               y := alpha*A'*x + beta*y ,
*               or
*               y := alpha*conjg(A')*x + beta*y ,
*               where:
*               alpha and beta are scalars,
*               x and y are vectors,
*               A is an m-by-n matrix.
*
*  Input Parameters
*
*  @param layout: Specifies whether two-dimensional array storage is row-major
*  ( CblasRowMajor ) or column-major ( CblasColMajor ).
*
*  @param trans:  Specifies the operation:
*                 if trans = CblasNoTrans , then y := alpha*A*x + beta*y ;
*                 if trans = CblasTrans , then y := alpha*A'*x + beta*y ;
*                 if trans = CblasConjTrans , then y := alpha *conjg(A')*x + beta*y .
*
*  @param m:      Specifies the number of rows of the matrix A. The value of m 
*                 must be at least zero.
*
*  @param n:      Specifies the number of columns of the matrix A. The value of n 
*                 must be at least zero.
*
*  @param alpha:  Specifies the scalar alpha.
*  @param a:       Array, size lda *k.
*
*                 For Layout = CblasColMajor , k is n . Before entry, the leading m -by- n part
*                 of the array a must contain the matrix A.
*                 For Layout = CblasRowMajor , k is m . Before entry, the leading n -by- m part
*                 of the array a must contain the matrix A.
*
*  @param lda:    Specifies the leading dimension of a as declared in the calling (sub)program.
*
*                 For Layout = CblasColMajor , the value of lda must be at least max(1, m).
*                 For Layout = CblasRowMajor , the value of lda must be at least max(1, n).
*
*  @param x:      Array, size at least (1+(n-1)*abs(incx)) when trans = CblasNoTrans and at least 
*                 (1 + (m - 1)*abs(incx)) otherwise. Before entry, the incremented array x must 
*                 contain the vector x.
*
*  @param incx:   Specifies the increment for the elements of x.
*                 The value of incx must not be zero.
*
*  @param beta:   Specifies the scalar beta. When beta is set to zero, then y need not 
*                 be set on input.
*
*  @param y:      Array, size at least (1 +(m - 1)*abs(incy)) when
*                 trans = CblasNoTrans and at least (1 +(n - 1)*abs(incy)) otherwise.
*                 Before entry with non-zero beta, the incremented array y must contain the
*                 vector y.
*  @param incy:   Specifies the increment for the elements of y.
*                 The value of incy must not be zero.
*
*  Output Parameters
*
*  @param y:      Updated vector y.
*
*/
void gemv(N, X)(in CBLAS_LAYOUT layout, in CBLAS_TRANSPOSE transA, in N m, in N n, in X alpha, 
	            in X* a, in N lda, in X* x, in N incX, in X beta, X* y, in N incY)
{
  	N i, j;
    N lenX, lenY;
  
    CBLAS_TRANSPOSE trans = (transA != CblasConjTrans) ? transA : CblasTrans;
  
    if (m == 0 || n == 0)
      return;
  
    if (alpha == 0 && beta == 1)
      return;
  
    if (trans == CblasNoTrans) {
      lenX = n;
      lenY = m;
    } else {
      lenX = m;
      lenY = n;
    }
  
    if (beta == 0) {
      N iy = OFFSET(lenY, incY);
      for (i = 0; i < lenY; ++i) {
        y[iy] = 0.0;
        iy += incY;
      }
    } else if (beta != 1) {
      N iy = OFFSET(lenY, incY);
      for (i = 0; i < lenY; ++i) {
        y[iy] *= beta;
        iy += incY;
      }
    }
  
    if (alpha == 0)
      return;
  
    if ((layout == CblasRowMajor && trans == CblasNoTrans)
        || (layout == CblasColMajor && trans == CblasTrans)) {
      N iy = OFFSET(lenY, incY);
      for (i = 0; i < lenY; ++i) {
        X temp = 0.0;
        N ix = OFFSET(lenX, incX);
        for (j = 0; j < lenX; ++j) {
          temp += x[ix] * a[lda * i + j];
          ix += incX;
        }
        y[iy] += alpha * temp;
        iy += incY;
      }
    } else if ((layout == CblasRowMajor && trans == CblasTrans)
               || (layout == CblasColMajor && trans == CblasNoTrans)) {
      N ix = OFFSET(lenX, incX);
      for (j = 0; j < lenX; ++j) {
        const X temp = alpha * x[ix];
        if (temp != 0.0) {
          N iy = OFFSET(lenY, incY);
          for (i = 0; i < lenY; ++i) {
            y[iy] += temp * a[lda * j + i];
            iy += incY;
          }
        }
        ix += incX;
      }
    } else {
      assert(0, "unrecognized operation");
    }
}


void gemv(N, X: Complex!V, V = typeof(X.re))(in CBLAS_LAYOUT layout, in CBLAS_TRANSPOSE transA, 
	        in N m, in N n, in X alpha, in X* a, in N lda, in X* x, in N incX, in X beta, X* y, 
	        in N incY)
{
  	N i, j;
    N lenX, lenY;

    CBLAS_TRANSPOSE trans = transA;

	X zero = complex(0, 0), one = complex(1, 0);
  
    if (m == 0 || n == 0)
      return;
  
    if (alpha == zero && beta == one)
      return;
  
    if (trans == CblasNoTrans) {
      lenX = n;
      lenY = m;
    } else {
      lenX = m;
      lenY = n;
    }
  
    if (beta == zero) {
      N iy = OFFSET(lenY, incY);
      for (i = 0; i < lenY; ++i) {
        y[iy] = zero;
        iy += incY;
      }
    } else if (beta != one) {
      N iy = OFFSET(lenY, incY);
      for (i = 0; i < lenY; ++i) {
        y[iy] *= beta;
        iy += incY;
      }
    }
  
    if (alpha == zero)
      return;
  
    if ((layout == CblasRowMajor && trans == CblasNoTrans)
        || (layout == CblasColMajor && trans == CblasTrans)) {
      N iy = OFFSET(lenY, incY);
      for (i = 0; i < lenY; ++i) {
        X temp = zero;
        N ix = OFFSET(lenX, incX);
        for (j = 0; j < lenX; ++j) {
          temp += x[ix] * a[lda * i + j];
          ix += incX;
        }
        y[iy] += alpha * temp;
        iy += incY;
      }
    } else if ((layout == CblasRowMajor && trans == CblasTrans)
               || (layout == CblasColMajor && trans == CblasNoTrans)) {
      N ix = OFFSET(lenX, incX);
      for (j = 0; j < lenX; ++j) {
        const X temp = alpha * x[ix];
        if (temp != zero) {
          N iy = OFFSET(lenY, incY);
          for (i = 0; i < lenY; ++i) {
            y[iy] += temp * a[lda * j + i];
            iy += incY;
          }
        }
        ix += incX;
      }
    } else if(layout == CblasColMajor && trans == CblasConjTrans){
      N iy = OFFSET(lenY, incY);
      for (i = 0; i < lenY; ++i) {
        X temp = zero;
        N ix = OFFSET(lenX, incX);
        for (j = 0; j < lenX; ++j) {
          temp += x[ix] * conj(a[lda * i + j]);
          ix += incX;
        }
        y[iy] += alpha * temp;
        iy += incY;
      }
    } else if(layout == CblasRowMajor && trans == CblasConjTrans){
      N ix = OFFSET(lenX, incX);
      for (j = 0; j < lenX; ++j) {
        const X temp = alpha * x[ix];
        if (temp != zero) {
          N iy = OFFSET(lenY, incY);
          for (i = 0; i < lenY; ++i) {
            y[iy] += temp * conj(a[lda * j + i]);
            iy += incY;
          }
        }
        ix += incX;
      }
    } else {
      assert(0, "unrecognized operation");
    }
}


/**
*  @title ger blas function: Performs a rank-1 update of a general matrix.
*
*  @description   The ger routines perform a matrix-vector operation defined as
*
*                 A := alpha*x*y'+ A,
*                 where:
*                 alpha is a scalar,
*                 x is an m-element vector,
*                 y is an n-element vector,
*                 A is an m-by-n general matrix.
*
*  Input Parameters
*
*  @param layout: Specifies whether two-dimensional array storage is row-major 
*                 (CblasRowMajor) or column-major (CblasColMajor).
*
*  @param m:      Specifies the number of rows of the matrix A.
*                 The value of m must be at least zero.
*
*  @param n:      Specifies the number of columns of the matrix A.
*                 The value of n must be at least zero.
*
*  @param alpha:  Specifies the scalar alpha.
*
*  @param x:      Array, size at least (1 + (m - 1)*abs(incx)) . Before entry, the
*                 incremented array x must contain the m-element vector x.
*
*  @param incx:   Specifies the increment for the elements of x.
*                 The value of incx must not be zero.
*
*  @param y:      Array, size at least (1 + (n - 1)*abs(incy)) . Before entry, the
*                 incremented array y must contain the n-element vector y.
*
*  @param incy:   Specifies the increment for the elements of y.
*                 The value of incy must not be zero.
*
*  @param a:      Array, size lda *k.
*
*                 For Layout = CblasColMajor , k is n . Before entry, the leading m -by- n part
*                 of the array a must contain the matrix A.
*
*                 For Layout = CblasRowMajor , k is m . Before entry, the leading n -by- m part
*                 of the array a must contain the matrix A.
*
*  @param lda:    Specifies the leading dimension of a as declared in the calling
*                 (sub)program.
*
*                 For layout = CblasColMajor , the value of lda must be at least max(1, m).
*                 For layout = CblasRowMajor , the value of lda must be at least max(1, n).
*  
*  Output Parameters
*  @param a:      Overwritten by the updated matrix.
*
*
*/
void ger(N, X)(in CBLAS_ORDER layout, in N m, in N n, in X alpha, in X* x, in N incX,
               in X* y, in N incY, X* A, in N lda)
{
	N i, j;

    if (layout == CblasRowMajor && isFloatingPoint!X) {
      N ix = OFFSET(m, incX);
      for (i = 0; i < m; ++i) {
        const X tmp = alpha * x[ix];
        N jy = OFFSET(n, incY);
        for (j = 0; j < n; ++j) {
          A[lda * i + j] += y[jy] * tmp;
          jy += incY;
        }
        ix += incX;
      }
    } else if (layout == CblasColMajor && isFloatingPoint!X) {
      N jy = OFFSET(n, incY);
      for (j = 0; j < n; ++j) {
        const X tmp = alpha * y[jy];
        N ix = OFFSET(m, incX);
        for (i = 0; i < m; ++i) {
          A[i + lda * j] += x[ix] * tmp;
          ix += incX;
        }
        jy += incY;
      }
    } else {
      assert(0, "unrecognized operation");
    }
}

template gerComplex(CBLAS_TRANSPOSE trans){

	static if(trans == CblasTrans){
		T transFunction(T)(T x){
			return x;
		}
	} else {
		alias transFunction = conj;
	}

    void gerComplex(N, X: Complex!V, V = typeof(X.re))(in CBLAS_ORDER layout, in N m, in N n, in X alpha,
    	           in X* x, in N incX, in X* y, in N incY, X* A, in N lda)
    {
    	N i, j;
    
        if (layout == CblasRowMajor){
          N ix = OFFSET(m, incX);
          for (i = 0; i < m; ++i) {
            const X tmp = alpha * x[ix];
            N jy = OFFSET(n, incY);
            for (j = 0; j < n; ++j) {
              A[lda * i + j] += transFunction(y[jy]) * tmp;
              jy += incY;
            }
            ix += incX;
          }
        } else if (layout == CblasColMajor){
          N jy = OFFSET(n, incY);
          for (j = 0; j < n; ++j) {
            const X tmp = alpha * transFunction(y[jy]);
            N ix = OFFSET(m, incX);
            for (i = 0; i < m; ++i) {
              A[i + lda * j] += x[ix] * tmp;
              ix += incX;
            }
            jy += incY;
          }
        } else {
          assert(0, "unrecognized operation");
        }
    }
}

/* For complex transpose and complex hermitian ... */
alias gerComplex!CblasTrans geru;
alias gerComplex!CblasConjTrans gerc;

/**
*  @title hbmv blas function: Computes a matrix-vector product using a
*                             Hermitian band matrix.
*
*  @description   The hbmv routines perform a matrix-vector operation defined as 
*                 y := alpha*A*x + beta*y, where:
*
*                 alpha and beta are scalars,
*                 x and y are n-element vectors,
*                 A is an n-by-n Hermitian band matrix, with k super-diagonals.
*
*  Input Parameters:
*
*  @param layout: Specifies whether two-dimensional array storage is row-major
*                 (CblasRowMajor) or column-major (CblasColMajor).
*
*  @param uplo:   Specifies whether the upper or lower triangular part of the Hermitian band
*                 matrix A is used:
*
*                 If uplo = CblasUpper , then the upper triangular part of the matrix A is
*                 used.
*                 If uplo = CblasLower , then the low triangular part of the matrix A is used.
*
*  @param n:      Specifies the order of the matrix A. The value of n must be at least zero.
*
*  @param k:      For uplo = CblasUpper : Specifies the number of super-diagonals of the
*                 matrix A.
*                 For uplo = CblasLower : Specifies the number of sub-diagonals of the
*                 matrix A.
*                 The value of k must satisfy k >= 0.
*
*  @param alpha:  Specifies the scalar alpha.
*
*  @param a:      Array, size lda*n .
*                 layout = CblasColMajor :
*
*                 Before entry with uplo = CblasUpper , the leading (k + 1) by n part of
*                 the array a must contain the upper triangular band part of the Hermitian
*                 matrix. The matrix must be supplied column-by-column, with the leading
*                 diagonal of the matrix in row k of the array, the first super-diagonal starting
*                 at position 1 in row (k - 1) , and so on. The top left k by k triangle of the
*                 array a is not referenced.
*                 The following program segment transfers the upper triangular part of a
*                 Hermitian band matrix from conventional full matrix storage (matrix, with
*                 leading dimension ldm) to band storage (a, with leading dimension lda):
*
*                 for (j = 0; j < n; j++) {
*                     m = k - j;
*                     for (i = max( 0, j - k); i <= j; i++) {
*                         a[(m+i) + j*lda] = matrix[i + j*ldm];
*                     }
*                 }
*
*                 Before entry with uplo = CblasLower , the leading (k + 1) by n part of
*                 the array a must contain the lower triangular band part of the Hermitian
*                 matrix, supplied column-by-column, with the leading diagonal of the matrix
*                 in row 0 of the array, the first sub-diagonal starting at position 0 in row 1,
*                 and so on. The bottom right k by k triangle of the array a is not referenced.
*
*                 The following program segment transfers the lower triangular part of a
*                 Hermitian band matrix from conventional full matrix storage (matrix, with
*                 leading dimension ldm) to band storage (a, with leading dimension lda):
*
*                 for (j = 0; j < n; j++) {
*                     m = -j;
*                     for (i = j; i < min(n, j + k + 1); i++) {
*                         a[(m+i) + j*lda] = matrix[i + j*ldm];
*                     }
*                 }
*
*                 layout = CblasRowMajor:
*
*                 Before entry with uplo = CblasUpper , the leading ( k + 1)-by- n part of
*                 array a must contain the upper triangular band part of the Hermitian
*                 matrix. The matrix must be supplied row-by-row, with the leading diagonal
*                 of the matrix in column 0 of the array, the first super-diagonal starting at
*                 position 0 in column 1, and so on. The bottom right k -by- k triangle of array
*                 a is not referenced.
*
*                 The following program segment transfers the upper triangular part of a
*                 Hermitian band matrix from row-major full matrix storage (matrix with
*                 leading dimension ldm ) to row-major band storage ( a , with leading
*                 dimension lda ):
*
*                 for (i = 0; i < n; i++) {
*                     m = -i;
*                     for (j = i; j < MIN(n, i+k+1); j++) {
*                         a[(m+j) + i*lda] = matrix[j + i*ldm];
*                     }
*                 }
*
*                 Before entry with uplo = CblasLower , the leading ( k + 1)-by- n part of
*                 array a must contain the lower triangular band part of the Hermitian matrix,
*                 supplied row-by-row, with the leading diagonal of the matrix in column k of
*                 the array, the first sub-diagonal starting at position 1 in column k -1, and so
*                 on. The top left k -by- k triangle of array a is not referenced.
*
*                 The following program segment transfers the lower triangular part of a
*                 Hermitian row-major band matrix from row-major full matrix storage
*                 (matrix, with leading dimension ldm ) to row-major band storage ( a , with
*                 leading dimension lda ):
*
*                 for (i = 0; i < n; i++) {
*                     m = k - i;
*                     for (j = max(0, i-k); j <= i; j++) {
*                         a[(m+j) + i*lda] = matrix[j + i*ldm];
*                     }
*                 }
*
*                 The imaginary parts of the diagonal elements need not be set and are
*                 assumed to be zero.
*
*  @param lda:    Specifies the leading dimension of a as declared in the calling
*                 (sub)program. The value of lda must be at least (k + 1) .
*
*  @param x:      Array, size at least (1 + (n - 1)*abs(incx)) . Before entry, the
*                 incremented array x must contain the vector x.
*
*  @param incx:   Specifies the increment for the elements of x.
*                 The value of incx must not be zero.
*
*  @param beta:   Specifies the scalar beta.
*
*  @param y:      Array, size at least (1 + (n - 1)*abs(incy)) . Before entry, the
*                 incremented array y must contain the vector y.
*
*  @param incy:   Specifies the increment for the elements of y.
*                 The value of incy must not be zero.
*                 
*  Output Parameters:
*                 
*  @param y:      Overwritten by the updated vector y.
*
*/
void hbmv(N, X: Complex!V, V = typeof(X.re))(in CBLAS_LAYOUT layout, in CBLAS_UPLO uplo, in N n, 
	            in N k, in X alpha, in X* a, in N lda, in X* x, in N incX , in X beta, X* y, in N incY)
{
  	N i, j, indexA; X Aij;
  	const N layoutIndicator = (layout == CblasColMajor) ? -1 : 1;
  	X zero = complex(0, 0), one = complex(1, 0);
    
    if(n == 0)
    	return;

    /* if alpha = zero beta is one return */
    if(alpha == zero && beta == one)
    	return;

    /* form  y := beta*y */
    if (beta == zero) {
        N iy = OFFSET(n, incY);
        for (i = 0; i < n; ++i) {
            y[iy] = zero;
            iy += incY;
        }
    } else if (beta != one) {
        N iy = OFFSET(n, incY);
        for (i = 0; i < n; ++i) {
            y[iy] *= beta;
            iy += incY;
        }
    }
    
    if(alpha == zero)
    	return;
    
    /* form  y := alpha*A*x + y */
    
    if ((layout == CblasRowMajor && uplo == CblasUpper)
        || (layout == CblasColMajor && uplo == CblasLower)) {
        N ix = OFFSET(n, incX);
        N iy = OFFSET(n, incY);
        for (i = 0; i < n; ++i) {
            X temp1 = alpha*x[ix];
            X temp2 = zero;
            const N j_min = i + 1;
            const N j_max = min(n, i + k + 1);
            N jx = OFFSET(n, incX) + j_min * incX;
            N jy = OFFSET(n, incY) + j_min * incY;
            y[iy] += a[lda * i].re * temp1;
            for (j = j_min; j < j_max; ++j) {
                indexA = lda * i + (j - i);
                Aij = a[indexA];
                Aij.im = layoutIndicator*Aij.im;
                y[jy] += conj(Aij)*temp1;
                temp2 += Aij*x[jx];
                jx += incX;
                jy += incY;
            }
            y[iy] += alpha*temp2;
            ix += incX;
            iy += incY;
        }
    } else if ((layout == CblasRowMajor && uplo == CblasLower)
               || (layout == CblasColMajor && uplo == CblasUpper)) {
        N ix = OFFSET(n, incX);
        N iy = OFFSET(n, incY);
        for (i = 0; i < n; ++i) {
            X temp1 = alpha*x[ix];
            X temp2 = zero;
            const N j_min = (k > i ? 0 : i - k);
            const N j_max = i;
            N jx = OFFSET(n, incX) + j_min * incX;
            N jy = OFFSET(n, incY) + j_min * incY;
            
            for (j = j_min; j < j_max; ++j) {
                indexA = i * lda + (k - i + j);
                Aij = a[indexA];
                Aij.im = layoutIndicator*Aij.im;
                y[jy] += conj(Aij)*temp1;
                temp2 += x[jx]*Aij;
                jx += incX;
                jy += incY;
            }
            
            indexA = lda * i + k;

            y[iy] += a[indexA].re*temp1;
            y[iy] += alpha*temp2;

            ix += incX;
            iy += incY;
        }
    
    } else {
      assert(0, "unrecognized operation");
    }
}


/** 
*  @title hemv blas function: Computes a matrix-vector product using a Hermitian matrix.
*
*  @description      The hemv routines perform a matrix-vector operation defined as
*                    y := alpha*A*x + beta*y, where:
*   
*                    alpha and beta are scalars,
*                    x and y are n-element vectors,
*                    A is an n-by-n Hermitian matrix.
*  
*  Input Parameters:
*  
*  @param layout:    Specifies whether two-dimensional array storage is row-major
*                    (CblasRowMajor) or column-major (CblasColMajor).
*
*  @param uplo:      Specifies whether the upper or lower triangular part of the array a is used.
*                    If uplo = CblasUpper , then the upper triangular of the array a is used.
*                    If uplo = CblasLower , then the low triangular of the array a is used.
*
*  @param n:         Specifies the order of the matrix A. The value of n must be at least zero.
*
*  @param alpha:     Specifies the scalar alpha.
*
*  @param a:         Array, size lda*n .
*                    Before entry with uplo = CblasUpper , the leading n-by-n upper triangular
*                    part of the array a must contain the upper triangular part of the Hermitian
*                    matrix and the strictly lower triangular part of a is not referenced. Before
*                    entry with uplo = CblasLower , the leading n-by-n lower triangular part of
*                    the array a must contain the lower triangular part of the Hermitian matrix
*                    and the strictly upper triangular part of a is not referenced.
*                    The imaginary parts of the diagonal elements need not be set and are
*                    assumed to be zero.
*
*  @param lda:       Specifies the leading dimension of a as declared in the calling
*                    (sub)program. The value of lda must be at least max(1, n).
*
*  @param x:         Array, size at least (1 + (n - 1)*abs(incx)) . Before entry, the
*                    incremented array x must contain the n-element vector x.
*
*  @param incX:      Specifies the increment for the elements of x.
*                    The value of incx must not be zero.
*
*  @param beta:      Specifies the scalar beta. When beta is supplied as zero then y need not be
*                    set on input.
*
*  @param y:         Array, size at least (1 + (n - 1)*abs(incY)) . Before entry, the
*                    incremented array y must contain the n-element vector y.
*
*  @param incY:      Specifies the increment for the elements of y.
*                    The value of incY must not be zero.
*
*  Output Parameters:
*
*  @param y:         Overwritten by the updated vector y.
*
*/
void hemv(N, X: Complex!V, V = typeof(X.re))(in CBLAS_ORDER layout, in CBLAS_UPLO uplo,
             in N n, in X alpha, in X* a, in N lda,
             in X* x, in N incX, in X beta, X* y, in N incY)
{
    N i, j, j_min, j_max, ix, iy;
    immutable N layoutIndicator = (layout == CblasColMajor) ? -1 : 1;
  	X zero = complex(0, 0), one = complex(1, 0), Aij, temp1, temp2;

    if(alpha == zero && beta == one)
    	return;

    /* form  y := beta*y */
    if(beta == zero){
        iy = OFFSET(n, incY);
        for (i = 0; i < n; i++) {
        	y[iy] = zero;
            iy += incY;
        }
    } else if(beta != one) {
        iy = OFFSET(n, incY);
        for (i = 0; i < n; i++) {
          y[iy] = y[iy]*beta;
          iy += incY;
        }
    }

    if(alpha == zero)
        return;

    /* form  y := alpha*A*x + y */
    if ((layout == CblasRowMajor && uplo == CblasUpper)
        || (layout == CblasColMajor && uplo == CblasLower)) {
        ix = OFFSET(n, incX);
        iy = OFFSET(n, incY);
        for (i = 0; i < n; ++i) {
            temp1 = alpha*x[ix];
            temp2 = zero;
            j_min = i + 1;
            j_max = n;
            N jx = OFFSET(n, incX) + j_min * incX;
            N jy = OFFSET(n, incY) + j_min * incY;
            y[iy] += temp1*a[lda * i + i].re;
            for (j = j_min; j < j_max; j++) {
                Aij = a[lda * i + j];
        	    Aij.im = layoutIndicator*Aij.im;
                y[jy] += temp1*conj(Aij);
                temp2 += x[jx]*Aij;
                jx += incX;
                jy += incY;
            }
            y[iy] += alpha*temp2;
            ix += incX;
            iy += incY;
        }
    } else if ((layout == CblasRowMajor && uplo == CblasLower)
               || (layout == CblasColMajor && uplo == CblasUpper)) {
        ix = OFFSET(n, incX) + (n - 1) * incX;
        iy = OFFSET(n, incY) + (n - 1) * incY;
        for (i = n; i > 0 && --i;) {
            temp1 = alpha*x[ix];
            temp2 = zero;
            j_min = 0;
            j_max = i;
            N jx = OFFSET(n, incX) + j_min * incX;
            N jy = OFFSET(n, incY) + j_min * incY;
            y[iy] += temp1*a[lda * i + i].re;
            for (j = j_min; j < j_max; j++) {
            	Aij = a[lda * i + j];
            	Aij.im = layoutIndicator*Aij.im;
                y[jy] += temp1*conj(Aij);
                temp2 += x[jx]*Aij;
                jx += incX;
                jy += incY;
            }
            y[iy] += alpha*temp2;
            ix -= incX;
            iy -= incY;
        }
    } else {
        assert(0, "unrecognized operation");
    }
}


/** 
*  @title her blas function: Performs a rank-1 update of a Hermitian matrix.
*
*  @description      The her routines perform a matrix-vector operation defined as
*                    A := alpha*x*conjg(x') + A, where:
*
*                    alpha is a real scalar,
*                    x is an n-element vector,
*                    A is an n-by-n Hermitian matrix.
*
*  Input Parameters:
*
*  @param layout:    Specifies whether two-dimensional array storage is row-major
*                    (CblasRowMajor) or column-major (CblasColMajor).
*
*  @param uplo:      Specifies whether the upper or lower triangular part of the array a is used.
*                    If uplo = CblasUpper , then the upper triangular of the array a is used.
*                    If uplo = CblasLower , then the low triangular of the array a is used.
*
*  @param n:         Specifies the order of the matrix A. The value of n must be at least zero.
*
*  @param alpha:     Specifies the scalar alpha.
*
*  @param x:         Array, dimension at least (1 + (n - 1)*abs(incx)) . Before entry, the
*                    incremented array x must contain the n-element vector x.
*
*  @param incx:      Specifies the increment for the elements of x.
*                    The value of incx must not be zero.
*
*  @param a:         Array, size lda*n .
*
*                    Before entry with uplo = CblasUpper , the leading n-by-n upper triangular
*                    part of the array a must contain the upper triangular part of the Hermitian
*                    matrix and the strictly lower triangular part of a is not referenced.
*
*                    Before entry with uplo = CblasLower , the leading n-by-n lower triangular
*                    part of the array a must contain the lower triangular part of the Hermitian
*                    matrix and the strictly upper triangular part of a is not referenced.
*                    The imaginary parts of the diagonal elements need not be set and are
*                    assumed to be zero.
*
*  @param lda:       Specifies the leading dimension of a as declared in the calling
*                    (sub)program. The value of lda must be at least max(1, n) .
*
*  Output Parameters:
*
*  @param a:         With uplo = CblasUpper , the upper triangular part of the array a is
*                    overwritten by the upper triangular part of the updated matrix.
*
*                    With uplo = CblasLower , the lower triangular part of the array a is
*                    overwritten by the lower triangular part of the updated matrix.
*
*                    The imaginary parts of the diagonal elements are set to zero.
*
*/
void her(N, X: Complex!V, V = typeof(X.re))(in CBLAS_ORDER layout, in CBLAS_UPLO uplo, in N n, in V alpha,
	            in X* x, in N incX, X* a, in N lda)
{
	N i, j, ix, jx;
    immutable N layoutIndicator = (layout == CblasColMajor) ? -1 : 1;
  	X zero = complex(0, 0), one = complex(1, 0), Aij, xij, tmp;
    
    if (alpha == 0)
        return;
    
    if ((layout == CblasRowMajor && uplo == CblasUpper)
        || (layout == CblasColMajor && uplo == CblasLower)) {
        ix = OFFSET(n, incX);
        for (i = 0; i < n; i++) {
            tmp = alpha*complex(x[ix].re, layoutIndicator*x[ix].im);
            jx = ix;
            
            xij = complex(x[jx].re, -layoutIndicator*x[jx].im);
            Aij = a[lda * i + i];
            Aij.re += xij.re*tmp.re - xij.im*tmp.im;
            Aij.im = 0;
            a[lda * i + i] = Aij;
            jx += incX;
            
            for (j = i + 1; j < n; j++) {
                xij = complex(x[jx].re, -layoutIndicator*x[jx].im);
                a[lda * i + j] += xij*tmp;
                jx += incX;
            }
            ix += incX;
        }
    } else if ((layout == CblasRowMajor && uplo == CblasLower)
               || (layout == CblasColMajor && uplo == CblasUpper)) {
        ix = OFFSET(n, incX);
        for (i = 0; i < n; i++) {
            tmp = alpha*complex(x[ix].re, layoutIndicator*x[ix].im);
            jx = OFFSET(n, incX);
            for (j = 0; j < i; j++) {
                xij = complex(x[jx].re, -layoutIndicator*x[jx].im);
                a[lda * i + j] += xij*tmp;
                jx += incX;
            }
            
            xij = complex(x[jx].re, -layoutIndicator*x[jx].im);
            Aij = a[lda * i + i];
            Aij.re += xij.re*tmp.re - xij.im*tmp.im;
            Aij.im = 0;
            a[lda * i + i] = Aij;
            jx += incX;
            ix += incX;
        }
    } else {
        assert(0, "unrecognized operation");
    }
}

/** 
*  @title her2 blas function: Performs a rank-2 update of a Hermitian matrix.
*
*  @description      The her2 routines perform a matrix-vector operation defined as
*                    A := alpha *x*conjg(y') + conjg(alpha)*y *conjg(x') + A,
*
*                    alpha is a real scalar,
*                    x is an n-element vector,
*                    A is an n-by-n Hermitian matrix.
*
*  Input Parameters:
*
*  @param layout:    Specifies whether two-dimensional array storage is row-major
*                    (CblasRowMajor) or column-major (CblasColMajor).
*
*  @param uplo:      Specifies whether the upper or lower triangular part of the array a is used.
*                    If uplo = CblasUpper, then the upper triangular of the array a is used.
*                    If uplo = CblasLower, then the low triangular of the array a is used.
*
*  @param n:         Specifies the order of the matrix A. The value of n must be at least zero.
*
*  @param alpha:     Specifies the scalar alpha.
*
*  @param x:         Array, size at least (1 + (n - 1)*abs(incx)). Before entry, the
*                    incremented array x must contain the n-element vector x.
*
*  @param incx:      Specifies the increment for the elements of x.
*                    The value of incx must not be zero.
*
*  @param y:         Array, size at least (1 + (n - 1)*abs(incy)). Before entry, the
*                    incremented array y must contain the n-element vector y.
*
*  @param incy:      Specifies the increment for the elements of y.
*                    The value of incy must not be zero.
*
*  @param a:         Array, size lda*n.
*
*                    Before entry with uplo = CblasUpper, the leading n-by-n upper triangular
*                    part of the array a must contain the upper triangular part of the Hermitian
*                    matrix and the strictly lower triangular part of a is not referenced.
*                    Before entry with uplo = CblasLower, the leading n-by-n lower triangular
*                    part of the array a must contain the lower triangular part of the Hermitian
*                    matrix and the strictly upper triangular part of a is not referenced.
*                    The imaginary parts of the diagonal elements need not be set and are
*                    assumed to be zero.
*
*  @param lda:       Specifies the leading dimension of a as declared in the calling
*                    (sub)program. The value of lda must be at least max(1, n) .
*  
*  Output Parameters:
*
*  @param a:         With uplo = CblasUpper, the upper triangular part of the array a is
*                    overwritten by the upper triangular part of the updated matrix.
*  
*                    With uplo = CblasLower, the lower triangular part of the array a is
*                    overwritten by the lower triangular part of the updated matrix.
*
*                    The imaginary parts of the diagonal elements are set to zero.
*
*/
void her2(N, X: Complex!V, V = typeof(X.re))(in CBLAS_ORDER layout, in CBLAS_UPLO uplo,
                in N n, in X alpha, in X* x, in N incX,
                in X* y, in N incY, X* a, in N lda)
{
	N i, j, ix, iy, jx, jy;
    immutable N layoutIndicator = (layout == CblasColMajor) ? -1 : 1;
  	X zero = complex(0, 0), one = complex(1, 0), Aij, tmp1, tmp2;

    if(alpha == zero)
        return;

    if((layout == CblasRowMajor && uplo == CblasUpper)
        || (layout == CblasColMajor && uplo == CblasLower)) {
        ix = OFFSET(n, incX);
        iy = OFFSET(n, incY);
        for (i = 0; i < n; ++i) {
        	/* tmp1 = alpha Xi */

            tmp1 = alpha*x[ix];
            
            /* tmp2 = conj(alpha) Yi */
            
            tmp2.re  = alpha.re*y[iy].re + alpha.im*y[iy].im;
            tmp2.im  = -alpha.im*y[iy].re + alpha.re*y[iy].im;
            
            jx = ix + incX;
            jy = iy + incY;
            
            /* Aij = alpha*Xi*conj(Yj) + conj(alpha)*Yi*conj(Xj) */

            Aij = a[lda * i + i];
            Aij.re += 2*(tmp1.re*y[iy].re + tmp1.im*y[iy].im);
            Aij.im = 0;
            a[lda * i + i] = Aij;
            
            for (j = i + 1; j < n; ++j) {
                Aij = a[lda*i + j];
                Aij.re += ((tmp1.re * y[jy].re + tmp1.im * y[jy].im) +
                                        (tmp2.re * x[jx].re + tmp2.im * x[jx].im));
                Aij.im += layoutIndicator*((tmp1.im * y[jy].re - tmp1.re * y[jy].im) +
                                            (tmp2.im * x[jx].re - tmp2.re * x[jx].im));
                a[lda * i + j] = Aij;
                jx += incX;
                jy += incY;
            }
            ix += incX;
            iy += incY;
        }
    } else if((layout == CblasRowMajor && uplo == CblasLower)
               || (layout == CblasColMajor && uplo == CblasUpper)) {
        ix = OFFSET(n, incX);
        iy = OFFSET(n, incY);
        for (i = 0; i < n; ++i) {
            
            tmp1 = alpha*x[ix];
            tmp2.re = alpha.re*y[iy].re + alpha.im*y[iy].im;
            tmp2.im = -alpha.im*y[iy].re + alpha.re*y[iy].im;
            
            jx = OFFSET(n, incX);
            jy = OFFSET(n, incY);
            
            /* Aij = alpha*Xi*conj(Yj) + conj(alpha)*Yi*conj(Xj) */
            
            for (j = 0; j < i; ++j) {
                Aij = a[lda * i + j];
                Aij.re += ((tmp1.re * y[jy].re + tmp1.im * y[jy].im)
                                        + (tmp2.re * x[jx].re + tmp2.im * x[jx].im));
                Aij.im += layoutIndicator*((tmp1.im * y[jy].re - tmp1.re * y[jy].im) +
                          (tmp2.im * x[jx].re - tmp2.re * x[jx].im));
                a[lda * i + j] = Aij;
                jx += incX;
                jy += incY;
            }
            
            Aij = a[lda * i + i];
            Aij.re += 2 * (tmp1.re * y[iy].re + tmp1.im * y[iy].im);
            Aij.im = 0;
            a[lda * i + i] = Aij;
            ix += incX;
            iy += incY;
        }
    } else {
        assert(0, "unrecognized operation");
    }
}



/** 
*  @title hpmv blas function: Computes a matrix-vector product using a Hermitian 
*                             packed matrix.
*
*  @description      The hpmv routines perform a matrix-vector operation defined as
*                    y := alpha*A*x + beta*y,
*
*                    alpha and beta are scalars,
*                    x is an n-element vector,
*                    A is an n-by-n Hermitian matrix, supplied in packed form.
*
*  Input Parameters:
*
*  @param layout:    Specifies whether two-dimensional array storage is row-major
*                    (CblasRowMajor) or column-major (CblasColMajor).
*
*  @param uplo:      Specifies whether the upper or lower triangular part of the matrix A is
*                    supplied in the packed array ap.
*
*                    If uplo = CblasUpper , then the upper triangular part of the matrix A is
*                    supplied in the packed array ap.
*
*                    If uplo = CblasLower , then the low triangular part of the matrix A is
*                    supplied in the packed array ap.
*
*  @param n:         Specifies the order of the matrix A. The value of n must be at least zero.
*
*  @param alpha:     Specifies the scalar alpha.
*
*  @param ap:        Array, size at least ((n*(n + 1))/2).
*
*                    For Layout = CblasColMajor :
*
*                    Before entry with uplo = CblasUpper , the array ap must contain the upper
*                    triangular part of the Hermitian matrix packed sequentially, column-by-
*                    column, so that ap[0] contains A 1, 1 , ap[1] and ap[2] contain A 1, 2 and
*                    A 2, 2 respectively, and so on. Before entry with uplo = CblasLower , the
*                    array ap must contain the lower triangular part of the Hermitian matrix
*                    packed sequentially, column-by-column, so that ap[0] contains A 1, 1 ,
*                    ap[1] and ap[2] contain A 2, 1 and A 3, 1 respectively, and so on.
*
*                    For Layout = CblasRowMajor :
*
*                    Before entry with uplo = CblasUpper , the array ap must contain the upper
*                    triangular part of the Hermitian matrix packed sequentially, row-by-row,
*                    ap[0] contains A 1, 1 , ap[1] and ap[2] contain A 1, 2 and A 1, 3 respectively,
*                    and so on. Before entry with uplo = CblasLower , the array ap must
*                    contain the lower triangular part of the Hermitian matrix packed
*                    sequentially, row-by-row, so that ap[0] contains A 1, 1 , ap[1] and ap[2]
*                    contain A 2, 1 and A 2, 2 respectively, and so on.
*
*                    The imaginary parts of the diagonal elements need not be set and are
*                    assumed to be zero.
*
*  @param x:         Array, size at least (1 +(n - 1)*abs(incx)) . Before entry, the
*                    incremented array x must contain the n-element vector x.
*
*  @param incx:      Specifies the increment for the elements of x.
*                    The value of incx must not be zero.
*
*  @param beta:      Specifies the scalar beta.
*                    When beta is equal to zero then y need not be set on input.
*
*  @param y:         Array, size at least (1 + (n - 1)*abs(incy)) . Before entry, the
*                    incremented array y must contain the n-element vector y.
*
*  @param incy:      Specifies the increment for the elements of y.
*                    The value of incy must not be zero.
*
*  Output Parameters:
*
*  @param y:         Overwritten by the updated vector y.
*
*/
void hpmv(N, X: Complex!V, V = typeof(X.re))(in CBLAS_ORDER layout, in CBLAS_UPLO uplo, in N n, 
	        in X alpha, in X* ap, in X* x, in N incX, in X beta, X* y, in N incY)
{
    N i, j;
    X Aij, temp1, temp2;
    V Aii;
    const N layoutIndicator = (layout == CblasColMajor) ? -1 : 1;
    X zero = complex(0, 0), one = complex(1, 0);

    if(alpha == zero && beta == one)
    	return;

    /* form  y := beta*y */
    if (beta == zero) {
        N iy = OFFSET(n, incY);
        for (i = 0; i < n; ++i) {
            y[iy] = zero;
            iy += incY;
        }
    } else if (beta != one) {
      N iy = OFFSET(n, incY);
      for (i = 0; i < n; ++i) {
        y[iy] = y[iy]*beta;
        iy += incY;
      }
    }

    if(alpha == zero)
    	return;

    /* form  y := alpha*A*x + y */

    if ((layout == CblasRowMajor && uplo == CblasUpper)
        || (layout == CblasColMajor && uplo == CblasLower)) {
        N ix = OFFSET(n, incX);
        N iy = OFFSET(n, incY);
        for (i = 0; i < n; ++i) {
            temp1 = alpha*x[ix];
            temp2 = zero;
            const N j_min = i + 1;
            const N j_max = n;
            N jx = OFFSET(n, incX) + j_min * incX;
            N jy = OFFSET(n, incY) + j_min * incY;
            /*BASE Aii_real = CONST_REAL(ap, TPUP(n, i, i));*/
            Aii = ap[TPUP(n, i, i)].re;
            /* Aii_imag is zero */
            y[iy] += temp1*Aii;
            for (j = j_min; j < j_max; ++j) {
                Aij = ap[TPUP(n, i, j)];
                Aij.im = layoutIndicator*Aij.im;
                y[jy] += temp1*conj(Aij);
                temp2 += x[jx]*Aij;
                jx += incX;
                jy += incY;
            }
            y[iy] += alpha*temp2;
            ix += incX;
            iy += incY;
        }
    } else if ((layout == CblasRowMajor && uplo == CblasLower)
               || (layout == CblasColMajor && uplo == CblasUpper)) {

        N ix = OFFSET(n, incX);
        N iy = OFFSET(n, incY);
        for (i = 0; i < n; ++i) {
            temp1 = alpha*x[ix];
            temp2 = zero;
            const N j_min = 0;
            const N j_max = i;
            N jx = OFFSET(n, incX) + j_min * incX;
            N jy = OFFSET(n, incY) + j_min * incY;
            /*BASE Aii_real = CONST_REAL(ap, TPLO(n, i, i));*/
            Aii = ap[TPLO(n, i, i)].re;
            /* Aii_imag is zero */
            y[iy] += temp1*Aii;
            for (j = j_min; j < j_max; ++j) {
                Aij = ap[TPLO(n, i, j)];
                Aij.im = layoutIndicator*Aij.im;
                y[jy] += temp1*conj(Aij);
                temp2 += x[jx]*Aij;
                jx += incX;
                jy += incY;
            }
            y[iy] += alpha*temp2;
            ix += incX;
            iy += incY;
        }

    } else {
        assert(0, "unrecognized operation");
    }
}

/** 
*  @title hpr blas function: Performs a rank-1 update of a Hermitian packed matrix.
*
*  @description      The hpr routines perform a matrix-vector operation defined as
*                    A := alpha*x*conjg(x') + A,
*
*                    alpha is a real scalars,
*                    x is an n-element vector,
*                    A is an n-by-n Hermitian matrix, supplied in packed form.
*
*  Input Parameters:
*
*  @param layout:    Specifies whether two-dimensional array storage is row-major
*                    (CblasRowMajor) or column-major (CblasColMajor).
*
*  @param uplo:      Specifies whether the upper or lower triangular part of the matrix A is
*                    supplied in the packed array ap.
*
*                    If uplo = CblasUpper, the upper triangular part of the matrix A is supplied
*                    in the packed array ap.
*
*                    If uplo = CblasLower, the low triangular part of the matrix A is supplied in
*                    the packed array ap.
*
*  @param n:         Specifies the order of the matrix A. The value of n must be at least zero.
*
*  @param alpha:     Specifies the scalar alpha.
*
*  @param x:         Array, size at least (1 + (n - 1)*abs(incx)) . Before entry, the
*                    incremented array x must contain the n-element vector x.
*
*  @param incx:      Specifies the increment for the elements of x. incx must not be zero.
*
*  @param ap:        Array, size at least ((n*(n + 1))/2).
*
*                    For Layout = CblasColMajor :
*
*                    Before entry with uplo = CblasUpper , the array ap must contain the upper
*                    triangular part of the Hermitian matrix packed sequentially, column-by-
*                    column, so that ap[0] contains A 1, 1 , ap[1] and ap[2] contain A 1, 2 and
*                    A 2, 2 respectively, and so on.
*
*                    Before entry with uplo = CblasLower , the array ap must contain the lower
*                    triangular part of the Hermitian matrix packed sequentially, column-by-
*                    column, so that ap[0] contains A 1, 1 , ap[1] and ap[2] contain A 2, 1 and
*                    A 3, 1 respectively, and so on.
*
*                    For Layout = CblasRowMajor :
*
*                    Before entry with uplo = CblasUpper , the array ap must contain the upper
*                    triangular part of the Hermitian matrix packed sequentially, row-by-row,
*                    ap[0] contains A 1, 1 , ap[1] and ap[2] contain A 1, 2 and A 1, 3 respectively,
*                    and so on.
*
*                    Before entry with uplo = CblasLower , the array ap must contain the lower
*                    triangular part of the Hermitian matrix packed sequentially, row-by-row, so
*                    that ap[0] contains A 1, 1 , ap[1] and ap[2] contain A 2, 1 and A 2, 2
*                    respectively, and so on.
*
*                    The imaginary parts of the diagonal elements need not be set and are
*                    assumed to be zero.
*
*  Output Parameters:
*
*  @param ap:        With uplo = CblasUpper , overwritten by the upper triangular part of the
*                    updated matrix.
*
*                    With uplo = CblasLower , overwritten by the lower triangular part of the
*                    updated matrix.
*
*                    The imaginary parts of the diagonal elements are set to zero.
*
*/
void hpr(N, X: Complex!V, V = typeof(X.re))(in CBLAS_ORDER layout, in CBLAS_UPLO uplo, 
	                                        in N n, in V alpha, in X* x, in N incX, X* ap)
{
    N i, j;
    const N layoutIndicator = (layout == CblasColMajor) ? -1 : 1;
    X zero = complex(0, 0), one = complex(1, 0), tmp, Xi, APij;

    if(alpha == zero)
    	return;
    
    if ((layout == CblasRowMajor && uplo == CblasUpper)
        || (layout == CblasColMajor && uplo == CblasLower)) {
        N ix = OFFSET(n, incX);
        for (i = 0; i < n; ++i) {
            tmp = alpha*x[ix];
            tmp.im = layoutIndicator*tmp.im;
            N jx = ix;
            Xi = x[jx];
            Xi.im = -layoutIndicator*Xi.im;
            APij = ap[TPUP(n, i, i)];
            APij.re += Xi.re*tmp.re - Xi.im*tmp.im;
            APij.im = 0;
            ap[TPUP(n, i, i)] = APij;
            jx += incX;
            
            for (j = i + 1; j < n; ++j) {
                Xi = x[jx];
                Xi.im = -layoutIndicator*Xi.im;
                ap[TPUP(n, i, j)] += Xi*tmp;
                jx += incX;
            }
            ix += incX;
        }
    } else if ((layout == CblasRowMajor && uplo == CblasLower)
               || (layout == CblasColMajor && uplo == CblasUpper)) {
        N ix = OFFSET(n, incX);
        for (i = 0; i < n; ++i) {
            tmp = alpha*x[ix];
            tmp.im = layoutIndicator*tmp.im;
            N jx = OFFSET(n, incX);
            for (j = 0; j < i; ++j) {
                Xi = x[jx];
                Xi.im = -layoutIndicator*Xi.im;
                ap[TPLO(n, i, j)] += Xi*tmp;
                jx += incX;
            }
            Xi = x[jx];
            Xi.im = -layoutIndicator*Xi.im;
            APij = ap[TPLO(n, i, i)];
            APij.re += Xi.re*tmp.re - Xi.im*tmp.im;
            APij.im = 0;
            ap[TPLO(n, i, i)] = APij;
            jx += incX;
            ix += incX;
        }
    } else {
        assert(0, "unrecognized operation");
    }
}


/** 
*  @title hpr2 blas function: Performs a rank-2 update of a Hermitian packed matrix.
*
*  @description      The hpr routines perform a matrix-vector operation defined as
*                    A := alpha*x*conjg(y') + conjg(alpha)*y*conjg(x') + A, where
*
*                    alpha is a real scalars,
*                    x is an n-element vector,
*                    A is an n-by-n Hermitian matrix, supplied in packed form.
*
*  Input Parameters:
*
*  @param layout:    Specifies whether two-dimensional array storage is row-major
*                    (CblasRowMajor) or column-major (CblasColMajor).
*
*  @param uplo:      Specifies whether the upper or lower triangular part of the matrix A is
*                    supplied in the packed array ap.
*
*                    If uplo = CblasUpper , then the upper triangular part of the matrix A is
*                    supplied in the packed array ap.
*
*                    If uplo = CblasLower , then the low triangular part of the matrix A is
*                    supplied in the packed array ap.
*
*  @param n:         Specifies the order of the matrix A. The value of n must be at least zero.
*
*  @param alpha:     Specifies the scalar alpha.
*
*  @param x:         Array, dimension at least (1 +(n - 1)*abs(incx)) . Before entry, the
*                    incremented array x must contain the n-element vector x.
*
*  @param incx:      Specifies the increment for the elements of x.
*                    The value of incx must not be zero.
*
*  @param y:         Array, size at least (1 +(n - 1)*abs(incy)) . Before entry, the
*                    incremented array y must contain the n-element vector y.
*
*  @param incy:      Specifies the increment for the elements of y.
*                    The value of incy must not be zero.
*
*  @param ap:        Array, size at least ((n*(n + 1))/2).
*
*                    For Layout = CblasColMajor :
*
*                    Before entry with uplo = CblasUpper , the array ap must contain the upper
*                    triangular part of the Hermitian matrix packed sequentially, column-by-
*                    column, so that ap[0] contains A 1, 1 , ap[1] and ap[2] contain A 1, 2 and
*                    A 2, 2 respectively, and so on.
*
*                    Before entry with uplo = CblasLower , the array ap must contain the lower
*                    triangular part of the Hermitian matrix packed sequentially, column-by-
*                    column, so that ap[0] contains A 1, 1 , ap[1] and ap[2] contain A 2, 1 and
*                    A 3, 1 respectively, and so on.
*
*                    For Layout = CblasRowMajor :
*
*                    Before entry with uplo = CblasUpper , the array ap must contain the upper
*                    triangular part of the Hermitian matrix packed sequentially, row-by-row,
*                    ap[0] contains A 1, 1 , ap[1] and ap[2] contain A 1, 2 and A 1, 3 respectively,
*                    and so on.
*
*                    Before entry with uplo = CblasLower , the array ap must contain the lower
*                    triangular part of the Hermitian matrix packed sequentially, row-by-row, so
*                    that ap[0] contains A 1, 1 , ap[1] and ap[2] contain A 2, 1 and A 2, 2
*                    respectively, and so on.
*
*                    The imaginary parts of the diagonal elements need not be set and are
*                    assumed to be zero.
*
*  Output Parameters:
*
*  @param ap:        With uplo = CblasUpper , overwritten by the upper triangular part of the
*                    updated matrix.
*
*                    With uplo = CblasLower , overwritten by the lower triangular part of the
*                    updated matrix.
*
*                    The imaginary parts of the diagonal elements need are set to zero.
*
*/
void hpr2(N, X: Complex!V, V = typeof(X.re))(in CBLAS_ORDER layout, in CBLAS_UPLO uplo,
             in N n, in X alpha, in X* x, in N incX,
             in X* y, in N incY, X* ap)
{
    N i, j;
    const N layoutIndicator = (layout == CblasColMajor) ? -1 : 1;
    X zero = complex(0, 0), one = complex(1, 0), tmp1, tmp2, Xi, APij;

    if(alpha == zero)
    	return;
    
    if ((layout == CblasRowMajor && uplo == CblasUpper)
        || (layout == CblasColMajor && uplo == CblasLower)) {
        N ix = OFFSET(n, incX);
        N iy = OFFSET(n, incY);
        for (i = 0; i < n; i++) {

            tmp1 = alpha*x[ix];
            tmp2 = conj(alpha)*y[iy];
            
            N jx = ix + incX;
            N jy = iy + incY;
            
            /* Aij = alpha*Xi*conj(Yj) + conj(alpha)*Yi*conj(Xj) */
            
            APij = ap[TPUP(n, i, i)];
            APij.re += 2*(tmp1.re*y[iy].re + tmp1.im*y[iy].im);
            APij.im = 0;
            ap[TPUP(n, i, i)] = APij;
            
            for (j = i + 1; j < n; j++) {
                APij = ap[TPUP(n, i, j)];
                APij.re += ((tmp1.re * y[jy].re + tmp1.im*y[jy].im)
                                            + (tmp2.re*x[jx].re + tmp2.im*x[jx].im));
                APij.im += layoutIndicator*((tmp1.im * y[jy].re - tmp1.re*y[jy].im) +
                                   (tmp2.im*x[jx].re - tmp2.re*x[jx].im));
                ap[TPUP(n, i, j)] = APij;
                jx += incX;
                jy += incY;
            }
            ix += incX;
            iy += incY;
        }
    } else if ((layout == CblasRowMajor && uplo == CblasLower)
               || (layout == CblasColMajor && uplo == CblasUpper)) {
        
        N ix = OFFSET(n, incX);
        N iy = OFFSET(n, incY);
        for (i = 0; i < n; i++) {
            tmp1 = alpha*x[ix];
            
            tmp2 = conj(alpha)*y[iy];
            
            N jx = OFFSET(n, incX);
            N jy = OFFSET(n, incY);
            
            /* Aij = alpha*Xi*conj(Yj) + conj(alpha)*Yi*conj(Xj) */
            
            for (j = 0; j < i; j++) {
                APij = ap[TPLO(n, i, j)];
                APij.re += ((tmp1.re * y[jy].re + tmp1.im*y[jy].im)
                                            + (tmp2.re*x[jx].re + tmp2.im*x[jx].im));
                APij.im += layoutIndicator*((tmp1.im * y[jy].re - tmp1.re*y[jy].im) +
                                   (tmp2.im*x[jx].re - tmp2.re*x[jx].im));
                ap[TPLO(n, i, j)] = APij;
                jx += incX;
                jy += incY;
            }
            
            APij = ap[TPLO(n, i, i)];
            APij.re += 2*(tmp1.re * y[iy].re + tmp1.im * y[iy].im);
            APij.im = 0;
            ap[TPLO(n, i, i)] = APij;

            ix += incX;
            iy += incY;
        }
    } else {
        assert(0, "unrecognized operation");
    }
}



/** 
*  @title sbmv blas function: Computes a matrix-vector product using a symmetric band matrix.
*
*  @description      The sbmv routines perform a matrix-vector operation defined as
*                    y := alpha*A*x + beta*y, where
*
*                    alpha and beta are scalars,
*                    x and y are n-element vectors,
*                    A is an n-by-n symmetric band matrix, with k super-diagonals.
*
*  Input Parameters:
*
*  @param layout:    Specifies whether two-dimensional array storage is row-major
*                    (CblasRowMajor) or column-major (CblasColMajor).
*
*  @param uplo:      Specifies whether the upper or lower triangular part of the band matrix A is
*                    used:
*                    if uplo = CblasUpper - upper triangular part;
*                    if uplo = CblasLower - low triangular part.
*
*  @param n:         Specifies the order of the matrix A. The value of n must be at least zero.
*
*  @param k:         Specifies the number of super-diagonals of the matrix A.
*                    The value of k must satisfy 0 ≤k.
*
*  @param alpha:     Specifies the scalar alpha.
*
*  @param a:         Array, size lda*n . Before entry with uplo = CblasUpper , the leading (k
*                    + 1) by n part of the array a must contain the upper triangular band part
*                    of the symmetric matrix, supplied column-by-column, with the leading
*                    diagonal of the matrix in row k of the array, the first super-diagonal starting
*                    at position 1 in row (k - 1) , and so on. The top left k by k triangle of the
*                    array a is not referenced.
*                    The following program segment transfers the upper triangular part of a
*                    symmetric band matrix from conventional full matrix storage (matrix, with
*                    leading dimension ldm) to band storage (a, with leading dimension lda):
*                    for (j = 0; j < n; j++) {
*                        m = k - j;
*                        for (i = max( 0, j - k); i <= j; i++) {
*                            a[(m+i) + j*lda] = matrix[i + j*ldm];
*                        }
*                    }
*                    Before entry with uplo = CblasLower , the leading (k + 1) by n part of
*                    the array a must contain the lower triangular band part of the symmetric
*                    matrix, supplied column-by-column, with the leading diagonal of the matrix
*                    in row 0 of the array, the first sub-diagonal starting at position 0 in row 1,
*                    and so on. The bottom right k by k triangle of the array a is not referenced.
*                    The following program segment transfers the lower triangular part of a
*                    symmetric band matrix from conventional full matrix storage (matrix, with
*                    leading dimension ldm) to band storage (a, with leading dimension lda):
*                    for (j = 0; j < n; j++) {
*                        m = -j;
*                        for (i = j; i < min(n, j + k + 1); i++) {
*                            a[(m+i) + j*lda] = matrix[i + j*ldm];
*                    }
*                    }
*                    Layout = CblasRowMajor:
*                    Before entry with uplo = CblasUpper , the leading ( k + 1)-by- n part of
*                    array a must contain the upper triangular band part of the symmetric
*                    matrix. The matrix must be supplied row-by-row, with the leading diagonal
*                    of the matrix in column 0 of the array, the first super-diagonal starting at
*                    position 0 in column 1, and so on. The bottom right k -by- k triangle of array
*                    a is not referenced.
*                    The following program segment transfers the upper triangular part of a
*                    symmetric band matrix from row-major full matrix storage (matrix with
*                    leading dimension ldm ) to row-major band storage ( a , with leading
*                    dimension lda ):
*                    for (i = 0; i < n; i++) {
*                        m = -i;
*                        for (j = i; j < MIN(n, i+k+1); j++) {
*                            a[(m+j) + i*lda] = matrix[j + i*ldm];
*                        }
*                    }
*                    Before entry with uplo = CblasLower , the leading ( k + 1)-by- n part of
*                    array a must contain the lower triangular band part of the symmetric
*                    matrix, supplied row-by-row, with the leading diagonal of the matrix in
*                    column k of the array, the first sub-diagonal starting at position 1 in column
*                    k -1, and so on. The top left k -by- k triangle of array a is not referenced.
*                    The following program segment transfers the lower triangular part of a
*                    symmetric row-major band matrix from row-major full matrix storage
*                    (matrix, with leading dimension ldm ) to row-major band storage ( a , with
*                    leading dimension lda ):
*                    for (i = 0; i < n; i++) {
*                        m = k - i;
*                        for (j = max(0, i-k); j <= i; j++) {
*                            a[(m+j) + i*lda] = matrix[j + i*ldm];
*                        }
*                    }
*
*  @param lda:       Specifies the leading dimension of a as declared in the calling
*                    (sub)program. The value of lda must be at least (k + 1).
*
*  @param x:         Array, size at least (1 + (n - 1)*abs(incx)) . Before entry, the
*                    incremented array x must contain the vector x.
*
*  @param incx:      Specifies the increment for the elements of x.
*                    The value of incx must not be zero.
*
*  @param beta:      Specifies the scalar beta.
*
*  @param y:         Array, size at least (1 + (n - 1)*abs(incy)) . Before entry, the
*                    incremented array y must contain the vector y.
*
*  @param incy:      Specifies the increment for the elements of y.
*                    The value of incy must not be zero.
*
*  Output Parameters
*
*  @param y:         Overwritten by the updated vector y.
*
*/
void sbmv(N, X)(in CBLAS_ORDER layout, in CBLAS_UPLO uplo,
             in N n, in N k, in X alpha, in X* a,
             in N lda, in X* x, in N incX,
             in X beta, X* y, in N incY)
{
    N i, j;
    X zero = 0, one = 1;
    
    if (n == 0)
        return;
    
    if (alpha == 0.0 && beta == 1.0)
        return;
    
    /* form  y := beta*y */
    if (beta == 0.0) {
        N iy = OFFSET(n, incY);
        for (i = 0; i < n; ++i) {
            y[iy] = 0.0;
            iy += incY;
        }
    } else if (beta != 1.0) {
        N iy = OFFSET(n, incY);
        for (i = 0; i < n; ++i) {
            y[iy] *= beta;
            iy += incY;
        }
    }
    
    if (alpha == 0.0)
      return;
    
    /* form  y := alpha*A*x + y */
    
    if ((layout == CblasRowMajor && uplo == CblasUpper)
        || (layout == CblasColMajor && uplo == CblasLower)) {
        N ix = OFFSET(n, incX);
        N iy = OFFSET(n, incY);
        
        for (i = 0; i < n; ++i) {
            X tmp1 = alpha * x[ix];
            X tmp2 = 0.0;
            immutable N j_min = i + 1;
            immutable N j_max = min(n, i + k + 1);
            N jx = OFFSET(n, incX) + j_min * incX;
            N jy = OFFSET(n, incY) + j_min * incY;
            y[iy] += tmp1 * a[0 + i * lda];
            for (j = j_min; j < j_max; ++j) {
                X Aij = a[(j - i) + i * lda];
                y[jy] += tmp1 * Aij;
                tmp2 += Aij * x[jx];
                jx += incX;
                jy += incY;
            }
            y[iy] += alpha * tmp2;
            ix += incX;
            iy += incY;
        }
    } else if ((layout == CblasRowMajor && uplo == CblasLower)
               || (layout == CblasColMajor && uplo == CblasUpper)) {
        N ix = OFFSET(n, incX);
        N iy = OFFSET(n, incY);
        
        for (i = 0; i < n; ++i) {
            X tmp1 = alpha * x[ix];
            X tmp2 = 0.0;
            immutable N j_min = (i > k) ? i - k : 0;
            immutable N j_max = i;
            N jx = OFFSET(n, incX) + j_min * incX;
            N jy = OFFSET(n, incY) + j_min * incY;
            for (j = j_min; j < j_max; ++j) {
                X Aij = a[(k - i + j) + i * lda];
                y[jy] += tmp1 * Aij;
                tmp2 += Aij * x[jx];
                jx += incX;
                jy += incY;
            }
            y[iy] += tmp1 * a[k + i * lda] + alpha * tmp2;
            ix += incX;
            iy += incY;
        }
    } else {
        assert(0, "unrecognized operation");
    }
}



/** 
*  @title spmv blas function: Computes a matrix-vector product using a symmetric packed matrix.
*
*  @description      The spmv routines perform a matrix-vector operation defined as
*                    y := alpha*A*x + beta*y, where
*
*                    alpha and beta are scalars,
*                    x and y are n-element vectors,
*                    A is an n-by-n symmetric matrix, supplied in packed form.
*
*  Input Parameters:
*
*  @param layout:    Specifies whether two-dimensional array storage is row-major
*                    (CblasRowMajor) or column-major (CblasColMajor).
*
*  @param uplo:      Specifies whether the upper or lower triangular part of the matrix A is
*                    supplied in the packed array ap.
*
*                    If uplo = CblasUpper, then the upper triangular part of the matrix A is
*                    supplied in the packed array ap.
*
*                    If uplo = CblasLower, then the low triangular part of the matrix A is
*                    supplied in the packed array ap.
*
*  @param n:         Specifies the order of the matrix A. The value of n must be at least zero.
*
*  @param alpha:     Specifies the scalar alpha.
*
*  @param ap:        Array, size at least ((n*(n + 1))/2).
*
*                    For layout = CblasColMajor :
*
*                    Before entry with uplo = CblasUpper , the array ap must contain the upper
*                    triangular part of the symmetric matrix packed sequentially, column-by-
*                    column, so that ap[0] contains A 1, 1 , ap[1] and ap[2] contain A 1, 2 and
*                    A 2, 2 respectively, and so on. Before entry with uplo = CblasLower , the
*                    array ap must contain the lower triangular part of the symmetric matrix
*                    packed sequentially, column-by-column, so that ap[0] contains A 1, 1 ,
*                    ap[1] and ap[2] contain A 2, 1 and A 3, 1 respectively, and so on.
*
*                    For layout = CblasRowMajor:
*
*                    Before entry with uplo = CblasUpper , the array ap must contain the upper
*                    triangular part of the symmetric matrix packed sequentially, row-by-row,
*                    ap[0] contains A 1, 1 , ap[1] and ap[2] contain A 1, 2 and A 1, 3 respectively,
*                    and so on. Before entry with uplo = CblasLower , the array ap must
*                    contain the lower triangular part of the symmetric matrix packed
*                    sequentially, row-by-row, so that ap[0] contains A 1, 1 , ap[1] and ap[2]
*                    contain A 2, 1 and A 2, 2 respectively, and so on.
*
*  @param x:         Array, size at least (1 + (n - 1)*abs(incx)) . Before entry, the
*                    incremented array x must contain the n-element vector x.
*
*  @param incx:      Specifies the increment for the elements of x.
*                    The value of incx must not be zero.
*
*  @param beta:      Specifies the scalar beta.
*                    When beta is supplied as zero, then y need not be set on input.
*
*  @param y:         Array, size at least (1 + (n - 1)*abs(incy)) . Before entry, the
*                    incremented array y must contain the n-element vector y.
*
*  @param incy:      Specifies the increment for the elements of y.
*                    The value of incy must not be zero.
*
*  Output Parameters
*
*  @param y:         Overwritten by the updated vector y.
*
*/
void spmv(X, N)(in CBLAS_ORDER layout, in CBLAS_UPLO uplo,
                in N n, in X alpha, in X* ap,
                in X* x, in N incX, in X beta, X* y,
                in N incY)
{
    N i, j;
    X zero = 0, one = 1;
    
    if (alpha == 0.0 && beta == 1.0)
      return;
    
    /* form  y := beta*y */
    if (beta == 0.0) {
        N iy = OFFSET(n, incY);
        for (i = 0; i < n; ++i) {
            y[iy] = 0.0;
            iy += incY;
        }
    } else if (beta != 1.0) {
        N iy = OFFSET(n, incY);
        for (i = 0; i < n; ++i) {
            y[iy] *= beta;
            iy += incY;
        }
    }
    
    if (alpha == 0.0)
        return;
    
    /* form  y := alpha*A*x + y */
    if ((layout == CblasRowMajor && uplo == CblasUpper)
        || (layout == CblasColMajor && uplo == CblasLower)) {
        N ix = OFFSET(n, incX);
        N iy = OFFSET(n, incY);
        for (i = 0; i < n; ++i) {
            X tmp1 = alpha * x[ix];
            X tmp2 = 0.0;
            immutable N j_min = i + 1;
            immutable N j_max = n;
            N jx = OFFSET(n, incX) + j_min * incX;
            N jy = OFFSET(n, incY) + j_min * incY;
            
            y[iy] += tmp1 * ap[TPUP(n, i, i)];
            
            for (j = j_min; j < j_max; ++j) {
                immutable X apk = ap[TPUP(n, i, j)];
                y[jy] += tmp1 * apk;
                tmp2 += apk * x[jx];
                jy += incY;
                jx += incX;
            }
            y[iy] += alpha * tmp2;
            ix += incX;
            iy += incY;
        }
    } else if ((layout == CblasRowMajor && uplo == CblasLower)
               || (layout == CblasColMajor && uplo == CblasUpper)) {
        N ix = OFFSET(n, incX);
        N iy = OFFSET(n, incY);
        for (i = 0; i < n; ++i) {
            X tmp1 = alpha * x[ix];
            X tmp2 = 0.0;
            
            immutable N j_min = 0;
            immutable N j_max = i;
            N jx = OFFSET(n, incX) + j_min * incX;
            N jy = OFFSET(n, incY) + j_min * incY;
            
            y[iy] += tmp1 * ap[TPLO(n, i, i)];
            
            for (j = j_min; j < j_max; ++j) {
                immutable X apk = ap[TPLO(n, i, j)];
                y[jy] += tmp1 * apk;
                tmp2 += apk * x[jx];
                jy += incY;
                jx += incX;
            }
            y[iy] += alpha * tmp2;
            ix += incX;
            iy += incY;
        }
    } else {
        assert(0, "unrecognized operation");
    }
}



/** 
*  @title spr blas function: Performs a rank-1 update of a symmetric packed matrix.
*
*  @description      The ?spr routines perform a matrix-vector operation defined as
*                    a:= alpha*x*x'+ A, where
*
*                    alpha is a real scalar,
*                    x is an n-element vector,
*                    A is an n-by-n symmetric matrix, supplied in packed form.
*
*  Input Parameters:
*
*  @param layout:    Specifies whether two-dimensional array storage is row-major
*                    (CblasRowMajor) or column-major (CblasColMajor).
*
*  @param uplo:      Specifies whether the upper or lower triangular part of the matrix A is
*                    supplied in the packed array ap.
*
*                    If uplo = CblasUpper , then the upper triangular part of the matrix A is
*                    supplied in the packed array ap.
*
*                    If uplo = CblasLower , then the low triangular part of the matrix A is
*                    supplied in the packed array ap.
*
*  @param n:         Specifies the order of the matrix A. The value of n must be at least zero.
*
*  @param alpha:     Specifies the scalar alpha.
*
*  @param x:         Array, size at least (1 + (n - 1)*abs(incx)) . Before entry, the
*                    incremented array x must contain the n-element vector x.
*
*  @param incx:      Specifies the increment for the elements of x.
*                    The value of incx must not be zero.
*
*  @param ap:        For layout = CblasColMajor :
*
*                    Before entry with uplo = CblasUpper , the array ap must contain the upper
*                    triangular part of the symmetric matrix packed sequentially, column-by-
*                    column, so that ap[0] contains A 1, 1 , ap[1] and ap[2] contain A 1, 2 and
*                    A 2, 2 respectively, and so on.
*
*                    Before entry with uplo = CblasLower , the array ap must contain the lower
*                    triangular part of the symmetric matrix packed sequentially, column-by-
*                    column, so that ap[0] contains A 1, 1 , ap[1] and ap[2] contain A 2, 1 and
*                    A 3, 1 respectively, and so on.
*
*                    For layout = CblasRowMajor :
*
*                    Before entry with uplo = CblasUpper , the array ap must contain the upper
*                    triangular part of the symmetric matrix packed sequentially, row-by-row,
*                    ap[0] contains A 1, 1 , ap[1] and ap[2] contain A 1, 2 and A 1, 3 respectively,
*                    and so on.
*
*                    Before entry with uplo = CblasLower , the array ap must contain the lower
*                    triangular part of the symmetric matrix packed sequentially, row-by-row, so
*                    that ap[0] contains A 1, 1 , ap[1] and ap[2] contain A 2, 1 and A 2, 2
*                    respectively, and so on.
*
*  Output Parameters:
*
*  @param            ap: With uplo = CblasUpper , overwritten by the upper triangular part of the
*                    updated matrix.
*
*                    With uplo = CblasLower , overwritten by the lower triangular part of the
*                    updated matrix.
*
*/
void spr(N, X)(in CBLAS_ORDER layout, in CBLAS_UPLO uplo, in N n, 
	     in X alpha, in X* x, in N incX, X* ap)
{
    N i, j;
    X zero = 0;
    
    if (n == 0)
        return;
    
    if (alpha == zero)
        return;
    
    if ((layout == CblasRowMajor && uplo == CblasUpper)
        || (layout == CblasColMajor && uplo == CblasLower)) {
        N ix = OFFSET(n, incX);
        for (i = 0; i < n; ++i) {
            immutable X tmp = alpha * x[ix];
            N jx = ix;
            for (j = i; j < n; ++j) {
                ap[TPUP(n, i, j)] += x[jx] * tmp;
                jx += incX;
            }
            ix += incX;
        }
    } else if ((layout == CblasRowMajor && uplo == CblasLower)
               || (layout == CblasColMajor && uplo == CblasUpper)) {
        N ix = OFFSET(n, incX);
        for (i = 0; i < n; ++i) {
            immutable X tmp = alpha * x[ix];
            N jx = OFFSET(n, incX);
            for (j = 0; j <= i; ++j) {
                ap[TPLO(n, i, j)] += x[jx] * tmp;
                jx += incX;
            }
            ix += incX;
        }
    } else {
        assert(0, "unrecognized operation");
    }
}


/** 
*  @title spr2 blas function: Performs a rank-2 update of a symmetric packed matrix.
*
*  @description      The spr2 routines perform a matrix-vector operation defined as
*                    A:= alpha*x*y'+ alpha*y*x' + A, where
*
*                    alpha is a real scalar,
*                    x is an n-element vector,
*                    A is an n-by-n symmetric matrix, supplied in packed form.
*
*  Input Parameters:
*
*  @param layout:    Specifies whether two-dimensional array storage is row-major
*                    (CblasRowMajor) or column-major (CblasColMajor).
*
*  @param uplo:      Specifies whether the upper or lower triangular part of the matrix A is
*                    supplied in the packed array ap.
*
*                    If uplo = CblasUpper, then the upper triangular part of the matrix A is
*                    supplied in the packed array ap.
*
*                    If uplo = CblasLower, then the low triangular part of the matrix A is
*                    supplied in the packed array ap.
*
*  @param n:         Specifies the order of the matrix A. The value of n must be at least zero.
*
*  @param alpha:     Specifies the scalar alpha.
*
*  @param x:         Array, size at least (1 + (n - 1)*abs(incx)) . Before entry, the
*                    incremented array x must contain the n-element vector x.
*
*  @param incx:      Specifies the increment for the elements of x.
*                    The value of incx must not be zero.
*
*  @param y:         Array, size at least (1 + (n - 1)*abs(incy)) . Before entry, the
*                    incremented array y must contain the n-element vector y.
*
*  @param incy:      Specifies the increment for the elements of y. The value of incy must not be
*                    zero.
*
*  @param ap:        For Layout = CblasColMajor:
*
*                    Before entry with uplo = CblasUpper , the array ap must contain the upper
*                    triangular part of the symmetric matrix packed sequentially, column-by-
*                    column, so that ap[0] contains A 1, 1 , ap[1] and ap[2] contain A 1, 2 and
*                    A 2, 2 respectively, and so on.
*
*                    Before entry with uplo = CblasLower , the array ap must contain the lower
*                    triangular part of the symmetric matrix packed sequentially, column-by-
*                    column, so that ap[0] contains A 1, 1 , ap[1] and ap[2] contain A 2, 1 and
*                    A 3, 1 respectively, and so on.
*
*                    For Layout = CblasRowMajor:
*
*                    Before entry with uplo = CblasUpper , the array ap must contain the upper
*                    triangular part of the symmetric matrix packed sequentially, row-by-row,
*                    ap[0] contains A 1, 1 , ap[1] and ap[2] contain A 1, 2 and A 1, 3 respectively,
*                    and so on.
*
*                    Before entry with uplo = CblasLower , the array ap must contain the lower
*                    triangular part of the symmetric matrix packed sequentially, row-by-row, so
*                    that ap[0] contains A 1, 1 , ap[1] and ap[2] contain A 2, 1 and A 2, 2
*                    respectively, and so on.
*
*  Output Parameters
*
*  @param ap:        With uplo = CblasUpper, overwritten by the upper triangular part of the
*                    updated matrix.
*
*                    With uplo = CblasLower, overwritten by the lower triangular part of the
*                    updated matrix.
*
*/
void spr2(N, X)(in CBLAS_ORDER layout, in CBLAS_UPLO uplo, in N n, in X alpha, 
	        in X* x, in N incX, in X* y, in N incY, X* ap)
{
    N i, j;
    X zero = 0;
    
    if (n == 0)
        return;
    
    if (alpha == zero)
        return;
    
    if ((layout == CblasRowMajor && uplo == CblasUpper)
        || (layout == CblasColMajor && uplo == CblasLower)) {
        N ix = OFFSET(n, incX);
        N iy = OFFSET(n, incY);
        for (i = 0; i < n; ++i) {
            immutable X tmp1 = alpha * x[ix];
            immutable X tmp2 = alpha * y[iy];
            N jx = ix;
            N jy = iy;
            for (j = i; j < n; ++j) {
                ap[TPUP(n, i, j)] += tmp1 * y[jy] + tmp2 * x[jx];
                jx += incX;
                jy += incY;
            }
            ix += incX;
            iy += incY;
        }
    } else if ((layout == CblasRowMajor && uplo == CblasLower)
               || (layout == CblasColMajor && uplo == CblasUpper)) {
        N ix = OFFSET(n, incX);
        N iy = OFFSET(n, incY);
        for (i = 0; i < n; ++i) {
            immutable X tmp1 = alpha * x[ix];
            immutable X tmp2 = alpha * y[iy];
            N jx = OFFSET(n, incX);
            N jy = OFFSET(n, incY);
            for (j = 0; j <= i; ++j) {
                ap[TPLO(n, i, j)] += tmp1 * y[jy] + tmp2 * x[jx];
                jx += incX;
                jy += incY;
            }
            ix += incX;
            iy += incY;
        }
    } else {
        assert(0, "unrecognized operation");
    }
}


/** 
*  @title syr blas function: Performs a rank-1 update of a symmetric matrix.
*
*  @description      The syr routines perform a matrix-vector operation defined as
*                    A := alpha*x*x' + A, where
*
*                    alpha is a real scalar,
*                    x is an n-element vector,
*                    A is an n-by-n symmetric matrix.
*
*  Input Parameters:
*
*  @param layout:    Specifies whether two-dimensional array storage is row-major
*                    (CblasRowMajor) or column-major (CblasColMajor).
*
*  @param uplo:      Specifies whether the upper or lower triangular part of the array a is used.
*                    If uplo = CblasUpper , then the upper triangular part of the array a is used.
*
*                    If uplo = CblasLower , then the low triangular part of the array a is used.
*
*  @param n:         Specifies the order of the matrix A. The value of n must be at least zero.
*
*  @param alpha:     Specifies the scalar alpha.
*
*  @param x:         Array, size at least (1 + (n-1)*abs(incx)) . Before entry, the
*                    incremented array x must contain the n-element vector x.
*
*  @param incx:      Specifies the increment for the elements of x.
*                    The value of incx must not be zero.
*
*  @param a:         Array, size lda*n.
*
*                    Before entry with uplo = CblasUpper , the leading n-by-n upper triangular
*                    part of the array a must contain the upper triangular part of the symmetric
*                    matrix A and the strictly lower triangular part of a is not referenced.
*                    Before entry with uplo = CblasLower , the leading n-by-n lower triangular
*                    part of the array a must contain the lower triangular part of the symmetric
*                    matrix A and the strictly upper triangular part of a is not referenced.
*
*  @param lda:       Specifies the leading dimension of a as declared in the calling
*                    (sub)program. The value of lda must be at least max(1, n).
*
*  Output Parameters:
*
*  @param a:         With uplo = CblasUpper , the upper triangular part of the array a is
*                    overwritten by the upper triangular part of the updated matrix.
*                    With uplo = CblasLower , the lower triangular part of the array a is
*                    overwritten by the lower triangular part of the updated matrix.
*
*/
void syr(N, X)(in CBLAS_ORDER layout, in CBLAS_UPLO uplo, in N n, in X alpha, in X* x, 
	           in N incX, X* a, in N lda)
{
    N i, j;
    X zero = 0;
    
    if (n == 0)
        return;
    
    if (alpha == zero)
        return;
    
    if ((layout == CblasRowMajor && uplo == CblasUpper)
        || (layout == CblasColMajor && uplo == CblasLower)) {
        N ix = OFFSET(n, incX);
        for (i = 0; i < n; ++i) {
            immutable X tmp = alpha * x[ix];
            N jx = ix;
            for (j = i; j < n; ++j) {
                a[lda * i + j] += x[jx] * tmp;
                jx += incX;
            }
            ix += incX;
        }
    } else if ((layout == CblasRowMajor && uplo == CblasLower)
               || (layout == CblasColMajor && uplo == CblasUpper)) {
        N ix = OFFSET(n, incX);
        for (i = 0; i < n; ++i) {
            immutable X tmp = alpha * x[ix];
            N jx = OFFSET(n, incX);
            for (j = 0; j <= i; ++j) {
                a[lda * i + j] += x[jx] * tmp;
                jx += incX;
            }
            ix += incX;
        }
    } else {
        assert(0, "unrecognized operation");
    }
}

/** 
*  @title syr2 blas function: Performs a rank-2 update of symmetric matrix.
*
*  @description      The syr2 routines perform a matrix-vector operation defined as
*                    A := alpha*x*y'+ alpha*y*x' + A, where
*
*                    alpha is a scalar,
*                    x and y are n-element vectors,
*                    A is an n-by-n symmetric matrix.
*
*  Input Parameters:
*
*  @param layout:    Specifies whether two-dimensional array storage is row-major
*                    (CblasRowMajor) or column-major (CblasColMajor).
*
*  @param uplo:      Specifies whether the upper or lower triangular part of the array a is used.
*
*                    If uplo = CblasUpper, then the upper triangular part of the array a is used.
*                    
*                    If uplo = CblasLower, then the low triangular part of the array a is used.
*
*  @param n:         Specifies the order of the matrix A. The value of n must be at least zero.
*
*  @param alpha:     Specifies the scalar alpha.
*
*  @param x:         Array, size at least (1 + (n - 1)*abs(incx)). Before entry, the
*                    incremented array x must contain the n-element vector x.
*
*  @param incx:      Specifies the increment for the elements of x.
*                    The value of incx must not be zero.
*
*  @param y:         Array, size at least (1 + (n - 1)*abs(incy)). Before entry, the
*                    incremented array y must contain the n-element vector y.
*
*  @param incy:      Specifies the increment for the elements of y. The value of incy must not be
*                    zero.
*
*  @param a:         Array, size lda*n.
*
*                    Before entry with uplo = CblasUpper , the leading n-by-n upper triangular
*                    part of the array a must contain the upper triangular part of the symmetric
*                    matrix and the strictly lower triangular part of a is not referenced.
*
*                    Before entry with uplo = CblasLower , the leading n-by-n lower triangular
*                    part of the array a must contain the lower triangular part of the symmetric
*                    matrix and the strictly upper triangular part of a is not referenced.
*  
*  @param lda        Specifies the leading dimension of a as declared in the calling
*                    (sub)program. The value of lda must be at least max(1, n).
*
*  Output Parameters:
*
*  @param a:         With uplo = CblasUpper , the upper triangular part of the array a is
*                    overwritten by the upper triangular part of the updated matrix.
*
*                    With uplo = CblasLower , the lower triangular part of the array a is
*                    overwritten by the lower triangular part of the updated matrix.
*
*/
void syr2(N, X)(in CBLAS_ORDER layout, in CBLAS_UPLO uplo, in N n, in X alpha, in X* x,
	            in N incX, in X* y, in N incY, X* a, in N lda)
{
    N i, j;
    X zero = 0;
    
    if (n == 0)
        return;
    
    if (alpha == zero)
        return;
    
    if ((layout == CblasRowMajor && uplo == CblasUpper)
        || (layout == CblasColMajor && uplo == CblasLower)) {
      N ix = OFFSET(n, incX);
      N iy = OFFSET(n, incY);
      for (i = 0; i < n; ++i) {
        immutable X tmp1 = alpha * x[ix];
        immutable X tmp2 = alpha * y[iy];
        N jx = ix;
        N jy = iy;
        for (j = i; j < n; ++j) {
            a[lda * i + j] += tmp1 * y[jy] + tmp2 * x[jx];
            jx += incX;
            jy += incY;
        }
        ix += incX;
        iy += incY;
      }
    } else if ((layout == CblasRowMajor && uplo == CblasLower)
               || (layout == CblasColMajor && uplo == CblasUpper)) {
      N ix = OFFSET(n, incX);
      N iy = OFFSET(n, incY);
      for (i = 0; i < n; ++i) {
        immutable X tmp1 = alpha * x[ix];
        immutable X tmp2 = alpha * y[iy];
        N jx = OFFSET(n, incX);
        N jy = OFFSET(n, incY);
        for (j = 0; j <= i; ++j) {
            a[lda * i + j] += tmp1 * y[jy] + tmp2 * x[jx];
            jx += incX;
            jy += incY;
        }
        ix += incX;
        iy += incY;
      }
    } else {
        assert(0, "unrecognized operation");
    }
}

/** 
*  @title tbmv blas function: Computes a matrix-vector product using a triangular band matrix.
*
*  @description      The ?bmv routines perform one of the matrix-vector operations defined as
*                    x := A*x , or x := A'*x , or x := conjg(A')*x, where
*
*                    x is an n-element vector,
*                    A is an n-by-n unit, or non-unit, upper or lower triangular band matrix, with (k +1) diagonals.
*
*  Input Parameters:
*
*  @param layout:    Specifies whether two-dimensional array storage is row-major
*                    (CblasRowMajor) or column-major (CblasColMajor).
*
*  @param uplo:      Specifies whether the matrix A is an upper or lower triangular matrix:
*                    uplo = CblasUpper
*                    if uplo = CblasLower, then the matrix is low triangular.
*
*  @param trans:      Specifies the operation:
*                    if trans = CblasNoTrans , then x := A*x ;
*                    if trans = CblasTrans , then x := A'*x ;
*                    if trans = CblasConjTrans , then x := conjg(A')*x
*
*  @param diag:      Specifies whether the matrix A is unit triangular:
*                    if diag = CblasUnit then the matrix is unit triangular;
*                    if diag = CblasNonUnit, then the matrix is not unit triangular.
*
*  @param n:         Specifies the order of the matrix A. The value of n must be at least zero.
*
*  @param k:         On entry with uplo = CblasUpper specifies the number of super-diagonals
*                    of the matrix A. On entry with uplo = CblasLower , k specifies the number
*                    of sub-diagonals of the matrix a.
*                    The value of k must satisfy 0 ≤k.
*
*  @param a:         Array, size lda*n .
*                    Layout = CblasColMajor :
*
*                    Before entry with uplo = CblasUpper , the leading (k + 1) by n part of
*                    the array a must contain the upper triangular band part of the matrix of
*                    coefficients, supplied column-by-column, with the leading diagonal of the
*                    matrix in row k of the array, the first super-diagonal starting at position 1 in
*                    row (k - 1) , and so on. The top left k by k triangle of the array a is not
*                    referenced. The following program segment transfers an upper triangular
*                    band matrix from conventional full matrix storage (matrix, with leading
*                    dimension ldm) to band storage (a, with leading dimension lda):
*                    for (j = 0; j < n; j++) {
*                        m = k - j;
*                        for (i = max( 0, j - k); i <= j; i++) {
*                            a[(m+i) + j*lda] = matrix[i + j*ldm];
*                        }
*                    }
*                    Before entry with uplo = CblasLower , the leading (k + 1) by n part of
*                    the array a must contain the lower triangular band part of the matrix of
*                    coefficients, supplied column-by-column, with the leading diagonal of the
*                    matrix in row 0 of the array, the first sub-diagonal starting at position 0 in
*                    row 1, and so on. The bottom right k by k triangle of the array a is not
*                    referenced. The following program segment transfers a lower triangular
*                    band matrix from conventional full matrix storage (matrix, with leading
*                    dimension ldm) to band storage (a, with leading dimension lda):
*                    for (j = 0; j < n; j++) {
*                        m = -j;
*                        for (i = j; i < min(n, j + k + 1); i++) {
*                            a[(m+i) + j*lda] = matrix[i + j*ldm];
*                        }
*                    }
*                    Note that when diag = CblasUnit , the elements of the array a
*                    corresponding to the diagonal elements of the matrix are not referenced,
*                    but are assumed to be unity.
*                    Layout = CblasRowMajor:
*                    Before entry with uplo = CblasUpper , the leading ( k + 1)-by- n part of
*                    array a must contain the upper triangular band part of the matrix of
*                    coefficients. The matrix must be supplied row-by-row, with the leading
*                    diagonal of the matrix in column 0 of the array, the first super-diagonal
*                    starting at position 0 in column 1, and so on. The bottom right k -by- k
*                    triangle of array a is not referenced.
*                    The following program segment transfers the upper triangular part of a
*                    Hermitian band matrix from row-major full matrix storage (matrix with
*                    leading dimension ldm ) to row-major band storage ( a , with leading
*                    dimension lda ):
*                    for (i = 0; i < n; i++) {
*                        m = -i;
*                        for (j = i; j < MIN(n, i+k+1); j++) {
*                            a[(m+j) + i*lda] = matrix[j + i*ldm];
*                        }
*                    }
*                    Before entry with uplo = CblasLower , the leading ( k + 1)-by- n part of
*                    array a must contain the lower triangular band part of the matrix of
*                    coefficients, supplied row-by-row, with the leading diagonal of the matrix in
*                    column k of the array, the first sub-diagonal starting at position 1 in column
*                    k -1, and so on. The top left k -by- k triangle of array a is not referenced.
*                    The following program segment transfers the lower triangular part of a
*                    Hermitian row-major band matrix from row-major full matrix storage
*                    (matrix, with leading dimension ldm ) to row-major band storage ( a , with
*                    leading dimension lda ):
*                    for (i = 0; i < n; i++) {
*                        m = k - i;
*                        for (j = max(0, i-k); j <= i; j++) {
*                            a[(m+j) + i*lda] = matrix[j + i*ldm];
*                        }
*                    }
*
*  @param lda:       Specifies the leading dimension of a as declared in the calling
*                    (sub)program. The value of lda must be at least (k + 1).
*
*  @param x:         Array, size at least (1 + (n - 1)*abs(incx)) . Before entry, the
*                    incremented array x must contain the n-element vector x.
*
*  @param incx:      Specifies the increment for the elements of x.
*                    The value of incx must not be zero.
*
*  Output Parameters:
*
*  @param x:         Overwritten with the transformed vector x.
*
*/
void tbmv(N, X)(in CBLAS_ORDER layout, in CBLAS_UPLO uplo,
                in CBLAS_TRANSPOSE transA, in CBLAS_DIAG diag,
                in N n, in N k, in X* a, in N lda, X* x, in N incX)
{
    N i, j;
    X zero = 0, one = 1;
    
    const N nonunit = (diag == CblasNonUnit);
    CBLAS_TRANSPOSE trans = (transA != CblasConjTrans) ? transA : CblasTrans;
    
    if (n == 0)
        return;
    
    if ((layout == CblasRowMajor && trans == CblasNoTrans && uplo == CblasUpper)
        || (layout == CblasColMajor && trans == CblasTrans && uplo == CblasLower)) {
        /* form  x := A*x */
        
        N ix = OFFSET(n, incX);
        for (i = 0; i < n; ++i) {
            X temp = (nonunit ? a[lda * i + 0] : 1.0) * x[ix];
            const N j_min = i + 1;
            const N j_max = min(n, i + k + 1);
            N jx = OFFSET(n, incX) + j_min * incX;
            
            for (j = j_min; j < j_max; ++j) {
                temp += x[jx] * a[lda * i + (j - i)];
                jx += incX;
            }
            
            x[ix] = temp;
            ix += incX;
        }
    } else if ((layout == CblasRowMajor && trans == CblasNoTrans && uplo == CblasLower)
               || (layout == CblasColMajor && trans == CblasTrans && uplo == CblasUpper)) {
    
        N ix = OFFSET(n, incX) + (n - 1) * incX;
        for (i = n; i > 0 && --i;) {
            X temp = (nonunit ? a[lda * i + k] : 1.0) * x[ix];
            const N j_min = (i > k ? i - k : 0);
            const N j_max = i;
            N jx = OFFSET(n, incX) + j_min * incX;
            for (j = j_min; j < j_max; ++j) {
                temp += x[jx] * a[lda * i + (k - i + j)];
                jx += incX;
            }
            x[ix] = temp;
            ix -= incX;
        }
    
    } else if ((layout == CblasRowMajor && trans == CblasTrans && uplo == CblasUpper)
               || (layout == CblasColMajor && trans == CblasNoTrans && uplo == CblasLower)) {
      /* form  x := A'*x */
        N ix = OFFSET(n, incX) + (n - 1) * incX;
        
        for (i = n; i > 0 && --i;) {
            X temp = zero;
            const N j_min = (k > i ? 0 : i - k);
            const N j_max = i;
            N jx = OFFSET(n, incX) + j_min * incX;
            for (j = j_min; j < j_max; ++j) {
                temp += x[jx] * a[lda * j + (i - j)];
                jx += incX;
            }
            if (nonunit) {
                x[ix] = temp + x[ix] * a[lda * i + 0];
            } else {
                x[ix] += temp;
            }
            ix -= incX;
        }
    } else if ((layout == CblasRowMajor && trans == CblasTrans && uplo == CblasLower)
               || (layout == CblasColMajor && trans == CblasNoTrans && uplo == CblasUpper)) {
    
        N ix = OFFSET(n, incX);
        for (i = 0; i < n; ++i) {
            X temp = zero;
            const N j_min = i + 1;
            const N j_max = min(n, i + k + 1);
            N jx = OFFSET(n, incX) + j_min * incX;
            for (j = j_min; j < j_max; ++j) {
                temp += x[jx] * a[lda * j + (k - j + i)];
                jx += incX;
            }
            if (nonunit) {
                x[ix] = temp + x[ix] * a[lda * i + k];
            } else {
                x[ix] += temp;
            }
            ix += incX;
        }
    }
}


/** 
*  @title tbsv blas function: Solves a system of linear equations whose coefficients are in a triangular band matrix.
*
*  @description      The ?tbsv routines solve one of the following systems of equations:
*                    A*x = b, or A'*x = b, or conjg(A')*x = b, where
*
*                    b and x are n-element vectors,
*                    A is an n-by-n unit, or non-unit, upper or lower triangular band matrix, with (k + 1) diagonals.
*                    The routine does not test for singularity or near-singularity.
*                    Such tests must be performed before calling this routine.
*
*  Input Parameters:
*
*  @param layout:    Specifies whether two-dimensional array storage is row-major
*                    (CblasRowMajor) or column-major (CblasColMajor).
*
*  @param uplo:      Specifies whether the matrix A is an upper or lower triangular matrix:
*                    if uplo = CblasUpper the matrix is upper triangular;
*                    if uplo = CblasLower , the matrix is low triangular.
*
*  @param trans:     Specifies the system of equations:
*                    if trans = CblasNoTrans , then A*x = b ;
*                    if trans = CblasTrans , then A'*x = b ;
*                    if trans = CblasConjTrans , then conjg(A')*x = b.
*
*  @param diag:      Specifies whether the matrix A is unit triangular:
*                    if diag = CblasUnit then the matrix is unit triangular;
*                    if diag = CblasNonUnit , then the matrix is not unit triangular.
*
*  @param n          Specifies the order of the matrix A. The value of n must be at least zero.
*
*  @param k          On entry with uplo = CblasUpper , k specifies the number of super-
*                    diagonals of the matrix A. On entry with uplo = CblasLower , k specifies
*                    the number of sub-diagonals of the matrix A.
*                    The value of k must satisfy 0 ≤k.
*                    Array, size lda*n .
*
*  @param a:         Layout = CblasColMajor :
*                    Before entry with uplo = CblasUpper , the leading (k + 1) by n part of
*                    the array a must contain the upper triangular band part of the matrix of
*                    coefficients, supplied column-by-column, with the leading diagonal of the
*                    matrix in row k of the array, the first super-diagonal starting at position 1 in
*                    row (k - 1) , and so on. The top left k by k triangle of the array a is not
*                    referenced.
*                    The following program segment transfers an upper triangular band matrix
*                    from conventional full matrix storage (matrix, with leading dimension ldm)
*                    to band storage (a, with leading dimension lda):
*                    for (j = 0; j < n; j++) {
*                        m = k - j;
*                        for (i = max( 0, j - k); i <= j; i++) {
*                            a[(m+i) + j*lda] = matrix[i + j*ldm];
*                        }
*                    }
*                    Before entry with uplo = CblasLower , the leading (k + 1) by n part of
*                    the array a must contain the lower triangular band part of the matrix of
*                    coefficients, supplied column-by-column, with the leading diagonal of the
*                    matrix in row 0 of the array, the first sub-diagonal starting at position 0 in
*                    row 1, and so on. The bottom right k by k triangle of the array a is not
*                    referenced.
*                    The following program segment transfers a lower triangular band matrix
*                    from conventional full matrix storage (matrix, with leading dimension ldm)
*                    to band storage (a, with leading dimension lda):
*                    for (j = 0; j < n; j++) {
*                        m = -j;
*                        for (i = j; i < min(n, j + k + 1); i++) {
*                            a[(m+i) + j*lda] = matrix[i + j*ldm];
*                        }
*                    }
*                    When diag = CblasUnit , the elements of the array a corresponding to the
*                    diagonal elements of the matrix are not referenced, but are assumed to be
*                    unity.
*                    Layout = CblasRowMajor:
*                    Before entry with uplo = CblasUpper , the leading ( k + 1)-by- n part of
*                    array a must contain the upper triangular band part of the matrix of
*                    coefficients. The matrix must be supplied row-by-row, with the leading
*                    diagonal of the matrix in column 0 of the array, the first super-diagonal
*                    starting at position 0 in column 1, and so on. The bottom right k -by- k
*                    triangle of array a is not referenced.
*                    The following program segment transfers the upper triangular part of a
*                    Hermitian band matrix from row-major full matrix storage (matrix with
*                    leading dimension ldm ) to row-major band storage ( a , with leading
*                    dimension lda ):
*                    for (i = 0; i < n; i++) {
*                        m = -i;
*                        for (j = i; j < MIN(n, i+k+1); j++) {
*                            a[(m+j) + i*lda] = matrix[j + i*ldm];
*                        }
*                    }
*                    Before entry with uplo = CblasLower , the leading ( k + 1)-by- n part of
*                    array a must contain the lower triangular band part of the matrix of
*                    coefficients, supplied row-by-row, with the leading diagonal of the matrix in
*                    column k of the array, the first sub-diagonal starting at position 1 in column
*                    k -1, and so on. The top left k -by- k triangle of array a is not referenced.
*                    The following program segment transfers the lower triangular part of a
*                    Hermitian row-major band matrix from row-major full matrix storage
*                    (matrix, with leading dimension ldm ) to row-major band storage ( a , with
*                    leading dimension lda ):
*                    for (i = 0; i < n; i++) {
*                        m = k - i;
*                        for (j = max(0, i-k); j <= i; j++) {
*                            a[(m+j) + i*lda] = matrix[j + i*ldm];
*                        }
*                    }
*
*  @param lda:       Specifies the leading dimension of a as declared in the calling
*                    (sub)program. The value of lda must be at least (k + 1).
*
*  @param x:         Array, size at least (1 + (n - 1)*abs(incx)) . Before entry, the
*                    incremented array x must contain the n-element right-hand side vector b.
*
*  @param incx:      Specifies the increment for the elements of x.
*                    The value of incx must not be zero.
*
*  Output Parameters:
*
*  @param x:         Overwritten with the solution vector x.
*
*/
void tbsv(N, X)(in CBLAS_ORDER layout, in CBLAS_UPLO uplo, in CBLAS_TRANSPOSE transA, in CBLAS_DIAG diag,
                in N n, in N k, in X* a, in N lda, X* x, in N incX)
{
    const N layoutIndicator = (transA == CblasConjTrans) ? -1 : 1;
    const N trans = (transA != CblasConjTrans) ? transA : CblasTrans;
    const N nonunit = (diag == CblasNonUnit);
    N i, j;
    
    if (n == 0)
        return;
    
    /* form  x := inv( A )*x */
    
    if ((layout == CblasRowMajor && trans == CblasNoTrans && uplo == CblasUpper)
        || (layout == CblasColMajor && trans == CblasTrans && uplo == CblasLower)) {
        /* backsubstitution */
        N ix = OFFSET(n, incX) + incX * (n - 1);
        for (i = n; i > 0 && i--;) {
            X tmp = x[ix];
            const N j_min = i + 1;
            const N j_max = min(n, i + k + 1);
            N jx = OFFSET(n, incX) + j_min * incX;
            for (j = j_min; j < j_max; j++) {
            	static if(!isComplex!X)
            	{
            		const X Aij = a[lda * i + (j - i)];
            	}else{
            		const X Aij = X(a[lda * i + (j - i)].re, layoutIndicator*a[lda * i + (j - i)].im);
            	}
                tmp -= Aij * x[jx];
                jx += incX;
            }
            if (nonunit) {
            	static if(!isComplex!X)
            	{
            		X aij = a[lda * i + 0];
            	}else{
            		X aij = X(a[lda * i + 0].re, layoutIndicator*a[lda * i + 0].im);
            	}
                x[ix] = tmp / aij;
            } else {
                x[ix] = tmp;
            }
          ix -= incX;
        }
    } else if ((layout == CblasRowMajor && trans == CblasNoTrans && uplo == CblasLower)
               || (layout == CblasColMajor && trans == CblasTrans && uplo == CblasUpper)) {
        /* forward substitution */
        N ix = OFFSET(n, incX);
        for (i = 0; i < n; i++) {
            X tmp = x[ix];
            const N j_min = (i > k ? i - k : 0);
            const N j_max = i;
            N jx = OFFSET(n, incX) + j_min * incX;
            for (j = j_min; j < j_max; j++) {
                static if(!isComplex!X)
            	{
            		const X Aij = a[lda * i + (k + j - i)];
            	}else{
            		const X Aij = X(a[lda * i + (k + j - i)].re, layoutIndicator*a[lda * i + (k + j - i)].im);
            	}
                tmp -= Aij * x[jx];
                jx += incX;
            }
            if (nonunit) {
            	static if(!isComplex!X)
            	{
            		X aij = a[lda * i + k];
            	}else{
            		X aij = X(a[lda * i + k].re, layoutIndicator*a[lda * i + k].im);
            	}
                x[ix] = tmp / aij;
            } else {
                x[ix] = tmp;
            }
            ix += incX;
        }
    } else if ((layout == CblasRowMajor && trans == CblasTrans && uplo == CblasUpper)
               || (layout == CblasColMajor && trans == CblasNoTrans && uplo == CblasLower)) {
        /* form  x := inv( A' )*x */
        /* forward substitution */
        N ix = OFFSET(n, incX);
        for (i = 0; i < n; i++) {
            X tmp = x[ix];
            const N j_min = (k > i ? 0 : i - k);
            const N j_max = i;
            N jx = OFFSET(n, incX) + j_min * incX;
            for (j = j_min; j < j_max; j++) {
                static if(!isComplex!X)
            	{
            		const X Aji = a[(i - j) + lda * j];
            	}else{
            		const X Aji = X(a[(i - j) + lda * j].re, layoutIndicator*a[(i - j) + lda * j].im);
            	}
                tmp -= Aji * x[jx];
                jx += incX;
            }
            if (nonunit) {
            	static if(!isComplex!X)
            	{
            		const X aij = a[lda * i];
            	}else{
            		const X aij = X(a[lda*i].re, layoutIndicator*a[lda*i].im);
            	}
                x[ix] = tmp / aij;
            } else {
                x[ix] = tmp;
            }
            ix += incX;
        }
    } else if ((layout == CblasRowMajor && trans == CblasTrans && uplo == CblasLower)
               || (layout == CblasColMajor && trans == CblasNoTrans && uplo == CblasUpper)) {
        /* backsubstitution */
        N ix = OFFSET(n, incX) + (n - 1) * incX;
        for (i = n; i > 0 && i--;) {
            X tmp = x[ix];
            const N j_min = i + 1;
            const N j_max = min(n, i + k + 1);
            N jx = OFFSET(n, incX) + j_min * incX;
            for (j = j_min; j < j_max; j++) {
                static if(!isComplex!X)
            	{
            		const X Aji = a[(k+i-j)+lda*j];
            	}else{
            		const X Aji = X(a[(k+i-j)+lda*j].re, layoutIndicator*a[(k+i-j)+lda*j].im);
            	}
                tmp -= Aji * x[jx];
                jx += incX;
            }
            if (nonunit) {
            	static if(!isComplex!X)
            	{
            		const X aij = a[k+lda*i];
            	}else{
            		const X aij = X(a[k+lda*i].re, layoutIndicator*a[k+lda*i].im);
            	}
                x[ix] = tmp/aij;
            } else {
                x[ix] = tmp;
            }
            ix -= incX;
        }
    } else {
        assert(0, "unrecognized operation");
    }
}

/** 
*  @title tpmv blas function:Computes a matrix-vector product using a triangular packed matrix.
*
*  @description      The ?tpmv routines perform one of the matrix-vector operations defined as
*                    x := A*x , or x := A'*x , or x := conjg(A')*x, where
*
*                    x is an n-element vector,
*                    A is an n-by-n unit, or non-unit, upper or lower triangular matrix, supplied in packed form.
*
*  Input Parameters:
*
*  @param layout:    Specifies whether two-dimensional array storage is row-major
*                    (CblasRowMajor) or column-major (CblasColMajor).
*
*  @param uplo:      Specifies whether the matrix A is upper or lower triangular:
*                    uplo = CblasUpper
*                    if uplo = CblasLower , then the matrix is low triangular.
*
*  @param trans:     Specifies the operation:
*                    if trans = CblasNoTrans , then x := A*x ;
*                    if trans = CblasTrans , then x := A'*x ;
*                    if trans = CblasConjTrans , then x := conjg(A')*x.
*
*  @param diag:      Specifies whether the matrix A is unit triangular:
*                    if diag = CblasUnit then the matrix is unit triangular;
*                    if diag = CblasNonUnit , then the matrix is not unit triangular.
*
*  @param n:         Specifies the order of the matrix A. The value of n must be at least zero.
*
*  @param ap:        Array, size at least ((n*(n + 1))/2) .
*                    For Layout = CblasColMajor :
*                    Before entry with uplo = CblasUpper , the array ap must contain the upper
*                    triangular matrix packed sequentially, column-by-column, so that
*                    respectively, and so on. Before entry with uplo = CblasLowerap[0]
*                    contains A 1, 1 , ap[1] and ap[2] contain A 1, 2 and A 2, 2 , the array ap must
*                    contain the lower triangular matrix packed sequentially, column-by-column,
*                    so that ap[0] contains A 1, 1 , ap[1] and ap[2] contain A 2, 1 and A 3, 1
*                    respectively, and so on. When diag = CblasUnit , the diagonal elements of
*                    a are not referenced, but are assumed to be unity.
*                    For Layout = CblasRowMajor :
*                    Before entry with uplo = CblasUpper , the array ap must contain the upper
*                    triangular matrix packed sequentially, row-by-row, ap[0] contains A 1, 1 ,
*                    ap[1] and ap[2] contain A 1, 2 and A 1, 3 respectively, and so on.
*                    Before entry with uplo = CblasLower , the array ap must contain the lower
*                    triangular matrix packed sequentially, row-by-row, so that ap[0] contains
*                    A 1, 1 , ap[1] and ap[2] contain A 2, 1 and A 2, 2 respectively, and so on.
*
*  @param x:         Array, size at least (1 + (n - 1)*abs(incx)). Before entry, the
*                    incremented array x must contain the n-element vector x.
*
*  @param incx:      Specifies the increment for the elements of x.
*                    The value of incx must not be zero.
*
*  Output Parameters
*
*  @param x:         Overwritten with the transformed vector x.
*
*/
void tpmv(N, X)(in CBLAS_ORDER order, in CBLAS_UPLO uplo, in CBLAS_TRANSPOSE transA, in CBLAS_DIAG diag,
                in N n, in X* ap, X* x, in N incX)
{
    N i, j;
    const int layoutIndicator = (transA == CblasConjTrans) ? -1 : 1;
    const int nonunit = (diag == CblasNonUnit);
    const int trans = (transA != CblasConjTrans) ? transA : CblasTrans;
    
    if (n == 0)
        return;
    
    if ((order == CblasRowMajor && trans == CblasNoTrans && uplo == CblasUpper)
        || (order == CblasColMajor && trans == CblasTrans && uplo == CblasLower)) {
        /* form  x:= A*x */
        N ix = OFFSET(n, incX);
        for (i = 0; i < n; i++) {
            X atmp = ap[TPUP(n, i, i)];
            static if(isComplex!X)
                atmp.im = layoutIndicator*atmp.im;
            X temp = (nonunit ? x[ix] * atmp : x[ix]);
            N jx = OFFSET(n, incX) + (i + 1) * incX;
            for (j = i + 1; j < n; j++) {
                atmp = ap[TPUP(n, i, j)];
                static if(isComplex!X)
                    atmp.im = layoutIndicator*atmp.im;
                temp += atmp * x[jx];
                jx += incX;
            }
            x[ix] = temp;
            ix += incX;
        }
    } else if ((order == CblasRowMajor && trans == CblasNoTrans && uplo == CblasLower)
               || (order == CblasColMajor && trans == CblasTrans && uplo == CblasUpper)) {
        N ix = OFFSET(n, incX) + (n - 1) * incX;
        for (i = n; i > 0 && i--;) {
            X atmp = ap[TPLO(n, i, i)];
            static if(isComplex!X)
                atmp.im = layoutIndicator*atmp.im;
            X temp = (nonunit ? x[ix] * atmp : x[ix]);
            N jx = OFFSET(n, incX);
            for (j = 0; j < i; j++) {
                atmp = ap[TPLO(n, i, j)];
                static if(isComplex!X)
                    atmp.im = layoutIndicator*atmp.im;
                temp += atmp * x[jx];
                jx += incX;
            }
            x[ix] = temp;
            ix -= incX;
        }
    } else if ((order == CblasRowMajor && trans == CblasTrans && uplo == CblasUpper)
               || (order == CblasColMajor && trans == CblasNoTrans && uplo == CblasLower)) {
        /* form  x := A'*x */
        N ix = OFFSET(n, incX) + (n - 1) * incX;
        for (i = n; i > 0 && i--;) {
            X atmp = ap[TPUP(n, i, i)];
            static if(isComplex!X)
                atmp.im = layoutIndicator*atmp.im;
            X temp = (nonunit ? x[ix] * atmp : x[ix]);
            N jx = OFFSET(n, incX);
            for (j = 0; j < i; j++) {
                atmp = ap[TPUP(n, j, i)];
                static if(isComplex!X)
                    atmp.im = layoutIndicator*atmp.im;
                temp += atmp * x[jx];
                jx += incX;
            }
            x[ix] = temp;
            ix -= incX;
        }
    } else if ((order == CblasRowMajor && trans == CblasTrans && uplo == CblasLower)
               || (order == CblasColMajor && trans == CblasNoTrans && uplo == CblasUpper)) {
        N ix = OFFSET(n, incX);
        for (i = 0; i < n; i++) {
            X atmp = ap[TPLO(n, i, i)];
            static if(isComplex!X)
                atmp.im = layoutIndicator*atmp.im;
            X temp = (nonunit ? x[ix] * atmp : x[ix]);
            N jx = OFFSET(n, incX) + (i + 1) * incX;
            for (j = i + 1; j < n; j++) {
                atmp = ap[TPLO(n, j, i)];
                static if(isComplex!X)
                    atmp.im = layoutIndicator*atmp.im;
                temp += atmp * x[jx];
                jx += incX;
            }
            x[ix] = temp;
            ix += incX;
        }
    } else {
      assert(0, "unrecognized operation");
    }
}


/** 
*  @title cblas_tpsv Solves a system of linear equations whose coefficients are in a triangular packed matrix.
*
*  @description      The tpsv routines solve one of the following systems of equations
*                    A*x = b , or A'*x = b , or conjg(A')*x = b, where
*
*                    b and x are n-element vectors,
*                    A is an n-by-n unit, or non-unit, upper or lower triangular matrix, supplied in packed form.
*                    This routine does not test for singularity or near-singularity.
*                    Such tests must be performed before calling this routine.
*
*  Input Parameters:
*
*  @param layout:    Specifies whether two-dimensional array storage is row-major
*                    (CblasRowMajor) or column-major (CblasColMajor).
*
*  @param uplo:      Specifies whether the matrix A is upper or lower triangular:
*                    uplo = CblasUpper
*                    if uplo = CblasLower , then the matrix is low triangular.
*
*  @param trans:     Specifies the system of equations:
*                    if trans = CblasNoTrans , then A*x = b ;
*                    if trans = CblasTrans , then A'*x = b ;
*                    if trans = CblasConjTrans , then conjg(A')*x = b .
*
*  @param diag:      Specifies whether the matrix A is unit triangular:
*                    if diag = CblasUnit then the matrix is unit triangular;
*                    if diag = CblasNonUnit , then the matrix is not unit triangular.
*
*  @param n:         Specifies the order of the matrix A. The value of n must be at least zero.
*
*  @param ap:        Array, size at least ((n*(n + 1))/2) .
*                    For Layout = CblasColMajor :
*                    Before entry with uplo = CblasUpper , the array ap must contain the upper
*                    triangular part of the triangular matrix packed sequentially, column-by-
*                    column, so that ap[0] contains A 1, 1 , ap[1] and ap[2] contain A 1, 2 and
*                    A 2, 2 respectively, and so on.
*                    Before entry with uplo = CblasLower , the array ap must contain the lower
*                    triangular part of the triangular matrix packed sequentially, column-by-
*                    column, so that ap[0] contains A 1, 1 , ap[1] and ap[2] contain A 2, 1 and
*                    A 3, 1 respectively, and so on.
*                    For Layout = CblasRowMajor :
*                    Before entry with uplo = CblasUpper , the array ap must contain the upper
*                    triangular part of the triangular matrix packed sequentially, row-by-row,
*                    ap[0] contains A 1, 1 , ap[1] and ap[2] contain A 1, 2 and A 1, 3 respectively,
*                    and so on. Before entry with uplo = CblasLower , the array ap must
*                    contain the lower triangular part of the triangular matrix packed
*                    sequentially, row-by-row, so that ap[0] contains A 1, 1 , ap[1] and ap[2]
*                    contain A 2, 1 and A 2, 2 respectively, and so on.
*                    When diag = CblasUnit , the diagonal elements of a are not referenced,
*                    but are assumed to be unity.
*
*  @param x:         Array, size at least (1 + (n - 1)*abs(incx)) . Before entry, the
*                    incremented array x must contain the n-element right-hand side vector b.
*
*  @param incx:      Specifies the increment for the elements of x.
*                    The value of incx must not be zero.
*
*  Output Parameters
*
*  @param x:         Overwritten with the solution vector x.
*
*/
void tpsv(N, X)(in CBLAS_ORDER order, in CBLAS_UPLO uplo,
                in CBLAS_TRANSPOSE transA, in CBLAS_DIAG diag,
                in N n, in X* ap, X* x, in N incX)
{
    N i, j;
    
    const N layoutIndicator = (transA == CblasConjTrans) ? -1 : 1;
    const N nonunit = (diag == CblasNonUnit);
    const N trans = (transA != CblasConjTrans) ? transA : CblasTrans;
    
    if (n == 0)
        return;
    
    /* form  x := inv( A )*x */
    
    if ((order == CblasRowMajor && trans == CblasNoTrans && uplo == CblasUpper)
        || (order == CblasColMajor && trans == CblasTrans && uplo == CblasLower)) {
        /* backsubstitution */
        N ix = OFFSET(n, incX) + incX * (n - 1);
        if (nonunit) {
        	X atmp = ap[TPUP(n, (n - 1), (n - 1))];
        	static if(isComplex!X)
        	    atmp.im = layoutIndicator*atmp.im;
            x[ix] = x[ix] / atmp;
        }
        ix -= incX;
        for (i = n - 1; i > 0 && i--;) {
            X tmp = x[ix];
            N jx = ix + incX;
            for (j = i + 1; j < n; j++) {
            	static if(isComplex!X)
            	    const X Aij = X(ap[TPUP(n, i, j)].re, layoutIndicator*ap[TPUP(n, i, j)].im);
            	else
            	    const X Aij = ap[TPUP(n, i, j)];
                tmp -= Aij * x[jx];
                jx += incX;
            }
            if (nonunit) {
            	X atmp = ap[TPUP(n, i, i)];
        	    static if(isComplex!X)
        	        atmp.im = layoutIndicator*atmp.im;
                x[ix] = tmp/atmp;
            } else {
                x[ix] = tmp;
            }
            ix -= incX;
        }
    } else if ((order == CblasRowMajor && trans == CblasNoTrans && uplo == CblasLower)
               || (order == CblasColMajor && trans == CblasTrans && uplo == CblasUpper)) {
    
        /* forward substitution */
        N ix = OFFSET(n, incX);
        if (nonunit) {
        	X atmp = ap[TPLO(n, 0, 0)];
        	    static if(isComplex!X)
        	        atmp.im = layoutIndicator*atmp.im;
            x[ix] = x[ix]/atmp;
        }
        ix += incX;
        for (i = 1; i < n; i++) {
            X tmp = x[ix];
            N jx = OFFSET(n, incX);
            for (j = 0; j < i; j++) {
            	static if(isComplex!X)
            	    const X Aij = X(ap[TPUP(n, i, j)].re, layoutIndicator*ap[TPUP(n, i, j)].im);
            	else
            	    const X Aij = ap[TPUP(n, i, j)];
                tmp -= Aij * x[jx];
                jx += incX;
            }
            if (nonunit) {
            	X atmp = ap[TPLO(n, i, j)];
        	    static if(isComplex!X)
        	        atmp.im = layoutIndicator*atmp.im;
                x[ix] = tmp/atmp;
            } else {
                x[ix] = tmp;
            }
            ix += incX;
        }
    } else if ((order == CblasRowMajor && trans == CblasTrans && uplo == CblasUpper)
               || (order == CblasColMajor && trans == CblasNoTrans && uplo == CblasLower)) {
    
        /* form  x := inv( A' )*x */
        
        /* forward substitution */
        N ix = OFFSET(n, incX);
        if (nonunit) {
        	X atmp = ap[TPLO(n, 0, 0)];
        	    static if(isComplex!X)
        	        atmp.im = layoutIndicator*atmp.im;
            x[ix] = x[ix]/atmp;
        }
        ix += incX;
        for (i = 1; i < n; i++) {
            X tmp = x[ix];
            N jx = OFFSET(n, incX);
            for (j = 0; j < i; j++) {
            	static if(isComplex!X)
            	    const X Aji = X(ap[TPUP(n, j, i)].re, layoutIndicator*ap[TPUP(n, j, i)].im);
            	else
            	    const X Aji = ap[TPUP(n, j, i)];
                tmp -= Aji * x[jx];
                jx += incX;
            }
            if (nonunit) {
            	X atmp = ap[TPLO(n, i, i)];
        	    static if(isComplex!X)
        	        atmp.im = layoutIndicator*atmp.im;
                x[ix] = tmp/atmp;
            } else {
                x[ix] = tmp;
            }
            ix += incX;
        }
    } else if ((order == CblasRowMajor && trans == CblasTrans && uplo == CblasLower)
               || (order == CblasColMajor && trans == CblasNoTrans && uplo == CblasUpper)) {
    
        /* backsubstitution */
        N ix = OFFSET(n, incX) + (n - 1) * incX;
        if (nonunit) {
        	X atmp = ap[TPLO(n, (n - 1), (n - 1))];
        	    static if(isComplex!X)
        	        atmp.im = layoutIndicator*atmp.im;
            x[ix] = x[ix]/atmp;
        }
        ix -= incX;
        for (i = n - 1; i > 0 && i--;) {
            X tmp = x[ix];
            N jx = ix + incX;
            for (j = i + 1; j < n; j++) {
            	static if(isComplex!X)
            	    const X Aji = X(ap[TPUP(n, j, i)].re, layoutIndicator*ap[TPUP(n, j, i)].im);
            	else
            	    const X Aji = ap[TPUP(n, j, i)];
                tmp -= Aji * x[jx];
                jx += incX;
            }
            if (nonunit) {
            	X atmp = ap[TPLO(n, i, i)];
        	    static if(isComplex!X)
        	        atmp.im = layoutIndicator*atmp.im;
                x[ix] = tmp/atmp;
            } else {
                x[ix] = tmp;
            }
            ix -= incX;
        }
    } else {
        assert(0, "unrecognized operation");
    }
}


/** 
*  @title trsv Solves a system of linear equations whose coefficients are in a triangular matrix.
*
*  @description      The trsv routines solve one of the systems of equations:
*                    A*x = b , or A'*x = b , or conjg(A')*x = b, where
*
*                    b and x are n-element vectors,
*                    A is an n-by-n unit, or non-unit, upper or lower triangular matrix.
*                    The routine does not test for singularity or near-singularity.
*                    Such tests must be performed before calling this routine.
*
*  Input Parameters:
*
*  @param layout:    Specifies whether two-dimensional array storage is row-major
*                    (CblasRowMajor) or column-major (CblasColMajor).
*
*  @param uplo:      Specifies whether the matrix A is upper or lower triangular:
*                    uplo = CblasUpper
*                    if uplo = CblasLower , then the matrix is low triangular.
*
*  @param trans:     Specifies the systems of equations:
*                    if trans = CblasNoTrans , then A*x = b ;
*                    if trans = CblasTrans , then A'*x = b ;
*                    if trans = CblasConjTrans , then oconjg(A')*x = b .
*
*  @param diag:      Specifies whether the matrix A is unit triangular:
*                    if diag = CblasUnit then the matrix is unit triangular;
*                    if diag = CblasNonUnit , then the matrix is not unit triangular.
*
*  @param n:         Specifies the order of the matrix A. The value of n must be at least zero.
*
*  @param a:         Array, size lda*n . Before entry with uplo = CblasUpper , the leading n-
*                    by-n upper triangular part of the array a must contain the upper triangular
*                    matrix and the strictly lower triangular part of a is not referenced. Before
*                    entry with uplo = CblasLower , the leading n-by-n lower triangular part of
*                    the array a must contain the lower triangular matrix and the strictly upper
*                    triangular part of a is not referenced.
*
*                    When diag = CblasUnit , the diagonal elements of a are not referenced
*                    either, but are assumed to be unity.
*
*  @param lda:       Specifies the leading dimension of a as declared in the calling
*                    (sub) program. The value of lda must be at least max(1, n) .
*
*  @param x:         Array, size at least (1 + (n - 1)*abs(incx)) . Before entry, the
*                    incremented array x must contain the n-element right-hand side vector b.
*
*  @param incx:      Specifies the increment for the elements of x.
*                    The value of incx must not be zero.
*
*  Output Parameters:
*
*  @param x:         Overwritten with the solution vector x.
*
*/
void trsv(N, X)(in CBLAS_ORDER order, in CBLAS_UPLO uplo,
                in CBLAS_TRANSPOSE transA, in CBLAS_DIAG diag,
                in N n, in X* a, in N lda, X* x, in N incX)
{
	const N layoutIndicator = (transA == CblasConjTrans) ? -1 : 1;
	const N trans = (transA != CblasConjTrans) ? transA : CblasTrans;
    const N nonunit = (diag == CblasNonUnit);
    N ix, jx;
    N i, j;
    
    if (n == 0)
        return;
    
    /* form  x := inv( A )*x */
    
    if ((order == CblasRowMajor && trans == CblasNoTrans && uplo == CblasUpper)
        || (order == CblasColMajor && trans == CblasTrans && uplo == CblasLower)) {
        /* backsubstitution */
        ix = OFFSET(n, incX) + incX * (n - 1);
        if (nonunit) {
        	X atmp = a[lda * (n - 1) + (n - 1)];
        	static if(isComplex!X)
        	        atmp.im = layoutIndicator*atmp.im;
            x[ix] = x[ix]/atmp;
        }
        ix -= incX;
        for (i = n - 1; i > 0 && i--;) {
            X tmp = x[ix];
            jx = ix + incX;
            for (j = i + 1; j < n; j++) {
            	static if(isComplex!X)
            	    const X Aij = X(a[lda*i + j].re, layoutIndicator*a[lda*i + j].im);
            	else
                    const X Aij = a[lda * i + j];
                tmp -= Aij * x[jx];
                jx += incX;
            }
            if (nonunit) {
            	static if(isComplex!X)
            	    const X aij = X(a[lda*i + i].re, layoutIndicator*a[lda*i + i].im);
            	else
                    const X aij = a[lda*i + i];
                x[ix] = tmp/aij;
            } else {
                x[ix] = tmp;
            }
            ix -= incX;
        }
    } else if ((order == CblasRowMajor && trans == CblasNoTrans && uplo == CblasLower)
               || (order == CblasColMajor && trans == CblasTrans && uplo == CblasUpper)) {
        
        /* forward substitution */
        ix = OFFSET(n, incX);
        if (nonunit) {
        	static if(isComplex!X)
            	    const X aij = X(a[0].re, layoutIndicator*a[0].im);
            	else
                    const X aij = a[0];
            x[ix] = x[ix]/aij;
        }
        ix += incX;
        for (i = 1; i < n; i++) {
            X tmp = x[ix];
            jx = OFFSET(n, incX);
            for (j = 0; j < i; j++) {
            	static if(isComplex!X)
            	    const X Aij = X(a[lda*i + j].re, layoutIndicator*a[lda*i + j].im);
            	else
                    const X Aij = a[lda*i + j];
                tmp -= Aij * x[jx];
                jx += incX;
            }
            if (nonunit) {
            	static if(isComplex!X)
            	    const X aij = X(a[lda*i + i].re, layoutIndicator*a[lda*i + i].im);
            	else
                    const X aij = a[lda*i + i];
                x[ix] = tmp/aij;
            } else {
                x[ix] = tmp;
            }
            ix += incX;
        }
    } else if ((order == CblasRowMajor && trans == CblasTrans && uplo == CblasUpper)
               || (order == CblasColMajor && trans == CblasNoTrans && uplo == CblasLower)) {
        
        /* form  x := inv( A' )*x */
        
        /* forward substitution */
        ix = OFFSET(n, incX);
        if (nonunit) {
        	static if(isComplex!X)
            	    const X aij = X(a[0].re, layoutIndicator*a[0].im);
            	else
                    const X aij = a[0];
            x[ix] = x[ix]/aij;
        }
        ix += incX;
        for (i = 1; i < n; i++) {
            X tmp = x[ix];
            jx = OFFSET(n, incX);
            for (j = 0; j < i; j++) {
            	static if(isComplex!X)
            	    const X Aji = X(a[lda*j + i].re, layoutIndicator*a[lda*j + i].im);
            	else
                    const X Aji = a[lda*j + i];
                tmp -= Aji * x[jx];
                jx += incX;
            }
            if (nonunit) {
            	static if(isComplex!X)
            	    const X aij = X(a[lda*i + i].re, layoutIndicator*a[lda*i + i].im);
            	else
                    const X aij = a[lda*i + i];
                x[ix] = tmp/aij;
            } else {
                x[ix] = tmp;
            }
            ix += incX;
        }
    } else if ((order == CblasRowMajor && trans == CblasTrans && uplo == CblasLower)
               || (order == CblasColMajor && trans == CblasNoTrans && uplo == CblasUpper)) {
        
        /* backsubstitution */
        ix = OFFSET(n, incX) + (n - 1) * incX;
        if (nonunit) {
        	static if(isComplex!X)
            	    const X aij = X(a[lda * (n - 1) + (n - 1)].re, layoutIndicator*a[lda * (n - 1) + (n - 1)].im);
            	else
                    const X aij = a[lda * (n - 1) + (n - 1)];
            x[ix] = x[ix] / aij;
        }
        ix -= incX;
        for (i = n - 1; i > 0 && i--;) {
            X tmp = x[ix];
            jx = ix + incX;
            for (j = i + 1; j < n; j++) {
            	static if(isComplex!X)
            	    const X Aji = X(a[lda*j + i].re, layoutIndicator*a[lda*j + i].im);
            	else
                    const X Aji = a[lda*j + i];
                tmp -= Aji * x[jx];
                jx += incX;
            }
            if (nonunit) {
            	static if(isComplex!X)
            	    const X aij = X(a[lda*i + i].re, layoutIndicator*a[lda*i + i].im);
            	else
                    const X aij = a[lda*i + i];
                x[ix] = tmp/aij;
            } else {
                x[ix] = tmp;
            }
            ix -= incX;
        }
    } else {
        assert(0, "unrecognized operation");
    }
}




