/**
*
*  @title: Level 1 BLAS algorithms for D
*  @author: Chibisi Chima-Okereke
*  @date: 2017-02-24
*  
*/

// dmd dblas.d -unittest -L-lopenblas -L-lpthread && ./dblas

module dblas.l1;
import std.complex: Complex, complex, conj;
import std.math: abs, fabs, sqrt, sgn, pow, approxEqual;
import std.traits : isIntegral, isUnsigned, isFloatingPoint;
// import std.stdio : writeln;
import std.random : Random, uniform, unpredictableSeed;


/* CBLAS functions functions for testing */
extern (C){
	void cblas_drot(int N, double *X, int incX, double *Y, int incY, double c, double  s);
	void cblas_drotg(double *a, double *b, double *c, double *s);
	void cblas_drotmg(double *d1, double *d2, double *b1, double b2, double *P);
	void cblas_drotm(int N, double *X, int incX, double *Y, int incY, double *P);
	void cblas_dswap(int n, double *x, int incx, double *y, int incy);
	void cblas_dscal(int N, double alpha, double *X, int incX);
	void cblas_dcopy(int n, double *x, int incx, double *y, int incy);
	void cblas_daxpy(int n, double alpha, double *x, int incx, double *y, int incy);
	double cblas_ddot(int n, double *x, int incx, double *y, int incy);
	double cblas_dsdot (int n, float *x, int incx, float *y, int incy);
	double cblas_dnrm2 (int N, double *X, int incX);
	double cblas_dasum (int n, double *x, int incx);
	int cblas_idamax(int n, double *x, int incx);
}

pragma(inline, true):
/* For generating random numbers for testing ... */
T[] runif(T)(int n, T x1 = 0.0, T x2 = 1.0)
{
	auto gen = Random(unpredictableSeed);
	T[] output;
	while(n){
		output ~= uniform(x1, x2, gen);
		--n;
	}
	return output;
}

T OFFSET(T)(in T N, in T incX)
{
	return incX > 0 ?  0 : ((N - 1) * -incX);
}

// isComplex taken from https://wiki.dlang.org/Is_expression
template isComplex(T)
{
	static if(is(T == Complex!CT, CT)){
		enum isComplex = true;
	}else{
		enum isComplex = false;
	}
}

/* Implement absolute values for complex numbers */
/* Square of absolute value of complex number */
T abs2(X: Complex!T, T = typeof(X.re))(X x)
{
	return (x.re*x.re + x.im*x.im);
}
/* Absolute value of complex number */
T abs(X: Complex!T, T = typeof(X.re))(X x)
{
	return sqrt(abs2(x));
}

/* alias for blas cabs1 */
alias abs cabs1;

/* Sigum function that allows complex numbers to be evaluated */
X csgn(X)(X x)
{
	return sgn(x);
}
T csgn(X: Complex!T, T = typeof(X.re))(X x)
{
	if(x.re != 0)
		return sgn(x.re);
	else
		return sgn(x.im);
}

T min(T)(T a, T b){
	return a > b ? a : b;
}
T max(T)(T a, T b){
	return a < b ? a : b;
}


/** TODO:
*   1. Unit tests and better examples. extern(C) equivalent functions from the blas library 
*      and make sure that the outputs are the same :-)
*/


/**
*  @title scal blas function: Computes the product of a vector by a scalar
*  @description Computes the product of a vector by a scalar x = a*x
*               where a is a scala and x is an n-element vector
*  @param n The number of elements in vector x
*  @param a The scala a
*  @param x Pointer to array size at least (1 + (n + 1)*abs(incx))
*  @param incx Specifies the increment for the elements of x
*  @return void but the input array x is now multiplied by a
*  @example
*           double[] x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
*           scal(x.length, 3.0, x.ptr, 1);
*           writeln(x);
*/
void scal(N, X)(in N n, in X a, X* x, in N incX)
{
	N ix = OFFSET(n, incX);
	for(int i = 0; i < n; ++i)
	{
		x[ix] *= a;
		ix += incX;
	}
}


unittest
{
	double[] x1 = runif(10, -10.0, 10.0);
	double[] x2 = x1;
	double a = runif(1, -10.0, 10.0)[0];
	int l1 = cast(int)x1.length;

	scal(x1.length, a, x1.ptr, 1);
	cblas_dscal(l1, a, x2.ptr, 1);

	assert(approxEqual(x1, x2), "scal test: x output failed.");
}


/**
*  @title asum blas function: Computes the sum of magnitues of the vector elements.
*  @description Computes the sum of magnitudes of elements of a real vector or the sum
*               of magnitudes of the real and imaginary parts of elements of a complex 
*               vector.
*  @param n The number of elements in vector x
*  @param x Pointer to array size at least (1 + (n + 1)*abs(incx))
*  @param incx Specifies the increment for the elements of x
*  @return the sum of x where the type is the of an element of X
*  @example
*           double[] x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
*           asum(x.length, x.ptr);
*           writeln(x);
*/
X asum(N, X)(N n, X* x, N incx)
if(isIntegral!(N))
{
	X res = fabs(*x); x += incx; --n;
	while(n)
	{
		res += fabs(*x);
		x += incx;
		--n;
	}
	return res;
}

V asum(N, X: Complex!V, V = typeof(X.re))(N n, X* x, N incx)
if(isIntegral!(N))
{
	V res = (fabs((*x).re) + fabs((*x).im)); x += incx; --n;
	while(n)
	{
		res += (fabs((*x).re) + fabs((*x).im));
		x += incx;
		--n;
	}
	return res;
}


/**
*  @title axpy blas function: Computes a vector-scalar product and assigns the result to a vector
*  @description The axpy routine performs a vector-vector operation defined as y := a*x + y where
*               a is a scalar, and x and y are vectors each with number of elements that equals n.
*               Input parameters
*  @param n The number of elements in vectors x and y
*  @param a Specifies the scalar a
*  @param x Pointer to array size at least (1 + (n + 1)*abs(incx))
*  @param y Pointer to array size at least (1 + (n + 1)*abs(incy))
*  @param incx Specifies the increment for the elements of x
*  @param incy Specifies the increment for the elements of y
*  @return void but the input array y is now multiplied by a
*  @example
*           double[] x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
*           double[] y = [7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
*           axpy(y.length, 3.0, x.ptr, 1, y.ptr, 1);
*           writeln(y);
*/
void axpy(N, A, X, Y)(N n, A a, X* x, N incx, Y* y, N incy)
if(isIntegral!(N))
{
	while(n){
		*y = a*(*x) + (*y);
		x += incx;
		y += incy;
		--n;
	}
}

unittest
{
	double[] x1 = runif(20, -10.0, 10.0);
	double[] x2 = x1;

	double[] y1 = runif(20, -10.0, 10.0);
	double[] y2 = y1;

	double a = runif(1, -10.0, 10.0)[0];
	int l1 = cast(int)x1.length;
	
	axpy(x1.length, a, x1.ptr, 1, y1.ptr, 1);
	cblas_daxpy(l1, a, x2.ptr, 1, y2.ptr, 1);

	assert(approxEqual(x1, x2), "axpy test: x output failed.");
	assert(approxEqual(y1, y2), "axpy test: y output failed.");
}

/**
*  @title copy blas function: Copies vector to another vector
*  @description The copy routine performs a vector-vector operation defined as
*               y = x, where x and y are vectors
*  @param n The number of elements in vectors x and y
*  @param x Pointer to array size at least (1 + (n + 1)*abs(incx))
*  @param y Pointer to array size at least (1 + (n + 1)*abs(incy))
*  @param incx Specifies the increment for the elements of x
*  @param incy Specifies the increment for the elements of y
*  @return void but the input array y now contains a copy of the 
*               elements of x
*  @example
*           double[] x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
*           double[] y = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
*           axpy(y.length, x.ptr, 1, y.ptr, 1);
*           writeln(y);
*/
void copy(N, X)(N n, X* x, N incx, X* y, N incy)
if(isIntegral!(N))
{
	while(n){
		*y = *x;
		x += incx;
		y += incy;
		--n;
	}
}

unittest
{
	double[] x1 = runif(10, -10.0, 10.0);
	double[] x2 = x1;

	double[] y1 = x1; y1[] = 0;
	double[] y2 = y1;

	int l1 = cast(int)x1.length;
	
	copy(x1.length, x1.ptr, 1, y1.ptr, 1);
	cblas_dcopy(l1, x2.ptr, 1, y2.ptr, 1);

	assert(approxEqual(x1, x2), "copy test: x output failed.");
	assert(approxEqual(y1, y2), "copy test: y output failed.");
}

/**
*  @title dot blas function: Computes a vector-vector dot product
*  @description The dot blas routine computes the inner product of two vectors
*               the accumulation of the intermediate results is the same type as
*               the elements of the inputs
*  @param n The number of elements in vectors x and y
*  @param x Pointer to array size at least (1 + (n + 1)*abs(incx))
*  @param y Pointer to array size at least (1 + (n + 1)*abs(incy))
*  @param incx Specifies the increment for the elements of x
*  @param incy Specifies the increment for the elements of y
*  @return returns dot product of vectors x and y
*  @example
*           double[] x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
*           double[] y = [7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
*           writeln(dot(y.length, x.ptr, 1, y.ptr, 1));
*/
X dot(N, X)(N n, X* x, in N incx, X* y, in N incy)
if(isIntegral!(N))
{
	X res = (*x)*(*y); --n;
	x += incx; y += incy;
	while(n){
		res += (*x)*(*y);
		x += incx;
		y += incy;
		--n;
	}
	return res;
}
// Overload for intial total res
X dot(N, X)(N n, X res, X* x, in N incx, X* y, in N incy)
if(isIntegral!(N))
{
	while(n){
		res += (*x)*(*y);
		x += incx;
		y += incy;
		--n;
	}
	return res;
}

// Alias for the complex case
alias dot dotu;

// Complex conjugate case
X dotc(N, X: Complex!V, V = typeof(X.re))(N n, X* x, in N incx, X* y, in N incy)
if(isIntegral!(N))
{
	X res = conj(*x) * (*y); --n;
	x += incx; y += incy;
	while(n){
		res += conj(*x) * (*y);
		x += incx;
		y += incy;
		--n;
	}
	return res;
}

unittest
{
	double[] x1 = runif(10, -10.0, 10.0);
	double[] y1 = runif(10, -10.0, 10.0);
	double[] x2 = x1, y2 = y1;
	int l2 = cast(int)x2.length;

	double out1 = dot(x1.length, x1.ptr, 1, y1.ptr, 1);
	double out2 = cblas_ddot(l2, x2.ptr, 1, y2.ptr, 1);
	
	assert(approxEqual(out1, out2), "dot test: x output failed.");
}


/**
*  @title nrm2 blas function: Computes the Euclidean norm of a vector
*  @description The nrm2 blas routine computes a vector reduction operation
*               defined as res = ||x||, where x is a vector and res is a value
*               containing the Euclidean norm of the elements of x
*  @param n The number of elements in vectors x
*  @param x Pointer to array size at least (1 + (n + 1)*abs(incx))
*  @param incx Specifies the increment for the elements of x
*  @return The Euclidean norm of the vector x
*  @example
*           double[] x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
*           nrm2(x.length, x.ptr, 1);
*           writeln(y);
*/
/*X nrm2(N, X)(N n, X* x, N incx)
if(isIntegral!N)
{
	X res = (*x) * (*x);
	x += incx; n--;
	while(n){
		res += ((*x) * (*x));
		x += incx; n--;
	}
	return sqrt(res);
}*/
/* Converted from GSL implementation */
X nrm2(N, X)(N n, X* x, N incx)
{
  X scale = 0.0;
  X ssq = 1.0;
  N i;
  N ix = 0;

  if (n <= 0 || incx <= 0) {
    return 0;
  } else if (n == 1) {
    return fabs(x[0]);
  }

  for(i = 0; i < n; i++) {
    const X z = x[ix];

    if (z != 0.0) {
      const X ax = fabs(z);

      if (scale < ax) {
        ssq = 1.0 + ssq * (scale / ax) * (scale / ax);
        scale = ax;
      } else {
        ssq += (ax / scale) * (ax / scale);
      }
    }

    ix += incx;
  }

  return scale * sqrt(ssq);
}
// Complex version
/*V nrm2(N, X: Complex!V, V = typeof(X.re))(N n, X* x, N incx)
if(isIntegral!N)
{
	//V res = ((*x) * conj(*x)).re;
	V res = abs2(*x);
	x += incx; n--;
	while(n){
		//res += ((*x) * conj(*x)).re;
		res += abs2(*x);
		x += incx; n--;
	}
	return sqrt(res);
}*/
/* Converted from GSL implementation */
V nrm2(N, X: Complex!V, V = typeof(X.re))(N n, X* x, N incx)
{
  V scale = 0.0;
  V ssq = 1.0;
  N i;
  N ix = 0;

  if (n == 0 || incx < 1) {
    return 0;
  }

  for (i = 0; i < n; i++) {
    const V z = cast(const V)x[ix].re;
    const V y = cast(const V)x[ix].im;

    if (z != 0.0) {
      const V ax = fabs(z);

      if (scale < ax) {
        ssq = 1.0 + ssq * (scale / ax) * (scale / ax);
        scale = ax;
      } else {
        ssq += (ax / scale) * (ax / scale);
      }
    }

    if (y != 0.0) {
      const V ay = fabs(y);

      if (scale < ay) {
        ssq = 1.0 + ssq * (scale / ay) * (scale / ay);
        scale = ay;
      } else {
        ssq += (ay / scale) * (ay / scale);
      }
    }

    ix += incx;
  }

  return scale * sqrt(ssq);
}

// X nrm2(N, X)(N n, X* x, N incx)
// double cblas_dnrm2 (int N, double *X, int incX);
unittest
{
	double[] x1 = runif(10, -10.0, 10.0);
	double[] x2 = x1;
	int l2 = cast(int)x2.length;

	double out1 = nrm2(x1.length, x1.ptr, 1);
	double out2 = cblas_dnrm2(l2, x2.ptr, 1);

	assert(approxEqual(out1, out2), "nrm2 test: x output failed.");
}


/**
*  @title rot blas function: Performs rotation of points in the plane
*  @description Given two complex vectors x and y, each vector of these vectors is replaced as follows
*               x_i = c*x_i + s*y_i
*               y_i = c*y_i - s*x_i
*  @param n The number of elements in vectors x and y
*  @param x Pointer to array size at least (1 + (n + 1)*abs(incx))
*  @param y Pointer to array size at least (1 + (n + 1)*abs(incy))
*  @param incx Specifies the increment for the elements of x
*  @param incy Specifies the increment for the elements of y
*  @return void but x and y are now altered as described previously
*  @example
*           double[] x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
*           double[] y = [7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
*           rot(x.length, x.ptr, 1, y.ptr, 1, 2.0, 3.0);
*           writeln(x);
*           writeln(y);
*/
void rot(N, X, Y)(N n, X* x, in N incx, X* y, in N incy, in Y c, in Y s)
if(isIntegral!N && isFloatingPoint!Y)
{
	X temp;
	while(n){
		temp = c*(*x) + s*(*y);
		*y = c*(*y) - s*(*x);
		*x = temp;
		x += incx;
		y += incy;
		--n;
	}
}

unittest
{
	double[] x1 = runif(10, -10.0, 10.0);
	double[] y1 = runif(10, -10.0, 10.0);
	double[] x2 = x1, y2 = y1;
	double[] params = runif(2, 1.0, 5.0);
	rot(x1.length, x1.ptr, 1, y1.ptr, 1, params[0], params[1]);
	int l1 = cast(int)x2.length;
	cblas_drot(l1, x2.ptr, 1, y2.ptr, 1, params[0], params[1]);
	assert(approxEqual(x1, x2), "rot test: x output failed.");
	assert(approxEqual(y1, y2), "rot test: y output failed.");
}

/**
*  @title rotg blas function: Computes the parameters for a Givens rotation
*  @description Given the cartesian coordinates (a, b) of a point, these routines
*               return the parameters c, s, r, and z associated with the Givens 
*               rotation. The parameters c and s define a unitary matrix such that:
*
*               |c  s| |a|   |r|
*               |    | | |   | |
*               |-s c| |b| = |0|
*
*               The parameter z is defined such that if |a| > |b|, z is s; otherwise
*               if c is not 0 z is 1/c; otherwise z is 1. Input Parameters:
*
*  @param a Provides the x-coordinate of the point p
*  @param b Provides the y-coordinate of the point p
*
*               Output Parameters:
*
*  @param a Contains the parameter r associated with the Givens rotation.
*  @param b Contains the parameter z associated with the Givens rotation.
*  @param c Contains the parameter c associated with the Givens rotation.
*  @param s Contains the parameter s associated with the Givens rotation.
*  @example
*
*/
void rotg(X)(X* a, X* b, X* c, X* s)
{
	X sA = csgn(*a); X sB = csgn(*b);
	X aA = abs(*a); X aB = abs(*b);
	X sigma = aA > aB ? sA : sB;
	X r = sigma*sqrt((*a)*(*a) + (*b)*(*b));
	*c = r != 0 ? (*a)/r : 1;
	*s = r != 0 ? (*b)/r : 0;
	X z = aA > aB ? (*s) : ((*c) != 0 && r != 0) ? 1/(*c) : ((*c) == 0 && r != 0) ? 1 : 0;
	*a = r; *b = z;
}

void rotg(X: Complex!T, T = typeof(X.re))(X* a, X* b, T* c, X* s)
{
	T aA = abs(*a); T aB2 = abs2(*b);
	X psi = (*a)/aA;
	X r;
	T sqrtA2B2 = sqrt(aA*aA + aB2);
	if(aA != 0){
	    r = psi*sqrtA2B2;
	    *c = aA/sqrtA2B2;
	    *s = (psi*conj(*b))/sqrtA2B2;
	}
	else{
		r = (*b);
		*c = 0;
		*s = complex(1, 0);
	}
	*a = r;
}

unittest
{
    double[] params = runif(4, 1.0, 5.0);
	double a1 = params[0], b1 = params[1], c1 = params[2], s1 = params[3];
	double a2 = params[0], b2 = params[1], c2 = params[2], s2 = params[3];
	rotg(&a1, &b1, &c1, &s1);
	cblas_drotg(&a2, &b2, &c2, &s2);
	
	assert(approxEqual(a1, a2), "rotg test: a output failed.");
	assert(approxEqual(b1, b2), "rotg test: b output failed.");
	assert(approxEqual(c1, c2), "rotg test: c output failed.");
	assert(approxEqual(s1, s2), "rotg test: s output failed.");
}

/**
*  @title rotm blas function: Performs Givens rotation of points in the plane
*  @description Given two vectors x and y, each vector element of these vectors is
*               replaced as follows:
*
*               |x_i|      |x_i|
*               |   |  =  H|   |
*               |y_i|      |y_i|
*
*         for i = 1 to n, where H is a modified Givens transformation matrix whose
*         values are stored in the param[1] to param[4] array.
*
*  Input parameters:
*
*  @param n Specifies the number of elements in vectors x and y.
*  @param x Array, size at least (1 + (n - 1)*abs(incx))
*  @param incx Specifies the increment for the elements of x.
*  @param y Array, size at least (1 + (n - 1)*abs(incy))
*  @param param Array, size 5. These elements of the param array are:
*         param[0] contains a switch, flag. param[1-4] contain h11, h12, h21, 
*         and h22.
*
*                          | h_11  h_12 |
*         flag = -1.0: H = |            |
*                          | h_21  h_22 |
*
*
*                          | 1.0   h_12 |
*         flag =  0.0: H = |            |
*                          | h_21  1.0  |
*
*
*                          | h_11  1.0  |
*         flag = 1.0:  H = |            |
*                          | -1.0  h_22 |
*
*
*                          | 1.0   0.0  |
*         flag = -2.0: H = |            |
*                          | 0.0   1.0  |
*
*         In the last three cases, the matrix entries of 1.0, -1.0, and 
*         0.0 are assumed based on the value of flag and are not required
*         to be set in the param vector.
*
*
*  Output Parameters:
*
*  @param x Each element x[i] is replaced by h_11*x[i] + h_12*y[i]
*  @param y Each element y[i] is replaced by h_21*x[i] + h_22*y[i]
*
*/
void rotm(N, X)(N n, X* x, in N incx, X* y, in N incy, X* param)
if(isIntegral!N && isFloatingPoint!X)
{
	X h11, h12, h21, h22, temp1, temp2;
	if(*param == -1)
	{
		param += 1; h11 = *param; param += 1; h21 = *param;
		param += 1; h12 = *param; param += 1; h22 = *param;
		while(n){
			temp1 = (*x)*h11; temp1 += (*y)*h12;
			temp2 = (*x)*h21; temp2 += (*y)*h22;
			*x = temp1; *y = temp2;
			--n; x += incx; y += incy;
		}
		return;
	}
	if(*param == 0)
	{
		param += 1; param += 1; h21 = *param;
		param += 1; h12 = *param;
		while(n){
			temp1 = (*x); temp1 += (*y)*h12;
			temp2 = (*x)*h21; temp2 += (*y);
			*x = temp1;
			*y = temp2;
			--n; x += incx; y += incy;
		}
		return;
	}
	if(*param == 1)
	{
		param += 1; h11 = *param; param += 1;
		param += 1; param += 1; h22 = *param;
		while(n){
			temp1 = (*x)*h11; temp1 += (*y);
			temp2 = -(*x); temp2 += (*y)*h22;
			*x = temp1;
			*y = temp2;
			--n; x += incx; y += incy;
		}
		return;
	}
	/* Don't have to do anything if *param = -2 */
	return;
}


unittest
{
    double[] param1 = runif(5, -5.0, 5.0), param2 = param1;
    double[] x1 = runif(10, -5.0, 5.0);
    double[] y1 = runif(10, -5.0, 5.0);
    double[] x2 = x1;
    double[] y2 = y1;
    
    int l1 = cast(int)x2.length;

	rotm(x1.length, x1.ptr, 1, y1.ptr, 1, param1.ptr);
	cblas_drotm(l1, x2.ptr, 1, y2.ptr, 1, param2.ptr);

	assert(approxEqual(x1, x2), "rotm test: x output failed.");
	assert(approxEqual(y1, y2), "rotm test: y output failed.");
}



/* rotmg Taken from the GSL library*/
/**
*  @title rotmg blas function: Computes the parameters for a modified Givens rotation
*  @description Given Cartesian coordinates (x1, y1) of an input vector, these routines
*               compute the components of a modified Givens transformation matrix H that
*               zeros the y-component of the resulting vector:
*
*               |x_1|      |x_1 sqrt(d1)|
*               |   |  =  H|            |
*               | 0 |      |y_1 sqrt(d1)|
*
*               This algorithm was essentially translated from GSL's cblas library;
*
*  Input parameters:
*
*  @param d1 Provides the scaling factor for the x-coordinate of the input vector
*  @param d2 Provides the scaling fector for the y-coordinate of the input vector
*  @param x1 Provides the x-coordinate of the input vector
*  @param y1 Provides the y-coordinate of the input vector
*
*
*  Output parameters:
*
*  @param d1 Provides the first diagonal element of the updated matrix
*  @param d2 Provides the second diagonal element of the updated matrix
*  @param x1 Provides the x-coordinate of the rotated vector defore scaling
*  @param param Array, size 5. These elements of the param array are:
*         param[0] contains a switch, flag. param[1-4] contain h11, h12, h21, 
*         and h22.
*
*                          | h_11  h_12 |
*         flag = -1.0: H = |            |
*                          | h_21  h_22 |
*
*
*                          | 1.0   h_12 |
*         flag =  0.0: H = |            |
*                          | h_21  1.0  |
*
*
*                          | h_11  1.0  |
*         flag = 1.0:  H = |            |
*                          | -1.0  h_22 |
*
*
*                          | 1.0   0.0  |
*         flag = -2.0: H = |            |
*                          | 0.0   1.0  |
*
*         In the last three cases, the matrix entries of 1.0, -1.0, and 
*         0.0 are assumed based on the value of flag and are not required
*         to be set in the param vector.
*
*
*  Output Parameters:
*
*  @param x Each element x[i] is replaced by h_11*x[i] + h_12*y[i]
*  @param y Each element y[i] is replaced by h_21*x[i] + h_22*y[i]
*
*/
void rotmg(X)(X* d1, X* d2, X* b1, in X b2, X* param)
if(isFloatingPoint!X)
{
	immutable X G = 4096.0, G2 = G * G;
    X D1 = *d1, D2 = *d2, x = *b1, y = b2;
    X h11, h12, h21, h22, u;

    X c, s;

    if (D1 < 0.0) {
	    param[0] = -1;
	    param[1] = 0;
	    param[2] = 0;
	    param[3] = 0;
	    param[4] = 0;
	    *d1 = 0;
	    *d2 = 0;
	    *b1 = 0;
	    return;
    }

    if (D2 * y == 0.0) {
        param[0] = -2;
        return;
    }

    c = fabs(D1 * x * x);
    s = fabs(D2 * y * y);

    if (c > s) {

	    param[0] = 0.0;

	    h11 = 1;
	    h12 = (D2 * y) / (D1 * x);
	    h21 = -y / x;
	    h22 = 1;

	    u = 1 - h21 * h12;

	    if (u <= 0.0) {
	        param[0] = -1;
	        param[1] = 0;
	        param[2] = 0;
	        param[3] = 0;
	        param[4] = 0;
	        *d1 = 0;
	        *d2 = 0;
	        *b1 = 0;
	      return;
	    }

	    D1 /= u;
	    D2 /= u;
	    x *= u;
	} else {

	    if (D2 * y * y < 0.0) {
	        param[0] = -1;
	        param[1] = 0;
	        param[2] = 0;
	        param[3] = 0;
	        param[4] = 0;
	        *d1 = 0;
	        *d2 = 0;
	        *b1 = 0;
	        return;
	    }

	    param[0] = 1;

	    h11 = (D1 * x) / (D2 * y);
	    h12 = 1;
	    h21 = -1;
	    h22 = x / y;

	    u = 1 + h11 * h22;

	    D1 /= u;
	    D2 /= u;

	    {
	        X tmp = D2;
	        D2 = D1;
	        D1 = tmp;
	    }

	    x = y * u;
    }

    while (D1 <= 1.0 / G2 && D1 != 0.0) {
	    param[0] = -1;
	    D1 *= G2;
	    x /= G;
	    h11 /= G;
	    h12 /= G;
	}

	while (D1 >= G2) {
	    param[0] = -1;
	    D1 /= G2;
	    x *= G;
	    h11 *= G;
	    h12 *= G;
	}

	while (fabs(D2) <= 1.0 / G2 && D2 != 0.0) {
	    param[0] = -1;
	    D2 *= G2;
	    h21 /= G;
	    h22 /= G;
	}

	while (fabs(D2) >= G2) {
	    param[0] = -1;
	    D2 /= G2;
	    h21 *= G;
	    h22 *= G;
	}

	*d1 = D1;
	*d2 = D2;
	*b1 = x;

	if (param[0] == -1.0) {
	    param[1] = h11;
	    param[2] = h21;
	    param[3] = h12;
	    param[4] = h22;
	} else if (param[0] == 0.0) {
	    param[2] = h21;
	    param[3] = h12;
	} else if (param[0] == 1.0) {
	    param[1] = h11;
	    param[4] = h22;
	}
}

unittest
{
    double[] params = runif(4, -5.0, 5.0);
	double d11 = params[0], d12 = d11, d21 = params[1], d22 = d21;
	double b11 = params[2], b12 = b11, b21 = params[3], b22 = b21;

	double[5] param1, param2;
	rotmg(&d11, &d21, &b11, b22, param1.ptr);
	cblas_drotmg(&d12, &d22, &b12, b22, param2.ptr);

	assert(approxEqual(d11, d12), "rotmg test: d1 output failed.");
	assert(approxEqual(d21, d22), "rotmg test: d2 output failed.");
	assert(approxEqual(b11, b12), "rotmg test: b1 output failed.");
	assert(approxEqual(b21, b22), "rotmg test: b2 output failed.");
}

/**
*  @title swap blas function: Swaps a vector with another vector
*  @description Given two vectors x and y, the swap routine returns
*               vectors y and x swapped, each replacing the other.
*
*  Input Parameters:
*
*  @param n The number of elements in vectors x and y
*  @param x Pointer to array size at least (1 + (n + 1)*abs(incx))
*  @param incx Specifies the increment for the elements of x
*  @param y Pointer to array size at least (1 + (n + 1)*abs(incy))
*  @param incy Specifies the increment for the elements of y
*
*  Output Parameters:
*  
*  @param x Contains the resultant vector x that containes the swapped elements in y.
*  @param y Contains the resultant vector y that containes the swapped elements in x.
* 
*/
void swap(N, X)(N n , X* x , in N incx , X* y , in N incy)
{
	X temp;
	while(n)
	{
		temp = *x;
		*x = *y;
		*y = temp;
		x += incx; y += incy;
		--n;
	}
}

unittest
{
    double[] x1 = runif(10, -5.0, 5.0), x2 = x1;
	double[] y1 = runif(10, -5.0, 5.0), y2 = y1;

    int l1 = cast(int)x1.length;
	swap(x1.length , x1.ptr , 1 , y1.ptr, 1);
	cblas_dswap(l1, x2.ptr, 1, y2.ptr, 1);

	assert(approxEqual(x1, x2), "swap test: x output failed.");
	assert(approxEqual(y1, y2), "swap test: y output failed.");
}

template CreateCmp(string cmp = ">")
{
	enum string CreateCmp = "\tif(abs(*x) " ~ cmp ~ " temp){\n\t\tend = x;\n\t\ttemp = abs(*x);\n\t}";
}

/* Legacy no longer required */
N MinMaxElOld(string cmp, N, X)(N n, X* x, N incx)
{
	if(incx < 0)
		return 0;
	/* Use two pointers to represent start and end */
	X temp = abs(*x); auto start = x, end = x;
	--n; x += incx; end = x;
	while(n)
	{
		/*if(abs(*x) < temp){
			end = x;
			temp = abs(*x);
		}*/
		mixin(CreateCmp!(cmp));
		--n; x += incx;
	}
	return end - start;
}

/* i?amin and i?amax by staged templates */
template MinMaxEl(string cmp)
{
	N MinMaxEl(N, X)(N n, X* x, N incx){
		if(incx < 0)
			return 0;
		/* Use two pointers to represent start and end */
		X temp = abs(*x); auto start = x, end = x;
		--n; x += incx; end = x;
		while(n)
		{
			/*if(abs(*x) < temp){
				end = x;
				temp = abs(*x);
			}*/
			mixin(CreateCmp!(cmp));
			--n; x += incx;
		}
		return end - start;
    }
}

/**
*  @title iamax blas function: Finds the index of the element with maximum absolute value
*  @description Given a vector x, the iamax (iamin) functions return the position of the vector
*               element x[i] that has the largest (smallest) absolute value for real numbers
*               and complex numbers. If n is not positive, 0 is returned.
*
*  Input Parameters:
*
*  @param n Specifies the number of elements in vector x;
*  @param x Pointer to array size at least (1 + (n + 1)*abs(incx))
*  @param incx Specifies the increment for the elements of x
*
*  Return values:
*  
*  @param Returns the position of vector element that has the largest absolute value such that the index
*         x[index] has the largest absolute value.
* 
*/
alias MinMaxEl!">" iamax;
alias MinMaxEl!"<" iamin;


unittest
{
    double[] x1 = runif(10, -10.0, 10.0), x2 = x1;

    int l2 = cast(int)x2.length;
	int out1 = cast(int)iamax(x1.length , x1.ptr, 1);
	int out2 = cblas_idamax(l2, x2.ptr, 1);

	assert(approxEqual(out1, out2), "iamax test: output failed.");
}



/******************************************************************************/

