/** 
*
* @title  BLAS implementation for D
* @author Chibisi Chima-Okereke
* @date   2017-03-01
*
*/

module dblas;
public import dblas.l1;
public import dblas.l2;
public import dblas.l3;

import std.stdio : writeln;
import std.complex: Complex, complex;

/* To compile: */
/* dub build dblas # or dub run ... */

/* C function for symm */
extern (C){
    void cblas_dsymm(in CBLAS_ORDER Order, in CBLAS_SIDE Side, in CBLAS_UPLO Uplo, in int M, in int N, in double alpha,
                     in double *A, in int lda, in double *B, in int ldb, in double beta, double *C, in int ldc);
}

/* Testing for symm */
void main(){
    CBLAS_ORDER order = CblasRowMajor;
    CBLAS_SIDE side = CblasRight;
    CBLAS_UPLO uplo = CblasUpper;
    int m = 1;
    int n = 2;
    double alpha = -1;
    double beta = 1;
    double[] a = [0.591, -0.01, -0.192, -0.376];
    double[] b = [0.561, 0.946];
    double[] c = [0.763, 0.189];

    Complex!double alphac = complex(-1, 0);
    Complex!double betac = complex(-0.3, 0.1);
    Complex!double[] ac = [complex(-0.835, 0.344), complex(0.975, 0.634), complex(0.312, -0.659), complex(-0.624, -0.175)];
    Complex!double[] bc = [complex(-0.707, -0.846), complex(0.825, -0.661)];
    Complex!double[] cc = [complex(0.352, -0.499), complex(0.267, 0.548)];

    int lda = 2, ldb = 2, ldc = 2;
    symm(order, side, uplo, 1, 2, alpha, a.ptr, lda, b.ptr, ldb, beta, c.ptr, ldc);
    symm(order, side, uplo, 1, 2, alphac, ac.ptr, lda, bc.ptr, ldb, betac, cc.ptr, ldc);
    writeln("symm: ", c);
    writeln("symm (Complex!double): ", cc);
}

