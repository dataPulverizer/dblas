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


import std.stdio : writeln;
import std.complex: Complex, complex;

/* To compile: */
/* dub build dblas # or dub run ... */

/* C function for tpmv */
extern (C){
    void cblas_dtrsv(in CBLAS_ORDER order, in CBLAS_UPLO Uplo, in CBLAS_TRANSPOSE TransA, in CBLAS_DIAG Diag,
                     in int N, in double *A, in int lda, double *X, in int incX);
}

/* Testing for tpmv */
void main(){
    CBLAS_LAYOUT order = CblasRowMajor;
    CBLAS_UPLO uplo = CblasUpper;
    CBLAS_TRANSPOSE transA = CblasNoTrans;
    CBLAS_DIAG diag = CblasNonUnit;

    double[] x = [0.348];
    double[] a = [0.995];
    trsv(order, uplo, transA, diag, 1, a.ptr, 1, x.ptr, -1);
    writeln("trsv (double): ", x);

    Complex!double[] xc = [complex(-0.627, 0.281)];
    Complex!double[] ac = [complex(0.977, -0.955)];
    trsv(order, uplo, transA, diag, 1, ac.ptr, 1, xc.ptr, -1);
    writeln("trsv (complex): ", xc);
}

