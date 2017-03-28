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
    void cblas_dtpmv (in CBLAS_ORDER order, in CBLAS_UPLO Uplo,
                 in CBLAS_TRANSPOSE TransA, in CBLAS_DIAG Diag,
                 in int N, in double* Ap, double* X, in int incX);
}

/* Testing for tpmv */
void main(){
    CBLAS_LAYOUT layout = CblasRowMajor;
    CBLAS_UPLO uplo = CblasUpper;
    CBLAS_TRANSPOSE transA = CblasNoTrans;
    CBLAS_DIAG diag = CblasNonUnit;

    double[] a = [-0.587, 0.14, 0.841];
    double[] x = [-0.213, 0.885];
    tpmv(layout, uplo, transA, diag, 2, a.ptr, x.ptr, -1);
    writeln("tpmv: ", x);
    Complex!double[] ac = [complex(0.254, 0.263), complex(-0.271, -0.595), complex(-0.182, -0.672)];
    Complex!double[] xc = [complex(-0.042, -0.705), complex(-0.255, -0.854)];
    tpmv(layout, uplo, transA, diag, 2, ac.ptr, xc.ptr, -1);
    writeln("tpmv: ", xc);
}

