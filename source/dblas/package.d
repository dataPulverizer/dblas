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
    void cblas_dtpsv(in CBLAS_ORDER order, in CBLAS_UPLO uplo,
             in CBLAS_TRANSPOSE transA, in CBLAS_DIAG diag,
             in int N, in double *Ap, double *X, in int incX);
}

/* Testing for tpmv */
void main(){
    CBLAS_LAYOUT order = CblasRowMajor;
    CBLAS_UPLO uplo = CblasUpper;
    CBLAS_TRANSPOSE transA = CblasNoTrans;
    CBLAS_DIAG diag = CblasNonUnit;

    double[] a = [-0.381, 0.53, 0.451];
    double[] x = [0.144, 0.032];
    tpsv(order, uplo, transA, diag, 2, a.ptr, x.ptr, -1);
    writeln("tpsv: ", x);

    Complex!double[] ac = [complex(0.052, 0.875), complex(0.751, -0.912), complex(0.832, -0.153)];
    Complex!double[] xc = [complex(0.344, -0.143), complex(-0.668, -0.945)];
    tpsv(order, uplo, transA, diag, 2, ac.ptr, xc.ptr, -1);
    writeln("tpsv: ", xc);
}

