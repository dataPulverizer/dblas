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

/* C function for dtbsv */
extern (C){
    void cblas_ztbsv (in CBLAS_ORDER order, in CBLAS_UPLO Uplo,
             in CBLAS_TRANSPOSE TransA, in CBLAS_DIAG Diag,
             in int N, in int K, in void *A, in int lda, void *X,
             in int incX);
}

/* Testing for dtbsv */
void main(){
    CBLAS_LAYOUT layout = CblasRowMajor;
    CBLAS_UPLO uplo = CblasUpper;
    CBLAS_TRANSPOSE transA = CblasNoTrans;
    CBLAS_DIAG diag = CblasNonUnit;

    Complex!double[] a = [complex(0.474, 0.715), complex(0.061, 0.532), complex(0.004, -0.318), 
        complex(0.37, -0.692), complex(-0.166, 0.039), complex(-0.946, 0.857), complex(-0.922, -0.491),
        complex(0.012, -0.217), complex(-0.674, -0.429)];
    Complex!double[] x = [complex(-0.123, 0.122), complex(0.981, 0.321), complex(0.942, 0.98)];
    tbsv(layout, uplo, transA, diag, 3, 1, a.ptr, 3, x.ptr, -1);
    writeln("tbsv: ", x);
    double[] ar = [-0.681, 0.209, 0.436, -0.369, 0.786, -0.84, 0.86, -0.233, 0.734];
    double[] xr = [-0.305, 0.61, -0.831];
    tbsv(layout, uplo, transA, diag, 3, 1, ar.ptr, 3, xr.ptr, -1);
    writeln("tbsv: ", xr);
}

