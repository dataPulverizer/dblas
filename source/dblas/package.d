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
    void cblas_dtbsv (const CBLAS_ORDER order, const CBLAS_UPLO Uplo,
             const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag,
             const int N, const int K, const double *A, const int lda,
             double *X, const int incX);

}

/* Testing for dtbsv */
void main(){
    CBLAS_LAYOUT layout = CblasRowMajor;
    CBLAS_UPLO uplo = CblasUpper;
    CBLAS_TRANSPOSE transA = CblasNoTrans;
    CBLAS_DIAG diag = CblasNonUnit;

    double[] a = [-0.681, 0.209, 0.436, -0.369, 0.786, -0.84, 0.86, -0.233, 0.734];
    double[] x = [-0.305, 0.61, -0.831];
    tbsv(layout, uplo, transA, diag, 3, 1, a.ptr, 3, x.ptr, -1);
    writeln(x);
}




