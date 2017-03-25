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

/* C function for syr2 */
extern (C){
  void cblas_dtbmv(in CBLAS_ORDER order, in CBLAS_UPLO Uplo,
             in CBLAS_TRANSPOSE TransA, in CBLAS_DIAG Diag,
             in int N, in int K, in double *A, in int lda,
             double *X, in int incX);
}

/* Testing for syr2 */
void main(){
    CBLAS_LAYOUT layout = CblasRowMajor;
    CBLAS_UPLO uplo = CblasUpper;
    CBLAS_TRANSPOSE trans = CblasNoTrans;
    CBLAS_DIAG diag = CblasNonUnit;
    double alpha = 0.0;
    double[] a = [0.439, -0.484, -0.952, -0.508, 0.381, -0.889, -0.192, -0.279, -0.155];
    double[] x = [-0.089, -0.688, -0.203];
    tbmv(layout, uplo, trans, diag, 3, 1, a.ptr, 3, x.ptr, -1);
    writeln(x);
}




