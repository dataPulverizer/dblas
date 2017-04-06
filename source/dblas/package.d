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

/* C function for trsm */
extern (C){
    void cblas_dtrsm(in CBLAS_ORDER Order, in CBLAS_SIDE Side, in CBLAS_UPLO Uplo, in CBLAS_TRANSPOSE TransA,
                     in CBLAS_DIAG Diag, in int M, in int N, in double alpha, in double* A, in int lda, double* B, in int ldb);
}

/* Testing for trsm */
void main(){

    CBLAS_ORDER order = CblasRowMajor;
    CBLAS_SIDE side = CblasLeft;
    CBLAS_UPLO uplo = CblasUpper;
    CBLAS_TRANSPOSE trans = CblasNoTrans;
    CBLAS_DIAG diag = CblasNonUnit;
    int m = 2, n = 3;
    double alpha = -0.3;
    double[] a = [-0.279, 0.058, 0.437, 0.462];
    double[] b = [0.578, 0.473, -0.34, -0.128, 0.503, 0.2];
    int lda = 2, ldb = 3;
    //double B_expected[] = [0.638784, 0.440702, -0.392589, 0.0831169, -0.326623, -0.12987];

    trsm(order, side, uplo, trans, diag, m, n, alpha, a.ptr, lda, b.ptr, ldb);
    writeln("trsm: ", b);


    Complex!double alphac = complex(0, 0);
    Complex!double[] ac = [complex(0.189, 0.519), complex(-0.455, -0.444), complex(-0.21, -0.507), complex(-0.591, 0.859)];
    Complex!double[] bc = [complex(-0.779, -0.484), complex(0.249, -0.107), complex(-0.755, -0.047), complex(0.941, 0.675),
                          complex(-0.757, 0.645), complex(-0.649, 0.242)];
    // Complex!double[] B_expected = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    trsm(order, side, uplo, trans, diag, m, n, alphac, ac.ptr, lda, bc.ptr, ldb);
    writeln("trsm (Complex!double): ", bc);
}

