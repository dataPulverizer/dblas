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

/* C function for syr2k */
extern (C){
    void cblas_dsyr2k (in CBLAS_ORDER Order, in CBLAS_UPLO Uplo, in CBLAS_TRANSPOSE Trans, in int N, in int K,
                       in double alpha, in double *A, in int lda, in double *B, in int ldb, in double beta, double *C, in int ldc);
}

/* Testing for syr2k */
void main(){

    CBLAS_ORDER order = CblasRowMajor;
    CBLAS_UPLO uplo = CblasUpper;
    CBLAS_TRANSPOSE trans = CblasNoTrans;
    int n = 1;
    int k = 2;
    double alpha = 0.1;
    double beta = 0;
    double[] a = [-0.225, 0.857];
    double[] b = [-0.933, 0.994];
    double[] c = [0.177];
    int lda = 2, ldb = 2, ldc = 1;

    //double C_expected[] = { 0.2123566 };
    syr2k(order, uplo, trans, n, k, alpha, a.ptr, lda, b.ptr, ldb, beta, c.ptr, ldc);
    writeln("syr2k: ", c);

    Complex!double alphac = complex(0, 0);
    Complex!double betac = complex(-0.3, 0.1);
    Complex!double[] ac = [complex(-0.315, 0.03), complex(0.281, 0.175)];
    Complex!double[] bc = [complex(-0.832, -0.964), complex(0.291, 0.476)];
    Complex!double[] cc = [complex(-0.341, 0.743)];
    //double C_expected[] = { 0.028, -0.257 };

    syr2k(order, uplo, trans, n, k, alphac, ac.ptr, lda, bc.ptr, ldb, betac, cc.ptr, ldc);
    writeln("syr2k (Complex!double): ", cc);
}

