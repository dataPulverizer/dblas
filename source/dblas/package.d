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

/* C function for hemm */
extern (C){
    void cblas_zherk(in CBLAS_ORDER Order, in CBLAS_UPLO Uplo, in CBLAS_TRANSPOSE Trans, in int N, in int K,
                     in double alpha, in void *A, in int lda, in double beta, void *C, in int ldc);
}

/* Testing for hemm */
void main(){
    CBLAS_TRANSPOSE transA = CblasNoTrans;

    CBLAS_ORDER order = CblasColMajor;
    CBLAS_UPLO uplo = CblasUpper;
    CBLAS_TRANSPOSE trans = CblasNoTrans;
    int n = 2;
    int k = 1;
    double alpha = 1;
    double beta = 1;
    Complex!double[] a = [complex(0.16, 0.464), complex(-0.623, 0.776)];
    Complex!double[] c = [complex(0.771, -0.449), complex(0.776, 0.112), complex(-0.134, 0.317), complex(0.547, -0.551)];
    int lda = 2, ldc = 2;

    // double C_expected[] = { 1.011896, 0.0, 0.776, 0.112, 0.126384, -0.096232, 1.537305, 0.0 };

    herk(order, uplo, trans, n, k, alpha, a.ptr, lda, beta, c.ptr, ldc);

    writeln("herk (Complex!double): ", c);
}

