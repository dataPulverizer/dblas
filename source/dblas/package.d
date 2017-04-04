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

/* C function for gemm */
extern (C){
    void
cblas_zhemm (in CBLAS_ORDER Order, in CBLAS_SIDE Side,
             in CBLAS_UPLO Uplo, in int M, in int N,
             in void *alpha, in void *A, in int lda, in void *B,
             in int ldb, in void *beta, void *C, in int ldc);
}

/* Testing for hemm */
void main(){
    CBLAS_LAYOUT order = CblasRowMajor;
    //CBLAS_TRANSPOSE transA = CblasNoTrans;
    //CBLAS_TRANSPOSE transB = CblasTrans;
    CBLAS_SIDE side = CblasLeft;
    CBLAS_UPLO uplo = CblasUpper;

    Complex!double[] a = [complex(-0.359, 0.089)];
    Complex!double[] b = [complex(-0.451, -0.337), complex(-0.901, -0.871)];
    Complex!double[] c = [complex(0.729, 0.631), complex(0.364, 0.246)];
    Complex!double alpha = complex(0, 0.1), beta = alpha;

    int ldc = 2, ldb = 2, lda = 1, m = 1, n = 2;
    hemm(order, side, uplo, m, n, alpha, a.ptr, lda, b.ptr, ldb, beta, c.ptr, ldc);
    writeln("hemm (Complex!double): ", c);
}

/*void hemm(N, X)(in CBLAS_ORDER order, in CBLAS_SIDE side, in CBLAS_UPLO uplo, in N m, in N n,
                in X* alpha, in X* a, in N lda, in X* b, in N ldb, in X* beta, X* c, in N ldc)*/

