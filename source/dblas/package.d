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

/* C function for syrk */
extern (C){
    void cblas_dsyrk (in CBLAS_ORDER Order, in CBLAS_UPLO Uplo, in CBLAS_TRANSPOSE Trans, in int N, in int K,
                      in double alpha, in double *A, in int lda, in double beta, double *C, in int ldc);
}

/* Testing for syrk */
void main(){
    CBLAS_ORDER order = CblasRowMajor;
    CBLAS_TRANSPOSE trans = CblasNoTrans;
    CBLAS_UPLO uplo = CblasUpper;

    int n = 2;
    int k = 1;
    double alpha = -1.0;
    double beta = 0.1;
    double[] a = [0.412, -0.229];
    double[] c = [0.628, -0.664, -0.268, 0.096];
    int lda = 1, ldc = 2;
    syrk(order, uplo, trans, n, k, alpha, a.ptr, lda, beta, c.ptr, ldc);

    Complex!double alphac = complex(1, 0);
    Complex!double betac = complex(1, 0);
    Complex!double[] ac = [complex(-0.049, -0.687), complex(-0.434, 0.294)];
    Complex!double[] cc = [complex(0.937, -0.113), complex(0.796, 0.293), complex(0.876, -0.199), complex(-0.757, -0.103)];
    syrk(order, uplo, trans, n, k, alphac, ac.ptr, lda, betac, cc.ptr, ldc);

    writeln("syrk: ", c);
    writeln("syrk (Complex!double): ", cc);
}

