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
    void cblas_zher2k(in CBLAS_ORDER Order, in CBLAS_UPLO Uplo, in CBLAS_TRANSPOSE Trans, in int N, in int K, 
                      in void *alpha, in void *A, in int lda, in void *B, in int ldb, in double beta, void *C, in int ldc);
}

/* Testing for trsm */
void main(){

    CBLAS_ORDER order = CblasRowMajor;
    CBLAS_UPLO uplo = CblasUpper;
    CBLAS_TRANSPOSE trans = CblasNoTrans;

    int n = 8, k = 2;
    double alpha = 1.0;
    double beta = 1.0;
    double[] a = [0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15];
    double[] b = [15, 7, 14, 6, 13, 5, 12, 4, 11, 3, 10, 2, 9, 1, 8, 0];
    double[] c = [0, 1, 3, 6, 10, 15, 21, 28,
                  0, 2, 4, 7, 11, 16, 22, 29, 
                  0, 0, 5, 8, 12, 17, 23, 30,
                  0, 0, 0, 9, 13, 18, 24, 31, 
                  0, 0, 0, 0, 14, 19, 25, 32,
                  0, 0, 0, 0, 0,  20, 26, 33, 
                  0, 0, 0, 0, 0,  0,  27, 34,
                  0, 0, 0, 0, 0,  0,  0,  35];
    int lda = 2, ldb = 2, ldc = 8;
    her2k(order, uplo, trans, n, k, alpha, a.ptr, lda, b.ptr, ldb, beta, c.ptr, ldc);
    writeln("her2k (CblasRowMajor, CblasUpper, CblasNoTrans): ", c);

    order = CblasRowMajor;
    uplo = CblasLower;
    trans = CblasTrans;

    n = 8, k = 3;
    alpha = 1.0, beta = 1.0;

    a = [0, 3, 6,  9,  12, 15, 18, 21,
         1, 4, 7, 10, 13, 16, 19, 22,
         2, 5, 8, 11, 14, 17, 20, 23];
    b = [1, 2, 3, 4, 5, 6, 7, 8,
         2, 3, 4, 5, 6, 7, 8, 9,
         3, 4, 5, 6, 7, 8, 9, 10];
    c = [ 0, 0,  0,  0,  0,  0,  0,  0, 
          1, 8,  0,  0,  0,  0,  0,  0,
          2, 9,  15, 0,  0,  0,  0,  0,
          3, 10, 16, 21, 0,  0,  0,  0,
          4, 11, 17, 22, 26, 0,  0,  0,
          5, 12, 18, 23, 27, 30, 0,  0,
          6, 13, 19, 24, 28, 31, 33, 0,
          7, 14, 20, 25, 29, 32, 34, 35];
    lda = 8, ldb = 8, ldc = 8;
    her2k(order, uplo, trans, n, k, alpha, a.ptr, lda, b.ptr, ldb, beta, c.ptr, ldc);
    writeln("her2k (CblasRowMajor, CblasLower, CblasTrans): ", c);

    order = CblasRowMajor;
    uplo = CblasUpper;
    trans = CblasTrans;

    n = 3, k = 5;
    Complex!double alphac = complex(1.0, 1.0), betac = complex(1.0, 0);
    Complex!double[] ac = [complex(2, 0), complex(3, 2), complex(4, 1),
                           complex(3, 3), complex(8, 0), complex(2, 5),
                           complex(1, 3), complex(2, 1), complex(6, 0),
                           complex(3, 3), complex(8, 0), complex(2, 5),
                           complex(1, 9), complex(3, 0), complex(6, 7)];

    Complex!double[] bc = [complex(4, 5), complex(6, 7), complex(8, 0),
                           complex(1, 9), complex(3, 0), complex(6, 7),
                           complex(3, 3), complex(8, 0), complex(2, 5),
                           complex(1, 3), complex(2, 1), complex(6, 0),
                           complex(2, 0), complex(3, 2), complex(4, 1)];

    Complex!double[] cc = [complex(6, 0), complex(3, 4), complex(9, 1),
                           complex(0, 0), complex(10, 0), complex(12, 2),
                           complex(0, 0), complex(0, 0), complex(3, 0),
                           complex(0, 0), complex(0, 0), complex(0, 0)];
    lda = 5, ldb = 5, ldc = 4;
    her2k(order, uplo, trans, n, k, alphac, ac.ptr, lda, bc.ptr, ldb, betac, cc.ptr, ldc);
    writeln("her2k (CblasRowMajor, CblasUpper, CblasTrans): ", cc);

    order = CblasRowMajor;
    uplo = CblasLower;
    trans = CblasNoTrans;
    n = 3, k = 5;
    alphac = complex(1, 1), betac = complex(1, 1);
    ac = [complex(2, 5), complex(3, 2), complex(4, 1), complex(1, 7), complex(0, 0),
          complex(3, 3), complex(8, 5), complex(2, 5), complex(2, 4), complex(1, 2),
          complex(1, 3), complex(2, 1), complex(6, 5), complex(3, 2), complex(2, 2)];
    bc = [complex(1, 5), complex(6, 2), complex(3, 1), complex(2, 0), complex(1, 0),
          complex(2, 4), complex(7, 5), complex(2, 5), complex(2, 4), complex(0, 0),
          complex(3, 5), complex(8, 1), complex(1, 5), complex(1, 0), complex(1, 1)];
    cc = [complex(2, 3), complex(0, 0), complex(0, 0),
          complex(1, 9), complex(3, 3), complex(0, 0),
          complex(4, 5), complex(6, 7), complex(8, 3),
          complex(0, 0), complex(0, 0), complex(0, 0)];
    lda = 3, ldb = 3, ldc = 4;
    her2k(order, uplo, trans, n, k, alphac, ac.ptr, lda, bc.ptr, ldb, betac, cc.ptr, ldc);
    writeln("her2k (CblasRowMajor, CblasLower, CblasNoTrans): ", cc);

    order = CblasColMajor;
    uplo = CblasUpper;
    trans = CblasNoTrans;

    n = 8, k = 2;
    alpha = 1.0;
    beta = 1.0;
    a = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
    b = [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0];
    c = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0,
                  3, 4, 5, 0, 0, 0, 0, 0, 0, 0, 6, 7, 8, 9, 0, 0, 0, 0, 0, 0,
                  10, 11, 12, 13, 14, 0, 0, 0, 0, 0, 15, 16, 17, 18, 19, 20, 0, 0, 0, 0,
                  21, 22, 23, 24, 25, 26, 27, 0, 0, 0,
                  28, 29, 30, 31, 32, 33, 34, 35, 0, 0];
    lda = 8, ldb = 8, ldc = 10;
    her2k(order, uplo, trans, n, k, alpha, a.ptr, lda, b.ptr, ldb, beta, c.ptr, ldc);
    writeln("her2k (CblasColMajor, CblasUpper, CblasNoTrans): ", c);
}

