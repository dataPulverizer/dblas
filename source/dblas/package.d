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

/* C function for trmm */
extern (C){
    void cblas_dtrmm (in CBLAS_ORDER Order, in CBLAS_SIDE Side, in CBLAS_UPLO Uplo, in CBLAS_TRANSPOSE TransA,
                      in CBLAS_DIAG Diag, in int M, in int N, in double alpha, in double* A, in int lda, double* B, in int ldb);
}

/* Testing for trmm */
void main(){

    CBLAS_ORDER order = CblasRowMajor;
    CBLAS_SIDE side = CblasLeft;
    CBLAS_UPLO uplo = CblasUpper;
    CBLAS_TRANSPOSE trans = CblasNoTrans;
    CBLAS_DIAG diag = CblasNonUnit;

    int m = 2;
    int n = 3;
    double alpha = -0.3;
    double[] a = [0.174, -0.308, 0.997, -0.484];
    double[] b = [-0.256, -0.178, 0.098, 0.004, 0.97, -0.408];
    int lda = 2, ldb = 3;
    //double B_expected[] = { 0.0137328, 0.0989196, -0.0428148, 5.808e-04, 0.140844, -0.0592416 };
    trmm(order, side, uplo, trans, diag, m, n, alpha, a.ptr, lda, b.ptr, ldb);
    writeln("trmm: ", b);

    Complex!double alphac = complex(0, 0);
    Complex!double[] ac = [complex(0.463, 0.033), complex(-0.929, 0.949), complex(0.864, 0.986), complex(0.393, 0.885)];
    Complex!double[] bc = [complex(-0.321, -0.852), complex(-0.337, -0.175), complex(0.607, -0.613), 
                          complex(0.688, 0.973), complex(-0.331, -0.35), complex(0.719, -0.553)];
    //Complex!double[] B_expected = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    trmm(order, side, uplo, trans, diag, m, n, alphac, ac.ptr, lda, bc.ptr, ldb);
    writeln("trmm (Complex!double): ", bc);
}

