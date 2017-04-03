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
    void cblas_dgemm(const CBLAS_ORDER Order, const CBLAS_TRANSPOSE TransA,
             const CBLAS_TRANSPOSE TransB, const int M, const int N,
             const int K, const double alpha, const double *A, const int lda,
             const double *B, const int ldb, const double beta, double *C,
             const int ldc);
}

/* Testing for gemm */
void main(){
    CBLAS_LAYOUT order = CblasRowMajor;
    CBLAS_TRANSPOSE transA = CblasNoTrans;
    CBLAS_TRANSPOSE transB = CblasTrans;

    double[] a = [0.939, 0.705, 0.977, 0.4];
    double[] b = [-0.089, -0.822, 0.937, 0.159, 0.789, -0.413, -0.172, 0.88];
    double[] c = [-0.619, 0.063];
    Complex!double[] ac = [complex(0.109, 0.892), complex(-0.723, 0.793), complex(0.109, -0.419), complex(-0.534, 0.448)];
    Complex!double[] bc = [complex(-0.875, -0.31), complex(-0.027, 0.067), complex(0.274, -0.126), complex(-0.548, 0.497),
                           complex(0.681, 0.388), complex(0.909, 0.889), complex(0.982, -0.074), complex(-0.788, 0.233)];
    Complex!double[] cc = [complex(0.503, 0.067), complex(0.239, 0.876)];
    gemm(order, transA, transB, 1, 2, 4, -0.3, a.ptr, 4, b.ptr, 4, 1, c.ptr, 2);
    gemm(order, transA, transB, 1, 2, 4, complex(0., 0.1), ac.ptr, 4, bc.ptr, 4, complex(1., 0.), cc.ptr, 2);
    writeln("gemm (double): ", c);
    writeln("gemm (Complex!double): ", cc);
}

