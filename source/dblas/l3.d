/**
*
*  @title: Level 3 BLAS algorithms for D
*  @author: Chibisi Chima-Okereke
*  @date: 2017-04-03
*  @description Level 3 BLAS algorithms for D translated from GSL library.
*  
*/

module dblas.l3;
import std.stdio : writeln;

import dblas.l1;
import dblas.l2;
import std.math: abs, fabs, sqrt;
import std.complex: Complex, complex, conj;



/** 
*  @title gemm Computes a matrix-matrix product with general matrices.
*
*  @description      The gemm routines compute a scalar-matrix-matrix product and add 
*                    the result to a scalar-matrix product, with general matrices. The operation 
*                    is defined as:
*                    C := alpha*op(A)*op(B) + beta*C, where
*
*                    op(X) is one of op(X) = X , or op(X) = X T , or op(X) = X H ,
*                    alpha and beta are scalars,
*                    A, B and C are matrices:
*                    * op(A) is an m-by-k matrix,
*                    * op(B) is a k-by-n matrix,
*                    * C is an m-by-n matrix.
*
*  Input Parameters:
*
*  @param order      Specifies whether two-dimensional array storage is row-major
*                    (CblasRowMajor) or column-major (CblasColMajor).
*
*  @param transa:    Specifies the form of op(A) used in the matrix multiplication:
*                    if transa = CblasNoTrans , then op(A) = A ;
*                    if transa = CblasTrans , then op(A) = A T ;
*                    if transa = CblasConjTrans , then op(A) = A H .
*
*  @param transb:    Specifies the form of op(B) used in the matrix multiplication:
*                    if transb = CblasNoTrans , then op(B) = B ;
*                    if transb = CblasTrans , then op(B) = B T ;
*                    if transb = CblasConjTrans , then op(B) = B H .
*
*  @param m:         Specifies the number of rows of the matrix op(A) and of the matrix C. The
*                    value of m must be at least zero.
*
*  @param n:         Specifies the number of columns of the matrix op(B) and the number of
*                    columns of the matrix C.
*                    The value of n must be at least zero.
*
*  @param k:         Specifies the number of columns of the matrix op(A) and the number of
*                    rows of the matrix op(B) .
*                    The value of k must be at least zero.
*
*  @param alpha:     Specifies the scalar alpha.
*
*  @param a:         ...
*
*  @param lda:       Specifies the leading dimension of a as declared in the calling (sub)program.
*
*  @param b:         ...
*
*  @param ldb:       Specifies the leading dimension of b as declared in the calling (sub)program.
*
*  @param beta:      Specifies the scalar beta.
*                    When beta is equal to zero, then c need not be set on input.
*
*  @param b:         ...
*
*  @param ldc:       Specifies the leading dimension of c as declared in the calling (sub)program.
*
*  Output Parameters:
*                    c Overwritten by the m -by- n matrix (alpha*op(A)*op(B) + beta*C).
*
*/
void gemm(N, X)(in CBLAS_ORDER order, in CBLAS_TRANSPOSE transA, in CBLAS_TRANSPOSE transB, in N m, in N n,
                in N K, in X alpha, in X* a, in N lda, in X* b, in N ldb, in X beta, X* c, in N ldc)
{
    N i, j, k;
    N n1, n2;
    N ldf, ldg;
    N conjF, conjG, transF, transG;
    const(X)* F, G;
    X zero = X(0), one = X(1);
    
    if (alpha == zero && beta == one)
        return;
    
    if (order == CblasRowMajor) {
        n1 = m;
        n2 = n;
        F = a;
        ldf = lda;
        conjF = (transA == CblasConjTrans) ? -1 : 1;
        transF = (transA == CblasConjTrans) ? CblasTrans : transA;
        G = b;
        ldg = ldb;
        conjG = (transB == CblasConjTrans) ? -1 : 1;
        transG = (transB == CblasConjTrans) ? CblasTrans : transB;
    } else {
        n1 = n;
        n2 = m;
        F = b;
        ldf = ldb;
        conjF = (transA == CblasConjTrans) ? -1 : 1;
        transF = (transB == CblasConjTrans) ? CblasTrans : transB;
        G = a;
        ldg = lda;
        conjG = (transB == CblasConjTrans) ? -1 : 1;
        transG = (transA == CblasConjTrans) ? CblasTrans : transA;
    }
    
    /* form  y := beta*y */
    if (beta == zero) {
        for (i = 0; i < n1; i++) {
            for (j = 0; j < n2; j++) {
                c[ldc * i + j] = zero;
            }
        }
    } else if (beta != one) {
        for (i = 0; i < n1; i++) {
            for (j = 0; j < n2; j++) {
                c[ldc * i + j] *= beta;
            }
        }
    }
    
    if (alpha == zero)
        return;
    
    if (transF == CblasNoTrans && transG == CblasNoTrans) {
        /* form  C := alpha*A*B + C */
        for (k = 0; k < K; k++) {
            for (i = 0; i < n1; i++) {
            	static if(isComplex!X)
                	    const X Fik = X(F[ldf*i + k].re, conjF*F[ldf*i + k].im);
                	else
                		const X Fik = F[ldf*i + k];
                const X temp = alpha*Fik;
                if (temp != zero) {
                    for (j = 0; j < n2; j++) {
                    	static if(isComplex!X)
                	        const X Gkj = X(G[ldg*k + j].re, conjG*F[ldg*k + j].im);
                	    else
                		    const X Gkj = G[ldg*k + j];
                        c[ldc * i + j] += temp * Gkj;
                    }
                }
            }
        }
    
    } else if (transF == CblasNoTrans && transG == CblasTrans) {
    
        /* form  C := alpha*A*B' + C */
        
        for (i = 0; i < n1; i++) {
            for (j = 0; j < n2; j++) {
                X temp = zero;
                for (k = 0; k < K; k++) {
                	static if(isComplex!X){
                	    const X Fik = X(F[ldf*i + k].re, conjF*F[ldf*i + k].im);
                	    const X Gjk = X(G[ldg*j + k].re, conjG*G[ldg*j + k].im);
                	}
                	else{
                		const X Fik = F[ldf*i + k];
                		const X Gjk = G[ldg*j + k];
                	}
                    temp += Fik*Gjk;
                }
                c[ldc * i + j] += alpha * temp;
            }
        }
    
    } else if (transF == CblasTrans && transG == CblasNoTrans) {
    
        for (k = 0; k < K; k++) {
            for (i = 0; i < n1; i++) {
            	static if(isComplex!X)
                	    const X Fki = X(F[ldf*k + i].re, conjF*F[ldf*k + i].im);
                	else
                		const X Fki = F[ldf*k + i];
                const X temp = alpha*Fki;
                if (temp != zero) {
                    for (j = 0; j < n2; j++) {
                    	static if(isComplex!X)
                	        const X Gkj = X(G[ldg*k + j].re, conjG*G[ldg*k + j].im);
                	    else
                		    const X Gkj = G[ldg*k + j];
                        c[ldc * i + j] += temp*Gkj;
                    }
                }
            }
        }
    
    } else if (transF == CblasTrans && transG == CblasTrans) {
    
        for (i = 0; i < n1; i++) {
            for (j = 0; j < n2; j++) {
                X temp = zero;
                for (k = 0; k < K; k++) {
                	static if(isComplex!X){
                	    const X Fki = X(F[ldf*k + i].re, conjF*F[ldf*k + i].im);
                	    const X Gjk = X(G[ldg*j + k].re, conjG*G[ldg*j + k].im);
                	}
                	else{
                		const X Fki = F[ldf*k + i];
                		const X Gjk = G[ldg*j + k];
                	}
                    temp += Fki * Gjk;
                }
                c[ldc * i + j] += alpha * temp;
            }
        }
    
    } else {
        assert(0, "unrecognized operation");
    }
}







