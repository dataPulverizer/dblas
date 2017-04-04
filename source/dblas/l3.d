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
                in N k, in X alpha, in X* a, in N lda, in X* b, in N ldb, in X beta, X* c, in N ldc)
{
    N i, j, k_;
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
        for (k_ = 0; k_ < k; k_++) {
            for (i = 0; i < n1; i++) {
                static if(isComplex!X)
                        const X Fik = X(F[ldf*i + k_].re, conjF*F[ldf*i + k_].im);
                    else
                        const X Fik = F[ldf*i + k_];
                const X temp = alpha*Fik;
                if (temp != zero) {
                    for (j = 0; j < n2; j++) {
                        static if(isComplex!X)
                            const X Gkj = X(G[ldg*k_ + j].re, conjG*F[ldg*k_ + j].im);
                        else
                            const X Gkj = G[ldg*k_ + j];
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
                for (k_ = 0; k_ < k; k_++) {
                    static if(isComplex!X){
                        const X Fik = X(F[ldf*i + k_].re, conjF*F[ldf*i + k_].im);
                        const X Gjk = X(G[ldg*j + k_].re, conjG*G[ldg*j + k_].im);
                    }
                    else{
                        const X Fik = F[ldf*i + k_];
                        const X Gjk = G[ldg*j + k_];
                    }
                    temp += Fik*Gjk;
                }
                c[ldc * i + j] += alpha * temp;
            }
        }
    
    } else if (transF == CblasTrans && transG == CblasNoTrans) {
    
        for (k_ = 0; k_ < k; k_++) {
            for (i = 0; i < n1; i++) {
                static if(isComplex!X)
                        const X Fki = X(F[ldf*k_ + i].re, conjF*F[ldf*k_ + i].im);
                    else
                        const X Fki = F[ldf*k_ + i];
                const X temp = alpha*Fki;
                if (temp != zero) {
                    for (j = 0; j < n2; j++) {
                        static if(isComplex!X)
                            const X Gkj = X(G[ldg*k_ + j].re, conjG*G[ldg*k_ + j].im);
                        else
                            const X Gkj = G[ldg*k_ + j];
                        c[ldc * i + j] += temp*Gkj;
                    }
                }
            }
        }
    
    } else if (transF == CblasTrans && transG == CblasTrans) {
    
        for (i = 0; i < n1; i++) {
            for (j = 0; j < n2; j++) {
                X temp = zero;
                for (k_ = 0; k_ < k; k_++) {
                    static if(isComplex!X){
                        const X Fki = X(F[ldf*k_ + i].re, conjF*F[ldf*k_ + i].im);
                        const X Gjk = X(G[ldg*j + k_].re, conjG*G[ldg*j + k_].im);
                    }
                    else{
                        const X Fki = F[ldf*k_ + i];
                        const X Gjk = G[ldg*j + k_];
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



/** 
*  @title gemm Computes a matrix-matrix product where one input matrix is Hermitian.
*
*  @description      The ?hemm routines compute a scalar-matrix-matrix product using a Hermitian matrix 
*                    A and a general matrix B and add the result to a scalar-matrix product using a general
*                    matrix C. The operation is defined as:
*                    C := alpha*A*B + beta*C
*                    or
*                    C := alpha*B*A + beta*C,
*                    where
*
*                    alpha and beta are scalars,
*                    A is a Hermitian matrix,
*                    B and C are m-by-n matrices.
*
*
*  Input Parameters:
*
*  @param order:     Specifies whether two-dimensional array storage is row-major
*                    (CblasRowMajor) or column-major (CblasColMajor).
*
*  @param side:      Specifies whether the Hermitian matrix A appears on the left or right in the
*                    operation as follows:
*                    if side = CblasLeft , then C := alpha*A*B + beta*C;
*                    if side = CblasRight , then C := alpha*B*A + beta*C.
*
*  @param uplo:      Specifies whether the upper or lower triangular part of the Hermitian matrix
*                    A is used:
*                    If uplo = CblasUpper , then the upper triangular part of the Hermitian
*                    matrix A is used.
*                    If uplo = CblasLower , then the low triangular part of the Hermitian matrix
*                    A is used.
*
*  @param m:         Specifies the number of rows of the matrix C.
*                    The value of m must be at least zero.
*
*  @param n:         Specifies the number of columns of the matrix C.
*                    The value of n must be at least zero.
*
*  @param alpha:     Specifies the scalar alpha.
*
*  @param a:         Array, size lda* ka , where ka is m when side = CblasLeft and is n
*                    otherwise. Before entry with side = CblasLeft , the m-by-m part of the
*                    array a must contain the Hermitian matrix, such that when uplo =
*                    CblasUpper , the leading m-by-m upper triangular part of the array a must
*                    contain the upper triangular part of the Hermitian matrix and the strictly
*                    lower triangular part of a is not referenced, and when uplo = CblasLower ,
*                    the leading m-by-m lower triangular part of the array a must contain the
*                    lower triangular part of the Hermitian matrix, and the strictly upper
*                    triangular part of a is not referenced.
*                    Before entry with side = CblasRight , the n-by-n part of the array a must
*                    contain the Hermitian matrix, such that when uplo = CblasUpper , the
*                    leading n-by-n upper triangular part of the array a must contain the upper
*                    triangular part of the Hermitian matrix and the strictly lower triangular part
*                    of a is not referenced, and when uplo = CblasLower , the leading n-by-n
*                    lower triangular part of the array a must contain the lower triangular part of
*                    the Hermitian matrix, and the strictly upper triangular part of a is not
*                    referenced. The imaginary parts of the diagonal elements need not be set,
*                    they are assumed to be zero.
*
*  @param lda:       Specifies the leading dimension of a as declared in the calling (sub)
*                    program. When side = CblasLeft then lda must be at least max(1, m) ,
*                    otherwise lda must be at least max(1,n).
*
*  @param b:         For Layout = CblasColMajor : array, size ldb*n. The leading m-by-n part
*                    of the array b must contain the matrix B.
*
*                    For Layout = CblasRowMajor : array, size ldb*m. The leading n-by-m part
*                    of the array b must contain the matrix B
*
*  @param ldb:       Specifies the leading dimension of b as declared in the calling
*                    (sub)program. When Layout = CblasColMajor , ldb must be at least
*                    max(1, m) ; otherwise, ldb must be at least max(1, n).
*
*  @param beta:      Specifies the scalar beta.
*                    When beta is supplied as zero, then c need not be set on input.
*                    For Layout = CblasColMajor : array, size ldc*n . Before entry, the leading
*                    m-by-n part of the array c must contain the matrix C, except when beta is
*                    zero, in which case c need not be set on entry.
*
*  @param c:         For Layout = CblasRowMajor : array, size ldc*m . Before entry, the leading
*                    n-by-m part of the array c must contain the matrix C, except when beta is
*                    zero, in which case c need not be set on entry.
*
*  @param ldc:       Specifies the leading dimension of c as declared in the calling
*                    (sub)program. When Layout = CblasColMajor , ldc must be at least
*                    max(1, m) ; otherwise, ldc must be at least max(1, n).
*
*  Output Parameters:
*  
*  @param c:         Overwritten by the m-by-n updated matrix.
*
*/
void hemm(N, X)(in CBLAS_ORDER order, in CBLAS_SIDE side, in CBLAS_UPLO uplo, in N m, in N n,
                in X alpha, in X* a, in N lda, in X* b, in N ldb, in X beta, X* c, in N ldc)
{
    N i, j, k;
    N n1, n2;
    N uplo_, side_;
    X zero = X(0), one = X(1);
    
    if(alpha == zero && beta == one)
        return;
    
    if (order == CblasRowMajor) {
        n1 = m;
        n2 = n;
        uplo_ = uplo;
        side_ = side;
    } else {
        n1 = n;
        n2 = m;
        uplo_ = (uplo == CblasUpper) ? CblasLower : CblasUpper;
        side_ = (side == CblasLeft) ? CblasRight : CblasLeft;
    }
    
    /* form  y := beta*y */
    if (beta == zero) {
        for (i = 0; i < n1; i++) {
            for (j = 0; j < n2; j++) {
                c[ldc * i + j] = zero;
            }
        }
    } else if (!(beta == one)) {
        for (i = 0; i < n1; i++) {
            for (j = 0; j < n2; j++) {
                c[ldc*i + j] = beta*c[ldc*i + j];
            }
        }
    }
    
    if (alpha == zero)
        return;
    
    if (side_ == CblasLeft && uplo_ == CblasUpper) {
        /* form  C := alpha*A*B + C */
        for (i = 0; i < n1; i++) {
            for (j = 0; j < n2; j++) {
                const X Bij = b[ldb * i + j];
                const X temp1 = alpha*Bij;
                X temp2 = zero;
                /* const BASE Aii_imag = 0.0; */
                const X Aii = X(a[i*lda + i].re, 0);
                c[i*ldc + j] += temp1 * Aii;
                for (k = i + 1; k < n1; k++) {
                    c[k*ldc + j] += conj(a[i*lda + k])*temp1;
                    temp2 += a[i*lda + k]*b[ldb*k + j];
                }
                c[i*ldc + j] += alpha*temp2;
          }
        }
    } else if (side_ == CblasLeft && uplo_ == CblasLower) {
      /* form  C := alpha*A*B + C */
        for (i = 0; i < n1; i++) {
            for (j = 0; j < n2; j++) {
                const temp1 = alpha*b[ldb*i + j];
                X temp2 = zero;
                for (k = 0; k < i; k++) {
                    c[k*ldc + j] += conj(a[i*lda + k])*temp1;
                    temp2 += a[i*lda + k]*b[ldb*k + j];
                }
                c[i*ldc + j] += temp1*a[i*lda + i].re;
                c[i*ldc + j] += alpha*temp2;
            }
        }
    } else if (side_ == CblasRight && uplo_ == CblasUpper) {
        /* form  C := alpha*B*A + C */
        for (i = 0; i < n1; i++) {
            for (j = 0; j < n2; j++) {
                const temp1 = alpha*b[ldb*i + j];
                X temp2 = zero;
                {
                    c[i*ldc + j] += temp1*a[j*lda + j].re;
                }
                for (k = j + 1; k < n2; k++) {
                    c[i*ldc + k] += temp1*a[j*lda + k];
                    temp2 += conj(a[j*lda + k])*b[ldb*i + k];
                }
                c[i*ldc + j] += alpha*temp2;
            }
        }
    } else if (side_ == CblasRight && uplo_ == CblasLower) {
        /* form  C := alpha*B*A + C */
        for (i = 0; i < n1; i++) {
            for (j = 0; j < n2; j++) {
                const X temp1 = alpha*b[ldb*i + j];
                X temp2 = zero;
                for (k = 0; k < j; k++) {
                    c[i*ldc + k] = temp1*a[j*lda + k];
                    temp2 += b[ldb*i + k]*conj(a[j*lda + k]);
                }
                c[i*ldc + j] += temp1*a[j*lda + j].re;
                c[i*ldc + j] += alpha*temp2;
            }
        }
    } else {
        assert(0, "unrecognized operation");
    }
}




void herk(N, X: Complex!V, V)(in CBLAS_ORDER order, in CBLAS_UPLO uplo, in CBLAS_TRANSPOSE trans, in N n, 
                in N k, in V alpha, in X* a, in N lda, in V beta, X* c, in N ldc)
{
    N i, j, k_;
    N uplo_, trans_;
    X zero = X(0), one = X(1);
    
    if (beta == 1.0 && (alpha == 0.0 || k == 0))
        return;
    
    if (order == CblasRowMajor) {
        uplo_ = uplo;
        trans_ = trans;
    } else {
        uplo_ = (uplo == CblasUpper) ? CblasLower : CblasUpper;
        trans_ = (trans == CblasNoTrans) ? CblasConjTrans : CblasNoTrans;
    }
    
    /* form  y := beta*y */
    if (beta == 0.0) {
        if (uplo_ == CblasUpper) {
            for (i = 0; i < n; i++) {
                for (j = i; j < n; j++) {
                    /*REAL(C, ldc * i + j) = 0.0;
                    IMAG(C, ldc * i + j) = 0.0;*/
                    c[ldc*i + j] = zero;
                }
            }
        } else {
            for (i = 0; i < n; i++) {
                for (j = 0; j <= i; j++) {
                    /*REAL(C, ldc * i + j) = 0.0;
                    IMAG(C, ldc * i + j) = 0.0;*/
                    c[ldc*i + j] = zero;
                }
            }
        }
    } else if (beta != 1.0) {
        if (uplo_ == CblasUpper) {
            for (i = 0; i < n; i++) {
                /*REAL(C, ldc * i + i) *= beta;
                IMAG(C, ldc * i + i) = 0;*/
                c[ldc*i + i] *= X(c[ldc*i + i].re*beta, 0);
                for (j = i + 1; j < n; j++) {
                    /*REAL(C, ldc * i + j) *= beta;
                    IMAG(C, ldc * i + j) *= beta;*/
                    c[ldc*i + j] *= beta;
                }
            }
        } else {
            for (i = 0; i < n; i++) {
                for (j = 0; j < i; j++) {
                    /*REAL(C, ldc * i + j) *= beta;
                    IMAG(C, ldc * i + j) *= beta;*/
                    c[ldc*i + j] *= beta;
                }
                /*REAL(C, ldc * i + i) *= beta;
                IMAG(C, ldc * i + i) = 0;*/
                c[ldc*i + i] = X(c[ldc*i + i].re*beta, 0);
            }
        }
    } else {
        /* set imaginary part of Aii to zero */
        for (i = 0; i < n; i++) {
            /* IMAG(C, ldc * i + i) = 0.0; */
            c[ldc*i + i] = X(c[ldc*i + i].re, 0);
        }
    }
    
    if (alpha == 0.0)
        return;
    
    if (uplo_ == CblasUpper && trans_ == CblasNoTrans) {
    
        for (i = 0; i < n; i++) {
            for (j = i; j < n; j++) {
                /*BASE temp_real = 0.0;
                BASE temp_imag = 0.0;*/
                X temp = zero;
                for (k_ = 0; k_ < k; k_++) {
                    /*const BASE Aik_real = CONST_REAL(A, i * lda + k_);
                    const BASE Aik_imag = CONST_IMAG(A, i * lda + k_);
                    const BASE Ajk_real = CONST_REAL(A, j * lda + k_);
                    const BASE Ajk_imag = -CONST_IMAG(A, j * lda + k_);
                    temp_real += Aik_real * Ajk_real - Aik_imag * Ajk_imag;
                    temp_imag += Aik_real * Ajk_imag + Aik_imag * Ajk_real;*/
                    temp += a[i*lda + k_]*X(a[j*lda + k_].re, -a[j*lda + k_].im);
                }
                /*REAL(C, i * ldc + j) += alpha * temp_real;
                IMAG(C, i * ldc + j) += alpha * temp_imag;*/
                c[i*ldc + j] += alpha*temp;
            }
        }
    
    } else if (uplo_ == CblasUpper && trans_ == CblasConjTrans) {
    
        for (i = 0; i < n; i++) {
            for (j = i; j < n; j++) {
                /*BASE temp_real = 0.0;
                BASE temp_imag = 0.0;*/
                X temp = zero;
                for (k_ = 0; k_ < k; k_++) {
                    /*const BASE Aki_real = CONST_REAL(A, k_ * lda + i);
                    const BASE Aki_imag = -CONST_IMAG(A, k_ * lda + i);
                    const BASE Akj_real = CONST_REAL(A, k_ * lda + j);
                    const BASE Akj_imag = CONST_IMAG(A, k_ * lda + j);
                    temp_real += Aki_real * Akj_real - Aki_imag * Akj_imag;
                    temp_imag += Aki_real * Akj_imag + Aki_imag * Akj_real;*/
                    temp += X(a[k_*lda + i].re, -a[k_*lda + i].im)*a[k_*lda + j];
                }
                /*REAL(C, i * ldc + j) += alpha * temp_real;
                IMAG(C, i * ldc + j) += alpha * temp_imag;*/
                c[i*ldc + j] += alpha*temp;
            }
        }
    
    } else if (uplo_ == CblasLower && trans_ == CblasNoTrans) {
    
        for (i = 0; i < n; i++) {
            for (j = 0; j <= i; j++) {
                /*BASE temp_real = 0.0;
                BASE temp_imag = 0.0;*/
                X temp = zero;
                for (k_ = 0; k_ < k; k_++) {
                    /*const BASE Aik_real = CONST_REAL(A, i * lda + k_);
                    const BASE Aik_imag = CONST_IMAG(A, i * lda + k_);
                    const BASE Ajk_real = CONST_REAL(A, j * lda + k_);
                    const BASE Ajk_imag = -CONST_IMAG(A, j * lda + k_);
                    temp_real += Aik_real * Ajk_real - Aik_imag * Ajk_imag;
                    temp_imag += Aik_real * Ajk_imag + Aik_imag * Ajk_real;*/
                    temp += a[i*lda + k_]*X(a[j*lda + k_].re, -a[j*lda + k_].im);
                }
                /*REAL(C, i * ldc + j) += alpha * temp_real;
                IMAG(C, i * ldc + j) += alpha * temp_imag;*/
                c[i*ldc + j] += alpha*temp;
            }
        }
    
    } else if (uplo_ == CblasLower && trans_ == CblasConjTrans) {
    
        for (i = 0; i < n; i++) {
            for (j = 0; j <= i; j++) {
                /*BASE temp_real = 0.0;
                BASE temp_imag = 0.0;*/
                X temp = zero;
                for (k_ = 0; k_ < k; k_++) {
                    /*const BASE Aki_real = CONST_REAL(A, k_ * lda + i);
                    const BASE Aki_imag = -CONST_IMAG(A, k_ * lda + i);
                    const BASE Akj_real = CONST_REAL(A, k_ * lda + j);
                    const BASE Akj_imag = CONST_IMAG(A, k_ * lda + j);
                    temp_real += Aki_real * Akj_real - Aki_imag * Akj_imag;
                    temp_imag += Aki_real * Akj_imag + Aki_imag * Akj_real;*/
                    temp += a[k_*lda + j]*X(a[k_*lda + i].re, -a[k_*lda + i].im);
                }
                /*REAL(C, i * ldc + j) += alpha * temp_real;
                IMAG(C, i * ldc + j) += alpha * temp_imag;*/
                c[i*ldc + j] += alpha*temp;
            }
        }
    
    } else {
        assert(0, "unrecognized operation");
    }
}



