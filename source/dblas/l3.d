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
*  @description      The hemm routines compute a scalar-matrix-matrix product using a Hermitian matrix 
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



/** 
*  @title herk Performs a Hermitian rank-k update.
*
*  @description      The herk routines perform a rank-k matrix-matrix operation using a general matrix A and a Hermitian
*                    matrix C. The operation is defined as
*                    C := alpha*A*A H + beta*C,
*                    or
*                    C := alpha*A H *A + beta*C,
*                    where
*
*                    alpha and beta are real scalars,
*                    C is an n-by-n Hermitian matrix,
*                    A is an n-by-k matrix in the first case and a k-by-n matrix in the second case.
*
*
*  Input Parameters:
*
*  @param order:     Specifies whether two-dimensional array storage is row-major
*                    (CblasRowMajor) or column-major (CblasColMajor).
*
*  @param uplo:      Specifies whether the upper or lower triangular part of the array c is used.
*                    If uplo = CblasUpper, then the upper triangular part of the array c is
*                    used.
*                    If uplo = CblasLower, then the low triangular part of the array c is used.
*
*  @param trans:     Specifies the operation:
*                    if trans = CblasNoTrans , then C := alpha*A*A H + beta*C ;
*                    if trans = CblasConjTrans , then C := alpha*A H *A + beta*C.
*
*  @param n:         Specifies the order of the matrix C. The value of n must be at least zero.
*
*  @param k:         With trans = CblasNoTrans , k specifies the number of columns of the
*                    matrix A, and with trans = CblasConjTrans , k specifies the number of
*                    rows of the matrix A.
*                    The value of k must be at least zero.
*
*  @param alpha:     Specifies the scalar alpha.
*
*  @param a:         ...
*
*  @param lda:       ...
*
*  @param beta:      Specifies the scalar beta.
*
*  @param c:         Array, size ldc by n.
*                    Before entry with uplo = CblasUpper , the leading n-by-n upper triangular
*                    part of the array c must contain the upper triangular part of the Hermitian
*                    matrix and the strictly lower triangular part of c is not referenced.
*                    Before entry with uplo = CblasLower , the leading n-by-n lower triangular
*                    part of the array c must contain the lower triangular part of the Hermitian
*                    matrix and the strictly upper triangular part of c is not referenced.
*                    The imaginary parts of the diagonal elements need not be set, they are
*                    assumed to be zero.
*
*  @param ldc:       Specifies the leading dimension of c as declared in the calling
*                    (sub)program. The value of ldc must be at least max(1, n).
*
*  Output Parameters
*
*  @param c:         With uplo = CblasUpper , the upper triangular part of the array c is
*                    overwritten by the upper triangular part of the updated matrix.
*  
*                    With uplo = CblasLower , the lower triangular part of the array c is
*                    overwritten by the lower triangular part of the updated matrix.
*                    The imaginary parts of the diagonal elements are set to zero.
*
*/
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
                    c[ldc*i + j] = zero;
                }
            }
        } else {
            for (i = 0; i < n; i++) {
                for (j = 0; j <= i; j++) {
                    c[ldc*i + j] = zero;
                }
            }
        }
    } else if (beta != 1.0) {
        if (uplo_ == CblasUpper) {
            for (i = 0; i < n; i++) {
                c[ldc*i + i] *= X(c[ldc*i + i].re*beta, 0);
                for (j = i + 1; j < n; j++) {
                    c[ldc*i + j] *= beta;
                }
            }
        } else {
            for (i = 0; i < n; i++) {
                for (j = 0; j < i; j++) {
                    c[ldc*i + j] *= beta;
                }
                c[ldc*i + i] = X(c[ldc*i + i].re*beta, 0);
            }
        }
    } else {
        /* set imaginary part of Aii to zero */
        for (i = 0; i < n; i++) {
            c[ldc*i + i] = X(c[ldc*i + i].re, 0);
        }
    }
    
    if (alpha == 0.0)
        return;
    
    if (uplo_ == CblasUpper && trans_ == CblasNoTrans) {
    
        for (i = 0; i < n; i++) {
            for (j = i; j < n; j++) {
                X temp = zero;
                for (k_ = 0; k_ < k; k_++) {
                    temp += a[i*lda + k_]*X(a[j*lda + k_].re, -a[j*lda + k_].im);
                }
                c[i*ldc + j] += alpha*temp;
            }
        }
    
    } else if (uplo_ == CblasUpper && trans_ == CblasConjTrans) {
    
        for (i = 0; i < n; i++) {
            for (j = i; j < n; j++) {
                X temp = zero;
                for (k_ = 0; k_ < k; k_++) {
                    temp += X(a[k_*lda + i].re, -a[k_*lda + i].im)*a[k_*lda + j];
                }
                c[i*ldc + j] += alpha*temp;
            }
        }
    
    } else if (uplo_ == CblasLower && trans_ == CblasNoTrans) {
    
        for (i = 0; i < n; i++) {
            for (j = 0; j <= i; j++) {
                X temp = zero;
                for (k_ = 0; k_ < k; k_++) {
                    temp += a[i*lda + k_]*X(a[j*lda + k_].re, -a[j*lda + k_].im);
                }
                c[i*ldc + j] += alpha*temp;
            }
        }
    
    } else if (uplo_ == CblasLower && trans_ == CblasConjTrans) {
    
        for (i = 0; i < n; i++) {
            for (j = 0; j <= i; j++) {
                X temp = zero;
                for (k_ = 0; k_ < k; k_++) {
                    temp += a[k_*lda + j]*X(a[k_*lda + i].re, -a[k_*lda + i].im);
                }
                c[i*ldc + j] += alpha*temp;
            }
        }
    
    } else {
        assert(0, "unrecognized operation");
    }
}

/** 
*  @title symm Computes a matrix-matrix product where one input matrix is symmetric.
*
*  @description      The symm routines compute a scalar-matrix-matrix product with one symmetric matrix and add 
*                    the result to a scalar-matrix product. The operation is defined as:
*                    C := alpha*A*B + beta*C,
*                    or
*                    C := alpha*B*A + beta*C,
*                    where
*
*                    alpha and beta are scalars,
*                    A is a symmetric matrix,
*                    B and C are m-by-n matrices.
*
*  Input Parameters:
*
*  @param order:     Specifies whether two-dimensional array storage is row-major
*                    (CblasRowMajor) or column-major (CblasColMajor).
*
*  @param side:      Specifies whether the symmetric matrix A appears on the left or right in the
*                    operation:
*                    if side = CblasLeft , then C := alpha*A*B + beta*C;
*                    if side = CblasRight , then C := alpha*B*A + beta*C.
*
*  @param uplo:      Specifies whether the upper or lower triangular part of the symmetric
*                    matrix A is used:
*
*                    if uplo = CblasUpper, then the upper triangular part is used;
*                    if uplo = CblasLower, then the lower triangular part is used.
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
*                    otherwise.
*
*                    Before entry with side = CblasLeft , the m-by-m part of the array a must
*                    contain the symmetric matrix, such that when uplo = CblasUpper , the
*                    leading m-by-m upper triangular part of the array a must contain the upper
*                    triangular part of the symmetric matrix and the strictly lower triangular part
*                    of a is not referenced, and when side = CblasLeft , the leading m-by-m
*                    lower triangular part of the array a must contain the lower triangular part of
*                    the symmetric matrix and the strictly upper triangular part of a is not
*                    referenced.
*
*                    Before entry with side = CblasRight , the n-by-n part of the array a must
*                    contain the symmetric matrix, such that when uplo = CblasUpper e array a
*                    must contain the upper triangular part of the symmetric matrix and the
*                    strictly lower triangular part of a is not referenced, and when side =
*                    CblasLeft , the leading n-by-n lower triangular part of the array a must
*                    contain the lower triangular part of the symmetric matrix and the strictly
*                    upper triangular part of a is not referenced.
*
*  @param lda:       Specifies the leading dimension of a as declared in the calling
*                    (sub)program. When side = CblasLeft then lda must be at least max(1,
*                    m) , otherwise lda must be at least max(1, n) .
*
*  @param b:         For Layout = CblasColMajor : array, size ldb*n. The leading m-by-n part
*                    of the array b must contain the matrix B.
*                    For Layout = CblasRowMajor : array, size ldb*m. The leading n-by-m part
*                    of the array b must contain the matrix B
*
*  @param ldb:       Specifies the leading dimension of b as declared in the calling
*                    (sub)program. When Layout = CblasColMajor , ldb must be at least
*                    max(1, m) ; otherwise, ldb must be at least max(1, n) .
*
*  @param beta:      Specifies the scalar beta.
*                    When beta is set to zero, then c need not be set on input.
*
*  @param c:         For Layout = CblasColMajor : array, size ldc*n . Before entry, the leading
*                    m-by-n part of the array c must contain the matrix C, except when beta is
*                    zero, in which case c need not be set on entry.
*                    For Layout = CblasRowMajor : array, size ldc*m . Before entry, the leading
*                    n-by-m part of the array c must contain the matrix C, except when beta is
*                    zero, in which case c need not be set on entry.
*
*  @param ldc:       Specifies the leading dimension of c as declared in the calling
*                    (sub)program. When Layout = CblasColMajor , ldc must be at least
*                    max(1, m) ; otherwise, ldc must be at least max(1, n).
*
*  Output Parameters
*
*  @param c:         Overwritten by the m-by-n updated matrix.
*
*/
void symm(N, X)(in CBLAS_ORDER order, in CBLAS_SIDE side, in CBLAS_UPLO uplo, in N m, in N n,
                in X alpha, in X* a, in N lda, in X* b, in N ldb, in X beta, X* c, in N ldc)
{
    N i, j, k;
    N n1, n2;
    N uplo_, side_;
    X zero = X(0), one = X(1);
    
    if (alpha == zero && beta == one)
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
    } else if (beta != one) {
        for (i = 0; i < n1; i++) {
            for (j = 0; j < n2; j++) {
                c[ldc * i + j] *= beta;
            }
        }
    }
    
    if (alpha == zero)
        return;
    
    if (side_ == CblasLeft && uplo_ == CblasUpper) {
    
        /* form  C := alpha*A*B + C */
    
        for (i = 0; i < n1; i++) {
            for (j = 0; j < n2; j++) {
                const X temp1 = alpha * b[ldb * i + j];
                X temp2 = zero;
                c[i * ldc + j] += temp1 * a[i * lda + i];
                for (k = i + 1; k < n1; k++) {
                    const X Aik = a[i * lda + k];
                    c[k * ldc + j] += Aik * temp1;
                    temp2 += Aik * b[ldb * k + j];
                }
                c[i * ldc + j] += alpha * temp2;
            }
        }
    
    } else if (side_ == CblasLeft && uplo_ == CblasLower) {
    
        /* form  C := alpha*A*B + C */
        
        for (i = 0; i < n1; i++) {
            for (j = 0; j < n2; j++) {
                const X temp1 = alpha * b[ldb * i + j];
                X temp2 = zero;
                for (k = 0; k < i; k++) {
                    const X Aik = a[i * lda + k];
                    c[k * ldc + j] += Aik * temp1;
                    temp2 += Aik * b[ldb * k + j];
                }
                c[i * ldc + j] += temp1 * a[i * lda + i] + alpha * temp2;
            }
        }
    
    } else if (side_ == CblasRight && uplo_ == CblasUpper) {
      
        /* form  C := alpha*B*A + C */
        
        for (i = 0; i < n1; i++) {
            for (j = 0; j < n2; j++) {
                const X temp1 = alpha * b[ldb * i + j];
                X temp2 = zero;
                c[i * ldc + j] += temp1 * a[j * lda + j];
                for (k = j + 1; k < n2; k++) {
                    const X Ajk = a[j * lda + k];
                    c[i * ldc + k] += temp1 * Ajk;
                    temp2 += b[ldb * i + k] * Ajk;
                }
                c[i * ldc + j] += alpha * temp2;
            }
        }
    
    } else if (side_ == CblasRight && uplo_ == CblasLower) {
      
        /* form  C := alpha*B*A + C */
        
        for (i = 0; i < n1; i++) {
            for (j = 0; j < n2; j++) {
                const X temp1 = alpha * b[ldb * i + j];
                X temp2 = zero;
                for (k = 0; k < j; k++) {
                    const X Ajk = a[j * lda + k];
                    c[i * ldc + k] += temp1 * Ajk;
                    temp2 += b[ldb * i + k] * Ajk;
                }
                c[i * ldc + j] += temp1 * a[j * lda + j] + alpha * temp2;
            }
        }
    
    } else {
        assert(0, "unrecognized operation");
    }
}


/** 
*  @title syrk Performs a symmetric rank-k update.
*
*  @description      The syrk routines perform a rank-k matrix-matrix operation for a symmetric matrix C using a general
*                    matrix A. The operation is defined as:
*                    C := alpha*A*A' + beta*C,
*                    or
*                    C := alpha*A'*A + beta*C,
*                    where
*
*                    alpha and beta are scalars,
*                    C is an n-by-n symmetric matrix,
*                    A is an n-by-k matrix in the first case and a k-by-n matrix in the second case.
*
*  Input Parameters:
*
*  @param order:     Specifies whether two-dimensional array storage is row-major
*                    (CblasRowMajor) or column-major (CblasColMajor).
*
*  @param uplo:      Specifies whether the upper or lower triangular part of the array c is used.
*                    If uplo = CblasUpper, then the upper triangular part of the array c is
*                    used.
*
*                    If uplo = CblasLower, then the low triangular part of the array c is used.
*
*  @param trans:     Specifies the operation:
*                    if trans = CblasNoTrans , then C := alpha*A*A' + beta*C ;
*                    if trans = CblasTrans , then C := alpha*A'*A + beta*C ;
*                    if trans = CblasConjTrans , then C := alpha*A'*A + beta*C.
*
*  @param n:         Specifies the order of the matrix C. The value of n must be at least zero.
*
*  @param k:         On entry with trans = CblasNoTrans , k specifies the number of columns of
*                    the matrix a, and on entry with trans = CblasTrans or
*                    trans = CblasConjTrans , k specifies the number of rows of the matrix a.
*                    The value of k must be at least zero.
*
*  @param alpha:     Specifies the scalar alpha.
*
*  @param a:         Array, size lda* ka , where ka is k when trans = CblasNoTrans , and is n
*                    otherwise. Before entry with trans = CblasNoTrans , the leading n-by-k
*                    part of the array a must contain the matrix A, otherwise the leading k-by-n
*                    part of the array a must contain the matrix A.
*
*  @param lda:       ...
*
*  @param beta:      Specifies the scalar beta.
*
*  @param c:         Array, size ldc* n . Before entry with uplo = CblasUpper , the leading n-
*                    by-n upper triangular part of the array c must contain the upper triangular
*                    part of the symmetric matrix and the strictly lower triangular part of c is not
*                    referenced.
*
*                    Before entry with uplo = CblasLower , the leading n-by-n lower triangular
*                    part of the array c must contain the lower triangular part of the symmetric
*                    matrix and the strictly upper triangular part of c is not referenced.
*
*  @param ldc:       Specifies the leading dimension of c as declared in the calling
*                    (sub)program. The value of ldc must be at least max(1, n).
*
*  Output Parameters:
*
*  @param c:         With uplo = CblasUpper , the upper triangular part of the array c is
*                    overwritten by the upper triangular part of the updated matrix.
*                    With uplo = CblasLower , the lower triangular part of the array c is
*                    overwritten by the lower triangular part of the updated matrix.
*
*/
void syrk(N, X)(in CBLAS_ORDER order, in CBLAS_UPLO uplo, in CBLAS_TRANSPOSE trans, in N n, 
                in N k, in X alpha, in X* a, in N lda, in X beta, X* c, in N ldc)
{
    N i, j, k_;
    N uplo_, trans_;
    X zero = X(0), one = X(1);
    
    if (alpha == zero && beta == one)
        return;
    
    if (order == CblasRowMajor) {
        uplo_ = uplo;
        trans_ = (trans == CblasConjTrans) ? CblasTrans : trans;
    } else {
        uplo_ = (uplo == CblasUpper) ? CblasLower : CblasUpper;
        
        if (trans == CblasTrans || trans == CblasConjTrans) {
            trans_ = CblasNoTrans;
        } else {
            trans_ = CblasTrans;
        }
    }
    
    /* form  y := beta*y */
    if (beta == zero) {
        if (uplo_ == CblasUpper) {
            for (i = 0; i < n; i++) {
                for (j = i; j < n; j++) {
                    c[ldc * i + j] = zero;
                }
            }
        } else {
            for (i = 0; i < n; i++) {
                for (j = 0; j <= i; j++) {
                    c[ldc * i + j] = zero;
                }
            }
        }
    } else if (beta != one) {
        if (uplo_ == CblasUpper) {
            for (i = 0; i < n; i++) {
                for (j = i; j < n; j++) {
                    c[ldc * i + j] *= beta;
                }
            }
        } else {
            for (i = 0; i < n; i++) {
                for (j = 0; j <= i; j++) {
                    c[ldc * i + j] *= beta;
                }
            }
        }
    }
    
    if (alpha == zero)
        return;
    
    if (uplo_ == CblasUpper && trans_ == CblasNoTrans) {
    
        for (i = 0; i < n; i++) {
            for (j = i; j < n; j++) {
                X temp = zero;
                for (k_ = 0; k_ < k; k_++) {
                    temp += a[i * lda + k_] * a[j * lda + k_];
                }
                c[i * ldc + j] += alpha * temp;
            }
        }
    
    } else if (uplo_ == CblasUpper && trans_ == CblasTrans) {
    
        for (i = 0; i < n; i++) {
            for (j = i; j < n; j++) {
                X temp = zero;
                for (k_ = 0; k_ < k; k_++) {
                    temp += a[k_ * lda + i] * a[k_ * lda + j];
                }
                c[i * ldc + j] += alpha * temp;
            }
        }
    
    } else if (uplo_ == CblasLower && trans_ == CblasNoTrans) {
    
        for (i = 0; i < n; i++) {
            for (j = 0; j <= i; j++) {
                X temp = zero;
                for (k_ = 0; k_ < k; k_++) {
                    temp += a[i * lda + k_] * a[j * lda + k_];
                }
                c[i * ldc + j] += alpha * temp;
            }
        }
    
    } else if (uplo_ == CblasLower && trans_ == CblasTrans) {
    
        for (i = 0; i < n; i++) {
            for (j = 0; j <= i; j++) {
                X temp = zero;
                for (k_ = 0; k_ < k; k_++) {
                    temp += a[k_ * lda + i] * a[k_ * lda + j];
                }
                c[i * ldc + j] += alpha * temp;
            }
        }
    
    } else {
        assert(0, "unrecognized operation");
    }
}

/** 
*  @title syr2k Performs a symmetric rank-2k update.
*
*  @description            The syr2k routines perform a rank-2k matrix-matrix operation for a symmetric matrix C using general
*                          matrices A and B. The operation is defined as:
*                          C := alpha*A*B' + alpha*B*A' + beta*C,
*                          or
*                          C := alpha*A'*B + alpha*B'*A + beta*C,
*                          where
*                          
*                          alpha and beta are scalars,
*                          C is an n-by-n symmetric matrix,
*                          A and B are n-by-k matrices in the first case, and k-by-n matrices in the second case.
*                          
*  Input Parameters:       
*                          
*  @param order:           Specifies whether two-dimensional array storage is row-major
*                          (CblasRowMajor) or column-major (CblasColMajor).
*
*  @param uplo:            Specifies whether the upper or lower triangular part of the array c is used.
*                          If uplo = CblasUpper, then the upper triangular part of the array c is
*                          used.
*                          If uplo = CblasLower, then the low triangular part of the array c is used.
*
*  @param trans:           Specifies the operation:
*                          if trans = CblasNoTrans , then C := alpha*A*B'+ alpha*B*A'+ beta*C ;
*                          if trans = CblasTrans , then C := alpha*A'*B + alpha*B'*A + beta*C ;
*                          if trans = CblasConjTrans , then C := alpha*A'*B + alpha*B'*A
*                          + beta*C.
*
*  @param n:               Specifies the order of the matrix C. The value of n must be at least zero.
*
*  @param k:               On entry with trans = CblasNoTrans , k specifies the number of columns of
*                          the matrices A and B, and on entry with trans = CblasTrans or
*                          trans = CblasConjTrans , k specifies the number of rows of the matrices A
*                          and B. The value of k must be at least zero.
*
*  @param alpha:           Specifies the scalar alpha.
*
*  @param a:               ...
*
*  @param lda:             Specifies the leading dimension of a as declared in the calling
*                          (sub)program.
*  @param b:               ...
*
*  @param ldb:             Specifies the leading dimension of a as declared in the calling
*                          (sub)program.
*
*  @param beta:            Specifies the scalar beta.
*
*  @param c:               Array, size ldc* n . Before entry with uplo = CblasUpper , the leading n-
*                          by-n upper triangular part of the array c must contain the upper triangular
*                          part of the symmetric matrix and the strictly lower triangular part of c is not
*                          referenced.
*
*                          Before entry with uplo = CblasLower , the leading n-by-n lower triangular
*                          part of the array c must contain the lower triangular part of the symmetric
*                          matrix and the strictly upper triangular part of c is not referenced.
*
*  @param ldc:             Specifies the leading dimension of c as declared in the calling
*                          (sub)program. The value of ldc must be at least max(1, n).
*
*  Output Parameters:
*
*  @param c:               With uplo = CblasUpper , the upper triangular part of the array c is
*                          overwritten by the upper triangular part of the updated matrix.
*                          
*                          With uplo = CblasLower , the lower triangular part of the array c is
*                          overwritten by the lower triangular part of the updated matrix.
*
*/
void syr2k(N, X)(in CBLAS_ORDER order, in CBLAS_UPLO uplo, in CBLAS_TRANSPOSE trans, in N n, in N k,
                 in X alpha, in X* a, in N lda, in X* b, in N ldb, in X beta, X* c, in N ldc)
{
    N i, j, k_;
    N uplo_, trans_;
    X zero = X(0), one = X(1);
    
    if (alpha == zero && beta == one)
        return;
    
    if (order == CblasRowMajor) {
        uplo_ = uplo;
        trans_ = (trans == CblasConjTrans) ? CblasTrans : trans;
    } else {
        uplo_ = (uplo == CblasUpper) ? CblasLower : CblasUpper;
        
        if (trans == CblasTrans || trans == CblasConjTrans) {
            trans_ = CblasNoTrans;
        } else {
            trans_ = CblasTrans;
        }
    }
    /* form  C := beta*C */
    if (beta == zero) {
        if (uplo_ == CblasUpper) {
            for (i = 0; i < n; i++) {
                for (j = i; j < n; j++) {
                    c[ldc * i + j] = zero;
                }
            }
        } else {
            for (i = 0; i < n; i++) {
                for (j = 0; j <= i; j++) {
                    c[ldc * i + j] = zero;
                }
            }
        }
    } else if (beta != one) {
        if (uplo_ == CblasUpper) {
            for (i = 0; i < n; i++) {
                for (j = i; j < n; j++) {
                    c[ldc * i + j] *= beta;
                }
            }
        } else {
            for (i = 0; i < n; i++) {
                for (j = 0; j <= i; j++) {
                    c[ldc * i + j] *= beta;
                }
            }
        }
    }
    if (alpha == zero)
        return;
    
    if (uplo_ == CblasUpper && trans_ == CblasNoTrans) {
    
        for (i = 0; i < n; i++) {
            for (j = i; j < n; j++) {
                X temp = zero;
                for (k_ = 0; k_ < k; k_++) {
                    temp += (a[i * lda + k_] * b[j * ldb + k_]
                             + b[i * ldb + k_] * a[j * lda + k_]);
                }
                c[i * ldc + j] += alpha * temp;
            }
        }
    } else if (uplo_ == CblasUpper && trans_ == CblasTrans) {
        for (k_ = 0; k_ < k; k_++) {
            for (i = 0; i < n; i++) {
                X temp1 = alpha * a[k_ * lda + i];
                X temp2 = alpha * b[k_ * ldb + i];
                for (j = i; j < n; j++) {
                    c[i * lda + j] += temp1 * b[k_ * ldb + j] + temp2 * a[k_ * lda + j];
                }
            }
        }
    } else if (uplo_ == CblasLower && trans_ == CblasNoTrans) {
        for (i = 0; i < n; i++) {
            for (j = 0; j <= i; j++) {
                X temp = zero;
                for (k_ = 0; k_ < k; k_++) {
                    temp += (a[i * lda + k_] * b[j * ldb + k_]
                             + b[i * ldb + k_] * a[j * lda + k_]);
                }
                c[i * ldc + j] += alpha * temp;
            }
        }
    } else if (uplo_ == CblasLower && trans == CblasTrans) {
    
        for (k_ = 0; k_ < k; k_++) {
            for (i = 0; i < n; i++) {
                X temp1 = alpha * a[k_ * lda + i];
                X temp2 = alpha * b[k_ * ldb + i];
                for (j = 0; j <= i; j++) {
                    c[i * lda + j] += temp1 * b[k_ * ldb + j] + temp2 * a[k_ * lda + j];
                }
            }
        }
    } else {
        assert(0, "unrecognized operation");
    }
}



/** 
*  @title trmm Computes a matrix-matrix product where one input matrix is triangular.
*
*  @description            The trmm routines compute a scalar-matrix-matrix product with one triangular matrix. The operation is
*                          defined as:
*                          B := alpha*op(A)*B,
*                          or
*                          B := alpha*B*op(A),
*                          where
*                          
*                          alpha is a scalar,
*                          B is an m-by-n matrix,
*                          A is a unit, or non-unit, upper or lower triangular matrix
*                          op(A) is one of op(A) = A, or op(A) = A' , or op(A) = conjg(A') .
*
*  Input Parameters:       
*                          
*  @param order:           Specifies whether two-dimensional array storage is row-major
*                          (CblasRowMajor) or column-major (CblasColMajor).
*
*  @param side:            Specifies whether op(A) appears on the left or right of B in the operation:
*                          if side = CblasLeft , then B := alpha*op(A)*B ;
*                          if side = CblasRight , then B := alpha*B*op(A) .
*  @param uplo:            Specifies whether the matrix A is upper or lower triangular:
*                          uplo = CblasUpper
*                          if uplo = CblasLower , then the matrix is low triangular.
*  @param transa:          Specifies the form of op(A) used in the matrix multiplication:
*                          if transa = CblasNoTrans , then op(A) = A ;
*                          if transa = CblasTrans , then op(A) = A' ;
*                          if transa = CblasConjTrans , then op(A) = conjg(A') .
*  @param diag:            Specifies whether the matrix A is unit triangular:
*                          if diag = CblasUnit then the matrix is unit triangular;
*                          if diag = CblasNonUnit , then the matrix is not unit triangular.
*  @param m:               Specifies the number of rows of B. The value of m must be at least zero.
*  @param n:               Specifies the number of columns of B. The value of n must be at least zero.
*  @param alpha:           Specifies the scalar alpha.
*                          When alpha is zero, then a is not referenced and b need not be set before
*                          entry.
*  @param a:               Array, size lda by k, where k is m when side = CblasLeft and is n when
*                          side = CblasRight . Before entry with uplo = CblasUpper , the leading k
*                          by k upper triangular part of the array a must contain the upper triangular
*                          matrix and the strictly lower triangular part of a is not referenced.
*                          Before entry with uplo = CblasLower , the leading k by k lower triangular
*                          part of the array a must contain the lower triangular matrix and the strictly
*                          upper triangular part of a is not referenced.
*                          When diag = CblasUnit , the diagonal elements of a are not referenced
*                          either, but are assumed to be unity.
*  @param lda:             Specifies the leading dimension of a as declared in the calling
*                          (sub)program. When side = CblasLeft , then lda must be at least max(1,
*                          m) , when side = CblasRight , then lda must be at least max(1, n) .
*  @param b:               For Layout = CblasColMajor : array, size ldb*n . Before entry, the leading
*                          m-by-n part of the array b must contain the matrix B.
*                          For Layout = CblasRowMajor : array, size ldb*m . Before entry, the leading
*                          n-by-m part of the array b must contain the matrix B.
*  @param ldb:             Specifies the leading dimension of b as declared in the calling
*                          (sub)program. When Layout = CblasColMajor , ldb must be at least
*                          max(1, m); otherwise, ldb must be at least max(1, n).
*
*  Output Parameters:
*
*  @param b:               Overwritten by the transformed matrix.
*
*/
void trmm(N, X)(in CBLAS_ORDER order, in CBLAS_SIDE side, in CBLAS_UPLO uplo, in CBLAS_TRANSPOSE transA,
                in CBLAS_DIAG diag, in N m, in N n, in X alpha, in X* a, in N lda, X* b, in N ldb)
{
    N i, j, k;
    N n1, n2;
    X zero = X(0), one = X(1);
    
    const N nonunit = (diag == CblasNonUnit);
    N side_, uplo_, trans;
    
    if (order == CblasRowMajor) {
        n1 = m;
        n2 = n;
        side_ = side;
        uplo_ = uplo;
        trans = (transA == CblasConjTrans) ? CblasTrans : transA;
    } else {
        n1 = n;
        n2 = m;
        side_ = (side == CblasLeft) ? CblasRight : CblasLeft;
        uplo_ = (uplo == CblasUpper) ? CblasLower : CblasUpper;
        trans = (transA == CblasConjTrans) ? CblasTrans : transA;
    }
    
    if (side_ == CblasLeft && uplo_ == CblasUpper && trans == CblasNoTrans) {
        /* form  B := alpha * TriU(A)*B */
        for (i = 0; i < n1; i++) {
          for (j = 0; j < n2; j++) {
            X temp = zero;
        
            if (nonunit) {
              temp = a[i * lda + i] * b[i * ldb + j];
            } else {
              temp = b[i * ldb + j];
            }
        
            for (k = i + 1; k < n1; k++) {
              temp += a[lda * i + k] * b[k * ldb + j];
            }
        
            b[ldb * i + j] = alpha * temp;
          }
        }
    } else if (side_ == CblasLeft && uplo_ == CblasUpper && trans == CblasTrans) {
        /* form  B := alpha * (TriU(A))' *B */
        for (i = n1; i > 0 && i--;) {
            for (j = 0; j < n2; j++) {
                X temp = zero;
                
                for (k = 0; k < i; k++) {
                    temp += a[lda * k + i] * b[k * ldb + j];
                }
                
                if (nonunit) {
                    temp += a[i * lda + i] * b[i * ldb + j];
                } else {
                    temp += b[i * ldb + j];
                }
                
                b[ldb * i + j] = alpha * temp;
            }
        }
    } else if (side_ == CblasLeft && uplo_ == CblasLower && trans == CblasNoTrans) {
        
        /* form  B := alpha * TriL(A)*B */
        
        for (i = n1; i > 0 && i--;) {
            for (j = 0; j < n2; j++) {
                X temp = zero;
                
                for (k = 0; k < i; k++) {
                    temp += a[lda * i + k] * b[k * ldb + j];
                }
                
                if (nonunit) {
                    temp += a[i * lda + i] * b[i * ldb + j];
                } else {
                    temp += b[i * ldb + j];
                }
                
                b[ldb * i + j] = alpha * temp;
            }
        }
        
    } else if (side_ == CblasLeft && uplo_ == CblasLower && trans == CblasTrans) {
        /* form  B := alpha * TriL(A)' *B */
        for (i = 0; i < n1; i++) {
            for (j = 0; j < n2; j++) {
                X temp = zero;
                
                if (nonunit) {
                    temp = a[i * lda + i] * b[i * ldb + j];
                } else {
                    temp = b[i * ldb + j];
                }
                
                for (k = i + 1; k < n1; k++) {
                    temp += a[lda * k + i] * b[k * ldb + j];
                }
                
                b[ldb * i + j] = alpha * temp;
            }
        }
    } else if (side_ == CblasRight && uplo_ == CblasUpper && trans == CblasNoTrans) {
        
        /* form  B := alpha * B * TriU(A) */
        
        for (i = 0; i < n1; i++) {
            for (j = n2; j > 0 && j--;) {
                X temp = zero;
                
                for (k = 0; k < j; k++) {
                    temp += a[lda * k + j] * b[i * ldb + k];
                }
                
                if (nonunit) {
                    temp += a[j * lda + j] * b[i * ldb + j];
                } else {
                    temp += b[i * ldb + j];
                }
                
                b[ldb * i + j] = alpha * temp;
            }
        }
        
    } else if (side_ == CblasRight && uplo_ == CblasUpper && trans == CblasTrans) {
    
      /* form  B := alpha * B * (TriU(A))' */
    
      for (i = 0; i < n1; i++) {
        for (j = 0; j < n2; j++) {
          X temp = zero;
    
          if (nonunit) {
            temp = a[j * lda + j] * b[i * ldb + j];
          } else {
            temp = b[i * ldb + j];
          }
    
          for (k = j + 1; k < n2; k++) {
            temp += a[lda * j + k] * b[i * ldb + k];
          }
    
          b[ldb * i + j] = alpha * temp;
        }
      }
    
    } else if (side_ == CblasRight && uplo_ == CblasLower && trans == CblasNoTrans) {
        /* form  B := alpha *B * TriL(A) */
        for (i = 0; i < n1; i++) {
            for (j = 0; j < n2; j++) {
                X temp = zero;
                
                if (nonunit) {
                    temp = a[j * lda + j] * b[i * ldb + j];
                } else {
                    temp = b[i * ldb + j];
                }
                
                for (k = j + 1; k < n2; k++) {
                    temp += a[lda * k + j] * b[i * ldb + k];
                }
                
                b[ldb * i + j] = alpha * temp;
            }
        }
    } else if (side_ == CblasRight && uplo_ == CblasLower && trans == CblasTrans) {
        /* form  B := alpha * B * TriL(A)' */
        for (i = 0; i < n1; i++) {
            for (j = n2; j > 0 && j--;) {
                X temp = zero;
                
                for (k = 0; k < j; k++) {
                    temp += a[lda * j + k] * b[i * ldb + k];
                }
                
                if (nonunit) {
                    temp += a[j * lda + j] * b[i * ldb + j];
                } else {
                    temp += b[i * ldb + j];
                }
                
                b[ldb * i + j] = alpha * temp;
            }
        }
    } else {
        assert("unrecognized operation");
    }
}


/** 
*  @title trsm Solves a triangular matrix equation.
*
*  @description            The ?trsm routines solve one of the following matrix equations:
*                          op(A)*X = alpha*B,
*                          or
*                          X*op(A) = alpha*B,
*                          where
*                          
*                          alpha is a scalar,
*                          X and B are m-by-n matrices,
*                          A is a unit, or non-unit, upper or lower triangular matrix
*                          op(A) is one of op(A) = A , or op(A) = A' , or op(A) = conjg(A').
*                          The matrix B is overwritten by the solution matrix X.
*
*  Input Parameters:       
*                          
*  @param order:           Specifies whether two-dimensional array storage is row-major
*                          (CblasRowMajor) or column-major (CblasColMajor).
*
*  @param side:            Specifies whether op(A) appears on the left or right of X in the equation:
*                          if side = CblasLeft , then op(A)*X = alpha*B ;
*                          if side = CblasRight , then X*op(A) = alpha*B.
*
*  @param uplo:            Specifies whether the matrix A is upper or lower triangular:
*                          uplo = CblasUpper
*                          if uplo = CblasLower , then the matrix is low triangular.
*
*  @param transa:          Specifies the form of op(A) used in the matrix multiplication:
*                          if transa = CblasNoTrans , then op(A) = A ;
*                          if transa = CblasTrans ;
*                          if transa = CblasConjTrans , then op(A) = conjg(A').
*
*  @param diag:            Specifies whether the matrix A is unit triangular:
*                          if diag = CblasUnit then the matrix is unit triangular;
*                          if diag = CblasNonUnit , then the matrix is not unit triangular.
*
*  @param m:               Specifies the number of rows of B. The value of m must be at least zero.
*
*  @param n:               Specifies the number of columns of B. The value of n must be at least zero.
*
*  @param alpha:           Specifies the scalar alpha.
*                          When alpha is zero, then a is not referenced and b need not be set before
*                          entry.
*
*  @param a:               Array, size lda* k , where k is m when side = CblasLeft and is n when
*                          side = CblasRight . Before entry with uplo = CblasUpper , the leading k
*                          by k upper triangular part of the array a must contain the upper triangular
*                          matrix and the strictly lower triangular part of a is not referenced.
*                          Before entry with uplo = CblasLower lower triangular part of the array a
*                          must contain the lower triangular matrix and the strictly upper triangular
*                          part of a is not referenced.
*                          When diag = CblasUnit , the diagonal elements of a are not referenced
*                          either, but are assumed to be unity.
*
*  @param lda:             Specifies the leading dimension of a as declared in the calling
*                          (sub)program. When side = CblasLeft , then lda must be at least max(1,
*                          m) , when side = CblasRight , then lda must be at least max(1, n) .
*
*  @param b:               For Layout = CblasColMajor : array, size ldb*n . Before entry, the leading
*                          m-by-n part of the array b must contain the matrix B.
*                          For Layout = CblasRowMajor : array, size ldb*m . Before entry, the leading
*                          n-by-m part of the array b must contain the matrix B.
*
*  @param ldb:             Specifies the leading dimension of b as declared in the calling
*                          (sub)program. When Layout = CblasColMajor , ldb must be at least
*                          max(1, m) ; otherwise, ldb must be at least max(1, n).
*
*  Output Parameters:
*
*  @param b:               Overwritten by the solution matrix X.
*
*/
void trsm(N, X)(in CBLAS_ORDER order, in CBLAS_SIDE side, in CBLAS_UPLO uplo, in CBLAS_TRANSPOSE transA,
                in CBLAS_DIAG diag, in N m, in N n, in X alpha, in X* a, in N lda, X* b, in N ldb)
{
    N i, j, k;
    N n1, n2;
    
    const N nonunit = (diag == CblasNonUnit);
    N side_, uplo_, trans;
    X zero = X(0), one = X(1);
    
    if (order == CblasRowMajor) {
        n1 = m;
        n2 = n;
        side_ = side;
        uplo_ = uplo;
        trans = (transA == CblasConjTrans) ? CblasTrans : transA;
    } else {
        n1 = n;
        n2 = m;
        side_ = (side == CblasLeft) ? CblasRight : CblasLeft;
        uplo_ = (uplo == CblasUpper) ? CblasLower : CblasUpper;
        trans = (transA == CblasConjTrans) ? CblasTrans : transA;
    }
    if (side_ == CblasLeft && uplo_ == CblasUpper && trans == CblasNoTrans) {
        /* form  B := alpha * inv(TriU(A)) *B */
        if (alpha != one) {
            for (i = 0; i < n1; i++) {
                for (j = 0; j < n2; j++) {
                    b[ldb * i + j] *= alpha;
                }
            }
        }
        for (i = n1; i > 0 && i--;) {
            if (nonunit) {
                X Aii = a[lda * i + i];
                for (j = 0; j < n2; j++) {
                    b[ldb * i + j] /= Aii;
                }
            }
            for (k = 0; k < i; k++) {
                const X Aki = a[k * lda + i];
                for (j = 0; j < n2; j++) {
                    b[ldb * k + j] -= Aki * b[ldb * i + j];
                }
            }
        }
    } else if (side_ == CblasLeft && uplo_ == CblasUpper && trans == CblasTrans) {
        /* form  B := alpha * inv(TriU(A))' *B */
        if (alpha != one) {
            for (i = 0; i < n1; i++) {
                for (j = 0; j < n2; j++) {
                    b[ldb * i + j] *= alpha;
                }
            }
        }
        for (i = 0; i < n1; i++) {
            if (nonunit) {
                X Aii = a[lda * i + i];
                for (j = 0; j < n2; j++) {
                    b[ldb * i + j] /= Aii;
                }
            }
            
            for (k = i + 1; k < n1; k++) {
                const X Aik = a[i * lda + k];
                for (j = 0; j < n2; j++) {
                    b[ldb * k + j] -= Aik * b[ldb * i + j];
                }
            }
        }
    } else if (side_ == CblasLeft && uplo_ == CblasLower && trans == CblasNoTrans) {
        /* form  B := alpha * inv(TriL(A))*B */
        if (alpha != one) {
            for (i = 0; i < n1; i++) {
                for (j = 0; j < n2; j++) {
                    b[ldb * i + j] *= alpha;
                }
            }
        }
        for (i = 0; i < n1; i++) {
            if (nonunit) {
                X Aii = a[lda * i + i];
                for (j = 0; j < n2; j++) {
                    b[ldb * i + j] /= Aii;
                }
            }
            for (k = i + 1; k < n1; k++) {
                const X Aki = a[k * lda + i];
                for (j = 0; j < n2; j++) {
                    b[ldb * k + j] -= Aki * b[ldb * i + j];
                }
            }
        }
    } else if (side_ == CblasLeft && uplo_ == CblasLower && trans == CblasTrans) {
        /* form  B := alpha * TriL(A)' *B */
        if (alpha != one) {
            for (i = 0; i < n1; i++) {
                for (j = 0; j < n2; j++) {
                    b[ldb * i + j] *= alpha;
                }
            }
        }
        for (i = n1; i > 0 && i--;) {
            if (nonunit) {
                X Aii = a[lda * i + i];
                for (j = 0; j < n2; j++) {
                    b[ldb * i + j] /= Aii;
                }
            }
            
            for (k = 0; k < i; k++) {
                const X Aik = a[i * lda + k];
                for (j = 0; j < n2; j++) {
                    b[ldb * k + j] -= Aik * b[ldb * i + j];
                }
            }
        }
    } else if (side_ == CblasRight && uplo_ == CblasUpper && trans == CblasNoTrans) {
        /* form  B := alpha * B * inv(TriU(A)) */
        if (alpha != one) {
            for (i = 0; i < n1; i++) {
                for (j = 0; j < n2; j++) {
                    b[ldb * i + j] *= alpha;
                }
            }
        }
        for (i = 0; i < n1; i++) {
            for (j = 0; j < n2; j++) {
                if (nonunit) {
                    X Ajj = a[lda * j + j];
                    b[ldb * i + j] /= Ajj;
                }
                
                X Bij = b[ldb * i + j];
                for (k = j + 1; k < n2; k++) {
                    b[ldb * i + k] -= a[j * lda + k] * Bij;
                }
            }
        }
    } else if (side_ == CblasRight && uplo_ == CblasUpper && trans == CblasTrans) {
        /* form  B := alpha * B * inv(TriU(A))' */
        if (alpha != one) {
            for (i = 0; i < n1; i++) {
                for (j = 0; j < n2; j++) {
                    b[ldb * i + j] *= alpha;
                }
            }
        }
        
        for (i = 0; i < n1; i++) {
            for (j = n2; j > 0 && j--;) {
                if (nonunit) {
                    X Ajj = a[lda * j + j];
                    b[ldb * i + j] /= Ajj;
                }
                X Bij = b[ldb * i + j];
                for (k = 0; k < j; k++) {
                    b[ldb * i + k] -= a[k * lda + j] * Bij;
                }
            }
        }
    } else if (side_ == CblasRight && uplo_ == CblasLower && trans == CblasNoTrans) {
        /* form  B := alpha * B * inv(TriL(A)) */
        if (alpha != one) {
            for (i = 0; i < n1; i++) {
                for (j = 0; j < n2; j++) {
                    b[ldb * i + j] *= alpha;
                }
            }
        }
        for (i = 0; i < n1; i++) {
            for (j = n2; j > 0 && j--;) {
                if (nonunit) {
                    X Ajj = a[lda * j + j];
                    b[ldb * i + j] /= Ajj;
                }
                
                X Bij = b[ldb * i + j];
                for (k = 0; k < j; k++) {
                    b[ldb * i + k] -= a[j * lda + k] * Bij;
                }
            }
        }
    } else if (side_ == CblasRight && uplo_ == CblasLower && trans == CblasTrans) {
        /* form  B := alpha * B * inv(TriL(A))' */
        if (alpha != one) {
            for (i = 0; i < n1; i++) {
                for (j = 0; j < n2; j++) {
                    b[ldb * i + j] *= alpha;
                }
            }
        }
        for (i = 0; i < n1; i++) {
            for (j = 0; j < n2; j++) {
                if (nonunit) {
                    X Ajj = a[lda * j + j];
                    b[ldb * i + j] /= Ajj;
                }
                
                X Bij = b[ldb * i + j];
                for (k = j + 1; k < n2; k++) {
                    b[ldb * i + k] -= a[k * lda + j] * Bij;
                }
            }
        }
    } else {
        assert(0, "unrecognized operation");
    }
}








