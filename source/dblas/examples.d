/** 
*
* @title  BLAS implementation for D
* @author Chibisi Chima-Okereke
* @date   2017-03-01
*
*/

module test;
public import dblas.l1;
public import dblas.l2;


import std.stdio : writeln;
import std.complex: Complex, complex;

/* To compile: */
/* dub build dblas # or dub run ... */


/* Somewhere to park the code used to run the functions as they were created to
** to make sure that they are running properly.
*/

/* C function for tpmv */
extern (C){
    void cblas_dtpmv (in CBLAS_ORDER order, in CBLAS_UPLO Uplo,
                 in CBLAS_TRANSPOSE TransA, in CBLAS_DIAG Diag,
                 in int N, in double* Ap, double* X, in int incX);
}

/* Testing for tpmv */
void test_tpmv(){
    CBLAS_LAYOUT layout = CblasRowMajor;
    CBLAS_UPLO uplo = CblasUpper;
    CBLAS_TRANSPOSE transA = CblasNoTrans;
    CBLAS_DIAG diag = CblasNonUnit;

    double[] a = [-0.587, 0.14, 0.841];
    double[] x = [-0.213, 0.885];
    tpmv(layout, uplo, transA, diag, 2, a.ptr, x.ptr, -1);
    writeln("tpmv: ", x);
    Complex!double[] ac = [complex(0.254, 0.263), complex(-0.271, -0.595), complex(-0.182, -0.672)];
    Complex!double[] xc = [complex(-0.042, -0.705), complex(-0.255, -0.854)];
    tpmv(layout, uplo, transA, diag, 2, ac.ptr, xc.ptr, -1);
    writeln("tpmv: ", xc);
}


/* C function for ztbsv */
extern (C){
    void cblas_ztbsv (in CBLAS_ORDER order, in CBLAS_UPLO Uplo, in CBLAS_TRANSPOSE TransA, in CBLAS_DIAG Diag,
             in int N, in int K, in void *A, in int lda, void *X, in int incX);
}

/* Testing for ztbsv */
void test_ztbsv(){
    CBLAS_LAYOUT layout = CblasRowMajor;
    CBLAS_UPLO uplo = CblasUpper;
    CBLAS_TRANSPOSE transA = CblasNoTrans;
    CBLAS_DIAG diag = CblasNonUnit;

    Complex!double[] a = [complex(0.474, 0.715), complex(0.061, 0.532), complex(0.004, -0.318), 
        complex(0.37, -0.692), complex(-0.166, 0.039), complex(-0.946, 0.857), complex(-0.922, -0.491),
        complex(0.012, -0.217), complex(-0.674, -0.429)];
    Complex!double[] x = [complex(-0.123, 0.122), complex(0.981, 0.321), complex(0.942, 0.98)];
    tbsv(layout, uplo, transA, diag, 3, 1, a.ptr, 3, x.ptr, -1);
    writeln("tbsv: ", x);
    double[] ar = [-0.681, 0.209, 0.436, -0.369, 0.786, -0.84, 0.86, -0.233, 0.734];
    double[] xr = [-0.305, 0.61, -0.831];
    tbsv(layout, uplo, transA, diag, 3, 1, ar.ptr, 3, xr.ptr, -1);
    writeln("tbsv: ", xr);
}


/* C function for dtbsv */
extern (C){
    void cblas_dtbsv (const CBLAS_ORDER order, const CBLAS_UPLO Uplo,
             const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag,
             const int N, const int K, const double *A, const int lda,
             double *X, const int incX);

}

/* Testing for dtbsv */
void test_dtbsv(){
    CBLAS_LAYOUT layout = CblasRowMajor;
    CBLAS_UPLO uplo = CblasUpper;
    CBLAS_TRANSPOSE transA = CblasNoTrans;
    CBLAS_DIAG diag = CblasNonUnit;

    double[] a = [-0.681, 0.209, 0.436, -0.369, 0.786, -0.84, 0.86, -0.233, 0.734];
    double[] x = [-0.305, 0.61, -0.831];
    tbsv(layout, uplo, transA, diag, 3, 1, a.ptr, 3, x.ptr, -1);
    writeln(x);
}



/* C function for tbmv */
extern (C){
  void cblas_dtbmv(in CBLAS_ORDER order, in CBLAS_UPLO Uplo,
             in CBLAS_TRANSPOSE TransA, in CBLAS_DIAG Diag,
             in int N, in int K, in double *A, in int lda,
             double *X, in int incX);
}

/* Testing for tbmv */
void test_tbmv(){
    CBLAS_LAYOUT layout = CblasRowMajor;
    CBLAS_UPLO uplo = CblasUpper;
    CBLAS_TRANSPOSE trans = CblasNoTrans;
    CBLAS_DIAG diag = CblasNonUnit;
    double alpha = 0.0;
    double[] a = [0.439, -0.484, -0.952, -0.508, 0.381, -0.889, -0.192, -0.279, -0.155];
    double[] x = [-0.089, -0.688, -0.203];
    tbmv(layout, uplo, trans, diag, 3, 1, a.ptr, 3, x.ptr, -1);
    writeln(x);
}


/* C function for syr2 */
extern (C){
    void cblas_dsyr2(in CBLAS_ORDER order, in CBLAS_UPLO Uplo,
             in int N, in double alpha, in double *X, in int incX,
             in double *Y, in int incY, double *A, in int lda);
}

/* Testing for syr2 */
void test_syr2(){
    CBLAS_LAYOUT layout = CblasRowMajor;
    CBLAS_UPLO uplo = CblasUpper;
    double alpha = 0.0;
    double[] a = [0.862];
    double[] x = [0.823];
    double[] y = [0.699];
    syr2(layout, uplo, 1, alpha, x.ptr, -1, y.ptr, -1, a.ptr, 1);
    writeln(a);
}


/* C function for syr */
extern (C){
    void cblas_dsyr(in CBLAS_ORDER order, in CBLAS_UPLO Uplo,
             in int N, in double alpha, in double *X, in int incX,
             double *A, in int lda);
}

/* Testing for syr */
void test_syr(){
    CBLAS_LAYOUT layout = CblasRowMajor;
    CBLAS_UPLO uplo = CblasUpper;
    double alpha = 0.1;
    double[] a = [-0.291];
    double[] x = [0.845];
    syr(layout, uplo, 1, alpha, x.ptr, -1, a.ptr, 1);
    writeln(a);
}


/* C function for spr2 */
extern (C){
    void cblas_dspr2 (in CBLAS_ORDER order, in CBLAS_UPLO Uplo,
             in int N, in double alpha, in double *X, in int incX,
             in double *Y, in int incY, double *Ap);
}

/* Testing for spr2 */
void test_spr2(){
    CBLAS_LAYOUT layout = CblasRowMajor;
    CBLAS_UPLO uplo = CblasUpper;
    double alpha = -1.0;
    double[] ap = [0.493, -0.175, -0.831];
    double[] x = [-0.163, 0.489];
    double[] y = [0.154, 0.769];
    spr2(layout, uplo, 2, alpha, x.ptr, -1, y.ptr, -1, ap.ptr);
    writeln(ap);
}

/* C function for spr */
extern (C){
    void cblas_dspr (in CBLAS_ORDER order, in CBLAS_UPLO Uplo,
                     in int N, in double alpha, in double* X, in int incX,
                     double* Ap);
}

/* Testing for spr */
void test_spr(){
    CBLAS_LAYOUT layout = CblasRowMajor;
    CBLAS_UPLO uplo = CblasUpper;
    double alpha = -0.3;
    double[] ap = [-0.764, -0.257, -0.064];
    double[] x = [0.455, -0.285];
    spr(layout, uplo, 2, alpha, x.ptr, -1, ap.ptr);
    writeln(ap);
}

/* C function for sbmv */
extern (C){
    void cblas_dspmv (in CBLAS_ORDER order, in CBLAS_UPLO Uplo,
                      in int N, in double alpha, in double *Ap,
                      in double *X, in int incX, in double beta, double *Y,
                      in int incY);
}

/* Testing for sbmv */
void test_spmv(){
    CBLAS_LAYOUT layout = CblasRowMajor;
    CBLAS_UPLO uplo = CblasUpper;
    double alpha = 0.1;
    double beta = -0.3;
    double[] a = [-0.174, 0.878, 0.478];
    double[] x = [0.503, 0.313];
    double[] y = [-0.565, -0.109];
    spmv(layout, uplo, 2, alpha, a.ptr, x.ptr, -1, beta, y.ptr, -1);
    writeln(y);
}


/* C function for sbmv */
extern (C){
    void cblas_dsbmv (in CBLAS_ORDER order, in CBLAS_UPLO uplo,
             in int N, in int K, in double alpha, in double *A,
             in int lda, in double *X, in int incX,
             in double beta, double *Y, in int incY);
}

/* Testing for sbmv */
void test_sbmv(){
    CBLAS_LAYOUT layout = CblasRowMajor;
    CBLAS_UPLO uplo = CblasUpper;
    double alpha = 1;
    double beta = 0;
    double[] a = [0.627, -0.312, 0.031, 0.308, 0.323, -0.578, 0.797, 0.545, -0.476];
    double[] x = [-0.542, 0.606, 0.727];
    double[] y = [0.755, 0.268, -0.99];
    sbmv(layout, uplo, 3, 1, alpha, a.ptr, 3, x.ptr, -1, beta, y.ptr, -1);
    writeln(y);
}


/* The C function for blas */
extern (C){
    void cblas_zhpr2(in CBLAS_ORDER order, in CBLAS_UPLO Uplo,
                     in int N, in void *alpha, in void *X, in int incX,
                     in void *Y, in int incY, void *Ap);
}

/* Testing for the hpr2 function */
void test_hpr2(){
    CBLAS_LAYOUT layout = CblasRowMajor;
    CBLAS_UPLO uplo = CblasUpper;
    Complex!double alpha = complex(-1.0, 0.0);
    Complex!double[] ap = [complex(0.159, -0.13)];
    Complex!double[] x = [complex(0.854, 0.851)];
    Complex!double[] y = [complex(0.526, -0.267)];
    hpr2(layout, uplo, 1, alpha, x.ptr, -1, y.ptr, -1, ap.ptr);
    writeln(ap);
}



/* The hpr C function for blas */
extern (C){
    void cblas_zhpr(in CBLAS_ORDER order, in CBLAS_UPLO Uplo, in int N, in double alpha, 
                    in void *X, in int incX, void *Ap);
}

/* Testing for the hpr function ... */
void test_hpr(){
    CBLAS_LAYOUT layout = CblasRowMajor;
    CBLAS_UPLO uplo = CblasUpper;
    double alpha = 0.1;
    Complex!double[] ap = [complex(-0.273, -0.499), complex(-0.305, -0.277), complex(0.238, -0.369)];
    Complex!double[] x = [complex(0.638, -0.905), complex(0.224, 0.182)];
    hpr(layout, uplo, 2, alpha, x.ptr, -1, ap.ptr);
    writeln(ap);
}


/* The hpmv C function from blas */
extern (C){
    void cblas_zhpmv(in CBLAS_ORDER order, in CBLAS_UPLO Uplo, in int N, in void* alpha,
                 in void *Ap, in void *X, in int incX, in void *beta, void *Y, in int incY);
}

/* void hpmv(in CBLAS_ORDER layout, in CBLAS_UPLO uplo, in N n, 
          in X* alpha, in X* ap, in X* x, in N incX, in X* beta, X* y, in N incY) */

void test_hpmv(){
    CBLAS_LAYOUT layout = CblasRowMajor;
    CBLAS_UPLO uplo = CblasUpper;
    Complex!double alphaC = complex(-0.3, 0.1);
    Complex!double betaC = complex(0.0, 1.0);
    Complex!double[] a = [complex(0.339, -0.102), complex(0.908, 0.097), complex(-0.808, 0.236)];
    Complex!double[] x = [complex(0.993, -0.502), complex(-0.653, 0.796)];
    Complex!double[] y = [complex(-0.35, 0.339), complex(-0.269, -0.122)];
    hpmv(layout, uplo, 2, alphaC, a.ptr, x.ptr, -1, betaC, y.ptr, -1);
    writeln(y);
}

/* her2 examples */
/*her2(in CBLAS_ORDER layout, in CBLAS_UPLO uplo,
                in N n, in X* alpha, in X* x, in N incX,
                in X* y, in N incY, X* a, in N lda)*/

void test_her2(){
    CBLAS_LAYOUT layout = CblasRowMajor;
    CBLAS_UPLO uplo = CblasUpper;
    Complex!double alphaC = complex(-1.0, 0.0);
    Complex!double[] a = [complex(-0.821, 0.954)];
    Complex!double[] x = [complex(0.532, 0.802)];
    Complex!double[] y = [complex(0.016, -0.334)];
    her2(layout, uplo, 1, alphaC, x.ptr, -1, y.ptr, -1, a.ptr, 1);
    writeln(a);
}


/* her examples */
extern (C){
    void cblas_zher(in CBLAS_ORDER order, in CBLAS_UPLO uplo, in int N, in double alpha, 
                    in void *X, in int incX, void *A, in int lda);
}

/* void her(in CBLAS_ORDER layout, in CBLAS_UPLO uplo, in N n, in V alpha, in X* x, in N incX, X* a, in N lda)
*/

void test_her(){
    CBLAS_LAYOUT layout = CblasRowMajor;
    CBLAS_UPLO uplo = CblasUpper;
    double alphaD = 1;
    Complex!double betaC = complex(-0.3, 0.1);
    Complex!double[] a = [complex(0.188, 0.856)];
    Complex!double[] x = [complex(-0.832, -0.151)];
    her(layout, uplo, 1, alphaD, x.ptr, -1, a.ptr, 1);
    writeln(a);
}


/* hemv examples */
extern (C){
void cblas_zhemv(in CBLAS_ORDER order, in CBLAS_UPLO uplo, in int N, 
                 in void *alpha, in void *A, in int lda, in void *X,
                 in int incX, in void *beta, void *Y, in int incY);
}

void test_hemv(){
    CBLAS_LAYOUT layout = CblasRowMajor;
    CBLAS_UPLO uplo = CblasUpper;
    Complex!double alphaC = complex(1.0,0.0);
    Complex!double betaC = complex(-0.3,0.1);
    Complex!double[] a = [complex(-0.434,0.837)];
    Complex!double[] x = [complex(0.209,-0.935)];
    Complex!double[] y = [complex(0.346,-0.412)];
    hemv(layout, uplo, 1, alphaC, a.ptr, 1, x.ptr, -1, betaC, y.ptr, -1);
    writeln(y);
}


/* hbmv examples */
extern (C){
    void cblas_zhbmv (in CBLAS_ORDER order, in CBLAS_UPLO Uplo,
               in int N, in int K, in void *alpha, in void *A,
               in int lda, in void *X, in int incX, in void *beta,
               void *Y, in int incY);
}

    /*void hbmv(in CBLAS_LAYOUT layout, in CBLAS_UPLO uplo, in N n, in N k, 
      in X* alpha, in X* a, in N lda, in X* x, in N incX , in X* beta, X* y, in N incY)*/
    //hbmv(order, uplo, N, k, alpha, A, lda, X, incX, beta, Y, incY);
void test_hbmv(){

    CBLAS_LAYOUT layout = CblasRowMajor;
    CBLAS_UPLO uplo = CblasUpper;
    Complex!double[] a = [complex(0.937, -0.035), complex(0.339, 0.847), complex(0.022, 0.153),
                  complex(-0.785, 0.193), complex(-0.731, -0.166), complex(-0.243, -0.319),
                  complex(0.173, -0.24), complex(0.079, -0.058), complex(0.124, 0.445)];
    Complex!double[] x = [complex(-0.093, -0.103), complex(-0.537, -0.151), complex(0.094, 0.954)];
    Complex!double[] y = [complex(0.029, -0.391), complex(-0.256, 0.031), complex(-0.478, 0.098)];
    Complex!double alpha = complex(0.0, 1.0);
    Complex!double beta = complex(-0.3, 0.1);
    hbmv(layout, uplo, 3, 1, alpha, a.ptr, 3, x.ptr, -1, beta, y.ptr, -1);
    writeln(y);
    double[] A = [0.937, -0.035, 0.339, 0.847, 0.022, 0.153, -0.785, 0.193, -0.731, -0.166,
                 -0.243, -0.319, 0.173, -0.24, 0.079, -0.058, 0.124, 0.445];
    double[] X = [-0.093, -0.103, -0.537, -0.151, 0.094, 0.954];
    double[] Y = [0.029, -0.391, -0.256, 0.031, -0.478, 0.098];
    double[] alphaD = [0.0, 1.0];
    double[] betaD = [-0.3, 0.1];
    cblas_zhbmv(layout, uplo, 3, 1, alphaD.ptr, A.ptr, 3, X.ptr, -1, betaD.ptr, Y.ptr, -1);
    writeln(Y);
}


void test_ger(){
  
  /*  void ger(in CBLAS_ORDER layout, in N m, in N n, in X alpha, in X* x, in N incX,
               in X* y, in N incY, X* A, in N lda)
  */
  CBLAS_ORDER layout = CblasRowMajor;
  double[] a = [1, 2, 3, 2, 2, 4, 3, 2, 2, 4, 2, 1];
  double[] x = [3, 2, 1, 4];
  double[] y = [1, 2, 3];
  double alpha = 1;
  //ger(layout, 4, 3, alpha, x.ptr, 1, y.ptr, 1, a.ptr, 3);
  //writeln(a);
  layout = CblasColMajor;
  a = [1, 2, 3, 4, 2, 2, 2, 2, 3, 4, 2, 1];
  y = [1, 0, 2, 0, 3];
  //ger(layout, 4, 3, alpha, x.ptr, 1, y.ptr, 2, a.ptr, 4);
  //writeln(a);
  layout = CblasRowMajor;
  Complex!double[] ac = [complex(1, 2), complex(3, 5), complex(2, 0), 
                         complex(2, 3), complex(7, 9), complex(4, 8),
                         complex(7, 4), complex(1, 4), complex(6, 0),
                         complex(8, 2), complex(2, 5), complex(8, 0),
                         complex(9, 1), complex(3, 6), complex(1, 0)];
  Complex!double[] xc = [complex(1, 2), complex(4, 0), complex(1, 1), complex(3, 4), complex(2, 0)];
  Complex!double[] yc = [complex(1, 2), complex(4, 0), complex(1, -1)];
  Complex!double alphaC = complex(1, 0);
  geru(layout, 5, 3, alphaC, xc.ptr, 1, yc.ptr, 1, ac.ptr, 3);
  writeln(ac);
  ac = [complex(1, 2), complex(3, 5), complex(2, 0), 
                         complex(2, 3), complex(7, 9), complex(4, 8),
                         complex(7, 4), complex(1, 4), complex(6, 0),
                         complex(8, 2), complex(2, 5), complex(8, 0),
                         complex(9, 1), complex(3, 6), complex(1, 0)];
  gerc(layout, 5, 3, alphaC, xc.ptr, 1, yc.ptr, 1, ac.ptr, 3);
  writeln(ac);
}


void test_gemv(){
  CBLAS_LAYOUT layout = CblasRowMajor;
  CBLAS_TRANSPOSE trans = CblasNoTrans;
  double[] a = [1, 2, 3, 2, 2, 4, 3, 2, 2, 4, 2, 1];
  double[] x = [3, 2, 1];
  double[] y = [4, 0, 5, 0, 2, 0, 3];
  double alpha = 1, beta = 1;
  /* gemv(layout, transA, m, n, alpha, a, lda, x, incX, beta, y, incY) */
  gemv(layout, trans, 4, 3, alpha, a.ptr, 3, x.ptr, 1, beta, y.ptr, 2);
  writeln(y);
  Complex!double[] ac = [complex(1, 2), complex(3, 5), complex(2, 0),
                         complex(2, 3), complex(7, 9), complex(4, 8),
                         complex(7, 4), complex(1, 4), complex(6, 0),
                         complex(8, 2), complex(2, 5), complex(8, 0),
                         complex(9, 1), complex(3, 6), complex(1, 0)];
  Complex!double alphaC = complex(1, 0), betaC = complex(1, 0);
  Complex!double[] xc = [complex(1, 2), complex(4, 0), complex(1, 1)];
  Complex!double[] yc = [complex(1, 2), complex(4, 0), complex(1, -1),
                         complex(3, 4), complex(2, 0)];
  gemv(layout, trans, 5, 3, alphaC, ac.ptr, 3, xc.ptr, 1, betaC, yc.ptr, 1);
  writeln(yc);
}


/* void sgbmv(N, X)(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE trans, N m, N n, N kl, N ku, 
                                X alpha, X* a, N lda, X* x, N incx, X beta, X* y, N incy) */

void test_gbmv(){
    // Some L2 Algos
    // /* Example 1 from ESSL GBMV */
    CBLAS_LAYOUT layout = CblasRowMajor;
    CBLAS_TRANSPOSE trans = CblasNoTrans;
    double[] a = [0, 0, 0, 1, 1, 1, 0, 0, 2, 2, 2, 2, 0, 3, 3, 3, 3, 0, 4, 4, 4, 4, 0, 0, 5, 5, 5, 0, 0, 0];
    double[] x = [1, 2, 3, 4];
    double[] y = [1, 2, 3, 4, 5];
    gbmv(layout, trans, 5, 4, 3, 2, 3.0, a.ptr, 6, x.ptr, 1, 2.0, y.ptr, 1);
    writeln(y);
    
    // Column Major example
    layout = CblasColMajor;
    trans = CblasNoTrans;
    a = [0, 0, 1, 2, 0, 1, 2, 3, 1, 2, 3, 4, 2, 3, 4, 5, 3, 4, 5, 0, 4, 5, 0, 0];
    x = [1, 2, 3, 4];
    y = [1, 2, 3, 4, 5];
    gbmv(layout, trans, 5, 4, 3, 2, 3.0, a.ptr, 6, x.ptr, 1, 2.0, y.ptr, 1);
    writeln(y);
    
    /* void sgbmv(layout, trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy) */
    /* Example 1 from ESSL GEMV */
    layout = CblasRowMajor;
    trans = CblasNoTrans;
    double[] A = [1, 2, 3, 2, 2, 4, 3, 2, 2, 4, 2, 1];
    a = row_major(A, 4, 3, 3, 2);
    x = [3.0, 2.0, 1.0];
    y = [4.0, 0.0, 5.0, 0.0, 2.0, 0.0, 3.0];
    gbmv(layout, trans, 4, 3, 3, 2, 1.0, a.ptr, 6, x.ptr, 1, 1.0, y.ptr, 2);
    writeln(y);

      /* Example 3 from ESSL GEMV */
    layout = CBLAS_LAYOUT.CblasRowMajor;
    trans = CBLAS_TRANSPOSE.CblasNoTrans;
    Complex!double[] AC = [complex(1, 2), complex(3, 5), complex(2, 0),
                           complex(2, 3), complex(7, 9), complex(4, 8),
                           complex(7, 4), complex(1, 4), complex(6, 0),
                           complex(8, 2), complex(2, 5), complex(8, 0),
                           complex(9, 1), complex(3, 6), complex(1, 0)];
    Complex!double[] ac = row_major(AC, 5, 3, 4, 2);
    Complex!double[] xc = [complex(1, 2), complex(4, 0), complex(1, 1)];
    Complex!double alphaC = complex(1.0, 0.0);
    Complex!double betaC = complex(1.0, 0.0);
    Complex!double[] yc = [complex(1, 2), complex(4, 0), complex(1, -1), complex(3, 4), complex(2, 0)];
    gbmv(layout, trans, 5, 3, 4, 2, alphaC, ac.ptr, 7, xc.ptr, 1, betaC, yc.ptr, 1);
    writeln(yc);

    /* Example 2 from ESSL GBMV */
    layout = CblasRowMajor;
    trans = CblasNoTrans;
    AC = [complex(1, 1), complex(2, 2), complex(3, 3), complex(4, 4), complex(0, 0),
          complex(1, 1), complex(2, 2), complex(3, 3), complex(4, 4), complex(5, 5),
          complex(1, 1), complex(2, 2), complex(3, 3), complex(4, 4), complex(5, 5),
          complex(0, 0), complex(2, 2), complex(3, 3), complex(4, 4), complex(5, 5)];
    ac = row_major(AC, 4, 5, 2, 3);
    alphaC = complex(1.0, 1.0);
    betaC = complex(10.0, 0.0);
    xc = [complex(1, 2), complex(2, 3), complex(3, 4), complex(4, 5), complex(5, 6)];
    yc = [complex(1, 2), complex(0, 0), complex(2, 3), complex(0, 0), complex(3, 4), 
          complex(0, 0), complex(4, 5), complex(0, 0)];
    gbmv(layout, trans, 4, 5, 2, 3, alphaC, ac.ptr, 6, xc.ptr, 1, betaC, yc.ptr, 2);
    writeln(yc);

    /* Example 2 from ESSL GBMV */
    layout = CblasColMajor;
    trans = CblasConjTrans;
    AC = [complex(1, 1), complex(2, 2), complex(3, 3), complex(4, 4), complex(0, 0),
          complex(1, 1), complex(2, 2), complex(3, 3), complex(4, 4), complex(5, 5),
          complex(1, 1), complex(2, 2), complex(3, 3), complex(4, 4), complex(5, 5),
          complex(0, 0), complex(2, 2), complex(3, 3), complex(4, 4), complex(5, 5)];
    ac = col_major(AC, 5, 4, 3, 2);
    xc = [complex(1, 2), complex(2, 3), complex(3, 4), complex(4, 5), complex(5, 6)];
    yc = [complex(1, 2), complex(0, 0), complex(2, 3), complex(0, 0), complex(3, 4), 
          complex(0, 0), complex(4, 5), complex(0, 0)];
    alphaC = complex(1.0, 1.0);
    betaC = complex(10.0, 0.0);
    gbmv(layout, trans, 5, 4, 3, 2, alphaC, ac.ptr, 6, xc.ptr, 1, betaC, yc.ptr, 2);
    writeln(yc);
    /* Example 4 from ESSL GBMV */
    layout = CblasRowMajor;
    trans = CblasNoTrans;
    A = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4];
    a = row_major(A, 4, 5, 3, 4);
    double alpha = 2, beta = 10;
    x = [1, 2, 3, 4, 5];
    y = [1, 0, 2, 0, 3, 0, 4, 0];
    gbmv(layout, trans, 4, 5, 3, 4, alpha, a.ptr, 8, x.ptr, 1, beta, y.ptr, 2);
    writeln(y);
}


/* void sgbmv(N, X)(layout, trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy) */


/*
void test_l1()
{
  // Some L1 algos
  double[] x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
  auto y = [complex(1.0, 1.5), complex(2.0, 2.5), complex(3.0, 3.5), complex(4.0, 4.5), 
           complex(5.0, 5.5), complex(6.0, 6.5), complex(7.0, 7.5)];
  scal(x.length, 3.0, x.ptr, 1);
  scal(y.length, 3.0, y.ptr, 1);
  writeln("scal output:", x);
  writeln("scal output:", y);
  writeln("asum output: ", asum(x.length, x.ptr, 1));
  writeln("asum output: ", asum(y.length, y.ptr, 1));
}
*/
