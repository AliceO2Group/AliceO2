# include <stdio.h>

# include "MCHClustering/dataStructure.h"

// theta
double *getVarX( double *theta, int K ) { return &theta[0*K]; };
double *getVarY( double *theta, int K ) { return &theta[1*K]; };
double *getMuX ( double *theta, int K ) { return &theta[2*K]; };
double *getMuY ( double *theta, int K ) { return &theta[3*K]; };
double *getW   ( double *theta, int K ) { return &theta[4*K]; };
double *getMuAndW( double *theta, int K ) { return &theta[2*K]; };
//
const double *getConstVarX( const double *theta, int K ) { return &theta[0*K]; };
const double *getConstVarY( const double *theta, int K ) { return &theta[1*K]; };
const double *getConstMuX ( const double *theta, int K ) { return &theta[2*K]; };
const double *getConstMuY ( const double *theta, int K ) { return &theta[3*K]; };
const double *getConstW   ( const double *theta, int K ) { return &theta[4*K]; };
const double *getConstMuAndW( const double *theta, int K ) { return &theta[2*K]; };

// xyDxy
double *getX( double *xyDxy, int N ) { return &xyDxy[0*N]; };
double *getY( double *xyDxy, int N ) { return &xyDxy[1*N]; };
double *getDX( double *xyDxy, int N ) { return &xyDxy[2*N]; };
double *getDY( double *xyDxy, int N ) { return &xyDxy[3*N]; };
//
const double *getConstX( const double *xyDxy, int N ) { return &xyDxy[0*N]; };
const double *getConstY( const double *xyDxy, int N ) { return &xyDxy[1*N]; };
const double *getConstDX( const double *xyDxy, int N ) { return &xyDxy[2*N]; };
const double *getConstDY( const double *xyDxy, int N ) { return &xyDxy[3*N]; };

// xySupInf
double *getXInf( double *xyInfSup, int N ) { return &xyInfSup[0*N]; };
double *getYInf( double *xyInfSup, int N ) { return &xyInfSup[1*N]; };
double *getXSup( double *xyInfSup, int N ) { return &xyInfSup[2*N]; };
double *getYSup( double *xyInfSup, int N ) { return &xyInfSup[3*N]; };
const double *getConstXInf( const double *xyInfSup, int N ) { return &xyInfSup[0*N]; };
const double *getConstYInf( const double *xyInfSup, int N ) { return &xyInfSup[1*N]; };
const double *getConstXSup( const double *xyInfSup, int N ) { return &xyInfSup[2*N]; };
const double *getConstYSup( const double *xyInfSup, int N ) { return &xyInfSup[3*N]; };

void copyTheta( const double *theta0, int K0, double *theta, int K1, int K) {
  const double *w    = getConstW   ( theta0, K0);
  const double *muX  = getConstMuX ( theta0, K0);
  const double *muY  = getConstMuY ( theta0, K0);
  const double *varX = getConstVarX( theta0, K0);
  const double *varY = getConstVarY( theta0, K0);
  double *wm    = getW   ( theta, K1);
  double *muXm  = getMuX ( theta, K1);
  double *muYm  = getMuY ( theta, K1);
  double *varXm = getVarX( theta, K1);
  double *varYm = getVarY( theta, K1);
  vectorCopy( w,    K, wm );
  vectorCopy( muX,  K, muXm );
  vectorCopy( muY,  K, muYm );
  vectorCopy( varX, K, varXm );
  vectorCopy( varY, K, varYm );
}

void copyXYdXY( const double *xyDxy0, int N0, double *xyDxy, int N1, int N) {
  const double *X0  = getConstX ( xyDxy0, N0);
  const double *Y0  = getConstY ( xyDxy0, N0);
  const double *DX0 = getConstDX( xyDxy0, N0);
  const double *DY0 = getConstDY( xyDxy0, N0);
  
  double *X  = getX ( xyDxy, N1);
  double *Y  = getY ( xyDxy, N1);
  double *DX = getDX( xyDxy, N1);
  double *DY = getDY( xyDxy, N1);

  vectorCopy( X0,  N, X );
  vectorCopy( Y0,  N, Y );
  vectorCopy( DX0, N, DX );
  vectorCopy( DY0, N, DY );
}

void printTheta( const char *str, const double *theta, int K) {
  const double *varX = getConstVarX(theta, K);
  const double *varY = getConstVarY(theta, K);
  const double *muX  = getConstMuX(theta, K);
  const double *muY  = getConstMuY(theta, K);
  const double *w    = getConstW(theta, K);

  printf("%s \n", str);
  printf("    k        w      muX       muY          sigX      sigY\n");
  for (int k=0; k < K; k++) {
    printf("  %.2d-th: %7.4g %9.4g %9.4g %9.4g %9.4g\n", 
            k, w[k], muX[k], muY[k], sqrt(varX[k]), sqrt(varY[k]) );    
  }
}
void xyDxyToxyInfSup( const double *xyDxy, int nxyDxy, 
                           double *xyInfSup ) {
  const double *X  = getConstX ( xyDxy, nxyDxy);
  const double *Y  = getConstY ( xyDxy, nxyDxy);
  const double *DX = getConstDX( xyDxy, nxyDxy);
  const double *DY = getConstDY( xyDxy, nxyDxy);
  double *XInf = getXInf( xyInfSup, nxyDxy);
  double *XSup = getXSup( xyInfSup, nxyDxy);
  double *YInf = getYInf( xyInfSup, nxyDxy);
  double *YSup = getYSup( xyInfSup, nxyDxy);
  for (int k=0; k < nxyDxy; k++) {
    // To avoid overwritting
    double xInf = X[k] - DX[k];
    double xSup = X[k] + DX[k];
    double yInf = Y[k] - DY[k];
    double ySup = Y[k] + DY[k];
    //
    XInf[k] = xInf; YInf[k] = yInf;  
    XSup[k] = xSup; YSup[k] = ySup;  
  }
}

// Mask operations
void maskedCopyXYdXY( const double *xyDxy,  int nxyDxy, const Mask_t *mask,  int nMask, 
                      double *xyDxyMasked, int nxyDxyMasked) {
  const double *X  = getConstX ( xyDxy, nxyDxy);
  const double *Y  = getConstY ( xyDxy, nxyDxy);
  const double *DX = getConstDX( xyDxy, nxyDxy);
  const double *DY = getConstDY( xyDxy, nxyDxy);
  double *Xm  = getX ( xyDxyMasked, nxyDxyMasked);
  double *Ym  = getY ( xyDxyMasked, nxyDxyMasked);
  double *DXm = getDX( xyDxyMasked, nxyDxyMasked);
  double *DYm = getDY( xyDxyMasked, nxyDxyMasked);
  vectorGather( X, mask, nMask, Xm );
  vectorGather( Y, mask, nMask, Ym );
  vectorGather( DX, mask, nMask, DXm );
  vectorGather( DY, mask, nMask, DYm );
}

void maskedCopyToXYInfSup( const double *xyDxy, int ndxyDxy,  const Mask_t *mask, int nMask, 
                           double *xyDxyMasked, int ndxyDxyMasked ) {
  const double *X  = getConstX ( xyDxy, ndxyDxy);
  const double *Y  = getConstY ( xyDxy, ndxyDxy);
  const double *DX = getConstDX( xyDxy, ndxyDxy);
  const double *DY = getConstDY( xyDxy, ndxyDxy);
  double *Xm  = getX ( xyDxyMasked, ndxyDxyMasked);
  double *Ym  = getY ( xyDxyMasked, ndxyDxyMasked);
  double *DXm = getDX( xyDxyMasked, ndxyDxyMasked);
  double *DYm = getDY( xyDxyMasked, ndxyDxyMasked);
  double *XmInf = getXInf( xyDxyMasked, ndxyDxyMasked);
  double *XmSup = getXSup( xyDxyMasked, ndxyDxyMasked);
  double *YmInf = getYInf( xyDxyMasked, ndxyDxyMasked);
  double *YmSup = getYSup( xyDxyMasked, ndxyDxyMasked);
  vectorGather( X,  mask, nMask, Xm );
  vectorGather( Y,  mask, nMask, Ym );
  vectorGather( DX, mask, nMask, DXm );
  vectorGather( DY, mask, nMask, DYm );
  for (int k=0; k < nMask; k++) {
    // To avoid overwritting
    double xInf = Xm[k] - DXm[k];
    double xSup = Xm[k] + DXm[k];
    double yInf = Ym[k] - DYm[k];
    double ySup = Ym[k] + DYm[k];
    //
    XmInf[k] = xInf; YmInf[k] = yInf;  
    XmSup[k] = xSup; YmSup[k] = ySup;  
  }
}

void maskedCopyTheta( const double *theta, int K, const Mask_t *mask, int nMask , double *maskedTheta, int maskedK) {
  const double *w    = getConstW   ( theta, K);
  const double *muX  = getConstMuX ( theta, K);
  const double *muY  = getConstMuY ( theta, K);
  const double *varX = getConstVarX( theta, K);
  const double *varY = getConstVarY( theta, K);
  double *wm    = getW   ( maskedTheta, maskedK);
  double *muXm  = getMuX ( maskedTheta, maskedK);
  double *muYm  = getMuY ( maskedTheta, maskedK);
  double *varXm = getVarX( maskedTheta, maskedK);
  double *varYm = getVarY( maskedTheta, maskedK);
  vectorGather( w,    mask, nMask, wm );
  vectorGather( muX,  mask, nMask, muXm );
  vectorGather( muY,  mask, nMask, muYm );
  vectorGather( varX, mask, nMask, varXm );
  vectorGather( varY, mask, nMask, varYm );
}
    
void printXYdXY( const char *str, const double *xyDxy, int NMax, int N, const double *val1, const double *val2) {
  const double *X = getConstX( xyDxy, NMax);
  const double *Y = getConstY( xyDxy, NMax);
  const double *DX = getConstDX( xyDxy, NMax);
  const double *DY = getConstDY( xyDxy, NMax);
 
  printf("%s\n", str);
  int nPrint = 0;
  if (val1 != 0) nPrint++;
  if (val2 != 0) nPrint++;
  if ((nPrint == 1) && (val2 != 0)) val1 = val2;
  
  if ( nPrint == 0) {
    for( PadIdx_t i=0; i < N; i++) {
      printf("  pad %2d: %9.3g %9.3g %9.3g %9.3g \n", i, X[i], Y[i], DX[i], DY[i]);
    }
  } else if( nPrint == 1) {
    for( PadIdx_t i=0; i < N; i++) {
      printf("  pad %2d: %9.3g %9.3g %9.3g %9.3g - %9.3g \n", i, X[i], Y[i], DX[i], DY[i], val1[i]);  
    }
  } else {
    for( PadIdx_t i=0; i < N; i++) {
      printf("  pad %d: %9.3g %9.3g %9.3g %9.3g - %9.3g %9.3g \n", i, X[i], Y[i], DX[i], DY[i], val1[i], val2[i] );  
    }
  }
}