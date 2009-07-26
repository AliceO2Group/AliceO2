//-*- Mode: C++ -*-
// $Id: AliHLTTPCCATrackParam.h 33907 2009-07-23 13:52:49Z sgorbuno $
// ************************************************************************
// This file is property of and copyright by the ALICE HLT Project        *
// ALICE Experiment at CERN, All rights reserved.                         *
// See cxx source for full Copyright notice                               *
//                                                                        *
//*************************************************************************


#ifndef ALIHLTTPCCATRACKPARAM_H
#define ALIHLTTPCCATRACKPARAM_H

#include "AliHLTTPCCADef.h"
#include "AliHLTTPCCAMath.h"

class AliHLTTPCCATrackLinearisation;

/**
 * @class AliHLTTPCCATrackParam
 *
 * AliHLTTPCCATrackParam class describes the track parametrisation
 * which is used by the AliHLTTPCCATracker slice tracker.
 *
 */
class AliHLTTPCCATrackParam
{
  public:

    struct AliHLTTPCCATrackFitParam {
      float fBethe, fE, fTheta2, fEP2, fSigmadE2, fK22, fK33, fK43, fK44;// parameters
    };

    GPUd() float X()      const { return fX;    }
    GPUd() float Y()      const { return fP[0]; }
    GPUd() float Z()      const { return fP[1]; }
    GPUd() float SinPhi() const { return fP[2]; }
    GPUd() float DzDs()   const { return fP[3]; }
    GPUd() float QPt()    const { return fP[4]; }
    GPUd() float SignCosPhi() const { return fSignCosPhi; }
    GPUd() float Chi2()  const { return fChi2; }
    GPUd() int   NDF()   const { return fNDF; }

    float Err2Y()      const { return fC[0]; }
    float Err2Z()      const { return fC[2]; }
    float Err2SinPhi() const { return fC[5]; }
    float Err2DzDs()   const { return fC[9]; }
    float Err2QPt()    const { return fC[14]; }

    GPUd() float GetX()      const { return fX; }
    GPUd() float GetY()      const { return fP[0]; }
    GPUd() float GetZ()      const { return fP[1]; }
    GPUd() float GetSinPhi() const { return fP[2]; }
    GPUd() float GetDzDs()   const { return fP[3]; }
    GPUd() float GetQPt()    const { return fP[4]; }
    GPUd() float GetSignCosPhi() const { return fSignCosPhi; }
    GPUd() float GetChi2()   const { return fChi2; }
    GPUd() int   GetNDF()    const { return fNDF; }

    GPUd() float GetKappa( float Bz ) const { return fP[4]*Bz; }
    GPUd() float GetCosPhi() const { return fSignCosPhi*CAMath::Sqrt( 1 - SinPhi()*SinPhi() ); }

    GPUd() float GetErr2Y()      const { return fC[0]; }
    GPUd() float GetErr2Z()      const { return fC[2]; }
    GPUd() float GetErr2SinPhi() const { return fC[5]; }
    GPUd() float GetErr2DzDs()   const { return fC[9]; }
    GPUd() float GetErr2QPt()    const { return fC[14]; }

    GPUhd() const float *Par() const { return fP; }
    GPUhd() const float *Cov() const { return fC; }

    const float *GetPar() const { return fP; }
    const float *GetCov() const { return fC; }

    GPUhd() void SetPar( int i, float v ) { fP[i] = v; }
    GPUhd() void SetCov( int i, float v ) { fC[i] = v; }

    GPUd() void SetX( float v )     {  fX = v;    }
    GPUd() void SetY( float v )     {  fP[0] = v; }
    GPUd() void SetZ( float v )     {  fP[1] = v; }
    GPUd() void SetSinPhi( float v ) {  fP[2] = v; }
    GPUd() void SetDzDs( float v )  {  fP[3] = v; }
    GPUd() void SetQPt( float v )   {  fP[4] = v; }
    GPUd() void SetSignCosPhi( float v ) {  fSignCosPhi = v >= 0 ? 1 : -1; }
    GPUd() void SetChi2( float v )  {  fChi2 = v; }
    GPUd() void SetNDF( int v )   { fNDF = v; }


    GPUd() float GetDist2( const AliHLTTPCCATrackParam &t ) const;
    GPUd() float GetDistXZ2( const AliHLTTPCCATrackParam &t ) const;


    GPUd() float GetS( float x, float y, float Bz  ) const;

    GPUd() void GetDCAPoint( float x, float y, float z,
                             float &px, float &py, float &pz, float Bz  ) const;


    GPUd() bool TransportToX( float x, float Bz, float maxSinPhi = .999 );
    GPUd() bool TransportToXWithMaterial( float x, float Bz, float maxSinPhi = .999 );

    GPUd() bool  TransportToX( float x, AliHLTTPCCATrackLinearisation &t0,
                               float Bz,  float maxSinPhi = .999, float *DL = 0 );

    GPUd() bool  TransportToX( float x, float sinPhi0, float cosPhi0,  float Bz, float maxSinPhi = .999 );


    GPUd() bool  TransportToXWithMaterial( float x,  AliHLTTPCCATrackLinearisation &t0,
                                           AliHLTTPCCATrackFitParam &par, float Bz, float maxSinPhi = .999 );

    GPUd() bool  TransportToXWithMaterial( float x,
                                           AliHLTTPCCATrackFitParam &par, float Bz, float maxSinPhi = .999 );



    GPUd() static float ApproximateBetheBloch( float beta2 );
    GPUd() static float BetheBlochGeant( float bg,
                                         float kp0 = 2.33,
                                         float kp1 = 0.20,
                                         float kp2 = 3.00,
                                         float kp3 = 173e-9,
                                         float kp4 = 0.49848
                                       );
    GPUd() static float BetheBlochSolid( float bg );
    GPUd() static float BetheBlochGas( float bg );


    GPUd() void CalculateFitParameters( AliHLTTPCCATrackFitParam &par, float mass = 0.13957 );
    GPUd() bool CorrectForMeanMaterial( float xOverX0,  float xTimesRho, const AliHLTTPCCATrackFitParam &par );

    GPUd() bool Rotate( float alpha, float maxSinPhi = .999 );
    GPUd() bool Rotate( float alpha, AliHLTTPCCATrackLinearisation &t0, float maxSinPhi = .999 );
    GPUd() bool Filter( float y, float z, float err2Y, float err2Z, float maxSinPhi = .999 );

    GPUd() bool CheckNumericalQuality() const;

    GPUd() void Print() const;

#ifndef CUDA_DEVICE_EMULATION
  private:
#endif

    float fX;      // x position
    float fSignCosPhi; // sign of cosPhi
    float fP[5];   // 'active' track parameters: Y, Z, SinPhi, DzDs, q/Pt
    float fC[15];  // the covariance matrix for Y,Z,SinPhi,..
    float fChi2;   // the chi^2 value
    int   fNDF;    // the Number of Degrees of Freedom
};

#endif
