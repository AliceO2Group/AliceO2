//-*- Mode: C++ -*-
// $Id: AliHLTTPCCATrackParam.h 35151 2009-10-01 13:35:10Z sgorbuno $
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
#include "AliHLTTPCCATrackParam2.h"

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

	GPUd() const AliHLTTPCCATrackParam2& GetParam() const { return fParam; }
	GPUd() void SetParam(const AliHLTTPCCATrackParam2& v) { fParam = v; }
	GPUd() void InitParam();

    GPUd() float X()      const { return fParam.X();    }
    GPUd() float Y()      const { return fParam.Y(); }
    GPUd() float Z()      const { return fParam.Z(); }
    GPUd() float SinPhi() const { return fParam.SinPhi(); }
    GPUd() float DzDs()   const { return fParam.DzDs(); }
    GPUd() float QPt()    const { return fParam.QPt(); }
    GPUd() float SignCosPhi() const { return fSignCosPhi; }
    GPUd() float Chi2()  const { return fChi2; }
    GPUd() int   NDF()   const { return fNDF; }

    GPUd() float Err2Y()      const { return fC[0]; }
    GPUd() float Err2Z()      const { return fC[2]; }
    GPUd() float Err2SinPhi() const { return fC[5]; }
    GPUd() float Err2DzDs()   const { return fC[9]; }
    GPUd() float Err2QPt()    const { return fC[14]; }

    GPUd() float GetX()      const { return fParam.GetX(); }
    GPUd() float GetY()      const { return fParam.GetY(); }
    GPUd() float GetZ()      const { return fParam.GetZ(); }
    GPUd() float GetSinPhi() const { return fParam.GetSinPhi(); }
    GPUd() float GetDzDs()   const { return fParam.GetDzDs(); }
    GPUd() float GetQPt()    const { return fParam.GetQPt(); }
    GPUd() float GetSignCosPhi() const { return fSignCosPhi; }
    GPUd() float GetChi2()   const { return fChi2; }
    GPUd() int   GetNDF()    const { return fNDF; }

    GPUd() float GetKappa( float Bz ) const { return fParam.GetKappa(Bz); }
    GPUd() float GetCosPhi() const { return fSignCosPhi*CAMath::Sqrt( 1 - SinPhi()*SinPhi() ); }

    GPUd() float GetErr2Y()      const { return fC[0]; }
    GPUd() float GetErr2Z()      const { return fC[2]; }
    GPUd() float GetErr2SinPhi() const { return fC[5]; }
    GPUd() float GetErr2DzDs()   const { return fC[9]; }
    GPUd() float GetErr2QPt()    const { return fC[14]; }

    GPUhd() const float *Par() const { return fParam.Par(); }
    GPUhd() const float *Cov() const { return fC; }

    GPUd() const float *GetPar() const { return fParam.GetPar(); }
	GPUd() float GetPar(int i) const { return(fParam.GetPar(i)); }
    GPUd() const float *GetCov() const { return fC; }
	GPUd() float GetCov(int i) {return fC[i]; }

    GPUhd() void SetPar( int i, float v ) { fParam.SetPar(i, v); }
    GPUhd() void SetCov( int i, float v ) { fC[i] = v; }

    GPUd() void SetX( float v )     {  fParam.SetX(v);    }
    GPUd() void SetY( float v )     {  fParam.SetY(v); }
    GPUd() void SetZ( float v )     {  fParam.SetZ(v); }
    GPUd() void SetSinPhi( float v ) {  fParam.SetSinPhi(v); }
    GPUd() void SetDzDs( float v )  {  fParam.SetDzDs(v); }
    GPUd() void SetQPt( float v )   {  fParam.SetQPt(v); }
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

  private:
	//WARNING, Track Param Data is copied in the GPU Tracklet Constructor element by element instead of using copy constructor!!!
	//This is neccessary for performance reasons!!!
	//Changes to Elements of this class therefore must also be applied to TrackletConstructor!!!
    float fC[15];  // the covariance matrix for Y,Z,SinPhi,..
    float fSignCosPhi; // sign of cosPhi
    float fChi2;   // the chi^2 value
    int   fNDF;    // the Number of Degrees of Freedom

	AliHLTTPCCATrackParam2 fParam; // Track Parameters
};

GPUd() inline void AliHLTTPCCATrackParam::InitParam()
{
  //Initialize Tracklet Parameters using default values
  SetSinPhi( 0 );
  SetDzDs( 0 );
  SetQPt( 0 );
  SetSignCosPhi( 1 );
  SetChi2( 0 );
  SetNDF( -3 );
  SetCov( 0, 1 );
  SetCov( 1, 0 );
  SetCov( 2, 1 );
  SetCov( 3, 0 );
  SetCov( 4, 0 );
  SetCov( 5, 1 );
  SetCov( 6, 0 );
  SetCov( 7, 0 );
  SetCov( 8, 0 );
  SetCov( 9, 1 );
  SetCov( 10, 0 );
  SetCov( 11, 0 );
  SetCov( 12, 0 );
  SetCov( 13, 0 );
  SetCov( 14, 10. );
}

#endif
