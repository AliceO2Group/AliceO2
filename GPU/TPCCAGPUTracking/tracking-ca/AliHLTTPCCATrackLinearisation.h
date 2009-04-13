// ************************************************************************
// This file is property of and copyright by the ALICE HLT Project        * 
// ALICE Experiment at CERN, All rights reserved.                         *
// See cxx source for full Copyright notice                               *
//                                                                        *
//*************************************************************************


#ifndef ALIHLTTPCCATRACKLINEARISATION_H
#define ALIHLTTPCCATRACKLINEARISATION_H

#include "AliHLTTPCCATrackParam.h"


/**
 * @class AliHLTTPCCATrackLinearisation
 *
 * AliHLTTPCCATrackLinearisation class describes the parameters which are used 
 * to linearise the transport equations for the track trajectory.
 *
 * The class is used during track (re)fit, when the AliHLTTPCTrackParam track is only 
 * partially fitted, and there is some apriory knowledge about trajectory. 
 * This apriory knowledge is used to linearise the transport equations. 
 *
 * In case the track is fully fitted, the best linearisation point is  
 * the track trajectory itself (AliHLTTPCCATrackLinearisation = AliHLTTPCTrackParam ).
 *
 */
class AliHLTTPCCATrackLinearisation
{
 public:

  AliHLTTPCCATrackLinearisation()
    :fSinPhi( 0 ), fCosPhi( 1 ), fDzDs( 0 ), fQPt( 0 ){}

  AliHLTTPCCATrackLinearisation( Float_t SinPhi1, Float_t CosPhi1, Float_t DzDs1, Float_t QPt1 )
    : fSinPhi( SinPhi1 ), fCosPhi( CosPhi1 ), fDzDs( DzDs1 ), fQPt( QPt1 )
    {}
  
  AliHLTTPCCATrackLinearisation( const AliHLTTPCCATrackParam &t );
											     
  GPUd() void Set( Float_t SinPhi1, Float_t CosPhi1, Float_t DzDs1, Float_t QPt1 );


  GPUd() Float_t SinPhi()const { return fSinPhi; }
  GPUd() Float_t CosPhi()const { return fCosPhi; }
  GPUd() Float_t DzDs()  const { return fDzDs; }
  GPUd() Float_t QPt()   const { return fQPt; }

  GPUd() Float_t GetSinPhi()const { return fSinPhi; }
  GPUd() Float_t GetCosPhi()const { return fCosPhi; }
  GPUd() Float_t GetDzDs()  const { return fDzDs; }
  GPUd() Float_t GetQPt()   const { return fQPt; }

  GPUd() void SetSinPhi( Float_t v ){  fSinPhi = v; }
  GPUd() void SetCosPhi( Float_t v ){  fCosPhi = v; }
  GPUd() void SetDzDs( Float_t v )  {  fDzDs   = v; }
  GPUd() void SetQPt( Float_t v )   {  fQPt = v; }

private:

  Float_t fSinPhi; // SinPhi
  Float_t fCosPhi; // CosPhi
  Float_t fDzDs;   // DzDs
  Float_t fQPt;    // QPt
};


inline AliHLTTPCCATrackLinearisation::AliHLTTPCCATrackLinearisation( const AliHLTTPCCATrackParam &t )
  : fSinPhi(t.SinPhi()), fCosPhi(0), fDzDs(t.DzDs()), fQPt( t.QPt() )
{
  if( fSinPhi > .999 ) fSinPhi = .999;
  else if( fSinPhi < -.999 ) fSinPhi = -.999;
  fCosPhi = CAMath::Sqrt(1 - fSinPhi*fSinPhi);
  if( t.SignCosPhi()<0 ) fCosPhi = -fCosPhi;
}
				

GPUd() inline void AliHLTTPCCATrackLinearisation::Set( Float_t SinPhi1, Float_t CosPhi1, 
						  Float_t DzDs1, Float_t QPt1 )
{
  SetSinPhi( SinPhi1 ); 
  SetCosPhi( CosPhi1 ); 
  SetDzDs( DzDs1 ); 
  SetQPt( QPt1 );
}

#endif
