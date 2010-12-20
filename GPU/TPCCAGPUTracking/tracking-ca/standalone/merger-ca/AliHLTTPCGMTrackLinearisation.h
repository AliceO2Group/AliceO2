// ************************************************************************
// This file is property of and copyright by the ALICE HLT Project        *
// ALICE Experiment at CERN, All rights reserved.                         *
// See cxx source for full Copyright notice                               *
//                                                                        *
//*************************************************************************


#ifndef ALIHLTTPCGMTRACKLINEARISATION_H
#define ALIHLTTPCGMTRACKLINEARISATION_H

#include "AliHLTTPCGMTrackParam.h"


/**
 * @class AliHLTTPCGMTrackLinearisation
 *
 * AliHLTTPCGMTrackLinearisation class describes the parameters which are used
 * to linearise the transport equations for the track trajectory.
 *
 * The class is used during track (re)fit, when the AliHLTTPCTrackParam track is only
 * partially fitted, and there is some apriory knowledge about trajectory.
 * This apriory knowledge is used to linearise the transport equations.
 *
 * In case the track is fully fitted, the best linearisation point is
 * the track trajectory itself (AliHLTTPCGMTrackLinearisation = AliHLTTPCGMTrackParam ).
 *
 */
class AliHLTTPCGMTrackLinearisation
{

  public:

    AliHLTTPCGMTrackLinearisation()
      : fSinPhi( 0. ), fCosPhi( 1. ), fSecPhi( 1. ), fDzDs( 0. ), fDlDs( 0.), fQPt( 0. ) {}

    AliHLTTPCGMTrackLinearisation( float SinPhi1, float CosPhi1, float SecPhi1, float DzDs1, float DlDs1, float QPt1 )
      : fSinPhi( SinPhi1 ), fCosPhi( CosPhi1 ), fSecPhi( SecPhi1 ), fDzDs( DzDs1 ), fDlDs( DlDs1), fQPt( QPt1 ) {}

    AliHLTTPCGMTrackLinearisation( const AliHLTTPCGMTrackParam &t );

    void Set( float SinPhi1, float CosPhi1, float SecPhi1, float DzDs1, float DlDs1, float QPt1 );

    
    float& SinPhi() { return fSinPhi; }
    float& CosPhi() { return fCosPhi; }
    float& SecPhi() { return fSecPhi; }
    float& DzDs()   { return fDzDs; }
    float& DlDs()   { return fDlDs; }
    float& QPt()    { return fQPt; }


 private:
    float fSinPhi; // SinPhi
    float fCosPhi; // CosPhi
    float fSecPhi; // 1/cos(phi)
    float fDzDs;   // DzDs
    float fDlDs;   // DlDs
    float fQPt;    // QPt
};


 inline AliHLTTPCGMTrackLinearisation::AliHLTTPCGMTrackLinearisation( const AliHLTTPCGMTrackParam &t )
    : fSinPhi( t.GetSinPhi() ), fCosPhi( 0. ), fSecPhi( 0. ), fDzDs( t.GetDzDs() ), fDlDs( 0. ), fQPt( t.GetQPt() )
{
  fSinPhi = min( fSinPhi,  .999f );
  fSinPhi = max( fSinPhi, -.999f );
  fCosPhi = sqrt( 1. - fSinPhi * fSinPhi );
  fSecPhi = 1./fCosPhi; //reciprocal(fCosPhi);
  fDlDs = sqrt(1.+fDzDs*fDzDs);
}


 inline void AliHLTTPCGMTrackLinearisation::Set( float SinPhi1, float CosPhi1, float SecPhi1,
    float DzDs1, float DlDs1, float QPt1 )
{
  fSinPhi = SinPhi1 ;
  fCosPhi = CosPhi1 ;
  fSecPhi = SecPhi1 ;
  fDzDs = DzDs1;
  fDlDs = DlDs1;
  fQPt = QPt1;
}


#endif //ALIHLTTPCGMTRACKLINEARISATION_H
