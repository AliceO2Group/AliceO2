//-*- Mode: C++ -*-
// $Id: AliHLTTPCCATrackParam2.h 35151 2009-10-01 13:35:10Z sgorbuno $
// ************************************************************************
// This file is property of and copyright by the ALICE HLT Project        *
// ALICE Experiment at CERN, All rights reserved.                         *
// See cxx source for full Copyright notice                               *
//                                                                        *
//*************************************************************************


#ifndef ALIHLTTPCCATRACKPARAM2_H
#define ALIHLTTPCCATRACKPARAM2_H

#include "AliHLTTPCCADef.h"

class AliHLTTPCCATrackLinearisation;

/**
 * @class AliHLTTPCCATrackParam
 *
 * AliHLTTPCCATrackParam class describes the track parametrisation
 * which is used by the AliHLTTPCCATracker slice tracker.
 * This class is used for transfer between tracker and merger and does not contain the covariance matrice
 */
class AliHLTTPCCATrackParam2
{
  public:

    GPUd() float X()      const { return fX;    }
    GPUd() float Y()      const { return fP[0]; }
    GPUd() float Z()      const { return fP[1]; }
    GPUd() float SinPhi() const { return fP[2]; }
    GPUd() float DzDs()   const { return fP[3]; }
    GPUd() float QPt()    const { return fP[4]; }

    GPUd() float GetX()      const { return fX; }
    GPUd() float GetY()      const { return fP[0]; }
    GPUd() float GetZ()      const { return fP[1]; }
    GPUd() float GetSinPhi() const { return fP[2]; }
    GPUd() float GetDzDs()   const { return fP[3]; }
    GPUd() float GetQPt()    const { return fP[4]; }

    GPUd() float GetKappa( float Bz ) const { return -fP[4]*Bz; }

    GPUhd() const float *Par() const { return fP; }
    GPUd() const float *GetPar() const { return fP; }
	GPUd() float GetPar(int i) const { return(fP[i]); }

    GPUhd() void SetPar( int i, float v ) { fP[i] = v; }

    GPUd() void SetX( float v )     {  fX = v;    }
    GPUd() void SetY( float v )     {  fP[0] = v; }
    GPUd() void SetZ( float v )     {  fP[1] = v; }
    GPUd() void SetSinPhi( float v ) {  fP[2] = v; }
    GPUd() void SetDzDs( float v )  {  fP[3] = v; }
    GPUd() void SetQPt( float v )   {  fP[4] = v; }

  private:
	//WARNING, Track Param Data is copied in the GPU Tracklet Constructor element by element instead of using copy constructor!!!
	//This is neccessary for performance reasons!!!
	//Changes to Elements of this class therefore must also be applied to TrackletConstructor!!!
    float fX;      // x position
    float fP[5];   // 'active' track parameters: Y, Z, SinPhi, DzDs, q/Pt
};

#endif
