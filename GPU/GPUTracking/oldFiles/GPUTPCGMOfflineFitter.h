//-*- Mode: C++ -*-

// ************************************************************************
// This file is property of and copyright by the ALICE HLT Project        *
// ALICE Experiment at CERN, All rights reserved.                         *
// See cxx source for full Copyright notice                               *
//                                                                        *
//*************************************************************************


#ifndef GPUTPCGMOfflineFitter_H
#define GPUTPCGMOfflineFitter_H

#if ( defined(GPUCA_ALIROOT_LIB) && !defined(GPUCA_GPUCODE) )


#include "GPUParam.h"
#include "AliTPCtracker.h"

class GPUTPCGMMergedTrack;
class GPUTPCGMMergedTrackHit;
class AliTPCclusterMI;
class GPUTPCGMPolynomialField;

class GPUTPCGMOfflineFitter :public AliTPCtracker
{
public:

  GPUTPCGMOfflineFitter();
  ~GPUTPCGMOfflineFitter();
  
  void Initialize( const GPUParam& hltParam, Long_t TimeStamp, bool isMC );
  
  void RefitTrack(  GPUTPCGMMergedTrack &track, const GPUTPCGMPolynomialField* field,  GPUTPCGMMergedTrackHit* clusters );

  int CreateTPCclusterMI( const GPUTPCGMMergedTrackHit &h, AliTPCclusterMI &c);
  
  bool FitOffline(  const GPUTPCGMPolynomialField* field, GPUTPCGMMergedTrack &gmtrack,  GPUTPCGMMergedTrackHit* clusters, int &N );

private:
  GPUParam fCAParam;
};

#endif

#endif
