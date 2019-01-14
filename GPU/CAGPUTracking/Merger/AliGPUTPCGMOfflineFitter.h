//-*- Mode: C++ -*-

// ************************************************************************
// This file is property of and copyright by the ALICE HLT Project        *
// ALICE Experiment at CERN, All rights reserved.                         *
// See cxx source for full Copyright notice                               *
//                                                                        *
//*************************************************************************


#ifndef AliGPUTPCGMOfflineFitter_H
#define AliGPUTPCGMOfflineFitter_H

#if ( defined(GPUCA_ALIROOT_LIB) && !defined(GPUCA_GPUCODE) )


#include "AliGPUCAParam.h"
#include "AliTPCtracker.h"

class AliGPUTPCGMMergedTrack;
class AliGPUTPCGMMergedTrackHit;
class AliTPCclusterMI;
class AliGPUTPCGMPolynomialField;

class AliGPUTPCGMOfflineFitter :public AliTPCtracker
{
public:

  AliGPUTPCGMOfflineFitter();
  ~AliGPUTPCGMOfflineFitter();
  
  void Initialize( const AliGPUCAParam& hltParam, Long_t TimeStamp, bool isMC );
  
  void RefitTrack(  AliGPUTPCGMMergedTrack &track, const AliGPUTPCGMPolynomialField* field,  AliGPUTPCGMMergedTrackHit* clusters );

  int CreateTPCclusterMI( const AliGPUTPCGMMergedTrackHit &h, AliTPCclusterMI &c);
  
  bool FitOffline(  const AliGPUTPCGMPolynomialField* field, AliGPUTPCGMMergedTrack &gmtrack,  AliGPUTPCGMMergedTrackHit* clusters, int &N );

private:
  AliGPUCAParam fCAParam;
};

#endif

#endif
