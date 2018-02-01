//-*- Mode: C++ -*-

// ************************************************************************
// This file is property of and copyright by the ALICE HLT Project        *
// ALICE Experiment at CERN, All rights reserved.                         *
// See cxx source for full Copyright notice                               *
//                                                                        *
//*************************************************************************


#ifndef AliHLTTPCGMOfflineFitter_H
#define AliHLTTPCGMOfflineFitter_H

#if ( !defined(HLTCA_STANDALONE) && !defined(HLTCA_GPUCODE) )


#include "AliHLTTPCCAParam.h"
#include "AliTPCtracker.h"

class AliHLTTPCGMMergedTrack;
class AliHLTTPCGMMergedTrackHit;
class AliTPCclusterMI;
class AliHLTTPCGMPolynomialField;

class AliHLTTPCGMOfflineFitter :public AliTPCtracker
{
public:

  AliHLTTPCGMOfflineFitter();  
  ~AliHLTTPCGMOfflineFitter();
  
  void Initialize( const AliHLTTPCCAParam& hltParam, Long_t TimeStamp, bool isMC );
  
  void RefitTrack(  AliHLTTPCGMMergedTrack &track, const AliHLTTPCGMPolynomialField* field,  AliHLTTPCGMMergedTrackHit* clusters );

  int CreateTPCclusterMI( const AliHLTTPCGMMergedTrackHit &h, AliTPCclusterMI &c);
  
  bool FitOffline(  const AliHLTTPCGMPolynomialField* field, AliHLTTPCGMMergedTrack &gmtrack,  AliHLTTPCGMMergedTrackHit* clusters, int &N );

private:
  AliHLTTPCCAParam fCAParam;
};

#endif

#endif 
