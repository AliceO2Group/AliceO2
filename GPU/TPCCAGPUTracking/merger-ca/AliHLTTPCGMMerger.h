//-*- Mode: C++ -*-
// ************************************************************************
// This file is property of and copyright by the ALICE HLT Project        *
// ALICE Experiment at CERN, All rights reserved.                         *
// See cxx source for full Copyright notice                               *
//                                                                        *
//*************************************************************************

#ifndef ALIHLTTPCGMMERGER_H
#define ALIHLTTPCGMMERGER_H

#include "AliHLTTPCCADef.h"
#include "AliHLTTPCCAParam.h"
#include "AliHLTTPCGMBorderTrack.h"
#include "AliHLTTPCGMSliceTrack.h"
#include "AliHLTTPCCAGPUTracker.h"
#include "AliHLTTPCGMPolynomialField.h"
#include "AliHLTTPCGMMergedTrack.h"

#if !defined(HLTCA_GPUCODE)
#include <iostream>
#include <cmath>
#endif //HLTCA_GPUCODE

class AliHLTTPCCASliceTrack;
class AliHLTTPCCASliceOutput;
class AliHLTTPCGMCluster;
class AliHLTTPCGMTrackParam;

/**
 * @class AliHLTTPCGMMerger
 *
 */
class AliHLTTPCGMMerger
{
  
public:
  
  AliHLTTPCGMMerger();
  ~AliHLTTPCGMMerger();
  
  void SetSliceParam( const AliHLTTPCCAParam &v, long int TimeStamp=0, bool isMC=0  );
  
  void Clear();
  void SetSliceData( int index, const AliHLTTPCCASliceOutput *SliceData );
  bool Reconstruct(bool resetTimers = false);
  
  Int_t NOutputTracks() const { return fNOutputTracks; }
  const AliHLTTPCGMMergedTrack * OutputTracks() const { return fOutputTracks; }
   
  const AliHLTTPCCAParam &SliceParam() const { return fSliceParam; }

  void SetGPUTracker(AliHLTTPCCAGPUTracker* gpu) {fGPUTracker = gpu;}
  void SetDebugLevel(int debug) {fDebugLevel = debug;}

  const AliHLTTPCGMPolynomialField& Field() const {return fField;}
  const AliHLTTPCGMPolynomialField* pField() const {return &fField;}

  int NClusters() const { return(fNClusters); }
  int NOutputTrackClusters() const { return(fNOutputTrackClusters); }
  AliHLTTPCGMMergedTrackHit* Clusters() const {return(fClusters);}

private:
  
  AliHLTTPCGMMerger( const AliHLTTPCGMMerger& );

  const AliHLTTPCGMMerger &operator=( const AliHLTTPCGMMerger& ) const;
  
  void MakeBorderTracks( int iSlice, int iBorder, AliHLTTPCGMBorderTrack B[], int &nB );

  void MergeBorderTracks( int iSlice1, AliHLTTPCGMBorderTrack B1[],  int N1,
			  int iSlice2, AliHLTTPCGMBorderTrack B2[],  int N2 );

  void ClearMemory();
  bool AllocateMemory();
  void UnpackSlices();
  void MergeWithingSlices();
  void MergeSlices();
  void CollectMergedTracks();
  void Refit(bool resetTimers);
  
  static const int fgkNSlices = 36;       //* N slices
  int fNextSliceInd[fgkNSlices];
  int fPrevSliceInd[fgkNSlices];

  AliHLTTPCGMPolynomialField fField;
  
  AliHLTTPCCAParam fSliceParam;           //* slice parameters (geometry, calibr, etc.)
  const AliHLTTPCCASliceOutput *fkSlices[fgkNSlices]; //* array of input slice tracks

  Int_t fNOutputTracks;
  Int_t fNOutputTrackClusters;
  AliHLTTPCGMMergedTrack *fOutputTracks;       //* array of output merged tracks
  
  AliHLTTPCGMSliceTrack *fSliceTrackInfos; //* additional information for slice tracks
  int fSliceTrackInfoStart[fgkNSlices];   //* slice starting index in fTrackInfos array;
  int fSliceNTrackInfos[fgkNSlices];      //* N of slice track infos in fTrackInfos array;
  int fSliceTrackGlobalInfoStart[fgkNSlices]; //* Same for global tracks
  int fSliceNGlobalTrackInfos[fgkNSlices]; //* Same for global tracks
  int fMaxSliceTracks;      // max N tracks in one slice
  AliHLTTPCGMMergedTrackHit *fClusters;
  AliHLTTPCGMBorderTrack *fBorderMemory; // memory for border tracks
  AliHLTTPCGMBorderTrack::Range *fBorderRangeMemory; // memory for border tracks

  AliHLTTPCCAGPUTracker* fGPUTracker;
  int fDebugLevel;

  int fNClusters;			//Total number of incoming clusters
  
};

#endif //ALIHLTTPCCAMERGER_H
