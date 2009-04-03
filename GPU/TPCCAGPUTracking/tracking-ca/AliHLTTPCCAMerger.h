//-*- Mode: C++ -*-
// ************************************************************************
// This file is property of and copyright by the ALICE HLT Project        * 
// ALICE Experiment at CERN, All rights reserved.                         *
// See cxx source for full Copyright notice                               *
//                                                                        *
//*************************************************************************

#ifndef ALIHLTTPCCAMERGER_H
#define ALIHLTTPCCAMERGER_H

#include "AliHLTTPCCADef.h"
#include "AliHLTTPCCAParam.h"
#include "AliHLTTPCCATrackParam.h"

#if !defined(HLTCA_GPUCODE)
#include <iostream>
#endif

class AliHLTTPCCASliceTrack;
class AliHLTTPCCASliceOutput;
class AliHLTTPCCAMergedTrack;
class AliHLTTPCCAMergerOutput;

/**
 * @class AliHLTTPCCAMerger
 * 
 */
class AliHLTTPCCAMerger
{

public:

  AliHLTTPCCAMerger();
  ~AliHLTTPCCAMerger();

  void SetSliceParam( const AliHLTTPCCAParam &v ){ fSliceParam = v; }

  void Clear();
  void SetSliceData( int index, const AliHLTTPCCASliceOutput *SliceData );
  void Reconstruct();

  const AliHLTTPCCAMergerOutput * Output() const { return fOutput; }
  
 private:

  AliHLTTPCCAMerger(const AliHLTTPCCAMerger&);
  const AliHLTTPCCAMerger &operator=(const AliHLTTPCCAMerger&) const;
  
  class AliHLTTPCCAClusterInfo;
  class AliHLTTPCCASliceTrackInfo;
  class AliHLTTPCCABorderTrack;

  void MakeBorderTracks( Int_t iSlice, Int_t iBorder, AliHLTTPCCABorderTrack B[], Int_t &nB);
  void SplitBorderTracks( Int_t iSlice1, AliHLTTPCCABorderTrack B1[], Int_t N1,
			  Int_t iSlice2, AliHLTTPCCABorderTrack B2[], Int_t N2 );

  static Float_t GetChi2( Float_t x1, Float_t y1, Float_t a00, Float_t a10, Float_t a11, 
			  Float_t x2, Float_t y2, Float_t b00, Float_t b10, Float_t b11  );

  void UnpackSlices();
  void Merging();
   
  Bool_t FitTrack( AliHLTTPCCATrackParam &T, Float_t &Alpha, 
		   AliHLTTPCCATrackParam t0, Float_t Alpha0, Int_t hits[], Int_t &NHits,  Bool_t dir=0 );
  
  static const Int_t fgkNSlices = 36;       //* N slices 
  AliHLTTPCCAParam fSliceParam;           //* slice parameters (geometry, calibr, etc.)
  const AliHLTTPCCASliceOutput *fkSlices[fgkNSlices]; //* array of input slice tracks
  AliHLTTPCCAMergerOutput *fOutput;       //* array of output merged tracks  
  AliHLTTPCCASliceTrackInfo *fTrackInfos; //* additional information for slice tracks
  Int_t fMaxTrackInfos;                   //* booked size of fTrackInfos array
  AliHLTTPCCAClusterInfo *fClusterInfos;  //* information about track clusters
  Int_t fMaxClusterInfos;                 //* booked size of fClusterInfos array
  Int_t fSliceTrackInfoStart[fgkNSlices];   //* slice starting index in fTrackInfos array;
  Int_t fSliceNTrackInfos[fgkNSlices];                //* N of slice track infos in fTrackInfos array;
};

#endif
