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

#if !defined(HLTCA_GPUCODE)
#include <iostream>
#endif

class AliHLTTPCCATrackParam;
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
  AliHLTTPCCAMerger(const AliHLTTPCCAMerger&);
  const AliHLTTPCCAMerger &operator=(const AliHLTTPCCAMerger&) const;
  ~AliHLTTPCCAMerger();

  void SetSliceParam( const AliHLTTPCCAParam &v ){ fSliceParam = v; }

  void Reconstruct( const AliHLTTPCCASliceOutput **Slices );

  const AliHLTTPCCAMergerOutput * Output() const { return fOutput; }
  
 private:
  
  class AliHLTTPCCAClusterInfo{

  public:
    
    UInt_t  ISlice()    const { return fISlice;    }
    UInt_t  IRow()      const { return fIRow;      }
    UInt_t  IClu()      const { return fIClu;      }
    UChar_t PackedAmp() const { return fPackedAmp; }
    Float_t Y()         const { return fY;         }
    Float_t Z()         const { return fZ;         }
    Float_t Err2Y()     const { return fErr2Y;     }
    Float_t Err2Z()     const { return fErr2Z;     }
    
    void SetISlice    ( UInt_t v  ) { fISlice    = v; }
    void SetIRow      ( UInt_t v  ) { fIRow      = v; }
    void SetIClu      ( UInt_t v  ) { fIClu      = v; }
    void SetPackedAmp ( UChar_t v ) { fPackedAmp = v; }
    void SetY         ( Float_t v ) { fY         = v; } 
    void SetZ         ( Float_t v ) { fZ         = v; } 
    void SetErr2Y     ( Float_t v ) { fErr2Y     = v; } 
    void SetErr2Z     ( Float_t v ) { fErr2Z     = v; } 

  private:

    UInt_t fISlice;            // slice number
    UInt_t fIRow;              // row number
    UInt_t fIClu;              // cluster number
    UChar_t fPackedAmp; // packed cluster amplitude
    Float_t fY;                // y position (slice coord.system)
    Float_t fZ;                // z position (slice coord.system)
    Float_t fErr2Y;            // Squared measurement error of y position
    Float_t fErr2Z;            // Squared measurement error of z position
  };

  struct AliHLTTPCCASliceTrackInfo{

    AliHLTTPCCATrackParam fInnerParam; // inner parameters
    AliHLTTPCCATrackParam fOuterParam; // outer parameters
    Float_t fInnerAlpha;                 // alpha angle for inner parameters
    Float_t fOuterAlpha;                 // alpha angle for outer parameters
    Int_t fNClusters;                  // N clusters
    Int_t fFirstClusterRef; //index of the first track cluster in the global cluster array
    Int_t fPrevNeighbour; // neighbour in the previous slise
    Int_t fNextNeighbour; // neighbour in the next slise
    Bool_t fUsed;         // is the slice track already merged
  };

  struct AliHLTTPCCABorderTrack{

    AliHLTTPCCABorderTrack(): fParam(), fITrack(0), fIRow(0), fNHits(0), fX(0), fOK(0){};
    AliHLTTPCCATrackParam fParam; // track parameters at the border
    Int_t fITrack;               // track index
    Int_t fIRow;                 // row number of the closest cluster
    Int_t fNHits;                // n hits
    Float_t fX;                  // X coordinate of the closest cluster
    Bool_t fOK;                  // is the trak rotated and extrapolated correctly
  };
  
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
  const AliHLTTPCCASliceOutput **fkSlices;       //* array of input slice tracks  
  AliHLTTPCCAMergerOutput *fOutput;       //* array of output merged tracks  
  AliHLTTPCCASliceTrackInfo *fTrackInfos; //* additional information for slice tracks
  Int_t fMaxTrackInfos;                   //* booked size of fTrackInfos array
  AliHLTTPCCAClusterInfo *fClusterInfos;  //* information about track clusters
  Int_t fMaxClusterInfos;                 //* booked size of fClusterInfos array
  Int_t fSliceTrackInfoStart[fgkNSlices];   //* slice starting index in fTrackInfos array;
  Int_t fSliceNTrackInfos[fgkNSlices];                //* N of slice track infos in fTrackInfos array;
};

#endif
