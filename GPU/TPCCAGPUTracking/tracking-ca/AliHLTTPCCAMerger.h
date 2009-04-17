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

    void SetSliceParam( const AliHLTTPCCAParam &v ) { fSliceParam = v; }

    void Clear();
    void SetSliceData( int index, const AliHLTTPCCASliceOutput *SliceData );
    void Reconstruct();

    const AliHLTTPCCAMergerOutput * Output() const { return fOutput; }

    bool FitTrack( AliHLTTPCCATrackParam &T, float &Alpha,
                   AliHLTTPCCATrackParam t0, float Alpha0, int hits[], int &NHits,  bool dir = 0 );

  private:

    AliHLTTPCCAMerger( const AliHLTTPCCAMerger& );
    const AliHLTTPCCAMerger &operator=( const AliHLTTPCCAMerger& ) const;

    class AliHLTTPCCAClusterInfo;
    class AliHLTTPCCASliceTrackInfo;
    class AliHLTTPCCABorderTrack;

    void MakeBorderTracks( int iSlice, int iBorder, AliHLTTPCCABorderTrack B[], int &nB );
    void SplitBorderTracks( int iSlice1, AliHLTTPCCABorderTrack B1[], int N1,
                            int iSlice2, AliHLTTPCCABorderTrack B2[], int N2 );

    static float GetChi2( float x1, float y1, float a00, float a10, float a11,
                          float x2, float y2, float b00, float b10, float b11  );

    void UnpackSlices();
    void Merging();



    static const int fgkNSlices = 36;       //* N slices
    AliHLTTPCCAParam fSliceParam;           //* slice parameters (geometry, calibr, etc.)
    const AliHLTTPCCASliceOutput *fkSlices[fgkNSlices]; //* array of input slice tracks
    AliHLTTPCCAMergerOutput *fOutput;       //* array of output merged tracks
    AliHLTTPCCASliceTrackInfo *fTrackInfos; //* additional information for slice tracks
    int fMaxTrackInfos;                   //* booked size of fTrackInfos array
    AliHLTTPCCAClusterInfo *fClusterInfos;  //* information about track clusters
    int fMaxClusterInfos;                 //* booked size of fClusterInfos array
    int fSliceTrackInfoStart[fgkNSlices];   //* slice starting index in fTrackInfos array;
    int fSliceNTrackInfos[fgkNSlices];                //* N of slice track infos in fTrackInfos array;
};

#endif
