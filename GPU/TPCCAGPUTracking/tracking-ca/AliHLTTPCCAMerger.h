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
#endif //HLTCA_GPUCODE

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

    class AliHLTTPCCAClusterInfo
    {

      public:

        unsigned char  ISlice()    const { return fISlice;    }
        unsigned char  IRow()    const { return fIRow;    }
        int  Id()      const { return fId;      }
        UChar_t PackedAmp() const { return fPackedAmp; }
        float X()         const { return fX;         }
        float Y()         const { return fY;         }
        float Z()         const { return fZ;         }
        float Err2Y()     const { return fErr2Y;     }
        float Err2Z()     const { return fErr2Z;     }

        void SetISlice    ( unsigned char v  ) { fISlice    = v; }
        void SetIRow    ( unsigned char v  ) { fIRow    = v; }
        void SetId      (  int v  ) { fId      = v; }
        void SetPackedAmp ( UChar_t v ) { fPackedAmp = v; }
        void SetX         ( float v ) { fX         = v; }
        void SetY         ( float v ) { fY         = v; }
        void SetZ         ( float v ) { fZ         = v; }
        void SetErr2Y     ( float v ) { fErr2Y     = v; }
        void SetErr2Z     ( float v ) { fErr2Z     = v; }

      private:

        unsigned char fISlice;            // slice number
        unsigned char fIRow;            // row number
        int fId;                 // cluster hlt number
        UChar_t fPackedAmp; // packed cluster amplitude
        float fX;                // x position (slice coord.system)
        float fY;                // y position (slice coord.system)
        float fZ;                // z position (slice coord.system)
        float fErr2Y;            // Squared measurement error of y position
        float fErr2Z;            // Squared measurement error of z position
    };

    AliHLTTPCCAMerger();
    ~AliHLTTPCCAMerger();

    void SetSliceParam( const AliHLTTPCCAParam &v ) { fSliceParam = v; }

    void Clear();
    void SetSliceData( int index, const AliHLTTPCCASliceOutput *SliceData );
    void Reconstruct();

    const AliHLTTPCCAMergerOutput * Output() const { return fOutput; }

    bool FitTrack( AliHLTTPCCATrackParam &T, float &Alpha,
                   AliHLTTPCCATrackParam t0, float Alpha0, int hits[], int &NHits,  bool dir,
		   bool final = 0, 
                   AliHLTTPCCAClusterInfo *infoArray = 0 );

    const AliHLTTPCCAParam &SliceParam() const { return fSliceParam; }

    static float GetChi2( float x1, float y1, float a00, float a10, float a11,
                          float x2, float y2, float b00, float b10, float b11  );

  private:

    AliHLTTPCCAMerger( const AliHLTTPCCAMerger& );
    const AliHLTTPCCAMerger &operator=( const AliHLTTPCCAMerger& ) const;

    class AliHLTTPCCASliceTrackInfo;
    class AliHLTTPCCABorderTrack;

    void MakeBorderTracks( int iSlice, int iBorder, AliHLTTPCCABorderTrack B[], int &nB );
    void MergeBorderTracks( int iSlice1, AliHLTTPCCABorderTrack B1[], int N1,
                            int iSlice2, AliHLTTPCCABorderTrack B2[], int N2 );


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

#endif //ALIHLTTPCCAMERGER_H
