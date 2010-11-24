//-*- Mode: C++ -*-
// ************************************************************************
// This file is property of and copyright by the ALICE HLT Project        *
// ALICE Experiment at CERN, All rights reserved.                         *
// See cxx source for full Copyright notice                               *
//                                                                        *
//*************************************************************************


#ifndef ALIHLTTPCCASLICEOUTCLUSTER_H
#define ALIHLTTPCCASLICEOUTCLUSTER_H

#include "AliHLTTPCCACompressedInputData.h"
#include "AliHLTTPCCADataCompressor.h"

/**
 * @class AliHLTTPCCASliceOutCluster
 * AliHLTTPCCASliceOutCluster class contains clusters which are assigned to slice tracks.
 * It is used to send the data from TPC slice trackers to the GlobalMerger
 */
class AliHLTTPCCASliceOutCluster
{
  public:

  GPUh() void Set( UInt_t id, UInt_t row, float x, float y, float z ){
    UInt_t rowtype;
    //if( row<64 ) rowtype = 0;
    //else if( row<128 ) rowtype = (UInt_t(2)<<30);
    //else rowtype = (1<<30);
    //fId = id|rowtype;
    if( row<64 ) rowtype = 0;
    else if( row<128 ) rowtype = 2;
    else rowtype = 1;
    fRowType = rowtype;
    fId = id;
    fX = x; fY = y; fZ = z;
  }

  GPUh() float GetX() const {return fX;}
  GPUh() float GetY() const {return fY;}
  GPUh() float GetZ() const {return fZ;}
  GPUh() UInt_t GetId() const {return fId; } //fId & 0x3FFFFFFF;}
  GPUh() UInt_t GetRowType() const {return fRowType; }//fId>>30;}

  private:

  UInt_t  fId; // Id ( slice, patch, cluster )    
  UInt_t  fRowType; // row type
  Float_t fX;// coordinates
  Float_t fY;// coordinates
  Float_t fZ;// coordinates
};

#endif 
