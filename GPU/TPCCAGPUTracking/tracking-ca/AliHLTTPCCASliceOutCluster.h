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

  GPUh() void Set( UInt_t Id, UInt_t row, float x, float y, float z ){
    fId = (Id&0xffffff)+(row<<25);
    fXYZp = AliHLTTPCCADataCompressor::PackXYZ( row, x, y, z );
  }

  GPUh() void Get( int iSlice, UInt_t &Id, UInt_t &row, float &x, float &y, float &z ) const{
    Id = (fId&0xffffff) + iSlice<<25;  
    row = fId>>25;
    AliHLTTPCCADataCompressor::UnpackXYZ( row, fXYZp, x, y, z  );
  }  
  
  private:
    UInt_t fId; // Id ( row, patch, cluster )
    AliHLTTPCCACompressedCluster fXYZp;// packed coordinates
};

#endif 
