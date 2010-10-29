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
    UInt_t patch = (id>>1)&(0x7<<21); 
    UInt_t cluster = id&0x1fffff;
    fId = (row<<24)+ patch + cluster;
    fXYZp = AliHLTTPCCADataCompressor::PackXYZ( row, x, y, z );
  }

  GPUh() void Get( int iSlice, UInt_t &Id, UInt_t &row, float &x, float &y, float &z ) const{
    row = fId>>24;
    UInt_t patch = (fId<<1)&(0x7<<22);
    UInt_t cluster = fId&0x1fffff;
    Id = (iSlice<<25) + patch + cluster;  
    AliHLTTPCCADataCompressor::UnpackXYZ( row, fXYZp, x, y, z  );
  }  
  
  private:
    UInt_t fId; // Id ( row, patch, cluster )
    AliHLTTPCCACompressedCluster fXYZp;// packed coordinates
};

#endif 
