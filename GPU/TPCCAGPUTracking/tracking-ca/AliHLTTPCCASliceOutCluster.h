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

  GPUh() void Set( int Id, int row, float x, float y, float z ){
    fId = Id;  fRow = (UChar_t) row; 
    fXYZp = AliHLTTPCCADataCompressor::PackXYZ( row, x, y, z );
  }

  GPUh() void Get( int &Id, int &row, float &x, float &y, float &z ) const{
    Id = fId;  row = fRow;
    AliHLTTPCCADataCompressor::UnpackXYZ( fRow, fXYZp, x, y, z  );
  }  
    
  private:
    Int_t fId; // Id ( slice, patch, cluster )
    UChar_t fRow; // row number
    AliHLTTPCCACompressedCluster fXYZp;// packed coordinates
};

#endif 
