//-*- Mode: C++ -*-
// ************************************************************************
// This file is property of and copyright by the ALICE HLT Project        *
// ALICE Experiment at CERN, All rights reserved.                         *
// See cxx source for full Copyright notice                               *
//                                                                        *
//*************************************************************************


#ifndef ALIHLTTPCCASLICEOUTCLUSTER_H
#define ALIHLTTPCCASLICEOUTCLUSTER_H

#include "AliHLTTPCCADef.h"
#ifdef HLTCA_STANDALONE
#include "AliHLTTPCRootTypes.h"
#endif


/**
 * @class AliHLTTPCCASliceOutCluster
 * AliHLTTPCCASliceOutCluster class contains clusters which are assigned to slice tracks.
 * It is used to send the data from TPC slice trackers to the GlobalMerger
 */
class AliHLTTPCCASliceOutCluster
{
  public:

  GPUh() void Set( UInt_t id, short row, short flags, float x, float y, float z ){
    fRow = row;
    fFlags = flags;
    fId = id;
    fX = x; fY = y; fZ = z;
  }

  GPUh() float GetX() const {return fX;}
  GPUh() float GetY() const {return fY;}
  GPUh() float GetZ() const {return fZ;}
  GPUh() UInt_t GetId() const {return fId; }
  GPUh() short GetRow() const {return fRow; }
  GPUh() short GetFlags() const {return fFlags; }

  private:

  UInt_t  fId; // Id ( slice, patch, cluster )    
  short  fRow; // row
  short  fFlags; //flags
  Float_t fX;// coordinates
  Float_t fY;// coordinates
  Float_t fZ;// coordinates
};

#endif 
