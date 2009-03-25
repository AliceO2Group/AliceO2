//-*- Mode: C++ -*-
// @(#) $Id$
// ************************************************************************
// This file is property of and copyright by the ALICE HLT Project        * 
// ALICE Experiment at CERN, All rights reserved.                         *
// See cxx source for full Copyright notice                               *
//                                                                        *
//*************************************************************************

#ifndef ALIHLTTPCCAHIT_H
#define ALIHLTTPCCAHIT_H

#include "AliHLTTPCCADef.h"

/**
 * @class AliHLTTPCCAHit
 *
 * The AliHLTTPCCAHit class is the internal representation
 * of the TPC clusters for the AliHLTTPCCATracker algorithm.
 *
 */
class AliHLTTPCCAHit
{
public:
  
  GPUhd() Float_t Y() const   { return fY;    }
  GPUhd() Float_t Z() const  { return fZ;    }

  GPUhd() void SetY( Float_t v ){ fY = v;    }
  GPUhd() void SetZ( Float_t v ){ fZ = v;    }
  
protected:
  
  Float_t fY, fZ;       // Y and Z position of the TPC cluster
  
};


#endif
