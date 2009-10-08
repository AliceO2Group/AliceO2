//-*- Mode: C++ -*-
// @(#) $Id: AliHLTTPCCAHit.h 35151 2009-10-01 13:35:10Z sgorbuno $
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

    GPUhd() float Y() const   { return fY;    }
    GPUhd() float Z() const  { return fZ;    }

    GPUhd() void SetY( float v ) { fY = v;    }
    GPUhd() void SetZ( float v ) { fZ = v;    }

  protected:
    float fY, fZ;       // Y and Z position of the TPC cluster

};


#endif
