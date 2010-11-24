//-*- Mode: C++ -*-
// $Id: AliHLTTPCGMCluster.h 39008 2010-02-18 17:33:32Z sgorbuno $
// ************************************************************************
// This file is property of and copyright by the ALICE HLT Project        *
// ALICE Experiment at CERN, All rights reserved.                         *
// See cxx source for full Copyright notice                               *
//                                                                        *
//*************************************************************************


#ifndef ALIHLTTPCGMCLUSTER_H
#define ALIHLTTPCGMCLUSTER_H
#include "TMath.h"
/**
 * @class AliHLTTPCGMCluster
 *
 * The class describes TPC clusters used in AliHLTTPCGMMerger
 */
class AliHLTTPCGMCluster
{
  
 public:

  unsigned char  ISlice()  const { return fISlice;    }
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

  //private:

  unsigned char fISlice;   // slice number
  unsigned char fIRow;     // row number
  int fId;                 // cluster Id
  UChar_t fPackedAmp;      // packed amplitude
  float fX;                // x position (slice coord.system)
  float fY;                // y position (slice coord.system)
  float fZ;                // z position (slice coord.system)
  float fErr2Y;            // Squared measurement error of y position
  float fErr2Z;            // Squared measurement error of z position
};

#endif
