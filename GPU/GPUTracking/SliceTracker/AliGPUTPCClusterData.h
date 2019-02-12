// **************************************************************************
// * This file is property of and copyright by the ALICE HLT Project        *
// * All rights reserved.                                                   *
// *                                                                        *
// * Primary Authors:                                                       *
// *     Copyright 2009       Matthias Kretz <kretz@kde.org>                *
// *                                                                        *
// * Permission to use, copy, modify and distribute this software and its   *
// * documentation strictly for non-commercial purposes is hereby granted   *
// * without fee, provided that the above copyright notice appears in all   *
// * copies and that both the copyright notice and this permission notice   *
// * appear in the supporting documentation. The authors make no claims     *
// * about the suitability of this software for any purpose. It is          *
// * provided "as is" without express or implied warranty.                  *
// **************************************************************************

#ifndef ALIHLTTPCCACLUSTERDATA_H
#define ALIHLTTPCCACLUSTERDATA_H

#include "AliGPUTPCDef.h"

struct AliGPUTPCClusterData
{
      int fId;
      short fRow;
      short fFlags;
      float fX;
      float fY;
      float fZ;
      float fAmp;
#ifdef GPUCA_FULL_CLUSTERDATA
      float fPad;
      float fTime;
      float fAmpMax;
      float fSigmaPad2;
      float fSigmaTime2;
#endif
};

#endif // CLUSTERDATA_H
