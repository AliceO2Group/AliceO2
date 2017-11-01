//-*- Mode: C++ -*-
// ************************************************************************
// This file is property of and copyright by the ALICE HLT Project        *
// ALICE Experiment at CERN, All rights reserved.                         *
// See cxx source for full Copyright notice                               *
//                                                                        *
//*************************************************************************


#ifndef ALIHLTTPCGMMERGEDTRACKHIT_H
#define ALIHLTTPCGMMERGEDTRACKHIT_H

struct AliHLTTPCGMMergedTrackHit
{
  float fX, fY, fZ;
  unsigned int fId;
  unsigned char fSlice, fRow, fLeg;
  char fState;
};

#endif 
