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
  
  enum hitState { flagSplitPad = 0x1, flagSplitTime = 0x2, flagSplit = 0x3, flagEdge = 0x4, flagSingle = 0x8, hwcfFlags = 0xF, flagRejectDistance = 0x10, flagRejectErr = 0x20, flagReject = 0x30 };
};

#endif 
