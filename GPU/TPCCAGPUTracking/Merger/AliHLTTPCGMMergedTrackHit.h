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
  unsigned char fSlice, fRow, fLeg, fState;
  unsigned short fAmp;
  
  enum hitState { flagSplitPad = 0x1, flagSplitTime = 0x2, flagSplit = 0x3, flagEdge = 0x4, flagSingle = 0x8, flagShared = 0x10, hwcfFlags = 0x1F, flagRejectDistance = 0x20, flagRejectErr = 0x40, flagReject = 0x60, flagNotFit = 0x80 };

#ifdef GMPropagatePadRowTime
public:
  float fPad;
  float fTime;
#endif
};

#endif 
