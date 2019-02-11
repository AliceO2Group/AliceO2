/**************************************************************************
 * Copyright(c) 1998-1999, ALICE Experiment at CERN, All rights reserved. *
 *                                                                        *
 * Author: The ALICE Off-line Project.                                    *
 * Contributors are mentioned in the code where appropriate.              *
 *                                                                        *
 * Permission to use, copy, modify and distribute this software and its   *
 * documentation strictly for non-commercial purposes is hereby granted   *
 * without fee, provided that the above copyright notice appears in all   *
 * copies and that both the copyright notice and this permission notice   *
 * appear in the supporting documentation. The authors make no claims     *
 * about the suitability of this software for any purpose. It is          *
 * provided "as is" without express or implied warranty.                  *
 **************************************************************************/

///////////////////////////////////////////////////////////////////////////////
//
// ESD format for TRD tracklet from FEE used for triggering
//
// Author: Jochen Klein <jochen.klein@cern.ch>
//
///////////////////////////////////////////////////////////////////////////////

#include "AliESDTrdTracklet.h"

ClassImp(AliESDTrdTracklet)

AliESDTrdTracklet::AliESDTrdTracklet() :
  AliVTrdTracklet(),
  fHCId(-1),
  fTrackletWord(0)
{
  // default ctor
  for (int i=3;i--;) fLabel[i] = -1;
}

AliESDTrdTracklet::AliESDTrdTracklet(UInt_t trackletWord, Short_t hcid, const Int_t* label) :
  AliVTrdTracklet(),
  fHCId(hcid),
  fTrackletWord(trackletWord)
{
  // ctor with given tracklet word (and label)
  if (label) {
    for (int i=3;i--;) fLabel[i] = label[i];
  }
  else {
    for (int i=3;i--;) fLabel[i] = -1;
  }
}

AliESDTrdTracklet::~AliESDTrdTracklet()
{
  // dtor
}

AliESDTrdTracklet::AliESDTrdTracklet(const AliESDTrdTracklet &trkl) :
  AliVTrdTracklet(trkl),
  fHCId(trkl.fHCId),
  fTrackletWord(trkl.fTrackletWord)
{
  // copy ctor
  for (int i=3;i--;) fLabel[i] = trkl.fLabel[i];
}

AliESDTrdTracklet& AliESDTrdTracklet::operator=(const AliESDTrdTracklet &trkl)
{
  // assignment operator

  if (this == &trkl)
    return *this;

  AliVTrdTracklet::operator=(trkl);
  fHCId = trkl.fHCId;
  fTrackletWord = trkl.fTrackletWord;
  for (int i=3;i--;) fLabel[i] = trkl.fLabel[i];
  
  return *this;
}

Int_t AliESDTrdTracklet::GetBinY() const
{
  // returns (signed) value of Y

  if (fTrackletWord & 0x1000) {
    return -((~(fTrackletWord-1)) & 0x1fff);
  }
  else {
    return (fTrackletWord & 0x1fff);
  }
}

Int_t AliESDTrdTracklet::GetBinDy() const
{
  // returns (signed) value of the deflection length

  if (fTrackletWord & (1 << 19)) {
    return -((~((fTrackletWord >> 13) - 1)) & 0x7f);
  }
  else {
    return ((fTrackletWord >> 13) & 0x7f);
  }
}
