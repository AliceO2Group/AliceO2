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

/* $Id: AliHLTTRDTrackletWord.cxx 28397 2008-09-02 09:33:00Z cblume $ */

////////////////////////////////////////////////////////////////////////////
//                                                                        //
//  A tracklet word in HLT - adapted from AliTRDtrackletWord              //
//                                                                        //
//                                                                        //
//                                                                        //
////////////////////////////////////////////////////////////////////////////

#include "AliHLTTRDTrackletWord.h"
#include "AliTRDgeometry.h"
#include "AliTRDpadPlane.h"
#include "AliLog.h"
#include "AliTRDtrackletWord.h"
#include "AliTRDtrackletMCM.h"

AliTRDgeometry* AliHLTTRDTrackletWord::fgGeo = 0x0;

AliHLTTRDTrackletWord::AliHLTTRDTrackletWord(UInt_t trackletWord) :
  fId(-1),
  fHCId(-1),
  fTrackletWord(trackletWord)
{
  if (!fgGeo)
    fgGeo = new AliTRDgeometry;
}

AliHLTTRDTrackletWord::AliHLTTRDTrackletWord(UInt_t trackletWord, Int_t hcid, Int_t id) :
  fId(id),
  fHCId(hcid),
  fTrackletWord(trackletWord)
{
  if (!fgGeo)
    fgGeo = new AliTRDgeometry;
}

AliHLTTRDTrackletWord::AliHLTTRDTrackletWord(const AliHLTTRDTrackletWord &rhs) :
  fId(rhs.fId),
  fHCId(rhs.fHCId),
  fTrackletWord(rhs.fTrackletWord)
{

  if (!fgGeo)
    fgGeo = new AliTRDgeometry;
}

AliHLTTRDTrackletWord::AliHLTTRDTrackletWord(const AliTRDtrackletWord &rhs) :
  fId(-1),
  fHCId(rhs.GetHCId()),
  fTrackletWord(rhs.GetTrackletWord())
{

  if (!fgGeo)
    fgGeo = new AliTRDgeometry;
}

AliHLTTRDTrackletWord::AliHLTTRDTrackletWord(const AliTRDtrackletMCM &rhs) :
  fId(-1),
  fHCId(rhs.GetHCId()),
  fTrackletWord(rhs.GetTrackletWord())
{

  if (!fgGeo)
    fgGeo = new AliTRDgeometry;
}

AliHLTTRDTrackletWord::~AliHLTTRDTrackletWord()
{

}

AliHLTTRDTrackletWord& AliHLTTRDTrackletWord::operator=(const AliHLTTRDTrackletWord &rhs)
{
  this->~AliHLTTRDTrackletWord();
  new(this) AliHLTTRDTrackletWord(rhs);
  return *this;
}

AliHLTTRDTrackletWord& AliHLTTRDTrackletWord::operator=(const AliTRDtrackletMCM &rhs)
{
  this->~AliHLTTRDTrackletWord();
  new(this) AliHLTTRDTrackletWord(rhs);
  return *this;
}

Int_t AliHLTTRDTrackletWord::GetYbin() const {
  // returns (signed) value of Y
  if (fTrackletWord & 0x1000) {
    return -((~(fTrackletWord-1)) & 0x1fff);
  }
  else {
    return (fTrackletWord & 0x1fff);
  }
}

Int_t AliHLTTRDTrackletWord::GetdY() const
{
  // returns (signed) value of the deflection length
  if (fTrackletWord & (1 << 19)) {
    return -((~((fTrackletWord >> 13) - 1)) & 0x7f);
  }
  else {
    return ((fTrackletWord >> 13) & 0x7f);
  }
}

Int_t AliHLTTRDTrackletWord::GetROB() const
{
  return 2 * (GetZbin() / 4) + (GetY() > 0 ? 1 : 0);
}

Int_t AliHLTTRDTrackletWord::GetMCM() const
{
  AliTRDpadPlane *pp = fgGeo->GetPadPlane(GetDetector());
  return (((Int_t) ((GetY()) / pp->GetWidthIPad()) + 72) / 18) % 4
    + 4 * (GetZbin() % 4);
}

