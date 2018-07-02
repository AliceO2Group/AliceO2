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

AliHLTTRDGeometry* AliHLTTRDTrackletWord::fgGeo = 0x0;

AliHLTTRDTrackletWord::AliHLTTRDTrackletWord(unsigned int trackletWord) :
  fId(-1),
  fHCId(-1),
  fTrackletWord(trackletWord)
{
  for (int i=3;i--;) fLabel[i] = -1;
  if (!fgGeo)
    fgGeo = new AliHLTTRDGeometry;
}

AliHLTTRDTrackletWord::AliHLTTRDTrackletWord(unsigned int trackletWord, int hcid, int id, int* label) :
  fId(id),
  fHCId(hcid),
  fTrackletWord(trackletWord)
{
  if (label) {
    for (int i=3;i--;) fLabel[i] = label[i];
  }
  else {
    for (int i=3;i--;) fLabel[i] = -1;
  }
  if (!fgGeo)
    fgGeo = new AliHLTTRDGeometry;
}

AliHLTTRDTrackletWord::AliHLTTRDTrackletWord(const AliHLTTRDTrackletWord &rhs) :
  fId(rhs.fId),
  fHCId(rhs.fHCId),
  fTrackletWord(rhs.fTrackletWord)
{
  for (int i=3;i--;) fLabel[i] =  rhs.fLabel[i];
  if (!fgGeo)
    fgGeo = new AliHLTTRDGeometry;
}

#ifdef HLTCA_BUILD_ALIROOT_LIB
#include "AliTRDtrackletWord.h"
#include "AliTRDtrackletMCM.h"


AliHLTTRDTrackletWord::AliHLTTRDTrackletWord(const AliTRDtrackletWord &rhs) :
  fId(-1),
  fHCId(rhs.GetHCId()),
  fTrackletWord(rhs.GetTrackletWord())
{
  for (int i=3;i--;) fLabel[i] = -1;
  if (!fgGeo)
    fgGeo = new AliHLTTRDGeometry;
}

AliHLTTRDTrackletWord::AliHLTTRDTrackletWord(const AliTRDtrackletMCM &rhs) :
  fId(-1),
  fHCId(rhs.GetHCId()),
  fTrackletWord(rhs.GetTrackletWord())
{
  for (int i=3;i--;) fLabel[i] =  rhs.GetLabel(i);
  if (!fgGeo)
    fgGeo = new AliHLTTRDGeometry;
}

AliHLTTRDTrackletWord& AliHLTTRDTrackletWord::operator=(const AliTRDtrackletMCM &rhs)
{
  this->~AliHLTTRDTrackletWord();
  new(this) AliHLTTRDTrackletWord(rhs);
  return *this;
}

#endif

AliHLTTRDTrackletWord::~AliHLTTRDTrackletWord()
{

}

AliHLTTRDTrackletWord& AliHLTTRDTrackletWord::operator=(const AliHLTTRDTrackletWord &rhs)
{
  this->~AliHLTTRDTrackletWord();
  new(this) AliHLTTRDTrackletWord(rhs);
  return *this;
}

int AliHLTTRDTrackletWord::GetYbin() const {
  // returns (signed) value of Y
  if (fTrackletWord & 0x1000) {
    return -((~(fTrackletWord-1)) & 0x1fff);
  }
  else {
    return (fTrackletWord & 0x1fff);
  }
}

int AliHLTTRDTrackletWord::GetdY() const
{
  // returns (signed) value of the deflection length
  if (fTrackletWord & (1 << 19)) {
    return -((~((fTrackletWord >> 13) - 1)) & 0x7f);
  }
  else {
    return ((fTrackletWord >> 13) & 0x7f);
  }
}

int AliHLTTRDTrackletWord::GetROB() const
{
  return 2 * (GetZbin() / 4) + (GetY() > 0 ? 1 : 0);
}

int AliHLTTRDTrackletWord::GetMCM() const
{
  return (((int) ((GetY()) / fgGeo->GetPadPlaneWithIPad(GetDetector())) + 72) / 18) % 4
    + 4 * (GetZbin() % 4);
}
