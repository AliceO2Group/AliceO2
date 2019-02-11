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
///
/// This is a class for containing all the VZERO DDL raw data
/// It is written to the ESD-friend file
///
///////////////////////////////////////////////////////////////////////////////

#include "AliESDVZEROfriend.h"

ClassImp(AliESDVZEROfriend)

//_____________________________________________________________________________
AliESDVZEROfriend::AliESDVZEROfriend():
  AliVVZEROfriend(),
  fTrigger(0),
  fTriggerMask(0)
{
  // default constructor
  for (Int_t iScaler = 0; iScaler < kNScalers; iScaler++)
    fScalers[iScaler] = 0;

  for (Int_t iBunch = 0; iBunch < kNBunches; iBunch++)
    fBunchNumbers[iBunch] = 0;

  for (Int_t iChannel = 0; iChannel < kNChannels; iChannel++) {
    fBBScalers[iChannel] = 0;
    fBGScalers[iChannel] = 0;
    for (Int_t iBunch = 0; iBunch < kNBunches; iBunch++) {
      fChargeMB[iChannel][iBunch] = 0;
      fIsIntMB[iChannel][iBunch]  = kFALSE;
      fIsBBMB[iChannel][iBunch]   = kFALSE;
      fIsBGMB[iChannel][iBunch]   = kFALSE;
    }
    for (Int_t iEv = 0; iEv < kNEvOfInt; iEv++) {
      fADC[iChannel][iEv]   = 0.0;
      fIsInt[iChannel][iEv] = kFALSE;
      fIsBB[iChannel][iEv]  = kFALSE;
      fIsBG[iChannel][iEv]  = kFALSE;
    }
    fTime[iChannel]  = 0.0;
    fWidth[iChannel] = 0.0;
  }
}

//_____________________________________________________________________________
AliESDVZEROfriend::~AliESDVZEROfriend()
{
  // destructor
}

//_____________________________________________________________________________
AliESDVZEROfriend::AliESDVZEROfriend(const AliESDVZEROfriend& vzerofriend):
  AliVVZEROfriend(vzerofriend),
  fTrigger(vzerofriend.fTrigger),
  fTriggerMask(vzerofriend.fTriggerMask)
{
  // copy constructor
  for (Int_t iScaler = 0; iScaler < kNScalers; iScaler++)
    fScalers[iScaler] = vzerofriend.fScalers[iScaler];

  for (Int_t iBunch = 0; iBunch < kNBunches; iBunch++)
    fBunchNumbers[iBunch] = vzerofriend.fBunchNumbers[iBunch];

  for (Int_t iChannel = 0; iChannel < kNChannels; iChannel++) {
    fBBScalers[iChannel] = vzerofriend.fBBScalers[iChannel];
    fBGScalers[iChannel] = vzerofriend.fBGScalers[iChannel];
    for (Int_t iBunch = 0; iBunch < kNBunches; iBunch++) {
      fChargeMB[iChannel][iBunch] = vzerofriend.fChargeMB[iChannel][iBunch];
      fIsIntMB[iChannel][iBunch]  = vzerofriend.fIsIntMB[iChannel][iBunch];
      fIsBBMB[iChannel][iBunch]   = vzerofriend.fIsBBMB[iChannel][iBunch];
      fIsBGMB[iChannel][iBunch]   = vzerofriend.fIsBGMB[iChannel][iBunch];
    }
    for (Int_t iEv = 0; iEv < kNEvOfInt; iEv++) {
      fADC[iChannel][iEv]   = vzerofriend.fADC[iChannel][iEv];
      fIsInt[iChannel][iEv] = vzerofriend.fIsInt[iChannel][iEv];
      fIsBB[iChannel][iEv]  = vzerofriend.fIsBB[iChannel][iEv];
      fIsBG[iChannel][iEv]  = vzerofriend.fIsBG[iChannel][iEv];
    }
    fTime[iChannel]  = vzerofriend.fTime[iChannel];
    fWidth[iChannel] = vzerofriend.fWidth[iChannel];
  }
}

//_____________________________________________________________________________
AliESDVZEROfriend& AliESDVZEROfriend::operator = (const AliESDVZEROfriend& vzerofriend)
{
  // assignment operator
  if(&vzerofriend == this) return *this;
  AliVVZEROfriend::operator=(vzerofriend);

  fTrigger = vzerofriend.fTrigger;
  fTriggerMask = vzerofriend.fTriggerMask;

  for (Int_t iScaler = 0; iScaler < kNScalers; iScaler++)
    fScalers[iScaler] = vzerofriend.fScalers[iScaler];

  for (Int_t iBunch = 0; iBunch < kNBunches; iBunch++)
    fBunchNumbers[iBunch] = vzerofriend.fBunchNumbers[iBunch];

  for (Int_t iChannel = 0; iChannel < kNChannels; iChannel++) {
    fBBScalers[iChannel] = vzerofriend.fBBScalers[iChannel];
    fBGScalers[iChannel] = vzerofriend.fBGScalers[iChannel];
    for (Int_t iBunch = 0; iBunch < kNBunches; iBunch++) {
      fChargeMB[iChannel][iBunch] = vzerofriend.fChargeMB[iChannel][iBunch];
      fIsIntMB[iChannel][iBunch]  = vzerofriend.fIsIntMB[iChannel][iBunch];
      fIsBBMB[iChannel][iBunch]   = vzerofriend.fIsBBMB[iChannel][iBunch];
      fIsBGMB[iChannel][iBunch]   = vzerofriend.fIsBGMB[iChannel][iBunch];
    }
    for (Int_t iEv = 0; iEv < kNEvOfInt; iEv++) {
      fADC[iChannel][iEv]   = vzerofriend.fADC[iChannel][iEv];
      fIsInt[iChannel][iEv] = vzerofriend.fIsInt[iChannel][iEv];
      fIsBB[iChannel][iEv]  = vzerofriend.fIsBB[iChannel][iEv];
      fIsBG[iChannel][iEv]  = vzerofriend.fIsBG[iChannel][iEv];
    }
    fTime[iChannel]  = vzerofriend.fTime[iChannel];
    fWidth[iChannel] = vzerofriend.fWidth[iChannel];
  }

  return *this;
}

void AliESDVZEROfriend::Reset()
{
  // Reset the contents of the object
  fTrigger = 0;
  fTriggerMask = 0;

  for (Int_t iScaler = 0; iScaler < kNScalers; iScaler++)
    fScalers[iScaler] = 0;

  for (Int_t iBunch = 0; iBunch < kNBunches; iBunch++)
    fBunchNumbers[iBunch] = 0;

  for (Int_t iChannel = 0; iChannel < kNChannels; iChannel++) {
    fBBScalers[iChannel] = 0;
    fBGScalers[iChannel] = 0;
    for (Int_t iBunch = 0; iBunch < kNBunches; iBunch++) {
      fChargeMB[iChannel][iBunch] = 0;
      fIsIntMB[iChannel][iBunch]  = kFALSE;
      fIsBBMB[iChannel][iBunch]   = kFALSE;
      fIsBGMB[iChannel][iBunch]   = kFALSE;
    }
    for (Int_t iEv = 0; iEv < kNEvOfInt; iEv++) {
      fADC[iChannel][iEv]   = 0.0;
      fIsInt[iChannel][iEv] = kFALSE;
      fIsBB[iChannel][iEv]  = kFALSE;
      fIsBG[iChannel][iEv]  = kFALSE;
    }
    fTime[iChannel]  = 0.0;
    fWidth[iChannel] = 0.0;
  }
  
}
