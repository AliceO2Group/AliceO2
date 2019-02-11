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
/// This is a class for containing all the AD DDL raw data
/// It is written to the ESD-friend file
///
///////////////////////////////////////////////////////////////////////////////

#include "AliESDADfriend.h"

ClassImp(AliESDADfriend)

//_____________________________________________________________________________
AliESDADfriend::AliESDADfriend():
  TObject(),
  fTrigger(0),
  fTriggerMask(0)
{
  // default constructor
  for (Int_t iScaler = 0; iScaler < kNScalers; iScaler++)
    fScalers[iScaler] = 0;

  for (Int_t iChannel = 0; iChannel < kNChannels; iChannel++) {
    fBBScalers[iChannel] = 0;
    fBGScalers[iChannel] = 0;
    
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
AliESDADfriend::~AliESDADfriend()
{
  // destructor
}

//_____________________________________________________________________________
AliESDADfriend::AliESDADfriend(const AliESDADfriend& adfriend):
  TObject(adfriend),
  fTrigger(adfriend.fTrigger),
  fTriggerMask(adfriend.fTriggerMask)
{
  // copy constructor
  for (Int_t iScaler = 0; iScaler < kNScalers; iScaler++)
    fScalers[iScaler] = adfriend.fScalers[iScaler];

  for (Int_t iChannel = 0; iChannel < kNChannels; iChannel++) {
    fBBScalers[iChannel] = adfriend.fBBScalers[iChannel];
    fBGScalers[iChannel] = adfriend.fBGScalers[iChannel];
    
    for (Int_t iEv = 0; iEv < kNEvOfInt; iEv++) {
      fADC[iChannel][iEv]   = adfriend.fADC[iChannel][iEv];
      fIsInt[iChannel][iEv] = adfriend.fIsInt[iChannel][iEv];
      fIsBB[iChannel][iEv]  = adfriend.fIsBB[iChannel][iEv];
      fIsBG[iChannel][iEv]  = adfriend.fIsBG[iChannel][iEv];
    }
    fTime[iChannel]  = adfriend.fTime[iChannel];
    fWidth[iChannel] = adfriend.fWidth[iChannel];
  }
}

//_____________________________________________________________________________
AliESDADfriend& AliESDADfriend::operator = (const AliESDADfriend& adfriend)
{
  // assignment operator
  if(&adfriend == this) return *this;
  TObject::operator=(adfriend);

  fTrigger = adfriend.fTrigger;
  fTriggerMask = adfriend.fTriggerMask;

  for (Int_t iScaler = 0; iScaler < kNScalers; iScaler++)
    fScalers[iScaler] = adfriend.fScalers[iScaler];

  for (Int_t iChannel = 0; iChannel < kNChannels; iChannel++) {
    fBBScalers[iChannel] = adfriend.fBBScalers[iChannel];
    fBGScalers[iChannel] = adfriend.fBGScalers[iChannel];
    
    for (Int_t iEv = 0; iEv < kNEvOfInt; iEv++) {
      fADC[iChannel][iEv]   = adfriend.fADC[iChannel][iEv];
      fIsInt[iChannel][iEv] = adfriend.fIsInt[iChannel][iEv];
      fIsBB[iChannel][iEv]  = adfriend.fIsBB[iChannel][iEv];
      fIsBG[iChannel][iEv]  = adfriend.fIsBG[iChannel][iEv];
    }
    fTime[iChannel]  = adfriend.fTime[iChannel];
    fWidth[iChannel] = adfriend.fWidth[iChannel];
  }

  return *this;
}

void AliESDADfriend::Reset()
{
  // Reset the contents of the object
  fTrigger = 0;
  fTriggerMask = 0;

  for (Int_t iScaler = 0; iScaler < kNScalers; iScaler++)
    fScalers[iScaler] = 0;

  for (Int_t iChannel = 0; iChannel < kNChannels; iChannel++) {
    fBBScalers[iChannel] = 0;
    fBGScalers[iChannel] = 0;
    
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
