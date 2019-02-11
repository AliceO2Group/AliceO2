/**************************************************************************
 * Copyright(c) 1998-2010, ALICE Experiment at CERN, All rights reserved. *
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

//***********************************************************
// Class AliTODPIDParams
// class to store PID parameters for TOF in OADB
// Author: P. Antonioli, pietro.antonioli@to.infn.it
//***********************************************************

#include <Riostream.h>
#include <TNamed.h>
#include "AliLog.h"
#include "AliTOFPIDParams.h"

ClassImp(AliTOFPIDParams)

//_____________________________________________________________________________
AliTOFPIDParams::AliTOFPIDParams():
  TNamed("default",""),
  fStartTime(AliPIDResponse::kBest_T0),
  fTOFresolution(90),
  fTOFtail(0.95),                                 
  fTOFmatchingLossMC(0),                      
  fTOFadditionalMismForMC(0),                  
  fTOFtimeOffset(0) 
{
  fSigPparams[0]=0.008;
  fSigPparams[1]=0.008;
  fSigPparams[2]=0.002;
  fSigPparams[3]=40.;
  fOADBentryTag="default";
}

//_____________________________________________________________________________
AliTOFPIDParams::AliTOFPIDParams(Char_t *name):
  TNamed(name,""),
  fStartTime(AliPIDResponse::kBest_T0),
  fTOFresolution(90),
  fTOFtail(0.95),                                 
  fTOFmatchingLossMC(0),                      
  fTOFadditionalMismForMC(0),                  
  fTOFtimeOffset(0)                           
{
  fSigPparams[0]=0.008;
  fSigPparams[1]=0.008;
  fSigPparams[2]=0.002;
  fSigPparams[3]=40.;
  fOADBentryTag="default";
}

//_____________________________________________________________________________
AliTOFPIDParams::~AliTOFPIDParams(){
}


//_____________________________________________________________________________
void AliTOFPIDParams::SetSigPparams(Float_t *d) 
{
  //
  // Setting the SigP values
  //
  if (d == 0x0){
    AliError(Form("Null pointer passed"));
  }
  else{
    for (Int_t i=0;i<kSigPparams;i++) fSigPparams[i]=d[i];
  }
  return;
}

