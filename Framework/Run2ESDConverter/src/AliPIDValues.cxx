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

///////////////////////////////////////////////////////////////////////////
//                       PID Values                                      //
//                                                                       //
//                                                                       //
/*

Class to store PID information for each particle species

*/
//                                                                       //
///////////////////////////////////////////////////////////////////////////


#include "AliPIDValues.h"

ClassImp(AliPIDValues)

AliPIDValues::AliPIDValues() :
  TObject(),
  fPIDStatus(AliPIDResponse::kDetPidOk)
{
  //
  // default constructor
  //
  Int_t nspecies=AliPID::kSPECIESCN;
  for (Int_t i=0; i<nspecies; ++i) fValues[i]=0.;
}

//_______________________________________________________________________
AliPIDValues::AliPIDValues(const AliPIDValues &val) :
  TObject(val),
  fPIDStatus(val.fPIDStatus)
{
  //
  // copy constructor
  //
  Int_t nspecies=AliPID::kSPECIESCN;
  for (Int_t i=0; i<nspecies; ++i) fValues[i]=val.fValues[i];
}

//_______________________________________________________________________
AliPIDValues::AliPIDValues(Double_t val[], Int_t nspecies, AliPIDResponse::EDetPidStatus status) :
  TObject(),
  fPIDStatus(AliPIDResponse::kDetPidOk)
{
  //
  // constructor with array of values
  //
  SetValues(val,nspecies,status);
}

//_______________________________________________________________________
AliPIDValues& AliPIDValues::operator= (const AliPIDValues &val)
{
  //
  // assignment operator
  //
  if (this!=&val){
    TObject::operator=(val);
    
    Int_t nspecies=AliPID::kSPECIESCN;
    for (Int_t i=0; i<nspecies; ++i) fValues[i]=val.fValues[i];
    fPIDStatus=val.fPIDStatus;
  }

  return *this;
}

//_______________________________________________________________________
void AliPIDValues::Copy(TObject &obj) const {
  // this overwrites the virtual TObject::Copy()
  // to allow run time copying without casting
  // in AliPIDValues
  
  if(this==&obj)return;
  AliPIDValues *robj = dynamic_cast<AliPIDValues*>(&obj);
  if(!robj)return; // not AliPIDValues
  *robj = *this;
}

//_______________________________________________________________________
void AliPIDValues::SetValues(const Double_t val[], Int_t nspecies, AliPIDResponse::EDetPidStatus status)
{
  //
  // set array of values
  //
  if (nspecies>AliPID::kSPECIESCN) nspecies=AliPID::kSPECIESCN;
  for (Int_t i=0; i<nspecies; ++i) fValues[i]=val[i];
  fPIDStatus=status;
}

//_______________________________________________________________________
AliPIDResponse::EDetPidStatus AliPIDValues::GetValues(Double_t val[], Int_t nspecies) const
{
  //
  // get array of values
  //
  if (nspecies>AliPID::kSPECIESCN) nspecies=AliPID::kSPECIESCN;
  for (Int_t i=0; i<nspecies; ++i) val[i]=fValues[i];
  return fPIDStatus;
}

//_______________________________________________________________________
Double_t AliPIDValues::GetValue(AliPID::EParticleType type) const
{
  //
  // get values for a specific particle type
  //
  return fValues[(Int_t)type];
}

