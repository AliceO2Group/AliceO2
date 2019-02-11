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

// $Id$

//-----------------------------------------------------------------------------
/// \class AliESDMuonPad
///
/// Class to describe the MUON pads in the Event Summary Data
///
/// \author Philippe Pillot, Subatech
//-----------------------------------------------------------------------------

#include "AliESDMuonPad.h"

#include "AliLog.h"

#include <Riostream.h>

using std::endl;
using std::cout;
/// \cond CLASSIMP
ClassImp(AliESDMuonPad)
/// \endcond

//_____________________________________________________________________________
AliESDMuonPad::AliESDMuonPad()
: TObject(),
  fADC(0),
  fCharge(0.)
{
  /// default constructor
}

//_____________________________________________________________________________
AliESDMuonPad::AliESDMuonPad (const AliESDMuonPad& pad)
: TObject(pad),
  fADC(pad.fADC),
  fCharge(pad.fCharge)
{
  /// Copy constructor
}

//_____________________________________________________________________________
AliESDMuonPad& AliESDMuonPad::operator=(const AliESDMuonPad& pad)
{
  /// Equal operator
  if (this == &pad) return *this;
  
  TObject::operator=(pad); // don't forget to invoke the base class' assignment operator
  
  fADC = pad.fADC;
  fCharge = pad.fCharge;
  
  return *this;
}

//_____________________________________________________________________________
void AliESDMuonPad::Copy(TObject &obj) const {
  
  /// This overwrites the virtual TOBject::Copy()
  /// to allow run time copying without casting
  /// in AliESDEvent

  if(this==&obj)return;
  AliESDMuonPad *robj = dynamic_cast<AliESDMuonPad*>(&obj);
  if(!robj)return; // not an AliESDMuonPad
  *robj = *this;

}

//_____________________________________________________________________________
void AliESDMuonPad::Print(Option_t */*option*/) const
{
  /// print cluster content
  UInt_t cId = GetUniqueID();
  
  cout<<Form("padID=%u (det=%d, manu=%d, manuChannel=%d, cathode=%d)",
	     cId, GetDetElemId(), GetManuId(), GetManuChannel(), GetCathode())<<endl;
  
  cout<<Form("    raw charge=%d, calibrated charge=%5.2f", GetADC(), GetCharge())<<endl;
}

