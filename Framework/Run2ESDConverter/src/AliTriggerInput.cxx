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

/* $Id$ */

///////////////////////////////////////////////////////////////////////////////
//
//  Class to define a Trigger Input from an specific detector                                                                                           //
//
//
//                        name         description     id mask
//    Ej:
//      AliTriggerInput( "V0_MB_L0", "VO minimum bias", 0x01 );
//      AliTriggerInput( "V0_SC_L0", "VO semi central", 0x02 );
//      AliTriggerInput( "V0_C_L0",  "VO central",      0x04 );
//
//    The name must be globaly unique. Spaces are not allowed.
//    As convention should start with detector name then an id
//    and the trigger level (L0, L1, L2) separated by "_"
//    for valid detector names see AliTriggerCluster::fgkDetectorName
//
//    A maximun of 60 inputs trigger are allow.
//    So, the id mask should set only bit from the position 1 to 60.
//
///////////////////////////////////////////////////////////////////////////////

#include <Riostream.h>
#include <TMath.h>

#include "AliLog.h"
#include "AliTriggerInput.h"

using std::endl;
using std::cout;
ClassImp( AliTriggerInput )

Bool_t AliTriggerInput::fgkIsTriggerDetector[AliDAQ::kNDetectors] = {1,0,0,1,1,1,0,1,0,0,0,1,0,1,1,1,1,1,1,1,0,1,0,0,0};
const char* AliTriggerInput::fgkCTPDetectorName[AliDAQ::kNDetectors] = {
  "SPD",
  "SDD",
  "SSD",
  "TPC",
  "TRD",
  "TOF",
  "HMPID",
  "PHOS",
  "CPV",
  "PMD",
  "MUON_TRK",
  "MUON_TRG",
  "FMD",
  "T0",
  "V0",
  "ZDC",
  "ACORDE",
  "CTP",
  "EMCal",
  "DAQ_TEST",
  "EMPTY",
  "AD",
  "MFT",
  "FIT",
  "HLT"
};
//_____________________________________________________________________________
  AliTriggerInput::AliTriggerInput():
    TNamed(),
    fMask( 0 ),
    fValue( 0 ),
    fSignature( -1),
    fLevel(0),
    fDetectorId(-1),
    fIsActive(kFALSE),
    fIndexCTP(0),
    fIndexSwitch(0)
{}
//____________________________________________________________________________
 AliTriggerInput::AliTriggerInput( AliTriggerInput & inp ): 
   TNamed( inp ),
   fMask( inp.fMask ),
   fValue( inp.fValue ),
   fSignature( inp.fSignature ),
   fLevel( inp.fLevel),
   fDetectorId( inp.fDetectorId),
   fIsActive(kFALSE),
   fIndexCTP(inp.fIndexCTP),
   fIndexSwitch(inp.fIndexSwitch)
   {}
//_____________________________________________________________________________
  AliTriggerInput::AliTriggerInput( TString name, TString det, UChar_t level, Int_t signature, Char_t number ):
    TNamed( name.Data(), det.Data() ),
    fMask((number > 0) ? 1 << (number-1) : 0 ),
    fValue(0),
    fSignature(signature),
    fLevel(level),
    fDetectorId(-1),
    fIsActive(kFALSE),
    fIndexCTP(number),
    fIndexSwitch(0)
{
   //  Standard constructor
   //
   //    The name must be globaly unique. Spaces are not allowed.
   //    For valid detector names see AliDAQ::fgkDetectorName

   // Check for valid detector name
   Int_t iDet = 0;
   for( iDet = 0; iDet < AliDAQ::kNDetectors; iDet++ ) {
     if( !fgkIsTriggerDetector[iDet] ) continue;
      if( det.CompareTo( fgkCTPDetectorName[iDet] ) == 0 ) {
	fTitle = AliDAQ::DetectorName(iDet);
	fDetectorId = iDet;
	break;
      }
      if( det.CompareTo( AliDAQ::DetectorName(iDet) ) == 0 ) {
	fDetectorId = iDet;
	break;
      }
   }
   if( iDet == AliDAQ::kNDetectors ) {
      AliError( Form( "%s is not a valid trigger input, it must contain a valid trigger detector name instead of (%s)", name.Data(), det.Data() ) );
   }
}
//_____________________________________________________________________________
  AliTriggerInput::AliTriggerInput( TString name, TString det, UChar_t level, Int_t signature, UInt_t indexCTP, UInt_t indexSwitch ):
    TNamed( name.Data(), det.Data() ),
    fMask((indexCTP > 0) ? 1 << (indexCTP-1) : 0 ),
    fValue(0),
    fSignature(signature),
    fLevel(level),
    fDetectorId(-1),
    fIsActive(kFALSE),
    fIndexCTP(indexCTP),
    fIndexSwitch(indexSwitch)
{
   //  Standard constructor
   //
   //    The name must be globaly unique. Spaces are not allowed.
   //    For valid detector names see AliDAQ::fgkDetectorName

   // Check for valid detector name
   Int_t iDet = 0;
   for( iDet = 0; iDet < AliDAQ::kNDetectors; iDet++ ) {
     if( !fgkIsTriggerDetector[iDet] ) continue;
      if( det.CompareTo( fgkCTPDetectorName[iDet] ) == 0 ) {
	fTitle = AliDAQ::DetectorName(iDet);
	fDetectorId = iDet;
	break;
      }
      if( det.CompareTo( AliDAQ::DetectorName(iDet) ) == 0 ) {
	fDetectorId = iDet;
	break;
      }
   }
   if( iDet == AliDAQ::kNDetectors ) {
      AliError( Form( "%s is not a valid trigger input, it must contain a valid trigger detector name instead of (%s)", name.Data(), det.Data() ) );
   }
}
//_____________________________________________________________________________
void AliTriggerInput::Print( const Option_t* ) const
{
   // Print
   cout << "Trigger Input:" << endl; 
   cout << "  Name:        " << GetName() << endl;
   cout << "  Detector:    " << GetTitle() << "(Id=" << (Int_t)fDetectorId << ")" << endl;
   cout << "  Level:       " << (Int_t)fLevel << endl;
   cout << "  Signature:   " << fSignature << endl;
   cout << "  CTP index:      " << fIndexCTP << endl;
   cout << "  Switch index:      " << fIndexSwitch << endl;
   if (IsActive())
     cout << "   Input is active      " << endl;
   else
     cout << "   Input is not active  " << endl;
   if (Status())
     cout << "   Input is fired      " << endl;
   else
     cout << "   Input is not fired  " << endl;
}

//_____________________________________________________________________________
TString AliTriggerInput::GetModule() const
{
  // Get the detector module name (in AliRoot simulation sense)
  TString name = "";
  if (fDetectorId >= 0 && fDetectorId < AliDAQ::kNDetectors)
    name = AliDAQ::OfflineModuleName((Int_t)fDetectorId);
  else
    AliError(Form("Invalid detector Id (%d)",(Int_t)fDetectorId));

  return name;
}
UInt_t AliTriggerInput::GetIndexCTP() const
{
 if(fIndexCTP==0){
   for(UInt_t i=0;i<24;i++){
     if((1<<i) & fMask) return (i+1);
   }
  return 0;
 }
 return fIndexCTP;
}
