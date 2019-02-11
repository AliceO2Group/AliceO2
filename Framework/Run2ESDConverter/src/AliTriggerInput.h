#ifndef ALITRIGGERINPUT_H
#define ALITRIGGERINPUT_H

/* Copyright(c) 1998-1999, ALICE Experiment at CERN, All rights reserved. *
 * See cxx source for full Copyright notice                               */

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

//    The name must be globaly unique. Spaces are not allowed.
//    As convention should start with detector name then an id
//    and the trigger level (L0, L1, L2)
//
//    A maximun of 60 inputs trigger are allow.
//    So, the id mask should set only bit from the position 1 to 60.
//
///////////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TNamed
#include <TNamed.h>
#endif

#include "AliDAQ.h"

class AliTriggerInput : public TNamed {

public:
                          AliTriggerInput();
                          AliTriggerInput( TString name, TString det, UChar_t level, Int_t signature = -1, Char_t number = -1);
                          AliTriggerInput( TString name, TString det, UChar_t level, Int_t signature, UInt_t indexCTP, UInt_t indexSwitch);
                          AliTriggerInput( AliTriggerInput & inp );
               virtual   ~AliTriggerInput() {}

  //  Setters
                  void    Set()   { if (fIsActive) fValue = fMask; }
                  void    Reset() { fValue = 0; }
         	  void    Enable() { fIsActive = kTRUE; }

  //  Getters
                Bool_t    Status() const   { return (Bool_t)fValue; }
             ULong64_t    GetValue() const { return fValue; }
             ULong64_t    GetMask() const  { return fMask; }
	        Int_t     GetSignature() const { return fSignature; }
	       TString    GetInputName() const { return GetName(); }
               TString    GetDetector() const { return GetTitle(); }
	       TString    GetModule() const;
	        Char_t    GetDetectorId() const { return fDetectorId; }
	       UChar_t    GetLevel() const { return fLevel; }
	        Bool_t    IsActive() const { return fIsActive; }
		UInt_t    GetIndexCTP() const;
		UInt_t    GetIndexSwitch() const { return fIndexSwitch; }

           virtual void   Print( const Option_t* opt ="" ) const;

	   static Bool_t  fgkIsTriggerDetector[AliDAQ::kNDetectors]; // List of trigger detectors
   static  const char*    fgkCTPDetectorName[AliDAQ::kNDetectors];
protected:
             ULong64_t    fMask;                //  Trigger ID mask (1 bit)
             ULong64_t    fValue;               //  Trigger Signal (0 = false, > 1 = true = fMask )
		Int_t     fSignature;           //  8 bit signature (internal CTP inputs can have longer signature)
		UChar_t   fLevel;               //  L0, L1 or L2
		Char_t    fDetectorId;          //  Alice-wide detector id, see AliDAQ class for details
	        Bool_t    fIsActive;            //  Is trigger input active (during simulation)
                UInt_t    fIndexCTP;            // input position as seen at CTP (fMask=(1<<(fIndexCTP-1)) [1..24]      
                UInt_t    fIndexSwitch;         // input position in connector to CTP board (as seen in IR2) [1..48]    
//                  void    fDectParameterTable;  //-> link to detector parameter table????

   ClassDef( AliTriggerInput, 5 )  // Define a Trigger Input
};


#endif
