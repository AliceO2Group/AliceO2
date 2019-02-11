#ifndef ALITRIGGERSCALERSRECORDESD_H
#define ALITRIGGERSCALERSRECORDESD_H

/* Copyright(c) 1998-1999, ALICE Experiment at CERN, All rights reserved. *
* See cxx source for full Copyright notice */
/* $Id$ */

///////////////////////////////////////////////////////////////////////////////
//
// Class to define the ALICE Trigger Scalers Record 
//
// Each record consists of 1 time stamp (4 words)  (AliTimeStamp)
// and an array with the scalers (AliTriggerScalers) for each trigger class 
// in partition  
//
//////////////////////////////////////////////////////////////////////////////
#include <TObjArray.h>
#include "AliTimeStamp.h"
class TObjArray;
class AliTimeStamp;
class AliTriggerScalersESD;

class AliTriggerScalersRecordESD : public TObject {

public:
                              AliTriggerScalersRecordESD();
                   virtual   ~AliTriggerScalersRecordESD() { fScalers.SetOwner(); fScalers.Delete(); }       
                      void    Reset();
                      void    AddTriggerScalers( AliTriggerScalersESD* scaler );
                      void    AddTriggerScalers( UChar_t classIndex, ULong64_t LOCB, ULong64_t LOCA,        
                                              ULong64_t L1CB, ULong64_t L1CA, ULong64_t L2CB, ULong64_t L2CA );                            
           const TObjArray*   GetTriggerScalers()  const { return  &fScalers; }
const AliTriggerScalersESD*   GetTriggerScalersForClass( const Int_t classindex ) const;       
        const AliTimeStamp*   GetTimeStamp() const {return &fTimestamp;};
	             UInt_t   GetTimeGroup() const {return fTimeGroup; };
                       void   SetTimeStamp(const AliTimeStamp *stamp){fTimestamp = *stamp;};
		       void   SetTimeGroup(const UInt_t group){fTimeGroup = group;};
               virtual void   Print( const Option_t* opt ="" ) const;
                      
                           AliTriggerScalersRecordESD( const AliTriggerScalersRecordESD &rec );
                      AliTriggerScalersRecordESD&   operator=(const AliTriggerScalersRecordESD& rec);      
               

private:  
                        AliTimeStamp fTimestamp;    // record timestamp
                        TObjArray    fScalers;      // Array of scalers (AliTriggerScalersESD) 
			      UInt_t fTimeGroup;    // Active time group
                        ClassDef( AliTriggerScalersRecordESD, 3 )  // Define a Record of Trigger Scalers 
};

#endif
