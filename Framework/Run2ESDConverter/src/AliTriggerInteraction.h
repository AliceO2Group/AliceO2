#ifndef ALITRIGGERINTERACTION_H
#define ALITRIGGERINTERACTION_H

/* Copyright(c) 1998-1999, ALICE Experiment at CERN, All rights reserved. *
 * See cxx source for full Copyright notice                               */

///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// This class represents the CTP interaction objects                         //
//                                                                           //
// The Interaction consists of Name and Logical function of 4 CTP inputs.    //
//                                                                           //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#include <TNamed.h>


class AliTriggerInteraction : public TNamed {

public:
                          AliTriggerInteraction();
                          AliTriggerInteraction( TString & name, TString & logic );
                          AliTriggerInteraction( const AliTriggerInteraction& interact );
               virtual   ~AliTriggerInteraction();
  AliTriggerInteraction&   operator=(const AliTriggerInteraction& interact);

                  Bool_t  CheckInputs(const TObjArray &inputs) const;
                  Bool_t  IsActive(const TObjArray &inputs) const;
		  Bool_t  Trigger(const TObjArray& inputs ) const;
           virtual void   Print( const Option_t* opt ="" ) const;
private:

  ClassDef( AliTriggerInteraction, 1 )  // Define a trigger interaction object
};

#endif
