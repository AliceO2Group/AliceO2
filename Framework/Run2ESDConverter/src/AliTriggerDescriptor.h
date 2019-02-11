#ifndef ALITRIGGERDESCRIPTOR_H
#define ALITRIGGERDESCRIPTOR_H

/* Copyright(c) 1998-1999, ALICE Experiment at CERN, All rights reserved. *
 * See cxx source for full Copyright notice                               */

///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// This class represents the CTP descriptor objects                          //
//                                                                           //
// The Descriptor consists of Name and 1 or Logical function of 4 CTP inputs+//
// additional inputs (added with an AND operator)                            // 
//                                                                           //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#include <TNamed.h>


class AliTriggerDescriptor : public TNamed {

public:
                          AliTriggerDescriptor();
                          AliTriggerDescriptor( TString & name, TString & cond );
                          AliTriggerDescriptor( const AliTriggerDescriptor& desc );
               virtual   ~AliTriggerDescriptor();
  AliTriggerDescriptor&   operator=(const AliTriggerDescriptor& desc);

                  Bool_t  CheckInputsAndFunctions(const TObjArray &inputs,const TObjArray &functions) const;
                  Bool_t  IsActive(const TObjArray &inputs,const TObjArray &functions) const;
		  Bool_t  Trigger( const TObjArray &inputs, const TObjArray &functions) const;
           virtual void   Print( const Option_t* opt ="" ) const;
private:

  ClassDef( AliTriggerDescriptor, 1 )  // Define a trigger descriptor object
};

#endif
