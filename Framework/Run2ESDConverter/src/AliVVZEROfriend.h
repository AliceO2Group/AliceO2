#ifndef ALIVVZEROFRIEND_H
#define ALIVVZEROFRIEND_H
/* Copyright(c) 1998-1999, ALICE Experiment at CERN, All rights reserved. *
 * See cxx source for full Copyright notice                               */

///////////////////////////////////////////////////////////////////////////////
///
/// This is a base class for containing all the VZERO DDL raw data
///
///////////////////////////////////////////////////////////////////////////////

#include <TObject.h>

class AliVVZEROfriend: public TObject {
  public :
    AliVVZEROfriend() {}
    virtual ~AliVVZEROfriend() {}

    AliVVZEROfriend(const AliVVZEROfriend& vzerofriend);
    AliVVZEROfriend& operator = (const AliVVZEROfriend& vzerofriend);

    virtual void Reset() {}

// Getters of various scalers and Minimum Bias flags :

   virtual ULong64_t          GetBBScalers(Int_t /*channel*/) const  
      { return        0; }
   virtual ULong64_t          GetBGScalers(Int_t /*channel*/) const  
      { return        0; }
   virtual UInt_t             GetTriggerScalers(Int_t /*num_scaler*/) const 
      { return        0; }
   virtual UInt_t             GetBunchNumbersMB(Int_t /*num_bunch*/) const 
      { return        0; }
   virtual UShort_t           GetChargeMB(Int_t /*channel*/, Int_t /*num_bunch*/) const  
      { return        0; } 
   virtual Bool_t             GetIntMBFlag(Int_t /*channel*/, Int_t /*num_bunch*/) const   
      { return        0; } 
   virtual Bool_t             GetBBMBFlag(Int_t /*channel*/, Int_t /*num_bunch*/) const   
      { return        0; }  
   virtual Bool_t             GetBGMBFlag(Int_t /*channel*/, Int_t /*num_bunch*/) const   
      { return        0; }      
       
// Getters of ADC signals, ADC pedestals, time information and corresponding flags :

    virtual Float_t           GetADC(Int_t /*channel*/) const
      { return 0.; }
    virtual Float_t           GetPedestal(Int_t /*channel*/, Int_t /*event*/) const
      { return 0.; }
    virtual Bool_t            GetIntegratorFlag(Int_t /*channel*/, Int_t /*event*/) const
      { return 0; }
    virtual Bool_t            GetBBFlag(Int_t /*channel*/, Int_t /*event*/) const
      { return 0; } 
    virtual Bool_t            GetBGFlag(Int_t /*channel*/, Int_t /*event*/) const
      { return 0; }   
    virtual Float_t            GetTime(Int_t /*channel*/) const
      { return 0.; }
    virtual Float_t            GetWidth(Int_t /*channel*/) const
      { return 0.; }

    enum EVVZEROfriendParams {
      kNChannels = 64, // number of electronic channels in V0 (FEE numbering)
      kNEvOfInt  = 21, // number of events of interest
      kNScalers  = 16, // number of scalers
      kNBunches  = 10  // number of bunches used in Minimum Bias information 
    };

  private:

    ClassDef(AliVVZEROfriend, 1) // container class for VZERO DDL raw data
};

#endif
