#ifndef ALIESDVZEROFRIEND_H
#define ALIESDVZEROFRIEND_H
/* Copyright(c) 1998-1999, ALICE Experiment at CERN, All rights reserved. *
 * See cxx source for full Copyright notice                               */

///////////////////////////////////////////////////////////////////////////////
///
/// This is a class for containing all the VZERO DDL raw data
/// It is written to the ESD-friend file
///
///////////////////////////////////////////////////////////////////////////////

#include "AliVVZEROfriend.h"

class AliESDVZEROfriend: public AliVVZEROfriend {
  public :
    AliESDVZEROfriend();
    virtual ~AliESDVZEROfriend();

    AliESDVZEROfriend(const AliESDVZEROfriend& vzerofriend);
    AliESDVZEROfriend& operator = (const AliESDVZEROfriend& vzerofriend);

    void Reset();

// Getters of various scalers and Minimum Bias flags :

   ULong64_t          GetBBScalers(Int_t channel) const  
      { return        fBBScalers[channel]; }
   ULong64_t          GetBGScalers(Int_t channel) const  
      { return        fBGScalers[channel]; }
   UInt_t             GetTriggerScalers(Int_t num_scaler) const 
      { return        fScalers[num_scaler]; }
   UInt_t             GetBunchNumbersMB(Int_t num_bunch) const 
      { return        fBunchNumbers[num_bunch]; }
   UShort_t           GetChargeMB(Int_t channel, Int_t num_bunch) const  
      { return        fChargeMB[channel][num_bunch]; } 
   Bool_t             GetIntMBFlag(Int_t channel, Int_t num_bunch) const   
      { return        fIsIntMB[channel][num_bunch]; } 
   Bool_t             GetBBMBFlag(Int_t channel, Int_t num_bunch) const   
      { return        fIsBBMB[channel][num_bunch]; }  
   Bool_t             GetBGMBFlag(Int_t channel, Int_t num_bunch) const   
      { return        fIsBGMB[channel][num_bunch]; }      
       
// Getters of ADC signals, ADC pedestals, time information and corresponding flags :

    Float_t           GetADC(Int_t channel) const
      { return fADC[channel][kNEvOfInt/2]; }
    Float_t           GetPedestal(Int_t channel, Int_t event) const
      { return fADC[channel][event]; }
    Bool_t            GetIntegratorFlag(Int_t channel, Int_t event) const
      { return fIsInt[channel][event]; }
    Bool_t            GetBBFlag(Int_t channel, Int_t event) const
      { return fIsBB[channel][event]; } 
    Bool_t            GetBGFlag(Int_t channel, Int_t event) const
      { return fIsBG[channel][event]; }   
    Float_t            GetTime(Int_t channel) const
      { return fTime[channel]; }
    Float_t            GetWidth(Int_t channel) const
      { return fWidth[channel]; }

    // Setters
    void              SetBBScalers(Int_t channel, ULong64_t scalers)
      { fBBScalers[channel] = scalers; }
    void              SetBGScalers(Int_t channel, ULong64_t scalers)
      { fBGScalers[channel] = scalers; }
    void              SetTriggerScalers(Int_t num_scaler, UInt_t scaler)
      { fScalers[num_scaler] = scaler; }
    void              SetBunchNumbersMB(Int_t num_bunch, UInt_t bunch)
      { fBunchNumbers[num_bunch] = bunch; }
    void              SetChargeMB(Int_t channel,Int_t num_bunch, UShort_t charge)
      { fChargeMB[channel][num_bunch] = charge; }
    void              SetIntMBFlag(Int_t channel,Int_t num_bunch, Bool_t flag)
      { fIsIntMB[channel][num_bunch] = flag; }
    void              SetBBMBFlag(Int_t channel,Int_t num_bunch, Bool_t flag)
      { fIsBBMB[channel][num_bunch] = flag; }
    void              SetBGMBFlag(Int_t channel,Int_t num_bunch, Bool_t flag)
      { fIsBGMB[channel][num_bunch] = flag; }

    void              SetPedestal(Int_t channel, Int_t event, Float_t adc)
      { fADC[channel][event] = adc; }
    void              SetIntegratorFlag(Int_t channel, Int_t event, Bool_t flag)
      { fIsInt[channel][event] = flag; }
    void              SetBBFlag(Int_t channel, Int_t event, Bool_t flag)
      { fIsBB[channel][event] = flag; }
    void              SetBGFlag(Int_t channel, Int_t event, Bool_t flag)
      { fIsBG[channel][event] = flag; }
    void              SetTime(Int_t channel, Float_t time)
      { fTime[channel] = time; }
    void              SetWidth(Int_t channel, Float_t width)
      { fWidth[channel] = width; }

    UShort_t          GetTriggerInputs() const
      { return fTrigger; }
    UShort_t          GetTriggerInputsMask() const
      { return fTriggerMask; }
    void              SetTriggerInputs(UShort_t inputs)
      { fTrigger = inputs; }
    void              SetTriggerInputsMask(UShort_t mask)
      { fTriggerMask = mask; }

  private:

    ULong64_t     fBBScalers[kNChannels];        // 'Beam-Beam' scalers for all channels
    ULong64_t     fBGScalers[kNChannels];        // 'Beam-Gas' scalers for all channels
    UInt_t        fScalers[kNScalers];           // Trigger scalers
    UInt_t        fBunchNumbers[kNBunches];      // Bunch numbers for the previous 10 MB events
    UShort_t      fChargeMB[kNChannels][kNBunches]; // ADC counts for all channels for the previous 10 MB events
    Bool_t        fIsIntMB[kNChannels][kNBunches];  // 'Integrator' flag for all channels for the previous 10 MB events
    Bool_t        fIsBBMB[kNChannels][kNBunches];   // 'Beam-Beam' flag for all channels for the previous 10 MB events
    Bool_t        fIsBGMB[kNChannels][kNBunches];   // 'Beam-Gas' for all channels for the previous 10 MB events

    Float_t       fADC[kNChannels][kNEvOfInt];   // ADC counts for all channels and all events of interest
    Bool_t        fIsInt[kNChannels][kNEvOfInt]; // 'Integrator' flag for all channels 
    Bool_t        fIsBB[kNChannels][kNEvOfInt];  // 'Beam-Beam' flag for all channels
    Bool_t        fIsBG[kNChannels][kNEvOfInt];  // 'Beam-Gas' flag for all channels
    Float_t       fTime[kNChannels];             // leading time for all channels - from HPTDC - in nanoseconds
    Float_t       fWidth[kNChannels];            // pulse width for all channels - from HPTDC - in nanoseconds

    UShort_t      fTrigger;        // VZERO trigger inputs
    UShort_t      fTriggerMask;    // VZERO trigger inputs mask

    ClassDef(AliESDVZEROfriend, 3) // container class for VZERO DDL raw data
};

#endif
