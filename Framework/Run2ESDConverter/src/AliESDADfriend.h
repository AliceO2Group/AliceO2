#ifndef ALIESDADFRIEND_H
#define ALIESDADFRIEND_H
/* Copyright(c) 1998-1999, ALICE Experiment at CERN, All rights reserved. *
 * See cxx source for full Copyright notice                               */

///////////////////////////////////////////////////////////////////////////////
///
/// This is a class for containing all the AD DDL raw data
/// It is written to the ESD-friend file
///
///////////////////////////////////////////////////////////////////////////////

#include <TObject.h>

class AliESDADfriend: public TObject {
  public :
    AliESDADfriend();
    virtual ~AliESDADfriend();

    AliESDADfriend(const AliESDADfriend& adfriend);
    AliESDADfriend& operator = (const AliESDADfriend& adfriend);

    void Reset();

// Getters of various scalers and Minimum Bias flags :

   ULong64_t          GetBBScalers(Int_t channel) const  
      { return        fBBScalers[channel]; }
   ULong64_t          GetBGScalers(Int_t channel) const  
      { return        fBGScalers[channel]; }
   UInt_t             GetTriggerScalers(Int_t num_scaler) const 
      { return        fScalers[num_scaler]; }
   
       
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

    enum EESDADfriendParams {
      kNChannels = 16, // number of electronic channels in AD (FEE numbering)
      kNEvOfInt  = 21, // number of events of interest
      kNScalers  = 16, // number of scalers
    };

  private:

    ULong64_t     fBBScalers[kNChannels];        // 'Beam-Beam' scalers for all channels
    ULong64_t     fBGScalers[kNChannels];        // 'Beam-Gas' scalers for all channels
    UInt_t        fScalers[kNScalers];           // Trigger scalers

    Float_t       fADC[kNChannels][kNEvOfInt];   // ADC counts for all channels and all events of interest
    Bool_t        fIsInt[kNChannels][kNEvOfInt]; // 'Integrator' flag for all channels 
    Bool_t        fIsBB[kNChannels][kNEvOfInt];  // 'Beam-Beam' flag for all channels
    Bool_t        fIsBG[kNChannels][kNEvOfInt];  // 'Beam-Gas' flag for all channels
    Float_t       fTime[kNChannels];             // leading time for all channels - from HPTDC - in nanoseconds
    Float_t       fWidth[kNChannels];            // pulse width for all channels - from HPTDC - in nanoseconds

    UShort_t      fTrigger;        // AD trigger inputs
    UShort_t      fTriggerMask;    // AD trigger inputs mask

    ClassDef(AliESDADfriend, 2) // container class for AD DDL raw data
};

#endif
