#ifndef ALIESDAD_H
#define ALIESDAD_H

//-------------------------------------------------------------------------
//     Container class for ESD AD data
//     Author: Michal Broz
//     Michal.Broz@cern.ch
//-------------------------------------------------------------------------

#include <AliVAD.h>

class AliESDAD : public AliVAD 
{
public:
  AliESDAD();
  AliESDAD(const AliESDAD&o);
  AliESDAD(UInt_t BBtriggerADA,   UInt_t BGtriggerADA,
	      UInt_t BBtriggerADC,   UInt_t BGtriggerADC,
	      Float_t *Multiplicity, Float_t *Adc, 
	      Float_t *Time, Float_t *Width, Bool_t *BBFlag, Bool_t *BGFlag);
  void Copy(TObject &obj) const;

  virtual ~AliESDAD() {};

  // Setters
  void SetBBtriggerADA(UInt_t BBtrigger) {fBBtriggerADA=BBtrigger;}
  void SetBGtriggerADA(UInt_t BGtrigger) {fBGtriggerADA=BGtrigger;}
  void SetBBtriggerADC(UInt_t BBtrigger) {fBBtriggerADC=BBtrigger;}
  void SetBGtriggerADC(UInt_t BGtrigger) {fBGtriggerADC=BGtrigger;}
  void SetMultiplicity(Float_t Multiplicity[16])
    {for(Int_t i=0;i<16;i++) fMultiplicity[i]=Multiplicity[i];}
  void SetADC(Float_t adc[16])
    {for(Int_t i=0;i<16;i++) fAdc[i]=adc[i];}
  void SetTime(Float_t time[16])
    {for(Int_t i=0;i<16;i++) fTime[i]=time[i];}
  void SetWidth(Float_t width[16])
    {for(Int_t i=0;i<16;i++) fWidth[i]=width[i];}    
  void SetBBFlag(Bool_t BBFlag[16])
    {for(Int_t i=0;i<16;i++) fBBFlag[i]=BBFlag[i];} 
  void SetBGFlag(Bool_t BGFlag[16])
    {for(Int_t i=0;i<16;i++) fBGFlag[i]=BGFlag[i];}   

  void SetADATime(Float_t time) {fADATime = time;}
  void SetADCTime(Float_t time) {fADCTime = time;}
  void SetADATimeError(Float_t err) {fADATimeError = err;}
  void SetADCTimeError(Float_t err) {fADCTimeError = err;}

  void SetADADecision(Decision des) {fADADecision = des;}
  void SetADCDecision(Decision des) {fADCDecision = des;}

  void SetTriggerChargeA(UShort_t chargeA) {fTriggerChargeA = chargeA;}
  void SetTriggerChargeC(UShort_t chargeC) {fTriggerChargeC = chargeC;}
  void SetTriggerBits(UShort_t triggerBits) {fTriggerBits = triggerBits;}
  
  void SetPFBBFlag(Int_t channel, Int_t clock, Bool_t flag) { fIsBB[channel][clock] = flag; }
  void SetPFBGFlag(Int_t channel, Int_t clock, Bool_t flag) { fIsBG[channel][clock] = flag; }
  
  void SetADCTail(Float_t adc[16])
    {for(Int_t i=0;i<16;i++) fAdcTail[i]=adc[i];}
    
  void SetADCTrigger(Float_t adc[16])
    {for(Int_t i=0;i<16;i++) fAdcTrigger[i]=adc[i];}

  // Getters  
  virtual Short_t  GetNbPMADA() const;
  virtual Short_t  GetNbPMADC() const;
  virtual Float_t  GetMTotADA() const;
  virtual Float_t  GetMTotADC() const; 

  virtual Float_t  GetMultiplicity(Int_t i) const;
  virtual Float_t  GetMultiplicityADA(Int_t i) const;
  virtual Float_t  GetMultiplicityADC(Int_t i) const;    
  virtual Float_t  GetAdc(Int_t i) const;
  virtual Float_t  GetAdcADA(Int_t i) const; 
  virtual Float_t  GetAdcADC(Int_t i) const;   
  virtual Float_t  GetTime(Int_t i) const;
  virtual Float_t  GetTimeADA(Int_t i) const;   
  virtual Float_t  GetTimeADC(Int_t i) const;    
  virtual Float_t  GetWidth(Int_t i) const;
  virtual Float_t  GetWidthADA(Int_t i) const;
  virtual Float_t  GetWidthADC(Int_t i) const;
  virtual Bool_t   BBTriggerADA(Int_t i) const;
  virtual Bool_t   BGTriggerADA(Int_t i) const;
  virtual Bool_t   BBTriggerADC(Int_t i) const;
  virtual Bool_t   BGTriggerADC(Int_t i) const;  
  virtual Bool_t   GetBBFlag(Int_t i) const;
  virtual Bool_t   GetBGFlag(Int_t i) const;

  virtual Float_t  GetADATime() const { return fADATime; }
  virtual Float_t  GetADCTime() const { return fADCTime; }
  virtual Float_t  GetADATimeError() const { return fADATimeError; }
  virtual Float_t  GetADCTimeError() const { return fADCTimeError; }

  virtual Decision GetADADecision() const { return fADADecision; }
  virtual Decision GetADCDecision() const { return fADCDecision; }

  virtual UShort_t GetTriggerChargeA() const { return fTriggerChargeA; }
  virtual UShort_t GetTriggerChargeC() const { return fTriggerChargeC; }
  virtual UShort_t GetTriggerBits() const { return fTriggerBits; }
  
  virtual Bool_t   GetPFBBFlag(Int_t channel, Int_t clock) const { return fIsBB[channel][clock]; } 
  virtual Bool_t   GetPFBGFlag(Int_t channel, Int_t clock) const { return fIsBG[channel][clock]; }
  
  virtual Float_t  GetAdcTail(Int_t i) const;
  virtual Float_t  GetAdcTailADA(Int_t i) const; 
  virtual Float_t  GetAdcTailADC(Int_t i) const; 
  
  virtual Float_t  GetAdcTrigger(Int_t i) const;
  virtual Float_t  GetAdcTriggerADA(Int_t i) const; 
  virtual Float_t  GetAdcTriggerADC(Int_t i) const;    
  
  AliESDAD &operator=(const AliESDAD& source);
    
protected:

  UInt_t  fBBtriggerADA;     // bit mask for Beam-Beam trigger in ADA
  UInt_t  fBGtriggerADA;     // bit mask for Beam-Gas trigger in ADA
  UInt_t  fBBtriggerADC;     // bit mask for Beam-Beam trigger in ADC
  UInt_t  fBGtriggerADC;     // bit mask for Beam-Gas trigger in ADC

  Float_t fMultiplicity[16]; //  multiplicity for each channel
  Float_t fAdc[16];          //  charge signal for each channel
  Float_t fTime[16];         //  time for each channel
  Float_t fWidth[16];        //  time width for each channel
  Bool_t  fBBFlag[16];       //  BB Flags from Online AD Electronics
  Bool_t  fBGFlag[16];       //  BG Flags from Online AD Electronics

  Float_t fADATime;          // Average time in ADA
  Float_t fADCTime;          // Average time in ADC
  Float_t fADATimeError;     // Error in the average time in ADA
  Float_t fADCTimeError;     // Error in the average time in ADC

  Decision fADADecision;     // ADA final decision based on average time of channels
  Decision fADCDecision;     // ADC final decision based on average time of channels

  UShort_t fTriggerChargeA;  // Sum of the trigger (clock=10) charge on A side
  UShort_t fTriggerChargeC;  // Sum of the trigger (clock=10) charge on C side
  UShort_t fTriggerBits;     // AD trigger bits as defined in the firmware
  
  Bool_t   fIsBB[16][21];  // BB flag for all channels and 21 clocks
  Bool_t   fIsBG[16][21];  // BG flag for all channels and 21 clocks
  
  Float_t fAdcTail[16];      //  tail of charge signal for each channel
  Float_t fAdcTrigger[16];   //  charge trigger signal for each channel

  ClassDef(AliESDAD,13)
};

#endif
