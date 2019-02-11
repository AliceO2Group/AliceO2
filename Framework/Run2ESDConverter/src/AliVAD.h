#ifndef ALIVAD_H
#define ALIVAD_H

//-------------------------------------------------------------------------
//     Base class for ESD and AOD AD data
//     Author: Michal Broz
//     Michal.Broz@cern.ch 
//-------------------------------------------------------------------------

#include "TObject.h"
#include "TMath.h"

class AliVAD : public TObject 
{
public:
  AliVAD() { }
  AliVAD(const AliVAD& source);
  AliVAD &operator=(const AliVAD& source);

  virtual ~AliVAD() { }

  enum {
    kCorrectedLeadingTime = BIT(14),
    kTriggerBitsFilled = BIT(15),
    kDecisionFilled = BIT(16),
    kOnlineBitsFilled = BIT(17),
    kCorrectedForSaturation = BIT(18),
    kRobustMeanTime = BIT(19),
    kTriggerChargeBitsFilled = BIT(20),
    kPastFutureFlagsFilled = BIT(21)
  };
  enum Decision { kADInvalid = -1, kADEmpty = 0, kADBB, kADBG, kADFake };
  enum TriggerBits {
    kBBAandBBC = 0,
    kBBAorBBC = 1,
    kBGAandBBC = 2,
    kBGA = 3,
    kBGCandBBA = 4,
    kBGC = 5,
    kCTA1andCTC1 = 6,
    kCTA1orCTC1 = 7,
    kCTA2andCTC2 = 8,
    kCTA2orCTC2 = 9,
    kMTAandMTC = 10,
    kMTAorMTC = 11,
    kBBA = 12,
    kBBC = 13,
    kBGAorBGC = 14,
    kBGAandBBCorBGCandBBA = 15
  };

  virtual Short_t  GetNbPMADA() const = 0;
  virtual Short_t  GetNbPMADC() const = 0;
  virtual Float_t  GetMTotADA() const = 0;
  virtual Float_t  GetMTotADC() const = 0; 

  virtual Float_t  GetMultiplicity(Int_t i) const = 0;
  virtual Float_t  GetMultiplicityADA(Int_t i) const = 0;
  virtual Float_t  GetMultiplicityADC(Int_t i) const = 0;    

  virtual Bool_t   BBTriggerADA(Int_t i) const = 0;
  virtual Bool_t   BGTriggerADA(Int_t i) const = 0;
  virtual Bool_t   BBTriggerADC(Int_t i) const = 0;
  virtual Bool_t   BGTriggerADC(Int_t i) const = 0;  
  virtual Bool_t   GetBBFlag(Int_t i) const = 0;
  virtual Bool_t   GetBGFlag(Int_t i) const = 0;

  virtual Float_t  GetADATime() const = 0;
  virtual Float_t  GetADCTime() const = 0;

  virtual Decision GetADADecision() const = 0;
  virtual Decision GetADCDecision() const = 0;

  virtual UShort_t GetTriggerChargeA() const = 0;
  virtual UShort_t GetTriggerChargeC() const = 0;
  virtual UShort_t GetTriggerBits() const = 0;
  
  virtual Bool_t   GetPFBBFlag(Int_t channel, Int_t clock) const  = 0;
  virtual Bool_t   GetPFBGFlag(Int_t channel, Int_t clock) const  = 0;

protected:  

  Bool_t OutOfRange(Int_t i, const char *s, Int_t upper) const;
    
  ClassDef(AliVAD,3)
};

#endif
