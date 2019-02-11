/**************************************************************************
 * Copyright(c) 1998-2007, ALICE Experiment at CERN, All rights reserved. *
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

//-------------------------------------------------------------------------
//     Container class for ESD AD data
//     Author: Michal Broz
//-------------------------------------------------------------------------

#include "AliESDAD.h"
#include "AliLog.h"

ClassImp(AliESDAD)

//__________________________________________________________________________
AliESDAD::AliESDAD()
  :AliVAD(),
   fBBtriggerADA(0),
   fBGtriggerADA(0),
   fBBtriggerADC(0),
   fBGtriggerADC(0),
   fADATime(-1024),
   fADCTime(-1024),
   fADATimeError(0),
   fADCTimeError(0),
   fADADecision(kADInvalid),
   fADCDecision(kADInvalid),
   fTriggerChargeA(0),
   fTriggerChargeC(0),
   fTriggerBits(0)
{   
   // Default constructor 
   for(Int_t j=0; j<16; j++){ 
      fMultiplicity[j] = 0.0;   
      fAdc[j]   = 0.0;   
      fTime[j]  = 0.0; 
      fWidth[j] = 0.0; 
      fBBFlag[j]= kFALSE;
      fBGFlag[j]= kFALSE;
      for(Int_t k = 0; k < 21; ++k) fIsBB[j][k] = fIsBG[j][k] = kFALSE; 
      fAdcTail[j]   = 0.0; 
      fAdcTrigger[j]   = 0.0; 
   }
}

//__________________________________________________________________________
AliESDAD::AliESDAD(const AliESDAD &o)
  :AliVAD(o),
   fBBtriggerADA(o.fBBtriggerADA),
   fBGtriggerADA(o.fBGtriggerADA),
   fBBtriggerADC(o.fBBtriggerADC),
   fBGtriggerADC(o.fBGtriggerADC),
   fADATime(o.fADATime),
   fADCTime(o.fADCTime),
   fADATimeError(o.fADATimeError),
   fADCTimeError(o.fADCTimeError),
   fADADecision(o.fADADecision),
   fADCDecision(o.fADCDecision),
   fTriggerChargeA(o.fTriggerChargeA),
   fTriggerChargeC(o.fTriggerChargeC),
   fTriggerBits(o.fTriggerBits)
{   
   // Default constructor 
   for(Int_t j=0; j<16; j++) {
       fMultiplicity[j] = o.fMultiplicity[j];
       fAdc[j]    = o.fAdc[j];
       fTime[j]   = o.fTime[j];
       fWidth[j]  = o.fWidth[j];
       fBBFlag[j] = o.fBBFlag[j];
       fBGFlag[j] = o.fBGFlag[j];
       for(Int_t k = 0; k < 21; ++k) {
	 fIsBB[j][k] = o.fIsBB[j][k];
	 fIsBG[j][k] = o.fIsBG[j][k];
       }
       fAdcTail[j]    = o.fAdcTail[j];
       fAdcTrigger[j]   = o.fAdcTrigger[j];
   }
}

//__________________________________________________________________________
AliESDAD::AliESDAD(UInt_t BBtriggerADA, UInt_t BGtriggerADA,
	      UInt_t BBtriggerADC, UInt_t BGtriggerADC,
	      Float_t *Multiplicity, Float_t *Adc, 
	      Float_t *Time, Float_t *Width, Bool_t *BBFlag, Bool_t *BGFlag)
  :AliVAD(),
   fBBtriggerADA(BBtriggerADA),
   fBGtriggerADA(BGtriggerADA),
   fBBtriggerADC(BBtriggerADC),
   fBGtriggerADC(BGtriggerADC),
   fADATime(-1024),
   fADCTime(-1024),
   fADATimeError(0),
   fADCTimeError(0),
   fADADecision(kADInvalid),
   fADCDecision(kADInvalid),
   fTriggerChargeA(0),
   fTriggerChargeC(0),
   fTriggerBits(0)
{
   // Constructor
   for(Int_t j=0; j<16; j++) {
       fMultiplicity[j] = Multiplicity[j];
       fAdc[j]    = Adc[j];
       fTime[j]   = Time[j];
       fWidth[j]  = Width[j];
       fBBFlag[j] = BBFlag[j];
       fBGFlag[j] = BGFlag[j];
       for(Int_t k = 0; k < 21; ++k) fIsBB[j][k] = fIsBG[j][k] = kFALSE;
       fAdcTail[j]    = 0.0;
       fAdcTrigger[j]   = 0.0; 
   }
}

//__________________________________________________________________________
AliESDAD& AliESDAD::operator=(const AliESDAD& o)
{

  if(this==&o) return *this;
  AliVAD::operator=(o);
  // Assignment operator
  fBBtriggerADA=o.fBBtriggerADA;
  fBGtriggerADA=o.fBGtriggerADA;
  fBBtriggerADC=o.fBBtriggerADC;
  fBGtriggerADC=o.fBGtriggerADC;

  fADATime = o.fADATime;
  fADCTime = o.fADCTime;
  fADATimeError = o.fADATimeError;
  fADCTimeError = o.fADCTimeError;
  fADADecision = o.fADADecision;
  fADCDecision = o.fADCDecision;
  fTriggerChargeA = o.fTriggerChargeA;
  fTriggerChargeC = o.fTriggerChargeC;
  fTriggerBits = o.fTriggerBits;

   for(Int_t j=0; j<16; j++) {
       fMultiplicity[j] = o.fMultiplicity[j];
       fAdc[j]    = o.fAdc[j];
       fTime[j]   = o.fTime[j];
       fWidth[j]  = o.fWidth[j];
       fBBFlag[j] = o.fBBFlag[j];
       fBGFlag[j] = o.fBGFlag[j];
       for(Int_t k = 0; k < 21; ++k) {
	 fIsBB[j][k] = o.fIsBB[j][k];
	 fIsBG[j][k] = o.fIsBG[j][k];
       }
       fAdcTail[j]    = o.fAdcTail[j];
       fAdcTrigger[j]   = o.fAdcTrigger[j];
   }
  return *this;
}

//______________________________________________________________________________
void AliESDAD::Copy(TObject &obj) const {
  
  // this overwrites the virtual TOBject::Copy()
  // to allow run time copying without casting
  // in AliESDEvent

  if(this==&obj)return;
  AliESDAD *robj = dynamic_cast<AliESDAD*>(&obj);
  if(!robj)return; // not an AliESDAD
  *robj = *this;

}

//__________________________________________________________________________
Short_t AliESDAD::GetNbPMADA() const
{
  // Returns the number of
  // fired PM in ADA
  Short_t n=0;
  for(Int_t i=8;i<16;i++) 
    if (fMultiplicity[i]>0) n++;
  return n;
}

//__________________________________________________________________________
Short_t AliESDAD::GetNbPMADC() const
{
  // Returns the number of
  // fired PM in ADC
  Short_t n=0;
  for(Int_t i=0;i<8;i++) 
    if (fMultiplicity[i]>0) n++;
  return n;
}

//__________________________________________________________________________
Float_t AliESDAD::GetMTotADA() const
{
  // returns total multiplicity
  // in ADA
  Float_t mul=0.0;
  for(Int_t i=8;i<16;i++) 
    mul+=  fMultiplicity[i];
  return mul;
}

//__________________________________________________________________________
Float_t AliESDAD::GetMTotADC() const
{
  // returns total multiplicity
  // in ADC
  Float_t mul=0.0;
  for(Int_t i=0;i<8;i++) 
    mul+=  fMultiplicity[i];
  return mul;
}


//__________________________________________________________________________
Float_t AliESDAD::GetMultiplicity(Int_t i) const

{
  // returns multiplicity in a
  // given cell of AD
  if (OutOfRange(i, "AliESDAD::GetMultiplicity:",16)) return -1;
  return fMultiplicity[i];
}

//__________________________________________________________________________
Float_t AliESDAD::GetMultiplicityADA(Int_t i) const

{
  // returns multiplicity in a
  // given cell of ADA
  if (OutOfRange(i, "AliESDAD::GetMultiplicityADA:",8)) return -1;
  return fMultiplicity[8+i];
}

//__________________________________________________________________________
Float_t AliESDAD::GetMultiplicityADC(Int_t i) const

{
  // returns multiplicity in a
  // given cell of ADC
  if (OutOfRange(i, "AliESDAD::GetMultiplicityADC:",8)) return -1;
  return fMultiplicity[i];
}

//__________________________________________________________________________
Float_t AliESDAD::GetAdc(Int_t i) const

{
  // returns ADC charge in a
  // given cell of AD
  if (OutOfRange(i, "AliESDAD::GetAdc:",16)) return -1;
  return fAdc[i];
}

//__________________________________________________________________________
Float_t AliESDAD::GetAdcADA(Int_t i) const

{
  // returns ADC charge in a
  // given cell of ADA
  if (OutOfRange(i, "AliESDAD::GetAdcADA:",8)) return -1;
  return fAdc[8+i];
}

//__________________________________________________________________________
Float_t AliESDAD::GetAdcADC(Int_t i) const

{
  // returns ADC charge in a
  // given cell of ADC
  if (OutOfRange(i, "AliESDAD::GetAdcADC:",8)) return -1;
  return fAdc[i];
}

//__________________________________________________________________________
Float_t AliESDAD::GetAdcTail(Int_t i) const

{
  // returns ADC charge in a
  // given cell of AD
  if (OutOfRange(i, "AliESDAD::GetAdcTail:",16)) return -1;
  return fAdcTail[i];
}

//__________________________________________________________________________
Float_t AliESDAD::GetAdcTailADA(Int_t i) const

{
  // returns ADC charge in a
  // given cell of ADA
  if (OutOfRange(i, "AliESDAD::GetAdcTailADA:",8)) return -1;
  return fAdcTail[8+i];
}

//__________________________________________________________________________
Float_t AliESDAD::GetAdcTailADC(Int_t i) const

{
  // returns ADC charge in a
  // given cell of ADC
  if (OutOfRange(i, "AliESDAD::GetAdcTailADC:",8)) return -1;
  return fAdcTail[i];
}

//__________________________________________________________________________
Float_t AliESDAD::GetAdcTrigger(Int_t i) const

{
  // returns ADC charge in a
  // given cell of AD
  if (OutOfRange(i, "AliESDAD::GetAdcTrigger:",16)) return -1;
  return fAdcTrigger[i];
}

//__________________________________________________________________________
Float_t AliESDAD::GetAdcTriggerADA(Int_t i) const

{
  // returns ADC charge in a
  // given cell of ADA
  if (OutOfRange(i, "AliESDAD::GetAdcTriggerADA:",8)) return -1;
  return fAdcTrigger[8+i];
}

//__________________________________________________________________________
Float_t AliESDAD::GetAdcTriggerADC(Int_t i) const

{
  // returns ADC charge in a
  // given cell of ADC
  if (OutOfRange(i, "AliESDAD::GetAdcTriggerADC:",8)) return -1;
  return fAdcTrigger[i];
}



//__________________________________________________________________________
Float_t AliESDAD::GetTime(Int_t i) const

{
  // returns leading time measured by TDC
  // in a given cell of AD
  if (OutOfRange(i, "AliESDAD::GetTime:",16)) return -1;
  return fTime[i];
}

//__________________________________________________________________________
Float_t AliESDAD::GetTimeADA(Int_t i) const

{
  // returns leading time measured by TDC
  // in a given cell of ADA
  if (OutOfRange(i, "AliESDAD::GetTimeADA:",8)) return -1;
  return fTime[8+i];
}

//__________________________________________________________________________
Float_t AliESDAD::GetTimeADC(Int_t i) const

{
  // returns leading time measured by TDC
  // in a given cell of ADC
  if (OutOfRange(i, "AliESDAD::GetTimeADC:",8)) return -1;
  return fTime[i];
}

//__________________________________________________________________________
Float_t AliESDAD::GetWidth(Int_t i) const

{
  // returns time signal width
  // in a given cell of AD
  if (OutOfRange(i, "AliESDAD::GetWidth:",16)) return -1;
  return fWidth[i];
}

//__________________________________________________________________________
Float_t AliESDAD::GetWidthADA(Int_t i) const

{
  // returns time signal width
  // in a given cell of ADA
  if (OutOfRange(i, "AliESDAD::GetWidthADA:",8)) return -1;
  return fWidth[8+i];
}

//__________________________________________________________________________
Float_t AliESDAD::GetWidthADC(Int_t i) const

{
  // returns time signal width
  // in a given cell of ADC
  if (OutOfRange(i, "AliESDAD::GetWidthADC:",8)) return -1;
  return fWidth[i];
}

//__________________________________________________________________________
Bool_t AliESDAD::BBTriggerADA(Int_t i) const
{
  // returns offline beam-beam flags in ADA
  // one bit per cell
  if (OutOfRange(i, "AliESDAD:::BBTriggerADA",8)) return kFALSE;
  UInt_t test = 1;
  return ( fBBtriggerADA & (test << i) ? kTRUE : kFALSE );
}

//__________________________________________________________________________
Bool_t AliESDAD::BGTriggerADA(Int_t i) const
{
  // returns offline beam-gas flags in ADA
  // one bit per cell
  if (OutOfRange(i, "AliESDAD:::BGTriggerADA",8)) return kFALSE;
  UInt_t test = 1;
  return ( fBGtriggerADA & (test << i) ? kTRUE : kFALSE );
}

//__________________________________________________________________________
Bool_t AliESDAD::BBTriggerADC(Int_t i) const
{
  // returns offline beam-beam flags in ADC
  // one bit per cell
  if (OutOfRange(i, "AliESDAD:::BBTriggerADC",8)) return kFALSE;
  UInt_t test = 1;
  return ( fBBtriggerADC & (test << i) ? kTRUE : kFALSE );
}

//__________________________________________________________________________
Bool_t AliESDAD::BGTriggerADC(Int_t i) const
{
  // returns offline beam-gasflags in ADC
  // one bit per cell
  if (OutOfRange(i, "AliESDAD:::BGTriggerADC",8)) return kFALSE;
  UInt_t test = 1;
  return ( fBGtriggerADC & (test << i) ? kTRUE : kFALSE );
}

//__________________________________________________________________________
Bool_t AliESDAD::GetBBFlag(Int_t i) const

{
  // returns online beam-beam flag in AD
  // one boolean per cell
  if (OutOfRange(i, "AliESDAD::GetBBFlag:",16)) return kFALSE;
  return fBBFlag[i];
}

//__________________________________________________________________________
Bool_t AliESDAD::GetBGFlag(Int_t i) const

{
  // returns online beam-gas flag in AD
  // one boolean per cell
  if (OutOfRange(i, "AliESDAD::GetBGFlag:",16)) return kFALSE;
  return fBGFlag[i];
}
