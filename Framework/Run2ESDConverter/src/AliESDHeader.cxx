/**************************************************************************
 * Copyright(c) 1998-1999, ALICE Experiment at CERN, All rights reserved. *
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
//                      Implementation of   Class AliESDHeader
//   Header data
//   for the ESD   
//   Origin: Christian Klein-Boesing, CERN, Christian.Klein-Boesing@cern.ch 
//-------------------------------------------------------------------------

#include "AliESDHeader.h"
#include "AliTriggerScalersESD.h"
#include "AliTriggerScalersRecordESD.h"
#include "AliTriggerIR.h"
#include "AliTriggerConfiguration.h"
#include "AliLog.h" 

ClassImp(AliESDHeader)

//______________________________________________________________________________
AliESDHeader::AliESDHeader() :
  AliVHeader(),
  fTriggerMask(0),
  fTriggerMaskNext50(0),
  fOrbitNumber(0),
  fTimeStamp(0),
  fEventType(0),
  fEventSpecie(0),
  fPeriodNumber(0),
  fEventNumberInFile(0),
  fBunchCrossNumber(0),
  fTriggerCluster(0),
  fL0TriggerInputs(0),
  fL1TriggerInputs(0),
  fL2TriggerInputs(0),
  fTriggerScalers(),
  fTriggerScalersDeltaEvent(),
  fTriggerScalersDeltaRun(),
  fTriggerInputsNames(kNTriggerInputs),
  fCTPConfig(NULL),
  fIRBufferArray(),
  fIRInt2InteractionsMap(0),
  fIRInt1InteractionsMap(0)
{
  // default constructor

  SetName("AliESDHeader");
  for(Int_t i = 0; i<kNMaxIR ; i++) fIRArray[i] = 0;
  fTriggerInputsNames.SetOwner(kTRUE);
  for (Int_t itype=0; itype<3; itype++) fTPCNoiseFilterCounter[itype]=0;
  fIRBufferArray.SetOwner(kTRUE);
}

AliESDHeader::~AliESDHeader() 
{
  // destructor
  for(Int_t i=0;i<kNMaxIR;i++)if(fIRArray[i])delete fIRArray[i];
  delete fCTPConfig;
  //  fIRBufferArray.Delete();
}


AliESDHeader::AliESDHeader(const AliESDHeader &header) :
  AliVHeader(header),
  fTriggerMask(header.fTriggerMask),
  fTriggerMaskNext50(header.fTriggerMaskNext50),
  fOrbitNumber(header.fOrbitNumber),
  fTimeStamp(header.fTimeStamp),
  fEventType(header.fEventType),
  fEventSpecie(header.fEventSpecie),
  fPeriodNumber(header.fPeriodNumber),
  fEventNumberInFile(header.fEventNumberInFile),
  fBunchCrossNumber(header.fBunchCrossNumber),
  fTriggerCluster(header.fTriggerCluster),
  fL0TriggerInputs(header.fL0TriggerInputs),
  fL1TriggerInputs(header.fL1TriggerInputs),
  fL2TriggerInputs(header.fL2TriggerInputs),
  fTriggerScalers(header.fTriggerScalers),
  fTriggerScalersDeltaEvent(header.fTriggerScalersDeltaEvent),
  fTriggerScalersDeltaRun(header.fTriggerScalersDeltaRun),
  fTriggerInputsNames(TObjArray(kNTriggerInputs)),
  fCTPConfig(header.fCTPConfig),
  fIRBufferArray(),
  fIRInt2InteractionsMap(header.fIRInt2InteractionsMap),
  fIRInt1InteractionsMap(header.fIRInt1InteractionsMap)
{
  // copy constructor
  for(Int_t i = 0; i<kNMaxIR ; i++) {
    if(header.fIRArray[i])fIRArray[i] = new AliTriggerIR(*header.fIRArray[i]);
    else fIRArray[i]=0;
  }
  for(Int_t i = 0; i < kNTriggerInputs; i++) {
    TNamed *str = (TNamed *)((header.fTriggerInputsNames).At(i));
    if (str) fTriggerInputsNames.AddAt(new TNamed(*str),i);
  }

  for(Int_t i = 0; i < (header.fIRBufferArray).GetEntries(); ++i) {
    AliTriggerIR *ir = (AliTriggerIR*)((header.fIRBufferArray).At(i));
    if (ir) fIRBufferArray.Add(new AliTriggerIR(*ir));
  }
  for (Int_t itype=0; itype<3; itype++) fTPCNoiseFilterCounter[itype]=header.fTPCNoiseFilterCounter[itype];
  fTriggerInputsNames.SetOwner(kTRUE);
  fIRBufferArray.SetOwner(kTRUE);
}

AliESDHeader& AliESDHeader::operator=(const AliESDHeader &header)
{ 
  // assigment operator
  if(this!=&header) {
    AliVHeader::operator=(header);
    fTriggerMask = header.fTriggerMask;
    fTriggerMaskNext50 = header.fTriggerMaskNext50;
    fOrbitNumber = header.fOrbitNumber;
    fTimeStamp = header.fTimeStamp;
    fEventType = header.fEventType;
    fEventSpecie = header.fEventSpecie;
    fPeriodNumber = header.fPeriodNumber;
    fEventNumberInFile = header.fEventNumberInFile;
    fBunchCrossNumber = header.fBunchCrossNumber;
    fTriggerCluster = header.fTriggerCluster;
    fL0TriggerInputs = header.fL0TriggerInputs;
    fL1TriggerInputs = header.fL1TriggerInputs;
    fL2TriggerInputs = header.fL2TriggerInputs;
    fTriggerScalers = header.fTriggerScalers;
    fTriggerScalersDeltaEvent = header.fTriggerScalersDeltaEvent;
    fTriggerScalersDeltaRun = header.fTriggerScalersDeltaRun;
    fIRInt2InteractionsMap = header.fIRInt2InteractionsMap;
    fIRInt1InteractionsMap = header.fIRInt1InteractionsMap;

    delete fCTPConfig;
    fCTPConfig = header.fCTPConfig;

    fTriggerInputsNames.Clear();
    for(Int_t i = 0; i < kNTriggerInputs; i++) {
      TNamed *str = (TNamed *)((header.fTriggerInputsNames).At(i));
      if (str) fTriggerInputsNames.AddAt(new TNamed(*str),i);
    }
    for(Int_t i = 0; i<kNMaxIR ; i++) {
      delete fIRArray[i];
       if(header.fIRArray[i])fIRArray[i] = new AliTriggerIR(*header.fIRArray[i]);
       else fIRArray[i]=0;
    }

    fIRBufferArray.Delete();
    for(Int_t i = 0; i < (header.fIRBufferArray).GetEntries(); ++i) {
      AliTriggerIR *ir = (AliTriggerIR*)((header.fIRBufferArray).At(i));
      if (ir) fIRBufferArray.Add(new AliTriggerIR(*ir));
    }
    for (Int_t itype=0; itype<3; itype++) fTPCNoiseFilterCounter[itype]=header.fTPCNoiseFilterCounter[itype];
  }
  return *this;
}

void AliESDHeader::Copy(TObject &obj) const 
{  
  // this overwrites the virtual TOBject::Copy()
  // to allow run time copying without casting
  // in AliESDEvent

  if(this==&obj)return;
  AliESDHeader *robj = dynamic_cast<AliESDHeader*>(&obj);
  if(!robj)return; // not an AliESDHeader
  *robj = *this;

}
//______________________________________________________________________________
void AliESDHeader::Reset()
{
  // reset all data members
  fTriggerMask       = 0;
  fTriggerMaskNext50 = 0;
  fOrbitNumber       = 0;
  fTimeStamp         = 0;
  fEventType         = 0;
  fEventSpecie       = 0;
  fPeriodNumber      = 0;
  fEventNumberInFile = 0;
  fBunchCrossNumber  = 0;
  fTriggerCluster    = 0;
  fL0TriggerInputs   = 0;
  fL1TriggerInputs   = 0;
  fL2TriggerInputs   = 0;
  fTriggerScalers.Reset();
  fTriggerScalersDeltaEvent.Reset();
  fTriggerScalersDeltaRun.Reset();
  fTriggerInputsNames.Clear();

  fIRInt2InteractionsMap.ResetAllBits();
  fIRInt1InteractionsMap.ResetAllBits();

  delete fCTPConfig;
  fCTPConfig = 0;
  for(Int_t i=0;i<kNMaxIR;i++)if(fIRArray[i]){
   delete fIRArray[i];
   fIRArray[i]=0;
  }
  for (Int_t itype=0; itype<3; itype++) fTPCNoiseFilterCounter[itype]=0;
  fIRBufferArray.Clear();
}
//______________________________________________________________________________
Bool_t AliESDHeader::AddTriggerIR(const AliTriggerIR* ir)
{
  // Add an IR object into the array
  // of IRs in the ESD header

 fIRBufferArray.Add(new AliTriggerIR(*ir));

 return kTRUE;
}
//______________________________________________________________________________
void AliESDHeader::Print(const Option_t *) const
{
  // Print some data members
  printf("Event # %d in file Bunch crossing # %d Orbit # %d Trigger %lld %lld\n",
	 GetEventNumberInFile(),
	 GetBunchCrossNumber(),
	 GetOrbitNumber(),
	 GetTriggerMask(),
	 GetTriggerMaskNext50());
         printf("List of the active trigger inputs: ");
  	 for(Int_t i = 0; i < kNTriggerInputs; i++) {
    	   TNamed *str = (TNamed *)((fTriggerInputsNames).At(i));
    	   if (str) printf("%i %s ",i,str->GetName());
         }
         printf("\n");
}

//______________________________________________________________________________
void AliESDHeader::SetActiveTriggerInputs(const char*name, Int_t index)
{
  // Fill the active trigger inputs names
  // into the corresponding fTriggerInputsNames (TObjArray of TNamed)
  if (index >= kNTriggerInputs || index < 0) {
    AliError(Form("Index (%d) is outside the allowed range (0,59)!",index));
    return;
  }

  fTriggerInputsNames.AddAt(new TNamed(name,NULL),index);
}
//______________________________________________________________________________
const char* AliESDHeader::GetTriggerInputName(Int_t index, Int_t trglevel) const
{
  // Get the trigger input name
  // at the specified position in the trigger mask and trigger level (0,1,2)
  TNamed *trginput = 0;
  if (trglevel == 0) trginput = (TNamed *)fTriggerInputsNames.At(index);
  if (trglevel == 1) trginput = (TNamed *)fTriggerInputsNames.At(index+24);  
  if (trglevel == 2) trginput = (TNamed *)fTriggerInputsNames.At(index+48); 
  if (trginput) return trginput->GetName();
  else return "";
}
//______________________________________________________________________________
TString AliESDHeader::GetActiveTriggerInputs() const
{
  // Returns the list with the names of the active trigger inputs
  TString trginputs;
  for(Int_t i = 0; i < kNTriggerInputs; i++) {
    TNamed *str = (TNamed *)((fTriggerInputsNames).At(i));
    if (str) {
      trginputs += " ";
      trginputs += str->GetName();
      trginputs += " ";
    }
  }

  return trginputs;
}
//______________________________________________________________________________
TString AliESDHeader::GetFiredTriggerInputs() const
{
  // Returns the list with the names of the fired trigger inputs
  TString trginputs;
  for(Int_t i = 0; i < kNTriggerInputs; i++) {
      TNamed *str = (TNamed *)((fTriggerInputsNames.At(i)));
      if (i < 24 && (fL0TriggerInputs & (1ul << i))) {
        if (str) {
	  trginputs += " ";
	  trginputs += str->GetName();
          trginputs += " ";
        }
      }
      if (i >= 24 && i < 48 && (fL1TriggerInputs & (1ul << (i-24)))) {
        if (str) {
	  trginputs += " ";
	  trginputs += str->GetName();
          trginputs += " ";
        }
      }
      if (i >= 48 && (fL2TriggerInputs & (1u << (i-48)))) {
        if (str) {
	  trginputs += " ";
	  trginputs += str->GetName();
          trginputs += " ";
        }
      }

  }
  return trginputs;
}
//______________________________________________________________________________
Bool_t AliESDHeader::IsTriggerInputFired(const char *name) const
{
  // Checks if the trigger input is fired 
 
  TNamed *trginput = (TNamed *)fTriggerInputsNames.FindObject(name);
  if (trginput == 0) return kFALSE;

  Int_t inputIndex = fTriggerInputsNames.IndexOf(trginput);
  if (inputIndex < 0) return kFALSE;
 
  if(inputIndex < 24){
    if (fL0TriggerInputs & (1lu << inputIndex)) return kTRUE;
  } else if(inputIndex < 48){
    if (fL1TriggerInputs & (1lu << (inputIndex-24))) return kTRUE;
  } else if(inputIndex < 60){
    if (fL2TriggerInputs & (1u << (inputIndex-48))) return kTRUE;
  }
  else {
    AliError(Form("Index (%d) is outside the allowed range (0,59)!",inputIndex));
    return kFALSE;
  }
  return kFALSE;
}
//________________________________________________________________________________
Int_t  AliESDHeader::GetTriggerIREntries(Int_t int1, Int_t int2, Float_t deltaTime) const
{
  // returns number of IR-s within time window deltaTime
  // all possible combinations of int1 and int2 int1 - zdc bit, int2 v0 bit
  //
  const AliTriggerIR *IR;
  // triggered event 
  Int_t nIR = GetTriggerIREntries();
  UInt_t orbit1 = GetOrbitNumber();
  const Double_t ot=0.0889218; //orbit time msec
  Float_t timediff; // time difference between orbits (msec)
  //
  Int_t nofIR;
  nofIR=0;
  // loop over IR-s
    for(Int_t i=0;i<nIR;i++){//1
      IR=GetTriggerIR(i);
      //
      UInt_t orbit2 = IR->GetOrbit();
      timediff = (orbit2<=orbit1) ? (Float_t)((orbit1-orbit2))*ot : 
	(Float_t)((16777215-orbit1+orbit2))*ot;
      if (timediff>deltaTime) continue; //timediff outside time window
      if((int1&int2) == -1){ //ignore both bits, just count IR-s within time window
	nofIR++;
        continue;
      }
      // now check if int1, int2 bits are set
      UInt_t nw = IR->GetNWord();
      Bool_t *bint1 = IR->GetInt1s();
      Bool_t *bint2 = IR->GetInt2s();
      //
      Int_t flag1,flag2;
      flag1=0;
      flag2=0;
      for(UInt_t j=0;j<nw;j++){//2
	if(bint1[j]) flag1=1; // at least one int1 set
	if(bint2[j]) flag2=1; //  at least one int2 set
        //printf("IR %d, bint1 %d, bint2 %d\n",i,bint1[j],bint2[j]);
      }//2
      // checking combinations
      //
      
      if((flag1*int1*flag2*int2)==1){// int1=1 & int2=1	 
          nofIR++;
          continue;       
      }
      if(int1 == -1){// ignore int1
        if(flag2&int2){// int2=1
          nofIR++;
          continue;
	}
        else if (!flag2&!int2){ //int2=0 
          nofIR++;
          continue;          
	}
      }
      
      if(int2 ==-1){//ignore int2
        if(flag1&int1){//int1=1
          nofIR++;
          continue;  
	}
        else if(!flag1&!int1){ //int1=0
          nofIR++;
          continue;  
	}
      }
      
      if((flag1*int1)&!flag2&!int2){// int1=1, int2=0
          nofIR++;
          continue;  
      }
      
      if((int2*flag2)&!int1&!flag1){// int1=0, int2=1
          nofIR++;
          continue;  
      } 
         
      

    }//1
  
    return nofIR;
}
//__________________________________________________________________________
TObjArray AliESDHeader::GetIRArray(Int_t int1, Int_t int2, Float_t deltaTime) const
{
  //
  // returns an array of IR-s within time window deltaTime
  // all possible combinations of int1 and int2 int1 - zdc bit, int2 v0 bit
  //
  const AliTriggerIR *IR;
  TObjArray arr;
  // triggered event 
  Int_t nIR = GetTriggerIREntries();
  UInt_t orbit1 = GetOrbitNumber();
  const Double_t ot=0.0889218; //orbit time msec
  Float_t timediff; // time difference between orbits (msec)
  //
  // loop over IR-s
    for(Int_t i=0;i<nIR;i++){//1
      IR=GetTriggerIR(i);
      //
      UInt_t orbit2 = IR->GetOrbit();
      timediff = (orbit2<=orbit1) ? (Float_t)((orbit1-orbit2))*ot : 
	(Float_t)((16777215-orbit1+orbit2))*ot;
      if (timediff>deltaTime) continue; //timediff outside time window
      if((int1&int2) == -1){ //ignore both bits, just count IR-s within time window
	arr.Add((AliTriggerIR*)IR); //add this IR
        continue;
      }
      // now check if int1, int2 bits are set
      UInt_t nw = IR->GetNWord();
      Bool_t *bint1 = IR->GetInt1s();
      Bool_t *bint2 = IR->GetInt2s();
      //
      Int_t flag1,flag2;
      flag1=0;
      flag2=0;
      for(UInt_t j=0;j<nw;j++){//2
	if(bint1[j]) flag1=1; // at least one int1 set
	if(bint2[j]) flag2=1; //  at least one int2 set
      }//2
      // checking combinations
      //
      if((flag1*int1*flag2*int2)==1){// int1=1 & int2=1
	  arr.Add((AliTriggerIR*)IR); //add this IR
          continue;       
      }
      if(int1 == -1){// ignore int1
        if(flag2&int2){// int2=1
 	  arr.Add((AliTriggerIR*)IR); //add this IR
          continue;
	}
        else if (!flag2&!int2){ //int2=0 
          arr.Add((AliTriggerIR*)IR); //add this IR
          continue;          
	}
      }
      if(int2 ==-1){//ignore int2
        if(flag1&int1){//int1=1
	  arr.Add((AliTriggerIR*)IR); //add this IR
          continue;  
	}
        else if(!flag1&!int1){ //int1=0
	  arr.Add((AliTriggerIR*)IR); //add this IR
          continue;  
	}
      }
      if ((flag1*int1)&!flag2&!int2){// int1=1, int2=0
	  arr.Add((AliTriggerIR*)IR); //add this IR
          continue;  
      }
      if ((int2*flag2)&!int1&!flag1){// int1=0, int2=1
	  arr.Add((AliTriggerIR*)IR); //add this IR
          continue;  
      }      

    }//1
  
  return arr;
}

//__________________________________________________________________________
void AliESDHeader::SetIRInteractionMap() const
{
  //
  // Function to compute the map of interations 
  // within 0TVX (int2) or V0A&V0C (int1) and the Event Id 
  // Note, the zero value is excluded
  //
  const AliTriggerIR *ir[5] = {GetTriggerIR(0),GetTriggerIR(1),GetTriggerIR(2),GetTriggerIR(3),GetTriggerIR(4)};

  Long64_t orb = (Long64_t)GetOrbitNumber();
  Long64_t bc = (Long64_t)GetBunchCrossNumber();
  
  Long64_t evId = orb*3564 + bc;

  for(Int_t i = 0; i < 5; ++i) {
    if (ir[i] == NULL || ir[i]->GetNWord() == 0) continue;
    Long64_t irOrb = (Long64_t)ir[i]->GetOrbit();
    Bool_t* int2 = ir[i]->GetInt2s();
    Bool_t* int1 = ir[i]->GetInt1s();
    UShort_t* bcs = ir[i]->GetBCs();
    for(UInt_t nW = 0; nW < ir[i]->GetNWord(); ++nW) {
      Long64_t intId = irOrb*3564 + (Long64_t)bcs[nW];
      if (int2[nW] == kTRUE) {
	  Int_t item = (intId-evId);
	  Int_t bin = FindIRIntInteractionsBXMap(item);
	  if(bin>=0) {
	    fIRInt2InteractionsMap.SetBitNumber(bin,kTRUE);
	  }
      }
      if (int1[nW] == kTRUE) {
	  Int_t item = (intId-evId);
	  Int_t bin = FindIRIntInteractionsBXMap(item);
	  if(bin>=0) {
	    fIRInt1InteractionsMap.SetBitNumber(bin,kTRUE);
	  }
      }
    }
  }

  fIRInt2InteractionsMap.Compact();
  fIRInt1InteractionsMap.Compact();
}

//__________________________________________________________________________
Int_t AliESDHeader::FindIRIntInteractionsBXMap(Int_t difference) const
{
  //
  // The mapping is of 181 bits, from -90 to +90
  //
  Int_t bin=-1;

  if(difference<-90 || difference>90) return bin;
  else { bin = 90 + difference; }
  
  return bin;
}

//__________________________________________________________________________
Int_t AliESDHeader::GetIRInt2ClosestInteractionMap() const
{
  //
  // Calculation of the closest interaction
  //
  SetIRInteractionMap();

  Int_t firstNegative=100;
  for(Int_t item=-1; item>=-90; item--) {
    Int_t bin = FindIRIntInteractionsBXMap(item);
    Bool_t isFired = fIRInt2InteractionsMap.TestBitNumber(bin);
    if(isFired) {
      firstNegative = item;
      break;
    }
  }
  Int_t firstPositive=100;
  for(Int_t item=1; item<=90; item++) {
    Int_t bin = FindIRIntInteractionsBXMap(item);
    Bool_t isFired = fIRInt2InteractionsMap.TestBitNumber(bin);
    if(isFired) {
      firstPositive = item;
      break;
    }
  }

  Int_t closest = firstPositive < TMath::Abs(firstNegative) ? firstPositive : TMath::Abs(firstNegative);
  if(firstPositive==100 && firstNegative==100) closest=0;
  return closest;
}

//__________________________________________________________________________
Int_t AliESDHeader::GetIRInt1ClosestInteractionMap(Int_t gap) const
{
  //
  // Calculation of the closest interaction
  // In case of VZERO (Int1) one has to introduce a gap
  // in order to avoid false positivies from after-pulses

  SetIRInteractionMap();

  Int_t firstNegative=100;
  for(Int_t item=-1; item>=-90; item--) {
    Int_t bin = FindIRIntInteractionsBXMap(item);
    Bool_t isFired = fIRInt1InteractionsMap.TestBitNumber(bin);
    if(isFired) {
      firstNegative = item;
      break;
    }
  }
  Int_t firstPositive=100;
  for(Int_t item=1+gap; item<=90; item++) {
    Int_t bin = FindIRIntInteractionsBXMap(item);
    Bool_t isFired = fIRInt1InteractionsMap.TestBitNumber(bin);
    if(isFired) {
      firstPositive = item;
      break;
    }
  }

  Int_t closest = firstPositive < TMath::Abs(firstNegative) ? firstPositive : TMath::Abs(firstNegative);
  if(firstPositive==100 && firstNegative==100) closest=0;
  return closest;
}

//__________________________________________________________________________
Int_t  AliESDHeader::GetIRInt2LastInteractionMap() const
{
  //
  // Calculation of the last interaction
  //
  SetIRInteractionMap();

  Int_t lastNegative=0;
  for(Int_t item=-90; item<=-1; item++) {
    Int_t bin = FindIRIntInteractionsBXMap(item);
    Bool_t isFired = fIRInt2InteractionsMap.TestBitNumber(bin);
    if(isFired) {
      lastNegative = item;
      break;
    }
  }
  Int_t lastPositive=0;
  for(Int_t item=90; item>=1; item--) {
    Int_t bin = FindIRIntInteractionsBXMap(item);
    Bool_t isFired = fIRInt2InteractionsMap.TestBitNumber(bin);
    if(isFired) {
      lastPositive = item;
      break;
    }
  }

  Int_t last = lastPositive > TMath::Abs(lastNegative) ? lastPositive : TMath::Abs(lastNegative);
  return last;
}
