// -*- mode: C++ -*- 
#ifndef ALIESDHEADER_H
#define ALIESDHEADER_H

/* Copyright(c) 1998-1999, ALICE Experiment at CERN, All rights reserved. *
 * See cxx source for full Copyright notice                               */

//-------------------------------------------------------------------------
//                      Class AliESDHeader
//   Header data
//   for the ESD   
//   Origin: Christian Klein-Boesing, CERN, Christian.Klein-Boesing@cern.ch 
//-------------------------------------------------------------------------

#include <TObjArray.h>
#include <TClonesArray.h>
#include <TBits.h>
#include "AliVHeader.h"
#include "AliTriggerScalersESD.h"
#include "AliTriggerScalersRecordESD.h"

class AliTriggerScalersESD;
class AliTriggerScalersRecordESD;
class AliTriggerIR;
class AliTriggerConfiguration;

class AliESDHeader: public AliVHeader {
public:
  AliESDHeader();
  virtual ~AliESDHeader();
  AliESDHeader(const AliESDHeader& header);
  AliESDHeader& operator=(const AliESDHeader& header);
  virtual void Copy(TObject &obj) const;

  void      SetTriggerMask(ULong64_t n) {fTriggerMask=n;}
  void      SetTriggerMaskNext50(ULong64_t n) {fTriggerMaskNext50=n;}
  void      SetOrbitNumber(UInt_t n) {fOrbitNumber=n;}
  void      SetTimeStamp(UInt_t timeStamp){fTimeStamp = timeStamp;}
  void      SetEventType(UInt_t eventType){fEventType = eventType;}
  void      SetEventSpecie(UInt_t eventSpecie){fEventSpecie = eventSpecie;}
  void      SetEventNumberInFile(Int_t n) {fEventNumberInFile=n;}
  void      SetBunchCrossNumber(UShort_t n) {fBunchCrossNumber=n;}
  void      SetPeriodNumber(UInt_t n) {fPeriodNumber=n;}
  void      SetTriggerCluster(UChar_t n) {fTriggerCluster = n;}
  Bool_t    AddTriggerIR(const AliTriggerIR* ir);
  void      SetCTPConfig(AliTriggerConfiguration* ctpConfig) {fCTPConfig=ctpConfig;};
//************Setters/Getters for Trigger Inputs and TriggerScalersRecordESD
  void SetL0TriggerInputs(UInt_t n) {fL0TriggerInputs=n;}
  void SetL1TriggerInputs(UInt_t n) {fL1TriggerInputs=n;}
  void SetL2TriggerInputs(UShort_t n) {fL2TriggerInputs=n;}
  UInt_t      GetL0TriggerInputs() const {return fL0TriggerInputs;}  
  UInt_t      GetL1TriggerInputs() const {return fL1TriggerInputs;} 
  UShort_t    GetL2TriggerInputs() const {return fL2TriggerInputs;} 
  void SetTriggerScalersRecord(AliTriggerScalersESD *scalerRun) {fTriggerScalers.AddTriggerScalers(scalerRun); }
  void SetTriggerScalersDeltaEvent(const AliTriggerScalersRecordESD *scalerRun) {fTriggerScalersDeltaEvent = *scalerRun; }
  void SetTriggerScalersDeltaRun(const AliTriggerScalersRecordESD *scalerRun) {fTriggerScalersDeltaRun = *scalerRun; }
  const AliTriggerScalersRecordESD* GetTriggerScalersRecord() const {return &fTriggerScalers; }
  const AliTriggerScalersRecordESD* GetTriggerScalersDeltaEvent() const {return &fTriggerScalersDeltaEvent; }
  const AliTriggerScalersRecordESD* GetTriggerScalersDeltaRun() const {return &fTriggerScalersDeltaRun; }
  const AliTriggerIR* GetTriggerIR(Int_t i) const { return (const AliTriggerIR*)fIRBufferArray[i]; }
  void SetActiveTriggerInputs(const char*name, Int_t index);
  const char* GetTriggerInputName(Int_t index, Int_t trglevel) const;
  TString     GetActiveTriggerInputs() const;
  TString     GetFiredTriggerInputs() const;
  Bool_t      IsTriggerInputFired(const char *name) const;
  const AliTriggerConfiguration*  GetCTPConfig() const { return fCTPConfig;}
  Int_t  FindIRIntInteractionsBXMap(Int_t difference) const;
  TBits  GetIRInt2InteractionMap() const { SetIRInteractionMap(); return fIRInt2InteractionsMap; }
  TBits  GetIRInt1InteractionMap() const { SetIRInteractionMap(); return fIRInt1InteractionsMap; }
  Int_t  GetIRInt2ClosestInteractionMap() const;
  Int_t  GetIRInt1ClosestInteractionMap(Int_t gap = 3) const;
  Int_t  GetIRInt2LastInteractionMap() const;
//**************************************************************************

  ULong64_t GetTriggerMask() const {return fTriggerMask;}
  ULong64_t GetTriggerMaskNext50() const {return fTriggerMaskNext50;}
  void      GetTriggerMaskAll(ULong64_t& low,ULong64_t& high) const {low=fTriggerMask;high=fTriggerMaskNext50;}
  UInt_t    GetOrbitNumber() const {return fOrbitNumber;}
  UInt_t    GetTimeStamp()  const { return fTimeStamp;}
  UInt_t    GetEventType()  const { return fEventType;}
  UInt_t    GetEventSpecie()  const { return fEventSpecie;}
  Int_t     GetEventNumberInFile() const {return fEventNumberInFile;}
  UShort_t  GetBunchCrossNumber() const {return fBunchCrossNumber;}
  UInt_t    GetPeriodNumber() const {return fPeriodNumber;}
  UChar_t   GetTriggerCluster() const {return fTriggerCluster;}
  Int_t     GetTriggerIREntries() const { return fIRBufferArray.GetEntriesFast();};
  Int_t     GetTriggerIREntries(Int_t int1, Int_t int2, Float_t deltaTime = 180.) const;
  TObjArray GetIRArray(Int_t int1, Int_t int2, Float_t deltaTime = 180.) const;
  void      Reset();
  void      Print(const Option_t *opt=0) const;

  enum {kNTriggerInputs = 60};   //24 L0, 24 L1 and 12 L2 inputs
  Char_t GetTPCNoiseFilterCounter(UInt_t index) {return fTPCNoiseFilterCounter[index%3];};
  void SetTPCNoiseFilterCounter(UInt_t index,UChar_t value) {fTPCNoiseFilterCounter[index%3]=value;};

private:
  void   SetIRInteractionMap() const;

private:

  // Event Identification
  ULong64_t    fTriggerMask;       // Trigger Type (mask) 1-50 bits
  ULong64_t    fTriggerMaskNext50; // Trigger Type (mask) 51-100 bits
  UInt_t       fOrbitNumber;       // Orbit Number
  UInt_t       fTimeStamp;         // Time stamp
  UInt_t       fEventType;         // Type of Event
  UInt_t       fEventSpecie;       // Reconstruction event specie (1-default,2-lowM,4-highM,8-cosmic,16-cal)
  UInt_t       fPeriodNumber;      // Period Number
  Int_t        fEventNumberInFile; // Running Event count in the file
  UShort_t     fBunchCrossNumber;  // Bunch Crossing Number
  UChar_t      fTriggerCluster;    // Trigger cluster (mask)
  UInt_t       fL0TriggerInputs;   // L0 Trigger Inputs (mask)
  UInt_t       fL1TriggerInputs;   // L1 Trigger Inputs (mask)
  UShort_t     fL2TriggerInputs;   // L2 Trigger Inputs (mask)
  AliTriggerScalersRecordESD fTriggerScalers;  //Trigger counters of triggered classes in event, interpolated to the event time
  AliTriggerScalersRecordESD fTriggerScalersDeltaEvent;  // Change in the trigger scalers between the two counter readings closest to the event time 
  AliTriggerScalersRecordESD fTriggerScalersDeltaRun;  // Total number of counts in the trigger scalers for the duration of the run
  enum {kNMaxIR = 3};              // Max number of interaction records (IR)
  AliTriggerIR*  fIRArray[kNMaxIR];// Array with trigger IRs 
  TObjArray    fTriggerInputsNames;// Array of TNamed of the active trigger inputs (L0,L1 and L2)
  AliTriggerConfiguration*  fCTPConfig; // Trigger configuration for the run
  TObjArray    fIRBufferArray;// Array with interaction records before and after triggered event
  mutable TBits   fIRInt2InteractionsMap;  // map of the Int2 events (normally 0TVX) near the event, that's Int2Id-EventId within -90 +90 BXs
  mutable TBits   fIRInt1InteractionsMap;  // map of the Int1 events (normally V0A&V0C) near the event, that's Int1Id-EventId within -90 +90 BXs
  UChar_t fTPCNoiseFilterCounter[3];  // filter counter [0]=sector, [1]-timebin/sector, [2]-padrowsector 


  ClassDef(AliESDHeader,14)
};

#endif
