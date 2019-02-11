// -*- mode: C++ -*- 
#ifndef ALIESD_H
#define ALIESD_H
/* Copyright(c) 1998-1999, ALICE Experiment at CERN, All rights reserved. *
 * See cxx source for full Copyright notice                               */


/* $Id: AliESD.h 52237 2011-10-20 19:26:08Z hristov $ */

//-------------------------------------------------------------------------
//                          Class AliESD
//   This is the class to deal with during the physical analysis of data
//      
//         Origin: Iouri Belikov, CERN, Jouri.Belikov@cern.ch 
//-------------------------------------------------------------------------

#include <TClonesArray.h>
#include <TObject.h>
#include <TArrayF.h>

#include "AliESDMuonTrack.h"
#include "AliESDPmdTrack.h"
#include "AliESDTrdTrack.h"
#include "AliESDVertex.h"
#include "AliESDcascade.h"
#include "AliESDkink.h"
#include "AliESDtrack.h"
#include "AliESDHLTtrack.h"
#include "AliESDCaloCluster.h"
#include "AliESDv0.h"
#include "AliESDFMD.h"
#include "AliESDVZERO.h"
#include "AliMultiplicity.h"
#include "AliRawDataErrorLog.h"
#include "AliESDACORDE.h"
#include "AliESDAD.h"

class AliESDfriend;

class AliESD : public TObject {
public:
  AliESD();
  AliESD(const AliESD&);  
  virtual ~AliESD(); 

  void SetESDfriend(const AliESDfriend *f);
  void GetESDfriend(AliESDfriend *f) const;

  void SetEventNumberInFile(Int_t n) {fEventNumberInFile=n;}
  void SetBunchCrossNumber(UShort_t n) {fBunchCrossNumber=n;}
  void SetOrbitNumber(UInt_t n) {fOrbitNumber=n;}
  void SetPeriodNumber(UInt_t n) {fPeriodNumber=n;}
  void SetRunNumber(Int_t n) {fRunNumber=n;}
  void SetTimeStamp(UInt_t timeStamp){fTimeStamp = timeStamp;}
  void SetEventType(UInt_t eventType){fEventType = eventType;}
  void SetTriggerMask(ULong64_t n) {fTriggerMask=n;}
  void SetTriggerMaskNext50(ULong64_t n) {fTriggerMaskNext50=n;}
  void SetTriggerCluster(UChar_t n) {fTriggerCluster = n;}
  void SetMagneticField(Float_t mf){fMagneticField = mf;}
  Float_t GetMagneticField() const {return fMagneticField;}
  
  AliESDtrack *GetTrack(Int_t i) const {
    return (AliESDtrack *)fTracks.UncheckedAt(i);
  }
  AliESDHLTtrack *GetHLTConfMapTrack(Int_t i) const {
    return (AliESDHLTtrack *)fHLTConfMapTracks.UncheckedAt(i);
  }
  AliESDHLTtrack *GetHLTHoughTrack(Int_t i) const {
    return (AliESDHLTtrack *)fHLTHoughTracks.UncheckedAt(i);
  }
  AliESDMuonTrack *GetMuonTrack(Int_t i) const {
    return (AliESDMuonTrack *)fMuonTracks.UncheckedAt(i);
  }
  AliESDPmdTrack *GetPmdTrack(Int_t i) const {
    return (AliESDPmdTrack *)fPmdTracks.UncheckedAt(i);
  }
  AliESDTrdTrack *GetTrdTrack(Int_t i) const {
    return (AliESDTrdTrack *)fTrdTracks.UncheckedAt(i);
  }

  Bool_t Clean(Float_t *cleanPars);
  Bool_t RemoveKink(Int_t i);
  Bool_t RemoveV0(Int_t i);
  Bool_t RemoveTrack(Int_t i);

  Int_t  AddTrack(const AliESDtrack *t) {
    AliESDtrack * track = new(fTracks[fTracks.GetEntriesFast()]) AliESDtrack(*t);track->SetID(fTracks.GetEntriesFast()-1);return  track->GetID();
  }
  void AddHLTConfMapTrack(const AliESDHLTtrack *t) {
    new(fHLTConfMapTracks[fHLTConfMapTracks.GetEntriesFast()]) AliESDHLTtrack(*t);
  }
  void AddHLTHoughTrack(const AliESDHLTtrack *t) {
    new(fHLTHoughTracks[fHLTHoughTracks.GetEntriesFast()]) AliESDHLTtrack(*t);
  }
  void AddMuonTrack(const AliESDMuonTrack *t) {
    new(fMuonTracks[fMuonTracks.GetEntriesFast()]) AliESDMuonTrack(*t);
  }
  void AddPmdTrack(const AliESDPmdTrack *t) {
    new(fPmdTracks[fPmdTracks.GetEntriesFast()]) AliESDPmdTrack(*t);
  }
  void AddTrdTrack(const AliESDTrdTrack *t) {
    new(fTrdTracks[fTrdTracks.GetEntriesFast()]) AliESDTrdTrack(*t);
  }

  AliESDv0 *GetV0(Int_t i) const {
    return (AliESDv0 *)fV0s.UncheckedAt(i);
  }
  Int_t AddV0(const AliESDv0 *v);

  AliESDcascade *GetCascade(Int_t i) const {
    return (AliESDcascade *)fCascades.UncheckedAt(i);
  }
  void AddCascade(const AliESDcascade *c) {
    new(fCascades[fCascades.GetEntriesFast()]) AliESDcascade(*c);
  }

  AliESDkink *GetKink(Int_t i) const {
    return (AliESDkink *)fKinks.UncheckedAt(i);
  }
  Int_t AddKink(const AliESDkink *c) {
    AliESDkink * kink = new(fKinks[fKinks.GetEntriesFast()]) AliESDkink(*c);
    kink->SetID(fKinks.GetEntriesFast());
    return fKinks.GetEntriesFast()-1;
  }

  AliESDCaloCluster *GetCaloCluster(Int_t i) const {
    return (AliESDCaloCluster *)fCaloClusters.UncheckedAt(i);
  }
  Int_t AddCaloCluster(const AliESDCaloCluster *c) {
    AliESDCaloCluster *clus = new(fCaloClusters[fCaloClusters.GetEntriesFast()]) AliESDCaloCluster(*c);
    clus->SetID(fCaloClusters.GetEntriesFast()-1);
    return fCaloClusters.GetEntriesFast()-1;
  }
    
  void AddPHOSTriggerPosition(TArrayF array)   { fPHOSTriggerPosition    = new TArrayF(array) ; }
  void AddPHOSTriggerAmplitudes(TArrayF array) { fPHOSTriggerAmplitudes  = new TArrayF(array) ; }
  void AddEMCALTriggerPosition(TArrayF array)  { fEMCALTriggerPosition   = new TArrayF(array) ; }
  void AddEMCALTriggerAmplitudes(TArrayF array){ fEMCALTriggerAmplitudes = new TArrayF(array) ; }

  void SetVertex(const AliESDVertex *vertex) {
     new (&fSPDVertex) AliESDVertex(*vertex);
  }
  const AliESDVertex *GetVertex() const {return &fSPDVertex;}

  void SetMultiplicity(const AliMultiplicity *mul) {
     new (&fSPDMult) AliMultiplicity(*mul);
  }
  const AliMultiplicity *GetMultiplicity() const {return &fSPDMult;}

  void SetPrimaryVertex(const AliESDVertex *vertex) {
     new (&fPrimaryVertex) AliESDVertex(*vertex);
  }
  const AliESDVertex *GetPrimaryVertex() const {return &fPrimaryVertex;}

  void SetDiamond(const AliESDVertex *vertex);
  Float_t GetDiamondX() const {return fDiamondXY[0];}
  Float_t GetDiamondY() const {return fDiamondXY[1];}
  Float_t GetSigma2DiamondX() const {return fDiamondCovXY[0];}
  Float_t GetSigma2DiamondY() const {return fDiamondCovXY[2];}
  void GetDiamondCovXY(Float_t cov[3]) const {
    for(Int_t i=0;i<3;i++) cov[i]=fDiamondCovXY[i];
  }

  Int_t  GetEventNumberInFile() const {return fEventNumberInFile;}
  UShort_t GetBunchCrossNumber() const {return fBunchCrossNumber;}
  UInt_t GetOrbitNumber() const {return fOrbitNumber;}
  UInt_t GetPeriodNumber() const {return fPeriodNumber;}
  Int_t  GetRunNumber() const {return fRunNumber;}
  UInt_t    GetTimeStamp()  const { return fTimeStamp;}
  UInt_t    GetEventType()  const { return fEventType;}
  ULong64_t GetTriggerMask() const {return fTriggerMask;}
  ULong64_t GetTriggerMaskNext50() const {return fTriggerMaskNext50;}
  UChar_t  GetTriggerCluster() const {return fTriggerCluster;}
  
  Int_t GetNumberOfTracks()     const {return fTracks.GetEntriesFast();}
  Int_t GetNumberOfHLTConfMapTracks()     const {return fHLTConfMapTracks.GetEntriesFast();}
  Int_t GetNumberOfHLTHoughTracks()     const {return fHLTHoughTracks.GetEntriesFast();}
  Int_t GetNumberOfMuonTracks() const {return fMuonTracks.GetEntriesFast();}
  Int_t GetNumberOfPmdTracks() const {return fPmdTracks.GetEntriesFast();}
  Int_t GetNumberOfTrdTracks() const {return fTrdTracks.GetEntriesFast();}
  Int_t GetNumberOfV0s()      const {return fV0s.GetEntriesFast();}
  Int_t GetNumberOfCascades() const {return fCascades.GetEntriesFast();}
  Int_t GetNumberOfKinks() const {return fKinks.GetEntriesFast();}
  Int_t GetNumberOfCaloClusters() const {return fCaloClusters.GetEntriesFast();}

  Int_t GetNumberOfEMCALClusters() const {return fEMCALClusters;}
  void  SetNumberOfEMCALClusters(Int_t clus) {fEMCALClusters = clus;}
  Int_t GetFirstEMCALCluster() const {return fFirstEMCALCluster;}
  void  SetFirstEMCALCluster(Int_t index) {fFirstEMCALCluster = index;}
  TArrayF *GetEMCALTriggerPosition() const {return  fEMCALTriggerPosition;}
  TArrayF *GetEMCALTriggerAmplitudes() const {return  fEMCALTriggerAmplitudes;}

  Int_t GetNumberOfPHOSClusters() const {return fPHOSClusters;}
  void  SetNumberOfPHOSClusters(Int_t part) { fPHOSClusters = part ; }
  void  SetFirstPHOSCluster(Int_t index) { fFirstPHOSCluster = index ; } 
  Int_t GetFirstPHOSCluster() const  { return fFirstPHOSCluster ; }
  TArrayF *GetPHOSTriggerPosition() const {return  fPHOSTriggerPosition;}
  TArrayF *GetPHOSTriggerAmplitudes() const {return  fPHOSTriggerAmplitudes;}


  Double32_t GetT0zVertex() const {return fT0zVertex;}
  void SetT0zVertex(Double32_t z) {fT0zVertex=z;}
  Double32_t GetT0() const {return fT0timeStart;}
  void SetT0(Double32_t timeStart) {fT0timeStart = timeStart;}
  Float_t GetT0clock() const {return fT0clock;}
  void SetT0clock(Float_t timeStart) {fT0clock = timeStart;}
  Double32_t GetT0TOF(Int_t i) const {return fT0TOF[i];}
  const Double32_t * GetT0TOF() const {return fT0TOF;}
  void SetT0TOF(Int_t icase, Float_t time) { fT0TOF[icase] = time;}
  Int_t GetT0Trig() const {return fT0trig;}
  void SetT0Trig(Int_t tvdc) {fT0trig = tvdc;}
  const Double32_t * GetT0time() const {return fT0time;}
  void SetT0time(Double32_t time[24]) {
  for (Int_t i=0; i<24; i++) fT0time[i] = time[i];
  }
  const Double32_t * GetT0amplitude() const {return fT0amplitude;}
  void SetT0amplitude(Double32_t amp[24]) {
    for (Int_t i=0; i<24; i++) fT0amplitude[i] = amp[i];
  }
    
  Float_t GetZDCN1Energy() const {return fZDCN1Energy;}
  Float_t GetZDCP1Energy() const {return fZDCP1Energy;}
  Float_t GetZDCN2Energy() const {return fZDCN2Energy;}
  Float_t GetZDCP2Energy() const {return fZDCP2Energy;}
  Float_t GetZDCEMEnergy() const {return fZDCEMEnergy;}
  Int_t   GetZDCParticipants() const {return fZDCParticipants;}
  void    SetZDC(Float_t n1Energy, Float_t p1Energy, Float_t emEnergy,
                 Float_t n2Energy, Float_t p2Energy, Int_t participants) 
   {fZDCN1Energy=n1Energy; fZDCP1Energy=p1Energy; fZDCEMEnergy=emEnergy;
    fZDCN2Energy=n2Energy; fZDCP2Energy=p2Energy; fZDCParticipants=participants;}

  void ResetV0s() { fV0s.Clear(); }
  void ResetCascades() { fCascades.Clear(); }
  void Reset();

  void  Print(Option_t *option="") const;

  void SetFMDData(AliESDFMD * obj) { fESDFMD = new AliESDFMD(*obj); }
  AliESDFMD *GetFMDData(){ return fESDFMD; }
   
  void SetVZEROData(AliESDVZERO * obj) { fESDVZERO = new AliESDVZERO(*obj); }
  AliESDVZERO *GetVZEROData(){ return fESDVZERO; }
  void SetACORDEData(AliESDACORDE * obj){ fESDACORDE = new AliESDACORDE(*obj); } 
  AliESDACORDE *GetACORDEDAta(){ return fESDACORDE; }
  void SetADData(AliESDAD * obj){ fESDAD = new AliESDAD(*obj); } 
  AliESDAD *GetADData(){ return fESDAD; }



  AliRawDataErrorLog *GetErrorLog(Int_t i) const {
    return (AliRawDataErrorLog *)fErrorLogs.UncheckedAt(i);
  }
  void  AddRawDataErrorLog(const AliRawDataErrorLog *log) {
    new(fErrorLogs[fErrorLogs.GetEntriesFast()]) AliRawDataErrorLog(*log);
  }
  Int_t GetNumberOfErrorLogs()   const {return fErrorLogs.GetEntriesFast();}

protected:
  AliESD &operator=(const AliESD& source);

  // Event Identification

  Int_t        fEventNumberInFile;// running Event count in the file
  UShort_t     fBunchCrossNumber;// Bunch Crossing Number
  UInt_t       fOrbitNumber;     // Orbit Number
  UInt_t       fPeriodNumber;    // Period Number
  Int_t        fRunNumber;       // Run Number
  UInt_t       fTimeStamp;       // Time stamp
  UInt_t       fEventType;       // Type of Event
  ULong64_t    fTriggerMask;     // Trigger Type (mask)
  ULong64_t    fTriggerMaskNext50; // Trigger Type (mask)
  UChar_t      fTriggerCluster;  // Trigger cluster (mask)
  Int_t        fRecoVersion;     // Version of reconstruction 
  Float_t      fMagneticField;   // Solenoid Magnetic Field in kG : for compatibility with AliMagF

  Float_t      fZDCN1Energy;      // reconstructed energy in the neutron ZDC
  Float_t      fZDCP1Energy;      // reconstructed energy in the proton ZDC
  Float_t      fZDCN2Energy;      // reconstructed energy in the neutron ZDC
  Float_t      fZDCP2Energy;      // reconstructed energy in the proton ZDC
  Float_t      fZDCEMEnergy;     // reconstructed energy in the electromagnetic ZDC
  Int_t        fZDCParticipants; // number of participants estimated by the ZDC


  Double32_t   fT0zVertex;       // vertex z position estimated by the T0
  AliESDVertex fSPDVertex;       // Primary vertex estimated by the SPD
  AliESDVertex fPrimaryVertex;   // Primary vertex estimated using ESD tracks
  Float_t      fDiamondXY[2];    // Interaction diamond (x,y) in RUN
  Float_t      fDiamondCovXY[3]; // Interaction diamond covariance (x,y) in RUN
  AliMultiplicity fSPDMult;      // SPD tracklet multiplicity

  Float_t      fT0clock;     // backward compatibility
  Double32_t   fT0TOF[3];     // interaction time in ns ( A&C, A, C)
  Double32_t   fT0timeStart;     // interaction time estimated by the T0
  Int_t        fT0trig;            // T0 trigger signals
  Double32_t   fT0time[24];      // best TOF on each T0 PMT
  Double32_t   fT0amplitude[24]; // number of particles(MIPs) on each T0 PMT


  TClonesArray fTracks;          // ESD tracks
  TClonesArray fHLTConfMapTracks;// HLT ESD tracks from Conformal Mapper method
  TClonesArray fHLTHoughTracks;  // HLT ESD tracks from Hough Transform method
  TClonesArray fMuonTracks;      // MUON ESD tracks
  TClonesArray fPmdTracks;       // PMD ESD tracks
  TClonesArray fTrdTracks;       // TRD ESD tracks (triggered)
  TClonesArray fV0s;             // V0 vertices
  TClonesArray fCascades;        // Cascade vertices
  TClonesArray fKinks;           // Kinks
  TClonesArray fCaloClusters;    // Calorimeter clusters for PHOS/EMCAL
  Int_t        fEMCALClusters;   // Number of EMCAL clusters (subset of caloclusters)
  Int_t        fFirstEMCALCluster; // First EMCAL cluster in the fCaloClusters list 
  TArrayF   *fEMCALTriggerPosition; ///(x,y,z of 2x2 and x,y,z of nxn) not position of centroid but of patch corner
  TArrayF   *fEMCALTriggerAmplitudes; //(2x2 max ampl, 2x2 amp out of patch,  nxn max ampl, nxn amp out of patch)

  Int_t        fPHOSClusters;     // Number of PHOS clusters (subset of caloclusters)
  Int_t        fFirstPHOSCluster; // First PHOS cluster in the fCaloClusters list 
  TArrayF   *fPHOSTriggerPosition; //(x,y,z of 2x2 and x,y,z of nxn), not position of centroid but of patch corner
  TArrayF   *fPHOSTriggerAmplitudes; //(2x2 max ampl, 2x2 amp out of patch,  nxn max ampl, nxn amp out of patch)

  AliESDFMD   *fESDFMD;   // FMD object containing rough multiplicity
  AliESDVZERO *fESDVZERO; // VZERO object containing rough multiplicity
  AliESDACORDE *fESDACORDE; // ACORDE ESD object containing bit pattern
  AliESDAD *fESDAD; // AD ESD object containing bit pattern
  TClonesArray fErrorLogs;        // Raw-data reading error messages

  ClassDef(AliESD,24)  //ESD class 
};
#endif 

