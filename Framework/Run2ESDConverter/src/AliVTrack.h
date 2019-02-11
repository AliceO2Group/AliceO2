#ifndef AliVTrack_H
#define AliVTrack_H
/* Copyright(c) 1998-2008, ALICE Experiment at CERN, All rights reserved. *
 * See cxx source for full Copyright notice                               */


//-------------------------------------------------------------------------
//     base class for ESD and AOD tracks
//     Author: A. Dainese
//-------------------------------------------------------------------------

#include <TBits.h>

#include "AliVParticle.h"
#include "AliPID.h"
#include "AliVMisc.h"

class AliVEvent;
class AliVVertex;
class AliESDVertex;
class AliExternalTrackParam;
class AliTPCdEdxInfo;
class AliDetectorPID;
class AliTOFHeader;

 
class AliVTrack: public AliVParticle {

public:
  enum  {
    kITSin        = 0x1
    ,kITSout      = 0x2
    ,kITSrefit    = 0x4
    ,kITSpid      = 0x8
    ,kTPCin       = 0x10
    ,kTPCout      = 0x20
    ,kTPCrefit    = 0x40
    ,kTPCpid      = 0x80
    ,kTRDin       = 0x100
    ,kTRDout      = 0x200
    ,kTRDrefit    = 0x400
    ,kTRDpid      = 0x800
    ,kTOFin       = 0x1000
    ,kTOFout      = 0x2000
    ,kTOFrefit    = 0x4000
    ,kTOFpid      = 0x8000
    ,kHMPIDout    = 0x10000
    ,kHMPIDpid    = 0x20000
    ,kEMCALmatch  = 0x40000
    ,kTRDbackup   = 0x80000
    ,kTOFmismatch = 0x100000
    ,kPHOSmatch   = 0x200000
    ,kITSupg      = 0x400000     // flag that in the ITSupgrade reco
    //
    ,kSkipFriend  = 0x800000     // flag to skip friend storage
    //
    ,kGlobalMerge = 0x1000000
    ,kMultInV0    = 0x2000000     //BIT(25): assumed to be belong to V0 in multiplicity estimates
    ,kMultSec     = 0x4000000     //BIT(26): assumed to be secondary (due to the DCA) in multiplicity estimates
    ,kEmbedded    = 0x8000000     // BIT(27), 1<<27: Is a track that has been embedded into the event
    //
    ,kITSpureSA   = 0x10000000
    ,kTRDStop     = 0x20000000
    ,kESDpid      = 0x40000000
    ,kTIME        = 0x80000000
  };
  // since with enum we cannot go above 32 bits, we have to define static constants here
  static const ULong64_t kTRDupdate; // Flag TRD updating the ESD kinematics
  
  enum {
    kTRDnPlanes = 6,
    kEMCALNoMatch = -4096,
    kTOFBCNA = -100
  };

  AliVTrack() { }
  virtual ~AliVTrack() { }
  AliVTrack(const AliVTrack& vTrack); 
  AliVTrack& operator=(const AliVTrack& vTrack);
  // constructor for reinitialisation of vtable
  AliVTrack( AliVConstructorReinitialisationFlag f) :AliVParticle(f){}

  virtual Bool_t  IsPureITSStandalone() const {return kFALSE;}
  virtual const AliVEvent* GetEvent() const {return 0;}
  virtual Int_t    GetID() const = 0;
  virtual UChar_t  GetITSClusterMap() const = 0;
  virtual UChar_t  GetITSSharedClusterMap() const {return 0;}
  virtual Bool_t   HasPointOnITSLayer(Int_t /*i*/) const { return kFALSE; }
  virtual Bool_t   HasSharedPointOnITSLayer(Int_t /*i*/) const { return kFALSE; }
  virtual void     GetITSdEdxSamples(Double_t s[4]) const {for (int i=4;i--;) s[i]=0;};
  virtual const TBits* GetTPCClusterMapPtr() const {return NULL;}
  virtual const TBits* GetTPCFitMapPtr()     const {return NULL;}
  virtual const TBits* GetTPCSharedMapPtr()  const {return NULL;}
  virtual Float_t  GetTPCClusterInfo(Int_t /*nNeighbours*/, Int_t /*type*/, Int_t /*row0*/=0, Int_t /*row1*/=159, Int_t /*type*/= 0) const {return 0.;}
  virtual Bool_t GetTPCdEdxInfo( AliTPCdEdxInfo & ) const { return 0; } // return 0 if AliTPCdEdxInfo does not exist
  virtual UShort_t GetTPCNcls() const { return 0;}
  virtual UShort_t GetTPCNclsF() const { return 0;}
  virtual Double_t GetTPCchi2() const {return 0;}
  virtual Double_t GetTRDslice(Int_t /*plane*/, Int_t /*slice*/) const { return -1.; }
  virtual Int_t    GetNumberOfTRDslices() const { return 0; }
  virtual UChar_t  GetTRDncls() const {return 0;}
  virtual UChar_t  GetTRDntrackletsPID() const { return 0;}
  virtual void     SetDetectorPID(const AliDetectorPID */*pid*/) {;}
  virtual const    AliDetectorPID* GetDetectorPID() const { return 0x0; }
  virtual Double_t GetTRDchi2()          const { return -1;}
  virtual Int_t    GetNumberOfClusters() const {return 0;}
  virtual Double_t GetITSchi2()          const {return 0;}
  virtual Float_t GetTPCCrossedRows() const {return 0;}

  virtual Bool_t RelateToVVertex(const AliVVertex* /*vtx*/,
			Double_t /*b*/, Double_t /*maxd*/,
			AliExternalTrackParam* /*cParam*/=0) {return kFALSE;}
  virtual Bool_t RelateToVVertexTPC(const AliVVertex* /*vtx*/,
			Double_t /*b*/, Double_t /*maxd*/,
			AliExternalTrackParam* /*cParam*/=0) {return kFALSE;}
  virtual Bool_t RelateToVVertexBxByBz(const AliVVertex * /*vtx*/,
			Double_t /*b*/[3], Double_t /*maxd*/,
			AliExternalTrackParam* /*cParam*/=0) {return kFALSE;}
  virtual Bool_t RelateToVVertexTPCBxByBz(const AliVVertex * /*vtx*/,
			Double_t /*b*/[3], Double_t /*maxd*/,
			AliExternalTrackParam* /*cParam*/=0) {return kFALSE;}
  
  virtual Bool_t RelateToVertex(const AliESDVertex * /*vtx*/,
			Double_t /*b*/, Double_t /*maxd*/,
			AliExternalTrackParam* /*cParam*/=0) {return kFALSE;}
  virtual Bool_t RelateToVertexTPC(const AliESDVertex * /*vtx*/,
			Double_t /*b*/, Double_t /*maxd*/,
			AliExternalTrackParam* /*cParam*/=0) {return kFALSE;}
  virtual Bool_t RelateToVertexBxByBz(const AliESDVertex * /*vtx*/,
			Double_t /*b*/[3], Double_t /*maxd*/,
			AliExternalTrackParam* /*cParam*/=0) {return kFALSE;}
  virtual Bool_t RelateToVertexTPCBxByBz(const AliESDVertex * /*vtx*/,
			Double_t /*b*/[3], Double_t /*maxd*/,
			AliExternalTrackParam* /*cParam*/=0) {return kFALSE;}
  
  virtual void GetImpactParameters(Float_t [], Float_t []) const {;}
  virtual void GetImpactParameters(Float_t &/*&xy*/,Float_t &/*&z*/) const {;}
  virtual void GetImpactParametersTPC(Float_t &/*&xy*/,Float_t &/*&z*/) const {;}
  virtual void GetImpactParametersTPC(Float_t [] /*p[2]*/, Float_t [] /*cov[3]*/) const {;}
  
  virtual Int_t GetEMCALcluster()     const {return kEMCALNoMatch;}
  virtual void SetEMCALcluster(Int_t)       {;}
  virtual Bool_t IsEMCAL()            const {return kFALSE;}

  virtual Double_t GetTrackPhiOnEMCal()  const {return -999;}
  virtual Double_t GetTrackEtaOnEMCal()  const {return -999;}
  virtual Double_t GetTrackPtOnEMCal()   const {return -999;}
  virtual Double_t GetTrackPOnEMCal()    const {return -999;}
  virtual Bool_t IsExtrapolatedToEMCAL() const {return GetTrackPtOnEMCal()!=-999;} 
  virtual void SetTrackPhiEtaPtOnEMCal(Double_t,Double_t,Double_t=-999) {;}

  virtual Int_t GetPHOScluster()      const {return -1;}
  virtual void SetPHOScluster(Int_t)        {;}
  virtual Bool_t IsPHOS()             const {return kFALSE;}
  virtual void   SetPIDForTracking(Int_t ) {}
  virtual Int_t  GetPIDForTracking() const {return -999;}
  
  //pid info
  virtual void     SetStatus(ULong64_t /*flags*/) {;}
  virtual void     ResetStatus(ULong64_t /*flags*/) {;}

  virtual Double_t  GetITSsignal()       const {return 0.;}
  virtual Double_t  GetITSsignalTunedOnData() const {return 0.;}
  virtual void      SetITSsignalTunedOnData(Double_t /*signal*/) {}
  virtual Double_t  GetTPCsignal()       const {return 0.;}
  virtual Double_t  GetTPCsignalTunedOnData() const {return 0.;}
  virtual void      SetTPCsignalTunedOnData(Double_t /*signal*/) {}
  virtual UShort_t  GetTPCsignalN()      const {return 0 ;}
  virtual Double_t  GetTPCmomentum()     const {return 0.;}
  virtual Double_t  GetTPCTgl()          const {return 0.;}
  virtual Int_t     GetTPCLabel()        const {return 0;}
  virtual Double_t  GetTgl()             const {return 0.;}
  virtual Double_t  GetTOFsignal()       const {return 0.;}
  virtual Double_t  GetTOFsignalTunedOnData() const {return 0.;}
  virtual void      SetTOFsignalTunedOnData(Double_t /*signal*/) {}
  virtual Double_t  GetHMPIDsignal()     const {return 0.;}
  virtual Double_t  GetTRDsignal()       const {return 0.;}
  virtual UChar_t GetTRDNchamberdEdx() const {return 0;}
  virtual UChar_t GetTRDNclusterdEdx() const {return 0;}

  virtual Double_t  GetHMPIDoccupancy()  const {return 0.;}
  
  virtual Int_t     GetHMPIDcluIdx()     const {return 0;}
  
  virtual void GetHMPIDtrk(Float_t &/*&x*/, Float_t &/*y*/, Float_t &/*th*/, Float_t &/*ph*/) const {;}  
  virtual void GetHMPIDmip(Float_t &/*x*/, Float_t &/*y*/, Int_t &/*q*/,Int_t &/*nph*/) const {;}
  
  virtual Bool_t GetOuterHmpPxPyPz(Double_t */*p*/) const {return kFALSE;}

  virtual const AliExternalTrackParam * GetInnerParam() const { return NULL;}
  virtual const AliExternalTrackParam * GetOuterParam() const { return NULL;}
  virtual const AliExternalTrackParam * GetTPCInnerParam() const { return NULL;}
  virtual const AliExternalTrackParam * GetConstrainedParam() const {return NULL;}

  virtual void      GetIntegratedTimes(Double_t */*times*/, Int_t nspec=AliPID::kSPECIESC) const;
  virtual Double_t  GetTRDmomentum(Int_t /*plane*/, Double_t */*sp*/=0x0) const {return 0.;}
  virtual void      GetHMPIDpid(Double_t */*p*/) const {;}
  virtual Double_t  GetIntegratedLength() const { return 0.;}
  
  virtual ULong64_t  GetStatus() const = 0;
  virtual Bool_t   GetXYZ(Double_t *p) const = 0;
  virtual Bool_t   GetXYZAt(Double_t /*x*/, Double_t /*b*/, Double_t* /*r*/ ) const;
  virtual Double_t GetBz() const;
  virtual void     GetBxByBz(Double_t b[3]) const;
  virtual Bool_t   GetCovarianceXYZPxPyPz(Double_t cv[21]) const = 0;
  virtual Bool_t   PropagateToDCA(const AliVVertex *vtx,Double_t b,Double_t maxd,Double_t dz[2],Double_t covar[3]) = 0;
  virtual Int_t    GetNcls(Int_t /*idet*/) const { return 0; }
  virtual Bool_t   GetPxPyPz(Double_t */*p*/) const { return kFALSE; }
  virtual void     SetID(Short_t /*id*/) {;}
  virtual Int_t    GetTOFBunchCrossing(Double_t = 0, Bool_t = kFALSE) const { return kTOFBCNA;}
  virtual Double_t GetTOFExpTDiff(Double_t /*b*/=0, Bool_t /*pidTPConly*/=kTRUE) const {return kTOFBCNA*25;}
  virtual const AliTOFHeader *GetTOFHeader() const {return NULL;};

  //---------------------------------------------------------------------------
  //--the calibration interface--
  //--to be used in online calibration/QA
  //--should also be implemented in ESD so it works offline as well
  //-----------
  virtual Int_t GetTrackParam         ( AliExternalTrackParam& ) const {return 0;}
  virtual Int_t GetTrackParamRefitted ( AliExternalTrackParam& ) const {return 0;}
  virtual Int_t GetTrackParamIp       ( AliExternalTrackParam& ) const {return 0;}
  virtual Int_t GetTrackParamTPCInner ( AliExternalTrackParam& ) const {return 0;}
  virtual Int_t GetTrackParamOp       ( AliExternalTrackParam& ) const {return 0;}
  virtual Int_t GetTrackParamCp       ( AliExternalTrackParam& ) const {return 0;}
  virtual Int_t GetTrackParamITSOut   ( AliExternalTrackParam& ) const {return 0;}

  virtual void  ResetTrackParamIp       ( const AliExternalTrackParam* ) {;}
  virtual void  ResetTrackParamOp       ( const AliExternalTrackParam* ) {;}
  virtual void  ResetTrackParamTPCInner ( const AliExternalTrackParam* ) {;}

  virtual Int_t GetNumberOfTPCClusters() const { return 0; } 
  virtual Int_t GetNumberOfITSClusters() const { return 0; } 
  virtual Int_t GetNumberOfTRDClusters() const { return 0; } 

  virtual Int_t             GetKinkIndex(Int_t /*i*/) const { return 0;}
  virtual Double_t          GetSigned1Pt()         const { return 0;}
  virtual Bool_t            IsOn(ULong64_t /*mask*/) const {return 0;}
  virtual Double_t          GetX()    const {return 0;}
  virtual Double_t          GetY()    const {return 0;}
  virtual Double_t          GetZ()    const {return 0;}
  virtual const Double_t   *GetParameter() const {return 0;}
  virtual Double_t          GetAlpha() const {return 0;}
  virtual Double_t          GetSnp()  const {return 0;}
  virtual Double_t          GetSigmaSnp2() const {return 0;}
  virtual UShort_t          GetTPCncls(Int_t /*row0*/=0, Int_t /*row1*/=159) const {return 0;}
  virtual Double_t          GetTOFsignalDz() const {return 0;}
  virtual Double_t          GetP() const {return 0;}
  virtual Double_t          GetSignedPt() const {return 0;}
  virtual Double_t          GetSign() const {return 0;}
  virtual void              GetDirection(Double_t []) const {;}
  virtual Double_t          GetLinearD(Double_t /*xv*/, Double_t /*yv*/) const {return 0;}
  virtual void              GetDZ(Double_t /*x*/,Double_t /*y*/,Double_t /*z*/,Double_t /*b*/,Float_t [] /*dz[2]*/) const {;}
  virtual Char_t            GetITSclusters(Int_t */**idx*/) const {return 0;}
  virtual UChar_t           GetTRDclusters(Int_t */**idx*/) const {return 0;}


  ClassDef(AliVTrack,1)  // base class for tracks
};

#endif
