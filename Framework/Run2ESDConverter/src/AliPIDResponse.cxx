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

/* $Id: AliPIDResponse.cxx 46193 2010-12-21 09:00:14Z wiechula $ */

//-----------------------------------------------------------------
//        Base class for handling the pid response               //
//        functions of all detectors                             //
//        and give access to the nsigmas                         //
//                                                               //
//   Origin: Jens Wiechula, Uni Tuebingen, jens.wiechula@cern.ch //
//-----------------------------------------------------------------

#include <TList.h>
#include <TObjArray.h>
#include <TPRegexp.h>
#include <TF1.h>
#include <TH2D.h>
#include <TSpline.h>
#include <TFile.h>
#include <TArrayI.h>
#include <TArrayF.h>
#include <TLinearFitter.h>
#include <TSystem.h>
#include <TMD5.h>
#include "TRandom.h"

#include <AliVEvent.h>
#include <AliVTrack.h>
#include <AliMCEvent.h>
#include <AliLog.h>
#include <AliPID.h>
#include <AliOADBContainer.h>
#include <AliTRDPIDResponseObject.h>
#include <AliTRDdEdxParams.h>
#include <AliTOFPIDParams.h>
#include <AliHMPIDPIDParams.h>

#include "AliPIDResponse.h"
#include "AliDetectorPID.h"

#include "AliMultSelectionBase.h"

ClassImp(AliPIDResponse);

Float_t AliPIDResponse::fgTOFmismatchProb = 0.0;

AliPIDResponse::AliPIDResponse(Bool_t isMC/*=kFALSE*/) :
TNamed("PIDResponse","PIDResponse"),
fITSResponse(isMC),
fTPCResponse(),
fTRDResponse(),
fTOFResponse(),
fHMPIDResponse(),
fEMCALResponse(),
fRange(5.),
fITSPIDmethod(kITSTruncMean),
fTuneMConData(kFALSE),
fTuneMConDataMask(kDetTOF|kDetTPC),
fIsMC(isMC),
fCachePID(kFALSE),
fOADBPath(),
fCustomTPCpidResponse(),
fCustomTPCpidResponseOADBFile(),
fCustomTPCetaMaps(),
fBeamType("PP"),
fLHCperiod(),
fMCperiodTPC(),
fMCperiodUser(),
fCurrentFile(),
fRecoPassName(),
fRecoPassNameUser(),
fCurrentAliRootRev(-1),
fRecoPass(0),
fRecoPassUser(-1),
fRun(-1),
fOldRun(-1),
fResT0A(75.),
fResT0C(65.),
fResT0AC(55.),
fTPCPIDResponseArray(NULL),
fArrPidResponseMaster(NULL),
fResolutionCorrection(NULL),
fOADBvoltageMaps(NULL),
fUseTPCEtaCorrection(kFALSE),
fUseTPCMultiplicityCorrection(kFALSE),
fUseTPCNewResponse(kTRUE),
fTRDPIDResponseObject(NULL),
fTRDdEdxParams(NULL),
fUseTRDEtaCorrection(kFALSE),
fUseTRDClusterCorrection(kFALSE),
fUseTRDCentralityCorrection(kFALSE),
fTOFtail(0.9),
fTOFPIDParams(NULL),
fHMPIDPIDParams(NULL),
fEMCALPIDParams(NULL),
fCurrentEvent(NULL),
fCurrentMCEvent(NULL),
fCurrCentrality(0.0),
fBeamTypeNum(kPP),
fNoTOFmism(kFALSE)
{
  //
  // default ctor
  //
  AliLog::SetClassDebugLevel("AliPIDResponse",0);
  AliLog::SetClassDebugLevel("AliESDpid",0);
  AliLog::SetClassDebugLevel("AliAODpidUtil",0);

}

//______________________________________________________________________________
AliPIDResponse::~AliPIDResponse()
{
  //
  // dtor
  //
  delete fTPCPIDResponseArray;
  delete fArrPidResponseMaster;
  delete fTRDPIDResponseObject;
  delete fTRDdEdxParams;
  delete fTOFPIDParams;
}

//______________________________________________________________________________
AliPIDResponse::AliPIDResponse(const AliPIDResponse &other) :
TNamed(other),
fITSResponse(other.fITSResponse),
fTPCResponse(other.fTPCResponse),
fTRDResponse(other.fTRDResponse),
fTOFResponse(other.fTOFResponse),
fHMPIDResponse(other.fHMPIDResponse),
fEMCALResponse(other.fEMCALResponse),
fRange(other.fRange),
fITSPIDmethod(other.fITSPIDmethod),
fTuneMConData(other.fTuneMConData),
fTuneMConDataMask(other.fTuneMConDataMask),
fIsMC(other.fIsMC),
fCachePID(other.fCachePID),
fOADBPath(other.fOADBPath),
fCustomTPCpidResponse(other.fCustomTPCpidResponse),
fCustomTPCpidResponseOADBFile(other.fCustomTPCpidResponseOADBFile),
fCustomTPCetaMaps(other.fCustomTPCetaMaps),
fBeamType("PP"),
fLHCperiod(),
fMCperiodTPC(),
fMCperiodUser(other.fMCperiodUser),
fCurrentFile(),
fRecoPassName(),
fRecoPassNameUser(),
fCurrentAliRootRev(other.fCurrentAliRootRev),
fRecoPass(0),
fRecoPassUser(other.fRecoPassUser),
fRun(-1),
fOldRun(-1),
fResT0A(75.),
fResT0C(65.),
fResT0AC(55.),
fTPCPIDResponseArray(NULL),
fArrPidResponseMaster(NULL),
fResolutionCorrection(NULL),
fOADBvoltageMaps(NULL),
fUseTPCEtaCorrection(other.fUseTPCEtaCorrection),
fUseTPCMultiplicityCorrection(other.fUseTPCMultiplicityCorrection),
fUseTPCNewResponse(other.fUseTPCNewResponse),
fTRDPIDResponseObject(NULL),
fTRDdEdxParams(NULL),
fUseTRDEtaCorrection(other.fUseTRDEtaCorrection),
fUseTRDClusterCorrection(other.fUseTRDClusterCorrection),
fUseTRDCentralityCorrection(other.fUseTRDCentralityCorrection),
fTOFtail(0.9),
fTOFPIDParams(NULL),
fHMPIDPIDParams(NULL),
fEMCALPIDParams(NULL),
fCurrentEvent(NULL),
fCurrentMCEvent(NULL),
fCurrCentrality(0.0),
fBeamTypeNum(kPP),
fNoTOFmism(other.fNoTOFmism)
{
  //
  // copy ctor
  //
}

//______________________________________________________________________________
AliPIDResponse& AliPIDResponse::operator=(const AliPIDResponse &other)
{
  //
  // copy ctor
  //
  if(this!=&other) {
    delete fArrPidResponseMaster;
    TNamed::operator=(other);
    fITSResponse=other.fITSResponse;
    fTPCResponse=other.fTPCResponse;
    fTRDResponse=other.fTRDResponse;
    fTOFResponse=other.fTOFResponse;
    fHMPIDResponse=other.fHMPIDResponse;
    fEMCALResponse=other.fEMCALResponse;
    fRange=other.fRange;
    fITSPIDmethod=other.fITSPIDmethod;
    fOADBPath=other.fOADBPath;
    fCustomTPCpidResponse=other.fCustomTPCpidResponse;
    fCustomTPCpidResponseOADBFile=other.fCustomTPCpidResponseOADBFile;
    fCustomTPCetaMaps=other.fCustomTPCetaMaps;
    fTuneMConData=other.fTuneMConData;
    fTuneMConDataMask=other.fTuneMConDataMask;
    fIsMC=other.fIsMC;
    fCachePID=other.fCachePID;
    fBeamType="PP";
    fBeamTypeNum=kPP;
    fLHCperiod="";
    fMCperiodTPC="";
    fMCperiodUser=other.fMCperiodUser;
    fCurrentFile="";
    fRecoPassName=other.fRecoPassName;
    fRecoPassNameUser=other.fRecoPassNameUser;
    fCurrentAliRootRev=other.fCurrentAliRootRev;
    fRecoPass=0;
    fRecoPassUser=other.fRecoPassUser;
    fRun=-1;
    fOldRun=-1;
    fResT0A=75.;
    fResT0C=65.;
    fResT0AC=55.;
    fTPCPIDResponseArray=NULL;
    fArrPidResponseMaster=NULL;
    fResolutionCorrection=NULL;
    fOADBvoltageMaps=NULL;
    fUseTPCEtaCorrection=other.fUseTPCEtaCorrection;
    fUseTPCMultiplicityCorrection=other.fUseTPCMultiplicityCorrection;
    fUseTPCNewResponse=other.fUseTPCNewResponse;
    fTRDPIDResponseObject=NULL;
    fTRDdEdxParams=NULL;
    fUseTRDEtaCorrection=other.fUseTRDEtaCorrection;
    fUseTRDClusterCorrection=other.fUseTRDClusterCorrection;
    fUseTRDCentralityCorrection=other.fUseTRDCentralityCorrection;
    fEMCALPIDParams=NULL;
    fTOFtail=0.9;
    fTOFPIDParams=NULL;
    fHMPIDPIDParams=NULL;
    fCurrentEvent=other.fCurrentEvent;
    fCurrentMCEvent=other.fCurrentMCEvent;
    fNoTOFmism = other.fNoTOFmism;

  }
  return *this;
}

//______________________________________________________________________________
Float_t AliPIDResponse::NumberOfSigmas(EDetector detector, const AliVParticle *vtrack, AliPID::EParticleType type) const
{
  //
  // NumberOfSigmas for 'detCode'
  //

  const AliVTrack *track=static_cast<const AliVTrack*>(vtrack);
  // look for cached value first
  const AliDetectorPID *detPID=track->GetDetectorPID();

  if ( detPID && detPID->HasNumberOfSigmas(detector)){
    return detPID->GetNumberOfSigmas(detector, type);
  } else if (fCachePID) {
    FillTrackDetectorPID(track, detector);
    detPID=track->GetDetectorPID();
    return detPID->GetNumberOfSigmas(detector, type);
  }

  return GetNumberOfSigmas(detector, track, type);
}

//______________________________________________________________________________
AliPIDResponse::EDetPidStatus AliPIDResponse::NumberOfSigmas(EDetector detCode, const AliVParticle *track,
                                                             AliPID::EParticleType type, Double_t &val) const
{
  //
  // NumberOfSigmas with detector status as return value
  //

  val=NumberOfSigmas(detCode, track, type);
  return CheckPIDStatus(detCode, (AliVTrack*)track);
}

//______________________________________________________________________________
// public buffered versions of the PID calculation
//

//______________________________________________________________________________
Float_t AliPIDResponse::NumberOfSigmasITS(const AliVParticle *vtrack, AliPID::EParticleType type) const
{
  //
  // Calculate the number of sigmas in the ITS
  //

  return NumberOfSigmas(kITS, vtrack, type);
}

//______________________________________________________________________________
Float_t AliPIDResponse::NumberOfSigmasTPC(const AliVParticle *vtrack, AliPID::EParticleType type) const
{
  //
  // Calculate the number of sigmas in the TPC
  //

  return NumberOfSigmas(kTPC, vtrack, type);
}

//______________________________________________________________________________
Float_t AliPIDResponse::NumberOfSigmasTPC( const AliVParticle *vtrack,
                                           AliPID::EParticleType type,
                                           AliTPCPIDResponse::ETPCdEdxSource dedxSource) const
{
  //get number of sigmas according the selected TPC gain configuration scenario
  const AliVTrack *track=static_cast<const AliVTrack*>(vtrack);

  Float_t nSigma=fTPCResponse.GetNumberOfSigmas(track, type, dedxSource, fUseTPCEtaCorrection, fUseTPCMultiplicityCorrection);

  return nSigma;
}

//______________________________________________________________________________
Float_t AliPIDResponse::NumberOfSigmasTRD(const AliVParticle *vtrack, AliPID::EParticleType type) const
{
  //
  // Calculate the number of sigmas in the TRD
  //
  return NumberOfSigmas(kTRD, vtrack, type);
}

//______________________________________________________________________________
Float_t AliPIDResponse::NumberOfSigmasTOF(const AliVParticle *vtrack, AliPID::EParticleType type) const
{
  //
  // Calculate the number of sigmas in the TOF
  //

  return NumberOfSigmas(kTOF, vtrack, type);
}

//______________________________________________________________________________
Float_t AliPIDResponse::NumberOfSigmasHMPID(const AliVParticle *vtrack, AliPID::EParticleType type) const
{
  //
  // Calculate the number of sigmas in the EMCAL
  //

  return NumberOfSigmas(kHMPID, vtrack, type);
}

//______________________________________________________________________________
Float_t AliPIDResponse::NumberOfSigmasEMCAL(const AliVParticle *vtrack, AliPID::EParticleType type) const
{
  //
  // Calculate the number of sigmas in the EMCAL
  //

  return NumberOfSigmas(kEMCAL, vtrack, type);
}

//______________________________________________________________________________
Float_t  AliPIDResponse::NumberOfSigmasEMCAL(const AliVParticle *vtrack, AliPID::EParticleType type, Double_t &eop, Double_t showershape[4])  const
{
  //
  // emcal nsigma with eop and showershape
  //
  AliVTrack *track=(AliVTrack*)vtrack;

  AliVCluster *matchedClus = NULL;

  Double_t mom     = -1.;
  Double_t pt      = -1.;
  Double_t EovP    = -1.;
  Double_t fClsE   = -1.;

  // initialize eop and shower shape parameters
  eop = -1.;
  for(Int_t i = 0; i < 4; i++){
    showershape[i] = -1.;
  }

  Int_t nMatchClus = -1;
  Int_t charge     = 0;

  // Track matching
  nMatchClus = track->GetEMCALcluster();
  if(nMatchClus > -1){

    mom    = track->P();
    pt     = track->Pt();
    charge = track->Charge();

    matchedClus = (AliVCluster*)fCurrentEvent->GetCaloCluster(nMatchClus);

    if(matchedClus){

      // matched cluster is EMCAL
      if(matchedClus->IsEMCAL()){

	fClsE       = matchedClus->E();
	EovP        = fClsE/mom;

	// fill used EMCAL variables here
	eop            = EovP; // E/p
	showershape[0] = matchedClus->GetNCells(); // number of cells in cluster
	showershape[1] = matchedClus->GetM02(); // long axis
	showershape[2] = matchedClus->GetM20(); // short axis
	showershape[3] = matchedClus->GetDispersion(); // dispersion

        // look for cached value first
        const AliDetectorPID *detPID=track->GetDetectorPID();
        const EDetector detector=kEMCAL;

        if ( detPID && detPID->HasNumberOfSigmas(detector)){
          return detPID->GetNumberOfSigmas(detector, type);
        } else if (fCachePID) {
          FillTrackDetectorPID(track, detector);
          detPID=track->GetDetectorPID();
          return detPID->GetNumberOfSigmas(detector, type);
        }

        // NSigma value really meaningful only for electrons!
        return fEMCALResponse.GetNumberOfSigmas(pt,EovP,type,charge);
      }
    }
  }
  return -999;
}

//______________________________________________________________________________
AliPIDResponse::EDetPidStatus AliPIDResponse::GetSignalDelta(EDetector detector, const AliVParticle *track, AliPID::EParticleType type, Double_t &val, Bool_t ratio/*=kFALSE*/) const
{
  //
  //
  //
  val=-9999.;
  switch (detector){
    case kITS:   return GetSignalDeltaITS(track,type,val,ratio); break;
    case kTPC:   return GetSignalDeltaTPC(track,type,val,ratio); break;
    case kTRD:   return GetSignalDeltaTRD(track,type,val,ratio); break;
    case kTOF:   return GetSignalDeltaTOF(track,type,val,ratio); break;
    case kHMPID: return GetSignalDeltaHMPID(track,type,val,ratio); break;
    default: return kDetNoSignal;
  }
  return kDetNoSignal;
}

//______________________________________________________________________________
Double_t AliPIDResponse::GetSignalDelta(EDetector detCode, const AliVParticle *track, AliPID::EParticleType type, Bool_t ratio/*=kFALSE*/) const
{
  //
  //
  //
  Double_t val=-9999.;
  EDetPidStatus stat=GetSignalDelta(detCode, track, type, val, ratio);
  if ( stat==kDetNoSignal ) val=-9999.;
  return val;
}

//______________________________________________________________________________
AliPIDResponse::EDetPidStatus AliPIDResponse::ComputePIDProbability  (EDetCode  detCode, const AliVTrack *track, Int_t nSpecies, Double_t p[]) const
{
  // Compute PID response of 'detCode'

  // find detector code from detector bit mask
  Int_t detector=-1;
  for (Int_t idet=0; idet<kNdetectors; ++idet) if ( (detCode&(1<<idet)) ) { detector=idet; break; }
  if (detector==-1) return kDetNoSignal;

  return ComputePIDProbability((EDetector)detector, track, nSpecies, p);
}

//______________________________________________________________________________
AliPIDResponse::EDetPidStatus AliPIDResponse::ComputePIDProbability  (EDetector detector,  const AliVTrack *track, Int_t nSpecies, Double_t p[]) const
{
  //
  // Compute PID response of 'detector'
  //

  const AliDetectorPID *detPID=track->GetDetectorPID();

  if ( detPID && detPID->HasRawProbability(detector)){
    return detPID->GetRawProbability(detector, p, nSpecies);
  } else if (fCachePID) {
    FillTrackDetectorPID(track, detector);
    detPID=track->GetDetectorPID();
    return detPID->GetRawProbability(detector, p, nSpecies);
  }

  //if no caching return values calculated from scratch
  return GetComputePIDProbability(detector, track, nSpecies, p);
}

//______________________________________________________________________________
AliPIDResponse::EDetPidStatus AliPIDResponse::ComputeITSProbability  (const AliVTrack *track, Int_t nSpecies, Double_t p[]) const
{
  // Compute PID response for the ITS
  return ComputePIDProbability(kITS, track, nSpecies, p);
}

//______________________________________________________________________________
AliPIDResponse::EDetPidStatus AliPIDResponse::ComputeTPCProbability  (const AliVTrack *track, Int_t nSpecies, Double_t p[]) const
{
  // Compute PID response for the TPC
  return ComputePIDProbability(kTPC, track, nSpecies, p);
}

//______________________________________________________________________________
AliPIDResponse::EDetPidStatus AliPIDResponse::ComputeTOFProbability  (const AliVTrack *track, Int_t nSpecies, Double_t p[]) const
{
  // Compute PID response for the
  return ComputePIDProbability(kTOF, track, nSpecies, p);
}

//______________________________________________________________________________
AliPIDResponse::EDetPidStatus AliPIDResponse::ComputeTRDProbability  (const AliVTrack *track, Int_t nSpecies, Double_t p[]) const
{
  // Compute PID response for the
  return ComputePIDProbability(kTRD, track, nSpecies, p);
}

//______________________________________________________________________________
AliPIDResponse::EDetPidStatus AliPIDResponse::ComputeEMCALProbability  (const AliVTrack *track, Int_t nSpecies, Double_t p[]) const
{
  // Compute PID response for the EMCAL
  return ComputePIDProbability(kEMCAL, track, nSpecies, p);
}
//______________________________________________________________________________
AliPIDResponse::EDetPidStatus AliPIDResponse::ComputePHOSProbability (const AliVTrack */*track*/, Int_t nSpecies, Double_t p[]) const
{
  // Compute PID response for the PHOS

  // set flat distribution (no decision)
  for (Int_t j=0; j<nSpecies; j++) p[j]=1./nSpecies;
  return kDetNoSignal;
}

//______________________________________________________________________________
AliPIDResponse::EDetPidStatus AliPIDResponse::ComputeHMPIDProbability(const AliVTrack *track, Int_t nSpecies, Double_t p[]) const
{
  // Compute PID response for the HMPID
  return ComputePIDProbability(kHMPID, track, nSpecies, p);
}

//______________________________________________________________________________
AliPIDResponse::EDetPidStatus AliPIDResponse::ComputeTRDProbability  (const AliVTrack *track, Int_t nSpecies, Double_t p[],AliTRDPIDResponse::ETRDPIDMethod PIDmethod) const
{
  // Compute PID response for the
  return GetComputeTRDProbability(track, nSpecies, p, PIDmethod);
}

//______________________________________________________________________________
AliPIDResponse::EDetPidStatus AliPIDResponse::CheckPIDStatus(EDetector detector, const AliVTrack *track) const
{
  // calculate detector pid status

  const Int_t iDetCode=(Int_t)detector;
  if (iDetCode<0||iDetCode>=kNdetectors) return kDetNoSignal;
  const AliDetectorPID *detPID=track->GetDetectorPID();

  if ( detPID ){
    return detPID->GetPIDStatus(detector);
  } else if (fCachePID) {
    FillTrackDetectorPID(track, detector);
    detPID=track->GetDetectorPID();
    return detPID->GetPIDStatus(detector);
  }

  // if not buffered and no buffering is requested
  return GetPIDStatus(detector, track);
}

//______________________________________________________________________________
void AliPIDResponse::InitialiseEvent(AliVEvent *event, Int_t pass, TString recoPassName/*=""*/, Int_t run/*=-1*/)
{
  //
  // Apply settings for the current event
  //
  fRecoPass=pass;
  fRecoPassName=recoPassName;


  fCurrentEvent=NULL;
  if (!event) return;
  fCurrentEvent=event;
  if (run>0) fRun=run;
  else fRun=event->GetRunNumber();

  if (fRun!=fOldRun){
    ExecNewRun();
    fOldRun=fRun;
  }

  //TPC resolution parametrisation PbPb
  if ( fResolutionCorrection ){
    Double_t corrSigma=fResolutionCorrection->Eval(GetTPCMultiplicityBin(event));
    fTPCResponse.SetSigma(3.79301e-03*corrSigma, 2.21280e+04);
  }

  // Set up TPC multiplicity for PbPb
  if (fUseTPCMultiplicityCorrection) {
    Int_t numESDtracks = event->GetNumberOfESDTracks();
    if (numESDtracks < 0) {
      AliError("Cannot obtain event multiplicity (number of ESD tracks < 0). If you are using AODs, this might be a too old production. Please disable the multiplicity correction to get a reliable PID result!");
      numESDtracks = 0;
    }
    fTPCResponse.SetCurrentEventMultiplicity(numESDtracks);
  }
  else
    fTPCResponse.SetCurrentEventMultiplicity(0);

  //TOF resolution
  SetTOFResponse(event, (AliPIDResponse::EStartTimeType_t)fTOFPIDParams->GetStartTimeMethod());

  // Get and set centrality
  fCurrCentrality = -1;
  fCurrCentrality = AliMultSelectionBase::GetMultiplicityPercentileWithFallback(event,"V0M");

  // Set centrality percentile for EMCAL
  fEMCALResponse.SetCentrality(fCurrCentrality);
  // Set centrality percentile for TRD
  fTRDResponse.SetCentrality(fCurrCentrality);

  // switch off some TOF channel according to OADB to match data TOF matching eff
  if (fTuneMConData && ((fTuneMConDataMask & kDetTOF) == kDetTOF) && fTOFPIDParams->GetTOFmatchingLossMC() > 0.01){
    Int_t ntrk = event->GetNumberOfTracks();
    for(Int_t i=0;i < ntrk;i++){
      AliVParticle *trk = event->GetTrack(i);
      Int_t channel = GetTOFResponse().GetTOFchannel(trk);
      Int_t swoffEachOfThem = Int_t(100./fTOFPIDParams->GetTOFmatchingLossMC() + 0.5);
      if(!(channel%swoffEachOfThem)) ((AliVTrack *) trk)->ResetStatus(AliVTrack::kTOFout);
    }
  }

}

//______________________________________________________________________________
void AliPIDResponse::ExecNewRun()
{
  //
  // Things to Execute upon a new run
  //
  SetRecoInfo();

  // ===| ITS part |============================================================
  SetITSParametrisation();

  // ===| TPC part |============================================================
  // new treatment for loading the TPC PID response if requested
  // for the moment fall back to old method if no PID response array is found
  // by the new method for backward compatibility
  Bool_t doOldTPCPID=kTRUE;
  if (fUseTPCNewResponse) {
    doOldTPCPID = !InitializeTPCResponse();
    if (doOldTPCPID) {
      AliWarning("No TPC response parametrisations found using the new method. Falling back to the old method.");
    }
  }

  if (doOldTPCPID) {
    SetTPCPidResponseMaster();
    SetTPCParametrisation();
  }
  SetTPCEtaMaps();

  // ===| TRD part |============================================================
  SetTRDPidResponseMaster();
  //has to precede InitializeTRDResponse(), otherwise the read-out fTRDdEdxParams is not pased in TRDResponse!
  CheckTRDLikelihoodParameter();
  SetTRDdEdxParams();
  SetTRDEtaMaps();
  SetTRDClusterMaps();
  SetTRDCentralityMaps();
  InitializeTRDResponse();

  // ===| TOF part |============================================================
  SetTOFPidResponseMaster();
  InitializeTOFResponse();

  // ===| EMCAL part |==========================================================
  SetEMCALPidResponseMaster();
  InitializeEMCALResponse();

  // ===| HMPID part |==========================================================
  SetHMPIDPidResponseMaster();
  InitializeHMPIDResponse();

  if (fCurrentEvent) fTPCResponse.SetMagField(fCurrentEvent->GetMagneticField());
  if (fCurrentEvent) fTRDResponse.SetMagField(fCurrentEvent->GetMagneticField());
}

//______________________________________________________________________________
Double_t AliPIDResponse::GetTPCMultiplicityBin(const AliVEvent * const event)
{
  //
  // Get TPC multiplicity in bins of 150
  //

  const AliVVertex* vertexTPC = event->GetPrimaryVertex();
  Double_t tpcMulti=0.;
  if(vertexTPC){
    Double_t vertexContribTPC=vertexTPC->GetNContributors();
    tpcMulti=vertexContribTPC/150.;
    if (tpcMulti>20.) tpcMulti=20.;
  }

  return tpcMulti;
}

//______________________________________________________________________________
void AliPIDResponse::SetRecoInfo()
{
  //
  // Set reconstruction information
  //

  //reset information
  fLHCperiod="";
  fMCperiodTPC="";

  fBeamType="";

  fBeamType="PP";
  fBeamTypeNum=kPP;

  Bool_t hasProdInfo=(fCurrentFile.BeginsWith("LHC"));

  TPRegexp reg(".*(LHC1[1-3][a-z]+[0-9]+[a-z_]*)[/_].*");
  if (hasProdInfo) reg=TPRegexp("LHC1[1-2][a-z]+[0-9]+[a-z_]*");
  TPRegexp reg12a17("LHC1[2-4][a-z]");

  //find the period by run number (UGLY, but not stored in ESD and AOD... )
  if (fRun>=114737&&fRun<=117223)      { fLHCperiod="LHC10B"; fMCperiodTPC="LHC10D1";  }
  else if (fRun>=118503&&fRun<=121040) { fLHCperiod="LHC10C"; fMCperiodTPC="LHC10D1";  }
  else if (fRun>=122195&&fRun<=126437) { fLHCperiod="LHC10D"; fMCperiodTPC="LHC10F6A"; }
  else if (fRun>=127710&&fRun<=130850) { fLHCperiod="LHC10E"; fMCperiodTPC="LHC10F6A"; }
  else if (fRun>=133004&&fRun<=135029) { fLHCperiod="LHC10F"; fMCperiodTPC="LHC10F6A"; }
  else if (fRun>=135654&&fRun<=136377) { fLHCperiod="LHC10G"; fMCperiodTPC="LHC10F6A"; }
  else if (fRun>=136851&&fRun<=139846) {
    fLHCperiod="LHC10H";
    fMCperiodTPC="LHC10H8";
    if (reg.MatchB(fCurrentFile)) fMCperiodTPC="LHC11A10";
    // exception for 13d2 and later
    if (fCurrentAliRootRev >= 62714) fMCperiodTPC="LHC13D2";
    fBeamType="PBPB";
    fBeamTypeNum=kPBPB;
  }
  else if (fRun>=139847&&fRun<=146974) { fLHCperiod="LHC11A"; fMCperiodTPC="LHC10F6A"; }
  //TODO: periods 11B (146975-150721), 11C (150722-155837) are not yet treated assume 11d for the moment
  else if (fRun>=146975&&fRun<=155837) { fLHCperiod="LHC11D"; fMCperiodTPC="LHC10F6A"; }
  else if (fRun>=155838&&fRun<=159649) { fLHCperiod="LHC11D"; fMCperiodTPC="LHC10F6A"; }
  // also for 11e (159650-162750),f(162751-165771) use 11d
  else if (fRun>=159650&&fRun<=162750) { fLHCperiod="LHC11D"; fMCperiodTPC="LHC10F6A"; }
  else if (fRun>=162751&&fRun<=165771) { fLHCperiod="LHC11D"; fMCperiodTPC="LHC10F6A"; }

  else if (fRun>=165772 && fRun<=170718) {
    fLHCperiod="LHC11H";
    fMCperiodTPC="LHC11A10";
    fBeamType="PBPB";
    fBeamTypeNum=kPBPB;
    if (reg12a17.MatchB(fCurrentFile)) fMCperiodTPC="LHC12A17";
  }
  if (fRun>=170719 && fRun<=177311) {
    fLHCperiod="LHC12A";
    fBeamType="PP";
    fBeamTypeNum=kPP;
    fMCperiodTPC="LHC10F6A";
    if (fCurrentAliRootRev >= 62714)
      fMCperiodTPC="LHC14E2";
  }
  // for the moment use LHC12b parameters up to LHC12d
  if (fRun>=177312 /*&& fRun<=179356*/) {
    fLHCperiod="LHC12B";
    fBeamType="PP";
    fBeamTypeNum=kPP;
    fMCperiodTPC="LHC10F6A";
    if (fCurrentAliRootRev >= 62714)
      fMCperiodTPC="LHC14E2";
  }
//   if (fRun>=179357 && fRun<=183173) { fLHCperiod="LHC12C"; fBeamType="PP"; fBeamTypeNum=kPP;/*fMCperiodTPC="";*/ }
//   if (fRun>=183174 && fRun<=186345) { fLHCperiod="LHC12D"; fBeamType="PP"; fBeamTypeNum=kPP;/*fMCperiodTPC="";*/ }
//   if (fRun>=186346 && fRun<=186635) { fLHCperiod="LHC12E"; fBeamType="PP"; fBeamTypeNum=kPP;/*fMCperiodTPC="";*/ }

//   if (fRun>=186636 && fRun<=188166) { fLHCperiod="LHC12F"; fBeamType="PP"; fBeamTypeNum=kPP;/*fMCperiodTPC="";*/ }
//   if (fRun >= 188167 && fRun <= 188355 ) { fLHCperiod="LHC12G"; fBeamType="PP"; fBeamTypeNum=kPP;/*fMCperiodTPC="";*/ }
//   if (fRun >= 188356 && fRun <= 188503 ) { fLHCperiod="LHC12G"; fBeamType="PPB"; fBeamTypeNum=kPPB;/*fMCperiodTPC="";*/ }
// for the moment use 12g parametrisation for all full gain runs (LHC12e+)

  // Dedicated splines for periods 12g and 12i(j) (and use more appropriate MC)
  if (fRun >= 188720 && fRun <= 192738) {
    fLHCperiod="LHC12H";
    fBeamType="PP";
    fBeamTypeNum=kPP;
    fMCperiodTPC="LHC10F6A";
    if (fCurrentAliRootRev >= 62714)
      fMCperiodTPC="LHC13B2_FIXn1";
  }
  if (fRun >= 192739 && fRun <= 194479) {
    fLHCperiod="LHC12I";
    fBeamType="PP";
    fBeamTypeNum=kPP;
    fMCperiodTPC="LHC10F6A";
    if (fCurrentAliRootRev >= 62714)
      fMCperiodTPC="LHC13B2_FIXn1";
  }

  // Use for pp and pPb 12E-G pass 1 the PPB runs
  if (fRecoPass==1 && fRun >= 186346 && fRun < 188719) { fLHCperiod="LHC12G"; fBeamType="PPB";fBeamTypeNum=kPPB; fMCperiodTPC="LHC12G"; }

  // settings for pass2 of 2012 data
  if (fRecoPass>=2 && fRun>=170719 && fRun<=194479) {
    fBeamType    = "PP";
    fBeamTypeNum = kPP;
    fMCperiodTPC = "";
    if (fRun >= 170719 && fRun <= 177311) { fLHCperiod="LHC12A"; }
    if (fRun >= 177312 && fRun <= 179356) { fLHCperiod="LHC12B"; }
    if (fRun >= 179357 && fRun <= 183173) { fLHCperiod="LHC12C"; }
    if (fRun >= 183174 && fRun <= 186345) { fLHCperiod="LHC12D"; }
    if (fRun >= 186346 && fRun <= 186635) { fLHCperiod="LHC12E"; }
    if (fRun >= 186636 && fRun <= 188166) { fLHCperiod="LHC12F"; }
    if (fRun >= 188167 && fRun <= 188719) { fLHCperiod="LHC12G"; }
    if (fRun >= 188720 && fRun <= 192738) { fLHCperiod="LHC12H"; }
    if (fRun >= 192739 && fRun <= 193766) { fLHCperiod="LHC12I"; }
    // no special parametrisations for 12J, use 12I instead
    if (fRun >= 193767 && fRun <= 194479) { /*fLHCperiod="LHC12J";*/ fLHCperiod="LHC12I"; }

    // overwriting for the PPB period
    if (fRun >= 188167 && fRun <= 188418) { fLHCperiod="LHC12G"; fBeamType="PPB";fBeamTypeNum=kPPB; fMCperiodTPC="LHC12G"; }
  }

  // New parametrisation for 2013 pPb runs
  if (fRun >= 194480) {
    fLHCperiod="LHC13B";
    fBeamType="PPB";
    fBeamTypeNum=kPPB;
    fMCperiodTPC="LHC12G";

    if (fCurrentAliRootRev >= 61605)
      fMCperiodTPC="LHC13B2_FIX";
    if (fCurrentAliRootRev >= 62714)
      fMCperiodTPC="LHC13B2_FIXn1";

    // High luminosity pPb runs require different parametrisations
    if (fRun >= 195875 && fRun <= 197411) {
      fLHCperiod="LHC13F";
    }
  }

  // New parametrisation for the first 2015 pp runs
  if (fRun >= 208505) { // <<< This is the first run in 15a
    fLHCperiod="LHC15F";
    fBeamType="PP";
    fBeamTypeNum=kPP;
    fMCperiodTPC="LHC15G3";
  }

  //exception new pp MC productions from 2011 (11a periods have 10f6a splines!)
  if (fBeamType=="PP" && reg.MatchB(fCurrentFile) && !fCurrentFile.Contains("LHC11a")) { fMCperiodTPC="LHC11B2"; fBeamType="PP";fBeamTypeNum=kPP; }
  // exception for 11f1
  if (fCurrentFile.Contains("LHC11f1")) fMCperiodTPC="LHC11F1";
  // exception for 12f1a, 12f1b and 12i3
  if (fCurrentFile.Contains("LHC12f1") || fCurrentFile.Contains("LHC12i3")) fMCperiodTPC="LHC12F1";
  // exception for 12c4
  if (fCurrentFile.Contains("LHC12c4")) fMCperiodTPC="LHC12C4";
  // exception for 13d1 11d anchored prod
  if (fLHCperiod=="LHC11D" && fCurrentFile.Contains("LHC13d1")) fMCperiodTPC="LHC13D1";
}

//______________________________________________________________________________
void AliPIDResponse::SetITSParametrisation()
{
  //
  // Set the ITS parametrisation
  //
}


//______________________________________________________________________________
void AliPIDResponse::AddPointToHyperplane(TH2D* h, TLinearFitter* linExtrapolation, Int_t binX, Int_t binY)
{
  if (h->GetBinContent(binX, binY) <= 1e-4)
    return; // Reject bins without content (within some numerical precision) or with strange content

  Double_t coord[2] = {0, 0};
  coord[0] = h->GetXaxis()->GetBinCenter(binX);
  coord[1] = h->GetYaxis()->GetBinCenter(binY);
  Double_t binError = h->GetBinError(binX, binY);
  if (binError <= 0) {
    binError = 1000; // Should not happen because bins without content are rejected for the map (TH2D* h)
    printf("ERROR: This should never happen: Trying to add bin in addPointToHyperplane with error not set....\n");
  }
  linExtrapolation->AddPoint(coord, h->GetBinContent(binX, binY, binError));
}


//______________________________________________________________________________
TH2D* AliPIDResponse::RefineHistoViaLinearInterpolation(TH2D* h, Double_t refineFactorX, Double_t refineFactorY)
{
  if (!h)
    return 0x0;

  // Interpolate to finer map
  TLinearFitter* linExtrapolation = new TLinearFitter(2, "hyp2", "");

  Double_t upperMapBoundY = h->GetYaxis()->GetBinUpEdge(h->GetYaxis()->GetNbins());
  Double_t lowerMapBoundY = h->GetYaxis()->GetBinLowEdge(1);
  Int_t nBinsX = 30;
  // Binning was find to yield good results, if 40 bins are chosen for the range 0.0016 to 0.02. For the new variable range,
  // scale the number of bins correspondingly
  Int_t nBinsY = TMath::Nint((upperMapBoundY - lowerMapBoundY) / (0.02 - 0.0016) * 40);
  Int_t nBinsXrefined = nBinsX * refineFactorX;
  Int_t nBinsYrefined = nBinsY * refineFactorY;

  TH2D* hRefined = new TH2D(Form("%s_refined", h->GetName()),  Form("%s (refined)", h->GetTitle()),
                            nBinsXrefined, h->GetXaxis()->GetBinLowEdge(1), h->GetXaxis()->GetBinUpEdge(h->GetXaxis()->GetNbins()),
                            nBinsYrefined, lowerMapBoundY, upperMapBoundY);

  for (Int_t binX = 1; binX <= nBinsXrefined; binX++)  {
    for (Int_t binY = 1; binY <= nBinsYrefined; binY++)  {

      hRefined->SetBinContent(binX, binY, 1); // Default value is 1

      Double_t centerX = hRefined->GetXaxis()->GetBinCenter(binX);
      Double_t centerY = hRefined->GetYaxis()->GetBinCenter(binY);

      /*OLD
      linExtrapolation->ClearPoints();

      // For interpolation: Just take the corresponding bin from the old histo.
      // For extrapolation: take the last available bin from the old histo.
      // If the boundaries are to be skipped, also skip the corresponding bins
      Int_t oldBinX = h->GetXaxis()->FindBin(centerX);
      if (oldBinX < 1)
        oldBinX = 1;
      if (oldBinX > nBinsX)
        oldBinX = nBinsX;

      Int_t oldBinY = h->GetYaxis()->FindBin(centerY);
      if (oldBinY < 1)
        oldBinY = 1;
      if (oldBinY > nBinsY)
        oldBinY = nBinsY;

      // Neighbours left column
      if (oldBinX >= 2) {
        if (oldBinY >= 2) {
          AddPointToHyperplane(h, linExtrapolation, oldBinX - 1, oldBinY - 1);
        }

        AddPointToHyperplane(h, linExtrapolation, oldBinX - 1, oldBinY);

        if (oldBinY < nBinsY) {
          AddPointToHyperplane(h, linExtrapolation, oldBinX - 1, oldBinY + 1);
        }
      }

      // Neighbours (and point itself) same column
      if (oldBinY >= 2) {
        AddPointToHyperplane(h, linExtrapolation, oldBinX, oldBinY - 1);
      }

      AddPointToHyperplane(h, linExtrapolation, oldBinX, oldBinY);

      if (oldBinY < nBinsY) {
        AddPointToHyperplane(h, linExtrapolation, oldBinX, oldBinY + 1);
      }

      // Neighbours right column
      if (oldBinX < nBinsX) {
        if (oldBinY >= 2) {
          AddPointToHyperplane(h, linExtrapolation, oldBinX + 1, oldBinY - 1);
        }

        AddPointToHyperplane(h, linExtrapolation, oldBinX + 1, oldBinY);

        if (oldBinY < nBinsY) {
          AddPointToHyperplane(h, linExtrapolation, oldBinX + 1, oldBinY + 1);
        }
      }


      // Fit 2D-hyperplane
      if (linExtrapolation->GetNpoints() <= 0)
        continue;

      if (linExtrapolation->Eval() != 0)// EvalRobust -> Takes much, much, [...], much more time (~hours instead of seconds)
        continue;

      // Fill the bin of the refined histogram with the extrapolated value
      Double_t interpolatedValue = linExtrapolation->GetParameter(0) + linExtrapolation->GetParameter(1) * centerX
                                 + linExtrapolation->GetParameter(2) * centerY;
      */
      Double_t interpolatedValue = h->Interpolate(centerX, centerY) ;
      hRefined->SetBinContent(binX, binY, interpolatedValue);
    }
  }


  // Problem: Interpolation does not work before/beyond center of first/last bin (as the name suggests).
  // Therefore, for each row in dEdx: Take last bin from old map and interpolate values from center and edge.
  // Assume line through these points and extropolate to last bin of refined map
  const Double_t firstOldXbinUpEdge = h->GetXaxis()->GetBinUpEdge(1);
  const Double_t firstOldXbinCenter = h->GetXaxis()->GetBinCenter(1);

  const Double_t oldXbinHalfWidth = firstOldXbinUpEdge - firstOldXbinCenter;

  const Double_t lastOldXbinLowEdge = h->GetXaxis()->GetBinLowEdge(h->GetNbinsX());
  const Double_t lastOldXbinCenter = h->GetXaxis()->GetBinCenter(h->GetNbinsX());

  for (Int_t binY = 1; binY <= nBinsYrefined; binY++)  {
    Double_t centerY = hRefined->GetYaxis()->GetBinCenter(binY);

    const Double_t interpolatedCenterFirstXbin = h->Interpolate(firstOldXbinCenter, centerY);
    const Double_t interpolatedUpEdgeFirstXbin = h->Interpolate(firstOldXbinUpEdge, centerY);

    const Double_t extrapolationSlopeFirstXbin = (interpolatedUpEdgeFirstXbin - interpolatedCenterFirstXbin) / oldXbinHalfWidth;
    const Double_t extrapolationOffsetFirstXbin = interpolatedCenterFirstXbin;


    const Double_t interpolatedCenterLastXbin = h->Interpolate(lastOldXbinCenter, centerY);
    const Double_t interpolatedLowEdgeLastXbin = h->Interpolate(lastOldXbinLowEdge, centerY);

    const Double_t extrapolationSlopeLastXbin = (interpolatedCenterLastXbin - interpolatedLowEdgeLastXbin) / oldXbinHalfWidth;
    const Double_t extrapolationOffsetLastXbin = interpolatedCenterLastXbin;

    for (Int_t binX = 1; binX <= nBinsXrefined; binX++)  {
      Double_t centerX = hRefined->GetXaxis()->GetBinCenter(binX);

      if (centerX < firstOldXbinCenter) {
        Double_t extrapolatedValue = extrapolationOffsetFirstXbin + (centerX - firstOldXbinCenter) * extrapolationSlopeFirstXbin;
        hRefined->SetBinContent(binX, binY, extrapolatedValue);
      }
      else if (centerX <= lastOldXbinCenter) {
        continue;
      }
      else {
        Double_t extrapolatedValue = extrapolationOffsetLastXbin + (centerX - lastOldXbinCenter) * extrapolationSlopeLastXbin;
        hRefined->SetBinContent(binX, binY, extrapolatedValue);
      }
    }
  }

  delete linExtrapolation;

  return hRefined;
}

//______________________________________________________________________________
void AliPIDResponse::SetTPCEtaMaps(Double_t refineFactorMapX, Double_t refineFactorMapY,
                                   Double_t refineFactorSigmaMapX, Double_t refineFactorSigmaMapY)
{
  //
  // Load the TPC eta correction maps from the OADB
  //

  if (fUseTPCEtaCorrection == kFALSE) {
    // Disable eta correction via setting no maps
    if (!fTPCResponse.SetEtaCorrMap(0x0))
      AliInfo("Request to disable TPC eta correction -> Eta correction has been disabled");
    else
      AliError("Request to disable TPC eta correction -> Some error occured when unloading the correction maps");

    if (!fTPCResponse.SetSigmaParams(0x0, 0))
      AliInfo("Request to disable TPC eta correction -> Using old parametrisation for sigma");
    else
      AliError("Request to disable TPC eta correction -> Some error occured when unloading the sigma maps");

    return;
  }

  TString dataType = "DATA";
  TString period = fLHCperiod.IsNull() ? "No period information" : fLHCperiod;

  if (fIsMC)  {
    if (!(fTuneMConData && ((fTuneMConDataMask & kDetTPC) == kDetTPC))) {
      period=fMCperiodTPC;
      dataType="MC";
    }
    fRecoPass = 1;

    if (!(fTuneMConData && ((fTuneMConDataMask & kDetTPC) == kDetTPC)) && fMCperiodTPC.IsNull()) {
      AliError("***** Risk for unreliable TPC PID detected:                      ********");
      AliError("      MC detected, but no MC period set -> Not changing eta maps!");
      return;
    }
  }

  Int_t recopass       = fRecoPass;
  TString recoPassName = fRecoPassName;
  if (fTuneMConData && ((fTuneMConDataMask & kDetTPC) == kDetTPC) ) {
    recopass     = fRecoPassUser;
    recoPassName = fRecoPassNameUser;
  }

  if (fTPCResponse.GetRecoPassNameUsed().IsDigit()){
    Int_t recoPassUsedForSplines=fTPCResponse.GetRecoPassNameUsed().Atoi();
    if (recoPassUsedForSplines<recopass) {
      AliInfoF("Reco pass used for splines (%d) differs from the requested reco pass (%d), the splines one will be used to match the eta maps",
               recoPassUsedForSplines, recopass);
      recopass=recoPassUsedForSplines;
    }
  }

  TString defaultObj = Form("Default_%s_pass%d", dataType.Data(), recopass);

  AliInfo(Form("Current period and reco pass: %s.pass%d (%s)", period.Data(), recopass, recoPassName.Data()));

  // Invalidate old maps
  fTPCResponse.SetEtaCorrMap(0x0);
  fTPCResponse.SetSigmaParams(0x0, 0);


  TString fileNameMaps(Form("%s/COMMON/PID/data/TPCetaMaps.root", fOADBPath.Data()));
  if (!fCustomTPCetaMaps.IsNull()) fileNameMaps=fCustomTPCetaMaps;

  // ===| Load the eta correction maps |=======================================
  //
  TString contNameNumeric=TString::Format("TPCetaMaps_%s_pass%d", dataType.Data(), recopass);
  TString contNameString =TString::Format("TPCetaMaps_%s_%s", dataType.Data(), recoPassName.Data());
  TString contName;
  AliOADBContainer etaMapsCont(Form("TPCetaMaps_%s_pass%d", dataType.Data(), recopass));

  // ---| try loading for specific pass name |----------------------------------
  AliInfoF("Trying to load map container for specific pass name %s.", recoPassName.Data());
  Int_t statusCont = etaMapsCont.InitFromFile(fileNameMaps.Data(), contNameString);

  if (statusCont) {
    // ---| fall back to numerical pass |---------------------------------------
    AliInfoF("No dedicated map container found for '%s', check numerical pass %d.", recoPassName.Data(), recopass);
    statusCont = etaMapsCont.InitFromFile(fileNameMaps.Data(), contNameNumeric);
    contName=contNameNumeric;
  }
  else{
    AliInfoF("Dedicated map container for '%s' found.", recoPassName.Data());
    contName=contNameString;
  }

  if (statusCont) {
    AliError("Failed initializing TPC eta correction maps from OADB -> Disabled eta correction");
    fUseTPCEtaCorrection = kFALSE;
  }
  else {
    AliInfo(Form("Loading TPC eta correction map from %s (%s)", fileNameMaps.Data(), contName.Data()));

    TH2D* etaMap = 0x0;

    if (fIsMC && !(fTuneMConData && ((fTuneMConDataMask & kDetTPC) == kDetTPC))) {
      TString searchMap = Form("TPCetaMaps_%s_%s_pass%d", dataType.Data(), period.Data(), recopass);
      etaMap = dynamic_cast<TH2D *>(etaMapsCont.GetDefaultObject(searchMap.Data()));
      if (!etaMap) {
        // Try default object
        etaMap = dynamic_cast<TH2D *>(etaMapsCont.GetDefaultObject(defaultObj.Data()));
      }
    }
    else {
      etaMap = dynamic_cast<TH2D *>(etaMapsCont.GetObject(fRun, defaultObj.Data()));
    }


    if (!etaMap) {
      AliError(Form("TPC eta correction map not found for run %d and also no default map found -> Disabled eta correction!!!", fRun));
      fUseTPCEtaCorrection = kFALSE;
    }
    else {
      TH2D* etaMapRefined = RefineHistoViaLinearInterpolation(etaMap, refineFactorMapX, refineFactorMapY);

      if (etaMapRefined) {
        if (!fTPCResponse.SetEtaCorrMap(etaMapRefined)) {
          AliError(Form("Failed to set TPC eta correction map for run %d -> Disabled eta correction!!!", fRun));
          fTPCResponse.SetEtaCorrMap(0x0);
          fUseTPCEtaCorrection = kFALSE;
        }
        else {
          AliInfo(Form("Loaded TPC eta correction map (refine factors %.2f/%.2f) from %s: %s (MD5(map) = %s)",
                       refineFactorMapX, refineFactorMapY, fileNameMaps.Data(), fTPCResponse.GetEtaCorrMap()->GetTitle(),
                       AliTPCPIDResponse::GetChecksum(fTPCResponse.GetEtaCorrMap()).Data()));
        }

        delete etaMapRefined;
      }
      else {
        AliError(Form("Failed to set TPC eta correction map for run %d (map was loaded, but couldn't be refined) -> Disabled eta correction!!!", fRun));
        fUseTPCEtaCorrection = kFALSE;
      }
    }
  }

  // If there was some problem loading the eta maps, it makes no sense to load the sigma maps (that require eta corrected data)
  if (fUseTPCEtaCorrection == kFALSE) {
    AliError("Failed to load TPC eta correction map required by sigma maps -> Using old parametrisation for sigma");
    return;
  }

  // ===| Load the sigma parametrisation (1/dEdx vs tanTheta_local (~eta)) |===
  //
  contNameNumeric=TString::Format("TPCetaSigmaMaps_%s_pass%d", dataType.Data(), recopass);
  contNameString =TString::Format("TPCetaSigmaMaps_%s_%s", dataType.Data(), recoPassName.Data());
  contName="";
  AliOADBContainer etaSigmaMapsCont(Form("TPCetaSigmaMaps_%s_pass%d", dataType.Data(), recopass));

  // ---| try loading for specific pass name |----------------------------------
  AliInfoF("Trying to load sigma map container for specific pass name %s.", recoPassName.Data());
  statusCont = etaSigmaMapsCont.InitFromFile(fileNameMaps.Data(), contNameString);

  if (statusCont) {
    // ---| fall back to numerical pass |---------------------------------------
    AliInfoF("No dedicated sigma map container found for '%s', check numerical pass %d.", recoPassName.Data(), recopass);
    statusCont = etaSigmaMapsCont.InitFromFile(fileNameMaps.Data(), contNameNumeric);
    contName=contNameNumeric;
  }
  else{
    AliInfoF("Dedicated sigma map container for '%s' found.", recoPassName.Data());
    contName=contNameString;
  }

  if (statusCont) {
    AliError("Failed initializing TPC eta sigma maps from OADB -> Using old sigma parametrisation");
  }
  else {
    AliInfo(Form("Loading TPC eta sigma map from %s (%s)", fileNameMaps.Data(), contName.Data()));

    TObjArray* etaSigmaPars = 0x0;

    if (fIsMC && !(fTuneMConData && ((fTuneMConDataMask & kDetTPC) == kDetTPC))) {
      TString searchMap = Form("TPCetaSigmaMaps_%s_%s_pass%d", dataType.Data(), period.Data(), recopass);
      etaSigmaPars = dynamic_cast<TObjArray *>(etaSigmaMapsCont.GetDefaultObject(searchMap.Data()));
      if (!etaSigmaPars) {
        // Try default object
        etaSigmaPars = dynamic_cast<TObjArray *>(etaSigmaMapsCont.GetDefaultObject(defaultObj.Data()));
      }
    }
    else {
      etaSigmaPars = dynamic_cast<TObjArray *>(etaSigmaMapsCont.GetObject(fRun, defaultObj.Data()));
    }

    if (!etaSigmaPars) {
      AliError(Form("TPC eta sigma parametrisation not found for run %d -> Using old sigma parametrisation!!!", fRun));
    }
    else {
      TH2D* etaSigmaPar1Map = dynamic_cast<TH2D *>(etaSigmaPars->FindObject("sigmaPar1Map"));
      TNamed* sigmaPar0Info = dynamic_cast<TNamed *>(etaSigmaPars->FindObject("sigmaPar0"));
      Double_t sigmaPar0 = 0.0;

      if (sigmaPar0Info) {
        TString sigmaPar0String = sigmaPar0Info->GetTitle();
        sigmaPar0 = sigmaPar0String.Atof();
      }
      else {
        // Something is weired because the object for parameter 0 could not be loaded -> New sigma parametrisation can not be used!
        etaSigmaPar1Map = 0x0;
      }

      TH2D* etaSigmaPar1MapRefined = RefineHistoViaLinearInterpolation(etaSigmaPar1Map, refineFactorSigmaMapX, refineFactorSigmaMapY);


      if (etaSigmaPar1MapRefined) {
        if (!fTPCResponse.SetSigmaParams(etaSigmaPar1MapRefined, sigmaPar0)) {
          AliError(Form("Failed to set TPC eta sigma map for run %d -> Using old sigma parametrisation!!!", fRun));
          fTPCResponse.SetSigmaParams(0x0, 0);
        }
        else {
          AliInfo(Form("Loaded TPC sigma correction map (refine factors %.2f/%.2f) from %s: %s (MD5(map) = %s, sigmaPar0 = %f)",
                       refineFactorSigmaMapX, refineFactorSigmaMapY, fileNameMaps.Data(), fTPCResponse.GetSigmaPar1Map()->GetTitle(),
                       AliTPCPIDResponse::GetChecksum(fTPCResponse.GetSigmaPar1Map()).Data(), sigmaPar0));
        }

        delete etaSigmaPar1MapRefined;
      }
      else {
        AliError(Form("Failed to set TPC eta sigma map for run %d (map was loaded, but couldn't be refined) -> Using old sigma parametrisation!!!",
                      fRun));
      }
    }
  }
}


//______________________________________________________________________________
Bool_t AliPIDResponse::InitializeTPCResponse()
{
  // Load the Array with TPC PID response information
  // This is the new method which will completely replace the old one at some point

  //
  // Setup old resolution parametrisation
  // TODO: This should be moved to the initialisation and vanish completely at some point

  //default
  fTPCResponse.SetSigma(3.79301e-03, 2.21280e+04);

  if (fRun>=122195){ //LHC10d
    fTPCResponse.SetSigma(2.30176e-02, 5.60422e+02);
  }

  if (fRun>=170719){ // LHC12a
    fTPCResponse.SetSigma(2.95714e-03, 1.01953e+05);
  }

  if (fRun>=177312){ // LHC12b
    fTPCResponse.SetSigma(3.74633e-03, 7.11829e+04 );
  }

  if (fRun>=186346){ // LHC12e
    fTPCResponse.SetSigma(8.62022e-04, 9.08156e+05);
  }

  
  AliInfo("---------------------------- TPC Response Configuration (New) ----------------------------");
  // ===| load TPC response array from OADB |===================================
  TString fileNamePIDresponse(Form("%s/COMMON/PID/data/TPCPIDResponseOADB.root", fOADBPath.Data()));
  if (!fCustomTPCpidResponseOADBFile.IsNull()) fileNamePIDresponse=fCustomTPCpidResponseOADBFile;


  // ---| In case of MC and NO tune on data fall back to old method |-----------
  if (fIsMC) {
    if(!(fTuneMConData && ((fTuneMConDataMask & kDetTPC) == kDetTPC))) return kFALSE;
  }

  // ---| set reco pass |-------------------------------------------------------
  Int_t   recopass     = fRecoPass;
  TString recoPassName = fRecoPassName;
  if (fIsMC && fTuneMConData && ((fTuneMConDataMask & kDetTPC) == kDetTPC) ) {
    recopass     = fRecoPassUser;
    recoPassName = fRecoPassNameUser;
  }

  const Bool_t returnValue = fTPCResponse.InitFromOADB(fRun, recopass, recoPassName, fileNamePIDresponse, fUseTPCMultiplicityCorrection);
  AliInfo("------------------------------------------------------------------------------------------");

  return returnValue;
}

//______________________________________________________________________________
void AliPIDResponse::SetTPCPidResponseMaster()
{
  //
  // Load the TPC pid response functions from the OADB
  // Load the TPC voltage maps from OADB
  //
  //don't load twice for the moment
   if (fArrPidResponseMaster) return;


  //reset the PID response functions
  delete fArrPidResponseMaster;
  fArrPidResponseMaster=NULL;

  TFile *f=NULL;

  TString fileNamePIDresponse(Form("%s/COMMON/PID/data/TPCPIDResponse.root", fOADBPath.Data()));
  if (!fCustomTPCpidResponse.IsNull()) fileNamePIDresponse=fCustomTPCpidResponse;

  f=TFile::Open(fileNamePIDresponse.Data());
  if (f && f->IsOpen() && !f->IsZombie()){
    fArrPidResponseMaster=dynamic_cast<TObjArray*>(f->Get("TPCPIDResponse"));
  }
  delete f;

  TString fileNameVoltageMaps(Form("%s/COMMON/PID/data/TPCvoltageSettings.root", fOADBPath.Data()));
  f=TFile::Open(fileNameVoltageMaps.Data());
  if (f && f->IsOpen() && !f->IsZombie()){
    fOADBvoltageMaps=dynamic_cast<AliOADBContainer*>(f->Get("TPCvoltageSettings"));
  }
  delete f;

  if (!fArrPidResponseMaster){
    AliFatal(Form("Could not retrieve the TPC pid response from: %s",fileNamePIDresponse.Data()));
    return;
  }
  fArrPidResponseMaster->SetOwner();

  if (!fOADBvoltageMaps)
  {
    AliFatal(Form("Could not retrieve the TPC voltage maps from: %s",fileNameVoltageMaps.Data()));
  }
  fArrPidResponseMaster->SetOwner();
}

//______________________________________________________________________________
void AliPIDResponse::SetTPCParametrisation()
{
  //
  // Change BB parametrisation for current run
  //

  AliInfo("---------------------------- TPC Response Configuration (Old) ----------------------------");

  //
  //reset old splines
  //
  fTPCResponse.ResetSplines();

  if (fLHCperiod.IsNull()) {
    AliError("No period set, not changing parametrisation");
    AliInfo("------------------------------------------------------------------------------------------");
    return;
  }

  //
  // Set default parametrisations for data and MC
  //

  //data type
  TString datatype="DATA";
  //in case of mc fRecoPass is per default 1
  if (fIsMC) {
      if(!(fTuneMConData && ((fTuneMConDataMask & kDetTPC) == kDetTPC))) datatype="MC";
      fRecoPass=1;
  } else {
    if (fRecoPass<=0) {
      fTPCResponse.SetUseDatabase(kFALSE);
      AliError("******** Risk for unreliable TPC PID detected               **********");
      AliError("         no proper reco pass was set, no splines can be set");
      AliError("         an outdate Bethe Bloch parametrisation will be used");
      AliInfo("------------------------------------------------------------------------------------------");
      return;
    }
  }

  // period
  TString period=fLHCperiod;
  if (fIsMC && !(fTuneMConData && ((fTuneMConDataMask & kDetTPC) == kDetTPC))) period=fMCperiodTPC;

  Int_t recopass = fRecoPass;
  if(fTuneMConData && ((fTuneMConDataMask & kDetTPC) == kDetTPC)) recopass = fRecoPassUser;

  AliInfo(Form("Searching splines for: %s %s PASS%d %s",datatype.Data(),period.Data(),recopass,fBeamType.Data()));
  Bool_t found=kFALSE;
  //
  //set the new PID splines
  //
  if (fArrPidResponseMaster){
    //for MC don't use period information
    //if (fIsMC) period="[A-Z0-9]*";
    //for MC use MC period information
    //pattern for the default entry (valid for all particles)
    TPRegexp reg(Form("TSPLINE3_%s_([A-Z]*)_%s_PASS%d_%s_MEAN(_*)([A-Z1-9]*)",datatype.Data(),period.Data(),recopass,fBeamType.Data()));

    //find particle id and gain scenario
    for (Int_t igainScenario=0; igainScenario<AliTPCPIDResponse::fgkNumberOfGainScenarios; igainScenario++)
    {
      TObject *grAll=NULL;
      TString gainScenario = AliTPCPIDResponse::GainScenarioName(igainScenario);
      gainScenario.ToUpper();
      //loop over entries and filter them
      for (Int_t iresp=0; iresp<fArrPidResponseMaster->GetEntriesFast();++iresp)
      {
        TObject *responseFunction=fArrPidResponseMaster->At(iresp);
        if (responseFunction==NULL) continue;
        TString responseName=responseFunction->GetName();

        if (!reg.MatchB(responseName)) continue;

        TObjArray *arr=reg.MatchS(responseName); if (!arr) continue;
        TObject* tmp=NULL;
        tmp=arr->At(1); if (!tmp) continue;
        TString particleName=tmp->GetName();
        tmp=arr->At(3); if (!tmp) continue;
        TString gainScenarioName=tmp->GetName();
        delete arr;
        if (particleName.IsNull()) continue;
        if (!grAll && particleName=="ALL" && gainScenarioName==gainScenario) grAll=responseFunction;
        else
        {
          for (Int_t ispec=0; ispec<(AliTPCPIDResponse::fgkNumberOfParticleSpecies); ++ispec)
          {
            TString particle=AliPID::ParticleName(ispec);
            particle.ToUpper();
            //std::cout<<responseName<<" "<<particle<<" "<<particleName<<" "<<gainScenario<<" "<<gainScenarioName<<std::endl;
            if ( particle == particleName && gainScenario == gainScenarioName )
            {
              fTPCResponse.SetResponseFunction( responseFunction,
                                                (AliPID::EParticleType)ispec,
                                                (AliTPCPIDResponse::ETPCgainScenario)igainScenario );
              fTPCResponse.SetUseDatabase(kTRUE);
              AliInfo(Form("Adding graph: %d %d - %s (MD5(spline) = %s)",ispec,igainScenario,responseFunction->GetName(),
                           AliTPCPIDResponse::GetChecksum((TSpline3*)responseFunction).Data()));
              found=kTRUE;
              break;
            }
          }
        }
      }

      // Retrieve responsefunction for pions - will (if available) be used for muons if there are no dedicated muon splines.
      // For light nuclei, try to set the proton spline, if no dedicated splines are available.
      // In both cases: Use default splines, if no dedicated splines and no pion/proton splines are available.
      TObject* responseFunctionPion = fTPCResponse.GetResponseFunction( (AliPID::EParticleType)AliPID::kPion,
                                                                        (AliTPCPIDResponse::ETPCgainScenario)igainScenario);
      TObject* responseFunctionProton = fTPCResponse.GetResponseFunction( (AliPID::EParticleType)AliPID::kProton,
                                                                          (AliTPCPIDResponse::ETPCgainScenario)igainScenario);

      for (Int_t ispec=0; ispec<(AliTPCPIDResponse::fgkNumberOfParticleSpecies); ++ispec)
      {
        if (!fTPCResponse.GetResponseFunction( (AliPID::EParticleType)ispec,
          (AliTPCPIDResponse::ETPCgainScenario)igainScenario))
        {
          if (ispec == AliPID::kMuon) { // Muons
            if (responseFunctionPion) {
              fTPCResponse.SetResponseFunction( responseFunctionPion,
                                                (AliPID::EParticleType)ispec,
                                                (AliTPCPIDResponse::ETPCgainScenario)igainScenario );
              fTPCResponse.SetUseDatabase(kTRUE);
              AliInfo(Form("Adding graph: %d %d - %s (MD5(spline) = %s)",ispec,igainScenario,responseFunctionPion->GetName(),
                           AliTPCPIDResponse::GetChecksum((TSpline3*)responseFunctionPion).Data()));
              found=kTRUE;
            }
            else if (grAll) {
              fTPCResponse.SetResponseFunction( grAll,
                                                (AliPID::EParticleType)ispec,
                                                (AliTPCPIDResponse::ETPCgainScenario)igainScenario );
              fTPCResponse.SetUseDatabase(kTRUE);
              AliInfo(Form("Adding graph: %d %d - %s (MD5(spline) = %s)",ispec,igainScenario,grAll->GetName(),
                           AliTPCPIDResponse::GetChecksum((TSpline3*)grAll).Data()));
              found=kTRUE;
            }
            //else
            //  AliError(Form("No splines found for muons (also no pion splines and no default splines) for gain scenario %d!", igainScenario));
          }
          else if (ispec >= AliPID::kSPECIES) { // Light nuclei
            if (responseFunctionProton) {
              fTPCResponse.SetResponseFunction( responseFunctionProton,
                                                (AliPID::EParticleType)ispec,
                                                (AliTPCPIDResponse::ETPCgainScenario)igainScenario );
              fTPCResponse.SetUseDatabase(kTRUE);
              AliInfo(Form("Adding graph: %d %d - %s (MD5(spline) = %s)",ispec,igainScenario,responseFunctionProton->GetName(),
                           AliTPCPIDResponse::GetChecksum((TSpline3*)responseFunctionProton).Data()));
              found=kTRUE;
            }
            else if (grAll) {
              fTPCResponse.SetResponseFunction( grAll,
                                                (AliPID::EParticleType)ispec,
                                                (AliTPCPIDResponse::ETPCgainScenario)igainScenario );
              fTPCResponse.SetUseDatabase(kTRUE);
              AliInfo(Form("Adding graph: %d %d - %s (MD5(spline) = %s)",ispec,igainScenario,grAll->GetName(),
                           AliTPCPIDResponse::GetChecksum((TSpline3*)grAll).Data()));
              found=kTRUE;
            }
            //else
            //  AliError(Form("No splines found for species %d (also no proton splines and no default splines) for gain scenario %d!",
            //                ispec, igainScenario));
          }
        }
      }
    }
  }
  else AliInfo("no fArrPidResponseMaster");

  if (!found){
    AliError("***** Risk for unreliable TPC PID detected:                      ********");
    AliError(Form("No splines found for: %s %s PASS%d %s",datatype.Data(),period.Data(),recopass,fBeamType.Data()));
  }


  //
  // Setup multiplicity correction (only used for non-pp collisions)
  //

  const Bool_t isPP = (fBeamType.CompareTo("PP") == 0);

  // 2013 pPb data taking at low luminosity
  const Bool_t isPPb2013LowLuminosity = period.Contains("LHC13B") || period.Contains("LHC13C") || period.Contains("LHC13D");
  // PbPb 2010, period 10h.pass2
  //TODO Needs further development const Bool_t is10hpass2 = period.Contains("LHC10H") && recopass == 2;


  // In case of MC without(!) tune on data activated for the TPC, don't use the multiplicity correction for the moment
  Bool_t isMCandNotTPCtuneOnData = fIsMC && !(fTuneMConData && ((fTuneMConDataMask & kDetTPC) == kDetTPC));

  // If correction is available, but disabled (highly NOT recommended!), print warning
  if (!fUseTPCMultiplicityCorrection && !isPP && !isMCandNotTPCtuneOnData) {
    //TODO: Needs further development if (is10hpass2 || isPPb2013LowLuminosity) {
    if (isPPb2013LowLuminosity) {
      AliWarning("Mulitplicity correction disabled, but correction parameters for this period exist. It is highly recommended to use enable the correction. Otherwise the splines might be off!");
    }
  }

  if (fUseTPCMultiplicityCorrection && !isPP && !isMCandNotTPCtuneOnData) {
    AliInfo("Multiplicity correction enabled!");

    //TODO After testing, load parameters from outside
    /*TODO no correction for MC
    if (period.Contains("LHC11A10"))  {//LHC11A10A
      AliInfo("Using multiplicity correction parameters for 11a10!");
      fTPCResponse.SetParameterMultiplicityCorrection(0, 6.90133e-06);
      fTPCResponse.SetParameterMultiplicityCorrection(1, -1.22123e-03);
      fTPCResponse.SetParameterMultiplicityCorrection(2, 1.80220e-02);
      fTPCResponse.SetParameterMultiplicityCorrection(3, 0.1);
      fTPCResponse.SetParameterMultiplicityCorrection(4, 6.45306e-03);

      fTPCResponse.SetParameterMultiplicityCorrectionTanTheta(0, -2.85505e-07);
      fTPCResponse.SetParameterMultiplicityCorrectionTanTheta(1, -1.31911e-06);
      fTPCResponse.SetParameterMultiplicityCorrectionTanTheta(2, -0.5);

      fTPCResponse.SetParameterMultiplicitySigmaCorrection(0, -4.29665e-05);
      fTPCResponse.SetParameterMultiplicitySigmaCorrection(1, 1.37023e-02);
      fTPCResponse.SetParameterMultiplicitySigmaCorrection(2, -6.36337e-01);
      fTPCResponse.SetParameterMultiplicitySigmaCorrection(3, 1.13479e-02);
    }
    else*/ if (isPPb2013LowLuminosity)  {// 2013 pPb data taking at low luminosity
      AliInfo("Using multiplicity correction parameters for 13b.pass2 (at least also valid for 13{c,d} and pass 3)!");

      fTPCResponse.SetParameterMultiplicityCorrection(0, -5.906e-06);
      fTPCResponse.SetParameterMultiplicityCorrection(1, -5.064e-04);
      fTPCResponse.SetParameterMultiplicityCorrection(2, -3.521e-02);
      fTPCResponse.SetParameterMultiplicityCorrection(3, 2.469e-02);
      fTPCResponse.SetParameterMultiplicityCorrection(4, 0);

      fTPCResponse.SetParameterMultiplicityCorrectionTanTheta(0, -5.32e-06);
      fTPCResponse.SetParameterMultiplicityCorrectionTanTheta(1, 1.177e-05);
      fTPCResponse.SetParameterMultiplicityCorrectionTanTheta(2, -0.5);

      fTPCResponse.SetParameterMultiplicitySigmaCorrection(0, 0.);
      fTPCResponse.SetParameterMultiplicitySigmaCorrection(1, 0.);
      fTPCResponse.SetParameterMultiplicitySigmaCorrection(2, 0.);
      fTPCResponse.SetParameterMultiplicitySigmaCorrection(3, 0.);

      /* Not too bad, but far from perfect in the details
      fTPCResponse.SetParameterMultiplicityCorrection(0, -6.27187e-06);
      fTPCResponse.SetParameterMultiplicityCorrection(1, -4.60649e-04);
      fTPCResponse.SetParameterMultiplicityCorrection(2, -4.26450e-02);
      fTPCResponse.SetParameterMultiplicityCorrection(3, 2.40590e-02);
      fTPCResponse.SetParameterMultiplicityCorrection(4, 0);

      fTPCResponse.SetParameterMultiplicityCorrectionTanTheta(0, -5.338e-06);
      fTPCResponse.SetParameterMultiplicityCorrectionTanTheta(1, 1.220e-05);
      fTPCResponse.SetParameterMultiplicityCorrectionTanTheta(2, -0.5);

      fTPCResponse.SetParameterMultiplicitySigmaCorrection(0, 7.89237e-05);
      fTPCResponse.SetParameterMultiplicitySigmaCorrection(1, -1.30662e-02);
      fTPCResponse.SetParameterMultiplicitySigmaCorrection(2, 8.91548e-01);
      fTPCResponse.SetParameterMultiplicitySigmaCorrection(3, 1.47931e-02);
      */
    }
    /*TODO: Needs further development
    else if (is10hpass2) {
      AliInfo("Using multiplicity correction parameters for 10h.pass2!");
      fTPCResponse.SetParameterMultiplicityCorrection(0, 3.21636e-07);
      fTPCResponse.SetParameterMultiplicityCorrection(1, -6.65876e-04);
      fTPCResponse.SetParameterMultiplicityCorrection(2, 1.28786e-03);
      fTPCResponse.SetParameterMultiplicityCorrection(3, 1.47677e-02);
      fTPCResponse.SetParameterMultiplicityCorrection(4, 0);

      fTPCResponse.SetParameterMultiplicityCorrectionTanTheta(0, 7.23591e-08);
      fTPCResponse.SetParameterMultiplicityCorrectionTanTheta(1, 2.7469e-06);
      fTPCResponse.SetParameterMultiplicityCorrectionTanTheta(2, -0.5);

      fTPCResponse.SetParameterMultiplicitySigmaCorrection(0, -1.22590e-05);
      fTPCResponse.SetParameterMultiplicitySigmaCorrection(1, 6.88888e-03);
      fTPCResponse.SetParameterMultiplicitySigmaCorrection(2, -3.20788e-01);
      fTPCResponse.SetParameterMultiplicitySigmaCorrection(3, 1.07345e-02);
    }
    */
    else {
      AliError(Form("Multiplicity correction is enabled, but no multiplicity correction parameters have been found for period %s.pass%d -> Mulitplicity correction DISABLED!", period.Data(), recopass));
      fUseTPCMultiplicityCorrection = kFALSE;
      fTPCResponse.ResetMultiplicityCorrectionFunctions();
    }
  }
  else {
    // Just set parameters such that overall correction factor is 1, i.e. no correction.
    // This is just a reasonable choice for the parameters for safety reasons. Disabling
    // the multiplicity correction will anyhow skip the calculation of the corresponding
    // correction factor inside THIS class. Nevertheless, experts can access the TPCPIDResponse
    // directly and use it for calculations - which should still give valid results, even if
    // the multiplicity correction is explicitely enabled in such expert calls.

    TString reasonForDisabling = "requested by user";
    if (fUseTPCMultiplicityCorrection) {
      if (isPP)
        reasonForDisabling = "pp collisions";
      else
        reasonForDisabling = "MC w/o tune on data";
    }

    AliInfo(Form("Multiplicity correction %sdisabled (%s)!", fUseTPCMultiplicityCorrection ? "automatically " : "",
                 reasonForDisabling.Data()));

    fUseTPCMultiplicityCorrection = kFALSE;
    fTPCResponse.ResetMultiplicityCorrectionFunctions();
  }

  if (fUseTPCMultiplicityCorrection) {
    for (Int_t i = 0; i <= 4 + 1; i++) {
      AliInfo(Form("parMultCorr: %d, %e", i, fTPCResponse.GetMultiplicityCorrectionFunction()->GetParameter(i)));
    }
    for (Int_t j = 0; j <= 2 + 1; j++) {
      AliInfo(Form("parMultCorrTanTheta: %d, %e", j, fTPCResponse.GetMultiplicityCorrectionFunctionTanTheta()->GetParameter(j)));
    }
    for (Int_t j = 0; j <= 3 + 1; j++) {
      AliInfo(Form("parMultSigmaCorr: %d, %e", j, fTPCResponse.GetMultiplicitySigmaCorrectionFunction()->GetParameter(j)));
    }
  }

  //
  // Setup old resolution parametrisation
  //

  //default
  fTPCResponse.SetSigma(3.79301e-03, 2.21280e+04);

  if (fRun>=122195){ //LHC10d
    fTPCResponse.SetSigma(2.30176e-02, 5.60422e+02);
  }

  if (fRun>=170719){ // LHC12a
    fTPCResponse.SetSigma(2.95714e-03, 1.01953e+05);
  }

  if (fRun>=177312){ // LHC12b
    fTPCResponse.SetSigma(3.74633e-03, 7.11829e+04 );
  }

  if (fRun>=186346){ // LHC12e
    fTPCResponse.SetSigma(8.62022e-04, 9.08156e+05);
  }

  if (fArrPidResponseMaster)
  fResolutionCorrection=(TF1*)fArrPidResponseMaster->FindObject(Form("TF1_%s_ALL_%s_PASS%d_%s_SIGMA",datatype.Data(),period.Data(),recopass,fBeamType.Data()));

  if (fResolutionCorrection) AliInfo(Form("Setting multiplicity correction function: %s  (MD5(corr function) = %s)",
                                          fResolutionCorrection->GetName(), AliTPCPIDResponse::GetChecksum(fResolutionCorrection).Data()));

  //read in the voltage map
  TVectorF* gsm = 0x0;
  if (fOADBvoltageMaps) gsm=dynamic_cast<TVectorF*>(fOADBvoltageMaps->GetObject(fRun));
  if (gsm)
  {
    fTPCResponse.SetVoltageMap(*gsm);
    TString vals;
    AliInfo(Form("Reading the voltage map for run %d\n",fRun));
    vals="IROC A: "; for (Int_t i=0; i<18; i++){vals+=Form("%.2f ",(*gsm)[i]);}
    AliInfo(vals.Data());
    vals="IROC C: "; for (Int_t i=18; i<36; i++){vals+=Form("%.2f ",(*gsm)[i]);}
    AliInfo(vals.Data());
    vals="OROC A: "; for (Int_t i=36; i<54; i++){vals+=Form("%.2f ",(*gsm)[i]);}
    AliInfo(vals.Data());
    vals="OROC C: "; for (Int_t i=54; i<72; i++){vals+=Form("%.2f ",(*gsm)[i]);}
    AliInfo(vals.Data());
  }
  else AliInfo("no voltage map, ideal default assumed");
  AliInfo("------------------------------------------------------------------------------------------");
}

//______________________________________________________________________________
void AliPIDResponse::SetTRDPidResponseMaster()
{
  //
  // Load the TRD pid params and references from the OADB
  //
  if(fTRDPIDResponseObject) return;
  AliOADBContainer contParams("contParams");

  Int_t statusResponse = contParams.InitFromFile(Form("%s/COMMON/PID/data/TRDPIDResponse.root", fOADBPath.Data()), "AliTRDPIDResponseObject");
  if(statusResponse){
    AliError("Failed initializing PID Response Object from OADB");
  } else {
    AliInfo(Form("Loading TRD Response from %s/COMMON/PID/data/TRDPIDResponse.root", fOADBPath.Data()));
    fTRDPIDResponseObject = dynamic_cast<AliTRDPIDResponseObject *>(contParams.GetObject(fRun));
    if(!fTRDPIDResponseObject){
      AliError(Form("TRD Response not found in run %d", fRun));
    }
  }
}
void AliPIDResponse::CheckTRDLikelihoodParameter(){
    Int_t nTracklets=1;
    Double_t level=0.9;
    Double_t params[4];
    Int_t centrality=0;
    Int_t iCharge=1;
    if (fTRDPIDResponseObject){
        if(!fTRDPIDResponseObject->GetThresholdParameters(nTracklets, level, params,centrality,AliTRDPIDResponse::kLQ1D,iCharge)){
            AliInfo("No Params for TRD Likelihood Threshold Parameters Found for Charge Dependence");
            AliInfo("Using Parameters for both charges");
            if((iCharge!=AliPID::kNoCharge)&&(!fTRDPIDResponseObject->GetThresholdParameters(nTracklets, level, params,centrality,AliTRDPIDResponse::kLQ1D,AliPID::kNoCharge))){
                AliError("No Params TRD Likelihood Threshold Parameters Found!!");
            }
        }
        else {
            AliInfo(Form("TRD Likelihood Threshold Parameters for Run %d Found",fRun));
        }
    }
}

//______________________________________________________________________________
void AliPIDResponse::InitializeTRDResponse(){
  //
  // Set PID Params and references to the TRD PID response
  //
    fTRDResponse.SetPIDResponseObject(fTRDPIDResponseObject);
    fTRDResponse.SetdEdxParams(fTRDdEdxParams);
}

//______________________________________________________________________________
void AliPIDResponse::SetTRDSlices(UInt_t TRDslicesForPID[2],AliTRDPIDResponse::ETRDPIDMethod method) const{

    if(fLHCperiod.Contains("LHC10D") || fLHCperiod.Contains("LHC10E")){
	// backward compatibility for setting with 8 slices
	TRDslicesForPID[0] = 0;
	TRDslicesForPID[1] = 7;
    }
    else{
	if(method==AliTRDPIDResponse::kLQ1D){
	    TRDslicesForPID[0] = 0; // first Slice contains normalized dEdx
	    TRDslicesForPID[1] = 0;
	}
	if((method==AliTRDPIDResponse::kLQ2D)||(method==AliTRDPIDResponse::kLQ3D)||(method==AliTRDPIDResponse::kLQ7D)){
	    TRDslicesForPID[0] = 1;
	    TRDslicesForPID[1] = 7;
	}
    }
    AliDebug(1,Form("Slice Range set to %d - %d",TRDslicesForPID[0],TRDslicesForPID[1]));
}
//______________________________________________________________________________
void AliPIDResponse::SetTRDdEdxParams()
{
  if(fTRDdEdxParams) return;

  const TString containerName = "TRDdEdxParamsContainer";
  AliOADBContainer cont(containerName.Data());

  const TString filePathNamePackage=Form("%s/COMMON/PID/data/TRDdEdxParams.root", fOADBPath.Data());

  const Int_t statusCont = cont.InitFromFile(filePathNamePackage.Data(), cont.GetName());
  if (statusCont){
    AliFatal("Failed initializing settings from OADB");
  }
  else{
    AliInfo(Form("Loading %s from %s\n", cont.GetName(), filePathNamePackage.Data()));

    fTRDdEdxParams = (AliTRDdEdxParams*)(cont.GetObject(fRun, "default"));

    if(!fTRDdEdxParams){
      AliError(Form("TRD dEdx Params default not found"));
    }
  }
}

//______________________________________________________________________________
void AliPIDResponse::SetTRDEtaMaps()
{
  //
  // Load the TRD eta correction map from the OADB
  //

    if (fIsMC) fUseTRDEtaCorrection = kFALSE;
    if (fUseTRDEtaCorrection == kFALSE) {
      //  fTRDResponse.SetEtaCorrMap(0,0x0);
	AliInfo("Request to disable TRD eta correction -> Eta correction has been disabled");
        return;
    }
    TH2D* etaMap[1];
    etaMap[0] = 0x0;


    const TString containerName = "TRDCorrectionMaps";
    AliOADBContainer cont(containerName.Data());

    const TString filePathNamePackage=Form("%s/COMMON/PID/data/TRDdEdxCorrectionMaps.root", fOADBPath.Data());

    const Int_t statusCont = cont.InitFromFile(filePathNamePackage.Data(), cont.GetName());
    if (statusCont){
	AliFatal("Failed initializing TRD Eta Correction settings from OADB");
        return;
    }
    else{
	AliInfo(Form("Loading %s from %s\n", cont.GetName(), filePathNamePackage.Data()));

	TObject* etaarray=(TObject*)cont.GetObject(fRun);

	if(etaarray){
		etaMap[0] = (TH2D *)etaarray->FindObject("TRDEtaMap");
		fTRDResponse.SetEtaCorrMap(0,etaMap[0]);
	}
	else{
	    AliError(Form("TRD Eta Correction Params not found"));
	    fUseTRDEtaCorrection = kFALSE;
            return;
	    //fTRDResponse.SetEtaCorrMap(0,0x0);
	}



	if (!etaMap[0]) {
	    AliError(Form("TRD Eta Correction Params not found"));
	    fUseTRDEtaCorrection = kFALSE;
            return;
	    //fTRDResponse.SetEtaCorrMap(0,0x0);
	}


    }


}

//______________________________________________________________________________
void AliPIDResponse::SetTRDClusterMaps()
{
  //
  // Load the TRD cluster correction map from the OADB
  //

    if (fIsMC) fUseTRDClusterCorrection = kFALSE;
    if (fUseTRDClusterCorrection == kFALSE) {
      //  fTRDResponse.SetEtaCorrMap(0,0x0);
	AliInfo("Request to disable TRD cluster correction -> Cluster correction has been disabled");
        return;
    }
    TH2D* clusterMap[3];
    for(Int_t i=0; i<3;i++) clusterMap[i] = 0x0;
    Int_t offset =4;

    const TString containerName = "TRDCorrectionMaps";
    AliOADBContainer cont(containerName.Data());

    const TString filePathNamePackage=Form("%s/COMMON/PID/data/TRDdEdxCorrectionMaps.root", fOADBPath.Data());

    const Int_t statusCont = cont.InitFromFile(filePathNamePackage.Data(), cont.GetName());
    if (statusCont){
	AliFatal("Failed initializing TRD Cluster Correction settings from OADB");
        return;
    }
    else{
	AliInfo(Form("Loading %s from %s\n", cont.GetName(), filePathNamePackage.Data()));

	TObject* clusterarray=(TObject*)cont.GetObject(fRun);

	if(clusterarray){
	    for(Int_t i=0;i<3;i++){
		clusterMap[i] = (TH2D *)clusterarray->FindObject(Form("TRDNclsMap_Nch%i",i+offset));
		fTRDResponse.SetClusterCorrMap(i,clusterMap[i]);
	    }
	}
	else{
	    AliError(Form("TRD Cluster Correction Params not found"));
	    fUseTRDClusterCorrection = kFALSE;
            return;
	    //fTRDResponse.SetEtaCorrMap(0,0x0);
	}



	if (!clusterMap[0]) {
	    AliError(Form("TRD Cluster Correction Params not found"));
	    fUseTRDClusterCorrection = kFALSE;
            return;
	    //fTRDResponse.SetEtaCorrMap(0,0x0);
	}


    }


}

//______________________________________________________________________________
void AliPIDResponse::SetTRDCentralityMaps()
{
  //
  // Load the TRD centrality correction map from the OADB
  //

    if (fIsMC) fUseTRDCentralityCorrection = kFALSE;
    if (fUseTRDCentralityCorrection == kFALSE) {
	AliInfo("Request to disable TRD centrality correction -> Centrality correction has been disabled");
        return;
    }
    TH2D* centralityMap[1];
    centralityMap[0] = 0x0;


    const TString containerName = "TRDCorrectionMaps";
    AliOADBContainer cont(containerName.Data());

    const TString filePathNamePackage=Form("%s/COMMON/PID/data/TRDdEdxCorrectionMaps.root", fOADBPath.Data());

    const Int_t statusCont = cont.InitFromFile(filePathNamePackage.Data(), cont.GetName());
    if (statusCont){
	AliInfo("Failed initializing TRD Centrality Correction settings from OADB or no correction parameters available");
        return;
    }
    else{
	AliInfo(Form("Loading %s from %s\n", cont.GetName(), filePathNamePackage.Data()));

	TObject* centralityarray=(TObject*)cont.GetObject(fRun);

	if(centralityarray){
		centralityMap[0] = (TH2D *)centralityarray->FindObject("TRDCentralityMap");
		fTRDResponse.SetCentralityCorrMap(0,centralityMap[0]);
	}
	else{
	    AliInfo(Form("TRD Centrality Correction Params not found"));
	    fUseTRDCentralityCorrection = kFALSE;
            return;
	    //fTRDResponse.SetCentralityCorrMap(0,0x0);
	}



	if (!centralityMap[0]) {
	    AliInfo(Form("TRD Centrality Correction Params not found"));
	    fUseTRDCentralityCorrection = kFALSE;
            return;
	}


    } 


}


//______________________________________________________________________________
void AliPIDResponse::SetTOFPidResponseMaster()
{
  //
  // Load the TOF pid params from the OADB
  //

  if (fTOFPIDParams) delete fTOFPIDParams;
  fTOFPIDParams=NULL;

  TFile *oadbf = new TFile(Form("%s/COMMON/PID/data/TOFPIDParams.root",fOADBPath.Data()));
  if (oadbf && oadbf->IsOpen()) {
    Int_t recoPass = fRecoPass;
    if (fTuneMConData && ((fTuneMConDataMask & kDetTOF) == kDetTOF) ) recoPass = fRecoPassUser;
    TString passName = Form("pass%d",recoPass);
    AliInfo(Form("Loading TOF Params from %s/COMMON/PID/data/TOFPIDParams.root for run %d (pass: %s)", fOADBPath.Data(),fRun,passName.Data()));
    AliOADBContainer *oadbc = (AliOADBContainer *)oadbf->Get("TOFoadb");
    if (oadbc) fTOFPIDParams = dynamic_cast<AliTOFPIDParams *>(oadbc->GetObject(fRun,"TOFparams",passName));
    oadbf->Close();
    delete oadbc;
  } else {
    AliError(Form("TOF OADB file not found!!! %s/COMMON/PID/data/TOFPIDParams.root",fOADBPath.Data()));
  }
  delete oadbf;

  if (!fTOFPIDParams) AliFatal("TOFPIDParams could not be retrieved");

  if (TString(fTOFPIDParams->GetOADBentryTag()) == "default") {
    AliWarning("******* Risk for unreliable TOF PID detected *********");
    if (!fIsMC && fRecoPass<=0) {
      AliWarningF("        Invalid reco pass for data (%d) was detected", fRecoPass);
    }
    AliWarning("        The default object was loaded");
  }
}

//______________________________________________________________________________
void AliPIDResponse::InitializeTOFResponse()
{
  //
  // Set PID Params to the TOF PID response
  //
  AliInfo("---------------------------- TOF Response Configuration ----------------------------");
  AliInfo(Form("TOF PID Params loaded from OADB [entryTag: %s]",fTOFPIDParams->GetOADBentryTag()));
  AliInfo(Form("  TOF resolution %5.2f [ps]",fTOFPIDParams->GetTOFresolution()));
  AliInfo(Form("  StartTime method %d",fTOFPIDParams->GetStartTimeMethod()));
  AliInfo(Form("  TOF res. mom. params: %6.3f %6.3f %6.3f %6.3f",
               fTOFPIDParams->GetSigParams(0),fTOFPIDParams->GetSigParams(1),fTOFPIDParams->GetSigParams(2),fTOFPIDParams->GetSigParams(3)));
  AliInfo(Form("  Start Time Offset %6.2f ps",fTOFPIDParams->GetTOFtimeOffset()));
  AliInfo(Form("  Fraction of tracks within gaussian behaviour: %6.4f",fTOFPIDParams->GetTOFtail()));

  if (fIsMC) {
    AliInfo("MC data:");
    if (fTuneMConData && ((fTuneMConDataMask & kDetTOF) == kDetTOF) ) {
      AliInfo(Form("  MC: TuneOnData option enabled for TOF data (target data reco pass is %d)",fRecoPassUser));
    } else AliInfo("  MC: TuneOnData option NOT enabled for TOF data");
  }
  AliInfo(Form("  MC: Fraction of tracks (percentage) to cut to fit matching in data: %6.2f%%",fTOFPIDParams->GetTOFmatchingLossMC()));
  AliInfo(Form("  MC: Fraction of random hits (percentage) to add to fit mismatch in data: %6.2f%%",fTOFPIDParams->GetTOFadditionalMismForMC()));

  for (Int_t i=0;i<4;i++) {
    fTOFResponse.SetTrackParameter(i,fTOFPIDParams->GetSigParams(i));
  }
  fTOFResponse.SetTimeResolution(fTOFPIDParams->GetTOFresolution());

  AliInfo("TZERO resolution loaded from ESDrun/AODheader");
  Float_t t0Spread[4];
  for (Int_t i=0;i<4;i++) t0Spread[i]=fCurrentEvent->GetT0spread(i);
  AliInfo(Form("  TZERO spreads from data: (A+C)/2 %f A %f C %f (A'-C')/2: %f",t0Spread[0],t0Spread[1],t0Spread[2],t0Spread[3]));
  Float_t a = t0Spread[1]*t0Spread[1]-t0Spread[0]*t0Spread[0]+t0Spread[3]*t0Spread[3];
  Float_t c = t0Spread[2]*t0Spread[2]-t0Spread[0]*t0Spread[0]+t0Spread[3]*t0Spread[3];
  if ( (t0Spread[0] > 50. && t0Spread[0] < 400.) && (a > 0.) && (c>0.)) {
    fResT0AC=t0Spread[3];
    fResT0A=TMath::Sqrt(a);
    fResT0C=TMath::Sqrt(c);
  } else {
    AliInfo("  TZERO spreads not present or inconsistent, loading default");
    fResT0A=75.;
    fResT0C=65.;
    fResT0AC=55.;
  }
  AliInfo(Form("  TZERO resolution set to: T0A: %f [ps] T0C: %f [ps] T0AC %f [ps]",fResT0A,fResT0C,fResT0AC));
  AliInfo("------------------------------------------------------------------------------------");

}

//______________________________________________________________________________
void AliPIDResponse::SetHMPIDPidResponseMaster()
{
  //
  // Load the HMPID pid params from the OADB
  //

  if (fHMPIDPIDParams) delete fHMPIDPIDParams;
  fHMPIDPIDParams=NULL;

  TFile *oadbf;
  if(!fIsMC) oadbf = new TFile(Form("%s/COMMON/PID/data/HMPIDPIDParams.root",fOADBPath.Data()));
  else       oadbf = new TFile(Form("%s/COMMON/PID/MC/HMPIDPIDParams.root",fOADBPath.Data()));
  if (oadbf && oadbf->IsOpen()) {
    AliInfo(Form("Loading HMPID Params from %s/COMMON/PID/data/HMPIDPIDParams.root", fOADBPath.Data()));
    AliOADBContainer *oadbc = (AliOADBContainer *)oadbf->Get("HMPoadb");
    if (oadbc) fHMPIDPIDParams = dynamic_cast<AliHMPIDPIDParams *>(oadbc->GetObject(fRun,"HMPparams"));
    oadbf->Close();
    delete oadbc;
  }
  delete oadbf;

  if (!fHMPIDPIDParams) AliFatal("HMPIDPIDParams could not be retrieved");
}

//______________________________________________________________________________
void AliPIDResponse::InitializeHMPIDResponse(){
  //
  // Set PID Params to the HMPID PID response
  //

  fHMPIDResponse.SetRefIndexArray(fHMPIDPIDParams->GetHMPIDrefIndex());
}

//______________________________________________________________________________
Bool_t AliPIDResponse::IdentifiedAsElectronTRD(const AliVTrack *vtrack,Double_t efficiencyLevel,Double_t centrality,AliTRDPIDResponse::ETRDPIDMethod PIDmethod) const {
    // old function for compatibility
    Int_t ntracklets=0;
    return IdentifiedAsElectronTRD(vtrack,ntracklets,efficiencyLevel,centrality,PIDmethod);
}

//______________________________________________________________________________
Bool_t AliPIDResponse::IdentifiedAsElectronTRD(const AliVTrack *vtrack, Int_t &ntracklets,Double_t efficiencyLevel,Double_t centrality,AliTRDPIDResponse::ETRDPIDMethod PIDmethod) const {
  //
  // Check whether track is identified as electron under a given electron efficiency hypothesis
  //
  // ntracklets is the number of tracklets that has been used to calculate the PID signal

  Double_t probs[AliPID::kSPECIES];

  ntracklets =CalculateTRDResponse(vtrack,probs,PIDmethod);

  // Take mean of the TRD momenta in the given tracklets
  Float_t p = 0, trdmomenta[AliVTrack::kTRDnPlanes];
  Int_t nmomenta = 0;
  for(Int_t iPl=0;iPl<AliVTrack::kTRDnPlanes;iPl++){
    if(vtrack->GetTRDmomentum(iPl) > 0.){
      trdmomenta[nmomenta++] = vtrack->GetTRDmomentum(iPl);
    }
  }
  p = TMath::Mean(nmomenta, trdmomenta);
  return fTRDResponse.IdentifiedAsElectron(ntracklets, probs, p, efficiencyLevel,centrality,PIDmethod,vtrack);
}

//______________________________________________________________________________
void AliPIDResponse::SetEMCALPidResponseMaster()
{
  //
  // Load the EMCAL pid response functions from the OADB
  //
  TObjArray* fEMCALPIDParamsRun      = NULL;
  TObjArray* fEMCALPIDParamsPass     = NULL;

  if(fEMCALPIDParams) return;
  AliOADBContainer contParams("contParams");

  Int_t statusPars = contParams.InitFromFile(Form("%s/COMMON/PID/data/EMCALPIDParams.root", fOADBPath.Data()), "AliEMCALPIDParams");
  if(statusPars){
    AliError("Failed initializing PID Params from OADB");
  }
  else {
    AliInfo(Form("Loading EMCAL Params from %s/COMMON/PID/data/EMCALPIDParams.root", fOADBPath.Data()));

    fEMCALPIDParamsRun = dynamic_cast<TObjArray *>(contParams.GetObject(fRun));
    if(fEMCALPIDParamsRun)  fEMCALPIDParamsPass = dynamic_cast<TObjArray *>(fEMCALPIDParamsRun->FindObject(Form("pass%d",fRecoPass)));
    if(fEMCALPIDParamsPass) fEMCALPIDParams     = dynamic_cast<TObjArray *>(fEMCALPIDParamsPass->FindObject(Form("EMCALPIDParams_Particles")));

    if(!fEMCALPIDParams){
      AliInfo(Form("EMCAL Params not found in run %d pass %d", fRun, fRecoPass));
      AliInfo("Will take the standard LHC11d instead ...");

      fEMCALPIDParamsRun = dynamic_cast<TObjArray *>(contParams.GetObject(156477));
      if(fEMCALPIDParamsRun)  fEMCALPIDParamsPass = dynamic_cast<TObjArray *>(fEMCALPIDParamsRun->FindObject(Form("pass%d",1)));
      if(fEMCALPIDParamsPass) fEMCALPIDParams     = dynamic_cast<TObjArray *>(fEMCALPIDParamsPass->FindObject(Form("EMCALPIDParams_Particles")));

      if(!fEMCALPIDParams){
	AliError(Form("DEFAULT EMCAL Params (LHC11d) not found in file %s/COMMON/PID/data/EMCALPIDParams.root", fOADBPath.Data()));
      }
    }
  }
}

//______________________________________________________________________________
void AliPIDResponse::InitializeEMCALResponse(){
  //
  // Set PID Params to the EMCAL PID response
  //
  fEMCALResponse.SetPIDParams(fEMCALPIDParams);

}

//______________________________________________________________________________
void AliPIDResponse::FillTrackDetectorPID(const AliVTrack *track, EDetector detector) const
{
  //
  // create detector PID information and setup the transient pointer in the track
  //

  // check if detector number is inside accepted range
  if (detector == kNdetectors) return;

  // get detector pid
  AliDetectorPID *detPID=const_cast<AliDetectorPID*>(track->GetDetectorPID());
  if (!detPID) {
    detPID=new AliDetectorPID;
    (const_cast<AliVTrack*>(track))->SetDetectorPID(detPID);
  }

  //check if values exist
  if (detPID->HasRawProbability(detector) && detPID->HasNumberOfSigmas(detector)) return;

  //TODO: which particles to include? See also the loops below...
  Double_t values[AliPID::kSPECIESC]={0};

  //probabilities
  EDetPidStatus status=GetComputePIDProbability(detector,track,AliPID::kSPECIESC,values);
  detPID->SetRawProbability(detector, values, (Int_t)AliPID::kSPECIESC, status);

  //nsigmas
  for (Int_t ipart=0; ipart<AliPID::kSPECIESC; ++ipart)
    values[ipart]=GetNumberOfSigmas(detector,track,(AliPID::EParticleType)ipart);
  // the pid status is the same for probabilities and nSigmas, so it is
  // fine to use the one from the probabilities also here
  detPID->SetNumberOfSigmas(detector, values, (Int_t)AliPID::kSPECIESC, status);

}

//______________________________________________________________________________
void AliPIDResponse::FillTrackDetectorPID()
{
  //
  // create detector PID information and setup the transient pointer in the track
  //

  if (!fCurrentEvent) return;

  for (Int_t itrack=0; itrack<fCurrentEvent->GetNumberOfTracks(); ++itrack){
    AliVTrack *track=dynamic_cast<AliVTrack*>(fCurrentEvent->GetTrack(itrack));
    if (!track) continue;

    for (Int_t idet=0; idet<kNdetectors; ++idet){
      FillTrackDetectorPID(track, (EDetector)idet);
    }
  }
}

//______________________________________________________________________________
void AliPIDResponse::SetTOFResponse(AliVEvent *vevent,EStartTimeType_t option){
  //
  // Set TOF response function
  // Input option for event_time used
  //

    Float_t t0spread = 0.; //vevent->GetEventTimeSpread();
    if(t0spread < 10) t0spread = 80;

    // T0-FILL and T0-TO offset (because of TOF misallignment
    Float_t starttimeoffset = 0;
    if(fTOFPIDParams && !(fIsMC)) starttimeoffset=fTOFPIDParams->GetTOFtimeOffset();
    if(fTOFPIDParams){
      fTOFtail = fTOFPIDParams->GetTOFtail();
      GetTOFResponse().SetTOFtail(fTOFtail);
    }

    // T0 from TOF algorithm
    Bool_t flagT0TOF=kFALSE;
    Bool_t flagT0T0=kFALSE;
    Float_t *startTime = new Float_t[fTOFResponse.GetNmomBins()];
    Float_t *startTimeRes = new Float_t[fTOFResponse.GetNmomBins()];
    Int_t *startTimeMask = new Int_t[fTOFResponse.GetNmomBins()];

    // T0-TOF arrays
    Float_t *estimatedT0event = new Float_t[fTOFResponse.GetNmomBins()];
    Float_t *estimatedT0resolution = new Float_t[fTOFResponse.GetNmomBins()];
    for(Int_t i=0;i<fTOFResponse.GetNmomBins();i++){
      estimatedT0event[i]=0.0;
      estimatedT0resolution[i]=0.0;
      startTimeMask[i] = 0;
    }

    Float_t resT0A=fResT0A;
    Float_t resT0C=fResT0C;
    Float_t resT0AC=fResT0AC;
    if(vevent->GetT0TOF()){ // check if T0 detector information is available
	flagT0T0=kTRUE;
    }


    AliTOFHeader *tofHeader = (AliTOFHeader*)vevent->GetTOFHeader();

    if (tofHeader) { // read global info and T0-TOF
      //      fTOFResponse.SetTimeResolution(tofHeader->GetTOFResolution()); // read from OADB in the initialization
      t0spread = tofHeader->GetT0spread(); // read t0 sprad
      if(t0spread < 10) t0spread = 80;

      flagT0TOF=kTRUE;
      for(Int_t i=0;i<fTOFResponse.GetNmomBins();i++){ // read T0-TOF default value
	startTime[i]=tofHeader->GetDefaultEventTimeVal();
	startTimeRes[i]=tofHeader->GetDefaultEventTimeRes();
	if(startTimeRes[i] < 1.e-5) startTimeRes[i] = t0spread;

	if(startTimeRes[i] > t0spread - 10 && TMath::Abs(startTime[i]) < 0.001) startTime[i] = -starttimeoffset; // apply offset for T0-fill
      }

      TArrayI *ibin=(TArrayI*)tofHeader->GetNvalues();
      TArrayF *t0Bin=(TArrayF*)tofHeader->GetEventTimeValues();
      TArrayF *t0ResBin=(TArrayF*)tofHeader->GetEventTimeRes();
      for(Int_t j=0;j < tofHeader->GetNbins();j++){ // fill T0-TOF in p-bins
	Int_t icurrent = (Int_t)ibin->GetAt(j);
	startTime[icurrent]=t0Bin->GetAt(j);
	startTimeRes[icurrent]=t0ResBin->GetAt(j);
	if(startTimeRes[icurrent] < 1.e-5) startTimeRes[icurrent] = t0spread;
	if(startTimeRes[icurrent] > t0spread - 10 && TMath::Abs(startTime[icurrent]) < 0.001) startTime[icurrent] = -starttimeoffset; // apply offset for T0-fill
      }
    }

    // for cut of 3 sigma on t0 spread
    Float_t t0cut = 3 * t0spread;
    if(t0cut < 500) t0cut = 500;

    if(option == kFILL_T0){ // T0-FILL is used
	for(Int_t i=0;i<fTOFResponse.GetNmomBins();i++){
	  estimatedT0event[i]=0.0-starttimeoffset;
	  estimatedT0resolution[i]=t0spread;
	}
	fTOFResponse.SetT0event(estimatedT0event);
	fTOFResponse.SetT0resolution(estimatedT0resolution);
    }

    if(option == kTOF_T0){ // T0-TOF is used when available (T0-FILL otherwise) from ESD
	if(flagT0TOF){
	    fTOFResponse.SetT0event(startTime);
	    fTOFResponse.SetT0resolution(startTimeRes);
	    for(Int_t i=0;i<fTOFResponse.GetNmomBins();i++){
	      if(startTimeRes[i]<t0spread) startTimeMask[i]=1;
	      fTOFResponse.SetT0binMask(i,startTimeMask[i]);
	    }
	}
	else{
	    for(Int_t i=0;i<fTOFResponse.GetNmomBins();i++){
	      estimatedT0event[i]=0.0-starttimeoffset;
	      estimatedT0resolution[i]=t0spread;
	      fTOFResponse.SetT0binMask(i,startTimeMask[i]);
	    }
	    fTOFResponse.SetT0event(estimatedT0event);
	    fTOFResponse.SetT0resolution(estimatedT0resolution);
	}
    }
    else if(option == kBest_T0){ // T0-T0 or T0-TOF are used when available (T0-FILL otherwise) from ESD
	Float_t t0AC=-10000;
	Float_t t0A=-10000;
	Float_t t0C=-10000;
	if(flagT0T0){
	    t0A= vevent->GetT0TOF()[1] - starttimeoffset;
	    t0C= vevent->GetT0TOF()[2] - starttimeoffset;
	    t0AC= vevent->GetT0TOF()[0] - starttimeoffset;
	    //t0AC= t0A/resT0A/resT0A + t0C/resT0C/resT0C;
	    //    resT0AC= 1./TMath::Sqrt(1./resT0A/resT0A + 1./resT0C/resT0C);
	    //    t0AC *= resT0AC*resT0AC;
	}

	Float_t t0t0Best = 0;
	Float_t t0t0BestRes = 9999;
	Int_t t0used=0;
	if(TMath::Abs(t0A) < t0cut && TMath::Abs(t0C) < t0cut && TMath::Abs(t0C-t0A) < 500){
	    t0t0Best = t0AC;
	    t0t0BestRes = resT0AC;
	    t0used=6;
	}
	else if(TMath::Abs(t0C) < t0cut){
	    t0t0Best = t0C;
	    t0t0BestRes = resT0C;
	    t0used=4;
	}
	else if(TMath::Abs(t0A) < t0cut){
	    t0t0Best = t0A;
	    t0t0BestRes = resT0A;
	    t0used=2;
	}

	if(flagT0TOF){ // if T0-TOF info is available
	    for(Int_t i=0;i<fTOFResponse.GetNmomBins();i++){
		if(t0t0BestRes < 999){
		  if(startTimeRes[i] < t0spread){
		    Double_t wtot = 1./startTimeRes[i]/startTimeRes[i] + 1./t0t0BestRes/t0t0BestRes;
		    Double_t t0best = startTime[i]/startTimeRes[i]/startTimeRes[i] + t0t0Best/t0t0BestRes/t0t0BestRes;
		    estimatedT0event[i]=t0best / wtot;
		    estimatedT0resolution[i]=1./TMath::Sqrt(wtot);
		    startTimeMask[i] = t0used+1;
		  }
		  else {
		    estimatedT0event[i]=t0t0Best;
		    estimatedT0resolution[i]=t0t0BestRes;
		    startTimeMask[i] = t0used;
		  }
		}
		else{
		  estimatedT0event[i]=startTime[i];
		  estimatedT0resolution[i]=startTimeRes[i];
		  if(startTimeRes[i]<t0spread) startTimeMask[i]=1;
		}
		fTOFResponse.SetT0binMask(i,startTimeMask[i]);
	    }
	    fTOFResponse.SetT0event(estimatedT0event);
	    fTOFResponse.SetT0resolution(estimatedT0resolution);
	}
	else{ // if no T0-TOF info is available
	    for(Int_t i=0;i<fTOFResponse.GetNmomBins();i++){
	      fTOFResponse.SetT0binMask(i,t0used);
	      if(t0t0BestRes < 999){
		estimatedT0event[i]=t0t0Best;
		estimatedT0resolution[i]=t0t0BestRes;
	      }
	      else{
		estimatedT0event[i]=0.0-starttimeoffset;
		estimatedT0resolution[i]=t0spread;
	      }
	    }
	    fTOFResponse.SetT0event(estimatedT0event);
	    fTOFResponse.SetT0resolution(estimatedT0resolution);
	}
    }

    else if(option == kT0_T0){ // T0-T0 is used when available (T0-FILL otherwise)
	Float_t t0AC=-10000;
	Float_t t0A=-10000;
	Float_t t0C=-10000;
	if(flagT0T0){
	    t0A= vevent->GetT0TOF()[1] - starttimeoffset;
	    t0C= vevent->GetT0TOF()[2] - starttimeoffset;
	    t0AC= vevent->GetT0TOF()[0] - starttimeoffset;
	    //    t0AC= t0A/resT0A/resT0A + t0C/resT0C/resT0C;
	    //    resT0AC= 1./TMath::Sqrt(1./resT0A/resT0A + 1./resT0C/resT0C);
	    //    t0AC *= resT0AC*resT0AC;
	}

	if(TMath::Abs(t0A) < t0cut && TMath::Abs(t0C) < t0cut && TMath::Abs(t0C-t0A) < 500){
	    for(Int_t i=0;i<fTOFResponse.GetNmomBins();i++){
	      estimatedT0event[i]=t0AC;
	      estimatedT0resolution[i]=resT0AC;
	      fTOFResponse.SetT0binMask(i,6);
	    }
	}
	else if(TMath::Abs(t0C) < t0cut){
	    for(Int_t i=0;i<fTOFResponse.GetNmomBins();i++){
	      estimatedT0event[i]=t0C;
	      estimatedT0resolution[i]=resT0C;
	      fTOFResponse.SetT0binMask(i,4);
	    }
	}
	else if(TMath::Abs(t0A) < t0cut){
	    for(Int_t i=0;i<fTOFResponse.GetNmomBins();i++){
	      estimatedT0event[i]=t0A;
	      estimatedT0resolution[i]=resT0A;
	      fTOFResponse.SetT0binMask(i,2);
	    }
	}
	else{
	    for(Int_t i=0;i<fTOFResponse.GetNmomBins();i++){
	      estimatedT0event[i]= 0.0 - starttimeoffset;
	      estimatedT0resolution[i]=t0spread;
	      fTOFResponse.SetT0binMask(i,0);
	    }
	}
	fTOFResponse.SetT0event(estimatedT0event);
	fTOFResponse.SetT0resolution(estimatedT0resolution);
    }

    delete [] startTime;
    delete [] startTimeRes;
    delete [] startTimeMask;
    delete [] estimatedT0event;
    delete [] estimatedT0resolution;
}

//______________________________________________________________________________
// private non cached versions of the PID calculation
//


//______________________________________________________________________________
Float_t AliPIDResponse::GetNumberOfSigmas(EDetector detector, const AliVParticle *vtrack, AliPID::EParticleType type) const
{
  //
  // NumberOfSigmas for 'detCode'
  //

  const AliVTrack *track=static_cast<const AliVTrack*>(vtrack);

  switch (detector){
    case kITS:   return GetNumberOfSigmasITS(track, type);   break;
    case kTPC:   return GetNumberOfSigmasTPC(track, type);   break;
    case kTRD:   return GetNumberOfSigmasTRD(track, type);   break;
    case kTOF:   return GetNumberOfSigmasTOF(track, type);   break;
    case kHMPID: return GetNumberOfSigmasHMPID(track, type); break;
    case kEMCAL: return GetNumberOfSigmasEMCAL(track, type); break;
    default: return -999.;
  }

  return -999.;
}

//______________________________________________________________________________
Float_t AliPIDResponse::GetNumberOfSigmasITS(const AliVParticle *vtrack, AliPID::EParticleType type) const
{
  //
  // Calculate the number of sigmas in the ITS
  //

  AliVTrack *track=(AliVTrack*)vtrack;

  const EDetPidStatus pidStatus=GetITSPIDStatus(track);
  if (pidStatus!=kDetPidOk) return -999.;

  // the following call is needed in order to fill the transient data member
  // fITSsignalTuned which is used in the ITSPIDResponse to judge
  // if using tuned on data
  if (fTuneMConData && ((fTuneMConDataMask & kDetITS) == kDetITS))
    GetITSsignalTunedOnData(track);

  return fITSResponse.GetNumberOfSigmas(track,type);
}

//______________________________________________________________________________
Float_t AliPIDResponse::GetNumberOfSigmasTPC(const AliVParticle *vtrack, AliPID::EParticleType type) const
{
  //
  // Calculate the number of sigmas in the TPC
  //

  AliVTrack *track=(AliVTrack*)vtrack;

  const EDetPidStatus pidStatus=GetTPCPIDStatus(track);
  if (pidStatus==kDetNoSignal) return -999.;

  // the following call is needed in order to fill the transient data member
  // fTPCsignalTuned which is used in the TPCPIDResponse to judge
  // if using tuned on data
  if (fTuneMConData && ((fTuneMConDataMask & kDetTPC) == kDetTPC))
    GetTPCsignalTunedOnData(track);

  return fTPCResponse.GetNumberOfSigmas(track, type, AliTPCPIDResponse::kdEdxDefault, fUseTPCEtaCorrection, fUseTPCMultiplicityCorrection);
}

//______________________________________________________________________________
Float_t AliPIDResponse::GetNumberOfSigmasTRD(const AliVParticle *vtrack, AliPID::EParticleType type) const
{
  //
  // Calculate the number of sigmas in the TRD
  //

  AliVTrack *track=(AliVTrack*)vtrack;

  const EDetPidStatus pidStatus=GetTRDPIDStatus(track);
  if (pidStatus!=kDetPidOk) return -999.;

  return fTRDResponse.GetNumberOfSigmas(track,type, fUseTRDEtaCorrection, fUseTRDClusterCorrection, fUseTRDCentralityCorrection);
}

//______________________________________________________________________________
Float_t AliPIDResponse::GetNumberOfSigmasTOF(const AliVParticle *vtrack, AliPID::EParticleType type) const
{
  //
  // Calculate the number of sigmas in the TOF
  //

  AliVTrack *track=(AliVTrack*)vtrack;

  const EDetPidStatus pidStatus=GetTOFPIDStatus(track);
  if (pidStatus!=kDetPidOk) return -999.;

  return GetNumberOfSigmasTOFold(vtrack, type);
}
//______________________________________________________________________________

Float_t AliPIDResponse::GetNumberOfSigmasHMPID(const AliVParticle *vtrack, AliPID::EParticleType type) const
{
  //
  // Calculate the number of sigmas in the HMPID
  //
  AliVTrack *track=(AliVTrack*)vtrack;

  const EDetPidStatus pidStatus=GetHMPIDPIDStatus(track);
  if (pidStatus!=kDetPidOk) return -999.;

  return fHMPIDResponse.GetNumberOfSigmas(track, type);
}

//______________________________________________________________________________
Float_t AliPIDResponse::GetNumberOfSigmasEMCAL(const AliVParticle *vtrack, AliPID::EParticleType type) const
{
  //
  // Calculate the number of sigmas in the EMCAL
  //

  AliVTrack *track=(AliVTrack*)vtrack;

  const EDetPidStatus pidStatus=GetEMCALPIDStatus(track);
  if (pidStatus!=kDetPidOk) return -999.;

  const Int_t nMatchClus = track->GetEMCALcluster();
  AliVCluster *matchedClus = (AliVCluster*)fCurrentEvent->GetCaloCluster(nMatchClus);

  const Double_t mom    = track->P();
  const Double_t pt     = track->Pt();
  const Int_t    charge = track->Charge();
  const Double_t fClsE  = matchedClus->E();
  const Double_t EovP   = fClsE/mom;

  return fEMCALResponse.GetNumberOfSigmas(pt,EovP,type,charge);
}

//______________________________________________________________________________
AliPIDResponse::EDetPidStatus AliPIDResponse::GetSignalDeltaITS(const AliVParticle *vtrack, AliPID::EParticleType type, Double_t &val, Bool_t ratio/*=kFALSE*/) const
{
  //
  // Signal minus expected Signal for ITS
  //
  AliVTrack *track=(AliVTrack*)vtrack;

  // the following call is needed in order to fill the transient data member
  // fITSsignalTuned which is used in the ITSPIDResponse to judge
  // if using tuned on data
  if (fTuneMConData && ((fTuneMConDataMask & kDetITS) == kDetITS))
    GetITSsignalTunedOnData(track);

  val=fITSResponse.GetSignalDelta(track,type,ratio);

  return GetITSPIDStatus(track);
}

//______________________________________________________________________________
AliPIDResponse::EDetPidStatus AliPIDResponse::GetSignalDeltaTPC(const AliVParticle *vtrack, AliPID::EParticleType type, Double_t &val, Bool_t ratio/*=kFALSE*/) const
{
  //
  // Signal minus expected Signal for TPC
  //
  AliVTrack *track=(AliVTrack*)vtrack;

  // the following call is needed in order to fill the transient data member
  // fTPCsignalTuned which is used in the TPCPIDResponse to judge
  // if using tuned on data
  if (fTuneMConData && ((fTuneMConDataMask & kDetTPC) == kDetTPC))
    GetTPCsignalTunedOnData(track);

  val=fTPCResponse.GetSignalDelta(track, type, AliTPCPIDResponse::kdEdxDefault, fUseTPCEtaCorrection, fUseTPCMultiplicityCorrection, ratio);

  return GetTPCPIDStatus(track);
}

//______________________________________________________________________________
AliPIDResponse::EDetPidStatus AliPIDResponse::GetSignalDeltaTRD(const AliVParticle *vtrack, AliPID::EParticleType type, Double_t &val, Bool_t ratio/*=kFALSE*/) const
{
  //
  // Signal minus expected Signal for TRD
  //
  AliVTrack *track=(AliVTrack*)vtrack;
  val=fTRDResponse.GetSignalDelta(track,type,ratio,fUseTRDEtaCorrection,fUseTRDClusterCorrection,fUseTRDCentralityCorrection);

  return GetTRDPIDStatus(track);
}

//______________________________________________________________________________
AliPIDResponse::EDetPidStatus AliPIDResponse::GetSignalDeltaTOF(const AliVParticle *vtrack, AliPID::EParticleType type, Double_t &val, Bool_t ratio/*=kFALSE*/) const
{
  //
  // Signal minus expected Signal for TOF
  //
  AliVTrack *track=(AliVTrack*)vtrack;
  val=GetSignalDeltaTOFold(track, type, ratio);

  return GetTOFPIDStatus(track);
}

//______________________________________________________________________________
AliPIDResponse::EDetPidStatus AliPIDResponse::GetSignalDeltaHMPID(const AliVParticle *vtrack, AliPID::EParticleType type, Double_t &val, Bool_t ratio/*=kFALSE*/) const
{
  //
  // Signal minus expected Signal for HMPID
  //
  AliVTrack *track=(AliVTrack*)vtrack;
  val=fHMPIDResponse.GetSignalDelta(track, type, ratio);

  return GetHMPIDPIDStatus(track);
}

//______________________________________________________________________________
AliPIDResponse::EDetPidStatus AliPIDResponse::GetComputePIDProbability  (EDetector detCode,  const AliVTrack *track, Int_t nSpecies, Double_t p[]) const
{
  //
  // Compute PID response of 'detCode'
  //

  switch (detCode){
    case kITS: return GetComputeITSProbability(track, nSpecies, p); break;
    case kTPC: return GetComputeTPCProbability(track, nSpecies, p); break;
    case kTRD: return GetComputeTRDProbability(track, nSpecies, p); break;
    case kTOF: return GetComputeTOFProbability(track, nSpecies, p); break;
    case kPHOS: return GetComputePHOSProbability(track, nSpecies, p); break;
    case kEMCAL: return GetComputeEMCALProbability(track, nSpecies, p); break;
    case kHMPID: return GetComputeHMPIDProbability(track, nSpecies, p); break;
    default: return kDetNoSignal;
  }
}

//______________________________________________________________________________
AliPIDResponse::EDetPidStatus AliPIDResponse::GetComputeITSProbability  (const AliVTrack *track, Int_t nSpecies, Double_t p[]) const
{
  //
  // Compute PID response for the ITS
  //

  // set flat distribution (no decision)
  for (Int_t j=0; j<nSpecies; j++) p[j]=1./nSpecies;

  const EDetPidStatus pidStatus=GetITSPIDStatus(track);
  if (pidStatus!=kDetPidOk) return pidStatus;

  if (track->GetDetectorPID()){
    return track->GetDetectorPID()->GetRawProbability(kITS, p, nSpecies);
  }

  //check for ITS standalone tracks
  Bool_t isSA=kTRUE;
  if( track->GetStatus() & AliVTrack::kTPCin ) isSA=kFALSE;

  Double_t mom=track->P();
  Double_t dedx=track->GetITSsignal();
  if (fTuneMConData && ((fTuneMConDataMask & kDetITS) == kDetITS)) dedx = GetITSsignalTunedOnData(track);
  (void) dedx;//mark as used to avoid warning

  Double_t momITS=mom;
  UChar_t clumap=track->GetITSClusterMap();
  Int_t nPointsForPid=0;
  for(Int_t i=2; i<6; i++){
    if(clumap&(1<<i)) ++nPointsForPid;
  }

  Bool_t mismatch=kTRUE/*, heavy=kTRUE*/;
  for (Int_t j=0; j<nSpecies; j++) {
    const Double_t chargeFactor = TMath::Power(AliPID::ParticleCharge(j),2.);
    //TODO: in case of the electron, use the SA parametrisation,
    //      this needs to be changed if ITS provides a parametrisation
    //      for electrons also for ITS+TPC tracks
    Double_t bethe=fITSResponse.Bethe(momITS,(AliPID::EParticleType)j,isSA || (j==(Int_t)AliPID::kElectron))*chargeFactor;
    Double_t sigma=fITSResponse.GetResolution(bethe,nPointsForPid,isSA || (j==(Int_t)AliPID::kElectron));
    Double_t nSigma=fITSResponse.GetNumberOfSigmas(track, (AliPID::EParticleType)j);
    if (TMath::Abs(nSigma) > fRange) {
      p[j]=TMath::Exp(-0.5*fRange*fRange)/sigma;
    } else {
      p[j]=TMath::Exp(-0.5*nSigma*nSigma)/sigma;
      mismatch=kFALSE;
    }
  }

  if (mismatch){
    for (Int_t j=0; j<nSpecies; j++) p[j]=1./nSpecies;
  }

  return kDetPidOk;
}
//______________________________________________________________________________
AliPIDResponse::EDetPidStatus AliPIDResponse::GetComputeTPCProbability  (const AliVTrack *track, Int_t nSpecies, Double_t p[]) const
{
  //
  // Compute PID response for the TPC
  //

  // set flat distribution (no decision)
  for (Int_t j=0; j<nSpecies; j++) p[j]=1./nSpecies;

  const EDetPidStatus pidStatus=GetTPCPIDStatus(track);
  if (pidStatus==kDetNoSignal) return pidStatus;

  Double_t dedx=track->GetTPCsignal();
  Bool_t mismatch=kTRUE/*, heavy=kTRUE*/;

  if (fTuneMConData && ((fTuneMConDataMask & kDetTPC) == kDetTPC)) dedx = GetTPCsignalTunedOnData(track);

  Double_t bethe = 0.;
  Double_t sigma = 0.;

  for (Int_t j=0; j<nSpecies; j++) {
    AliPID::EParticleType type=AliPID::EParticleType(j);

    bethe=fTPCResponse.GetExpectedSignal(track, type, AliTPCPIDResponse::kdEdxDefault, fUseTPCEtaCorrection, fUseTPCMultiplicityCorrection);
    sigma=fTPCResponse.GetExpectedSigma(track, type, AliTPCPIDResponse::kdEdxDefault, fUseTPCEtaCorrection, fUseTPCMultiplicityCorrection);

    if (TMath::Abs(dedx-bethe) > fRange*sigma) {
      p[j]=TMath::Exp(-0.5*fRange*fRange)/sigma;
    } else {
      p[j]=TMath::Exp(-0.5*(dedx-bethe)*(dedx-bethe)/(sigma*sigma))/sigma;
      mismatch=kFALSE;
    }
  }

  if (mismatch){
    for (Int_t j=0; j<nSpecies; j++) p[j]=1./nSpecies;
  }

  return pidStatus;
}
//______________________________________________________________________________
AliPIDResponse::EDetPidStatus AliPIDResponse::GetComputeTOFProbability  (const AliVTrack *track, Int_t nSpecies, Double_t p[],Bool_t kNoMism) const
{
  //
  // Compute PID probabilities for TOF
  //

  fgTOFmismatchProb = 1E-8;

  // centrality --> fCurrCentrality
  // Beam type --> fBeamTypeNum
  // N TOF cluster --> TOF header --> to get the TOF header we need to add a virtual method in AliVTrack extended to ESD and AOD tracks
  // isMC --> fIsMC
  Float_t pt = track->Pt();
  Float_t mismPropagationFactor[10] = {1.,1.,1.,1.,1.,1.,1.,1.,1.,1.};
  if(! (kNoMism | fNoTOFmism)){ // this flag allows to disable mismatch for iterative procedure to get priors
    mismPropagationFactor[3] = 1 + TMath::Exp(1 - 1.12*pt);// it has to be alligned with the one in AliPIDCombined
    mismPropagationFactor[4] = 1 + 1./(4.71114 - 5.72372*pt + 2.94715*pt*pt);// it has to be alligned with the one in AliPIDCombined

  Int_t nTOFcluster = 0;
  if(track->GetTOFHeader() && track->GetTOFHeader()->GetTriggerMask() && track->GetTOFHeader()->GetNumberOfTOFclusters() > -1){ // N TOF clusters available
    nTOFcluster = track->GetTOFHeader()->GetNumberOfTOFclusters();
    if(fIsMC) nTOFcluster = Int_t(nTOFcluster * 1.5); // +50% in MC
  }
  else{
    switch(fBeamTypeNum){
      case kPP: // pp
	nTOFcluster = 80;
	break;
      case kPPB: // pPb 5.05 ATeV
	nTOFcluster = Int_t(308 - 2.12*fCurrCentrality + TMath::Exp(4.917 -0.1604*fCurrCentrality));
	break;
      case kPBPB: // PbPb 2.76 ATeV
	nTOFcluster = Int_t(TMath::Exp(9.4 - 0.022*fCurrCentrality));
	break;
    }
  }

    switch(fBeamTypeNum){ // matching window factors for 3 cm and 10 cm (about (10/3)^2)
    case kPP: // pp 7 TeV
      nTOFcluster *= 10;
      break;
    case kPPB: // pPb 5.05 ATeV
      nTOFcluster *= 10;
      break;
    case kPBPB: // pPb 5.05 ATeV
      //       nTOFcluster *= 1;
      break;
    }

    if(nTOFcluster < 0) nTOFcluster = 10;


    fgTOFmismatchProb=fTOFResponse.GetMismatchProbability(track->GetTOFsignal(),track->Eta()) * nTOFcluster *6E-6 * (1 + 2.90505e-01/pt/pt); // mism weight * tof occupancy (including matching window factor) * pt dependence

  }

  // set flat distribution (no decision)
  for (Int_t j=0; j<nSpecies; j++) p[j]=1./nSpecies;

  const EDetPidStatus pidStatus=GetTOFPIDStatus(track);
  if (pidStatus!=kDetPidOk) return pidStatus;

  const Double_t meanCorrFactor = 0.07/fTOFtail; // Correction factor on the mean because of the tail (should be ~ 0.1 with tail = 1.1)

  for (Int_t j=0; j<nSpecies; j++) {
    AliPID::EParticleType type=AliPID::EParticleType(j);
    const Double_t nsigmas=GetNumberOfSigmasTOFold(track,type) + meanCorrFactor;

    const Double_t expTime = fTOFResponse.GetExpectedSignal(track,type);
    const Double_t sig     = fTOFResponse.GetExpectedSigma(track->P(),expTime,AliPID::ParticleMassZ(type));

    if(nsigmas < fTOFtail)
      p[j] = TMath::Exp(-0.5*nsigmas*nsigmas)/sig;
    else
      p[j] = TMath::Exp(-(nsigmas - fTOFtail*0.5)*fTOFtail)/sig;

    p[j] += fgTOFmismatchProb*mismPropagationFactor[j];
  }

  return kDetPidOk;
}

Int_t AliPIDResponse::CalculateTRDResponse(const AliVTrack *track,Double_t p[],AliTRDPIDResponse::ETRDPIDMethod PIDmethod) const
{
    // new function for backward compatibility
    // returns number of tracklets PID

    UInt_t TRDslicesForPID[2];
    SetTRDSlices(TRDslicesForPID,PIDmethod);

    Float_t mom[6]={0.};
    Double_t dedx[48]={0.};  // Allocate space for the maximum number of TRD slices
    Int_t nslices = TRDslicesForPID[1] - TRDslicesForPID[0] + 1;
    for(UInt_t ilayer = 0; ilayer < 6; ilayer++){
	mom[ilayer] = track->GetTRDmomentum(ilayer);
	for(UInt_t islice = TRDslicesForPID[0]; islice <= TRDslicesForPID[1]; islice++){
            // do not consider tracklets with no momentum (should be non functioning chambers)
            if(mom[ilayer]>0) dedx[ilayer*nslices+islice-TRDslicesForPID[0]] = track->GetTRDslice(ilayer, islice);
	}
    }

    return fTRDResponse.GetResponse(nslices, dedx, mom, p,PIDmethod, kTRUE, track);

}
//______________________________________________________________________________
AliPIDResponse::EDetPidStatus AliPIDResponse::GetComputeTRDProbability  (const AliVTrack *track, Int_t nSpecies, Double_t p[],AliTRDPIDResponse::ETRDPIDMethod PIDmethod) const
{
  //
  // Compute PID probabilities for the TRD
  //

  // set flat distribution (no decision)
  for (Int_t j=0; j<nSpecies; j++) p[j]=1./nSpecies;
  const EDetPidStatus pidStatus=GetTRDPIDStatus(track);
  if (pidStatus!=kDetPidOk) return pidStatus;

  CalculateTRDResponse(track,p,PIDmethod);

  return kDetPidOk;
}

//______________________________________________________________________________
AliPIDResponse::EDetPidStatus AliPIDResponse::GetComputeEMCALProbability  (const AliVTrack *track, Int_t nSpecies, Double_t p[]) const
{
  //
  // Compute PID response for the EMCAL
  //

  for (Int_t j=0; j<nSpecies; j++) p[j]=1./nSpecies;

  const EDetPidStatus pidStatus=GetEMCALPIDStatus(track);
  if (pidStatus!=kDetPidOk) return pidStatus;

  const Int_t nMatchClus = track->GetEMCALcluster();
  AliVCluster *matchedClus = (AliVCluster*)fCurrentEvent->GetCaloCluster(nMatchClus);

  const Double_t mom    = track->P();
  const Double_t pt     = track->Pt();
  const Int_t    charge = track->Charge();
  const Double_t fClsE  = matchedClus->E();
  const Double_t EovP   = fClsE/mom;

  // compute the probabilities
  fEMCALResponse.ComputeEMCALProbability(nSpecies,pt,EovP,charge,p);
  return kDetPidOk;
}

//______________________________________________________________________________
AliPIDResponse::EDetPidStatus AliPIDResponse::GetComputePHOSProbability (const AliVTrack */*track*/, Int_t nSpecies, Double_t p[]) const
{
  //
  // Compute PID response for the PHOS
  //

  // set flat distribution (no decision)
  for (Int_t j=0; j<nSpecies; j++) p[j]=1./nSpecies;
  return kDetNoSignal;
}

//______________________________________________________________________________
AliPIDResponse::EDetPidStatus AliPIDResponse::GetComputeHMPIDProbability(const AliVTrack *track, Int_t nSpecies, Double_t p[]) const
{
  //
  // Compute PID response for the HMPID
  //

  // set flat distribution (no decision)
  for (Int_t j=0; j<nSpecies; j++) p[j]=1./nSpecies;

  const EDetPidStatus pidStatus=GetHMPIDPIDStatus(track);
  if (pidStatus!=kDetPidOk) return pidStatus;

  fHMPIDResponse.GetProbability(track,nSpecies,p);

  return kDetPidOk;
}

//______________________________________________________________________________
AliPIDResponse::EDetPidStatus AliPIDResponse::GetITSPIDStatus(const AliVTrack *track) const
{
  // compute ITS pid status

  // check status bits
  if ((track->GetStatus()&AliVTrack::kITSin)==0 &&
    (track->GetStatus()&AliVTrack::kITSout)==0) return kDetNoSignal;

  const Float_t dEdx=track->GetITSsignal();
  if (dEdx<=0) return kDetNoSignal;

  // requite at least 3 pid clusters
  const UChar_t clumap=track->GetITSClusterMap();
  Int_t nPointsForPid=0;
  for(Int_t i=2; i<6; i++){
    if(clumap&(1<<i)) ++nPointsForPid;
  }

  if(nPointsForPid<3) {
    return kDetNoSignal;
  }

  return kDetPidOk;
}

//______________________________________________________________________________
AliPIDResponse::EDetPidStatus AliPIDResponse:: GetTPCPIDStatus(const AliVTrack *track) const
{
  // compute TPC pid status

  // check quality of the track
  if ( (track->GetStatus()&AliVTrack::kTPCin )==0 && (track->GetStatus()&AliVTrack::kTPCout)==0 ) return kDetNoSignal;

  // check pid values
  const Double_t dedx=track->GetTPCsignal();
  const UShort_t signalN=track->GetTPCsignalN();
  if (signalN<10 || dedx<10) return kDetNoSignal;

  if (!fTPCResponse.GetResponseFunction(AliPID::kPion)) return kDetNoParams;

  return kDetPidOk;
}

//______________________________________________________________________________
AliPIDResponse::EDetPidStatus AliPIDResponse::GetTRDPIDStatus(const AliVTrack *track) const
{
  // compute TRD pid status

  if((track->GetStatus()&AliVTrack::kTRDout)==0) return kDetNoSignal;
  return kDetPidOk;
}

//______________________________________________________________________________
AliPIDResponse::EDetPidStatus AliPIDResponse::GetTOFPIDStatus(const AliVTrack *track) const
{
  // compute TOF pid status

  if ((track->GetStatus()&AliVTrack::kTOFout)==0) return kDetNoSignal;
  if ((track->GetStatus()&AliVTrack::kTIME)==0) return kDetNoSignal;

  return kDetPidOk;
}

//______________________________________________________________________________
Float_t AliPIDResponse::GetTOFMismatchProbability(const AliVTrack *track) const
{
  // compute mismatch probability cross-checking at 5 sigmas with TPC
  // currently just implemented as a 5 sigma compatibility cut

  if(!track) return fgTOFmismatchProb;

  // check pid status
  const EDetPidStatus tofStatus=GetTOFPIDStatus(track);
  if (tofStatus!=kDetPidOk) return 0.;

  //mismatch
  const EDetPidStatus tpcStatus=GetTPCPIDStatus(track);
  if (tpcStatus==kDetNoSignal) return 0.;

  const Double_t meanCorrFactor = 0.11/fTOFtail; // Correction factor on the mean because of the tail (should be ~ 0.1 with tail = 1.1)
  Bool_t mismatch = kTRUE/*, heavy = kTRUE*/;
  for (Int_t j=0; j<AliPID::kSPECIESC; j++) {
    AliPID::EParticleType type=AliPID::EParticleType(j);
    const Double_t nsigmas=GetNumberOfSigmasTOFold(track,type) + meanCorrFactor;

    if (TMath::Abs(nsigmas)<5.){
      const Double_t nsigmasTPC=GetNumberOfSigmasTPC(track,type);
      if (TMath::Abs(nsigmasTPC)<5.) mismatch=kFALSE;
    }
  }

  if (mismatch){
    return 1.;
  }

  return 0.;
}

//______________________________________________________________________________
AliPIDResponse::EDetPidStatus AliPIDResponse:: GetHMPIDPIDStatus(const AliVTrack *track) const
{
  // compute HMPID pid status

  Int_t ch = track->GetHMPIDcluIdx()/1000000;
  Double_t HMPIDsignal = track->GetHMPIDsignal();

  if((track->GetStatus()&AliVTrack::kHMPIDpid)==0 || ch<0 || ch>6 || HMPIDsignal<0) return kDetNoSignal;

  return kDetPidOk;
}

//______________________________________________________________________________
AliPIDResponse::EDetPidStatus AliPIDResponse:: GetPHOSPIDStatus(const AliVTrack */*track*/) const
{
  // compute PHOS pid status
  return kDetNoSignal;
}

//______________________________________________________________________________
AliPIDResponse::EDetPidStatus AliPIDResponse:: GetEMCALPIDStatus(const AliVTrack *track) const
{
  // compute EMCAL pid status


  // Track matching
  const Int_t nMatchClus = track->GetEMCALcluster();
  if (nMatchClus<0) return kDetNoSignal;

  AliVCluster *matchedClus = (AliVCluster*)fCurrentEvent->GetCaloCluster(nMatchClus);

  if (!(matchedClus && matchedClus->IsEMCAL())) return kDetNoSignal;

  const Int_t charge = track->Charge();
  if (TMath::Abs(charge)!=1) return kDetNoSignal;

  if (!(fEMCALPIDParams && fEMCALPIDParams->At(AliPID::kElectron))) return kDetNoParams;

  return kDetPidOk;

}

//______________________________________________________________________________
AliPIDResponse::EDetPidStatus AliPIDResponse::GetPIDStatus(EDetector detector, const AliVTrack *track) const
{
  //
  // check pid status for a track
  //

  switch (detector){
    case kITS:   return GetITSPIDStatus(track);   break;
    case kTPC:   return GetTPCPIDStatus(track);   break;
    case kTRD:   return GetTRDPIDStatus(track);   break;
    case kTOF:   return GetTOFPIDStatus(track);   break;
    case kPHOS:  return GetPHOSPIDStatus(track);  break;
    case kEMCAL: return GetEMCALPIDStatus(track); break;
    case kHMPID: return GetHMPIDPIDStatus(track); break;
    default: return kDetNoSignal;
  }
  return kDetNoSignal;

}

//_________________________________________________________________________
Float_t AliPIDResponse::GetITSsignalTunedOnData(const AliVTrack *t) const
{
  /// Create gaussian signal response based on the dE/dx response observed in data
  /// Currently only for deuterons and triton. The other particles are fine in MC

  Float_t dedx = t->GetITSsignalTunedOnData();
  if(dedx > 0) return dedx;

  dedx = t->GetITSsignal();
  ((AliVTrack*)t)->SetITSsignalTunedOnData(dedx);
  if(dedx < 20) return dedx;

  Int_t nITSpid = 0;
  for(Int_t i=2; i<6; i++){
    if(t->HasPointOnITSLayer(i)) nITSpid++;
  }

  Bool_t isSA=kTRUE;
  if( t->GetStatus() & AliVTrack::kTPCin ) isSA=kFALSE;

  // check if we have MC information
  if (!fCurrentMCEvent) {
    AliError("Tune On Data requested, but MC event not set. Call 'SetCurrentMCEvent' before!");
    return dedx;
  }

  // get MC particle
  AliVParticle *mcPart = fCurrentMCEvent->GetTrack(TMath::Abs(t->GetLabel()));

  if (mcPart != NULL) { // protect against label-0 track (initial proton in Pythia events)
    AliPID::EParticleType type = AliPID::kPion;
    Bool_t kGood = kFALSE;
    Int_t iS = TMath::Abs(mcPart->PdgCode());

    for (Int_t ipart=0; ipart<AliPID::kSPECIESC; ++ipart) {
      if (iS == AliPID::ParticleCode(ipart)) {
        type = static_cast<AliPID::EParticleType>(ipart);
        kGood=kTRUE;
        break;
      }
    }

    // NOTE: currently only tune for deuterons and triton
    if(kGood && (iS == AliPID::ParticleCode(AliPID::kDeuteron) || (iS == AliPID::ParticleCode(AliPID::kTriton))) ) {
      Double_t bethe = fITSResponse.Bethe(t->P(),type,isSA);
      Double_t sigma = fITSResponse.GetResolution(bethe,nITSpid,isSA,t->P(),type);
      dedx = gRandom->Gaus(bethe,sigma);
    }
  }

  const_cast<AliVTrack*>(t)->SetITSsignalTunedOnData(dedx);
  return dedx;
}

//_________________________________________________________________________
Float_t AliPIDResponse::GetTPCsignalTunedOnData(const AliVTrack *t) const
{
  /// Create gaussian signal response based on the dE/dx response observed in data

  Float_t dedx = t->GetTPCsignalTunedOnData();

  if(dedx > 0) return dedx;

  dedx = t->GetTPCsignal();
  ((AliVTrack*)t)->SetTPCsignalTunedOnData(dedx);
  if(dedx < 20) return dedx;

  // check if we have MC information
  if (!fCurrentMCEvent) {
    AliError("Tune On Data requested, but MC event not set. Call 'SetCurrentMCEvent' before!");
    return dedx;
  }

  // get MC particle
  AliVParticle *mcPart = fCurrentMCEvent->GetTrack(TMath::Abs(t->GetLabel()));

  if (mcPart != NULL) { // protect against label-0 track (initial proton in Pythia events)
    AliPID::EParticleType type = AliPID::kPion;
    Bool_t kGood = kFALSE;
    Int_t iS = TMath::Abs(mcPart->PdgCode());

    for (Int_t ipart=0; ipart<AliPID::kSPECIESC; ++ipart) {
      if (iS == AliPID::ParticleCode(ipart)) {
        type = static_cast<AliPID::EParticleType>(ipart);
        kGood=kTRUE;
        break;
      }
    }

    if(kGood){
      //TODO maybe introduce different dEdxSources?
      Double_t bethe = fTPCResponse.GetExpectedSignal(t, type, AliTPCPIDResponse::kdEdxDefault, UseTPCEtaCorrection(),
                                                      UseTPCMultiplicityCorrection());
      Double_t sigma = fTPCResponse.GetExpectedSigma(t, type, AliTPCPIDResponse::kdEdxDefault, UseTPCEtaCorrection(),
                                                     UseTPCMultiplicityCorrection());
      dedx = gRandom->Gaus(bethe,sigma);
      //              if(iS == AliPID::ParticleCode(AliPID::kHe3) || iS == AliPID::ParticleCode(AliPID::kAlpha)) dedx *= 5;
    }
  }

  const_cast<AliVTrack*>(t)->SetTPCsignalTunedOnData(dedx);
  return dedx;
}

//_________________________________________________________________________
Float_t AliPIDResponse::GetTOFsignalTunedOnData(const AliVTrack *t) const
{
  /// Calculate the TOF signal tuned on data by adding a tail
  Double_t tofSignal = t->GetTOFsignalTunedOnData();

  if(tofSignal <  99999) return (Float_t)tofSignal; // it has been already set

  // read additional mismatch fraction
  Float_t addmism = GetTOFPIDParams()->GetTOFadditionalMismForMC();
  if(addmism > 1.){
    Float_t centr = GetCurrentCentrality();
    if(centr > 50) addmism *= 0.1667;
    else if(centr > 20) addmism *= 0.33;
  }

  tofSignal = t->GetTOFsignal() + fTOFResponse.GetTailRandomValue(t->Pt(),t->Eta(),t->GetTOFsignal(),addmism);
  const_cast<AliVTrack*>(t)->SetTOFsignalTunedOnData(tofSignal);
  return (Float_t)tofSignal;
}
