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
//-----------------------------------------------------------------
//           Implementation of the ESD track class
//   ESD = Event Summary Data
//   This is the class to deal with during the phisics analysis of data
//      Origin: Iouri Belikov, CERN
//      e-mail: Jouri.Belikov@cern.ch
//
//
//
//  What do you need to know before starting analysis
//  (by Marian Ivanov: marian.ivanov@cern.ch)
//
//
//   AliESDtrack:
//   1.  What is the AliESDtrack
//   2.  What informations do we store
//   3.  How to use the information for analysis
//   
//
//   1.AliESDtrack is the container of the information about the track/particle
//     reconstructed during Barrel Tracking.
//     Content:
//        a.) Track parameters and covariance - AliExternalTrackParam  - snapshots along trajectory at differnt tracking steps 
//            current, fIp, fTPCinner, fCp, fOp, fHMPIDp, + friendTrack (fITSout, fTPCout, fTRDin) 
//        b.) Flags - per detector status bits
//        c.) Track fit quality -chi2, number of clusters
//        d.) Detector PID information
//        d.) Different detector specific information (e.g number of tracklets in TRD, TOF cluster descriptor ...)
//
//
//     The track information is propagated from one tracking detector to 
//     other using the functionality of AliESDtrack (AliExternalTrackParam - current parameters)  
//
//     Barrel tracking uses Kalman filtering technique (no global fit model is used). Kalman provides optimal local 
//     track parameters estimate at given position under certian assumptions. 
//     Following approximations were used:  
//           a.) gaussian Multiple scattering
//           b.) gaussian energy loss
//           c.) gaussian error of the space point residuals  
//             
//     Kalman filter take into account following effects which are 
//     difficult to handle using global fit:
//        a.) Multiple scattering
//        b.) Energy loss
//        c.) Non homogenous magnetic field
//
//     In general case, following barrel detectors are contributing to 
//     the Kalman track information:
//         a. TPC
//         b. ITS
//         c. TRD
//
//      Track findind/fitting procedure is done in 3 steps:
//         1. Cluster2Track(in)   - inward sequence TPC->ITS    
//         2. PropagateBack(out)   - outward sequence ITS->TPC->TRD -> Outer PID detectors
//         3. RefitInward(refit)     - inward sequence TRD->TPC->ITS
//      After each recosntruction step detector status is updated in the data member fFlags
//      fFlags|=k<DetectorName><step> where step={1:in,2:out,3:refit,}
//      For some of detectors a special flags were implemented. Declaration of list of all flags can be find in $ALICE_ROOT/STEER/STEERBase/AliVTrack.h
//
//  
//      The current track parameter is updated after each detector (see bellow).
//      In specical cases a track  snapshots (AliExternalTrackParam) are stored
//
// 
//
//      For some type of analysis (+visualization) track local parameters at 
//      different position are neccesary. A snapshots during the track 
//      propagation are created and stored either in track itself (for analysis purposes) or assiciated friend track (for calibration and debugging purposes)
//      (See AliExternalTrackParam class for desctiption of variables and functionality)
//      Snapshots:
//      a.)  Current parameters (AliESDtrack itself) 
//         Description:  as obtained in the last succesfull tracking step
//         Contributors: best case TRD->TPC->ITS after RefitInward
//         Recomended way to get track parameters. It includes all of the information.
//         NOTICE - By default the track parameters are stored at the DCA point to the primary vertex. 
//                  Optimal for primary tracks, far from optimal for deeply secondary tracks.
//                  To get optimal track parameters at the secondary vertex, OnTheFly V0s with associated track parameters should be used
//
//      b.) Constrained parameters (fCp)
//         Description: 
//            Kalman track  updated with the Primary vertex information with corresponding error (soft  constraint - see http://en.wikipedia.org/wiki/Constraint_(mathematics)#Hard_and_soft_constraints) 
//         Function:  
//            const AliExternalTrackParam *GetConstrainedParam() const {return fCp;}
//         Contributors: best case TRD->TPC->ITS after RefitInward
//         Recommended usage: Use only for tracks selected as primary (check GetConstrainedChi2())
//         NOTICE - not real constraint only soft constraint
//
//      c.) Inner parameters (fIp) -  
//         Description: Track parameters at inner wall of the TPC 
//         Function:
//           const AliExternalTrackParam *GetInnerParam() const { return fIp;}
//         Contributors: general case TRD->TPC (during RefitInward)
//         Recomended usage: To provide momenta for the TPC PID and to estimate quality of the track determination for further 
//
//      d.)  TPCinnerParam (fTPCinner):
//         Description: TPC only parameters at DCA to the primary vertex (corrected for the material between TPC and vertex)
//         Function:
//                const AliExternalTrackParam *GetTPCInnerParam() const {return fTPCInner;}
//         Contributors:  TPC only from in step  1 (Cluster2Tracks)
//         Recomended usage: Requested for HBT study 
//                           (smaller correlations as using also ITS information)
//         NOTICE: Optimal for primary, far from optimal for secondary tracks (similar to track parameters a.) 
//                 ! We should always use the c.) fIp in case of the TPC PID analysis, or undo material budget correction!
//
//      e.) Outer parameters - (fOp)
//         Description: track parameters during PropagateBack in the last sucessfull propagation  
//                  Reason to generate backup  OuterParameters
//                        a.) Local inclination angle bigger than threshold - 
//                            Low momenta tracks 
//                        b.) Catastrofic (large relative>~20 %)energy loss in material outside of the TPC
//                        c.) No additional  space points contributing to track 
//         Function:
//            const AliExternalTrackParam *GetOuterParam() const { return fOp;}
//         Contributors:  general case - ITS-> TPC -> TRD ( during PropagateBack )
//         Recomended usage:             
//              a.) Tracking: Starting parameter for Refit inward 
//              b.) Visualization
//              c.) QA
//         NOTICE: Should be not used for the physic analysis
//         Function:
//            const AliExternalTrackParam *GetOuterParam() const { return fOp;}
//
//-----------------------------------------------------------------

#include <TMath.h>
#include <TParticle.h>
#include <TDatabasePDG.h>
#include <TMatrixD.h>

#include "AliESDVertex.h"
#include "AliESDtrack.h"
#include "AliESDEvent.h"
#include "AliKalmanTrack.h"
#include "AliVTrack.h"
#include "AliLog.h"
#include "AliTrackPointArray.h"
#include "TPolyMarker3D.h"
#include "AliTrackerBase.h"
#include "AliTPCdEdxInfo.h"
#include "AliDetectorPID.h"
#include "TTreeStream.h"
#include "TObjArray.h"

ClassImp(AliESDtrack)

Bool_t AliESDtrack::fgTrackEMuAsPi = kTRUE;

void SetPIDValues(Double_t * dest, const Double_t * src, Int_t n) {
  // This function copies "n" PID weights from "scr" to "dest"
  // and normalizes their sum to 1 thus producing conditional probabilities.
  // The negative weights are set to 0.
  // In case all the weights are non-positive they are replaced by
  // uniform probabilities

  if (n<=0) return;

  Float_t uniform = 1./(Float_t)n;

  Float_t sum = 0;
  for (Int_t i=0; i<n; i++) 
    if (src[i]>=0) {
      sum+=src[i];
      dest[i] = src[i];
    }
    else {
      dest[i] = 0;
    }

  if(sum>0)
    for (Int_t i=0; i<n; i++) dest[i] /= sum;
  else
    for (Int_t i=0; i<n; i++) dest[i] = uniform;
}

//_______________________________________________________________________
AliESDtrack::AliESDtrack() : 
  AliExternalTrackParam(),
  fCp(0),
  fIp(0),
  fTPCInner(0),
  fOp(0),
  fHMPIDp(0),  
  fFriendTrack(NULL),
  fTPCFitMap(159),//number of padrows
  fTPCClusterMap(159),//number of padrows
  fTPCSharedMap(159),//number of padrows
  fFrTrackID(0),
  fFlags(0),
  fID(0),
  fLabel(0),
  fITSLabel(0),
  fTPCLabel(0),
  fTRDLabel(0),
  fTOFLabel(NULL),
  fTOFCalChannel(-1),
  fTOFindex(-1),
  fHMPIDqn(0),
  fHMPIDcluIdx(-1),
  fCaloIndex(kEMCALNoMatch),
  fR(0),
  fITSr(0),
  fTPCr(0),
  fTRDr(0),
  fTOFr(0),
  fHMPIDr(0),
  fHMPIDtrkTheta(0),
  fHMPIDtrkPhi(0),
  fHMPIDsignal(0),
  fTrackTime(0),
  fTrackLength(0),
  fdTPC(0),fzTPC(0),
  fCddTPC(0),fCdzTPC(0),fCzzTPC(0),
  fCchi2TPC(0),
  fD(0),fZ(0),
  fCdd(0),fCdz(0),fCzz(0),
  fCchi2(0),
  fITSchi2(0),
  fTPCchi2(0),
  fTPCchi2Iter1(0),
  fTRDchi2(0),
  fTOFchi2(0),
  fHMPIDchi2(0),
  fGlobalChi2(0),
  fITSsignal(0),
  fITSsignalTuned(0),
  fTPCsignal(0),
  fTPCsignalTuned(0),
  fTPCsignalS(0),
  fTPCdEdxInfo(0),
  fTRDsignal(0),
  fTRDQuality(0),
  fTRDBudget(0),
  fTOFsignal(99999),
  fTOFsignalTuned(99999),
  fTOFsignalToT(99999),
  fTOFsignalRaw(99999),
  fTOFsignalDz(999),
  fTOFsignalDx(999),
  fTOFdeltaBC(999),
  fTOFl0l1(999),
  fCaloDx(0),
  fCaloDz(0),
  fHMPIDtrkX(0),
  fHMPIDtrkY(0),
  fHMPIDmipX(0),
  fHMPIDmipY(0),
  fTPCncls(0),
  fTPCnclsF(0),
  fTPCsignalN(0),
  fTPCnclsIter1(0),
  fTPCnclsFIter1(0),
  fITSncls(0),
  fITSClusterMap(0),
  fITSSharedMap(0),
  fTRDncls(0),
  fTRDncls0(0),
  fTRDntracklets(0),
  fTRDNchamberdEdx(0),
  fTRDNclusterdEdx(0),
  fTRDnSlices(0),
  fTRDslices(0x0),
  fVertexID(-2),// -2 means an orphan track 
  fPIDForTracking(AliPID::kPion),
  fPIDForTrackingIn(AliPID::kPion),
  fESDEvent(0),
  fCacheNCrossedRows(-10),
  fCacheChi2TPCConstrainedVsGlobal(-10),
  fCacheChi2TPCConstrainedVsGlobalVertex(0),
  fDetectorPID(0x0),
  fTrackPhiOnEMCal(-999),
  fTrackEtaOnEMCal(-999),
  fTrackPtOnEMCal(-999),
  fNtofClusters(0),
  fTOFcluster(NULL)
{
  //
  // The default ESD constructor 
  //
  if (!OnlineMode()) fFriendTrack=new AliESDfriendTrack();

  Int_t i;
  for (i=kNITSchi2Std;i--;) fITSchi2Std[i] = 0;
  
  for (i=0; i<3; i++)   { fKinkIndexes[i]=0;}
  for (i=0; i<3; i++)   { fV0Indexes[i]=0;}
  for (i=0;i<kTRDnPlanes;i++) {
    fTRDTimBin[i]=0;
  }
  for (i=0;i<4;i++) {fITSdEdxSamples[i]=0.;}
  for (i=0;i<4;i++) {fTPCPoints[i]=0;}
  for (i=0;i<10;i++) {fTOFInfo[i]=0;}
  for (i=0;i<12;i++) {fITSModule[i]=-1;}
}

bool AliESDtrack::fgkOnlineMode=false;

//_______________________________________________________________________
AliESDtrack::AliESDtrack(const AliESDtrack& track):
  AliExternalTrackParam(track),
  fCp(0),
  fIp(0),
  fTPCInner(0),
  fOp(0),
  fHMPIDp(0),  
  fFriendTrack(0),
  fTPCFitMap(track.fTPCFitMap),
  fTPCClusterMap(track.fTPCClusterMap),
  fTPCSharedMap(track.fTPCSharedMap),
  fFrTrackID(track.fFrTrackID),
  fFlags(track.fFlags),
  fID(track.fID),
  fLabel(track.fLabel),
  fITSLabel(track.fITSLabel),
  fTPCLabel(track.fTPCLabel),
  fTRDLabel(track.fTRDLabel),
  fTOFLabel(NULL),
  fTOFCalChannel(track.fTOFCalChannel),
  fTOFindex(track.fTOFindex),
  fHMPIDqn(track.fHMPIDqn),
  fHMPIDcluIdx(track.fHMPIDcluIdx),
  fCaloIndex(track.fCaloIndex),
  fR(0),
  fITSr(0),
  fTPCr(0),
  fTRDr(0),
  fTOFr(0),
  fHMPIDr(0),
  fHMPIDtrkTheta(track.fHMPIDtrkTheta),
  fHMPIDtrkPhi(track.fHMPIDtrkPhi),
  fHMPIDsignal(track.fHMPIDsignal),
  fTrackTime(NULL),
  fTrackLength(track.fTrackLength),
  fdTPC(track.fdTPC),fzTPC(track.fzTPC),
  fCddTPC(track.fCddTPC),fCdzTPC(track.fCdzTPC),fCzzTPC(track.fCzzTPC),
  fCchi2TPC(track.fCchi2TPC),
  fD(track.fD),fZ(track.fZ),
  fCdd(track.fCdd),fCdz(track.fCdz),fCzz(track.fCzz),
  fCchi2(track.fCchi2),
  fITSchi2(track.fITSchi2),
  fTPCchi2(track.fTPCchi2),
  fTPCchi2Iter1(track.fTPCchi2Iter1),
  fTRDchi2(track.fTRDchi2),
  fTOFchi2(track.fTOFchi2),
  fHMPIDchi2(track.fHMPIDchi2),
  fGlobalChi2(track.fGlobalChi2),
  fITSsignal(track.fITSsignal),
  fITSsignalTuned(track.fITSsignalTuned),
  fTPCsignal(track.fTPCsignal),
  fTPCsignalTuned(track.fTPCsignalTuned),
  fTPCsignalS(track.fTPCsignalS),
  fTPCdEdxInfo(0),
  fTRDsignal(track.fTRDsignal),
  fTRDQuality(track.fTRDQuality),
  fTRDBudget(track.fTRDBudget),
  fTOFsignal(track.fTOFsignal),
  fTOFsignalTuned(track.fTOFsignalTuned),
  fTOFsignalToT(track.fTOFsignalToT),
  fTOFsignalRaw(track.fTOFsignalRaw),
  fTOFsignalDz(track.fTOFsignalDz),
  fTOFsignalDx(track.fTOFsignalDx),
  fTOFdeltaBC(track.fTOFdeltaBC),
  fTOFl0l1(track.fTOFl0l1),
  fCaloDx(track.fCaloDx),
  fCaloDz(track.fCaloDz),
  fHMPIDtrkX(track.fHMPIDtrkX),
  fHMPIDtrkY(track.fHMPIDtrkY),
  fHMPIDmipX(track.fHMPIDmipX),
  fHMPIDmipY(track.fHMPIDmipY),
  fTPCncls(track.fTPCncls),
  fTPCnclsF(track.fTPCnclsF),
  fTPCsignalN(track.fTPCsignalN),
  fTPCnclsIter1(track.fTPCnclsIter1),
  fTPCnclsFIter1(track.fTPCnclsIter1),
  fITSncls(track.fITSncls),
  fITSClusterMap(track.fITSClusterMap),
  fITSSharedMap(track.fITSSharedMap),
  fTRDncls(track.fTRDncls),
  fTRDncls0(track.fTRDncls0),
  fTRDntracklets(track.fTRDntracklets),
  fTRDNchamberdEdx(track.fTRDNchamberdEdx),
  fTRDNclusterdEdx(track.fTRDNclusterdEdx),
  fTRDnSlices(track.fTRDnSlices),
  fTRDslices(0x0),
  fVertexID(track.fVertexID),
  fPIDForTracking(AliPID::kPion),
  fPIDForTrackingIn(AliPID::kPion),
  fESDEvent(track.fESDEvent),
  fCacheNCrossedRows(track.fCacheNCrossedRows),
  fCacheChi2TPCConstrainedVsGlobal(track.fCacheChi2TPCConstrainedVsGlobal),
  fCacheChi2TPCConstrainedVsGlobalVertex(track.fCacheChi2TPCConstrainedVsGlobalVertex),
  fDetectorPID(0x0),
  fTrackPhiOnEMCal(track.fTrackPhiOnEMCal),
  fTrackEtaOnEMCal(track.fTrackEtaOnEMCal),
  fTrackPtOnEMCal(track.fTrackPtOnEMCal),
  fNtofClusters(track.fNtofClusters),
  fTOFcluster(NULL)
{
  //
  //copy constructor
  //
  for (Int_t i=kNITSchi2Std;i--;) fITSchi2Std[i] = track.fITSchi2Std[i];

  if(track.fTrackTime){
    fTrackTime = new Double32_t[AliPID::kSPECIESC];
    for (Int_t i=0;i<AliPID::kSPECIESC;i++) fTrackTime[i]=track.fTrackTime[i];
  }

  if (track.fR) {
    fR = new Double32_t[AliPID::kSPECIES]; 
    for (Int_t i=AliPID::kSPECIES;i--;)  fR[i]=track.fR[i];
  }
  if (track.fITSr) {
    fITSr = new Double32_t[AliPID::kSPECIES]; 
    for (Int_t i=AliPID::kSPECIES;i--;)  fITSr[i]=track.fITSr[i]; 
  }
  //
  if (track.fTPCr) {
    fTPCr = new Double32_t[AliPID::kSPECIES]; 
    for (Int_t i=AliPID::kSPECIES;i--;) fTPCr[i]=track.fTPCr[i]; 
  }

  for (Int_t i=0;i<4;i++) {fITSdEdxSamples[i]=track.fITSdEdxSamples[i];}
  for (Int_t i=0;i<4;i++) {fTPCPoints[i]=track.fTPCPoints[i];}
  for (Int_t i=0; i<3;i++)   { fKinkIndexes[i]=track.fKinkIndexes[i];}
  for (Int_t i=0; i<3;i++)   { fV0Indexes[i]=track.fV0Indexes[i];}
  //
  for (Int_t i=0;i<kTRDnPlanes;i++) {
    fTRDTimBin[i]=track.fTRDTimBin[i];
  }

  if (fTRDnSlices) {
    fTRDslices=new Double32_t[fTRDnSlices];
    for (Int_t i=0; i<fTRDnSlices; i++) fTRDslices[i]=track.fTRDslices[i];
  }

  if (track.fDetectorPID) fDetectorPID = new AliDetectorPID(*track.fDetectorPID);

  if (track.fTRDr) {
    fTRDr  = new Double32_t[AliPID::kSPECIES];
    for (Int_t i=AliPID::kSPECIES;i--;) fTRDr[i]=track.fTRDr[i]; 
  }

  if (track.fTOFr) {
    fTOFr = new Double32_t[AliPID::kSPECIES];
    for (Int_t i=AliPID::kSPECIES;i--;) fTOFr[i]=track.fTOFr[i];
  }

  if(track.fTOFLabel){
    if(!fTOFLabel) fTOFLabel = new Int_t[3];
    for (Int_t i=0;i<3;i++) fTOFLabel[i]=track.fTOFLabel[i];
  }

  for (Int_t i=0;i<10;i++) fTOFInfo[i]=track.fTOFInfo[i];
  for (Int_t i=0;i<12;i++) fITSModule[i]=track.fITSModule[i];
  
  if (track.fHMPIDr) {
    fHMPIDr = new Double32_t[AliPID::kSPECIES];
    for (Int_t i=AliPID::kSPECIES;i--;) fHMPIDr[i]=track.fHMPIDr[i];
  }

  if (track.fCp) fCp=new AliExternalTrackParam(*track.fCp);
  if (track.fIp) fIp=new AliExternalTrackParam(*track.fIp);
  if (track.fTPCInner) fTPCInner=new AliExternalTrackParam(*track.fTPCInner);
  if (track.fOp) fOp=new AliExternalTrackParam(*track.fOp);
  if (track.fHMPIDp) fHMPIDp=new AliExternalTrackParam(*track.fHMPIDp);
  if (track.fTPCdEdxInfo) fTPCdEdxInfo = new AliTPCdEdxInfo(*track.fTPCdEdxInfo);

  
  if (track.fFriendTrack) fFriendTrack=new AliESDfriendTrack(*(track.fFriendTrack));

  if(fNtofClusters > 0){
    fTOFcluster = new Int_t[fNtofClusters];
        for(Int_t i=0;i < fNtofClusters;i++) fTOFcluster[i] = track.fTOFcluster[i];
  }
}

//_______________________________________________________________________
AliESDtrack::AliESDtrack(const AliVTrack *track) : 
  AliExternalTrackParam(track),
  fCp(0),
  fIp(0),
  fTPCInner(0),
  fOp(0),
  fHMPIDp(0),  
  fFriendTrack(0),
  fTPCFitMap(159),//number of padrows
  fTPCClusterMap(159),//number of padrows
  fTPCSharedMap(159),//number of padrows
  fFrTrackID(0),
  fFlags(0),
  fID(),
  fLabel(0),
  fITSLabel(0),
  fTPCLabel(0),
  fTRDLabel(0),
  fTOFLabel(NULL),
  fTOFCalChannel(-1),
  fTOFindex(-1),
  fHMPIDqn(0),
  fHMPIDcluIdx(-1),
  fCaloIndex(kEMCALNoMatch),
  fR(0),
  fITSr(0),
  fTPCr(0),
  fTRDr(0),
  fTOFr(0),
  fHMPIDr(0),
  fHMPIDtrkTheta(0),
  fHMPIDtrkPhi(0),
  fHMPIDsignal(0),
  fTrackTime(NULL),
  fTrackLength(0),
  fdTPC(0),fzTPC(0),
  fCddTPC(0),fCdzTPC(0),fCzzTPC(0),
  fCchi2TPC(0),
  fD(0),fZ(0),
  fCdd(0),fCdz(0),fCzz(0),
  fCchi2(0),
  fITSchi2(0),
  fTPCchi2(0),
  fTPCchi2Iter1(0),
  fTRDchi2(0),
  fTOFchi2(0),
  fHMPIDchi2(0),
  fGlobalChi2(0),
  fITSsignal(0),
  fITSsignalTuned(0),
  fTPCsignal(0),
  fTPCsignalTuned(0),
  fTPCsignalS(0),
  fTPCdEdxInfo(0),
  fTRDsignal(0),
  fTRDQuality(0),
  fTRDBudget(0),
  fTOFsignal(99999),
  fTOFsignalTuned(99999),
  fTOFsignalToT(99999),
  fTOFsignalRaw(99999),
  fTOFsignalDz(999),
  fTOFsignalDx(999),
  fTOFdeltaBC(999),
  fTOFl0l1(999),
  fCaloDx(0),
  fCaloDz(0),
  fHMPIDtrkX(0),
  fHMPIDtrkY(0),
  fHMPIDmipX(0),
  fHMPIDmipY(0),
  fTPCncls(0),
  fTPCnclsF(0),
  fTPCsignalN(0),
  fTPCnclsIter1(0),
  fTPCnclsFIter1(0),
  fITSncls(0),
  fITSClusterMap(0),
  fITSSharedMap(0),
  fTRDncls(0),
  fTRDncls0(0),
  fTRDntracklets(0),
  fTRDNchamberdEdx(0),
  fTRDNclusterdEdx(0),
  fTRDnSlices(0),
  fTRDslices(0x0),
  fVertexID(-2),  // -2 means an orphan track
  fPIDForTracking(track->GetPIDForTracking()),
  fPIDForTrackingIn(track->GetPIDForTracking()),
  fESDEvent(0),
  fCacheNCrossedRows(-10),
  fCacheChi2TPCConstrainedVsGlobal(-10),
  fCacheChi2TPCConstrainedVsGlobalVertex(0),
  fDetectorPID(0x0),
  fTrackPhiOnEMCal(-999),
  fTrackEtaOnEMCal(-999),
  fTrackPtOnEMCal(-999),
  fNtofClusters(0),
  fTOFcluster(NULL)
{
  //
  // ESD track from AliVTrack.
  // This is not a copy constructor !
  //

  if (track->InheritsFrom("AliExternalTrackParam")) {
     AliError("This is not a copy constructor. Use AliESDtrack(const AliESDtrack &) !");
     AliWarning("Calling the default constructor...");
     AliESDtrack();
     return;
  }

  // Reset all the arrays
  Int_t i;
  for (i=kNITSchi2Std;i--;) fITSchi2Std[i] = 0;
  
  for (i=0; i<3; i++)   { fKinkIndexes[i]=0;}
  for (i=0; i<3; i++)   { fV0Indexes[i]=-1;}
  for (i=0;i<kTRDnPlanes;i++) {
    fTRDTimBin[i]=0;
  }
  for (i=0;i<4;i++) {fITSdEdxSamples[i]=0.;}
  for (i=0;i<4;i++) {fTPCPoints[i]=0;}
  for (i=0;i<10;i++) {fTOFInfo[i]=0;}
  for (i=0;i<12;i++) {fITSModule[i]=-1;}


  // Set ITS cluster map
  fITSClusterMap=track->GetITSClusterMap();
  fITSSharedMap=0;
  fITSchi2=track->GetITSchi2();
  fITSncls=0;
  for(i=0; i<6; i++) {
    if(HasPointOnITSLayer(i)) fITSncls++;
  }

  // Set TPC ncls 
  fTPCncls=track->GetTPCNcls();
  fTPCnclsF=track->GetTPCNclsF();
  // TPC cluster maps
  const TBits* bmap = track->GetTPCClusterMapPtr();
  if (bmap) SetTPCClusterMap(*bmap);
  bmap = GetTPCFitMapPtr();
  if (bmap) SetTPCFitMap(*bmap);
  bmap = GetTPCSharedMapPtr();
  if (bmap) SetTPCSharedMap(*bmap);
  // Set TPC chi2
  fTPCchi2 = track->GetTPCchi2();

  // Impact parameters
  if (track->InheritsFrom("AliAODTrack")) {
    Float_t ip[2], ipCov[3];
    track->GetImpactParameters(ip,ipCov);
    fD = ip[0];
    fZ = ip[1];
    fCdd = ipCov[0];
    fCdz = ipCov[1];
    fCzz = ipCov[2];
  }

  //
  // Set the combined PID
  const Double_t *pid = track->PID();
  if(pid) {
    fR = new Double32_t[AliPID::kSPECIES];
    for (i=AliPID::kSPECIES; i--;) fR[i]=pid[i];
  }
  //
  // calo matched cluster id
  SetEMCALcluster(track->GetEMCALcluster());
  // AliESD track label
  //
  // PID info
  fITSsignal = track->GetITSsignal();
  fITSsignalTuned = track->GetITSsignalTunedOnData();
  double itsdEdx[4];
  track->GetITSdEdxSamples(itsdEdx);
  SetITSdEdxSamples(itsdEdx);
  //
  SetTPCsignal(track->GetTPCsignal(),fTPCsignalS,track->GetTPCsignalN()); // No signalS in AODPi
  
  AliTPCdEdxInfo dEdxInfo;
  if (track->GetTPCdEdxInfo( dEdxInfo )) SetTPCdEdxInfo(new AliTPCdEdxInfo(dEdxInfo));
  //
  SetTRDsignal(track->GetTRDsignal());
  int ntrdsl = track->GetNumberOfTRDslices();
  if (ntrdsl>0) {
    SetNumberOfTRDslices((ntrdsl+2)*kTRDnPlanes);
    for (int ipl=kTRDnPlanes;ipl--;){
      for (int isl=ntrdsl;isl--;) SetTRDslice(track->GetTRDslice(ipl,isl),ipl,isl);
      Double_t sp, p = track->GetTRDmomentum(ipl, &sp);
      SetTRDmomentum(p, ipl, &sp);
    }
  }
  //
  fTRDncls = track->GetTRDncls();
  fTRDntracklets &= 0xff & track->GetTRDntrackletsPID();
  fTRDchi2 = track->GetTRDchi2();
  //
  SetTOFsignal(track->GetTOFsignal());
  Double_t expt[AliPID::kSPECIESC];
  track->GetIntegratedTimes(expt,AliPID::kSPECIESC);
  SetIntegratedTimes(expt);
  //
  SetTrackPhiEtaPtOnEMCal(track->GetTrackPhiOnEMCal(),track->GetTrackEtaOnEMCal(),track->GetTrackPtOnEMCal());
  //
  SetLabel(track->GetLabel());
  // Set the status
  SetStatus(track->GetStatus());
  //
  // Set the ID
  SetID(track->GetID());
  //
}

//_______________________________________________________________________
AliESDtrack::AliESDtrack(TParticle * part) : 
  AliExternalTrackParam(),
  fCp(0),
  fIp(0),
  fTPCInner(0),
  fOp(0),
  fHMPIDp(0),  
  fFriendTrack(0),
  fTPCFitMap(159),//number of padrows
  fTPCClusterMap(159),//number of padrows
  fTPCSharedMap(159),//number of padrows
  fFrTrackID(0),
  fFlags(0),
  fID(0),
  fLabel(0),
  fITSLabel(0),
  fTPCLabel(0),
  fTRDLabel(0),
  fTOFLabel(NULL),
  fTOFCalChannel(-1),
  fTOFindex(-1),
  fHMPIDqn(0),
  fHMPIDcluIdx(-1),
  fCaloIndex(kEMCALNoMatch),
  fR(0),
  fITSr(0),
  fTPCr(0),
  fTRDr(0),
  fTOFr(0),
  fHMPIDr(0),
  fHMPIDtrkTheta(0),
  fHMPIDtrkPhi(0),
  fHMPIDsignal(0),
  fTrackTime(NULL),
  fTrackLength(0),
  fdTPC(0),fzTPC(0),
  fCddTPC(0),fCdzTPC(0),fCzzTPC(0),
  fCchi2TPC(0),
  fD(0),fZ(0),
  fCdd(0),fCdz(0),fCzz(0),
  fCchi2(0),
  fITSchi2(0),
  fTPCchi2(0),
  fTPCchi2Iter1(0),  
  fTRDchi2(0),
  fTOFchi2(0),
  fHMPIDchi2(0),
  fGlobalChi2(0),
  fITSsignal(0),
  fITSsignalTuned(0),
  fTPCsignal(0),
  fTPCsignalTuned(0),
  fTPCsignalS(0),
  fTPCdEdxInfo(0),
  fTRDsignal(0),
  fTRDQuality(0),
  fTRDBudget(0),
  fTOFsignal(99999),
  fTOFsignalTuned(99999),
  fTOFsignalToT(99999),
  fTOFsignalRaw(99999),
  fTOFsignalDz(999),
  fTOFsignalDx(999),
  fTOFdeltaBC(999),
  fTOFl0l1(999),
  fCaloDx(0),
  fCaloDz(0),
  fHMPIDtrkX(0),
  fHMPIDtrkY(0),
  fHMPIDmipX(0),
  fHMPIDmipY(0),
  fTPCncls(0),
  fTPCnclsF(0),
  fTPCsignalN(0),
  fTPCnclsIter1(0),
  fTPCnclsFIter1(0),
  fITSncls(0),
  fITSClusterMap(0),
  fITSSharedMap(0),
  fTRDncls(0),
  fTRDncls0(0),
  fTRDntracklets(0),
  fTRDNchamberdEdx(0),
  fTRDNclusterdEdx(0),
  fTRDnSlices(0),
  fTRDslices(0x0),
  fVertexID(-2),  // -2 means an orphan track
  fPIDForTracking(AliPID::kPion),
  fPIDForTrackingIn(AliPID::kPion),
  fESDEvent(0),
  fCacheNCrossedRows(-10),
  fCacheChi2TPCConstrainedVsGlobal(-10),
  fCacheChi2TPCConstrainedVsGlobalVertex(0),
  fDetectorPID(0x0),
  fTrackPhiOnEMCal(-999),
  fTrackEtaOnEMCal(-999),
  fTrackPtOnEMCal(-999),
  fNtofClusters(0),
  fTOFcluster(NULL)
{
  //
  // ESD track from TParticle
  //

  // Reset all the arrays
  Int_t i;
  for (i=kNITSchi2Std;i--;) fITSchi2Std[i] = 0;
  
  for (i=0; i<3; i++)   { fKinkIndexes[i]=0;}
  for (i=0; i<3; i++)   { fV0Indexes[i]=-1;}
  for (i=0;i<kTRDnPlanes;i++) {
    fTRDTimBin[i]=0;
  }
  for (i=0;i<4;i++) {fITSdEdxSamples[i]=0.;}
  for (i=0;i<4;i++) {fTPCPoints[i]=0;}
  for (i=0;i<10;i++) {fTOFInfo[i]=0;}
  for (i=0;i<12;i++) {fITSModule[i]=-1;}

  // Calculate the AliExternalTrackParam content

  Double_t xref;
  Double_t alpha;
  Double_t param[5];
  Double_t covar[15];

  // Calculate alpha: the rotation angle of the corresponding local system (TPC sector)
  alpha = part->Phi()*180./TMath::Pi();
  if (alpha<0) alpha+= 360.;
  if (alpha>360) alpha -= 360.;

  Int_t sector = (Int_t)(alpha/20.);
  alpha = 10. + 20.*sector;
  alpha /= 180;
  alpha *= TMath::Pi();

  // Covariance matrix: no errors, the parameters are exact
  for (i=0; i<15; i++) covar[i]=0.;

  // Get the vertex of origin and the momentum
  TVector3 ver(part->Vx(),part->Vy(),part->Vz());
  TVector3 mom(part->Px(),part->Py(),part->Pz());

  // Rotate to the local coordinate system (TPC sector)
  ver.RotateZ(-alpha);
  mom.RotateZ(-alpha);

  // X of the referense plane
  xref = ver.X();

  Int_t pdgCode = part->GetPdgCode();

  Double_t charge = 
    TDatabasePDG::Instance()->GetParticle(pdgCode)->Charge();

  param[0] = ver.Y();
  param[1] = ver.Z();
  param[2] = TMath::Sin(mom.Phi());
  param[3] = mom.Pz()/mom.Pt();
  param[4] = TMath::Sign(1/mom.Pt(),charge);

  // Set AliExternalTrackParam
  Set(xref, alpha, param, covar);

  // Set the PID
  Int_t indexPID = 99;
  if (pdgCode<0) pdgCode = -pdgCode;
  for (i=0;i<AliPID::kSPECIESC;i++) if (pdgCode==AliPID::ParticleCode(i)) {indexPID = i; break;}

  if (indexPID < AliPID::kSPECIESC) fPIDForTrackingIn = fPIDForTracking = indexPID;

  // AliESD track label
  SetLabel(part->GetUniqueID());

}

//_______________________________________________________________________
AliESDtrack::~AliESDtrack(){ 
  //
  // This is destructor according Coding Conventrions 
  //
  //printf("Delete track\n");
  delete fIp; 
  delete fTPCInner; 
  delete fOp;
  delete fHMPIDp;
  delete fCp; 
  delete fFriendTrack;
  delete fTPCdEdxInfo;
  if(fTRDnSlices)
    delete[] fTRDslices;

  //Reset cached values - needed for TClonesArray in AliESDInputHandler
  fCacheNCrossedRows = -10.;
  fCacheChi2TPCConstrainedVsGlobal = -10.;
  if(fCacheChi2TPCConstrainedVsGlobalVertex) fCacheChi2TPCConstrainedVsGlobalVertex = 0;

  if(fTOFcluster)
    delete[] fTOFcluster;
  fTOFcluster = NULL;
  fNtofClusters=0;

  delete fDetectorPID;

  delete[] fR;
  delete[] fITSr;
  delete[] fTPCr;
  delete[] fTRDr;
  delete[] fTOFr;
  delete[] fHMPIDr;
  //
  if(fTrackTime) delete[] fTrackTime; 
  if(fTOFLabel) delete[] fTOFLabel;
}

//_______________________________________________________________________
AliESDtrack &AliESDtrack::operator=(const AliESDtrack &source)
{
  // == operator

  if(&source == this) return *this;
  AliExternalTrackParam::operator=(source);

  
  if(source.fCp){
    // we have the trackparam: assign or copy construct
    if(fCp)*fCp = *source.fCp;
    else fCp = new AliExternalTrackParam(*source.fCp);
  }
  else{
    // no track param delete the old one
    delete fCp;
    fCp = 0;
  }

  if(source.fIp){
    // we have the trackparam: assign or copy construct
    if(fIp)*fIp = *source.fIp;
    else fIp = new AliExternalTrackParam(*source.fIp);
  }
  else{
    // no track param delete the old one
    delete fIp;
    fIp = 0;
  }


  if(source.fTPCInner){
    // we have the trackparam: assign or copy construct
    if(fTPCInner) *fTPCInner = *source.fTPCInner;
    else fTPCInner = new AliExternalTrackParam(*source.fTPCInner);
  }
  else{
    // no track param delete the old one
    delete fTPCInner;
    fTPCInner = 0;
  }

  if(source.fTPCdEdxInfo) {
    if(fTPCdEdxInfo) *fTPCdEdxInfo = *source.fTPCdEdxInfo;
    fTPCdEdxInfo = new AliTPCdEdxInfo(*source.fTPCdEdxInfo);
  }

  if(source.fOp){
    // we have the trackparam: assign or copy construct
    if(fOp) *fOp = *source.fOp;
    else fOp = new AliExternalTrackParam(*source.fOp);
  }
  else{
    // no track param delete the old one
    delete fOp;
    fOp = 0;
  }

  
  if(source.fHMPIDp){
    // we have the trackparam: assign or copy construct
    if(fHMPIDp) *fHMPIDp = *source.fHMPIDp;
    else fHMPIDp = new AliExternalTrackParam(*source.fHMPIDp);
  }
  else{
    // no track param delete the old one
    delete fHMPIDp;
    fHMPIDp = 0;
  }

  // copy also the friend track 
  // use copy constructor
  if(source.fFriendTrack){
    // we have the trackparam: assign or copy construct
    delete fFriendTrack; fFriendTrack=new AliESDfriendTrack(*source.fFriendTrack);
  }
  else{
    // no track param delete the old one
    delete fFriendTrack; fFriendTrack= 0;
  }

  fTPCFitMap = source.fTPCFitMap; 
  fTPCClusterMap = source.fTPCClusterMap; 
  fTPCSharedMap  = source.fTPCSharedMap;  
  // the simple stuff
  fFrTrackID = source.fFrTrackID;
  fFlags    = source.fFlags; 
  fID       = source.fID;             
  fLabel    = source.fLabel;
  fITSLabel = source.fITSLabel;
  for(int i = 0; i< 12;++i){
    fITSModule[i] = source.fITSModule[i];
  }
  fTPCLabel = source.fTPCLabel; 
  fTRDLabel = source.fTRDLabel;
  if(source.fTOFLabel){
    if(!fTOFLabel) fTOFLabel = new Int_t[3];
    for(int i = 0; i< 3;++i){
      fTOFLabel[i] = source.fTOFLabel[i];    
    }
  }
  fTOFCalChannel = source.fTOFCalChannel;
  fTOFindex      = source.fTOFindex;
  fHMPIDqn       = source.fHMPIDqn;
  fHMPIDcluIdx   = source.fHMPIDcluIdx; 
  fCaloIndex    = source.fCaloIndex;
  for (int i=kNITSchi2Std;i--;) fITSchi2Std[i] = source.fITSchi2Std[i];
  for(int i = 0; i< 3;++i){
    fKinkIndexes[i] = source.fKinkIndexes[i]; 
    fV0Indexes[i]   = source.fV0Indexes[i]; 
  }

  if (source.fR) {
    if (!fR) fR = new Double32_t[AliPID::kSPECIES]; 
    for (Int_t i=AliPID::kSPECIES;i--;)  fR[i]=source.fR[i];
  }
  else {delete[] fR; fR = 0;}

  if (source.fITSr) {
    if (!fITSr) fITSr = new Double32_t[AliPID::kSPECIES]; 
    for (Int_t i=AliPID::kSPECIES;i--;)  fITSr[i]=source.fITSr[i]; 
  }
  else {delete[] fITSr; fITSr = 0;}
  //
  if (source.fTPCr) {
    if (!fTPCr) fTPCr = new Double32_t[AliPID::kSPECIES]; 
    for (Int_t i=AliPID::kSPECIES;i--;) fTPCr[i]=source.fTPCr[i]; 
  }
  else {delete[] fTPCr; fTPCr = 0;}

  if (source.fTRDr) {
    if (!fTRDr) fTRDr  = new Double32_t[AliPID::kSPECIES];
    for (Int_t i=AliPID::kSPECIES;i--;) fTRDr[i]=source.fTRDr[i]; 
  }
  else {delete[] fTRDr; fTRDr = 0;}

  if (source.fTOFr) {
    if (!fTOFr) fTOFr = new Double32_t[AliPID::kSPECIES];
    for (Int_t i=AliPID::kSPECIES;i--;) fTOFr[i]=source.fTOFr[i];
  }
  else {delete[] fTOFr; fTOFr = 0;}

  if (source.fHMPIDr) {
    if (!fHMPIDr) fHMPIDr = new Double32_t[AliPID::kSPECIES];
    for (Int_t i=AliPID::kSPECIES;i--;) fHMPIDr[i]=source.fHMPIDr[i];
  }
  else {delete[] fHMPIDr; fHMPIDr = 0;}
  
  fPIDForTracking = source.fPIDForTracking;
  fPIDForTrackingIn = source.fPIDForTrackingIn;

  fHMPIDtrkTheta = source.fHMPIDtrkTheta;
  fHMPIDtrkPhi   = source.fHMPIDtrkPhi;
  fHMPIDsignal   = source.fHMPIDsignal; 

  
  if(fTrackTime){
    delete[] fTrackTime;
  }
  if(source.fTrackTime){
    fTrackTime = new Double32_t[AliPID::kSPECIESC];
    for(Int_t i=0;i < AliPID::kSPECIESC;i++)
      fTrackTime[i] = source.fTrackTime[i];  
  }

  fTrackLength   = source. fTrackLength;
  fdTPC  = source.fdTPC; 
  fzTPC  = source.fzTPC; 
  fCddTPC = source.fCddTPC;
  fCdzTPC = source.fCdzTPC;
  fCzzTPC = source.fCzzTPC;
  fCchi2TPC = source.fCchi2TPC;

  fD  = source.fD; 
  fZ  = source.fZ; 
  fCdd = source.fCdd;
  fCdz = source.fCdz;
  fCzz = source.fCzz;
  fCchi2     = source.fCchi2;

  fITSchi2   = source.fITSchi2;             
  fTPCchi2   = source.fTPCchi2;            
  fTPCchi2Iter1   = source.fTPCchi2Iter1;            
  fTRDchi2   = source.fTRDchi2;      
  fTOFchi2   = source.fTOFchi2;      
  fHMPIDchi2 = source.fHMPIDchi2;      

  fGlobalChi2 = source.fGlobalChi2;      

  fITSsignal  = source.fITSsignal;     
  fITSsignalTuned = source.fITSsignalTuned;
  for (Int_t i=0;i<4;i++) {fITSdEdxSamples[i]=source.fITSdEdxSamples[i];}
  fTPCsignal  = source.fTPCsignal;     
  fTPCsignalTuned  = source.fTPCsignalTuned;
  fTPCsignalS = source.fTPCsignalS;    
  for(int i = 0; i< 4;++i){
    fTPCPoints[i] = source.fTPCPoints[i];  
  }
  fTRDsignal = source.fTRDsignal;
  fTRDNchamberdEdx = source.fTRDNchamberdEdx;
  fTRDNclusterdEdx = source.fTRDNclusterdEdx;

  for(int i = 0;i < kTRDnPlanes;++i){
    fTRDTimBin[i] = source.fTRDTimBin[i];   
  }

  if(fTRDnSlices)
    delete[] fTRDslices;
  fTRDslices=0;
  fTRDnSlices=source.fTRDnSlices;
  if (fTRDnSlices) {
    fTRDslices=new Double32_t[fTRDnSlices];
    for(int j = 0;j < fTRDnSlices;++j) fTRDslices[j] = source.fTRDslices[j];
  }

  fTRDQuality =   source.fTRDQuality;     
  fTRDBudget  =   source.fTRDBudget;      
  fTOFsignal  =   source.fTOFsignal;     
  fTOFsignalTuned  = source.fTOFsignalTuned;
  fTOFsignalToT = source.fTOFsignalToT;   
  fTOFsignalRaw = source.fTOFsignalRaw;  
  fTOFsignalDz  = source.fTOFsignalDz;      
  fTOFsignalDx  = source.fTOFsignalDx;      
  fTOFdeltaBC   = source.fTOFdeltaBC;
  fTOFl0l1      = source.fTOFl0l1;
 
  for(int i = 0;i<10;++i){
    fTOFInfo[i] = source.fTOFInfo[i];    
  }

  fHMPIDtrkX = source.fHMPIDtrkX; 
  fHMPIDtrkY = source.fHMPIDtrkY; 
  fHMPIDmipX = source.fHMPIDmipX;
  fHMPIDmipY = source.fHMPIDmipY; 

  fTPCncls    = source.fTPCncls;      
  fTPCnclsF   = source.fTPCnclsF;     
  fTPCsignalN = source.fTPCsignalN;   
  fTPCnclsIter1    = source.fTPCnclsIter1;      
  fTPCnclsFIter1   = source.fTPCnclsFIter1;     

  fITSncls = source.fITSncls;       
  fITSClusterMap = source.fITSClusterMap; 
  fITSSharedMap = source.fITSSharedMap; 
  fTRDncls   = source.fTRDncls;       
  fTRDncls0  = source.fTRDncls0;      
  fTRDntracklets  = source.fTRDntracklets; 
  fVertexID = source.fVertexID;
  fPIDForTracking = source.fPIDForTracking;
  fPIDForTrackingIn = source.fPIDForTrackingIn;

  fCacheNCrossedRows = source.fCacheNCrossedRows;
  fCacheChi2TPCConstrainedVsGlobal = source.fCacheChi2TPCConstrainedVsGlobal;
  fCacheChi2TPCConstrainedVsGlobalVertex = source.fCacheChi2TPCConstrainedVsGlobalVertex;

  delete fDetectorPID;
  fDetectorPID=0x0;
  if (source.fDetectorPID) fDetectorPID = new AliDetectorPID(*source.fDetectorPID);
  
  fTrackPhiOnEMCal= source.fTrackPhiOnEMCal;
  fTrackEtaOnEMCal= source.fTrackEtaOnEMCal;
  fTrackPtOnEMCal= source.fTrackPtOnEMCal;

  if(fTOFcluster){
    delete[] fTOFcluster;
  }
  fNtofClusters = source.fNtofClusters;
  if(fNtofClusters > 0){
    fTOFcluster = new Int_t[fNtofClusters];
        for(Int_t i=0;i < fNtofClusters;i++) fTOFcluster[i] = source.fTOFcluster[i];
  }

  return *this;
}



void AliESDtrack::Copy(TObject &obj) const {
  
  // this overwrites the virtual TOBject::Copy()
  // to allow run time copying without casting
  // in AliESDEvent

  if(this==&obj)return;
  AliESDtrack *robj = dynamic_cast<AliESDtrack*>(&obj);
  if(!robj)return; // not an AliESDtrack
  *robj = *this;

}



void AliESDtrack::AddCalibObject(TObject * object){
  //
  // add calib object to the list
  //
  if (!fFriendTrack) fFriendTrack  = new AliESDfriendTrack;
  if (!fFriendTrack) return;
  fFriendTrack->AddCalibObject(object);
}

TObject *  AliESDtrack::GetCalibObject(Int_t index){
  //
  // return calib objct at given position
  //
  if (!fFriendTrack) return 0;
  return fFriendTrack->GetCalibObject(index);
}


Bool_t AliESDtrack::FillTPCOnlyTrack(AliESDtrack &track){
  
  // Fills the information of the TPC-only first reconstruction pass
  // into the passed ESDtrack object. For consistency fTPCInner is also filled
  // again



  // For data produced before r26675
  // RelateToVertexTPC was not properly called during reco
  // so you'll have to call it again, before FillTPCOnlyTrack
  //  Float_t p[2],cov[3];
  // track->GetImpactParametersTPC(p,cov); 
  // if(p[0]==0&&p[1]==0) // <- Default values
  //  track->RelateToVertexTPC(esd->GetPrimaryVertexTPC(),esd->GetMagneticField(),kVeryBig);
  

  if(!fTPCInner)return kFALSE;

  // fill the TPC track params to the global track parameters
  track.Set(fTPCInner->GetX(),fTPCInner->GetAlpha(),fTPCInner->GetParameter(),fTPCInner->GetCovariance());
  track.fD = fdTPC;
  track.fZ = fzTPC;
  track.fCdd = fCddTPC;
  track.fCdz = fCdzTPC;
  track.fCzz = fCzzTPC;

  // copy the inner params
  if(track.fIp) *track.fIp = *fIp;
  else track.fIp = new AliExternalTrackParam(*fIp);

  // copy the TPCinner parameters
  if(track.fTPCInner) *track.fTPCInner = *fTPCInner;
  else track.fTPCInner = new AliExternalTrackParam(*fTPCInner);
  track.fdTPC   = fdTPC;
  track.fzTPC   = fzTPC;
  track.fCddTPC = fCddTPC;
  track.fCdzTPC = fCdzTPC;
  track.fCzzTPC = fCzzTPC;
  track.fCchi2TPC = fCchi2TPC;

  // copy all other TPC specific parameters

  // replace label by TPC label
  track.fLabel    = fTPCLabel;
  track.fTPCLabel = fTPCLabel;

  track.fTPCchi2 = fTPCchi2; 
  track.fTPCchi2Iter1 = fTPCchi2Iter1; 
  track.fTPCsignal = fTPCsignal;
  track.fTPCsignalTuned = fTPCsignalTuned;
  track.fTPCsignalS = fTPCsignalS;
  for(int i = 0;i<4;++i)track.fTPCPoints[i] = fTPCPoints[i];

  track.fTPCncls    = fTPCncls;     
  track.fTPCnclsF   = fTPCnclsF;     
  track.fTPCsignalN =  fTPCsignalN;
  track.fTPCnclsIter1    = fTPCnclsIter1;     
  track.fTPCnclsFIter1   = fTPCnclsFIter1;     

  // PID 
  if (fTPCr) {
    if (!track.fTPCr) track.fTPCr = new Double32_t[AliPID::kSPECIES];
    for(int i=AliPID::kSPECIES;i--;) track.fTPCr[i] = fTPCr[i];
  }
  //
  if (fR) {
    if (!track.fR) track.fR = new Double32_t[AliPID::kSPECIES];
    for(int i=AliPID::kSPECIES;i--;) track.fR[i] = fR[i];
  }
    
  track.fTPCFitMap = fTPCFitMap;
  track.fTPCClusterMap = fTPCClusterMap;
  track.fTPCSharedMap = fTPCSharedMap;


  // reset the flags
  track.fFlags = kTPCin;
  track.fID    = fID;

  track.fFlags |= fFlags & kTPCpid; //copy the TPCpid status flag
 
  for (Int_t i=0;i<3;i++) track.fKinkIndexes[i] = fKinkIndexes[i];
  
  return kTRUE;
    
}

//_______________________________________________________________________
void AliESDtrack::MakeMiniESDtrack(){
  // Resets everything except
  // fFlags: Reconstruction status flags 
  // fLabel: Track label
  // fID:  Unique ID of the track
  // Impact parameter information
  // fR[AliPID::kSPECIES]: combined "detector response probability"
  // Running track parameters in the base class (AliExternalTrackParam)
  
  fTrackLength = 0;

  if(fTrackTime)
    for (Int_t i=0;i<AliPID::kSPECIESC;i++) fTrackTime[i] = 0;

  // Reset track parameters constrained to the primary vertex
  delete fCp;fCp = 0;

  // Reset track parameters at the inner wall of TPC
  delete fIp;fIp = 0;
  delete fTPCInner;fTPCInner=0;
  // Reset track parameters at the inner wall of the TRD
  delete fOp;fOp = 0;
  // Reset track parameters at the HMPID
  delete fHMPIDp;fHMPIDp = 0;


  // Reset ITS track related information
  fITSchi2 = 0;
  fITSncls = 0;       
  fITSClusterMap=0;
  fITSSharedMap=0;
  fITSsignal = 0;     
  fITSsignalTuned = 0;
  for (Int_t i=0;i<4;i++) fITSdEdxSamples[i] = 0.;
  if (fITSr) for (Int_t i=0;i<AliPID::kSPECIES;i++) fITSr[i]=0; 
  fITSLabel = 0;       

  // Reset TPC related track information
  fTPCchi2 = 0;       
  fTPCchi2Iter1 = 0;       
  fTPCncls = 0;       
  fTPCnclsF = 0;       
  fTPCnclsIter1 = 0;       
  fTPCnclsFIter1 = 0;  
  fTPCFitMap = 0;       
  fTPCClusterMap = 0;  
  fTPCSharedMap = 0;  
  fTPCsignal= 0;      
  fTPCsignalTuned= 0;
  fTPCsignalS= 0;      
  fTPCsignalN= 0;      
  if (fTPCr) for (Int_t i=0;i<AliPID::kSPECIES;i++) fTPCr[i] = 0; 
  fTPCLabel=0;       
  for (Int_t i=0;i<4;i++) fTPCPoints[i] = 0;
  for (Int_t i=0; i<3;i++)   fKinkIndexes[i] = 0;
  for (Int_t i=0; i<3;i++)   fV0Indexes[i] = 0;

  // Reset TRD related track information
  fTRDchi2 = 0;        
  fTRDncls = 0;       
  fTRDncls0 = 0;       
  fTRDsignal = 0;      
  fTRDNchamberdEdx = 0;
  fTRDNclusterdEdx = 0;

  for (Int_t i=0;i<kTRDnPlanes;i++) {
    fTRDTimBin[i]  = 0;
  }
  if (fTRDr) for (Int_t i=0;i<AliPID::kSPECIES;i++) fTRDr[i] = 0; 
  fTRDLabel = 0;       
  fTRDQuality  = 0;
  fTRDntracklets = 0;
  if(fTRDnSlices)
    delete[] fTRDslices;
  fTRDslices=0x0;
  fTRDnSlices=0;
  fTRDBudget  = 0;

  // Reset TOF related track information
  fTOFchi2 = 0;        
  fTOFindex = -1;       
  fTOFsignal = 99999;      
  fTOFCalChannel = -1;
  fTOFsignalToT = 99999;
  fTOFsignalRaw = 99999;
  fTOFsignalDz = 999;
  fTOFsignalDx = 999;
  fTOFdeltaBC = 999;
  fTOFl0l1 = 999;
  if (fTOFr) for (Int_t i=0;i<AliPID::kSPECIES;i++) fTOFr[i] = 0;
  for (Int_t i=0;i<10;i++) fTOFInfo[i] = 0;

  // Reset HMPID related track information
  fHMPIDchi2 = 0;     
  fHMPIDqn = 0;     
  fHMPIDcluIdx = -1;     
  fHMPIDsignal = 0;     
  if (fHMPIDr) for (Int_t i=0;i<AliPID::kSPECIES;i++) fHMPIDr[i] = 0;
  fHMPIDtrkTheta = 0;     
  fHMPIDtrkPhi = 0;      
  fHMPIDtrkX = 0;     
  fHMPIDtrkY = 0;      
  fHMPIDmipX = 0;
  fHMPIDmipY = 0;
  fCaloIndex = kEMCALNoMatch;

  // reset global track chi2
  fGlobalChi2 = 0;

  fVertexID = -2; // an orphan track
  fPIDForTrackingIn = fPIDForTracking = AliPID::kPion;
  //
  delete fFriendTrack; fFriendTrack = 0;
} 

//_______________________________________________________________________
Int_t AliESDtrack::GetPID(Bool_t tpcOnly) const 
{
  // Returns the particle most probable id. For backward compatibility first the prob. arrays
  // will be checked, but normally the GetPIDForTracking will be returned
  Int_t i;
  const Double32_t *prob = 0;
  if (tpcOnly) { // check if TPCpid is valid
    if (!fTPCr) return GetPIDForTracking();
    prob = fTPCr;
    for (i=0; i<AliPID::kSPECIES-1; i++) if (prob[i] != prob[i+1]) break;
    if (i == AliPID::kSPECIES-1) prob = 0; // not valid, try with combined pid
  }
  if (!prob) { // either requested TPCpid is not valid or comb.pid is requested 
    if (!fR) return GetPIDForTracking();
    prob = fR;
    for (i=0; i<AliPID::kSPECIES-1; i++) if (prob[i] != prob[i+1]) break;
    if (i == AliPID::kSPECIES-1) return GetPIDForTracking();  // If all the probabilities are equal, return the pion mass
  }
  //
  Float_t max=0.;
  Int_t k=-1;
  for (i=0; i<AliPID::kSPECIES; i++) if (prob[i]>max) {k=i; max=prob[i];}
  //
  if (k==0) { // dE/dx "crossing points" in the TPC
    /*
    Double_t p=GetP();
    if ((p>0.38)&&(p<0.48))
      if (prob[0]<prob[3]*10.) return AliPID::kKaon;
    if ((p>0.75)&&(p<0.85))
      if (prob[0]<prob[4]*10.) return AliPID::kProton;
    */
    return AliPID::kElectron;
  }
  if (k==1) return AliPID::kMuon; 
  if (k==2||k==-1) return AliPID::kPion;
  if (k==3) return AliPID::kKaon;
  if (k==4) return AliPID::kProton;
  //  AliWarning("Undefined PID !");
  return GetPIDForTracking();
}

//_______________________________________________________________________
Int_t AliESDtrack::GetTOFBunchCrossing(Double_t b, Bool_t pidTPConly) const 
{
  // Returns the number of bunch crossings after trigger (assuming 25ns spacing)
  const double kSpacing = 25; // min interbanch spacing in ns
  if (!IsOn(kTOFout)) return kTOFBCNA;
  double tdif = GetTOFExpTDiff(b,pidTPConly);
  return TMath::Nint(tdif/kSpacing);
  //
}

//_______________________________________________________________________
Double_t AliESDtrack::GetTOFExpTDiff(Double_t b, Bool_t pidTPConly) const 
{
  // Returns the time difference in ns between TOF signal and expected time
  const double kps2ns = 1e-3; // we need ns
  const double kNoInfo = kTOFBCNA*25; // no info
 if (!IsOn(kTOFout)) return kNoInfo; // no info
  //
  double tdif = GetTOFsignal();
  if (IsOn(kTIME)) { // integrated time info is there
    int pid = GetPID(pidTPConly);
    Double_t times[AliPID::kSPECIESC];
    // old esd has only AliPID::kSPECIES times
    GetIntegratedTimes(times,pid>=AliPID::kSPECIES ? AliPID::kSPECIESC : AliPID::kSPECIES); 
    tdif -= times[pid];
  }
  else { // assume integrated time info from TOF radius and momentum
    const double kRTOF = 385.;
    const double kCSpeed = 3.e-2; // cm/ps
    double p = GetP();
    if (p<0.01) return kNoInfo;
    double m = GetMass(pidTPConly);
    double curv = GetC(b);
    double path = TMath::Abs(curv)>kAlmost0 ? // account for curvature
      2./curv*TMath::ASin(kRTOF*curv/2.)*TMath::Sqrt(1.+GetTgl()*GetTgl()) : kRTOF;
    tdif -= path/kCSpeed*TMath::Sqrt(1.+m*m/(p*p));
  }
  return tdif*kps2ns;
}

//_______________________________________________________________________
Double_t AliESDtrack::GetTOFExpTDiffSpec(AliPID::EParticleType specie,Double_t b) const 
{
  // Returns the time difference in ns between TOF signal and expected time for given specii
  const double kps2ns = 1e-3; // we need ns
  const double kNoInfo = kTOFBCNA*25; // no info
 if (!IsOn(kTOFout)) return kNoInfo; // no info
  //
  double tdif = GetTOFsignal();
  if (IsOn(kTIME)) { // integrated time info is there
    Double_t times[AliPID::kSPECIESC];
    // old esd has only AliPID::kSPECIES times
    GetIntegratedTimes(times,int(specie)>=int(AliPID::kSPECIES) ? AliPID::kSPECIESC : AliPID::kSPECIES); 
    tdif -= times[specie];
  }
  else { // assume integrated time info from TOF radius and momentum
    const double kRTOF = 385.;
    const double kCSpeed = 3.e-2; // cm/ps
    double p = GetP();
    if (p<0.01) return kNoInfo;
    double m = GetMass(specie);
    double curv = GetC(b);
    double path = TMath::Abs(curv)>kAlmost0 ? // account for curvature
      2./curv*TMath::ASin(kRTOF*curv/2.)*TMath::Sqrt(1.+GetTgl()*GetTgl()) : kRTOF;
    tdif -= path/kCSpeed*TMath::Sqrt(1.+m*m/(p*p));
  }
  return tdif*kps2ns;
}

//______________________________________________________________________________
Double_t AliESDtrack::M() const
{
  // Returns the assumed mass
  // (the pion mass, if the particle can't be identified properly).
  static Bool_t printerr=kTRUE;
  if (printerr) {
     AliWarning("WARNING !!! ... THIS WILL BE PRINTED JUST ONCE !!!");
     printerr = kFALSE;
     AliWarning("This is the ESD mass. Use it with care !"); 
  }
  return GetMass(); 
}
  
//______________________________________________________________________________
Double_t AliESDtrack::E() const
{
  // Returns the energy of the particle given its assumed mass.
  // Assumes the pion mass if the particle can't be identified properly.
  
  Double_t m = M();
  Double_t p = P();
  return TMath::Sqrt(p*p + m*m);
}

//______________________________________________________________________________
Double_t AliESDtrack::Y() const
{
  // Returns the rapidity of a particle given its assumed mass.
  // Assumes the pion mass if the particle can't be identified properly.
  
  Double_t e = E();
  Double_t pz = Pz();
  if (e != TMath::Abs(pz)) { // energy was not equal to pz
    return 0.5*TMath::Log((e+pz)/(e-pz));
  } else { // energy was equal to pz
    return -999.;
  }
}

//_______________________________________________________________________
Bool_t AliESDtrack::UpdateTrackParams(const AliKalmanTrack *t, ULong64_t flags){
  //
  // This function updates track's running parameters 
  //
  Bool_t rc=kTRUE;

  SetStatus(flags);
  fLabel=t->GetLabel();

  if (t->IsStartedTimeIntegral()) {
    SetStatus(kTIME);
    Double_t times[AliPID::kSPECIESC];
    t->GetIntegratedTimes(times); 
    SetIntegratedTimes(times);
    SetIntegratedLength(t->GetIntegratedLength());
  }

  Set(t->GetX(),t->GetAlpha(),t->GetParameter(),t->GetCovariance());
  if (fFriendTrack) {
    if (flags==kITSout) fFriendTrack->SetITSOut(*t);
    if (flags==kTPCout) fFriendTrack->SetTPCOut(*t);
    if (flags==kTRDrefit) fFriendTrack->SetTRDIn(*t);
  }
  
  switch (flags) {
    
  case kITSin: 
    fITSchi2Std[0] = t->GetChi2();
    //
  case kITSout: 
    fITSchi2Std[1] = t->GetChi2();
  case kITSrefit:
    {
    fITSchi2Std[2] = t->GetChi2();
    fITSClusterMap=0;
    fITSncls=t->GetNumberOfClusters();
    if (fFriendTrack) {
    Int_t indexITS[AliESDfriendTrack::kMaxITScluster];
    for (Int_t i=0;i<AliESDfriendTrack::kMaxITScluster;i++) {
	indexITS[i]=t->GetClusterIndex(i);

	if (i<fITSncls) {
	  Int_t l=(indexITS[i] & 0xf0000000) >> 28;
           SETBIT(fITSClusterMap,l);                 
        }
    }
    fFriendTrack->SetITSIndices(indexITS,AliESDfriendTrack::kMaxITScluster);
    }

    fITSchi2=t->GetChi2();
    fITSsignal=t->GetPIDsignal();
    fITSLabel = t->GetLabel();
    // keep in fOp the parameters outside ITS for ITS stand-alone tracks 
    if (flags==kITSout) { 
      if (!fOp) fOp=new AliExternalTrackParam(*t);
      else 
        fOp->Set(t->GetX(),t->GetAlpha(),t->GetParameter(),t->GetCovariance());
    }   
    }
    break;
    
  case kTPCin: case kTPCrefit:
    {
    fTPCLabel = t->GetLabel();
    if (flags==kTPCin)  {
      fTPCInner=new AliExternalTrackParam(*t); 
      fTPCnclsIter1=t->GetNumberOfClusters();    
      fTPCchi2Iter1=t->GetChi2();
    }
    if (!fIp) fIp=new AliExternalTrackParam(*t);
    else 
      fIp->Set(t->GetX(),t->GetAlpha(),t->GetParameter(),t->GetCovariance());
    }
    // Intentionally no break statement; need to set general TPC variables as well
  case kTPCout:
    {
    if (flags & kTPCout){
      if (!fOp) fOp=new AliExternalTrackParam(*t);
      else 
        fOp->Set(t->GetX(),t->GetAlpha(),t->GetParameter(),t->GetCovariance());
    }
    fTPCncls=t->GetNumberOfClusters();    
    fTPCchi2=t->GetChi2();
    
    if (fFriendTrack) {  // Copy cluster indices
      Int_t indexTPC[AliESDfriendTrack::kMaxTPCcluster];
      for (Int_t i=0;i<AliESDfriendTrack::kMaxTPCcluster;i++)         
	indexTPC[i]=t->GetClusterIndex(i);
      fFriendTrack->SetTPCIndices(indexTPC,AliESDfriendTrack::kMaxTPCcluster);
    }
    fTPCsignal=t->GetPIDsignal();
    }
    break;

  case kTRDin: case kTRDrefit:
    break;
  case kTRDout:
    {
    fTRDLabel = t->GetLabel(); 
    fTRDchi2  = t->GetChi2();
    fTRDncls  = t->GetNumberOfClusters();
    if (fFriendTrack) {
      Int_t indexTRD[AliESDfriendTrack::kMaxTRDcluster];
      for (Int_t i=0;i<AliESDfriendTrack::kMaxTRDcluster;i++) indexTRD[i]=-2;
      for (Int_t i=0;i<6;i++) indexTRD[i]=t->GetTrackletIndex(i);
      fFriendTrack->SetTRDIndices(indexTRD,AliESDfriendTrack::kMaxTRDcluster);
    }    
    
    //commented out by Xianguo
    //fTRDsignal=t->GetPIDsignal();
    }
    break;
  case kTRDbackup:
    if (!fOp) fOp=new AliExternalTrackParam(*t);
    else 
      fOp->Set(t->GetX(),t->GetAlpha(),t->GetParameter(),t->GetCovariance());
    fTRDncls0 = t->GetNumberOfClusters(); 
    break;
  case kTOFin: 
    break;
  case kTOFout: 
    break;
  case kTRDStop:
    break;
  case kHMPIDout:
  if (!fHMPIDp) fHMPIDp=new AliExternalTrackParam(*t);
    else 
      fHMPIDp->Set(t->GetX(),t->GetAlpha(),t->GetParameter(),t->GetCovariance());
    break;
  default: 
    AliError("Wrong flag !");
    return kFALSE;
  }

  return rc;
}

//_______________________________________________________________________
void AliESDtrack::GetExternalParameters(Double_t &x, Double_t p[5]) const {
  //---------------------------------------------------------------------
  // This function returns external representation of the track parameters
  //---------------------------------------------------------------------
  x=GetX();
  for (Int_t i=0; i<5; i++) p[i]=GetParameter()[i];
}

//_______________________________________________________________________
void AliESDtrack::GetExternalCovariance(Double_t cov[15]) const {
  //---------------------------------------------------------------------
  // This function returns external representation of the cov. matrix
  //---------------------------------------------------------------------
  for (Int_t i=0; i<15; i++) cov[i]=AliExternalTrackParam::GetCovariance()[i];
}

//_______________________________________________________________________
Bool_t AliESDtrack::GetConstrainedExternalParameters
                 (Double_t &alpha, Double_t &x, Double_t p[5]) const {
  //---------------------------------------------------------------------
  // This function returns the constrained external track parameters
  //---------------------------------------------------------------------
  if (!fCp) return kFALSE;
  alpha=fCp->GetAlpha();
  x=fCp->GetX();
  for (Int_t i=0; i<5; i++) p[i]=fCp->GetParameter()[i];
  return kTRUE;
}

//_______________________________________________________________________
Bool_t 
AliESDtrack::GetConstrainedExternalCovariance(Double_t c[15]) const {
  //---------------------------------------------------------------------
  // This function returns the constrained external cov. matrix
  //---------------------------------------------------------------------
  if (!fCp) return kFALSE;
  for (Int_t i=0; i<15; i++) c[i]=fCp->GetCovariance()[i];
  return kTRUE;
}

Bool_t
AliESDtrack::GetInnerExternalParameters
                 (Double_t &alpha, Double_t &x, Double_t p[5]) const {
  //---------------------------------------------------------------------
  // This function returns external representation of the track parameters 
  // at the inner layer of TPC
  //---------------------------------------------------------------------
  if (!fIp) return kFALSE;
  alpha=fIp->GetAlpha();
  x=fIp->GetX();
  for (Int_t i=0; i<5; i++) p[i]=fIp->GetParameter()[i];
  return kTRUE;
}

Bool_t 
AliESDtrack::GetInnerExternalCovariance(Double_t cov[15]) const {
 //---------------------------------------------------------------------
 // This function returns external representation of the cov. matrix 
 // at the inner layer of TPC
 //---------------------------------------------------------------------
  if (!fIp) return kFALSE;
  for (Int_t i=0; i<15; i++) cov[i]=fIp->GetCovariance()[i];
  return kTRUE;
}

void 
AliESDtrack::SetOuterParam(const AliExternalTrackParam *p, ULong_t flags) {
  //
  // This is a direct setter for the outer track parameters
  //
  SetStatus(flags);
  if (fOp) delete fOp;
  fOp=new AliExternalTrackParam(*p);
}

void 
AliESDtrack::SetOuterHmpParam(const AliExternalTrackParam *p, ULong_t flags) {
  //
  // This is a direct setter for the outer track parameters
  //
  SetStatus(flags);
  if (fHMPIDp) delete fHMPIDp;
  fHMPIDp=new AliExternalTrackParam(*p);
}

Bool_t 
AliESDtrack::GetOuterExternalParameters
                 (Double_t &alpha, Double_t &x, Double_t p[5]) const {
  //---------------------------------------------------------------------
  // This function returns external representation of the track parameters 
  // at the inner layer of TRD
  //---------------------------------------------------------------------
  if (!fOp) return kFALSE;
  alpha=fOp->GetAlpha();
  x=fOp->GetX();
  for (Int_t i=0; i<5; i++) p[i]=fOp->GetParameter()[i];
  return kTRUE;
}

Bool_t 
AliESDtrack::GetOuterHmpExternalParameters
                 (Double_t &alpha, Double_t &x, Double_t p[5]) const {
  //---------------------------------------------------------------------
  // This function returns external representation of the track parameters 
  // at the inner layer of TRD
  //---------------------------------------------------------------------
  if (!fHMPIDp) return kFALSE;
  alpha=fHMPIDp->GetAlpha();
  x=fHMPIDp->GetX();
  for (Int_t i=0; i<5; i++) p[i]=fHMPIDp->GetParameter()[i];
  return kTRUE;
}

Bool_t 
AliESDtrack::GetOuterExternalCovariance(Double_t cov[15]) const {
 //---------------------------------------------------------------------
 // This function returns external representation of the cov. matrix 
 // at the inner layer of TRD
 //---------------------------------------------------------------------
  if (!fOp) return kFALSE;
  for (Int_t i=0; i<15; i++) cov[i]=fOp->GetCovariance()[i];
  return kTRUE;
}

Bool_t 
AliESDtrack::GetOuterHmpExternalCovariance(Double_t cov[15]) const {
 //---------------------------------------------------------------------
 // This function returns external representation of the cov. matrix 
 // at the inner layer of TRD
 //---------------------------------------------------------------------
  if (!fHMPIDp) return kFALSE;
  for (Int_t i=0; i<15; i++) cov[i]=fHMPIDp->GetCovariance()[i];
  return kTRUE;
}

Int_t AliESDtrack::GetNcls(Int_t idet) const
{
  // Get number of clusters by subdetector index
  //
  Int_t ncls = 0;
  switch(idet){
  case 0:
    ncls = fITSncls;
    break;
  case 1:
    ncls = fTPCncls;
    break;
  case 2:
    ncls = fTRDncls;
    break;
  case 3:
    if (fTOFindex != -1)
      ncls = 1;
    break;
  case 4: //PHOS
    break;
  case 5: //HMPID
    if ((fHMPIDcluIdx >= 0) && (fHMPIDcluIdx < 7000000)) {
      if ((fHMPIDcluIdx%1000000 != 9999) && (fHMPIDcluIdx%1000000 != 99999)) {
	ncls = 1;
      }
    }    
    break;
  default:
    break;
  }
  return ncls;
}

Int_t AliESDtrack::GetClusters(Int_t idet, Int_t *idx) const
{
  // Get cluster index array by subdetector index
  //
  Int_t ncls = 0;
  switch(idet){
  case 0:
    ncls = GetITSclusters(idx);
    break;
  case 1:
    ncls = GetTPCclusters(idx);
    break;
  case 2:
    ncls = GetTRDclusters(idx);
    break;
  case 3:
    if (fTOFindex != -1) {
      idx[0] = fTOFindex;
      ncls = 1;
    }
    break;
  case 4: //PHOS
    break;
  case 5:
    if ((fHMPIDcluIdx >= 0) && (fHMPIDcluIdx < 7000000)) {
      if ((fHMPIDcluIdx%1000000 != 9999) && (fHMPIDcluIdx%1000000 != 99999)) {
	idx[0] = GetHMPIDcluIdx();
	ncls = 1;
      }
    }    
    break;
  case 6: //EMCAL
    break;
  default:
    break;
  }
  return ncls;
}

//_______________________________________________________________________
void AliESDtrack::GetIntegratedTimes(Double_t *times, Int_t nspec) const 
{
  // get integrated time for requested N species
  if (nspec<1) return;
  if(fNtofClusters>0 && GetESDEvent()){
    TClonesArray *tofclArray = GetESDEvent()->GetESDTOFClusters();
    AliESDTOFCluster *tofcl = (AliESDTOFCluster *) tofclArray->At(fTOFcluster[0]);
    //
    for(int i=tofcl->GetNMatchableTracks();i--;){
      if(tofcl->GetTrackIndex(i) == GetID()) {
	for (int j=nspec; j--;) times[j]=tofcl->GetIntegratedTime(j,i);
	return;
      }
    }
  }
  else if(fNtofClusters>0) 
    AliInfo("No AliESDEvent available here!\n");

  // Returns the array with integrated times for each particle hypothesis
  if(fTrackTime)
    for (int i=nspec; i--;) times[i]=fTrackTime[i];
  else
// The line below is wrong since it does not honor the nspec value
// The "times" array may have only AliPID::kSPECIES size, as called by:
// AliESDpidCuts::AcceptTrack()
//    for (int i=AliPID::kSPECIESC; i--;) times[i]=0.0;
    for (int i=nspec; i--;) times[i]=0.0;
}
//_______________________________________________________________________
Double_t AliESDtrack::GetIntegratedLength() const{
  Int_t index = -1;
  if(fNtofClusters>0 && GetESDEvent()){
    TClonesArray *tofclArray = GetESDEvent()->GetESDTOFClusters();
    AliESDTOFCluster *tofcl = (AliESDTOFCluster *) tofclArray->At(fTOFcluster[0]);

    for(Int_t i=0;i < tofcl->GetNMatchableTracks();i++){
      if(tofcl->GetTrackIndex(i) == GetID()) index = i;
    }
    
    if(fNtofClusters>0 && index > -1)
      return tofcl->GetLength(index);
  }
  else if(fNtofClusters>0) AliInfo("No AliESDEvent available here!\n");

  return fTrackLength;
}

//_______________________________________________________________________
void AliESDtrack::SetIntegratedTimes(const Double_t *times) 
{
  // Sets the array with integrated times for each particle hypotesis
  if(!fTrackTime) fTrackTime = new Double32_t[AliPID::kSPECIESC];
  for (int i=AliPID::kSPECIESC; i--;) fTrackTime[i]=times[i];
}

//_______________________________________________________________________
void AliESDtrack::SetITSpid(const Double_t *p) 
{
  // Sets values for the probability of each particle type (in ITS)
  if (!fITSr) fITSr = new Double32_t[AliPID::kSPECIESC];
  SetPIDValues(fITSr,p,AliPID::kSPECIES);
  SetStatus(AliESDtrack::kITSpid);
}

//_______________________________________________________________________
void AliESDtrack::GetITSpid(Double_t *p) const {
  // Gets the probability of each particle type (in ITS)
  for (Int_t i=0; i<AliPID::kSPECIES; i++) p[i] = fITSr ? fITSr[i] : 0;
}

//_______________________________________________________________________
Char_t AliESDtrack::GetITSclusters(Int_t *idx) const {
  //---------------------------------------------------------------------
  // This function returns indices of the assgined ITS clusters 
  //---------------------------------------------------------------------
  if (idx && fFriendTrack) {
    Int_t *index=fFriendTrack->GetITSindices();
    for (Int_t i=0; i<AliESDfriendTrack::kMaxITScluster; i++) {
      if ( (i>=fITSncls) && (i<6) ) idx[i]=-1;
      else {
	if (index) {
	  idx[i]=index[i];
	}
	else idx[i]= -2;
      }
    }
  }
  return fITSncls;
}

//_______________________________________________________________________
Bool_t AliESDtrack::GetITSModuleIndexInfo(Int_t ilayer,Int_t &idet,Int_t &status,
					 Float_t &xloc,Float_t &zloc) const {
  //----------------------------------------------------------------------
  // This function encodes in the module number also the status of cluster association
  // "status" can have the following values: 
  // 1 "found" (cluster is associated), 
  // 2 "dead" (module is dead from OCDB), 
  // 3 "skipped" (module or layer forced to be skipped),
  // 4 "outinz" (track out of z acceptance), 
  // 5 "nocls" (no clusters in the road), 
  // 6 "norefit" (cluster rejected during refit), 
  // 7 "deadzspd" (holes in z in SPD)
  // Also given are the coordinates of the crossing point of track and module
  // (in the local module ref. system)
  // WARNING: THIS METHOD HAS TO BE SYNCHRONIZED WITH AliITStrackV2::GetModuleIndexInfo()!
  //----------------------------------------------------------------------

  if(fITSModule[ilayer]==-1) {
    idet = -1;
    status=0;
    xloc=-99.; zloc=-99.;
    return kFALSE;
  }

  Int_t module = fITSModule[ilayer];

  idet = Int_t(module/1000000);

  module -= idet*1000000;

  status = Int_t(module/100000);

  module -= status*100000;

  Int_t signs = Int_t(module/10000);

  module-=signs*10000;

  Int_t xInt = Int_t(module/100);
  module -= xInt*100;

  Int_t zInt = module;

  if(signs==1) { xInt*=1; zInt*=1; }
  if(signs==2) { xInt*=1; zInt*=-1; }
  if(signs==3) { xInt*=-1; zInt*=1; }
  if(signs==4) { xInt*=-1; zInt*=-1; }

  xloc = 0.1*(Float_t)xInt;
  zloc = 0.1*(Float_t)zInt;

  if(status==4) idet = -1;

  return kTRUE;
}

//_______________________________________________________________________
UShort_t AliESDtrack::GetTPCclusters(Int_t *idx) const {
  //---------------------------------------------------------------------
  // This function returns indices of the assgined ITS clusters 
  //---------------------------------------------------------------------
  if (idx && fFriendTrack) {
    Int_t *index=fFriendTrack->GetTPCindices();

    if (index){
      memcpy(idx,index,sizeof(int)*AliESDfriendTrack::kMaxTPCcluster);
      //RS for (Int_t i=0; i<AliESDfriendTrack::kMaxTPCcluster; i++) idx[i]=index[i];
    }
    else {
      for (Int_t i=0; i<AliESDfriendTrack::kMaxTPCcluster; i++) idx[i]=-2;
    }
  }
  return fTPCncls;
}

//_______________________________________________________________________
Float_t AliESDtrack::GetTPCCrossedRows() const
{
  // This function calls GetTPCClusterInfo with some default parameters which are used in the track selection and caches the outcome
  // because GetTPCClusterInfo is quite time-consuming
  
  if (fCacheNCrossedRows > -1)
    return fCacheNCrossedRows;
  
  fCacheNCrossedRows = GetTPCClusterInfo(2, 1);
  return fCacheNCrossedRows;
}

//_______________________________________________________________________
Float_t AliESDtrack::GetTPCClusterInfo(Int_t nNeighbours/*=3*/, Int_t type/*=0*/, Int_t row0, Int_t row1, Int_t bitType ) const
{
  //
  // TPC cluster information
  // type 0: get fraction of found/findable clusters with neighbourhood definition
  //      1: findable clusters with neighbourhood definition
  //      2: found clusters
  //      3: get fraction of found/findable clusters with neighbourhood definition - requiring before and after row
  //      4: findable clusters with neighbourhood definition - before and after row
  // bitType:
  //      0 - all cluster used
  //      1 - clusters  used for the kalman update
  // definition of findable clusters:
  //            a cluster is defined as findable if there is another cluster
  //           within +- nNeighbours pad rows. The idea is to overcome threshold
  //           effects with a very simple algorithm.
  //

  
  Int_t found=0;
  Int_t findable=0;
  Int_t last=-nNeighbours;
  const TBits & clusterMap = (bitType%2==0) ? fTPCClusterMap : fTPCFitMap;

  
  Int_t upperBound=clusterMap.GetNbits();
  if (upperBound>row1) upperBound=row1;
  if (type>=3){ // requires cluster before and after
    for (Int_t i=row0; i<upperBound; ++i){
      Int_t beforeAfter=0;
      if (clusterMap[i]) {
	last=i;
	++found;
	++findable;
	continue;
      }
      if ((i-last)<=nNeighbours) {
	++beforeAfter;
      }
      //look to nNeighbours after
      for (Int_t j=i+1; j<i+1+nNeighbours; ++j){
	if (clusterMap[j]){
	  ++beforeAfter;
	  break;
	}
      }
      if (beforeAfter>1) ++findable;
    }
    if (type==3) return Float_t(found)/Float_t(TMath::Max(findable,1));
    if (type==4) return findable;
    return 0;
  }

  for (Int_t i=row0; i<upperBound; ++i){
    //look to current row
    if (clusterMap[i]) {
      last=i;
      ++found;
      ++findable;
      continue;
    }
    //look to nNeighbours before
    if ((i-last)<=nNeighbours) {
      ++findable;
      continue;
    }
    //look to nNeighbours after
    for (Int_t j=i+1; j<i+1+nNeighbours; ++j){
      if (clusterMap[j]){
        ++findable;
        break;
      }
    }
  }
  if (type==2) return found;
  if (type==1) return findable;
  
  if (type==0){
    Float_t fraction=0;
    if (findable>0) 
      fraction=(Float_t)found/(Float_t)findable;
    else 
      fraction=0;
    return fraction;
  }  
  return 0;  // undefined type - default value
}

//_______________________________________________________________________
Float_t AliESDtrack::GetTPCClusterDensity(Int_t nNeighbours/*=3*/, Int_t type/*=0*/, Int_t row0, Int_t row1, Int_t bitType ) const
{
  //
  // TPC cluster density -  only rows where signal before and after given row are used
  //                     -  slower function
  // type 0: get fraction of found/findable clusters with neighbourhood definition
  //      1: findable clusters with neighbourhood definition
  //      2: found clusters
  // bitType:
  //      0 - all cluster used
  //      1 - clusters  used for the kalman update
  // definition of findable clusters:
  //            a cluster is defined as findable if there is another cluster
  //           within +- nNeighbours pad rows. The idea is to overcome threshold
  //           effects with a very simple algorithm.
  //  
  Int_t found=0;
  Int_t findable=0;
  //  Int_t last=-nNeighbours;
  const TBits & clusterMap = (bitType%2==0) ? fTPCClusterMap : fTPCFitMap;
  Int_t upperBound=clusterMap.GetNbits();
  if (upperBound>row1) upperBound=row1;
  for (Int_t i=row0; i<upperBound; ++i){
    Bool_t isUp=kFALSE;
    Bool_t isDown=kFALSE;
    for (Int_t idelta=1; idelta<=nNeighbours; idelta++){
      if (i-idelta>=0 && clusterMap[i-idelta]) isDown=kTRUE;
      if (i+idelta<upperBound && clusterMap[i+idelta]) isUp=kTRUE;
    }
    if (isUp&&isDown){
      ++findable;
      if (clusterMap[i]) ++found;
    }
  }
  if (type==2) return found;
  if (type==1) return findable;
  
  if (type==0){
    Float_t fraction=0;
    if (findable>0) 
      fraction=(Float_t)found/(Float_t)findable;
    else 
      fraction=0;
    return fraction;
  }  
  return 0;  // undefined type - default value
}




//_______________________________________________________________________
Double_t AliESDtrack::GetTPCdensity(Int_t row0, Int_t row1) const{
  //
  // GetDensity of the clusters on given region between row0 and row1
  // Dead zone effect takin into acoount
  //
  if (!fFriendTrack) return 0.0;
  Int_t good  = 0;
  Int_t found = 0;
  //  
  Int_t *index=fFriendTrack->GetTPCindices();
  for (Int_t i=row0;i<=row1;i++){     
    Int_t idx = index[i];
    if (idx!=-1)  good++;             // track outside of dead zone
    if (idx>0)    found++;
  }
  Float_t density=0.5;
  if (good>TMath::Max((row1-row0)*0.5,0.0)) density = Float_t(found)/Float_t(good);
  return density;
}

//_______________________________________________________________________
void AliESDtrack::SetTPCpid(const Double_t *p) {  
  // Sets values for the probability of each particle type (in TPC)
  if (!fTPCr) fTPCr = new Double32_t[AliPID::kSPECIES];
  SetPIDValues(fTPCr,p,AliPID::kSPECIES);
  SetStatus(AliESDtrack::kTPCpid);
}

//_______________________________________________________________________
void AliESDtrack::GetTPCpid(Double_t *p) const {
  // Gets the probability of each particle type (in TPC)
  for (Int_t i=0; i<AliPID::kSPECIES; i++) p[i] = fTPCr ? fTPCr[i] : 0;
}

//_______________________________________________________________________
UChar_t AliESDtrack::GetTRDclusters(Int_t *idx) const {
  //---------------------------------------------------------------------
  // This function returns indices of the assgined TRD clusters 
  //---------------------------------------------------------------------
  if (idx && fFriendTrack) {
    Int_t *index=fFriendTrack->GetTRDindices();

    if (index) {
      for (Int_t i=0; i<AliESDfriendTrack::kMaxTRDcluster; i++) idx[i]=index[i];
    }
    else {
      for (Int_t i=0; i<AliESDfriendTrack::kMaxTRDcluster; i++) idx[i]=-2;
    }
  }
  return fTRDncls;
}

//_______________________________________________________________________
UChar_t AliESDtrack::GetTRDtracklets(Int_t *idx) const {
//
// This function returns the number of TRD tracklets used in tracking
// and it fills the indices of these tracklets in the array "idx" as they 
// are registered in the TRD track list. 
// 
// Caution :
//   1. The idx array has to be allocated with a size >= AliESDtrack::kTRDnPlanes
//   2. The idx array store not only the index but also the layer of the tracklet. 
//      Therefore tracks with TRD gaps contain default values for indices [-1] 

  if (!fFriendTrack) return 0;
  if (!idx) return GetTRDntracklets();
  Int_t *index=fFriendTrack->GetTRDindices();
  Int_t n = 0;
  for (Int_t i=0; i<kTRDnPlanes; i++){ 
    if (index){
      if(index[i]>=0) n++;
      idx[i]=index[i];
    }
    else idx[i] = -2;
  }
  return n;
}

//_______________________________________________________________________
void AliESDtrack::SetTRDpid(const Double_t *p) {  
  // Sets values for the probability of each particle type (in TRD)
  if (!fTRDr) fTRDr = new Double32_t[AliPID::kSPECIES];
  SetPIDValues(fTRDr,p,AliPID::kSPECIES);
  SetStatus(AliESDtrack::kTRDpid);
}

//_______________________________________________________________________
void AliESDtrack::GetTRDpid(Double_t *p) const {
  // Gets the probability of each particle type (in TRD)
  for (Int_t i=0; i<AliPID::kSPECIES; i++) p[i] = fTRDr ? fTRDr[i]:0;
}

//_______________________________________________________________________
void    AliESDtrack::SetTRDpid(Int_t iSpecies, Float_t p)
{
  // Sets the probability of particle type iSpecies to p (in TRD)
  if (!fTRDr) {
    fTRDr = new Double32_t[AliPID::kSPECIES];
    for (Int_t i=AliPID::kSPECIES; i--;) fTRDr[i] = 0;
  }
  fTRDr[iSpecies] = p;
}

Double_t AliESDtrack::GetTRDpid(Int_t iSpecies) const
{
  // Returns the probability of particle type iSpecies (in TRD)
  return fTRDr ? fTRDr[iSpecies] : 0;
}

//____________________________________________________
Int_t AliESDtrack::GetNumberOfTRDslices() const 
{
  // built in backward compatibility
  Int_t idx = fTRDnSlices - (kTRDnPlanes<<1);
  return idx<18 ? fTRDnSlices/kTRDnPlanes : idx/kTRDnPlanes;
}

//____________________________________________________
Double_t AliESDtrack::GetTRDmomentum(Int_t plane, Double_t *sp) const
{
//Returns momentum estimation and optional its error (sp)
// in TRD layer "plane".

  if (!fTRDnSlices) {
    AliDebug(2, "No TRD info allocated for this track.");
    return -1.;
  }
  if ((plane<0) || (plane>=kTRDnPlanes)) {
    AliWarning(Form("Request for TRD plane[%d] outside range.", plane)); 
    return -1.;
  }

  Int_t idx = fTRDnSlices-(kTRDnPlanes<<1)+plane;
  // Protection for backward compatibility
  if(idx<(GetNumberOfTRDslices()*kTRDnPlanes)) return -1.;

  if(sp) (*sp) = fTRDslices[idx+kTRDnPlanes];
  return fTRDslices[idx];
}

//____________________________________________________
Double_t  AliESDtrack::GetTRDslice(Int_t plane, Int_t slice) const {
  //Gets the charge from the slice of the plane

  if(!fTRDslices) {
    //AliError("No TRD slices allocated for this track !");
    return -1.;
  }
  if ((plane<0) || (plane>=kTRDnPlanes)) {
    AliError("Info for TRD plane not available !");
    return -1.;
  }
  Int_t ns=GetNumberOfTRDslices();
  if ((slice<-1) || (slice>=ns)) {
    //AliError("Wrong TRD slice !");  
    return -1.;
  }

  if(slice>=0) return fTRDslices[plane*ns + slice];

  // return average of the dEdx measurements
  Double_t q=0.; Double32_t *s = &fTRDslices[plane*ns];
  for (Int_t i=0; i<ns; i++, s++) if((*s)>0.) q+=(*s);
  return q/ns;
}

//____________________________________________________
void  AliESDtrack::SetNumberOfTRDslices(Int_t n) {
  //Sets the number of slices used for PID 
  if (fTRDnSlices) return;

  fTRDnSlices=n;
  fTRDslices=new Double32_t[fTRDnSlices];
  
  // set-up correctly the allocated memory
  memset(fTRDslices, 0, n*sizeof(Double32_t));
  for (Int_t i=GetNumberOfTRDslices(); i--;) fTRDslices[i]=-1.;
}

//____________________________________________________
void  AliESDtrack::SetTRDslice(Double_t q, Int_t plane, Int_t slice) {
  //Sets the charge q in the slice of the plane
  if(!fTRDslices) {
    AliError("No TRD slices allocated for this track !");
    return;
  }
  if ((plane<0) || (plane>=kTRDnPlanes)) {
    AliError("Info for TRD plane not allocated !");
    return;
  }
  Int_t ns=GetNumberOfTRDslices();
  if ((slice<0) || (slice>=ns)) {
    AliError(Form("Wrong TRD slice %d/%d, NSlices=%d",plane,slice,ns));
    return;
  }
  Int_t n=plane*ns + slice;
  fTRDslices[n]=q;
}


//____________________________________________________
void AliESDtrack::SetTRDmomentum(Double_t p, Int_t plane, Double_t *sp)
{
  if(!fTRDslices) {
    AliError("No TRD slices allocated for this track !");
    return;
  }
  if ((plane<0) || (plane>=kTRDnPlanes)) {
    AliError("Info for TRD plane not allocated !");
    return;
  }

  Int_t idx = fTRDnSlices-(kTRDnPlanes<<1)+plane;
  // Protection for backward compatibility
  if(idx<GetNumberOfTRDslices()*kTRDnPlanes) return;

  if(sp) fTRDslices[idx+kTRDnPlanes] = (*sp);
  fTRDslices[idx] = p;
}


//_______________________________________________________________________
void AliESDtrack::SetTOFpid(const Double_t *p) {  
  // Sets the probability of each particle type (in TOF)
  if (!fTOFr) fTOFr = new Double32_t[AliPID::kSPECIES];
  SetPIDValues(fTOFr,p,AliPID::kSPECIES);
  SetStatus(AliESDtrack::kTOFpid);
}

//_______________________________________________________________________
void AliESDtrack::SetTOFLabel(const Int_t *p) {  

  if(fNtofClusters>0){
    TClonesArray *tofclArray = GetESDEvent()->GetESDTOFClusters();
    AliESDTOFCluster *tofcl = (AliESDTOFCluster *) tofclArray->At(fTOFcluster[0]);
    AliESDTOFHit* hit = tofcl->GetTOFHit(0);

    if(hit) hit->SetTOFLabel(p);
  }
  else{
    // Sets  (in TOF)
    if(!fTOFLabel) fTOFLabel = new Int_t[3]; 
    for (Int_t i=0; i<3; i++) fTOFLabel[i]=p[i];
  }
}

//_______________________________________________________________________
void AliESDtrack::GetTOFpid(Double_t *p) const {
  // Gets probabilities of each particle type (in TOF)
  for (Int_t i=0; i<AliPID::kSPECIES; i++) p[i] = fTOFr ? fTOFr[i]:0;
}

//_______________________________________________________________________
void AliESDtrack::GetTOFLabel(Int_t *p) const {
  // Gets (in TOF)
  if(fNtofClusters>0){
    TClonesArray *tofclArray = GetESDEvent()->GetESDTOFClusters();
    AliESDTOFCluster *tofcl = (AliESDTOFCluster *) tofclArray->At(fTOFcluster[0]);

    for (Int_t i=0; i<3; i++) p[i]=tofcl->GetLabel(i);
  }
  else{
    if(fTOFLabel) for (Int_t i=0; i<3; i++) p[i]=fTOFLabel[i];
    else for (int i=3;i--;) p[i] = -1;
  }
}

//_______________________________________________________________________
void AliESDtrack::GetTOFInfo(Float_t *info) const {
  // Gets (in TOF)
  for (Int_t i=0; i<10; i++) info[i]=fTOFInfo[i];
}

//_______________________________________________________________________
void AliESDtrack::SetTOFInfo(Float_t*info) {
  // Gets (in TOF)
  for (Int_t i=0; i<10; i++) fTOFInfo[i]=info[i];
}



//_______________________________________________________________________
void AliESDtrack::SetHMPIDpid(const Double_t *p) {  
  // Sets the probability of each particle type (in HMPID)
  if (!fHMPIDr) fHMPIDr = new Double32_t[AliPID::kSPECIES];
  SetPIDValues(fHMPIDr,p,AliPID::kSPECIES);
  SetStatus(AliESDtrack::kHMPIDpid);
}

//_______________________________________________________________________
void  AliESDtrack::SetTPCdEdxInfo(AliTPCdEdxInfo * dEdxInfo){ 
  if(fTPCdEdxInfo) delete fTPCdEdxInfo;
  fTPCdEdxInfo = dEdxInfo; 
}

//_______________________________________________________________________
void AliESDtrack::GetHMPIDpid(Double_t *p) const {
  // Gets probabilities of each particle type (in HMPID)
  for (Int_t i=0; i<AliPID::kSPECIES; i++) p[i] = fHMPIDr ? fHMPIDr[i]:0;
}



//_______________________________________________________________________
void AliESDtrack::SetESDpid(const Double_t *p) {  
  // Sets the probability of each particle type for the ESD track
  if (!fR) fR = new Double32_t[AliPID::kSPECIES];
  SetPIDValues(fR,p,AliPID::kSPECIES);
  SetStatus(AliESDtrack::kESDpid);
}

//_______________________________________________________________________
void AliESDtrack::GetESDpid(Double_t *p) const {
  // Gets probability of each particle type for the ESD track
  for (Int_t i=0; i<AliPID::kSPECIES; i++) p[i] = fR ? fR[i]:0;
}

//_______________________________________________________________________
Bool_t AliESDtrack::RelateToVVertexTPC(const AliVVertex *vtx, 
Double_t b, Double_t maxd, AliExternalTrackParam *cParam) {
  //
  // Try to relate the TPC-only track parameters to the vertex "vtx", 
  // if the (rough) transverse impact parameter is not bigger then "maxd". 
  //            Magnetic field is "b" (kG).
  //
  // a) The TPC-only paramters are extapolated to the DCA to the vertex.
  // b) The impact parameters and their covariance matrix are calculated.
  // c) An attempt to constrain the TPC-only params to the vertex is done.
  //    The constrained params are returned via "cParam".
  //
  // In the case of success, the returned value is kTRUE
  // otherwise, it's kFALSE)
  // 

  if (!fTPCInner) return kFALSE;
  if (!vtx) return kFALSE;

  Double_t dz[2],cov[3];
  if (!fTPCInner->PropagateToDCA(vtx, b, maxd, dz, cov)) return kFALSE;

  fdTPC = dz[0];
  fzTPC = dz[1];  
  fCddTPC = cov[0];
  fCdzTPC = cov[1];
  fCzzTPC = cov[2];
  
  Double_t covar[6]; vtx->GetCovarianceMatrix(covar);
  Double_t p[2]={GetParameter()[0]-dz[0],GetParameter()[1]-dz[1]};
  Double_t c[3]={covar[2],0.,covar[5]};

  Double_t chi2=GetPredictedChi2(p,c);
  if (chi2>kVeryBig) return kFALSE;

  fCchi2TPC=chi2;

  if (!cParam) return kTRUE;

  *cParam = *fTPCInner;
  if (!cParam->Update(p,c)) return kFALSE;

  return kTRUE;
}

//_______________________________________________________________________
Bool_t AliESDtrack::RelateToVVertexTPCBxByBz(const AliVVertex *vtx, 
Double_t b[3], Double_t maxd, AliExternalTrackParam *cParam) {
  //
  // Try to relate the TPC-only track parameters to the vertex "vtx", 
  // if the (rough) transverse impact parameter is not bigger then "maxd". 
  //
  // All three components of the magnetic field ,"b[3]" (kG), 
  // are taken into account.
  //
  // a) The TPC-only paramters are extapolated to the DCA to the vertex.
  // b) The impact parameters and their covariance matrix are calculated.
  // c) An attempt to constrain the TPC-only params to the vertex is done.
  //    The constrained params are returned via "cParam".
  //
  // In the case of success, the returned value is kTRUE
  // otherwise, it's kFALSE)
  // 

  if (!fTPCInner) return kFALSE;
  if (!vtx) return kFALSE;

  Double_t dz[2],cov[3];
  if (!fTPCInner->PropagateToDCABxByBz(vtx, b, maxd, dz, cov)) return kFALSE;

  fdTPC = dz[0];
  fzTPC = dz[1];  
  fCddTPC = cov[0];
  fCdzTPC = cov[1];
  fCzzTPC = cov[2];
  
  Double_t covar[6]; vtx->GetCovarianceMatrix(covar);
  Double_t p[2]={GetParameter()[0]-dz[0],GetParameter()[1]-dz[1]};
  Double_t c[3]={covar[2],0.,covar[5]};

  Double_t chi2=GetPredictedChi2(p,c);
  if (chi2>kVeryBig) return kFALSE;

  fCchi2TPC=chi2;

  if (!cParam) return kTRUE;

  *cParam = *fTPCInner;
  if (!cParam->Update(p,c)) return kFALSE;

  return kTRUE;
}

//_______________________________________________________________________
Bool_t AliESDtrack::RelateToVVertex(const AliVVertex *vtx, 
Double_t b, Double_t maxd, AliExternalTrackParam *cParam) {
  //
  // Try to relate this track to the vertex "vtx", 
  // if the (rough) transverse impact parameter is not bigger then "maxd". 
  //            Magnetic field is "b" (kG).
  //
  // a) The track gets extapolated to the DCA to the vertex.
  // b) The impact parameters and their covariance matrix are calculated.
  // c) An attempt to constrain this track to the vertex is done.
  //    The constrained params are returned via "cParam".
  //
  // In the case of success, the returned value is kTRUE
  // (otherwise, it's kFALSE)
  //  

  if (!vtx) return kFALSE;

  Double_t dz[2],cov[3];
  if (!PropagateToDCA(vtx, b, maxd, dz, cov)) return kFALSE;

  fD = dz[0];
  fZ = dz[1];  
  fCdd = cov[0];
  fCdz = cov[1];
  fCzz = cov[2];
  
  Double_t covar[6]; vtx->GetCovarianceMatrix(covar);
  Double_t p[2]={GetParameter()[0]-dz[0],GetParameter()[1]-dz[1]};
  Double_t c[3]={covar[2],0.,covar[5]};

  Double_t chi2=GetPredictedChi2(p,c);
  if (chi2>kVeryBig) return kFALSE;

  fCchi2=chi2;


  //--- Could now these lines be removed ? ---
  delete fCp;
  fCp=new AliExternalTrackParam(*this);  

  if (!fCp->Update(p,c)) {delete fCp; fCp=0; return kFALSE;}
  //----------------------------------------

  // fVertexID = vtx->GetID(); //No GetID() in AliVVertex

  if (!cParam) return kTRUE;

  *cParam = *this;
  if (!cParam->Update(p,c)) return kFALSE; 

  return kTRUE;
}

//_______________________________________________________________________
Bool_t AliESDtrack::RelateToVVertexBxByBz(const AliVVertex *vtx, 
Double_t b[3], Double_t maxd, AliExternalTrackParam *cParam) {
  //
  // Try to relate this track to the vertex "vtx", 
  // if the (rough) transverse impact parameter is not bigger then "maxd". 
  //            Magnetic field is "b" (kG).
  //
  // a) The track gets extapolated to the DCA to the vertex.
  // b) The impact parameters and their covariance matrix are calculated.
  // c) An attempt to constrain this track to the vertex is done.
  //    The constrained params are returned via "cParam".
  //
  // In the case of success, the returned value is kTRUE
  // (otherwise, it's kFALSE)
  //  

  if (!vtx) return kFALSE;

  Double_t dz[2],cov[3];
  if (!PropagateToDCABxByBz(vtx, b, maxd, dz, cov)) return kFALSE;

  fD = dz[0];
  fZ = dz[1];  
  fCdd = cov[0];
  fCdz = cov[1];
  fCzz = cov[2];
  
  Double_t covar[6]; vtx->GetCovarianceMatrix(covar);
  Double_t p[2]={GetParameter()[0]-dz[0],GetParameter()[1]-dz[1]};
  Double_t c[3]={covar[2],0.,covar[5]};

  Double_t chi2=GetPredictedChi2(p,c);
  if (chi2>kVeryBig) return kFALSE;

  fCchi2=chi2;


  //--- Could now these lines be removed ? ---
  delete fCp;
  fCp=new AliExternalTrackParam(*this);  

  if (!fCp->Update(p,c)) {delete fCp; fCp=0; return kFALSE;}
  //----------------------------------------

  // fVertexID = vtx->GetID(); // No GetID in AliVVertex

  if (!cParam) return kTRUE;

  *cParam = *this;
  if (!cParam->Update(p,c)) return kFALSE; 

  return kTRUE;
}










//_______________________________________________________________________
Bool_t AliESDtrack::RelateToVertexTPC(const AliESDVertex *vtx, 
Double_t b, Double_t maxd, AliExternalTrackParam *cParam) {
  //
  // Try to relate the TPC-only track parameters to the vertex "vtx", 
  // if the (rough) transverse impact parameter is not bigger then "maxd". 
  //            Magnetic field is "b" (kG).
  //
  // a) The TPC-only paramters are extapolated to the DCA to the vertex.
  // b) The impact parameters and their covariance matrix are calculated.
  // c) An attempt to constrain the TPC-only params to the vertex is done.
  //    The constrained params are returned via "cParam".
  //
  // In the case of success, the returned value is kTRUE
  // otherwise, it's kFALSE)
  // 

  if (!fTPCInner) return kFALSE;
  if (!vtx) return kFALSE;

  Double_t dz[2],cov[3];
  if (!fTPCInner->PropagateToDCA(vtx, b, maxd, dz, cov)) return kFALSE;

  fdTPC = dz[0];
  fzTPC = dz[1];  
  fCddTPC = cov[0];
  fCdzTPC = cov[1];
  fCzzTPC = cov[2];
  
  Double_t covar[6]; vtx->GetCovMatrix(covar);
  Double_t p[2]={GetParameter()[0]-dz[0],GetParameter()[1]-dz[1]};
  Double_t c[3]={covar[2],0.,covar[5]};

  Double_t chi2=GetPredictedChi2(p,c);
  if (chi2>kVeryBig) return kFALSE;

  fCchi2TPC=chi2;

  if (!cParam) return kTRUE;

  *cParam = *fTPCInner;
  if (!cParam->Update(p,c)) return kFALSE;

  return kTRUE;
}

//_______________________________________________________________________
Bool_t AliESDtrack::RelateToVertexTPCBxByBz(const AliESDVertex *vtx, 
Double_t b[3], Double_t maxd, AliExternalTrackParam *cParam) {
  //
  // Try to relate the TPC-only track parameters to the vertex "vtx", 
  // if the (rough) transverse impact parameter is not bigger then "maxd". 
  //
  // All three components of the magnetic field ,"b[3]" (kG), 
  // are taken into account.
  //
  // a) The TPC-only paramters are extapolated to the DCA to the vertex.
  // b) The impact parameters and their covariance matrix are calculated.
  // c) An attempt to constrain the TPC-only params to the vertex is done.
  //    The constrained params are returned via "cParam".
  //
  // In the case of success, the returned value is kTRUE
  // otherwise, it's kFALSE)
  // 

  if (!fTPCInner) return kFALSE;
  if (!vtx) return kFALSE;

  Double_t dz[2],cov[3];
  if (!fTPCInner->PropagateToDCABxByBz(vtx, b, maxd, dz, cov)) return kFALSE;

  fdTPC = dz[0];
  fzTPC = dz[1];  
  fCddTPC = cov[0];
  fCdzTPC = cov[1];
  fCzzTPC = cov[2];
  
  Double_t covar[6]; vtx->GetCovMatrix(covar);
  Double_t p[2]={GetParameter()[0]-dz[0],GetParameter()[1]-dz[1]};
  Double_t c[3]={covar[2],0.,covar[5]};

  Double_t chi2=GetPredictedChi2(p,c);
  if (chi2>kVeryBig) return kFALSE;

  fCchi2TPC=chi2;

  if (!cParam) return kTRUE;

  *cParam = *fTPCInner;
  if (!cParam->Update(p,c)) return kFALSE;

  return kTRUE;
}

//_______________________________________________________________________
Bool_t AliESDtrack::RelateToVertex(const AliESDVertex *vtx, 
Double_t b, Double_t maxd, AliExternalTrackParam *cParam) {
  //
  // Try to relate this track to the vertex "vtx", 
  // if the (rough) transverse impact parameter is not bigger then "maxd". 
  //            Magnetic field is "b" (kG).
  //
  // a) The track gets extapolated to the DCA to the vertex.
  // b) The impact parameters and their covariance matrix are calculated.
  // c) An attempt to constrain this track to the vertex is done.
  //    The constrained params are returned via "cParam".
  //
  // In the case of success, the returned value is kTRUE
  // (otherwise, it's kFALSE)
  //  

  if (!vtx) return kFALSE;

  Double_t dz[2],cov[3];
  if (!PropagateToDCA(vtx, b, maxd, dz, cov)) return kFALSE;

  fD = dz[0];
  fZ = dz[1];  
  fCdd = cov[0];
  fCdz = cov[1];
  fCzz = cov[2];
  
  Double_t covar[6]; vtx->GetCovMatrix(covar);
  Double_t p[2]={GetParameter()[0]-dz[0],GetParameter()[1]-dz[1]};
  Double_t c[3]={covar[2],0.,covar[5]};

  Double_t chi2=GetPredictedChi2(p,c);
  if (chi2>kVeryBig) return kFALSE;

  fCchi2=chi2;


  //--- Could now these lines be removed ? ---
  delete fCp;
  fCp=new AliExternalTrackParam(*this);  

  if (!fCp->Update(p,c)) {delete fCp; fCp=0; return kFALSE;}
  //----------------------------------------

  fVertexID = vtx->GetID();

  if (!cParam) return kTRUE;

  *cParam = *this;
  if (!cParam->Update(p,c)) return kFALSE; 

  return kTRUE;
}

//_______________________________________________________________________
Bool_t AliESDtrack::RelateToVertexBxByBz(const AliESDVertex *vtx, 
Double_t b[3], Double_t maxd, AliExternalTrackParam *cParam) {
  //
  // Try to relate this track to the vertex "vtx", 
  // if the (rough) transverse impact parameter is not bigger then "maxd". 
  //            Magnetic field is "b" (kG).
  //
  // a) The track gets extapolated to the DCA to the vertex.
  // b) The impact parameters and their covariance matrix are calculated.
  // c) An attempt to constrain this track to the vertex is done.
  //    The constrained params are returned via "cParam".
  //
  // In the case of success, the returned value is kTRUE
  // (otherwise, it's kFALSE)
  //  

  if (!vtx) return kFALSE;

  Double_t dz[2],cov[3];
  if (!PropagateToDCABxByBz(vtx, b, maxd, dz, cov)) return kFALSE;

  fD = dz[0];
  fZ = dz[1];  
  fCdd = cov[0];
  fCdz = cov[1];
  fCzz = cov[2];
  
  Double_t covar[6]; vtx->GetCovMatrix(covar);
  Double_t p[2]={GetParameter()[0]-dz[0],GetParameter()[1]-dz[1]};
  Double_t c[3]={covar[2],0.,covar[5]};

  Double_t chi2=GetPredictedChi2(p,c);
  if (chi2>kVeryBig) return kFALSE;

  fCchi2=chi2;


  //--- Could now these lines be removed ? ---
  delete fCp;
  fCp=new AliExternalTrackParam(*this);  

  if (!fCp->Update(p,c)) {delete fCp; fCp=0; return kFALSE;}
  //----------------------------------------

  fVertexID = vtx->GetID();

  if (!cParam) return kTRUE;

  *cParam = *this;
  if (!cParam->Update(p,c)) return kFALSE; 

  return kTRUE;
}

//_______________________________________________________________________
void AliESDtrack::Print(Option_t *) const {
  // Prints info on the track
  AliExternalTrackParam::Print();
  printf("ESD track info\n") ; 
  Double_t p[AliPID::kSPECIES] ;
  Int_t index = 0 ; 
  if( IsOn(kITSpid) ){
    printf("From ITS: ") ; 
    GetITSpid(p) ; 
    for(index = 0 ; index < AliPID::kSPECIES; index++) 
      printf("%f, ", p[index]) ;
    printf("\n           signal = %f\n", GetITSsignal()) ;
  } 
  if( IsOn(kTPCpid) ){
    printf("From TPC: ") ; 
    GetTPCpid(p) ; 
    for(index = 0 ; index < AliPID::kSPECIES; index++) 
      printf("%f, ", p[index]) ;
    printf("\n           signal = %f\n", GetTPCsignal()) ;
  }
  if( IsOn(kTRDpid) ){
    printf("From TRD: ") ; 
    GetTRDpid(p) ; 
    for(index = 0 ; index < AliPID::kSPECIES; index++) 
      printf("%f, ", p[index]) ;
    printf("\n           signal = %f\n", GetTRDsignal()) ;
    printf("\n           NchamberdEdx = %d\n", GetTRDNchamberdEdx()) ;
    printf("\n           NclusterdEdx = %d\n", GetTRDNclusterdEdx()) ;
  }
  if( IsOn(kTOFpid) ){
    printf("From TOF: ") ; 
    GetTOFpid(p) ; 
    for(index = 0 ; index < AliPID::kSPECIES; index++) 
      printf("%f, ", p[index]) ;
    printf("\n           signal = %f\n", GetTOFsignal()) ;
  }
  if( IsOn(kHMPIDpid) ){
    printf("From HMPID: ") ; 
    GetHMPIDpid(p) ; 
    for(index = 0 ; index < AliPID::kSPECIES; index++) 
      printf("%f, ", p[index]) ;
    printf("\n           signal = %f\n", GetHMPIDsignal()) ;
  }
} 


//
// Draw functionality
// Origin: Marian Ivanov, Marian.Ivanov@cern.ch
//
void AliESDtrack::FillPolymarker(TPolyMarker3D *pol, Float_t magF, Float_t minR, Float_t maxR, Float_t stepR){
  //
  // Fill points in the polymarker
  //
  TObjArray arrayRef;
  arrayRef.AddLast(new AliExternalTrackParam(*this));
  if (fIp) arrayRef.AddLast(new AliExternalTrackParam(*fIp));
  if (fOp) arrayRef.AddLast(new AliExternalTrackParam(*fOp));
  if (fHMPIDp) arrayRef.AddLast(new AliExternalTrackParam(*fHMPIDp));
  //
  Double_t mpos[3]={0,0,0};
  Int_t entries=arrayRef.GetEntries();
  for (Int_t i=0;i<entries;i++){
    Double_t pos[3];
    ((AliExternalTrackParam*)arrayRef.At(i))->GetXYZ(pos);
    mpos[0]+=pos[0]/entries;
    mpos[1]+=pos[1]/entries;
    mpos[2]+=pos[2]/entries;    
  }
  // Rotate to the mean position
  //
  Float_t fi= TMath::ATan2(mpos[1],mpos[0]);
  for (Int_t i=0;i<entries;i++){
    Bool_t res = ((AliExternalTrackParam*)arrayRef.At(i))->Rotate(fi);
    if (!res) delete arrayRef.RemoveAt(i);
  }
  Int_t counter=0;
  for (Double_t r=minR; r<maxR; r+=stepR){
    Double_t sweight=0;
    Double_t mlpos[3]={0,0,0};
    for (Int_t i=0;i<entries;i++){
      Double_t point[3]={0,0,0};
      AliExternalTrackParam *param = ((AliExternalTrackParam*)arrayRef.At(i));
      if (!param) continue;
      if (param->GetXYZAt(r,magF,point)){
	Double_t weight = 1./(10.+(r-param->GetX())*(r-param->GetX()));
	sweight+=weight;
	mlpos[0]+=point[0]*weight;
	mlpos[1]+=point[1]*weight;
	mlpos[2]+=point[2]*weight;
      }
    }
    if (sweight>0){
      mlpos[0]/=sweight;
      mlpos[1]/=sweight;
      mlpos[2]/=sweight;      
      pol->SetPoint(counter,mlpos[0],mlpos[1], mlpos[2]);
      //      printf("xyz\t%f\t%f\t%f\n",mlpos[0], mlpos[1],mlpos[2]);
      counter++;
    }
  }
}

//_______________________________________________________________________
void AliESDtrack::SetITSdEdxSamples(const Double_t s[4]) {
  //
  // Store the dE/dx samples measured by the two SSD and two SDD layers.
  // These samples are corrected for the track segment length. 
  //
  for (Int_t i=0; i<4; i++) fITSdEdxSamples[i]=s[i];
}

//_______________________________________________________________________
void AliESDtrack::GetITSdEdxSamples(Double_t s[4]) const {
  //
  // Get the dE/dx samples measured by the two SSD and two SDD layers.  
  // These samples are corrected for the track segment length.
  //
  for (Int_t i=0; i<4; i++) s[i]=fITSdEdxSamples[i];
}


UShort_t   AliESDtrack::GetTPCnclsS(Int_t i0,Int_t i1) const{
  //
  // get number of shared TPC clusters
  //
  return  fTPCSharedMap.CountBits(i0)-fTPCSharedMap.CountBits(i1);
}

UShort_t   AliESDtrack::GetTPCncls(Int_t i0,Int_t i1) const{
  //
  // get number of TPC clusters
  //
  return  fTPCClusterMap.CountBits(i0)-fTPCClusterMap.CountBits(i1);
}

//____________________________________________________________________
Double_t AliESDtrack::GetChi2TPCConstrainedVsGlobal(const AliESDVertex* vtx) const
{
  // Calculates the chi2 between the TPC track (TPCinner) constrained to the primary vertex and the global track
  //
  // Returns -1 in case the calculation failed
  //
  // Value is cached as a non-persistent member.
  //
  // Code adapted from original code by GSI group (Jacek, Marian, Michael)
  
  // cache, ignoring that a different vertex might be passed
  if (fCacheChi2TPCConstrainedVsGlobalVertex == vtx)
    return fCacheChi2TPCConstrainedVsGlobal;
  
  fCacheChi2TPCConstrainedVsGlobal = -1;
  fCacheChi2TPCConstrainedVsGlobalVertex = vtx;
  
  Double_t x[3];
  GetXYZ(x);
  Double_t b[3];
  AliTrackerBase::GetBxByBz(x,b);

  if (!fTPCInner)  { 
    AliWarning("Could not get TPC Inner Param.");
    return fCacheChi2TPCConstrainedVsGlobal;
  }
  
  // clone for constraining
  AliExternalTrackParam* tpcInnerC = new AliExternalTrackParam(*fTPCInner);
  if (!tpcInnerC) { 
    AliWarning("Clone of TPCInnerParam failed.");
    return fCacheChi2TPCConstrainedVsGlobal;  
  }
  
  // transform to the track reference frame 
  Bool_t isOK = tpcInnerC->Rotate(GetAlpha());
  isOK &= tpcInnerC->PropagateTo(GetX(), b[2]);
  if (!isOK) { 
    delete tpcInnerC;
    tpcInnerC = 0; 
    AliWarning("Rotation/Propagation of track failed.") ; 
    return fCacheChi2TPCConstrainedVsGlobal;    
  }  

  // constrain TPCinner 
  isOK = tpcInnerC->ConstrainToVertex(vtx, b);
  
  // transform to the track reference frame 
  isOK &= tpcInnerC->Rotate(GetAlpha());
  isOK &= tpcInnerC->PropagateTo(GetX(), b[2]);

  if (!isOK) {
    AliWarning("ConstrainTPCInner failed.") ;
    delete tpcInnerC;
    tpcInnerC = 0; 
    return fCacheChi2TPCConstrainedVsGlobal;  
  }
  
  // calculate chi2 between vi and vj vectors
  // with covi and covj covariance matrices
  // chi2ij = (vi-vj)^(T)*(covi+covj)^(-1)*(vi-vj)
  TMatrixD deltaT(5,1);
  TMatrixD delta(1,5);
  TMatrixD covarM(5,5);

  for (Int_t ipar=0; ipar<5; ipar++) {
    deltaT(ipar,0) = tpcInnerC->GetParameter()[ipar] - GetParameter()[ipar];
    delta(0,ipar) = tpcInnerC->GetParameter()[ipar] - GetParameter()[ipar];

    for (Int_t jpar=0; jpar<5; jpar++) {
      Int_t index = GetIndex(ipar,jpar);
      covarM(ipar,jpar) = GetCovariance()[index]+tpcInnerC->GetCovariance()[index];
    }
  }
  // chi2 distance TPC constrained and TPC+ITS
  TMatrixD covarMInv = covarM.Invert();
  TMatrixD mat2 = covarMInv*deltaT;
  TMatrixD chi2 = delta*mat2; 
  
  delete tpcInnerC; 
  tpcInnerC = 0;
  
  fCacheChi2TPCConstrainedVsGlobal = chi2(0,0);
  return fCacheChi2TPCConstrainedVsGlobal;
}

void AliESDtrack::SetDetectorPID(const AliDetectorPID *pid)
{
  //
  // Set the detector PID
  //
  if (fDetectorPID) delete fDetectorPID;
  fDetectorPID=pid;
  
}

Double_t AliESDtrack::GetLengthInActiveZone( Int_t mode, Double_t deltaY, Double_t deltaZ, Double_t bz, Double_t exbPhi , TTreeSRedirector * pcstream) const {
  //
  // Input parameters:
  //   mode  - type of external track parameters 
  //   deltaY - user defined "dead region" in cm
  //   deltaZ - user defined "active region" in cm (250 cm drift lenght - 14 cm L1 delay
  //   bz     - magnetic field 
  //   exbPhi - optional rotation due to the ExB effect
  // return value:
  //   the length of the track in cm in "active volume" of the TPC
  //
  if (mode==0) return GetLengthInActiveZone(this, deltaY,deltaZ,bz, exbPhi,pcstream);
  if (mode==1) return GetLengthInActiveZone(fIp, deltaY,deltaZ,bz, exbPhi,pcstream);
  if (mode==2) return GetLengthInActiveZone(fOp, deltaY,deltaZ,bz, exbPhi,pcstream);
  return 0;
}

Double_t AliESDtrack::GetLengthInActiveZone(const AliExternalTrackParam  *paramT, Double_t deltaY, Double_t deltaZ, Double_t bz, Double_t exbPhi , TTreeSRedirector * pcstream) {
  //
  // Numerical code to calculate the length of the track in active region of the TPC
  // ( can be speed up if somebody wants to invest time - analysical version shoult be possible) 
  //
  // Input parameters:
  //   paramT - external track parameters 
  //   deltaY - user defined "dead region" in cm
  //   deltaZ - user defined "active region" in cm (250 cm drift lenght - 14 cm L1 delay
  //   bz     - magnetic field 
  //   exbPhi - optional rotation due to the ExB effect
  // return value:
  //   the length of the track in cm in "active volume" of the TPC
  //
  const Double_t rIn=85;
  const Double_t rOut=245;
  Double_t xyz[3], pxyz[3];
  if (paramT->GetXYZAt(rIn,bz,xyz)){
    paramT->GetPxPyPzAt(rIn,bz,pxyz);
  }else{
    paramT->GetXYZ(xyz);
    paramT->GetPxPyPz(pxyz);
  }
  //
  Double_t dca   = -paramT->GetD(0,0,bz);  // get impact parameter distance to point (0,0)
  Double_t radius= TMath::Abs(1/paramT->GetC(bz));  //
  Double_t sign  = paramT->GetSign()*TMath::Sign(1.,bz)*(-1.);
  Double_t R0    = TMath::Sqrt(xyz[0]*xyz[0]+xyz[1]*xyz[1]);   // radius at current point
  Double_t phiR0 = TMath::ATan2(xyz[1],xyz[0]);                // angle of given point
  Double_t dPhiR0= -TMath::ASin((dca*dca-2*dca*radius*sign+R0*R0)/(2*R0*(dca-radius*sign)));
  Double_t phi0  = phiR0-(dPhiR0);  // global phi offset to be added
  //
  //
  AliExternalTrackParam paramR=(*paramT);
  Double_t length=0;
  for (Double_t R=rIn; R<=rOut; R++){
    Double_t sinPhi=(dca*dca-2*dca*radius*sign+R*R)/(2*R*(dca-radius*sign));
    if (TMath::Abs(sinPhi)>=1) continue;
    Double_t dphi     = -TMath::ASin(sinPhi);
    Double_t phi      = phi0+dphi;                           // global phi
    Int_t    sector   = TMath::Nint(9*phi/(TMath::Pi()));
    Double_t dPhiEdge = phi-(sector*TMath::Pi()/9)+exbPhi;   // distance to sector boundary in rphi
    Double_t dX   = R*TMath::Cos(phi)-xyz[0];
    Double_t dY   = R*TMath::Sin(phi)-xyz[1];
    Double_t deltaPhi = 2*TMath::ASin(0.5*TMath::Sqrt(dX*dX+dY*dY)/radius);
    Double_t z = xyz[2]+deltaPhi*radius*paramT->GetTgl();
    if (TMath::Abs(dPhiEdge*R)>deltaY && TMath::Abs(z)<deltaZ){
      length++;
    }
    //    Double_t deltaZ= dphi*radius; 
    if (pcstream){
      //should we keep debug possibility ?
      AliExternalTrackParam paramTcopy=(*paramT);
      paramR.Rotate(phi);
      paramR.PropagateTo(R,bz);
      (*pcstream)<<"debugEdge"<<
    "length="<<length<< // track length
    "radius="<<radius<< // radius
	"R="<<R<<                   // radius
	"dphiEdge="<<dPhiEdge<<     // distance to edge 
	"phi0="<<phi0<<	            // phi0 -phi at the track initial position
	"phi="<<phi<<               // 
	"z="<<z<<
	"pT.="<<&paramTcopy<<
	"pR.="<<&paramR<<
	"\n";
    }
  }
  return length;
}

Double_t AliESDtrack::GetMassForTracking() const
{
  int pid = fPIDForTracking;
  if (pid<AliPID::kPion && fgTrackEMuAsPi) pid = AliPID::kPion;
  double m = AliPID::ParticleMass(pid);
  return (fPIDForTracking==AliPID::kHe3 || fPIDForTracking==AliPID::kAlpha) ? -m : m;
}


void    AliESDtrack::SetTOFclusterArray(Int_t /*ncluster*/,Int_t */*TOFcluster*/){
  AliInfo("Method has to be implemented!");
//   fNtofClusters=ncluster;
//   if(TOFcluster == fTOFcluster) return;
//   if(fTOFcluster){ // reset previous content    
//     delete[] fTOFcluster;
//     fTOFcluster = NULL;
//     fNtofClusters=0;
//   }

//   if(ncluster){ // set new content
//     fTOFcluster = new Int_t[fNtofClusters];
//     for(Int_t i=0;i < fNtofClusters;i++) fTOFcluster[i] = TOFcluster[i];
//   }
//   else
//     fTOFcluster = 0;
}

//____________________________________________
void  AliESDtrack::SuppressTOFMatches()
{
  // remove reference to this track from TOF clusters
  if (!fNtofClusters || !GetESDEvent()) return;
  TClonesArray *tofclArray = GetESDEvent()->GetESDTOFClusters();
  for (;fNtofClusters--;) {
    AliESDTOFCluster* clTOF = (AliESDTOFCluster*)tofclArray->At(fTOFcluster[fNtofClusters]);
    clTOF->SuppressMatchedTrack(GetID());
    if (!clTOF->GetNMatchableTracks()) { // remove this cluster
      int last = tofclArray->GetEntriesFast()-1;
      AliESDTOFCluster* clTOFL = (AliESDTOFCluster*)tofclArray->At(last);
      if (last != fTOFcluster[fNtofClusters]) {
	*clTOF = *clTOFL; // move last cluster to the place of eliminated one
	// fix the references on this cluster
	clTOF->FixSelfReferences(last,fTOFcluster[fNtofClusters]);
      }
      tofclArray->RemoveAt(last);
    }
  }
}

//____________________________________________
void  AliESDtrack::ReplaceTOFTrackID(int oldID, int newID)
{
  // replace the ID in TOF clusters references to this track
  if (!fNtofClusters || !GetESDEvent()) return;
  TClonesArray *tofclArray = GetESDEvent()->GetESDTOFClusters();
  if (!tofclArray || tofclArray->GetEntries()<1) return;
  for (int it=fNtofClusters;it--;) {
    AliESDTOFCluster* clTOF = (AliESDTOFCluster*)tofclArray->At(fTOFcluster[it]);
    clTOF->ReplaceMatchedTrackID(oldID,newID);
  }
}

//____________________________________________
void  AliESDtrack::ReplaceTOFClusterID(int oldID, int newID)
{
  // replace the referenc on TOF cluster oldID by newID
  if (!fNtofClusters || !GetESDEvent()) return;
  for (int it=fNtofClusters;it--;) {
    if (fTOFcluster[it] == oldID) {
      fTOFcluster[it] = newID;
      return;
    }
  }
}

//____________________________________________
void  AliESDtrack::ReplaceTOFMatchID(int oldID, int newID)
{
  // replace in the ESDTOFCluster associated with this track the id of the corresponding
  // ESDTOFMatch from oldID to newID
  if (!fNtofClusters || !GetESDEvent()) return;
  TClonesArray *tofclArray = GetESDEvent()->GetESDTOFClusters();
  for (int it=fNtofClusters;it--;) {
    AliESDTOFCluster* clTOF = (AliESDTOFCluster*)tofclArray->At(fTOFcluster[it]);
    clTOF->ReplaceMatchID(oldID,newID);
  }
}

//____________________________________________
void AliESDtrack::AddTOFcluster(Int_t icl)
{
  fNtofClusters++;
  
  Int_t *old = fTOFcluster;
  fTOFcluster = new Int_t[fNtofClusters];

  for(Int_t i=0;i < fNtofClusters-1;i++) fTOFcluster[i] = old[i];
  fTOFcluster[fNtofClusters-1] = icl;

  if(fNtofClusters-1)  delete[] old; // delete previous content    
 
}

//____________________________________________
void AliESDtrack::SetTOFsignal(Double_t tof)
{
  if(fNtofClusters>0 && GetESDEvent()){
    TClonesArray *tofclArray = GetESDEvent()->GetESDTOFClusters();
    AliESDTOFCluster *tofcl = (AliESDTOFCluster *) tofclArray->At(fTOFcluster[0]);
    AliESDTOFHit* hit = tofcl->GetTOFHit(0);
    if(hit) hit->SetTime(tof);
  }
  else{
    if(fNtofClusters>0) AliInfo("No AliESDEvent available here!\n");
    fTOFsignal=tof;
  }
}
//____________________________________________
void AliESDtrack::SetTOFCalChannel(Int_t index){
  if(fNtofClusters>0 && GetESDEvent()){
    TClonesArray *tofclArray = GetESDEvent()->GetESDTOFClusters();
    AliESDTOFCluster *tofcl = (AliESDTOFCluster *) tofclArray->At(fTOFcluster[0]);
    AliESDTOFHit* hit = tofcl->GetTOFHit(0);
    if(hit) hit->SetTOFchannel(index);
  }
  else{
    if(fNtofClusters>0) AliInfo("No AliESDEvent available here!\n");
    fTOFCalChannel=index;
  }
}
//____________________________________________
void AliESDtrack::SetTOFsignalToT(Double_t ToT){
  if(fNtofClusters>0 && GetESDEvent()){
    TClonesArray *tofclArray = GetESDEvent()->GetESDTOFClusters();
    AliESDTOFCluster *tofcl = (AliESDTOFCluster *) tofclArray->At(fTOFcluster[0]);
    AliESDTOFHit* hit = tofcl->GetTOFHit(0);
    if(hit) hit->SetTOT(ToT);
  }
  else{
    if(fNtofClusters>0) AliInfo("No AliESDEvent available here!\n");
    fTOFsignalToT=ToT;
  }
}
//____________________________________________
void AliESDtrack::SetTOFsignalRaw(Double_t tof){
  if(fNtofClusters>0 && GetESDEvent()){
    TClonesArray *tofclArray = GetESDEvent()->GetESDTOFClusters();
    AliESDTOFCluster *tofcl = (AliESDTOFCluster *) tofclArray->At(fTOFcluster[0]);
    AliESDTOFHit* hit = tofcl->GetTOFHit(0);
    if(hit) hit->SetTimeRaw(tof);
  }
  else{
    if(fNtofClusters>0) AliInfo("No AliESDEvent available here!\n");
    fTOFsignalRaw=tof;
  }
}
//____________________________________________
void AliESDtrack::SetTOFsignalDz(Double_t dz){
  Int_t index = -1;
  AliESDTOFCluster *tofcl;

  if(fNtofClusters>0 && GetESDEvent()){
    TClonesArray *tofclArray = GetESDEvent()->GetESDTOFClusters();
    tofcl = (AliESDTOFCluster *) tofclArray->At(fTOFcluster[0]);

    for(Int_t i=0;i < tofcl->GetNMatchableTracks();i++){
      if(tofcl->GetTrackIndex(i) == GetID()) index = i;
    }

  }
  if(index > -1){
    AliESDTOFMatch* match = tofcl->GetTOFMatch(index);
    if(match){
      match->SetDz(dz);
    }
  }
  else{
    if(fNtofClusters>0) AliInfo("No AliESDEvent available here!\n");
    fTOFsignalDz=dz;
  }


}
//____________________________________________
void AliESDtrack::SetTOFsignalDx(Double_t dx){
  Int_t index = -1;
  AliESDTOFCluster *tofcl;

  if(fNtofClusters>0 && GetESDEvent()){
    TClonesArray *tofclArray = GetESDEvent()->GetESDTOFClusters();
    tofcl = (AliESDTOFCluster *) tofclArray->At(fTOFcluster[0]);

    for(Int_t i=0;i < tofcl->GetNMatchableTracks();i++){
      if(tofcl->GetTrackIndex(i) == GetID()) index = i;
    }

  }
  if(index > -1){
    AliESDTOFMatch* match = tofcl->GetTOFMatch(index);
    if(match){
      match->SetDx(dx);
    }
  }
  else{
    if(fNtofClusters>0) AliInfo("No AliESDEvent available here!\n");
    fTOFsignalDx=dx;
  }
}
//____________________________________________
void AliESDtrack::SetTOFDeltaBC(Short_t deltaBC){
  if(fNtofClusters>0 && GetESDEvent()){
    TClonesArray *tofclArray = GetESDEvent()->GetESDTOFClusters();
    AliESDTOFCluster *tofcl = (AliESDTOFCluster *) tofclArray->At(fTOFcluster[0]);
    AliESDTOFHit* hit = tofcl->GetTOFHit(0);
    if(hit) hit->SetDeltaBC(deltaBC);
  }
  else{
    if(fNtofClusters>0) AliInfo("No AliESDEvent available here!\n");
    fTOFdeltaBC=deltaBC;
  }
}
//____________________________________________
void AliESDtrack::SetTOFL0L1(Short_t l0l1){
  if(fNtofClusters>0 && GetESDEvent()){
    TClonesArray *tofclArray = GetESDEvent()->GetESDTOFClusters();
    AliESDTOFCluster *tofcl = (AliESDTOFCluster *) tofclArray->At(fTOFcluster[0]);
    AliESDTOFHit* hit = tofcl->GetTOFHit(0);
    if(hit) hit->SetL0L1Latency(l0l1);
  }
  else{
    if(fNtofClusters>0) AliInfo("No AliESDEvent available here!\n");
    fTOFl0l1=l0l1;
  }
}
//____________________________________________
Double_t AliESDtrack::GetTOFsignal() const 
{
  if(fNtofClusters>0 && GetESDEvent()){
    TClonesArray *tofclArray = GetESDEvent()->GetESDTOFClusters();
    AliESDTOFCluster *tofcl = (AliESDTOFCluster *) tofclArray->At(fTOFcluster[0]);

    return tofcl->GetTime();
  }
  else if(fNtofClusters>0) AliInfo("No AliESDEvent available here!\n");

  return fTOFsignal;
}

//____________________________________________
Double_t AliESDtrack::GetTOFsignalToT() const 
{
  if(fNtofClusters>0 && GetESDEvent()){
    TClonesArray *tofclArray = GetESDEvent()->GetESDTOFClusters();
    AliESDTOFCluster *tofcl = (AliESDTOFCluster *) tofclArray->At(fTOFcluster[0]);

    return tofcl->GetTOT();
  }
  else if(fNtofClusters>0) AliInfo("No AliESDEvent available here!\n");

  return fTOFsignalToT;
}

//____________________________________________
Double_t AliESDtrack::GetTOFsignalRaw() const 
{
  if(fNtofClusters>0 && GetESDEvent()){
    TClonesArray *tofclArray = GetESDEvent()->GetESDTOFClusters();
    AliESDTOFCluster *tofcl = (AliESDTOFCluster *) tofclArray->At(fTOFcluster[0]);

    return tofcl->GetTimeRaw();
  }
  else if(fNtofClusters>0) AliInfo("No AliESDEvent available here!\n");

  return fTOFsignalRaw;
}

//____________________________________________
Double_t AliESDtrack::GetTOFsignalDz() const 
{

  AliESDTOFCluster *tofcl;

  Int_t index = -1;
  if(fNtofClusters>0 && GetESDEvent()){
    TClonesArray *tofclArray = GetESDEvent()->GetESDTOFClusters();
    tofcl = (AliESDTOFCluster *) tofclArray->At(fTOFcluster[0]);

    for(Int_t i=0;i < tofcl->GetNMatchableTracks();i++){
      if(tofcl->GetTrackIndex(i) == GetID()) index = i;
    }
  }
  else if(fNtofClusters>0) AliInfo("No AliESDEvent available here!\n");

  if(fNtofClusters>0 && index > -1){
    return tofcl->GetDz(index);
  }
  return fTOFsignalDz;
}

//____________________________________________
Double_t AliESDtrack::GetTOFsignalDx() const 
{
  AliESDTOFCluster *tofcl;

  Int_t index = -1;
  if(fNtofClusters>0 && GetESDEvent()){
    TClonesArray *tofclArray = GetESDEvent()->GetESDTOFClusters();
    tofcl = (AliESDTOFCluster *) tofclArray->At(fTOFcluster[0]);
    for(Int_t i=0;i < tofcl->GetNMatchableTracks();i++){
      if(tofcl->GetTrackIndex(i) == GetID()) index = i;
    }
  }
  else if(fNtofClusters>0) AliInfo("No AliESDEvent available here!\n");
  if(fNtofClusters>0 && index > -1){
    return tofcl->GetDx(index);
  }
  return fTOFsignalDx;
}

//____________________________________________
Short_t  AliESDtrack::GetTOFDeltaBC() const 
{
  if(fNtofClusters>0 && GetESDEvent()){
    TClonesArray *tofclArray = GetESDEvent()->GetESDTOFClusters();
    AliESDTOFCluster *tofcl = (AliESDTOFCluster *) tofclArray->At(fTOFcluster[0]);
    return tofcl->GetDeltaBC();
  }
  else if(fNtofClusters>0) AliInfo("No AliESDEvent available here!\n");

  return fTOFdeltaBC;
}

//____________________________________________
Short_t  AliESDtrack::GetTOFL0L1() const 
{
  if(fNtofClusters>0 && GetESDEvent()){
    TClonesArray *tofclArray = GetESDEvent()->GetESDTOFClusters();
    AliESDTOFCluster *tofcl = (AliESDTOFCluster *) tofclArray->At(fTOFcluster[0]);

    return tofcl->GetL0L1Latency();
  }
  else if(fNtofClusters>0) AliInfo("No AliESDEvent available here!\n");

  return fTOFl0l1;
}

//____________________________________________
Int_t   AliESDtrack::GetTOFCalChannel() const 
{
  if(fNtofClusters>0 && GetESDEvent()){
    TClonesArray *tofclArray = GetESDEvent()->GetESDTOFClusters();
    AliESDTOFCluster *tofcl = (AliESDTOFCluster *) tofclArray->At(fTOFcluster[0]);

    return tofcl->GetTOFchannel();
  }
  else if(fNtofClusters>0) AliInfo("No AliESDEvent available here!\n");

  return fTOFCalChannel;
}

//____________________________________________
Int_t   AliESDtrack::GetTOFcluster() const 
{
  if(fNtofClusters>0 && GetESDEvent()){
    TClonesArray *tofclArray = GetESDEvent()->GetESDTOFClusters();
    AliESDTOFCluster *tofcl = (AliESDTOFCluster *) tofclArray->At(fTOFcluster[0]);

    return tofcl->GetClusterIndex();
  }
  else if(fNtofClusters>0) AliInfo("No AliESDEvent available here!\n");

  return fTOFindex;
}

//____________________________________________
Int_t   AliESDtrack::GetTOFclusterN() const
{
  return fNtofClusters;
}

//____________________________________________
Bool_t  AliESDtrack::IsTOFHitAlreadyMatched() const{
  if(fNtofClusters>0 && GetESDEvent()){
    TClonesArray *tofclArray = GetESDEvent()->GetESDTOFClusters();
    AliESDTOFCluster *tofcl = (AliESDTOFCluster *) tofclArray->At(fTOFcluster[0]);

    if (tofcl->GetNMatchableTracks() > 1)
      return kTRUE;
  }
  else if(fNtofClusters>0) AliInfo("No AliESDEvent available here!\n");

  return kFALSE;
}

//____________________________________________
void AliESDtrack::ReMapTOFcluster(Int_t ncl,Int_t *mapping){
  for(Int_t i=0;i<fNtofClusters;i++){
    if(fTOFcluster[i]<ncl && fTOFcluster[i]>-1)
      fTOFcluster[i] = mapping[fTOFcluster[i]];
    else
      AliInfo(Form("TOF cluster re-mapping in AliESDtrack: out of range (%i > %i)\n",fTOFcluster[i],ncl));
  }
}

//____________________________________________
void AliESDtrack::SortTOFcluster(){
  TClonesArray *tofclArray = GetESDEvent()->GetESDTOFClusters();

  for(Int_t i=0;i<fNtofClusters-1;i++){
    for(Int_t j=i+1;j<fNtofClusters;j++){
      AliESDTOFCluster *tofcl = (AliESDTOFCluster *) tofclArray->At(fTOFcluster[i]);
      Int_t index1 = -1;
      for(Int_t it=0;it < tofcl->GetNMatchableTracks();it++){
         if(tofcl->GetTrackIndex(it) == GetID()) index1 = it;
      }
      Double_t timedist1 = 10000;
      for(Int_t isp=0; isp< AliPID::kSPECIESC;isp++){
	Double_t timec = TMath::Abs(tofcl->GetTime() - tofcl->GetIntegratedTime(isp));
	if(timec < timedist1) timedist1 = timec;
      }
      timedist1 *= 0.03; // in cm
      Double_t radius1 = tofcl->GetDx(index1)*tofcl->GetDx(index1) + tofcl->GetDz(index1)*tofcl->GetDz(index1) + timedist1*timedist1;

      AliESDTOFCluster *tofcl2 = (AliESDTOFCluster *) tofclArray->At(fTOFcluster[j]);
      Int_t index2 = -1;
      for(Int_t it=0;it < tofcl2->GetNMatchableTracks();it++){
         if(tofcl2->GetTrackIndex(it) == GetID()) index2 = it;
      }
      if(index1 == -1 || index2 == -1){
      }
      Double_t timedist2 = 10000;
      for(Int_t isp=0; isp< AliPID::kSPECIESC;isp++){
	Double_t timec = TMath::Abs(tofcl2->GetTime() - tofcl2->GetIntegratedTime(isp));
	if(timec < timedist2) timedist2 = timec;
      }
      timedist2 *= 0.03; // in cm
      Double_t radius2 = tofcl2->GetDx(index2)*tofcl2->GetDx(index2) + tofcl2->GetDz(index2)*tofcl2->GetDz(index2) + timedist2*timedist2;

      if(radius2 < radius1){
        Int_t change = fTOFcluster[i];
        fTOFcluster[i] = fTOFcluster[j];
        fTOFcluster[j] = change;
      }
    }
  }
}

//____________________________________________
const AliTOFHeader* AliESDtrack::GetTOFHeader() const {
  return fESDEvent ? fESDEvent->GetTOFHeader() : 0x0;
}

//___________________________________________
void AliESDtrack::SetID(Short_t id) 
{
  // set track ID taking care about dependencies
  if (fNtofClusters) ReplaceTOFTrackID(fID,id); 
  fID=id;
}


Double_t  AliESDtrack::GetdEdxInfo(Int_t regionID, Int_t calibID, Int_t qID, Int_t valueID){
  //
  // Interface to get the calibrated dEdx information 
  // For details of arguments and return values see 
  //     AliTPCdEdxInfo::GetdEdxInfo(Int_t regionID, Int_t calibID, Int_t valueID)
  //
  if (!fTPCdEdxInfo) return 0;
  if (!fIp) return 0;
  return fTPCdEdxInfo->GetdEdxInfo(fIp, regionID, calibID, qID, valueID);
}


Double_t AliESDtrack::GetdEdxInfoTRD(Int_t method, Double_t p0, Double_t p1, Double_t p2){
  //
  // Methods
  // mean values:
  //     0.)   linear
  //     1.)   logarithmic
  //     2.)   1/sqrt
  //     3.)   power()
  // time COG:
  //     4.)   linear
  //     5.)   logarithmic
  //     6.)   square
  Int_t nSlicesPerLayer=GetNumberOfTRDslices();
  Int_t nSlicesAll=GetNumberOfTRDslices()*kTRDnPlanes;

  if (method<=3){
    Double_t sumAmp=0;
    Int_t    sumW=0;
    for (Int_t ibin=0; ibin<nSlicesAll; ibin++){
      if (fTRDslices[ibin]<=0) continue; 
      sumW++;
      if (method==0) sumAmp+=fTRDslices[ibin];
      if (method==1) sumAmp+=TMath::Log(TMath::Abs(fTRDslices[ibin])+p0);
      if (method==2) sumAmp+=1/TMath::Sqrt(TMath::Abs(fTRDslices[ibin])+p0);
      if (method==3) sumAmp+=TMath::Power(TMath::Abs(fTRDslices[ibin])+p0,p1);
    }
    if (sumW==0) return 0;
    Double_t dEdx=sumAmp/sumW;
    if (method==1) dEdx= TMath::Exp(dEdx);
    if (method==2) dEdx= 1/(dEdx*dEdx);
    if (method==3) dEdx= TMath::Power(dEdx,1/p1);
    return dEdx;
  }
  if (method>3){
    Double_t sumWT=0;
    Double_t sumW=0;
    for (Int_t ibin=0; ibin<nSlicesAll; ibin++){
      if (fTRDslices[ibin]<=0) continue; 
      Double_t time=(ibin%nSlicesPerLayer);
      Double_t weight=fTRDslices[ibin];
      if (method==5) weight=TMath::Log((weight+p0)/p0);
      if (method==6) weight=TMath::Power(weight+p0,p1);
      sumWT+=time*weight;
      sumW+=weight;
    }
    if (sumW<=0) return 0;
    Double_t meanTime=sumWT/sumW;
    return meanTime;
  }
  return 0;
}

void AliESDtrack::SetImpactParameters( const Float_t p[2], const Float_t cov[3], const Float_t chi2, const AliExternalTrackParam *cParam)
{
  // set impact parameters
  
  fD = p[0];
  fZ = p[1];  
  fCdd = cov[0];
  fCdz = cov[1];
  fCzz = cov[2];
  fCchi2=chi2;
  delete fCp;
  if( cParam ) fCp=new AliExternalTrackParam(*cParam);  
}

void AliESDtrack::SetImpactParametersTPC( const Float_t p[2], const Float_t cov[3], const Float_t chi2 )
{
  // set impact parameters TPC
  
  fdTPC = p[0];
  fzTPC = p[1];  
  fCddTPC = cov[0];
  fCdzTPC = cov[1];
  fCzzTPC = cov[2];
  fCchi2TPC = chi2;
}

void  AliESDtrack::SetTrackEMuAsPi(Bool_t val)
{
  // when true, track mu and e with pion mass (run 2)
  fgTrackEMuAsPi = val;
  AliInfoClassF("Track e and mu with pion mass: %d",fgTrackEMuAsPi);
}
