// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   Controller.h
/// @author ruben.shahoyan@cern.ch, michael.lettrich@cern.ch
/// @since  2021-02-01
/// @brief  Steering class for the global alignment

#include "Align/Controller.h"
#include "Align/AlignConfig.h"
#include "Framework/Logger.h"
#include "Align/utils.h"
#include "Align/AlignmentPoint.h"
#include "Align/AlignableDetector.h"
#include "Align/AlignableVolume.h"
#include "Align/AlignableDetectorITS.h"
#include "Align/AlignableDetectorTRD.h"
#include "Align/AlignableDetectorTOF.h"
#include "Align/EventVertex.h"
#include "Align/ResidualsControllerFast.h"
#include "Align/GeometricalConstraint.h"
#include "Align/DOFStatistics.h"
#include "ReconstructionDataFormats/VtxTrackIndex.h"
#include "ReconstructionDataFormats/PrimaryVertex.h"
#include "DataFormatsGlobalTracking/RecoContainer.h"
#include "ReconstructionDataFormats/VtxTrackRef.h"
#include "TRDBase/TrackletTransformer.h"
#include "MathUtils/Utils.h"

#include <TMath.h>
#include <TString.h>

#include <TROOT.h>
#include <TSystem.h>
#include <TRandom.h>
#include <TH1F.h>
#include <TList.h>
#include <cstdio>
#include <TGeoGlobalMagField.h>
#include "CommonUtils/NameConf.h"
#include "MathUtils/SymMatrixSolver.h"
#include "DataFormatsParameters/GRPObject.h"

#include "SimulationDataFormat/MCUtils.h"
#include "Steer/MCKinematicsReader.h"
#include "CommonUtils/TreeStreamRedirector.h"
#include <unordered_map>

using namespace TMath;
using namespace o2::align::utils;
using namespace o2::dataformats;
using namespace o2::globaltracking;

namespace o2
{
namespace align
{
using GIndex = o2::dataformats::VtxTrackIndex;
using PropagatorD = o2::base::PropagatorD;
using MatCorrType = PropagatorD::MatCorrType;

void Controller::ProcStat::print() const
{
  // TODO RS
  //  const Char_t* Controller::sStatClName[Controller::kNStatCl] = {"Inp: ", "Acc: "};
  //  const Char_t* Controller::sStatName[Controller::kMaxStat] =
  //  {"runs", "Ev.Coll", "Ev.Cosm", "Trc.Coll", "Trc.Cosm"};
}

const Char_t* Controller::sMPDataExt = ".mille";
const Char_t* Controller::sMPDataTxtExt = ".mille_txt";

const Char_t* Controller::sDetectorName[Controller::kNDetectors] = {"ITS", "TPC", "TRD", "TOF", "HMPID"}; //RSREM

//const int Controller::mgkSkipLayers[Controller::kNLrSkip] = {AliGeomManager::kPHOS1, AliGeomManager::kPHOS2,
//                                                                 AliGeomManager::kMUON, AliGeomManager::kEMCAL}; TODO(milettri, shahoian): needs detector IDs previously stored in AliGeomManager
const int Controller::sSkipLayers[Controller::kNLrSkip] = {0, 0, 0, 0}; // TODO(milettri, shahoian): needs AliGeomManager - remove this line after fix.

//________________________________________________________________
Controller::Controller(DetID::mask_t detmask, GTrackID::mask_t trcmask, bool useMC)
  : mDetMask(detmask), mMPsrc(trcmask), mUseMC(useMC)
{
  init();
}

//________________________________________________________________
Controller::~Controller()
{
  // d-tor
  closeMPRecOutput();
  closeMilleOutput();
  closeResidOutput();
  //
}

//________________________________________________________________
void Controller::init()
{
  if (mDetMask[DetID::ITS]) {
    addDetector(new AlignableDetectorITS(this));
  }
  if (mDetMask[DetID::TRD]) {
    addDetector(new AlignableDetectorTRD(this));
  }
  if (mDetMask[DetID::TOF]) {
    addDetector(new AlignableDetectorTOF(this));
  }
  for (int src = GIndex::NSources; src--;) {
    if (mMPsrc[src]) {
      mTrackSources.push_back(src);
    }
  }
  mVtxSens = std::make_unique<EventVertex>(this);
}

//________________________________________________________________
void Controller::process()
{
  static int nTF = 0;
  o2::steer::MCKinematicsReader mcReader;
  if (mUseMC) {
    if (!mcReader.initFromDigitContext("collisioncontext.root")) {
      throw std::invalid_argument("initialization of MCKinematicsReader failed");
    }
  }
  auto timerStart = std::chrono::system_clock::now();
  int nVtx = 0, nVtxAcc = 0, nTrc = 0, nTrcAcc = 0;
  for (auto id = DetID::First; id <= DetID::Last; id++) {
    auto* det = getDetector(id);
    if (det) {
      det->prepareDetectorData(); // in case the detector needs to preprocess the RecoContainer data
    }
  }
  auto primVertices = mRecoData->getPrimaryVertices();
  auto primVer2TRefs = mRecoData->getPrimaryVertexMatchedTrackRefs();
  auto primVerGIs = mRecoData->getPrimaryVertexMatchedTracks();
  const auto& algConf = AlignConfig::Instance();
  // process vertices with contributor tracks
  std::unordered_map<GIndex, bool> ambigTable;
  int nvRefs = primVer2TRefs.size();
  bool fieldON = std::abs(PropagatorD::Instance()->getNominalBz()) > 0.1;

  for (int ivref = 0; ivref < nvRefs; ivref++) {
    const o2::dataformats::PrimaryVertex* vtx = (ivref < nvRefs - 1) ? &primVertices[ivref] : nullptr;
    bool useVertexConstrain = false;
    if (vtx) {
      auto nContrib = vtx->getNContributors();
      useVertexConstrain = nContrib >= algConf.vtxMinCont && nContrib <= algConf.vtxMaxCont;
    }
    auto& trackRef = primVer2TRefs[ivref];
    if (algConf.verbose > 1) {
      LOGP(info, "processing vtref {} of {} with {} tracks, {}", ivref, nvRefs, trackRef.getEntries(), vtx ? vtx->asString() : std::string{});
    }
    nVtx++;
    bool newVtx = true;
    for (int src : mTrackSources) {
      if ((GIndex::getSourceDetectorsMask(src) & mDetMask).none()) { // do we need this source?
        continue;
      }
      int start = trackRef.getFirstEntryOfSource(src), end = start + trackRef.getEntriesOfSource(src);
      for (int ti = start; ti < end; ti++) {
        auto trackIndex = primVerGIs[ti];
        if (trackIndex.isAmbiguous()) {
          auto& ambSeen = ambigTable[trackIndex];
          if (ambSeen) { // processed
            continue;
          }
          ambSeen = true;
        }
        int npnt = 0;
        auto contributorsGID = mRecoData->getSingleDetectorRefs(trackIndex);

        std::string trComb;
        for (int ig = 0; ig < GIndex::NSources; ig++) {
          if (contributorsGID[ig].isIndexSet()) {
            trComb += " " + contributorsGID[ig].asString();
          }
        }
        if (algConf.verbose > 1) {
          LOG(info) << "processing track " << trackIndex.asString() << " contributors: " << trComb;
        }
        resetForNextTrack();
        nTrc++;
        // RS const auto& trcOut = mRecoData->getTrackParamOut(trackIndex);
        auto trcOut = mRecoData->getTrackParamOut(trackIndex);
        const auto& trcIn = mRecoData->getTrackParam(trackIndex);
        // check detectors contributions
        AlignableDetector* det = nullptr;
        int ndet = 0, npntDet = 0;

        if ((det = getDetector(DetID::ITS))) {
          if (contributorsGID[GIndex::ITS].isIndexSet() && (npntDet = det->processPoints(contributorsGID[GIndex::ITS], false))) {
            npnt += npntDet;
            ndet++;
          } else if (mAllowAfterburnerTracks && contributorsGID[GIndex::ITSAB].isIndexSet() && (npntDet = det->processPoints(contributorsGID[GIndex::ITSAB], false)) > 0) {
            npnt += npntDet;
            ndet++;
          } else {
            continue;
          }
        }
        if ((det = getDetector(DetID::TRD)) && contributorsGID[GIndex::TRD].isIndexSet() && (npntDet = det->processPoints(contributorsGID[GIndex::TRD], false)) > 0) {
          npnt += npntDet;
          ndet++;
        }
        if ((det = getDetector(DetID::TOF)) && contributorsGID[GIndex::TOF].isIndexSet() && (npntDet = det->processPoints(contributorsGID[GIndex::TOF], false)) > 0) {
          npnt += npntDet;
          ndet++;
        }
        // other detectors
        if (algConf.verbose > 1) {
          LOGP(info, "processing track {} {} of vtref {}, Ndets:{}, Npoints: {}, use vertex: {} | Kin: {} Kout: {}", ti, trackIndex.asString(), ivref, ndet, npnt, useVertexConstrain && trackIndex.isPVContributor(), trcIn.asString(), trcOut.asString());
        }
        if (ndet < algConf.minDetectors) {
          continue;
        }
        if (npnt < algConf.minPointTotal) {
          if (algConf.verbose > 0) {
            LOGP(info, "too few points {} < {}", npnt, algConf.minPointTotal);
          }
          continue;
        }
        bool vtxCont = false;
        if (trackIndex.isPVContributor() && useVertexConstrain) {
          mAlgTrack->copyFrom(trcIn); // copy kinematices of inner track just for propagation to the vertex
          if (addVertexConstraint(*vtx)) {
            mAlgTrack->setRefPoint(mRefPoint.get()); // set vertex as a reference point
            vtxCont = true;
          }
        }
        mAlgTrack->copyFrom(trcOut); // copy kinematices of outer track as the refit will be done inward
        mAlgTrack->setFieldON(fieldON);
        mAlgTrack->sortPoints();

        int pntMeas = mAlgTrack->getInnerPointID() - 1;
        if (pntMeas < 0) { // this should not happen
          mAlgTrack->Print("p meas");
          LOG(error) << "AliAlgTrack->GetInnerPointID() cannot be 0";
        }
        if (!mAlgTrack->iniFit()) {
          if (algConf.verbose > 0) {
            LOGP(warn, "iniFit failed");
          }
          continue;
        }

        // compare refitted and original track
        if (mDebugOutputLevel) {
          trackParam_t trcAlgRef(*mAlgTrack.get());
          std::array<double, 5> dpar{};
          std::array<double, 15> dcov{};
          for (int i = 0; i < 5; i++) {
            dpar[i] = trcIn.getParam(i);
          }
          for (int i = 0; i < 15; i++) {
            dcov[i] = trcIn.getCov()[i];
          }
          trackParam_t trcOrig(trcIn.getX(), trcIn.getAlpha(), dpar, dcov, trcIn.getCharge());
          if (PropagatorD::Instance()->propagateToAlphaX(trcOrig, trcAlgRef.getAlpha(), trcAlgRef.getX(), true)) {
            (*mDBGOut) << "trcomp"
                       << "orig=" << trcOrig << "fit=" << trcAlgRef << "\n";
          }
        }
        // RS: this is to substitute the refitter track by MC truth, just for debugging
        /*
        if (mUseMC) {
          auto lbl = mRecoData->getTrackMCLabel(trackIndex);
          if (lbl.isValid()) {
            o2::MCTrack mcTrack = *mcReader.getTrack(lbl);
            std::array<float,3> xyz{(float)mcTrack.GetStartVertexCoordinatesX(),(float)mcTrack.GetStartVertexCoordinatesY(),(float)mcTrack.GetStartVertexCoordinatesZ()},
              pxyz{(float)mcTrack.GetStartVertexMomentumX(),(float)mcTrack.GetStartVertexMomentumY(),(float)mcTrack.GetStartVertexMomentumZ()};
            std::array<float,21> cv21{10., 0.,10., 0.,0.,10., 0.,0.,0.,1.,   0.,0.,0.,0.,1., 0.,0.,0.,0.,0.,1.};
            trcOut.set(xyz, pxyz, cv21, trcOut.getSign(), false);
            mAlgTrack->copyFrom(trcOut);
          }
        }
        */
        if (!mAlgTrack->processMaterials()) {
          if (algConf.verbose > 0) {
            LOGP(warn, "processMaterials failed");
          }
          continue;
        }
        mAlgTrack->defineDOFs();
        if (!mAlgTrack->calcResidDeriv()) {
          if (algConf.verbose > 0) {
            LOGP(warn, "calcResidDeriv failed");
          }
          continue;
        }

        if (mUseMC && mDebugOutputLevel) {
          auto lbl = mRecoData->getTrackMCLabel(trackIndex);
          if (lbl.isValid()) {
            std::vector<float> pntX, pntY, pntZ, trcX, trcY, trcZ, prpX, prpY, prpZ, alpha, xsens, pntXTF, pntYTF, pntZTF, resY, resZ;
            std::vector<int> detid, volid;

            o2::MCTrack mcTrack = *mcReader.getTrack(lbl);
            trackParam_t recTrack{*mAlgTrack};
            for (int ip = 0; ip < mAlgTrack->getNPoints(); ip++) {
              double tmp[3], tmpg[3];
              auto* pnt = mAlgTrack->getPoint(ip);
              auto* sens = pnt->getSensor();
              detid.emplace_back(pnt->getDetID());
              volid.emplace_back(pnt->getVolID());
              TGeoHMatrix t2g;
              sens->getMatrixT2G(t2g);
              t2g.LocalToMaster(pnt->getXYZTracking(), tmpg);
              pntX.emplace_back(tmpg[0]);
              pntY.emplace_back(tmpg[1]);
              pntZ.emplace_back(tmpg[2]);
              double xyz[3]{pnt->getXTracking(), pnt->getYTracking(), pnt->getZTracking()};
              xyz[1] += mAlgTrack->getResidual(0, ip);
              xyz[2] += mAlgTrack->getResidual(1, ip);
              t2g.LocalToMaster(xyz, tmpg);
              trcX.emplace_back(tmpg[0]);
              trcY.emplace_back(tmpg[1]);
              trcZ.emplace_back(tmpg[2]);

              pntXTF.emplace_back(pnt->getXTracking());
              pntYTF.emplace_back(pnt->getYTracking());
              pntZTF.emplace_back(pnt->getZTracking());
              resY.emplace_back(mAlgTrack->getResidual(0, ip));
              resZ.emplace_back(mAlgTrack->getResidual(1, ip));

              alpha.emplace_back(pnt->getAlphaSens());
              xsens.emplace_back(pnt->getXSens());
            }
            (*mDBGOut) << "mccomp"
                       << "mcTr=" << mcTrack << "recTr=" << recTrack << "gid=" << trackIndex << "lbl=" << lbl << "vtxConst=" << vtxCont
                       << "pntX=" << pntX << "pntY=" << pntY << "pntZ=" << pntZ
                       << "trcX=" << trcX << "trcY=" << trcY << "trcZ=" << trcZ
                       << "alp=" << alpha << "xsens=" << xsens
                       << "pntXTF=" << pntXTF << "pntYTF=" << pntYTF << "pntZTF=" << pntZTF
                       << "resY=" << resY << "resZ=" << resZ
                       << "detid=" << detid << "volid=" << volid << "\n";
          }
        }
        nTrcAcc++;
        if (newVtx) {
          newVtx = false;
          nVtxAcc++;
        }
        storeProcessedTrack(trackIndex);
      }
    }
  }
  auto timerEnd = std::chrono::system_clock::now();
  std::chrono::duration<float, std::milli> duration = timerEnd - timerStart;
  LOGP(info, "Processed TF {}: {} vertices ({} used), {} tracks ({} used) in {} ms", nTF, nVtx, nVtxAcc, nTrc, nTrcAcc, duration.count());
  nTF++;
}

//________________________________________________________________
void Controller::initDetectors()
{
  // init all detectors geometry
  //
  if (getInitGeomDone()) {
    return;
  }
  //
  mAlgTrack = std::make_unique<AlignmentTrack>();
  mRefPoint = std::make_unique<AlignmentPoint>();
  //
  int dofCnt = 0;
  // special fake sensor for vertex constraint point
  // it has special T2L matrix adjusted for each track, no need to init it here
  mVtxSens = std::make_unique<EventVertex>(this);
  mVtxSens->setInternalID(1);
  mVtxSens->prepareMatrixL2G();
  mVtxSens->prepareMatrixL2GIdeal();
  dofCnt += mVtxSens->getNDOFs();
  //
  for (auto id = DetID::First; id <= DetID::Last; id++) {
    auto* det = getDetector(id);
    if (det) {
      dofCnt += det->initGeom();
    }
  }
  if (!dofCnt) {
    LOG(fatal) << "No DOFs found";
  }
  //
  //
  for (auto id = DetID::First; id <= DetID::Last; id++) {
    auto* det = getDetector(id);
    if (!det || det->isDisabled()) {
      continue;
    }
    det->cacheReferenceCCDB();
  }
  //
  assignDOFs();
  LOG(info) << "Booked " << dofCnt << " global parameters";
  //
  setInitGeomDone();
  //
}

//________________________________________________________________
void Controller::initDOFs()
{
  // scan all free global parameters, link detectors to array of params
  //
  if (getInitDOFsDone()) {
    LOG(info) << "initDOFs was already done, just reassigning " << getNDOFs() << "DOFs arrays/labels";
    assignDOFs();
    return;
  }
  const auto& conf = AlignConfig::Instance();

  mNDOFs = 0;
  int ndfAct = 0;
  assignDOFs();
  int nact = 0;
  mVtxSens->initDOFs();
  for (auto id = DetID::First; id <= DetID::Last; id++) {
    AlignableDetector* det = getDetector(id);
    if (det && !det->isDisabled()) {
      det->initDOFs();
      nact++;
      ndfAct += det->getNDOFs();
    }
  }
  for (int i = 0; i < NTrackTypes; i++) {
    if (nact < conf.minDetAcc[i]) {
      LOG(fatal) << nact << " detectors are active, while " << conf.minDetAcc[i] << " in track are asked";
    }
  }
  LOG(info) << mNDOFs << " global parameters " << mNDet << " detectors, " << ndfAct << " in " << nact << " active detectors";
  addAutoConstraints();
  setInitDOFsDone();
}

//________________________________________________________________
void Controller::assignDOFs()
{
  // add parameters/labels arrays to volumes. If the Controller is read from the file, this method need
  // to be called (of initDOFs should be called)
  //
  int ndfOld = -1;
  if (mNDOFs > 0) {
    ndfOld = mNDOFs;
  }
  mNDOFs = 0;
  //
  // reserve
  int ndofTOT = mVtxSens->getNDOFs();
  for (auto id = DetID::First; id <= DetID::Last; id++) {
    AlignableDetector* det = getDetector(id);
    if (!det) {
      continue;
    }
    ndofTOT += det->getNDOFs();
  }
  mGloParVal.clear();
  mGloParErr.clear();
  mGloParLab.clear();
  mLbl2ID.clear();
  mGloParVal.reserve(ndofTOT);
  mGloParErr.reserve(ndofTOT);
  mGloParLab.reserve(ndofTOT);

  mVtxSens->assignDOFs();

  for (auto id = DetID::First; id <= DetID::Last; id++) {
    AlignableDetector* det = getDetector(id);
    if (!det) {
      continue;
    }
    det->assignDOFs();
  }
  LOG(info) << "Assigned parameters/labels arrays for " << mNDOFs << " DOFs";
  if (ndfOld > 0 && ndfOld != mNDOFs) {
    LOG(error) << "Recalculated NDOFs=" << mNDOFs << " not equal to saved NDOFs=" << ndfOld;
  }
  // build Lbl -> parID table
  for (int i = 0; i < ndofTOT; i++) {
    int& id = mLbl2ID[mGloParLab[i]];
    if (id != 0) {
      LOGP(fatal, "parameters {} and {} share the same label {}", id - 1, i, mGloParLab[i]);
    }
    id = i + 1;
  }
  //
}

//_________________________________________________________
void Controller::addDetector(AlignableDetector* det)
{
  // add detector constructed externally to alignment framework
  mDetectors[det->getDetID()].reset(det);
  mNDet++;
}

//_________________________________________________________
bool Controller::checkDetectorPattern(DetID::mask_t patt) const
{
  //validate detector pattern
  return ((patt & mObligatoryDetPattern[mTracksType]) == mObligatoryDetPattern[mTracksType]) &&
         patt.count() >= AlignConfig::Instance().minDetAcc[mTracksType];
}

//_________________________________________________________
bool Controller::checkDetectorPoints(const int* npsel) const
{
  //validate detectors pattern according to number of selected points
  int ndOK = 0;
  for (auto id = DetID::First; id <= DetID::Last; id++) {
    AlignableDetector* det = getDetector(id);
    if (!det || det->isDisabled(mTracksType)) {
      continue;
    }
    if (npsel[id] < det->getNPointsSel(mTracksType)) {
      if (det->isObligatory(mTracksType)) {
        return false;
      }
      continue;
    }
    ndOK++;
  }
  return ndOK >= AlignConfig::Instance().minDetAcc[mTracksType];
}

//_________________________________________________________
bool Controller::storeProcessedTrack(o2::dataformats::GlobalTrackID tid)
{
  // write alignment track
  bool res = true;
  const auto& conf = AlignConfig::Instance();
  if (conf.MilleOut) {
    res &= fillMilleData();
  }
  if (conf.MPRecOut) {
    res &= fillMPRecData(tid);
  }
  if ((conf.controlFraction > gRandom->Rndm()) && mAlgTrack->testLocalSolution()) {
    res &= fillControlData(tid);
  }
  //
  if (!res) {
    LOGP(error, "storeProcessedTrack failed");
  }
  return res;
}

//_________________________________________________________
bool Controller::fillMilleData()
{
  // store MP2 data in Mille format
  if (!mMille) {
    const auto& conf = AlignConfig::Instance();
    mMilleFileName = fmt::format("{}_{:08d}_{:010d}{}", AlignConfig::Instance().mpDatFileName, mTimingInfo.runNumber, mTimingInfo.tfCounter, conf.MilleOutBin ? sMPDataExt : sMPDataTxtExt);
    mMille = std::make_unique<Mille>(mMilleFileName.c_str(), conf.MilleOutBin);
  }
  if (!mAlgTrack->getDerivDone()) {
    LOG(error) << "Track derivatives are not yet evaluated";
    return false;
  }
  int np = mAlgTrack->getNPoints(), nDGloTot = 0; // total number global derivatives stored
  int nParETP = mAlgTrack->getNLocExtPar();       // numnber of local parameters for reference track param
  int nVarLoc = mAlgTrack->getNLocPar();          // number of local degrees of freedom in the track
  //
  const int* gloParID = mAlgTrack->getGloParID(); // IDs of global DOFs this track depends on
  for (int ip = 0; ip < np; ip++) {
    AlignmentPoint* pnt = mAlgTrack->getPoint(ip);
    if (pnt->containsMeasurement()) {
      int gloOffs = pnt->getDGloOffs(); // 1st entry of global derivatives for this point
      int nDGlo = pnt->getNGloDOFs();   // number of global derivatives (number of DOFs it depends on)
      if (!pnt->isStatOK()) {
        pnt->incrementStat();
      }
      int milleIBufferG[nDGlo];
      float milleDBufferG[nDGlo];
      float milleDBufferL[nVarLoc];
      std::memset(milleIBufferG, 0, sizeof(int) * nDGlo);
      std::memset(milleDBufferG, 0, sizeof(float) * nDGlo);
      std::memset(milleDBufferL, 0, sizeof(float) * nVarLoc);
      // local der. array cannot be 0-suppressed by Mille construction, need to reset all to 0
      for (int idim = 0; idim < 2; idim++) {                    // 2 dimensional orthogonal measurement
        const double* deriv = mAlgTrack->getDResDLoc(idim, ip); // array of Dresidual/Dparams_loc
        // derivatives over reference track parameters
        for (int j = 0; j < nParETP; j++) {
          milleDBufferL[j] = isZeroAbs(deriv[j]) ? 0. : deriv[j];
        }
        //
        // point may depend on material variables within these limits
        for (int j = pnt->getMinLocVarID(); j < pnt->getMaxLocVarID(); j++) {
          milleDBufferL[j] = isZeroAbs(deriv[j]) ? 0. : deriv[j];
        }
        // derivatives over global params: this array can be 0-suppressed, no need to reset
        int nGlo = 0;
        deriv = mAlgTrack->getDResDGlo(idim, gloOffs);
        const int* gloIDP(gloParID + gloOffs);
        for (int j = 0; j < nDGlo; j++) {
          milleDBufferG[nGlo] = isZeroAbs(deriv[j]) ? 0. : deriv[j]; // value of derivative
          milleIBufferG[nGlo++] = getGloParLab(gloIDP[j]);           // global DOF ID + 1 (Millepede needs positive labels)
        }
        mMille->mille(nVarLoc, milleDBufferL, nGlo, milleDBufferG, milleIBufferG, mAlgTrack->getResidual(idim, ip), Sqrt(pnt->getErrDiag(idim)));
        nDGloTot += nGlo;
      }
    }
    if (pnt->containsMaterial()) {     // material point can add 4 or 5 otrhogonal pseudo-measurements
      int nmatpar = pnt->getNMatPar(); // residuals (correction expectation value)
      //      const float* expMatCorr = pnt->getMatCorrExp(); // expected corrections (diagonalized)
      const float* expMatCov = pnt->getMatCorrCov(); // their diagonalized error matrix
      int offs = pnt->getMaxLocVarID() - nmatpar;    // start of material variables
      // here all derivatives are 1 = dx/dx
      float milleDBufferL[nVarLoc];
      std::memset(milleDBufferL, 0, sizeof(float) * nVarLoc);
      for (int j = 0; j < nmatpar; j++) { // mat. "measurements" don't depend on global params
        int j1 = j + offs;
        milleDBufferL[j1] = 1.0; // only 1 non-0 derivative
        // mMille->mille(nVarLoc,milleDBufferL,0, nullptr, nullptr, expMatCorr[j], Sqrt(expMatCov[j]));
        // expectation for MS effect is 0
        mMille->mille(nVarLoc, milleDBufferL, 0, nullptr, nullptr, 0, Sqrt(expMatCov[j]));
        milleDBufferL[j1] = 0.0; // reset buffer
      }
    } // material "measurement"
  }   // loop over points
  //
  if (!nDGloTot) {
    LOG(info) << "Track does not depend on free global parameters, discard";
    mMille->clear();
    return false;
  }
  mMille->finalise(); // store the record
  return true;
}

//_________________________________________________________
bool Controller::fillMPRecData(o2::dataformats::GlobalTrackID tid)
{
  // store MP2 in MPRecord format
  if (!mMPRecFile) {
    initMPRecOutput();
  }
  mMPRecord.clear();
  if (!mMPRecord.fillTrack(*mAlgTrack.get(), mGloParLab)) {
    return false;
  }
  mMPRecord.setRun(mRunNumber);
  mMPRecord.setFirstTFOrbit(mTimingInfo.firstTForbit);
  mMPRecord.setTrackID(tid);
  mMPRecTree->Fill();
  return true;
}

//_________________________________________________________
bool Controller::fillControlData(o2::dataformats::GlobalTrackID tid)
{
  // store control residuals
  if (!mResidFile) {
    initResidOutput();
  }
  int nps, np = mAlgTrack->getNPoints();
  nps = (!mRefPoint->containsMeasurement()) ? np - 1 : np; // ref point is dummy?
  if (nps < 0) {
    return true;
  }
  mCResid.clear();
  if (!mCResid.fillTrack(*mAlgTrack.get(), AlignConfig::Instance().KalmanResid)) {
    return false;
  }
  mCResid.setRun(mRunNumber);
  mCResid.setFirstTFOrbit(mTimingInfo.firstTForbit);
  mCResid.setBz(o2::base::PropagatorD::Instance()->getNominalBz());
  mCResid.setTrackID(tid);
  // if (isCosmic()) {
  //   mCResid.setInvTrackID(tid);
  // }
  mResidTree->Fill();
  return true;
}

//_________________________________________________________
void Controller::setTimingInfo(const o2::framework::TimingInfo& ti)
{
  mTimingInfo = ti;
  LOGP(info, "TIMING {} {}", ti.runNumber, ti.creation);
  if (ti.runNumber != mRunNumber) {
    mRunNumber = ti.runNumber;
    acknowledgeNewRun();
  }
}

//_________________________________________________________
void Controller::acknowledgeNewRun()
{
  LOG(warning) << __PRETTY_FUNCTION__ << " yet incomplete";

  // o2::base::GeometryManager::loadGeometry();
  // o2::base::PropagatorImpl<double>::initFieldFromGRP();
  // std::unique_ptr<o2::parameters::GRPObject> grp{o2::parameters::GRPObject::loadFrom()};

  // FIXME(milettri): needs AliESDEvent
  //   // load needed info for new run
  //   if (run == mRunNumber){
  //     return;} // nothing to do
  //   if (run > 0) {
  //     mStat[kAccStat][kRun]++;
  //   }
  //   if (mRunNumber > 0){
  //   mRunNumber = run;
  //   LOG(info) << "Processing new run " << mRunNumber;
  //   //
  //   // setup magnetic field
  //   if (fESDEvent &&
  //       (!TGeoGlobalMagField::Instance()->GetField() ||
  //        !smallerAbs(fESDEvent->GetMagneticField() - AliTrackerBase::GetBz(), 5e-4))) {
  //     fESDEvent->InitMagneticField();
  //   }
  //   //
  //   if (!mUseRecoOCDB) {
  //     LOG(warning) << "Reco-time OCDB will NOT be preloaded";
  //     return;
  //   }
  //   LoadRecoTimeOCDB();
  //   //
  //   for (auto id=DetID::First; id<=DetID::Last; id++) {
  //     AlignableDetector* det = getDetector(id);
  //     if (!det->isDisabled()){
  //       det->acknowledgeNewRun(run);}
  //   }
  //   //
  //   // bring to virgin state
  //   // CleanOCDB();
  //   //
  //   // LoadRefOCDB(); //??? we need to get back reference OCDB ???
  //   //
  //   mStat[kInpStat][kRun]++;
  //   //
}

// FIXME(milettri): needs OCDB
////_________________________________________________________
//bool Controller::LoadRecoTimeOCDB()
//{
//  // Load OCDB paths used for the reconstruction of data being processed
//  // In order to avoid unnecessary uploads, the objects are not actually
//  // loaded/cached but just added as specific paths with version
//  LOG(info) << "Preloading Reco-Time OCDB for run " << mRunNumber << " from ESD UserInfo list";
//  //
//  CleanOCDB();
//  //
//  if (!mRecoOCDBConf.IsNull() && !gSystem->AccessPathName(mRecoOCDBConf.c_str(), kFileExists)) {
//    LOG(info) << "Executing reco-time OCDB setup macro " << mRecoOCDBConf.c_str();
//    gROOT->ProcessLine(Form(".x %s(%d)", mRecoOCDBConf.c_str(), mRunNumber));
//    if (AliCDBManager::Instance()->IsDefaultStorageSet()){
//      return true;}
//    LOG(fatal) << "macro " << mRecoOCDBConf.c_str() << " failed to configure reco-time OCDB";
//  } else
//    LOG(warning) << "No reco-time OCDB config macro" << mRecoOCDBConf.c_str() << "  is found, will use ESD:UserInfo";
//  //
//  if (!mESDTree){
//    LOG(fatal) << "Cannot preload Reco-Time OCDB since the ESD tree is not set";}
//  const TTree* tr = mESDTree; // go the the real ESD tree
//  while (tr->GetTree() && tr->GetTree() != tr)
//    tr = tr->GetTree();
//  //
//  const TList* userInfo = const_cast<TTree*>(tr)->GetUserInfo();
//  TMap* cdbMap = (TMap*)userInfo->FindObject("cdbMap");
//  TList* cdbList = (TList*)userInfo->FindObject("cdbList");
//  //
//  if (!cdbMap || !cdbList) {
//    userInfo->Print();
//    LOG(fatal) << "Failed to extract cdbMap and cdbList from UserInfo list";
//  }
//  //
//  return PreloadOCDB(mRunNumber, cdbMap, cdbList);
//}

//____________________________________________
void Controller::Print(const Option_t* opt) const
{
  // print info
  TString opts = opt;
  opts.ToLower();
  printf("%5d DOFs in %d detectors\n", mNDOFs, mNDet);
  if (getMPAlignDone()) {
    printf("ALIGNMENT FROM MILLEPEDE SOLUTION IS APPLIED\n");
  }
  //
  for (auto id = DetID::First; id <= DetID::Last; id++) {
    AlignableDetector* det = getDetector(id);
    if (!det) {
      continue;
    }
    det->Print(opt);
  }
  if (!opts.IsNull()) {
    printf("\nSpecial sensor for Vertex Constraint\n");
    mVtxSens->Print(opt);
  }
  //
  if (mRefRunNumber >= 0) {
    printf("(%d)", mRefRunNumber);
  }
  AlignConfig::Instance().printKeyValues();
  //
  if (opts.Contains("stat")) {
    printStatistics();
  }
}

//________________________________________________________
void Controller::printStatistics() const
{
  // print processing stat
  mStat.print();
}

//________________________________________________________
void Controller::resetForNextTrack()
{
  // reset detectors for next track
  mRefPoint->clear();
  mAlgTrack->Clear();
  for (auto id = DetID::First; id <= DetID::Last; id++) {
    AlignableDetector* det = getDetector(id);
    if (det) {
      det->reset();
    }
  }
}

//____________________________________________
bool Controller::testLocalSolution()
{
  // test track local solution
  int npnt = mAlgTrack->getNPoints(), nlocpar = mAlgTrack->getNLocPar();
  double mat[nlocpar][nlocpar], rhs[nlocpar];
  std::memset(mat, 0, sizeof(double) * nlocpar * nlocpar);
  std::memset(rhs, 0, sizeof(double) * nlocpar);
  for (int ip = npnt; ip--;) {
    AlignmentPoint* pnt = mAlgTrack->getPoint(ip);
    if (pnt->containsMeasurement()) {
      for (int idim = 2; idim--;) {                                                           // each point has 2 position residuals
        double resid = mAlgTrack->getResidual(idim, ip), sg2inv = 1. / pnt->getErrDiag(idim); // residual and its inv. error
        auto deriv = mAlgTrack->getDResDLoc(idim, ip);                                        // array of Dresidual/Dparams
        for (int parI = 0; parI < nlocpar; parI++) {
          rhs[parI] -= deriv[parI] * resid * sg2inv;
          for (int parJ = parI; parJ < nlocpar; parJ++) {
            mat[parI][parJ] += deriv[parI] * deriv[parJ] * sg2inv;
          }
        }
      } // loop over 2 orthogonal measurements at the point
    }   // derivarives at measured points
    // if the point contains material, consider its expected kinks, eloss as measurements
    if (pnt->containsMaterial()) { // at least 4 parameters: 2 spatial + 2 angular kinks with 0 expectaction
      int npm = pnt->getNMatPar();
      // const float* expMatCorr = pnt->getMatCorrExp(); // expected correction (diagonalized) // RS??
      const float* expMatCov = pnt->getMatCorrCov(); // its error
      int offs = pnt->getMaxLocVarID() - npm;
      for (int ipar = 0; ipar < npm; ipar++) {
        int parI = offs + ipar;
        // expected
        // rhs[parI] -= expMatCorr[ipar]/expMatCov[ipar]; // consider expectation as measurement // RS??
        mat[parI][parI] += 1. / expMatCov[ipar]; // this measurement is orthogonal to all others
      }
    } // material effect descripotion params
    //
  }
  o2::math_utils::SymMatrixSolver solver(nlocpar, 1);
  for (int i = 0; i < nlocpar; i++) {
    for (int j = i; j < nlocpar; j++) {
      solver.A(i, j) = mat[i][j];
    }
    solver.B(i, 0) = rhs[i];
  }
  solver.solve();
  // increment current params by new solution
  auto& pars = mAlgTrack->getLocParsV();
  for (int i = 0; i < nlocpar; i++) {
    pars[i] += solver.B(i, 0);
  }
  mAlgTrack->calcResiduals();
  return true;
}

//____________________________________________
void Controller::initMPRecOutput()
{
  // prepare MP record output
  mMPRecFile.reset(TFile::Open(fmt::format("{}_{:08d}_{:010d}{}", AlignConfig::Instance().mpDatFileName, mTimingInfo.runNumber, mTimingInfo.tfCounter, ".root").c_str(), "recreate"));
  mMPRecTree = std::make_unique<TTree>("mpTree", "MPrecord Tree");
  mMPRecTree->Branch("mprec", "o2::align::Millepede2Record", &mMPRecordPtr);
}

//____________________________________________
void Controller::initResidOutput()
{
  // prepare residual output
  mResidFile.reset(TFile::Open(fmt::format("{}_{:08d}_{:010d}{}", AlignConfig::Instance().residFileName, mTimingInfo.runNumber, mTimingInfo.tfCounter, ".root").c_str(), "recreate"));
  mResidTree = std::make_unique<TTree>("res", "Control Residuals");
  mResidTree->Branch("t", "o2::align::ResidualsController", &mCResidPtr);
}

//____________________________________________
void Controller::closeMPRecOutput()
{
  // close output
  if (!mMPRecFile) {
    return;
  }
  LOGP(info, "Writing tree {} with {} entries to {}", mMPRecTree->GetName(), mMPRecTree->GetEntries(), mMPRecFile->GetName());
  mMPRecFile->cd();
  mMPRecTree->Write();
  mMPRecTree.reset();
  mMPRecFile->Close();
  mMPRecFile.reset();
}

//____________________________________________
void Controller::closeResidOutput()
{
  // close output
  if (!mResidFile) {
    return;
  }
  LOG(info) << "Closing " << mResidFile->GetName();
  mResidFile->cd();
  mResidTree->Write();
  mResidTree.reset();
  mResidFile->Close();
  mResidFile.reset();
  mCResid.clear();
}

//____________________________________________
void Controller::closeMilleOutput()
{
  // close output
  if (mMille) {
    LOG(info) << "Closing " << mMilleFileName;
  }
  mMille.reset();
}

//____________________________________________
void Controller::setObligatoryDetector(DetID detID, int trtype, bool v)
{
  // mark detector presence obligatory in the track of given type
  AlignableDetector* det = getDetector(detID);
  if (!det) {
    LOG(error) << "Detector " << detID << " is not defined";
  }
  if (v) {
    mObligatoryDetPattern[trtype] |= detID.getMask();
  } else {
    mObligatoryDetPattern[trtype] &= ~detID.getMask();
  }
  if (det->isObligatory(trtype) != v) {
    det->setObligatory(trtype, v);
  }
  //
}

//____________________________________________
bool Controller::addVertexConstraint(const o2::dataformats::PrimaryVertex& vtx)
{
  auto* prop = PropagatorD::Instance();
  const auto& conf = AlignConfig::Instance();
  o2::dataformats::DCA dcaInfo;
  AlignmentTrack::trackParam_t trcDCA(*mAlgTrack.get());
  if (!prop->propagateToDCABxByBz(vtx, trcDCA, conf.maxStep, MatCorrType(conf.matCorType), &dcaInfo)) {
    return false;
  }
  // RS FIXME do selections if needed
  mVtxSens->setAlpha(trcDCA.getAlpha());
  double xyz[3] = {vtx.getX(), vtx.getY(), vtx.getZ()}, xyzT[3];
  double c[3] = {0.5 * (vtx.getSigmaX2() + vtx.getSigmaY2()), 0., vtx.getSigmaZ2()};

  mVtxSens->applyCorrection(xyz);
  mVtxSens->getMatrixT2L().MasterToLocal(xyz, xyzT);

  mRefPoint->setSensor(mVtxSens.get());
  // RS FIXME the Xsensor is 0 ?
  mRefPoint->setXYZTracking(xyzT);
  mRefPoint->setYZErrTracking(c);
  mRefPoint->setAlphaSens(mVtxSens->getAlpTracking()); // RS FIXME Cannot this be done in setSensor?
  mRefPoint->setContainsMeasurement(true);
  mRefPoint->init();
  return true;
}

//______________________________________________________
void Controller::writeCalibrationResults() const
{
  // writes output calibration
  AlignableDetector* det;
  for (auto id = DetID::First; id <= DetID::Last; id++) {
    if (!(det = getDetector(id)) || det->isDisabled()) {
      continue;
    }
    det->writeCalibrationResults();
  }
}

//________________________________________________________
AlignableDetector* Controller::getDetOfDOFID(int id) const
{
  // return detector owning DOF with this ID
  for (auto id = DetID::First; id <= DetID::Last; id++) {
    AlignableDetector* det = getDetector(id);
    if (det && det->ownsDOFID(id)) {
      return det;
    }
  }
  return nullptr;
}

//________________________________________________________
AlignableVolume* Controller::getVolOfDOFID(int id) const
{
  // return volume owning DOF with this ID
  for (auto id = DetID::First; id <= DetID::Last; id++) {
    AlignableDetector* det = getDetector(id);
    if (det && det->ownsDOFID(id)) {
      return det->getVolOfDOFID(id);
    }
  }
  if (mVtxSens && mVtxSens->ownsDOFID(id)) {
    return mVtxSens.get();
  }
  return nullptr;
}

//________________________________________________________
void Controller::terminate(bool doStat)
{
  // finalize processing
  if (doStat) {
    if (mVtxSens) {
      mVtxSens->fillDOFStat(mDOFStat);
    }
  }
  //
  for (auto id = DetID::First; id <= DetID::Last; id++) {
    if (getDetector(id)) {
      getDetector(id)->terminate();
    }
  }
  closeMPRecOutput();
  closeMilleOutput();
  closeResidOutput();
  Print("stat");
  //
}

//________________________________________________________
Char_t* Controller::getDOFLabelTxt(int idf) const
{
  // get DOF full label
  AlignableVolume* vol = getVolOfDOFID(idf);
  if (vol) {
    return Form("%d_%s_%s", getGloParLab(idf), vol->getSymName(),
                vol->getDOFName(idf - vol->getFirstParGloID()));
  }
  //
  // this might be detector-specific calibration dof
  AlignableDetector* det = getDetOfDOFID(idf);
  if (det) {
    return Form("%d_%s_%s", getGloParLab(idf), det->GetName(),
                det->getCalibDOFName(idf - det->getFirstParGloID()));
  }
  return nullptr;
}

//********************* interaction with PEDE **********************

//______________________________________________________
void Controller::genPedeSteerFile(const Option_t* opt) const
{
  // produce steering file template for PEDE + params and constraints
  //
  enum { kOff,
         kOn,
         kOnOn };
  const char* cmt[3] = {"  ", "! ", "!!"};
  const char* kSolMeth[] = {"inversion", "diagonalization", "fullGMRES", "sparseGMRES", "cholesky", "HIP"};
  const int kDefNIter = 3;     // default number of iterations to ask
  const float kDefDelta = 0.1; // def. delta to exit
  TString opts = opt;
  opts.ToLower();
  LOG(info) << "Generating MP2 templates:\n "
            << "Steering   :\t" << AlignConfig::Instance().mpSteerFileName << "\n"
            << "Parameters :\t" << AlignConfig::Instance().mpParFileName << "\n"
            << "Constraints:\t" << AlignConfig::Instance().mpConFileName << "\n";
  //
  FILE* parFl = fopen(AlignConfig::Instance().mpParFileName.c_str(), "w+");
  FILE* strFl = fopen(AlignConfig::Instance().mpSteerFileName.c_str(), "w+");
  //
  // --- template of steering file
  fprintf(strFl, "%-20s%s %s\n", AlignConfig::Instance().mpParFileName.c_str(), cmt[kOnOn], "parameters template");
  fprintf(strFl, "%-20s%s %s\n", AlignConfig::Instance().mpConFileName.c_str(), cmt[kOnOn], "constraints template");
  //
  fprintf(strFl, "\n\n%s %s\n", cmt[kOnOn], "MUST uncomment 1 solving methods and tune it");
  //
  int nm = sizeof(kSolMeth) / sizeof(char*);
  for (int i = 0; i < nm; i++) {
    fprintf(strFl, "%s%s %-20s %2d %.2f %s\n", cmt[kOn], "method", kSolMeth[i], kDefNIter, kDefDelta, cmt[kOnOn]);
  }
  //
  const float kDefChi2F0 = 20., kDefChi2F = 3.; // chi2 factors for 1st and following iterations
  const float kDefDWFrac = 0.2;                 // cut outliers with downweighting above this factor
  const int kDefOutlierDW = 4;                  // start Cauchy function downweighting from iteration
  const int kDefEntries = 25;                   // min entries per DOF to allow its variation
  //
  fprintf(strFl, "\n\n%s %s\n", cmt[kOnOn], "Optional settings");
  fprintf(strFl, "\n%s%-20s %.2f %.2f %s %s\n", cmt[kOn], "chisqcut", kDefChi2F0, kDefChi2F,
          cmt[kOnOn], "chi2 cut factors for 1st and next iterations");
  fprintf(strFl, "%s%-20s %2d %s %s\n", cmt[kOn], "outlierdownweighting", kDefOutlierDW,
          cmt[kOnOn], "iteration for outliers downweighting with Cauchi factor");
  fprintf(strFl, "%s%-20s %.3f %s %s\n", cmt[kOn], "dwfractioncut", kDefDWFrac,
          cmt[kOnOn], "cut outliers with downweighting above this factor");
  fprintf(strFl, "%s%-20s %2d %s %s\n", cmt[kOn], "entries", kDefEntries,
          cmt[kOnOn], "min entries per DOF to allow its variation");
  //
  fprintf(strFl, "\n\n\n%s%-20s %s %s\n\n\n", cmt[kOff], "CFiles", cmt[kOnOn], "put below *.mille files list");
  //
  if (mVtxSens) {
    mVtxSens->writePedeInfo(parFl, opt);
  }
  //
  for (auto id = DetID::First; id <= DetID::Last; id++) {
    AlignableDetector* det = getDetector(id);
    if (!det || det->isDisabled()) {
      continue;
    }
    det->writePedeInfo(parFl, opt);
    //
  }
  //
  writePedeConstraints();
  //
  fclose(strFl);
  fclose(parFl);
  //
}

//___________________________________________________________
bool Controller::readParameters(const std::string& parfile, bool useErrors)
{
  // read parameters file (millepede output)
  if (mNDOFs < 1) {
    LOG(error) << "Something is wrong in init, no DOFs found: mNDOFs=" << mNDOFs << " N GloParVal=" << mGloParVal.size() << " N GloParErr=" << mGloParErr.size();
  }
  std::ifstream inpf(parfile);
  if (!inpf.good()) {
    LOGP(fatal, "Failed on input filename {}", parfile);
  }
  mGloParVal.resize(mNDOFs);
  if (useErrors) {
    mGloParErr.resize(mNDOFs);
  }
  int cnt = 0;
  TString fline;
  fline.ReadLine(inpf);
  fline = fline.Strip(TString::kBoth, ' ');
  fline.ToLower();
  if (!fline.BeginsWith("parameter")) {
    LOGP(fatal, "First line of {} is not parameter keyword: {}", parfile, fline.Data());
  }
  double v0, v1, v2;
  int lab, asg = 0, asg0 = 0;
  while (fline.ReadLine(inpf)) {
    cnt++;
    fline = fline.Strip(TString::kBoth, ' ');
    if (fline.BeginsWith("!") || fline.BeginsWith("*")) {
      continue;
    } // ignore comment
    int nr = sscanf(fline.Data(), "%d%lf%lf%lf", &lab, &v0, &v1, &v2);
    if (nr < 3) {
      LOG(error) << "Expected to read at least 3 numbers, got " << nr << ", this is NOT milleped output";
      LOG(fatal) << "line (" << cnt << ") was: " << fline.Data();
    }
    if (nr == 3) {
      asg0++;
    }
    int parID = label2ParID(lab);
    if (parID < 0 || parID >= mNDOFs) {
      LOG(fatal) << "Invalid label " << lab << " at line " << cnt << " -> ParID=" << parID;
    }
    mGloParVal[parID] = -v0;
    if (useErrors) {
      mGloParErr[parID] = v1;
    }
    asg++;
    //
  };
  LOG(info) << "Read " << cnt << " lines, assigned " << asg << " values, " << asg0 << " dummy";
  //
  return true;
}

//______________________________________________________
void Controller::checkConstraints(const char* params)
{
  // check how the constraints are satisfied with already uploaded or provided params
  //
  if (params && !readParameters(params)) {
    LOG(error) << "Failed to load parameters from " << params;
    return;
  }
  //
  int ncon = getNConstraints();
  for (int icon = 0; icon < ncon; icon++) {
    getConstraint(icon).checkConstraint();
  }
  //
}

//___________________________________________________________
void Controller::MPRec2Mille(const std::string& mprecfile, const std::string& millefile, bool bindata)
{
  // converts MPRecord tree to millepede binary format
  TFile* flmpr = TFile::Open(mprecfile.c_str());
  if (!flmpr) {
    LOG(fatal) << "Failed to open MPRecord file " << mprecfile;
    return;
  }
  TTree* mprTree = (TTree*)flmpr->Get("mpTree");
  if (!mprTree) {
    LOG(fatal) << "No mpTree in xMPRecord file " << mprecfile;
    return;
  }
  MPRec2Mille(mprTree, millefile, bindata);
  delete mprTree;
  flmpr->Close();
  delete flmpr;
}

//___________________________________________________________
void Controller::MPRec2Mille(TTree* mprTree, const std::string& millefile, bool bindata)
{
  // converts MPRecord tree to millepede binary format
  //
  TBranch* br = mprTree->GetBranch("mprec");
  if (!br) {
    LOG(error) << "provided tree does not contain branch mprec";
    return;
  }
  Millepede2Record mrec, *rec = &mrec;
  br->SetAddress(&rec);
  int nent = mprTree->GetEntries();
  std::string mlname = millefile;
  if (mlname.empty()) {
    mlname = "mpRec2mpData";
  }
  if (!o2::utils::Str::endsWith(mlname, sMPDataExt)) {
    mlname += sMPDataExt;
  }
  Mille mille(mlname, bindata);
  std::vector<float> buffLoc;
  for (int i = 0; i < nent; i++) {
    br->GetEntry(i);
    int nr = rec->getNResid(); // number of residual records
    int nloc = rec->getNVarLoc();
    auto recDGlo = rec->getArrGlo();
    auto recDLoc = rec->getArrLoc();
    auto recLabLoc = rec->getArrLabLoc();
    auto recLabGlo = rec->getArrLabGlo();
    //
    for (int ir = 0; ir < nr; ir++) {
      buffLoc.clear();
      buffLoc.resize(nloc);
      int ndglo = rec->getNDGlo(ir);
      int ndloc = rec->getNDLoc(ir);
      // fill 0-suppressed array from MPRecord to non-0-suppressed array of Mille
      for (int l = ndloc; l--;) {
        buffLoc[recLabLoc[l]] = recDLoc[l];
      }
      //
      mille.mille(nloc, buffLoc.data(), ndglo, recDGlo, recLabGlo, rec->getResid(ir), rec->getResErr(ir));
      //
      recLabGlo += ndglo; // next record
      recDGlo += ndglo;
      recLabLoc += ndloc;
      recDLoc += ndloc;
    }
    mille.finalise();
  }
  br->SetAddress(nullptr);
}

//____________________________________________________________
void Controller::printLabels() const
{
  // print global IDs and Labels
  for (int i = 0; i < mNDOFs; i++) {
    printf("%5d %s\n", i, getDOFLabelTxt(i));
  }
}

//____________________________________________________________
int Controller::label2ParID(int lab) const
{
  // convert Mille label to ParID (slow)
  auto it = mLbl2ID.find(lab);
  if (it == mLbl2ID.end()) {
    LOGP(fatal, "Label {} is not mapped to any parameter", lab);
  }
  return it->second - 1;
}

//____________________________________________________________
void Controller::addAutoConstraints()
{
  // add default constraints on children cumulative corrections within the volumes
  for (auto id = DetID::First; id <= DetID::Last; id++) {
    AlignableDetector* det = getDetector(id);
    if (!det || det->isDisabled()) {
      continue;
    }
    det->addAutoConstraints();
  }
  LOG(info) << "Added " << getNConstraints() << " automatic constraints";
}

//____________________________________________________________
void Controller::writePedeConstraints() const
{
  // write constraints file
  FILE* conFl = fopen(AlignConfig::Instance().mpConFileName.c_str(), "w+");
  //
  int nconstr = getNConstraints();
  for (int icon = 0; icon < nconstr; icon++) {
    getConstraint(icon).writeChildrenConstraints(conFl);
  }
  //
  fclose(conFl);
}

//____________________________________________________________
void Controller::fixLowStatFromDOFStat(int thresh)
{
  // fix DOFs having stat below threshold
  //
  if (mNDOFs != mDOFStat.getNDOFs()) {
    LOG(error) << "Discrepancy between NDOFs=" << mNDOFs << " of and statistics object: " << mDOFStat.getNDOFs();
    return;
  }
  for (int parID = 0; parID < mNDOFs; parID++) {
    if (mDOFStat.getStat(parID) >= thresh) {
      continue;
    }
    mGloParErr[parID] = -999.;
  }
  //
}

//______________________________________________
void Controller::checkSol(TTree* mpRecTree, bool store,
                          bool verbose, bool loc, const char* outName)
{
  // do fast check of pede solution with MPRecord tree
  ResidualsControllerFast* rLG = store ? new ResidualsControllerFast() : nullptr;
  ResidualsControllerFast* rL = store && loc ? new ResidualsControllerFast() : nullptr;
  TTree *trLG = nullptr, *trL = nullptr;
  TFile* outFile = nullptr;
  if (store) {
    TString outNS = outName;
    if (outNS.IsNull()) {
      outNS = "resFast";
    }
    if (!outNS.EndsWith(".root")) {
      outNS += ".root";
    }
    outFile = TFile::Open(outNS.Data(), "recreate");
    trLG = new TTree("resFLG", "Fast residuals with LG correction");
    trLG->Branch("rLG", "ResidualsControllerFast", &rLG);
    //
    if (rL) {
      trL = new TTree("resFL", "Fast residuals with L correction");
      trL->Branch("rL", "ResidualsControllerFast", &rL);
    }
  }
  //
  Millepede2Record* rec = new Millepede2Record();
  mpRecTree->SetBranchAddress("mprec", &rec);
  int nrec = mpRecTree->GetEntriesFast();
  for (int irec = 0; irec < nrec; irec++) {
    mpRecTree->GetEntry(irec);
    checkSol(rec, rLG, rL, verbose, loc);
    // store even in case of failure, to have the trees aligned with controlRes
    if (trLG) {
      trLG->Fill();
    }
    if (trL) {
      trL->Fill();
    }
  }
  //
  // save
  if (trLG) {
    outFile->cd();
    trLG->Write();
    delete trLG;
    if (trL) {
      trL->Write();
      delete trL;
    }
    outFile->Close();
    delete outFile;
  }
  delete rLG;
  delete rL;
  //
}

//______________________________________________
bool Controller::checkSol(Millepede2Record* rec,
                          ResidualsControllerFast* rLG, ResidualsControllerFast* rL,
                          bool verbose, bool loc)
{
  LOG(fatal) << __PRETTY_FUNCTION__ << " is disabled";
  //FIXME(milettri): needs AliSymMatrix
  //  // Check pede solution using derivates, rather than updated geometry
  //  // If loc==true, also produces residuals for current geometry,
  //  // neglecting global corrections
  //  //
  //  if (rL){
  //    loc = true;} // if local sol. tree asked, always evaluate it
  //  //
  //  int nres = rec->getNResid();
  //  //
  //  const float* recDGlo = rec->getArrGlo();
  //  const float* recDLoc = rec->getArrLoc();
  //  const short* recLabLoc = rec->getArrLabLoc();
  //  const int* recLabGlo = rec->getArrLabGlo();
  //  int nvloc = rec->getNVarLoc();
  //  //
  //  // count number of real measurement duplets and material correction fake 4-plets
  //  int nPoints = 0;
  //  int nMatCorr = 0;
  //  for (int irs = 0; irs < nres; irs++) {
  //    if (rec->getNDGlo(irs) > 0) {
  //      if (irs == nres - 1 || rec->getNDGlo(irs + 1) == 0){
  //        LOG(fatal) << ("Real coordinate measurements must come in pairs");}
  //      nPoints++;
  //      irs++; // skip 2nd
  //      continue;
  //    } else if (rec->getResid(irs) == 0 && rec->getVolID(irs) == -1) { // material corrections have 0 residual
  //      nMatCorr++;
  //    } else { // might be fixed parameter, global derivs are skept
  //      nPoints++;
  //      irs++; // skip 2nd
  //      continue;
  //    }
  //  }
  //  //
  //  if (nMatCorr % 4){
  //    LOG(warning) << "Error? NMatCorr=" << nMatCorr << " is not multiple of 4";}
  //  //
  //  if (rLG) {
  //    rLG->Clear();
  //    rLG->setNPoints(nPoints);
  //    rLG->setNMatSol(nMatCorr);
  //    rLG->setCosmic(rec->isCosmic());
  //  }
  //  if (rL) {
  //    rL->Clear();
  //    rL->setNPoints(nPoints);
  //    rL->setNMatSol(nMatCorr);
  //    rL->setCosmic(rec->isCosmic());
  //  }
  //  //
  //  AliSymMatrix* matpG = new AliSymMatrix(nvloc);
  //  TVectorD *vecp = 0, *vecpG = new TVectorD(nvloc);
  //  //
  //  if (loc){
  //    vecp = new TVectorD(nvloc);}
  //  //
  //  float chi2Ini = 0, chi2L = 0, chi2LG = 0;
  //  //
  //  // residuals, accounting for global solution
  //  double* resid = new double[nres];
  //  int* volID = new int[nres];
  //  for (int irs = 0; irs < nres; irs++) {
  //    double resOr = rec->getResid(irs);
  //    resid[irs] = resOr;
  //    //
  //    int ndglo = rec->getNDGlo(irs);
  //    int ndloc = rec->getNDLoc(irs);
  //    volID[irs] = 0;
  //    for (int ig = 0; ig < ndglo; ig++) {
  //      int lbI = recLabGlo[ig];
  //      int idP = label2ParID(lbI);
  //      if (idP < 0){
  //        LOG(fatal) << "Did not find parameted for label " << lbI;}
  //      double parVal = getGloParVal()[idP];
  //      //      resid[irs] -= parVal*recDGlo[ig];
  //      resid[irs] += parVal * recDGlo[ig];
  //      if (!ig) {
  //        AlignableVolume* vol = getVolOfDOFID(idP);
  //        if (vol){
  //          volID[irs] = vol->getVolID();}
  //        else
  //          volID[irs] = -2; // calibration DOF !!! TODO
  //      }
  //    }
  //    //
  //    double sg2inv = rec->getResErr(irs);
  //    sg2inv = 1. / (sg2inv * sg2inv);
  //    //
  //    chi2Ini += resid[irs] * resid[irs] * sg2inv; // chi accounting for global solution only
  //    //
  //    // Build matrix to solve local parameters
  //    for (int il = 0; il < ndloc; il++) {
  //      int lbLI = recLabLoc[il]; // id of local variable
  //      (*vecpG)[lbLI] -= recDLoc[il] * resid[irs] * sg2inv;
  //      if (loc){
  //        (*vecp)[lbLI] -= recDLoc[il] * resOr * sg2inv;}
  //      for (int jl = il + 1; jl--;) {
  //        int lbLJ = recLabLoc[jl]; // id of local variable
  //        (*matpG)(lbLI, lbLJ) += recDLoc[il] * recDLoc[jl] * sg2inv;
  //      }
  //    }
  //    //
  //    recLabGlo += ndglo; // prepare for next record
  //    recDGlo += ndglo;
  //    recLabLoc += ndloc;
  //    recDLoc += ndloc;
  //    //
  //  }
  //  //
  //  if (rL){
  //    rL->setChi2Ini(chi2Ini);}
  //  if (rLG){
  //    rLG->setChi2Ini(chi2Ini);}
  //  //
  //  TVectorD vecSol(nvloc);
  //  TVectorD vecSolG(nvloc);
  //  //
  //  if (!matpG->SolveChol(*vecpG, vecSolG, false)) {
  //    LOG(info) << "Failed to solve track corrected for globals";
  //    delete matpG;
  //    matpG = 0;
  //  } else if (loc) { // solution with local correction only
  //    if (!matpG->SolveChol(*vecp, vecSol, false)) {
  //      LOG(info) << "Failed to solve track corrected for globals";
  //      delete matpG;
  //      matpG = 0;
  //    }
  //  }
  //  delete vecpG;
  //  delete vecp;
  //  if (!matpG) { // failed
  //    delete[] resid;
  //    delete[] volID;
  //    if (rLG){
  //      rLG->Clear();}
  //    if (rL){
  //      rL->Clear();}
  //    return false;
  //  }
  //  // check
  //  recDGlo = rec->getArrGlo();
  //  recDLoc = rec->getArrLoc();
  //  recLabLoc = rec->getArrLabLoc();
  //  recLabGlo = rec->getArrLabGlo();
  //  //
  //  if (verbose) {
  //    printf(loc ? "Sol L/LG:\n" : "Sol LG:\n");
  //    int nExtP = (nvloc % 4) ? 5 : 4;
  //    for (int i = 0; i < nExtP; i++){
  //      loc ? printf("%+.3e/%+.3e ", vecSol[i], vecSolG[i]) : printf("%+.3e ", vecSolG[i]);}
  //    printf("\n");
  //    bool nln = true;
  //    int cntL = 0;
  //    for (int i = nExtP; i < nvloc; i++) {
  //      nln = true;
  //      loc ? printf("%+.3e/%+.3e ", vecSol[i], vecSolG[i]) : printf("%+.3e ", vecSolG[i]);
  //      if (((++cntL) % 4) == 0) {
  //        printf("\n");
  //        nln = false;
  //      }
  //    }
  //    if (!nln){
  //      printf("\n");}
  //    if (loc){
  //      printf("%3s (%9s) %6s | [ %7s:%7s ] [ %7s:%7s ]\n", "Pnt", "Label",
  //             "Sigma", "resid", "pull/L ", "resid", "pull/LG");}
  //    else{
  //      printf("%3s (%9s) %6s | [ %7s:%7s ]\n", "Pnt", "Label",
  //             "Sigma", "resid", "pull/LG");}
  //  }
  //  int idMeas = -1, pntID = -1, matID = -1;
  //  for (int irs = 0; irs < nres; irs++) {
  //    double resOr = rec->getResid(irs);
  //    double resL = resOr;
  //    double resLG = resid[irs];
  //    double sg = rec->getResErr(irs);
  //    double sg2Inv = 1 / (sg * sg);
  //    //
  //    int ndglo = rec->getNDGlo(irs);
  //    int ndloc = rec->getNDLoc(irs);
  //    //
  //    for (int il = 0; il < ndloc; il++) {
  //      int lbLI = recLabLoc[il]; // id of local variable
  //      resL += recDLoc[il] * vecSol[lbLI];
  //      resLG += recDLoc[il] * vecSolG[lbLI];
  //    }
  //    //
  //    chi2L += resL * resL * sg2Inv;    // chi accounting for global solution only
  //    chi2LG += resLG * resLG * sg2Inv; // chi accounting for global solution only
  //    //
  //    if (ndglo || resOr != 0) { // real measurement
  //      idMeas++;
  //      if (idMeas > 1){
  //        idMeas = 0;}
  //      if (idMeas == 0){
  //        pntID++;} // measurements come in pairs
  //      int lbl = rec->getVolID(irs);
  //      lbl = ndglo ? recLabGlo[0] : 0; // TMP, until VolID is filled // RS!!!!
  //      if (rLG) {
  //        rLG->setResSigMeas(pntID, idMeas, resLG, sg);
  //        if (idMeas == 0){
  //          rLG->setLabel(pntID, lbl, volID[irs]);}
  //      }
  //      if (rL) {
  //        rL->setResSigMeas(pntID, idMeas, resL, sg);
  //        if (idMeas == 0){
  //          rL->setLabel(pntID, lbl, volID[irs]);}
  //      }
  //    } else {
  //      matID++; // mat.correcitons come in 4-plets, but we fill each separately
  //      //
  //      if (rLG){
  //        rLG->setMatCorr(matID, resLG, sg);}
  //      if (rL){
  //        rL->setMatCorr(matID, resL, sg);}
  //    }
  //    //
  //    if (verbose) {
  //      int lbl = rec->getVolID(irs);
  //      lbl = ndglo ? recLabGlo[0] : (resOr == 0 ? -1 : 0); // TMP, until VolID is filled // RS!!!!
  //      if (loc){
  //        printf("%3d (%9d) %6.4f | [%+.2e:%+7.2f] [%+.2e:%+7.2f]\n",
  //               irs, lbl, sg, resL, resL / sg, resLG, resLG / sg);}
  //      else
  //        printf("%3d (%9d) %6.4f | [%+.2e:%+7.2f]\n",
  //               irs, lbl, sg, resLG, resLG / sg);
  //    }
  //    //
  //    recLabGlo += ndglo; // prepare for next record
  //    recDGlo += ndglo;
  //    recLabLoc += ndloc;
  //    recDLoc += ndloc;
  //  }
  //  if (rL){
  //    rL->setChi2(chi2L);}
  //  if (rLG){
  //    rLG->setChi2(chi2LG);}
  //  //
  //  if (verbose) {
  //    printf("Chi: G = %e | LG = %e", chi2Ini, chi2LG);
  //    if (loc){
  //      printf(" | L = %e", chi2L);}
  //    printf("\n");
  //  }
  //  // store track corrections
  //  int nTrCor = nvloc - matID - 1;
  //  for (int i = 0; i < nTrCor; i++) {
  //    if (rLG){
  //      rLG->getTrCor()[i] = vecSolG[i];}
  //    if (rL){
  //      rL->getTrCor()[i] = vecSol[i];}
  //  }
  //  //
  //  delete[] resid;
  //  delete[] volID;
  return true;
}

//______________________________________________
void Controller::applyAlignmentFromMPSol()
{
  // apply alignment from millepede solution array to reference alignment level
  LOG(info) << "Applying alignment from Millepede solution";
  for (auto id = DetID::First; id <= DetID::Last; id++) {
    AlignableDetector* det = getDetector(id);
    if (!det || det->isDisabled()) {
      continue;
    }
    det->applyAlignmentFromMPSol();
  }
  setMPAlignDone();
  //
}

//______________________________________________
void Controller::expandGlobalsBy(int n)
{
  // expand global param contaiers by n
  int snew = n + mGloParVal.size();
  mGloParVal.resize(snew);
  mGloParErr.resize(snew);
  mGloParLab.resize(snew);
  mNDOFs += n;
}

} // namespace align
} // namespace o2
