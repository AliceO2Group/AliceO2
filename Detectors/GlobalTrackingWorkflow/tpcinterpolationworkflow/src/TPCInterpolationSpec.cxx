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

/// @file  TPCInterpolationSpec.cxx

#include <vector>
#include <unordered_map>
#include <algorithm>
#include <cmath>

#include "DataFormatsITS/TrackITS.h"
#include "ITSBase/GeometryTGeo.h"
#include "ReconstructionDataFormats/TrackTPCITS.h"
#include "DataFormatsTPC/TrackTPC.h"
#include "DataFormatsTPC/ClusterNative.h"
#include "DataFormatsTPC/Defs.h"
#include "DataFormatsTPC/WorkflowHelper.h"
#include "DataFormatsTRD/TrackTRD.h"
#include "DetectorsBase/GeometryManager.h"
#include "DetectorsBase/Propagator.h"
#include "TPCInterpolationWorkflow/TPCInterpolationSpec.h"
#include "DataFormatsGlobalTracking/RecoContainer.h"
#include "DataFormatsGlobalTracking/RecoContainerCreateTracksVariadic.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "SpacePoints/SpacePointsCalibParam.h"
#include "SpacePoints/SpacePointsCalibConfParam.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/DeviceSpec.h"
#include "Steer/MCKinematicsReader.h"
#include "SimulationDataFormat/TrackReference.h"
#include "SimulationDataFormat/O2DatabasePDG.h"

using namespace o2::framework;
using namespace o2::globaltracking;
using GTrackID = o2::dataformats::GlobalTrackID;
using DetID = o2::detectors::DetID;

namespace o2
{
namespace tpc
{

TPCInterpolationDPL::~TPCInterpolationDPL() = default;

void TPCInterpolationDPL::init(InitContext& ic)
{
  //-------- init geometry and field --------//
  mTimer.Stop();
  mTimer.Reset();
  o2::base::GRPGeomHelper::instance().setRequest(mGGCCDBRequest);
  mSlotLength = ic.options().get<uint32_t>("sec-per-slot");
  mProcessSeeds = ic.options().get<bool>("process-seeds");
  mMatCorr = ic.options().get<int>("matCorrType");
  if (mProcessSeeds && mSources != mSourcesMap) {
    LOG(fatal) << "process-seeds option is not compatible with using different track sources for vDrift and map extraction";
  }
  int lane = ic.services().get<const o2::framework::DeviceSpec>().inputTimesliceId;
  int maxLanes = ic.services().get<const o2::framework::DeviceSpec>().maxInputTimeslices;
  mInterpolation.setLane(lane, maxLanes);
  if (mUseMC) {
    if (!mSendTrackData) {
      LOG(warning) << "MC truth is stored aligned with the track data, but send-track-data is not set: no MC truth will be sent";
    }
    mMCReader = std::make_unique<o2::steer::MCKinematicsReader>();
    auto mcContext = ic.options().get<std::string>("mc-collision-context");
    if (!mMCReader->initFromDigitContext(mcContext)) {
      LOG(fatal) << "Could not initialize the MC kinematics reader from " << mcContext;
    }
  }
}

void TPCInterpolationDPL::updateTimeDependentParams(ProcessingContext& pc)
{
  o2::base::GRPGeomHelper::instance().checkUpdates(pc);
  mTPCVDriftHelper.extractCCDBInputs(pc);
  static bool initOnceDone = false;
  if (!initOnceDone) { // this params need to be queried only once
    initOnceDone = true;
    // other init-once stuff
    const auto& param = SpacePointsCalibConfParam::Instance();
    mInterpolation.setSqrtS(o2::base::GRPGeomHelper::instance().getGRPLHCIF()->getSqrtS());
    mInterpolation.setNHBPerTF(o2::base::GRPGeomHelper::getNHBFPerTF());
    mInterpolation.init(mSources, mSourcesMap);
    if (mProcessITSTPConly) {
      mInterpolation.setProcessITSTPConly();
    }
    int nTfs = mSlotLength / (o2::base::GRPGeomHelper::getNHBFPerTF() * o2::constants::lhc::LHCOrbitMUS * 1e-6);
    bool limitTracks = (param.maxTracksPerCalibSlot < 0) ? false : true;
    int nTracksPerTfMax = (nTfs > 0 && limitTracks) ? param.maxTracksPerCalibSlot / nTfs : -1;
    if (nTracksPerTfMax > 0) {
      LOGP(info, "We will stop processing tracks after validating {} tracks per TF, since we want to accumulate {} tracks for a slot with {} TFs",
           nTracksPerTfMax, param.maxTracksPerCalibSlot, nTfs);
      if (param.additionalTracksMap > 0) {
        int nTracksAdditional = param.additionalTracksMap / nTfs;
        LOGP(info, "In addition up to {} additional tracks are processed per TF", nTracksAdditional);
        mInterpolation.setAddTracksForMapPerTF(nTracksAdditional);
      }
    } else if (nTracksPerTfMax < 0) {
      LOG(info) << "The number of processed tracks per TF is not limited";
    } else {
      LOG(error) << "No tracks will be processed. maxTracksPerCalibSlot must be greater than slot length in TFs";
    }
    mInterpolation.setMaxTracksPerTF(nTracksPerTfMax);
    mInterpolation.setMatCorr(static_cast<o2::base::Propagator::MatCorrType>(mMatCorr));
    if (mProcessSeeds) {
      mInterpolation.setProcessSeeds();
    }
    o2::its::GeometryTGeo::Instance()->fillMatrixCache(o2::math_utils::bit2Mask(o2::math_utils::TransformType::T2GRot) | o2::math_utils::bit2Mask(o2::math_utils::TransformType::T2L));
    mInterpolation.setExtDetResid(mExtDetResid);
    mInterpolation.setITSClusterDictionary(mITSDict);
    if (mDebugOutput) {
      mInterpolation.setDumpTrackPoints();
    }
  }
  // we may have other params which need to be queried regularly
  if (mTPCVDriftHelper.isUpdated()) {
    LOGP(info, "Updating TPC fast transform map with new VDrift factor of {} wrt reference {} and DriftTimeOffset correction {} wrt {} from source {}",
         mTPCVDriftHelper.getVDriftObject().corrFact, mTPCVDriftHelper.getVDriftObject().refVDrift,
         mTPCVDriftHelper.getVDriftObject().timeOffsetCorr, mTPCVDriftHelper.getVDriftObject().refTimeOffset,
         mTPCVDriftHelper.getSourceName());
    mInterpolation.setTPCVDrift(mTPCVDriftHelper.getVDriftObject());
    mTPCVDriftHelper.acknowledgeUpdate();
  }
}

void TPCInterpolationDPL::finaliseCCDB(ConcreteDataMatcher& matcher, void* obj)
{
  if (o2::base::GRPGeomHelper::instance().finaliseCCDB(matcher, obj)) {
    return;
  }
  if (mTPCVDriftHelper.accountCCDBInputs(matcher, obj)) {
    return;
  }
  if (matcher == ConcreteDataMatcher("ITS", "CLUSDICT", 0)) {
    LOG(info) << "cluster dictionary updated";
    mITSDict = (const o2::itsmft::TopologyDictionary*)obj;
    return;
  }
}

void TPCInterpolationDPL::run(ProcessingContext& pc)
{
  mTimer.Start(false);
  RecoContainer recoData;
  recoData.collectData(pc, *mDataRequest.get());
  updateTimeDependentParams(pc);
  mInterpolation.prepareInputTrackSample(recoData);
  mInterpolation.process();
  mTimer.Stop();
  LOGF(info, "TPC interpolation timing: Cpu: %.3e Real: %.3e s", mTimer.CpuTime(), mTimer.RealTime());
  pc.outputs().snapshot(Output{"GLO", "UNBINNEDRES", 0}, mInterpolation.getClusterResiduals());
  pc.outputs().snapshot(Output{"GLO", "DETINFORES", 0}, mInterpolation.getClusterResidualsDetInfo());
  pc.outputs().snapshot(Output{"GLO", "TRKREFS", 0}, mInterpolation.getTrackDataCompact());
  if (mSendTrackData) {
    pc.outputs().snapshot(Output{"GLO", "TRKDATA", 0}, mInterpolation.getReferenceTracks());
  }
  if (mDebugOutput) {
    pc.outputs().snapshot(Output{"GLO", "TRKDATAEXT", 0}, mInterpolation.getTrackDataExtended());
  }
  if (mUseMC && mSendTrackData) {
    fillMCTruth(recoData);
    pc.outputs().snapshot(Output{"GLO", "TRKDATAMC", 0}, mTrackDataMC);
  }
  mInterpolation.reset();
}

void TPCInterpolationDPL::fillMCTruth(const RecoContainer& recoData)
{
  // MC truth for every stored TrackData: labels of the ITS-TPC part of the seed and of its ITS and TPC parts, the truth at
  // the ITS outer parameters (the ITS track reference nearest to TrackData::par, propagated to its x with the material
  // correction of the workflow and the mass of the true particle) and the truth at the TPC entrance (the first TPC track
  // reference in time, in the sector frame)
  const auto& trkData = mInterpolation.getReferenceTracks();
  mTrackDataMC.clear();
  mTrackDataMC.resize(trkData.size());
  struct Lookup {
    o2::MCCompLabel lbl;
    uint32_t idx;
    bool its; // ITS outer (true) or TPC entrance (false)
  };
  std::vector<Lookup> lookups;
  lookups.reserve(2 * trkData.size());
  for (size_t i = 0; i < trkData.size(); ++i) {
    auto& mc = mTrackDataMC[i];
    auto gidSet = recoData.getSingleDetectorRefs(trkData[i].gid);
    auto gidITS = gidSet[GTrackID::ITS].isIndexSet() ? gidSet[GTrackID::ITS] : gidSet[GTrackID::ITSAB];
    if (gidSet[GTrackID::ITSTPC].isIndexSet()) {
      mc.label = recoData.getTrackMCLabel(gidSet[GTrackID::ITSTPC]);
    }
    if (gidITS.isIndexSet()) {
      mc.labelITS = recoData.getTrackMCLabel(gidITS);
    }
    if (gidSet[GTrackID::TPC].isIndexSet()) {
      mc.labelTPC = recoData.getTrackMCLabel(gidSet[GTrackID::TPC]);
    }
    if (mc.labelITS.isValid() && mc.labelTPC.isValid() && mc.labelITS.getTrackEventSourceID() != mc.labelTPC.getTrackEventSourceID()) {
      mc.flags |= TrackDataMC::FakeITSTPC;
    }
    const auto& lblITS = mc.labelITS.isValid() ? mc.labelITS : mc.label;
    const auto& lblTPC = mc.labelTPC.isValid() ? mc.labelTPC : mc.label;
    if (lblITS.isValid()) {
      lookups.push_back({lblITS, uint32_t(i), true});
    }
    if (lblTPC.isValid()) {
      lookups.push_back({lblTPC, uint32_t(i), false});
    }
  }
  // the reader loads the kinematics of a whole event (can be >100 MB): process event by event and release it right after
  std::sort(lookups.begin(), lookups.end(), [](const Lookup& a, const Lookup& b) {
    return a.lbl.getSourceID() != b.lbl.getSourceID() ? a.lbl.getSourceID() < b.lbl.getSourceID() : a.lbl.getEventID() < b.lbl.getEventID();
  });
  auto pdgToPID = [](int pdg) {
    switch (std::abs(pdg)) {
      case 11:
        return o2::track::PID(o2::track::PID::Electron);
      case 13:
        return o2::track::PID(o2::track::PID::Muon);
      case 321:
        return o2::track::PID(o2::track::PID::Kaon);
      case 2212:
        return o2::track::PID(o2::track::PID::Proton);
      case 1000010020:
        return o2::track::PID(o2::track::PID::Deuteron);
      case 1000010030:
        return o2::track::PID(o2::track::PID::Triton);
      case 1000020030:
        return o2::track::PID(o2::track::PID::Helium3);
      case 1000020040:
        return o2::track::PID(o2::track::PID::Alpha);
      default:
        return o2::track::PID(o2::track::PID::Pion);
    }
  };
  auto refToPar = [&pdgToPID](const o2::TrackReference& ref, int charge, int pdg, bool sectorAlpha) {
    std::array<float, 3> xyz{ref.X(), ref.Y(), ref.Z()};
    std::array<float, 3> pxyz{ref.Px(), ref.Py(), ref.Pz()};
    return o2::track::TrackPar(xyz, pxyz, charge, sectorAlpha, pdgToPID(pdg));
  };
  const auto matCorr = static_cast<o2::base::Propagator::MatCorrType>(mMatCorr);
  auto prop = o2::base::Propagator::Instance();
  int curSrc = -1;
  int curEv = -1;
  for (const auto& lk : lookups) {
    const auto& lbl = lk.lbl;
    if (lbl.getSourceID() != curSrc || lbl.getEventID() != curEv) {
      if (curSrc >= 0) {
        mMCReader->releaseTracksForSourceAndEvent(curSrc, curEv);
      }
      curSrc = lbl.getSourceID();
      curEv = lbl.getEventID();
    }
    const auto& trk = trkData[lk.idx];
    auto& mc = mTrackDataMC[lk.idx];
    const auto* mcTrk = mMCReader->getTrack(lbl);
    int pdg = mcTrk ? mcTrk->GetPdgCode() : 0;
    const auto* pPDG = mcTrk ? O2DatabasePDG::Instance()->GetParticle(pdg) : nullptr;
    int charge = pPDG ? int(std::lround(pPDG->Charge() / 3.)) : 0; // TParticlePDG charge is in units of |e|/3
    auto refs = mMCReader->getTrackRefs(lbl.getSourceID(), lbl.getEventID(), lbl.getTrackID());
    if (lk.its) { // ITS outer: track reference of the ITS part nearest to TrackData::par
      mc.pdg = pdg;
      const o2::TrackReference* best = nullptr;
      float bestD2 = 1e30f;
      auto xyzReco = trk.par.getXYZGlo();
      for (const auto& ref : refs) {
        if (ref.getDetectorId() != DetID::ITS) {
          continue;
        }
        float dx = ref.X() - xyzReco.X();
        float dy = ref.Y() - xyzReco.Y();
        float dz = ref.Z() - xyzReco.Z();
        float d2 = dx * dx + dy * dy + dz * dz;
        if (d2 < bestD2) {
          bestD2 = d2;
          best = &ref;
        }
      }
      if (best && charge) {
        auto par = refToPar(*best, charge, pdg, false);
        if (par.rotateParam(trk.par.getAlpha()) && prop->PropagateToXBxByBz(par, trk.par.getX(), 0.999f, o2::base::Propagator::MAX_STEP, matCorr)) {
          mc.parITSOut = par;
          mc.distITSRef = std::sqrt(bestD2);
          mc.flags |= TrackDataMC::HasITSOut;
        }
      }
    } else { // TPC entrance: first TPC track reference in time of the TPC part
      if (!mc.labelITS.isValid() && !mc.label.isValid()) {
        mc.pdg = pdg; // no ITS lookup for this track
      }
      const o2::TrackReference* first = nullptr;
      for (const auto& ref : refs) {
        if (ref.getDetectorId() == DetID::TPC && (!first || ref.getTime() < first->getTime())) {
          first = &ref;
        }
      }
      if (first && charge) {
        mc.parTPCIn = refToPar(*first, charge, pdg, true);
        mc.flags |= TrackDataMC::HasTPCIn;
        // association check (loopers, wrong leg, fake): truth at the innermost TPC cluster of the track vs that cluster
        const auto& clRes = mInterpolation.getClusterResiduals();
        const UnbinnedResid* inner = nullptr;
        for (int ic = trk.clIdx.getFirstEntry(); ic < trk.clIdx.getFirstEntry() + trk.clIdx.getEntries(); ++ic) {
          if (clRes[ic].row < constants::MAXGLOBALPADROW && (!inner || clRes[ic].row < inner->row)) {
            inner = &clRes[ic];
          }
        }
        if (inner) {
          auto par = mc.parTPCIn;
          float yCl = inner->y * param::MaxY / 0x7fff + inner->dy * param::MaxResid / 0x7fff;
          float zCl = inner->z * param::MaxZ / 0x7fff + inner->dz * param::MaxResid / 0x7fff;
          if (par.rotateParam(o2::math_utils::sector2Angle(inner->sec)) && prop->PropagateToXBxByBz(par, param::RowX[inner->row], 0.999f, o2::base::Propagator::MAX_STEP, matCorr)) {
            mc.distTPCRef = std::hypot(par.getY() - yCl, par.getZ() - zCl);
          }
        }
      }
    }
  }
  if (curSrc >= 0) {
    mMCReader->releaseTracksForSourceAndEvent(curSrc, curEv);
  }
}

void TPCInterpolationDPL::endOfStream(EndOfStreamContext& ec)
{
  mInterpolation.finalize();
  LOGF(info, "TPC residuals extraction total timing: Cpu: %.3e Real: %.3e s in %d slots",
       mTimer.CpuTime(), mTimer.RealTime(), mTimer.Counter() - 1);
}

DataProcessorSpec getTPCInterpolationSpec(GTrackID::mask_t srcCls, GTrackID::mask_t srcVtx, GTrackID::mask_t srcTrk, GTrackID::mask_t srcTrkMap, bool useMC, bool processITSTPConly, bool sendTrackData, bool debugOutput, bool extDetResid, bool itsStag)
{
  auto dataRequest = std::make_shared<DataRequest>();
  dataRequest->setITSPerLayer(itsStag);
  std::vector<OutputSpec> outputs;

  dataRequest->requestTracks(srcVtx, false);
  dataRequest->requestClusters(srcCls, false);
  dataRequest->requestPrimaryVertices(false);
  if (useMC) { // the MC truth needs only the labels of the ITS-TPC tracks and of their ITS and TPC parts
    dataRequest->requestTracks(GTrackID::getSourcesMask("ITS,TPC,ITS-TPC"), true);
  }

  auto ggRequest = std::make_shared<o2::base::GRPGeomRequest>(false,                             // orbitResetTime
                                                              true,                              // GRPECS=true
                                                              true,                              // GRPLHCIF
                                                              true,                              // GRPMagField
                                                              true,                              // askMatLUT
                                                              o2::base::GRPGeomRequest::Aligned, // geometry
                                                              dataRequest->inputs,
                                                              true);
  o2::tpc::VDriftHelper::requestCCDBInputs(dataRequest->inputs);
  outputs.emplace_back("GLO", "UNBINNEDRES", 0, Lifetime::Timeframe);
  outputs.emplace_back("GLO", "DETINFORES", 0, Lifetime::Timeframe);
  outputs.emplace_back("GLO", "TRKREFS", 0, Lifetime::Timeframe);
  if (sendTrackData) {
    outputs.emplace_back("GLO", "TRKDATA", 0, Lifetime::Timeframe);
  }
  if (debugOutput) {
    outputs.emplace_back("GLO", "TRKDATAEXT", 0, Lifetime::Timeframe);
  }
  if (useMC && sendTrackData) {
    outputs.emplace_back("GLO", "TRKDATAMC", 0, Lifetime::Timeframe);
  }

  return DataProcessorSpec{
    "tpc-track-interpolation",
    dataRequest->inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<TPCInterpolationDPL>(dataRequest, srcTrk, srcTrkMap, ggRequest, useMC, processITSTPConly, sendTrackData, debugOutput, extDetResid)},
    Options{
      {"matCorrType", VariantType::Int, 2, {"material correction type (definition in Propagator.h)"}},
      {"sec-per-slot", VariantType::UInt32, 300u, {"number of seconds per calibration time slot (put 0 for infinite slot length)"}},
      {"process-seeds", VariantType::Bool, false, {"do not remove duplicates, e.g. for ITS-TPC-TRD track also process its seeding ITS-TPC part"}},
      {"mc-collision-context", VariantType::String, "collisioncontext.root", {"collision context used to access the MC kinematics and track references (MC only)"}}}};
}

} // namespace tpc
} // namespace o2
