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

#include <vector>
#include <TStopwatch.h>
#include "DataFormatsGlobalTracking/RecoContainer.h"
#include "DataFormatsGlobalTracking/RecoContainerCreateTracksVariadic.h"
#include "ReconstructionDataFormats/TrackTPCITS.h"
#include "ReconstructionDataFormats/GlobalTrackID.h"
#include "DetectorsBase/Propagator.h"
#include "DetectorsBase/GeometryManager.h"
#include "SimulationDataFormat/MCEventLabel.h"
#include "SimulationDataFormat/MCUtils.h"
#include "CommonDataFormat/BunchFilling.h"
#include "CommonUtils/NameConf.h"
#include "DataFormatsFT0/RecPoints.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/CCDBParamSpec.h"
#include "FT0Reconstruction/InteractionTag.h"
#include "ITSMFTBase/DPLAlpideParam.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "DetectorsBase/GRPGeomHelper.h"
#include "GlobalTrackingStudy/TrackingStudy.h"
#include "TPCBase/ParameterElectronics.h"
#include "ReconstructionDataFormats/PrimaryVertex.h"
#include "ReconstructionDataFormats/PrimaryVertexExt.h"
#include "DataFormatsFT0/RecPoints.h"
#include "CommonUtils/TreeStreamRedirector.h"
#include "ReconstructionDataFormats/VtxTrackRef.h"
#include "ReconstructionDataFormats/DCA.h"
#include "Steer/MCKinematicsReader.h"

namespace o2::trackstudy
{

using namespace o2::framework;
using DetID = o2::detectors::DetID;
using DataRequest = o2::globaltracking::DataRequest;

using PVertex = o2::dataformats::PrimaryVertex;
using V2TRef = o2::dataformats::VtxTrackRef;
using VTIndex = o2::dataformats::VtxTrackIndex;
using GTrackID = o2::dataformats::GlobalTrackID;
using TBracket = o2::math_utils::Bracketf_t;

using timeEst = o2::dataformats::TimeStampWithError<float, float>;

class TrackingStudySpec : public Task
{
 public:
  TrackingStudySpec(std::shared_ptr<DataRequest> dr, std::shared_ptr<o2::base::GRPGeomRequest> gr, GTrackID::mask_t src, bool useMC)
    : mDataRequest(dr), mGGCCDBRequest(gr), mTracksSrc(src), mUseMC(useMC) {}
  ~TrackingStudySpec() final = default;
  void init(InitContext& ic) final;
  void run(ProcessingContext& pc) final;
  void endOfStream(EndOfStreamContext& ec) final;
  void finaliseCCDB(ConcreteDataMatcher& matcher, void* obj) final;
  void process(o2::globaltracking::RecoContainer& recoData);

 private:
  void updateTimeDependentParams(ProcessingContext& pc);
  std::shared_ptr<DataRequest> mDataRequest;
  std::shared_ptr<o2::base::GRPGeomRequest> mGGCCDBRequest;
  bool mUseMC{false}; ///< MC flag
  std::unique_ptr<o2::utils::TreeStreamRedirector> mDBGOut;
  std::unique_ptr<o2::utils::TreeStreamRedirector> mDBGOutVtx;
  float mITSROFrameLengthMUS = 0.;
  int mMaxNeighbours = 3;
  float mMaxVTTimeDiff = 25.; // \mus
  GTrackID::mask_t mTracksSrc{};
  o2::steer::MCKinematicsReader mcReader; // reader of MC information
};

void TrackingStudySpec::init(InitContext& ic)
{
  o2::base::GRPGeomHelper::instance().setRequest(mGGCCDBRequest);
  mDBGOut = std::make_unique<o2::utils::TreeStreamRedirector>("trackStudy.root", "recreate");
  mDBGOutVtx = std::make_unique<o2::utils::TreeStreamRedirector>("trackStudyVtx.root", "recreate");

  mMaxVTTimeDiff = ic.options().get<float>("max-vtx-timediff");
  mMaxNeighbours = ic.options().get<int>("max-vtx-neighbours");
}

void TrackingStudySpec::run(ProcessingContext& pc)
{
  o2::globaltracking::RecoContainer recoData;
  recoData.collectData(pc, *mDataRequest.get()); // select tracks of needed type, with minimal cuts, the real selected will be done in the vertexer
  updateTimeDependentParams(pc);                 // Make sure this is called after recoData.collectData, which may load some conditions
  process(recoData);
}

void TrackingStudySpec::updateTimeDependentParams(ProcessingContext& pc)
{
  o2::base::GRPGeomHelper::instance().checkUpdates(pc);
  static bool initOnceDone = false;
  if (!initOnceDone) { // this params need to be queried only once
    initOnceDone = true;
    // Note: reading of the ITS AlpideParam needed for ITS timing is done by the RecoContainer
    auto grp = o2::base::GRPGeomHelper::instance().getGRPECS();
    const auto& alpParams = o2::itsmft::DPLAlpideParam<o2::detectors::DetID::ITS>::Instance();
    if (!grp->isDetContinuousReadOut(DetID::ITS)) {
      mITSROFrameLengthMUS = alpParams.roFrameLengthTrig / 1.e3; // ITS ROFrame duration in \mus
    } else {
      mITSROFrameLengthMUS = alpParams.roFrameLengthInBC * o2::constants::lhc::LHCBunchSpacingNS * 1e-3; // ITS ROFrame duration in \mus
    }
  }
}

void TrackingStudySpec::process(o2::globaltracking::RecoContainer& recoData)
{
  auto pvvec = recoData.getPrimaryVertices();
  auto trackIndex = recoData.getPrimaryVertexMatchedTracks(); // Global ID's for associated tracks
  auto vtxRefs = recoData.getPrimaryVertexMatchedTrackRefs(); // references from vertex to these track IDs
  auto prop = o2::base::Propagator::Instance();
  auto FITInfo = recoData.getFT0RecPoints();
  static int TFCount = 0;
  int nv = vtxRefs.size() - 1;
  o2::dataformats::PrimaryVertexExt pveDummy;
  std::vector<o2::dataformats::PrimaryVertexExt> pveVec;

  for (int iv = 0; iv < nv; iv++) {
    LOGP(debug, "processing PV {} of {}", iv, nv);

    const auto& vtref = vtxRefs[iv];
    auto pv = pvvec[iv];
    auto& pve = pveVec.emplace_back();
    static_cast<o2::dataformats::PrimaryVertex&>(pve) = pv;

    float bestTimeDiff = 1000, bestTime = -999;
    int bestFTID = -1;
    if (mTracksSrc[GTrackID::FT0]) {
      for (int ift0 = vtref.getFirstEntryOfSource(GTrackID::FT0); ift0 < vtref.getFirstEntryOfSource(GTrackID::FT0) + vtref.getEntriesOfSource(GTrackID::FT0); ift0++) {
        const auto& ft0 = FITInfo[trackIndex[ift0]];
        if (ft0.isValidTime(o2::ft0::RecPoints::TimeMean) && ft0.getTrigger().getAmplA() + ft0.getTrigger().getAmplC() > 20) {
          auto fitTime = ft0.getInteractionRecord().differenceInBCMUS(recoData.startIR);
          if (std::abs(fitTime - pv.getTimeStamp().getTimeStamp()) < bestTimeDiff) {
            bestTimeDiff = fitTime - pv.getTimeStamp().getTimeStamp();
            bestFTID = trackIndex[ift0];
          }
        }
      }
    } else {
      LOGP(warn, "FT0 is not requested, cannot set complete vertex info");
    }
    if (bestFTID >= 0) {
      pve.FT0A = FITInfo[bestFTID].getTrigger().getAmplA();
      pve.FT0Amp = pve.FT0A + FITInfo[bestFTID].getTrigger().getAmplC();
      pve.FT0Time = FITInfo[bestFTID].getInteractionRecord().differenceInBCMUS(recoData.startIR);
    }
    pve.VtxID = iv;
    for (int is = 0; is < GTrackID::NSources; is++) {
      bool skipTracks = (!GTrackID::getSourceDetectorsMask(is)[GTrackID::ITS] || !mTracksSrc[is] || !recoData.isTrackSourceLoaded(is));
      int idMin = vtref.getFirstEntryOfSource(is), idMax = idMin + vtref.getEntriesOfSource(is);
      for (int i = idMin; i < idMax; i++) {
        auto vid = trackIndex[i];
        bool pvCont = vid.isPVContributor();
        if (pvCont) {
          pve.nSrc[is]++;
        }
        if (skipTracks) {
          continue;
        }
        bool ambig = vid.isAmbiguous();
        auto trc = recoData.getTrackParam(vid);
        float xmin = trc.getX();
        o2::dataformats::DCA dca;
        if (!prop->propagateToDCA(pv, trc, prop->getNominalBz(), 2., o2::base::PropagatorF::MatCorrType::USEMatCorrLUT, &dca)) {
          continue;
        }
        float ttime = 0, ttimeE = 0;
        recoData.getTrackTime(vid, ttime, ttimeE);
        (*mDBGOut) << "dca"
                   << "tfID=" << TFCount << "ttime=" << ttime << "ttimeE=" << ttimeE
                   << "gid=" << vid << "pv=" << pv << "trc=" << trc << "pvCont=" << pvCont << "ambig=" << ambig << "dca=" << dca << "xmin=" << xmin << "\n";
      }
    }
  }
  int nvtot = mMaxNeighbours < 0 ? -1 : (int)pveVec.size();
  for (int cnt = 0; cnt < nvtot; cnt++) {
    const auto& pve = pveVec[cnt];
    float tv = pve.getTimeStamp().getTimeStamp();
    std::vector<o2::dataformats::PrimaryVertexExt> pveT(mMaxNeighbours); // neighbours in time
    std::vector<o2::dataformats::PrimaryVertexExt> pveZ(mMaxNeighbours); // neighbours in Z
    std::vector<int> idT(mMaxNeighbours), idZ(mMaxNeighbours);
    std::vector<float> dT(mMaxNeighbours), dZ(mMaxNeighbours);
    for (int i = 0; i < mMaxNeighbours; i++) {
      idT[i] = idZ[i] = -1;
      dT[i] = mMaxVTTimeDiff;
      dZ[i] = 1e9;
    }
    int cntM = cnt - 1, cntP = cnt + 1;
    for (; cntM >= 0; cntM--) { // backward
      const auto& vt = pveVec[cntM];
      auto dtime = std::abs(tv - vt.getTimeStamp().getTimeStamp());
      if (dtime > mMaxVTTimeDiff) {
        continue;
      }
      for (int i = 0; i < mMaxNeighbours; i++) {
        if (dT[i] > dtime) {
          dT[i] = dtime;
          idT[i] = cntM;
          break;
        }
      }
      auto dz = std::abs(pve.getZ() - vt.getZ());
      for (int i = 0; i < mMaxNeighbours; i++) {
        if (dZ[i] > dz) {
          dZ[i] = dz;
          idZ[i] = cntM;
          break;
        }
      }
    }
    for (; cntP < nvtot; cntP++) { // forward
      const auto& vt = pveVec[cntP];
      auto dtime = std::abs(tv - vt.getTimeStamp().getTimeStamp());
      if (dtime > mMaxVTTimeDiff) {
        continue;
      }
      for (int i = 0; i < mMaxNeighbours; i++) {
        if (dT[i] > dtime) {
          dT[i] = dtime;
          idT[i] = cntP;
          break;
        }
      }
      auto dz = std::abs(pve.getZ() - vt.getZ());
      for (int i = 0; i < mMaxNeighbours; i++) {
        if (dZ[i] > dz) {
          dZ[i] = dz;
          idZ[i] = cntP;
          break;
        }
      }
    }
    for (int i = 0; i < mMaxNeighbours; i++) {
      if (idT[i] != -1) {
        pveT[i] = pveVec[idT[i]];
      } else {
        break;
      }
    }
    for (int i = 0; i < mMaxNeighbours; i++) {
      if (idZ[i] != -1) {
        pveZ[i] = pveVec[idZ[i]];
      } else {
        break;
      }
    }
    (*mDBGOutVtx) << "pvExt"
                  << "pve=" << pve
                  << "pveT=" << pveT
                  << "pveZ=" << pveZ
                  << "tfID=" << TFCount
                  << "\n";
  }

  TFCount++;
}

void TrackingStudySpec::endOfStream(EndOfStreamContext& ec)
{
  mDBGOut.reset();
  mDBGOutVtx.reset();
}

void TrackingStudySpec::finaliseCCDB(ConcreteDataMatcher& matcher, void* obj)
{
  if (o2::base::GRPGeomHelper::instance().finaliseCCDB(matcher, obj)) {
    return;
  }
}

DataProcessorSpec getTrackingStudySpec(GTrackID::mask_t srcTracks, GTrackID::mask_t srcClusters, bool useMC)
{
  std::vector<OutputSpec> outputs;
  auto dataRequest = std::make_shared<DataRequest>();

  dataRequest->requestTracks(srcTracks, useMC);
  dataRequest->requestClusters(srcClusters, useMC);
  dataRequest->requestPrimaryVertertices(useMC);
  auto ggRequest = std::make_shared<o2::base::GRPGeomRequest>(false,                             // orbitResetTime
                                                              true,                              // GRPECS=true
                                                              true,                              // GRPLHCIF
                                                              true,                              // GRPMagField
                                                              true,                              // askMatLUT
                                                              o2::base::GRPGeomRequest::Aligned, // geometry
                                                              dataRequest->inputs,
                                                              true);

  return DataProcessorSpec{
    "track-study",
    dataRequest->inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<TrackingStudySpec>(dataRequest, ggRequest, srcTracks, useMC)},
    Options{
      {"max-vtx-neighbours", VariantType::Int, 2, {"Max PV neighbours fill, no PV study if < 0"}},
      {"max-vtx-timediff", VariantType::Float, 25.f, {"Max PV time difference to consider"}},
    }};
}

} // namespace o2::trackstudy
