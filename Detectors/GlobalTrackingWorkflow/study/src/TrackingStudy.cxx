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
#include "DataFormatsCalibration/MeanVertexObject.h"
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
#include "MathUtils/fit.h"

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
  float getDCAYCut(float pt) const;
  std::shared_ptr<DataRequest> mDataRequest;
  std::shared_ptr<o2::base::GRPGeomRequest> mGGCCDBRequest;
  bool mUseMC{false}; ///< MC flag
  std::unique_ptr<o2::utils::TreeStreamRedirector> mDBGOut;
  std::unique_ptr<o2::utils::TreeStreamRedirector> mDBGOutVtx;
  float mITSROFrameLengthMUS = 0.;
  int mMaxNeighbours = 3;
  float mMaxVTTimeDiff = 80.; // \mus
  float mTPCDCAYCut = 2.;
  float mTPCDCAZCut = 2.;
  float mMinX = 6.;
  float mMaxEta = 0.8;
  float mMinPt = 0.1;
  int mMinTPCClusters = 60;
  std::string mDCAYFormula = "0.0105 + 0.0350 / pow(x, 1.1)";

  GTrackID::mask_t mTracksSrc{};
  o2::dataformats::MeanVertexObject mMeanVtx{};
  o2::steer::MCKinematicsReader mcReader; // reader of MC information
};

void TrackingStudySpec::init(InitContext& ic)
{
  o2::base::GRPGeomHelper::instance().setRequest(mGGCCDBRequest);
  mDBGOut = std::make_unique<o2::utils::TreeStreamRedirector>("trackStudy.root", "recreate");
  mDBGOutVtx = std::make_unique<o2::utils::TreeStreamRedirector>("trackStudyVtx.root", "recreate");

  mMaxVTTimeDiff = ic.options().get<float>("max-vtx-timediff");
  mMaxNeighbours = ic.options().get<int>("max-vtx-neighbours");
  mTPCDCAYCut = ic.options().get<float>("max-tpc-dcay");
  mTPCDCAZCut = ic.options().get<float>("max-tpc-dcaz");
  mMinX = ic.options().get<float>("min-x-prop");
  mMaxEta = ic.options().get<float>("max-eta");
  mMinPt = ic.options().get<float>("min-pt");
  mMinTPCClusters = ic.options().get<int>("min-tpc-clusters");
  mDCAYFormula = ic.options().get<std::string>("dcay-vs-pt");
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
    pc.inputs().get<o2::dataformats::MeanVertexObject*>("meanvtx");
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
  int nv = vtxRefs.size();
  o2::dataformats::PrimaryVertexExt pveDummy;
  o2::dataformats::PrimaryVertex vtxDummy(mMeanVtx.getPos(), {}, {}, 0);
  std::vector<o2::dataformats::PrimaryVertexExt> pveVec(nv - 1);
  const auto& alpParams = o2::itsmft::DPLAlpideParam<o2::detectors::DetID::ITS>::Instance();
  float tBiasITS = alpParams.roFrameBiasInBC * o2::constants::lhc::LHCBunchSpacingMUS;
  std::vector<float> dtvec, dzvec;
  const o2::ft0::InteractionTag& ft0Params = o2::ft0::InteractionTag::Instance();

  for (int iv = 0; iv < nv; iv++) {
    LOGP(debug, "processing PV {} of {}", iv, nv);
    const auto& vtref = vtxRefs[iv];
    if (iv != nv - 1) {
      auto& pve = pveVec[iv];
      static_cast<o2::dataformats::PrimaryVertex&>(pve) = pvvec[iv];
      dtvec.clear();
      dzvec.clear();
      dtvec.reserve(pve.getNContributors());
      dzvec.reserve(pve.getNContributors());
      float bestTimeDiff = 1000, bestTime = -999;
      int bestFTID = -1;
      if (mTracksSrc[GTrackID::FT0]) {
        for (int ift0 = vtref.getFirstEntryOfSource(GTrackID::FT0); ift0 < vtref.getFirstEntryOfSource(GTrackID::FT0) + vtref.getEntriesOfSource(GTrackID::FT0); ift0++) {
          const auto& ft0 = FITInfo[trackIndex[ift0]];
          if (ft0Params.isSelected(ft0)) {
            auto fitTime = ft0.getInteractionRecord().differenceInBCMUS(recoData.startIR);
            if (std::abs(fitTime - pve.getTimeStamp().getTimeStamp()) < bestTimeDiff) {
              bestTimeDiff = fitTime - pve.getTimeStamp().getTimeStamp();
              bestFTID = trackIndex[ift0];
            }
          }
        }
      } else {
        LOGP(warn, "FT0 is not requested, cannot set complete vertex info");
      }
      if (bestFTID >= 0) {
        pve.FT0A = FITInfo[bestFTID].getTrigger().getAmplA();
        pve.FT0C = FITInfo[bestFTID].getTrigger().getAmplC();
        pve.FT0Time = FITInfo[bestFTID].getInteractionRecord().differenceInBCMUS(recoData.startIR);
      }
      pve.VtxID = iv;
    }
    float meanT = 0, meanZ = 0, rmsT = 0, rmsZ = 0;
    float meanTW = 0, meanZW = 0, rmsTW = 0, rmsZW = 0, WT = 0, WZ = 0;
    float meanT0 = 0, rmsT0 = 0;
    float meanTW0 = 0, rmsTW0 = 0, WT0 = 0;
    int nContAdd = 0, nContAdd0 = 0, ntITS = 0;
    int nAdjusted = 0;
    for (int is = 0; is < GTrackID::NSources; is++) {
      DetID::mask_t dm = GTrackID::getSourceDetectorsMask(is);
      bool skipTracks = !mTracksSrc[is] || !recoData.isTrackSourceLoaded(is) || !(dm[DetID::ITS] || dm[DetID::TPC]);
      int idMin = vtref.getFirstEntryOfSource(is), idMax = idMin + vtref.getEntriesOfSource(is);
      for (int i = idMin; i < idMax; i++) {
        auto vid = trackIndex[i];
        bool pvCont = vid.isPVContributor();
        if (pvCont) {
          pveVec[iv].nSrc[is]++;
        }
        if (skipTracks) {
          continue;
        }
        bool ambig = vid.isAmbiguous();
        auto trc = recoData.getTrackParam(vid);
        float xmin = trc.getX();
        o2::dataformats::DCA dca;
        if (!prop->propagateToDCA(iv == nv - 1 ? vtxDummy : pvvec[iv], trc, prop->getNominalBz(), 2., o2::base::PropagatorF::MatCorrType::USEMatCorrLUT, &dca)) {
          continue;
        }
        bool hasITS = GTrackID::getSourceDetectorsMask(is)[GTrackID::ITS];
        bool acceptGlo = true;
        while (1) {
          // do we cound this track for global multiplicity?
          if (!(acceptGlo = abs(trc.getEta()) < mMaxEta && trc.getPt() > mMinPt)) {
            break;
          }
          if (!(acceptGlo = std::abs(dca.getY()) < (hasITS ? getDCAYCut(trc.getPt()) : mTPCDCAYCut) && std::abs(dca.getZ()) < mTPCDCAYCut && xmin < mMinX)) {
            break;
          }
          GTrackID tpcTrID;
          if (GTrackID::getSourceDetectorsMask(is)[GTrackID::TPC] && recoData.isTrackSourceLoaded(GTrackID::TPC) && (tpcTrID = recoData.getTPCContributorGID(vid))) {
            auto& tpcTr = recoData.getTPCTrack(tpcTrID);
            if (!(acceptGlo = tpcTr.getNClusters() >= mMinTPCClusters)) {
              break;
            }
          }
          if (iv != nv - 1) {
            pveVec[iv].nSrcA[is]++;
            if (ambig) {
              pveVec[iv].nSrcAU[is]++;
            }
          }
          break;
        }

        if (!hasITS) {
          continue;
        }
        float ttime = 0, ttimeE = 0;
        recoData.getTrackTime(vid, ttime, ttimeE);
        bool acceptForPV0 = pvCont;
        if (vid.getSource() == GTrackID::ITS) {
          ttimeE *= mITSROFrameLengthMUS;
          ttime += ttimeE + tBiasITS;
          ttimeE *= 1. / sqrt(3);
          if (++ntITS > 0) { // do not allow ITS in the MAD
            acceptForPV0 = false;
          }
        } else if (vid.getSource() == GTrackID::ITSTPC) {
          float bcf = ttime / o2::constants::lhc::LHCBunchSpacingMUS + o2::constants::lhc::LHCMaxBunches;
          int bcWrtROF = int(bcf - alpParams.roFrameBiasInBC) % alpParams.roFrameLengthInBC;
          if (bcWrtROF == 0) {
            float dbc = bcf - (int(bcf / alpParams.roFrameBiasInBC)) * alpParams.roFrameBiasInBC;
            if (std::abs(dbc) < 1e-6 && (++nAdjusted) > 1) {
              acceptForPV0 = false; // do not allow more than 1 adjusted track MAD
            }
          }
        } else if (vid.getSource() == GTrackID::TPC) {
          ttimeE *= o2::constants::lhc::LHCBunchSpacingMUS * 8;
        }
        if (pvCont) {
          float dt = ttime - pveVec[iv].getTimeStamp().getTimeStamp();
          float tW = 1. / (ttimeE * ttimeE), zW = 1. / trc.getSigmaZ2();
          dzvec.push_back(dca.getZ());
          WT += tW;
          WZ += zW;
          meanT += dt;
          meanTW += dt * tW;
          meanZ += dca.getZ();
          meanZW += dca.getZ() * zW;

          rmsT += dt * dt;
          rmsTW += dt * dt * tW;
          rmsZ += dca.getZ() * dca.getZ();
          rmsZW += dca.getZ() * dca.getZ() * zW;
          nContAdd++;
          if (acceptForPV0) {
            dtvec.push_back(dt);
            WT0 += tW;
            meanT0 += dt;
            meanTW0 += dt * tW;
            rmsT0 += dt * dt;
            rmsTW0 += dt * dt * tW;
            nContAdd0++;
          }
          LOGP(debug, "dt={} dz={}, tW={}, zW={} t={} tE={} {}", dt, dca.getZ(), tW, zW, ttime, ttimeE, vid.asString());
        }
        if (acceptGlo) {
          (*mDBGOut) << "dca"
                     << "tfID=" << TFCount << "ttime=" << ttime << "ttimeE=" << ttimeE
                     << "gid=" << vid << "pv=" << (iv == nv - 1 ? vtxDummy : pvvec[iv]) << "trc=" << trc << "pvCont=" << pvCont << "ambig=" << ambig << "dca=" << dca << "xmin=" << xmin << "\n";
        }
      }
    }

    if (iv != nv - 1) {
      auto& pve = pveVec[iv];
      if (nContAdd) {
        rmsT /= nContAdd;
        rmsZ /= nContAdd;
        meanT /= nContAdd;
        meanZ /= nContAdd;
        pve.rmsT = (rmsT - meanT * meanT);
        pve.rmsT = pve.rmsT > 0 ? std::sqrt(pve.rmsT) : 0;
        pve.rmsZ = rmsZ - meanZ * meanZ;
        pve.rmsZ = pve.rmsZ > 0 ? std::sqrt(pve.rmsZ) : 0;
      }
      if (nContAdd0) {
        rmsT0 /= nContAdd0;
        meanT0 /= nContAdd0;
        pve.rmsT0 = (rmsT0 - meanT0 * meanT0);
        pve.rmsT0 = pve.rmsT0 > 0 ? std::sqrt(pve.rmsT0) : 0;
      }
      if (WT0 > 0) {
        rmsTW0 /= WT0;
        meanTW0 /= WT0;
        pve.rmsTW0 = (rmsTW0 - meanTW0 * meanTW0);
        pve.rmsTW0 = pve.rmsTW0 > 0 ? std::sqrt(pve.rmsTW0) : 0;
      }
      //
      if (WT > 0 && WZ > 0) {
        rmsTW /= WT;
        meanTW /= WT;
        pve.rmsTW = (rmsTW - meanTW * meanTW);
        pve.rmsTW = pve.rmsTW > 0 ? std::sqrt(pve.rmsTW) : 0;
        rmsZW /= WZ;
        meanZW /= WZ;
        pve.rmsZW = rmsZW - meanZW * meanZW;
        pve.rmsZW = pve.rmsZ > 0 ? std::sqrt(pve.rmsZ) : 0;
      }
      pve.tMAD = o2::math_utils::MAD2Sigma(dtvec.size(), dtvec.data());
      pve.zMAD = o2::math_utils::MAD2Sigma(dzvec.size(), dzvec.data());
    }
  }
  int nvtot = mMaxNeighbours < 0 ? -1 : (int)pveVec.size();

  auto insSlot = [maxSlots = mMaxNeighbours](std::vector<float>& vc, float v, int slot, std::vector<int>& vid, int id) {
    for (int i = maxSlots - 1; i > slot; i--) {
      std::swap(vc[i], vc[i - 1]);
      std::swap(vid[i], vid[i - 1]);
    }
    vc[slot] = v;
    vid[slot] = id;
  };

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
          insSlot(dT, dtime, i, idT, cntM);
          break;
        }
      }
      auto dz = std::abs(pve.getZ() - vt.getZ());
      for (int i = 0; i < mMaxNeighbours; i++) {
        if (dZ[i] > dz) {
          insSlot(dZ, dz, i, idZ, cntM);
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
          insSlot(dT, dtime, i, idT, cntP);
          break;
        }
      }
      auto dz = std::abs(pve.getZ() - vt.getZ());
      for (int i = 0; i < mMaxNeighbours; i++) {
        if (dZ[i] > dz) {
          insSlot(dZ, dz, i, idZ, cntP);
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
  if (matcher == ConcreteDataMatcher("GLO", "MEANVERTEX", 0)) {
    LOG(info) << "Imposing new MeanVertex: " << ((const o2::dataformats::MeanVertexObject*)obj)->asString();
    mMeanVtx = *(const o2::dataformats::MeanVertexObject*)obj;
    return;
  }
}

float TrackingStudySpec::getDCAYCut(float pt) const
{
  static TF1 fun("dcayvspt", mDCAYFormula.c_str(), 0, 20);
  return fun.Eval(pt);
}

DataProcessorSpec getTrackingStudySpec(GTrackID::mask_t srcTracks, GTrackID::mask_t srcClusters, bool useMC)
{
  std::vector<OutputSpec> outputs;
  auto dataRequest = std::make_shared<DataRequest>();

  dataRequest->requestTracks(srcTracks, useMC);
  dataRequest->requestClusters(srcClusters, useMC);
  dataRequest->requestPrimaryVertertices(useMC);
  dataRequest->inputs.emplace_back("meanvtx", "GLO", "MEANVERTEX", 0, Lifetime::Condition, ccdbParamSpec("GLO/Calib/MeanVertex", {}, 1));
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
      {"max-vtx-neighbours", VariantType::Int, 3, {"Max PV neighbours fill, no PV study if < 0"}},
      {"max-vtx-timediff", VariantType::Float, 90.f, {"Max PV time difference to consider"}},
      {"dcay-vs-pt", VariantType::String, "0.0105 + 0.0350 / pow(x, 1.1)", {"Formula for global tracks DCAy vs pT cut"}},
      {"min-tpc-clusters", VariantType::Int, 60, {"Cut on TPC clusters"}},
      {"max-tpc-dcay", VariantType::Float, 2.f, {"Cut on TPC dcaY"}},
      {"max-tpc-dcaz", VariantType::Float, 2.f, {"Cut on TPC dcaZ"}},
      {"max-eta", VariantType::Float, 0.8f, {"Cut on track eta"}},
      {"min-pt", VariantType::Float, 0.1f, {"Cut on track pT"}},
      {"min-x-prop", VariantType::Float, 6.f, {"track should be propagated to this X at least"}},
    }};
}

} // namespace o2::trackstudy
