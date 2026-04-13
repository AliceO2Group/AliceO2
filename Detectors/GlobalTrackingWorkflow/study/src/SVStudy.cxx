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
#include "DetectorsVertexing/SVertexerParams.h"
#include "ReconstructionDataFormats/TrackTPCITS.h"
#include "ReconstructionDataFormats/V0.h"
#include "ReconstructionDataFormats/GlobalTrackID.h"
#include "DataFormatsCalibration/MeanVertexObject.h"
#include "DetectorsBase/Propagator.h"
#include "DetectorsBase/GeometryManager.h"
#include "SimulationDataFormat/MCEventLabel.h"
#include "SimulationDataFormat/MCUtils.h"
#include "SimulationDataFormat/MCTrack.h"
#include "CommonDataFormat/BunchFilling.h"
#include "CommonUtils/NameConf.h"
#include "DataFormatsFT0/RecPoints.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/CCDBParamSpec.h"
#include "FT0Reconstruction/InteractionTag.h"
#include "DataFormatsITSMFT/DPLAlpideParam.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "DetectorsBase/GRPGeomHelper.h"
#include "GlobalTrackingStudy/TrackingStudy.h"
#include "TPCBase/ParameterElectronics.h"
#include "ReconstructionDataFormats/PrimaryVertex.h"
#include "DataFormatsFT0/RecPoints.h"
#include "DataFormatsITSMFT/TrkClusRef.h"
#include "CommonUtils/TreeStreamRedirector.h"
#include "ReconstructionDataFormats/VtxTrackRef.h"
#include "ReconstructionDataFormats/DCA.h"
#include "Steer/MCKinematicsReader.h"
#include "DCAFitter/DCAFitterN.h"
#include "MathUtils/fit.h"
#include "GlobalTrackingStudy/V0Ext.h"
#include "GPUO2InterfaceConfiguration.h"
// #include "GPUSettingsO2.h"
#include "GPUParam.h"
#include "GPUParam.inc"
#include "GPUTPCGeometry.h"
#include "GPUO2InterfaceRefit.h"
#include "GPUO2InterfaceUtils.h"

namespace o2::svstudy
{

using namespace o2::framework;
using DetID = o2::detectors::DetID;
using DataRequest = o2::globaltracking::DataRequest;

using PVertex = o2::dataformats::PrimaryVertex;
using V2TRef = o2::dataformats::VtxTrackRef;
using VTIndex = o2::dataformats::VtxTrackIndex;
using GTrackID = o2::dataformats::GlobalTrackID;
using TBracket = o2::math_utils::Bracketf_t;
using V0ID = o2::dataformats::V0Index;

using timeEst = o2::dataformats::TimeStampWithError<float, float>;

class SVStudySpec final : public Task
{
 public:
  SVStudySpec(std::shared_ptr<DataRequest> dr, std::shared_ptr<o2::base::GRPGeomRequest> gr, GTrackID::mask_t src, bool useTPCCl, bool useMC)
    : mDataRequest(dr), mGGCCDBRequest(gr), mTracksSrc(src), mUseTPCCl(useTPCCl), mUseMC(useMC) {}
  ~SVStudySpec() final = default;
  void init(InitContext& ic) final;
  void run(ProcessingContext& pc) final;
  void endOfStream(EndOfStreamContext& ec) final;
  void finaliseCCDB(ConcreteDataMatcher& matcher, void* obj) final;
  void process(o2::globaltracking::RecoContainer& recoData);
  o2::dataformats::V0Ext processV0(int iv, o2::globaltracking::RecoContainer& recoData);

 private:
  void updateTimeDependentParams(ProcessingContext& pc);
  bool refitV0(const V0ID& id, o2::dataformats::V0& v0, o2::globaltracking::RecoContainer& recoData);
  std::shared_ptr<DataRequest> mDataRequest;
  std::shared_ptr<o2::base::GRPGeomRequest> mGGCCDBRequest;
  bool mUseMC{false}; ///< MC flag
  std::unique_ptr<o2::utils::TreeStreamRedirector> mDBGOut;
  float mSelK0 = -1;
  bool mRefit = false;
  bool mUseTPCCl = false;
  float mMaxEta = 0.8;
  float mBz = 0;
  int mNHBPerTF = 0;
  int mNTPCOccBinLength = 0; ///< TPC occ. histo bin length in TBs
  float mNTPCOccBinLengthInv;
  float mTPCTBinMUSInv = 0.f;
  GTrackID::mask_t mTracksSrc{};
  o2::vertexing::DCAFitterN<2> mFitterV0;
  std::vector<float> mTBinClOccAft, mTBinClOccBef;
  std::unique_ptr<o2::steer::MCKinematicsReader> mcReader; // reader of MC information
  std::shared_ptr<o2::gpu::GPUParam> mParam = nullptr;
};

void SVStudySpec::init(InitContext& ic)
{
  o2::base::GRPGeomHelper::instance().setRequest(mGGCCDBRequest);
  mDBGOut = std::make_unique<o2::utils::TreeStreamRedirector>("svStudy.root", "recreate");
  mRefit = ic.options().get<bool>("refit");
  mSelK0 = ic.options().get<float>("sel-k0");
  mMaxEta = ic.options().get<float>("max-eta");
  if (mUseMC) {
    mcReader = std::make_unique<o2::steer::MCKinematicsReader>("collisioncontext.root");
  }
}

void SVStudySpec::run(ProcessingContext& pc)
{
  o2::globaltracking::RecoContainer recoData;
  recoData.collectData(pc, *mDataRequest.get()); // select tracks of needed type, with minimal cuts, the real selected will be done in the vertexer
  updateTimeDependentParams(pc);                 // Make sure this is called after recoData.collectData, which may load some conditions

  size_t occupancyMapSizeBytes = o2::gpu::GPUO2InterfaceRefit::fillOccupancyMapGetSize(mNHBPerTF, mParam.get());
  gsl::span<const unsigned int> TPCRefitterOccMap = recoData.occupancyMapTPC;
  o2::gpu::GPUO2InterfaceUtils::paramUseExternalOccupancyMap(mParam.get(), mNHBPerTF, TPCRefitterOccMap.data(), occupancyMapSizeBytes);

  mTBinClOccBef.resize(1);
  mTBinClOccAft.resize(1);
  if (recoData.inputsTPCclusters && mUseTPCCl) {
    mNTPCOccBinLength = mParam->rec.tpc.occupancyMapTimeBins;
    mTBinClOccBef.clear();
    mTBinClOccAft.clear();
    // prepare TPC occupancy data
    if (mNTPCOccBinLength > 1 && recoData.occupancyMapTPC.size()) {
      mNTPCOccBinLengthInv = 1. / mNTPCOccBinLength;
      int nTPCBins = mNHBPerTF * o2::constants::lhc::LHCMaxBunches / 8, ninteg = 0;
      int nTPCOccBins = nTPCBins * mNTPCOccBinLengthInv, sumBins = std::max(1, int(o2::constants::lhc::LHCMaxBunches / 8 * mNTPCOccBinLengthInv));
      mTBinClOccAft.resize(nTPCOccBins);
      mTBinClOccBef.resize(nTPCOccBins);
      float sm = 0., tb = 0.5 * mNTPCOccBinLength;
      std::vector<float> mltHistTB(nTPCOccBins);
      for (int i = 0; i < nTPCOccBins; i++) {
        mltHistTB[i] = mParam->GetUnscaledMult(tb);
        tb += mNTPCOccBinLength;
      }
      for (int i = nTPCOccBins; i--;) {
        sm += mltHistTB[i];
        if (i + sumBins < nTPCOccBins) {
          sm -= mltHistTB[i + sumBins];
        }
        mTBinClOccAft[i] = sm;
      }
      sm = 0;
      for (int i = 0; i < nTPCOccBins; i++) {
        sm += mltHistTB[i];
        if (i - sumBins > 0) {
          sm -= mltHistTB[i - sumBins];
        }
        mTBinClOccBef[i] = sm;
      }
    }
  }

  process(recoData);
}

void SVStudySpec::updateTimeDependentParams(ProcessingContext& pc)
{
  o2::base::GRPGeomHelper::instance().checkUpdates(pc);
  static bool initOnceDone = false;
  if (!initOnceDone) { // this params need to be queried only once
    initOnceDone = true;
    const auto& svparam = o2::vertexing::SVertexerParams::Instance();
    // Note: reading of the ITS AlpideParam needed for ITS timing is done by the RecoContainer
    mFitterV0.setBz(mBz);
    mFitterV0.setUseAbsDCA(svparam.useAbsDCA);
    mFitterV0.setPropagateToPCA(false);
    mFitterV0.setMaxR(svparam.maxRIni);
    mFitterV0.setMinParamChange(svparam.minParamChange);
    mFitterV0.setMinRelChi2Change(svparam.minRelChi2Change);
    mFitterV0.setMaxDZIni(svparam.maxDZIni);
    mFitterV0.setMaxDXYIni(svparam.maxDXYIni);
    mFitterV0.setMaxChi2(svparam.maxChi2);
    mFitterV0.setMatCorrType(o2::base::Propagator::MatCorrType(svparam.matCorr));
    mFitterV0.setUsePropagator(svparam.usePropagator);
    mFitterV0.setRefitWithMatCorr(svparam.refitWithMatCorr);
    mFitterV0.setMaxStep(svparam.maxStep);
    mFitterV0.setMaxSnp(svparam.maxSnp);
    mFitterV0.setMinXSeed(svparam.minXSeed);

    mNHBPerTF = o2::base::GRPGeomHelper::instance().getGRPECS()->getNHBFPerTF();
    if (!mParam) {
      // for occupancy estimator
      mParam = o2::gpu::GPUO2InterfaceUtils::getFullParamShared(0.f, mNHBPerTF);
    }
    auto& elParam = o2::tpc::ParameterElectronics::Instance();
    mTPCTBinMUSInv = 1. / elParam.ZbinWidth; // 1./TPC bin in microseconds
  }
  mBz = o2::base::Propagator::Instance()->getNominalBz();
  mFitterV0.setBz(mBz);
}

o2::dataformats::V0Ext SVStudySpec::processV0(int iv, o2::globaltracking::RecoContainer& recoData)
{
  o2::dataformats::V0Ext v0ext;
  auto invalidate = [&v0ext]() { v0ext.v0ID.setVertexID(-1); };

  auto v0s = recoData.getV0s();
  auto v0IDs = recoData.getV0sIdx();
  static int tfID = 0;

  const auto& v0id = v0IDs[iv];
  if (mRefit && !refitV0(v0id, v0ext.v0, recoData)) {
    invalidate();
    return v0ext;
  }
  const auto& v0sel = mRefit ? v0ext.v0 : v0s[iv];
  if (mMaxEta < std::abs(v0sel.getEta())) {
    invalidate();
    return v0ext;
  }
  if (mSelK0 > 0 && std::abs(std::sqrt(v0sel.calcMass2AsK0()) - 0.497) > mSelK0) {
    invalidate();
    return v0ext;
  }
  if (!mRefit) {
    v0ext.v0 = v0sel;
  }
  v0ext.v0ID = v0id;
  const auto clRefs = recoData.getTPCTracksClusterRefs();
  o2::MCCompLabel lb[2];
  const o2::MCTrack* mcTrks[2];
  for (int ip = 0; ip < 2; ip++) {
    auto& prInfo = v0ext.prInfo[ip];
    auto gid = v0ext.v0ID.getProngID(ip);
    auto gidset = recoData.getSingleDetectorRefs(gid);
    lb[ip] = recoData.getTrackMCLabel(gid);
    if (lb[ip].isValid()) {
      prInfo.corrGlo = !lb[ip].isFake();
    }
    // get TPC tracks, if any
    if (gidset[GTrackID::TPC].isSourceSet()) {
      const auto& tpcTr = recoData.getTPCTrack(gidset[GTrackID::TPC]);
      prInfo.trackTPC = tpcTr;
      prInfo.nClTPC = tpcTr.getNClusters();
      lb[ip] = recoData.getTrackMCLabel(gidset[GTrackID::TPC]);
      if (lb[ip].isValid()) {
        prInfo.corrTPC = !lb[ip].isFake();
      }
      if (mParam && mUseTPCCl) {
        uint8_t clSect = 0, clRow = 0;
        uint32_t clIdx = 0;
        tpcTr.getClusterReference(clRefs, tpcTr.getNClusterReferences() - 1, clSect, clRow, clIdx);
        const auto& clus = recoData.getTPCClusters().clusters[clSect][clRow][clIdx];
        prInfo.lowestRow = clRow;
        int npads = o2::gpu::GPUTPCGeometry::NPads(clRow);
        prInfo.padFromEdge = uint8_t(clus.getPad());
        if (prInfo.padFromEdge > npads / 2) {
          prInfo.padFromEdge = npads - 1 - prInfo.padFromEdge;
        }
      }
    }
    // get ITS tracks, if any
    if (gid.includesDet(DetID::ITS)) {
      auto gidITS = recoData.getITSContributorGID(gid);
      if (gidset[GTrackID::ITS].isSourceSet()) {
        const auto& itsTr = recoData.getITSTrack(gidset[GTrackID::ITS]);
        prInfo.nClITS = itsTr.getNClusters();
        lb[ip] = recoData.getTrackMCLabel(gidset[GTrackID::ITS]);
        if (lb[ip].isValid()) {
          prInfo.corrITS = !lb[ip].isFake();
        }
        for (int il = 0; il < 7; il++) {
          if (itsTr.hasHitOnLayer(il)) {
            prInfo.pattITS |= 0x1 << il;
          }
        }
      } else {
        const auto& itsTrf = recoData.getITSABRefs()[gidset[GTrackID::ITSAB]];
        prInfo.nClITS = itsTrf.getNClusters();
        lb[ip] = recoData.getTrackMCLabel(gidset[GTrackID::ITSAB]);
        if (lb[ip].isValid()) {
          prInfo.corrITS = !lb[ip].isFake();
        }
        for (int il = 0; il < 7; il++) {
          if (itsTrf.hasHitOnLayer(il)) {
            prInfo.pattITS |= 0x1 << il;
          }
        }
        prInfo.pattITS |= 0x1 << 31; // flag AB
      }
      if (gidset[GTrackID::ITSTPC].isSourceSet()) {
        auto mtc = recoData.getTPCITSTrack(gidset[GTrackID::ITSTPC]);
        lb[ip] = recoData.getTrackMCLabel(gidset[GTrackID::ITSTPC]);
        prInfo.chi2ITSTPC = mtc.getChi2Match();
        if (lb[ip].isValid()) {
          prInfo.corrITSTPC = !lb[ip].isFake();
        }
      }
    }
    if (mUseMC && lb[ip].isValid()) { // temp store of mctrks
      mcTrks[ip] = mcReader->getTrack(lb[ip]);
    }
  }
  if (mUseMC && (mcTrks[0] != nullptr) && (mcTrks[1] != nullptr)) {
    // check majority vote on mother particle otherwise leave pdg -1
    if (lb[0].getSourceID() == lb[1].getSourceID() && lb[0].getEventID() == lb[1].getEventID() &&
        mcTrks[0]->getMotherTrackId() == mcTrks[1]->getMotherTrackId() && mcTrks[0]->getMotherTrackId() >= 0) {
      const auto mother = mcReader->getTrack(lb[0].getSourceID(), lb[0].getEventID(), mcTrks[0]->getMotherTrackId());
      v0ext.mcPID = mother->GetPdgCode();
    }
  }
  return v0ext;
}

void SVStudySpec::process(o2::globaltracking::RecoContainer& recoData)
{
  auto v0IDs = recoData.getV0sIdx();
  auto nv0 = v0IDs.size();
  if (nv0 > recoData.getV0s().size()) {
    mRefit = true;
  }
  std::map<int, std::vector<int>> pv2sv;
  static int tfID = 0;
  for (int iv = 0; iv < nv0; iv++) {
    const auto v0id = v0IDs[iv];
    pv2sv[v0id.getVertexID()].push_back(iv);
  }
  std::vector<o2::dataformats::V0Ext> v0extVec;
  for (auto it : pv2sv) {
    int pvID = it.first;
    auto& vv = it.second;
    if (pvID < 0 || vv.size() == 0) {
      continue;
    }
    v0extVec.clear();
    for (int iv0 : vv) {
      auto v0ext = processV0(iv0, recoData);
      if (v0ext.v0ID.getVertexID() < 0) {
        continue;
      }
      v0extVec.push_back(v0ext);
    }
    if (v0extVec.size()) {
      const auto& pv = recoData.getPrimaryVertex(pvID);
      float tpcOccBef = 0., tpcOccAft = 0.;
      int tb = pv.getTimeStamp().getTimeStamp() * mTPCTBinMUSInv * mNTPCOccBinLengthInv;
      tpcOccBef = tb < 0 ? mTBinClOccBef[0] : (tb >= mTBinClOccBef.size() ? mTBinClOccBef.back() : mTBinClOccBef[tb]);
      tpcOccAft = tb < 0 ? mTBinClOccAft[0] : (tb >= mTBinClOccAft.size() ? mTBinClOccAft.back() : mTBinClOccAft[tb]);

      (*mDBGOut) << "v0"
                 << "orbit=" << recoData.startIR.orbit << "tfID=" << tfID << "tpcOccBef=" << tpcOccBef << "tpcOccAft=" << tpcOccAft
                 << "v0Ext=" << v0extVec
                 << "pv=" << pv
                 << "\n";
    }
  }
  tfID++;
}

bool SVStudySpec::refitV0(const V0ID& id, o2::dataformats::V0& v0, o2::globaltracking::RecoContainer& recoData)
{
  auto seedP = recoData.getTrackParam(id.getProngID(0));
  auto seedN = recoData.getTrackParam(id.getProngID(1));
  bool isTPConly = (id.getProngID(0).getSource() == GTrackID::TPC) || (id.getProngID(1).getSource() == GTrackID::TPC);
  const auto& svparam = o2::vertexing::SVertexerParams::Instance();
  if (svparam.mTPCTrackPhotonTune && isTPConly) {
    mFitterV0.setMaxDZIni(svparam.mTPCTrackMaxDZIni);
    mFitterV0.setMaxDXYIni(svparam.mTPCTrackMaxDXYIni);
    mFitterV0.setMaxChi2(svparam.mTPCTrackMaxChi2);
    mFitterV0.setCollinear(true);
  }
  int nCand = mFitterV0.process(seedP, seedN);
  if (svparam.mTPCTrackPhotonTune && isTPConly) { // restore
    // Reset immediately to the defaults
    mFitterV0.setMaxDZIni(svparam.maxDZIni);
    mFitterV0.setMaxDXYIni(svparam.maxDXYIni);
    mFitterV0.setMaxChi2(svparam.maxChi2);
    mFitterV0.setCollinear(false);
  }
  if (nCand == 0) { // discard this pair
    return false;
  }
  const int cand = 0;
  if (!mFitterV0.isPropagateTracksToVertexDone(cand) && !mFitterV0.propagateTracksToVertex(cand)) {
    return false;
  }
  const auto& trPProp = mFitterV0.getTrack(0, cand);
  const auto& trNProp = mFitterV0.getTrack(1, cand);
  std::array<float, 3> pP{}, pN{};
  trPProp.getPxPyPzGlo(pP);
  trNProp.getPxPyPzGlo(pN);
  std::array<float, 3> pV0 = {pP[0] + pN[0], pP[1] + pN[1], pP[2] + pN[2]};
  auto p2V0 = pV0[0] * pV0[0] + pV0[1] * pV0[1] + pV0[2] * pV0[2];
  const auto& pv = recoData.getPrimaryVertex(id.getVertexID());
  const auto v0XYZ = mFitterV0.getPCACandidatePos(cand);
  float dx = v0XYZ[0] - pv.getX(), dy = v0XYZ[1] - pv.getY(), dz = v0XYZ[2] - pv.getZ(), prodXYZv0 = dx * pV0[0] + dy * pV0[1] + dz * pV0[2];
  float cosPA = prodXYZv0 / std::sqrt((dx * dx + dy * dy + dz * dz) * p2V0);
  new (&v0) o2::dataformats::V0(v0XYZ, pV0, mFitterV0.calcPCACovMatrixFlat(cand), trPProp, trNProp);
  v0.setDCA(mFitterV0.getChi2AtPCACandidate(cand));
  v0.setCosPA(cosPA);
  return true;
}

void SVStudySpec::endOfStream(EndOfStreamContext& ec)
{
  mDBGOut.reset();
}

void SVStudySpec::finaliseCCDB(ConcreteDataMatcher& matcher, void* obj)
{
  if (o2::base::GRPGeomHelper::instance().finaliseCCDB(matcher, obj)) {
    return;
  }
}

DataProcessorSpec getSVStudySpec(GTrackID::mask_t srcTracks, GTrackID::mask_t srcCls, bool useMC)
{
  std::vector<OutputSpec> outputs;
  auto dataRequest = std::make_shared<DataRequest>();

  dataRequest->requestTracks(srcTracks, useMC);
  dataRequest->requestClusters(srcCls, false);
  dataRequest->requestPrimaryVertices(useMC);
  dataRequest->requestSecondaryVertices(useMC);
  dataRequest->inputs.emplace_back("meanvtx", "GLO", "MEANVERTEX", 0, Lifetime::Condition, ccdbParamSpec("GLO/Calib/MeanVertex", {}, 1));
  auto ggRequest = std::make_shared<o2::base::GRPGeomRequest>(true,                           // orbitResetTime
                                                              true,                           // GRPECS=true
                                                              false,                          // GRPLHCIF
                                                              true,                           // GRPMagField
                                                              true,                           // askMatLUT
                                                              o2::base::GRPGeomRequest::None, // geometry
                                                              dataRequest->inputs,
                                                              true);
  bool useTPCcl = srcCls[GTrackID::TPC];
  return DataProcessorSpec{
    "sv-study",
    dataRequest->inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<SVStudySpec>(dataRequest, ggRequest, srcTracks, useTPCcl, useMC)},
    Options{
      {"refit", VariantType::Bool, false, {"refit SVertices"}},
      {"sel-k0", VariantType::Float, -1.f, {"If positive, select K0s with this mass margin"}},
      {"max-eta", VariantType::Float, 1.2f, {"Cut on track eta"}},
    }};
}

} // namespace o2::svstudy
