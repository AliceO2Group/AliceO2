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

#include <random>
#include <vector>
#include <TStopwatch.h>
#include "CommonConstants/LHCConstants.h"
#include "DataFormatsGlobalTracking/RecoContainer.h"
#include "DataFormatsGlobalTracking/RecoContainerCreateTracksVariadic.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "TPCCalibration/VDriftHelper.h"
#include "TPCCalibration/CorrectionMapsLoader.h"
#include "ReconstructionDataFormats/GlobalTrackID.h"
#include "DetectorsBase/Propagator.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/Task.h"
#include "MathUtils/Tsallis.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "ReconstructionDataFormats/TrackCosmics.h"
#include "DataFormatsTPC/Constants.h"
#include "DetectorsBase/GRPGeomHelper.h"
#include "TPCWorkflow/TPCRefitter.h"
#include "GPUTPCGMMergedTrackHit.h"
#include "GPUO2InterfaceRefit.h"
#include "TPCBase/ParameterElectronics.h"
#include "CommonUtils/TreeStreamRedirector.h"
#include "Steer/MCKinematicsReader.h"
#include "DetectorsRaw/HBFUtils.h"

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

class TPCRefitterSpec final : public Task
{
 public:
  enum StudyType {
    TPC = 0x1,     ///< TPConly
    ITSTPC = 0x2,  ///< TPC + ITS matched tracks
    Cosmics = 0x4, ///< Cosmics
  };
  enum WriterType {
    Streamer = 0x1,  ///< Write per track streamer information
    TFVectors = 0x2, ///< Writer vectors per TF
  };
  TPCRefitterSpec(std::shared_ptr<DataRequest> dr, std::shared_ptr<o2::base::GRPGeomRequest> gr, const o2::tpc::CorrectionMapsLoaderGloOpts& sclOpts, GTrackID::mask_t src, bool useMC)
    : mDataRequest(dr), mGGCCDBRequest(gr), mTracksSrc(src), mUseMC(useMC)
  {
    mTPCCorrMapsLoader.setLumiScaleType(sclOpts.lumiType);
    mTPCCorrMapsLoader.setLumiScaleMode(sclOpts.lumiMode);
  }
  ~TPCRefitterSpec() final = default;
  void init(InitContext& ic) final;
  void run(ProcessingContext& pc) final;
  void endOfStream(EndOfStreamContext& ec) final;
  void finaliseCCDB(ConcreteDataMatcher& matcher, void* obj) final;
  void process(o2::globaltracking::RecoContainer& recoData);
  bool getDCAs(const o2::track::TrackPar& track, float& dcar, float& dcaz);

 private:
  void updateTimeDependentParams(ProcessingContext& pc);
  std::shared_ptr<DataRequest> mDataRequest;
  std::shared_ptr<o2::base::GRPGeomRequest> mGGCCDBRequest;
  o2::tpc::VDriftHelper mTPCVDriftHelper{};
  o2::tpc::CorrectionMapsLoader mTPCCorrMapsLoader{};
  bool mUseMC{false}; ///< MC flag
  bool mUseGPUModel{false};
  float mXRef = 83.;
  float mDCAMinPt = 1.;
  int mTFStart = 0;
  int mTFEnd = 999999999;
  int mTFCount = -1;
  int mDCAMinNCl = 80;
  int mStudyType = 0;         ///< Bitmask of 'StudyType'
  int mWriterType = 0;        ///< Bitmask of 'WriterType'
  float mSqrt{13600};         ///< centre of mass energy
  float mSamplingFactor{0.1}; ///< sampling factor in case sampling is used for unbinned data
  bool mUseR = false;
  bool mEnableDCA = false;
  int mWriteTrackClusters = 0;                                      ///< bitmask of which cluster information to dump to the tree: 0x1 = cluster native, 0x2 = corrected cluster positions, 0x4 = uncorrected cluster positions, 0x8 occupancy info
  bool mDoSampling{false};                                          ///< perform sampling of unbinned data
  bool mDoRefit{true};                                              ///< perform refit of TPC track
  std::vector<size_t> mClusterOccupancy;                            ///< binned occupancy of all clusters
  std::vector<size_t> mITSTPCTrackOccupanyTPCTime;                  ///< binned occupancy for ITS-TPC matched tracks using the TPC track time
  std::vector<size_t> mITSTPCTrackOccupanyCombinedTime;             ///< binned occupancy for ITS-TPC matched tracks using the combined track time
  std::unique_ptr<o2::utils::TreeStreamRedirector> mDBGOutTPC;      ///< per track streamer for TPC tracks
  std::unique_ptr<o2::utils::TreeStreamRedirector> mDBGOutITSTPC;   ///< per track streamer for ITS-TPC tracks
  std::unique_ptr<o2::utils::TreeStreamRedirector> mDBGOutTPCTF;    ///< per TF streamer for TPC tracks
  std::unique_ptr<o2::utils::TreeStreamRedirector> mDBGOutITSTPCTF; ///< per TF streamer for ITS-TPC tracks
  std::unique_ptr<o2::utils::TreeStreamRedirector> mDBGOutCosmics;  ///< per track streamer for TPC tracks
  std::unique_ptr<o2::utils::TreeStreamRedirector> mDBGOutCl;       ///< TPC cluster streamer
  float mITSROFrameLengthMUS = 0.;
  GTrackID::mask_t mTracksSrc{};
  o2::steer::MCKinematicsReader mcReader; ///< reader of MC information
  std::mt19937 mGenerator;                ///< random generator for sampling
  float mVdriftTB;                        ///< VDrift expressed in cm/TimeBin
  float mTPCTBBias;                       ///< Time bin bias
  uint32_t mTimeBinsPerTF{};              ///< number of time bins in TF
  uint32_t mOccupancyBinsPerTF{};         ///< number of time bins in TF
  uint32_t mTimeBinsPerDrift{500};        ///< number of time bins assumed in one drift
  //
  // Input data
  //
  gsl::span<const o2::tpc::TPCClRefElem> mTPCTrackClusIdx;            ///< input TPC track cluster indices span
  gsl::span<const o2::tpc::TrackTPC> mTPCTracksArray;                 ///< input TPC tracks span
  gsl::span<const o2::dataformats::TrackTPCITS> mITSTPCTracksArray;   ///< input TPC-ITS tracks span
  gsl::span<const o2::its::TrackITS> mITSTracksArray;                 ///< input ITS tracks span
  gsl::span<const o2::dataformats::TrackCosmics> mCosmics;            ///< input ITS tracks span
  gsl::span<const unsigned char> mTPCRefitterShMap;                   ///< externally set TPC clusters sharing map
  gsl::span<const unsigned int> mTPCRefitterOccMap;                   ///< externally set TPC clusters occupancy map
  const o2::tpc::ClusterNativeAccess* mTPCClusterIdxStruct = nullptr; ///< struct holding the TPC cluster indices
  gsl::span<const o2::MCCompLabel> mTPCTrkLabels;                     ///< input TPC Track MC labels
  std::unique_ptr<o2::gpu::GPUO2InterfaceRefit> mTPCRefitter;         ///< TPC refitter used for TPC tracks refit during the reconstruction
  std::vector<o2::InteractionTimeRecord> mIntRecs;

  void fillOccupancyVectors(o2::globaltracking::RecoContainer& recoData);
  bool processTPCTrack(o2::tpc::TrackTPC tr, o2::MCCompLabel lbl, o2::utils::TreeStreamRedirector* streamer, const o2::its::TrackITS* its = nullptr, const o2::dataformats::TrackTPCITS* itstpc = nullptr, bool outward = false, float time0custom = -1);
  void processCosmics(o2::globaltracking::RecoContainer& recoData);
};

void TPCRefitterSpec::init(InitContext& ic)
{
  o2::base::GRPGeomHelper::instance().setRequest(mGGCCDBRequest);
  mXRef = ic.options().get<float>("target-x");
  mUseR = ic.options().get<bool>("use-r-as-x");
  mEnableDCA = ic.options().get<bool>("enable-dcas");
  mUseGPUModel = ic.options().get<bool>("use-gpu-fitter");
  mTFStart = ic.options().get<int>("tf-start");
  mTFEnd = ic.options().get<int>("tf-end");
  mDCAMinPt = ic.options().get<float>("dcaMinPt");
  mDCAMinNCl = ic.options().get<float>("dcaMinNCl");
  mSqrt = ic.options().get<float>("sqrts");
  mSamplingFactor = ic.options().get<float>("sampling-factor");
  mDoSampling = ic.options().get<bool>("do-sampling");
  mDoRefit = ic.options().get<bool>("do-refit");
  mStudyType = ic.options().get<int>("study-type");
  mWriterType = ic.options().get<int>("writer-type");
  mWriteTrackClusters = ic.options().get<int>("write-track-clusters");
  const auto occBinsPerDrift = ic.options().get<uint32_t>("occupancy-bins-per-drift");
  mTimeBinsPerTF = (o2::raw::HBFUtils::Instance().nHBFPerTF * o2::constants::lhc::LHCMaxBunches) / 8 + 2 * mTimeBinsPerDrift; // add one drift before and after the TF
  mOccupancyBinsPerTF = static_cast<uint32_t>(std::ceil(float(mTimeBinsPerTF * occBinsPerDrift) / mTimeBinsPerDrift));
  mClusterOccupancy.resize(mOccupancyBinsPerTF);
  mITSTPCTrackOccupanyTPCTime.resize(mOccupancyBinsPerTF);
  mITSTPCTrackOccupanyCombinedTime.resize(mOccupancyBinsPerTF);
  LOGP(info, "Using {} bins for the occupancy per TF", mOccupancyBinsPerTF);

  if ((mWriterType & WriterType::Streamer) == WriterType::Streamer) {
    if ((mStudyType & StudyType::TPC) == StudyType::TPC) {
      mDBGOutTPC = std::make_unique<o2::utils::TreeStreamRedirector>("tpctracks-study-streamer.root", "recreate");
    }
    if ((mStudyType & StudyType::ITSTPC) == StudyType::ITSTPC) {
      mDBGOutITSTPC = std::make_unique<o2::utils::TreeStreamRedirector>("itstpctracks-study-streamer.root", "recreate");
    }
    if ((mStudyType & StudyType::Cosmics) == StudyType::Cosmics) {
      mDBGOutCosmics = std::make_unique<o2::utils::TreeStreamRedirector>("cosmics-study-streamer.root", "recreate");
    }
  }
  if (ic.options().get<bool>("dump-clusters")) {
    mDBGOutCl = std::make_unique<o2::utils::TreeStreamRedirector>("tpc-trackStudy-cl.root", "recreate");
  }

  if (mXRef < 0.) {
    mXRef = 0.;
  }
  mGenerator = std::mt19937(std::random_device{}());
  mTPCCorrMapsLoader.init(ic);
}

void TPCRefitterSpec::run(ProcessingContext& pc)
{
  ++mTFCount;
  if (mTFCount < mTFStart || mTFCount > mTFEnd) {
    LOGP(info, "Skipping TF {}", mTFCount);
    return;
  }

  o2::globaltracking::RecoContainer recoData;
  recoData.collectData(pc, *mDataRequest.get()); // select tracks of needed type, with minimal cuts, the real selected will be done in the vertexer
  updateTimeDependentParams(pc);                 // Make sure this is called after recoData.collectData, which may load some conditions
  fillOccupancyVectors(recoData);
  process(recoData);

  if (mTFCount >= mTFEnd) {
    LOGP(info, "Stopping processing after TF {}", mTFCount);
    pc.services().get<o2::framework::ControlService>().endOfStream();
    return;
  }
}

void TPCRefitterSpec::updateTimeDependentParams(ProcessingContext& pc)
{
  o2::base::GRPGeomHelper::instance().checkUpdates(pc);
  mTPCVDriftHelper.extractCCDBInputs(pc);
  mTPCCorrMapsLoader.extractCCDBInputs(pc);
  static bool initOnceDone = false;
  if (!initOnceDone) { // this params need to be queried only once
    initOnceDone = true;
    // none at the moment
  }
  // we may have other params which need to be queried regularly
  bool updateMaps = false;
  if (mTPCCorrMapsLoader.isUpdated()) {
    mTPCCorrMapsLoader.acknowledgeUpdate();
    updateMaps = true;
  }
  if (mTPCVDriftHelper.isUpdated()) {
    LOGP(info, "Updating TPC fast transform map with new VDrift factor of {} wrt reference {} and DriftTimeOffset correction {} wrt {} from source {}",
         mTPCVDriftHelper.getVDriftObject().corrFact, mTPCVDriftHelper.getVDriftObject().refVDrift,
         mTPCVDriftHelper.getVDriftObject().timeOffsetCorr, mTPCVDriftHelper.getVDriftObject().refTimeOffset,
         mTPCVDriftHelper.getSourceName());
    mTPCVDriftHelper.acknowledgeUpdate();
    updateMaps = true;
  }
  if (updateMaps) {
    mTPCCorrMapsLoader.updateVDrift(mTPCVDriftHelper.getVDriftObject().corrFact, mTPCVDriftHelper.getVDriftObject().refVDrift, mTPCVDriftHelper.getVDriftObject().getTimeOffset());
  }
}

void TPCRefitterSpec::fillOccupancyVectors(o2::globaltracking::RecoContainer& recoData)
{
  // reset counters
  std::fill(mClusterOccupancy.begin(), mClusterOccupancy.end(), 0u);
  std::fill(mITSTPCTrackOccupanyTPCTime.begin(), mITSTPCTrackOccupanyTPCTime.end(), 0u);
  std::fill(mITSTPCTrackOccupanyCombinedTime.begin(), mITSTPCTrackOccupanyCombinedTime.end(), 0u);

  // fill cluster occupancy
  const auto& clusterIndex = recoData.inputsTPCclusters->clusterIndex;
  using namespace o2::tpc::constants;
  for (int sector = 0; sector < MAXSECTOR; ++sector) {
    for (int padrow = 0; padrow < MAXGLOBALPADROW; ++padrow) {
      for (size_t icl = 0; icl < clusterIndex.nClusters[sector][padrow]; ++icl) {
        const auto& cl = clusterIndex.clusters[sector][padrow][icl];
        // shift by one TPC drift to allow seeing pile-up
        const auto tpcTime = cl.getTime() + mTimeBinsPerDrift;
        const uint32_t clOccPos = static_cast<uint32_t>(tpcTime * mOccupancyBinsPerTF / mTimeBinsPerTF);
        if (clOccPos >= mOccupancyBinsPerTF) {
          LOGP(error, "cluster with time {} outside TPC acceptanc", cl.getTime());
        } else {
          ++mClusterOccupancy[clOccPos];
        }
      }
    }
  }

  // fill track occupancy for its-tpc matched tracks
  auto tpcTracks = recoData.getTPCTracks();
  auto itstpcTracks = recoData.getTPCITSTracks();
  const auto& paramEle = o2::tpc::ParameterElectronics::Instance();

  for (const auto& tpcitsTrack : itstpcTracks) {
    const auto idxTPC = tpcitsTrack.getRefTPC().getIndex();
    if (idxTPC >= tpcTracks.size()) {
      LOGP(fatal, "TPC index {} out of array size {}", idxTPC, tpcTracks.size());
    }
    const auto& tpcTrack = tpcTracks[idxTPC];
    // shift by one TPC drift to allow seeing pile-up
    const auto tpcTime = tpcTrack.getTime0() + mTimeBinsPerDrift;
    if (tpcTime >= 0) {
      const uint32_t clOccPosTPC = static_cast<uint32_t>(tpcTime * mOccupancyBinsPerTF / mTimeBinsPerTF);
      if (clOccPosTPC < mITSTPCTrackOccupanyTPCTime.size()) {
        ++mITSTPCTrackOccupanyTPCTime[clOccPosTPC];
      } else {
        LOGP(warn, "TF {}: TPC occupancy index {} out of range {}", mTFCount, clOccPosTPC, mITSTPCTrackOccupanyTPCTime.size());
      }
    }
    // convert mus to time bins
    // shift by one TPC drift to allow seeing pile-up
    const auto tpcitsTime = tpcitsTrack.getTimeMUS().getTimeStamp() / paramEle.ZbinWidth + mTimeBinsPerDrift;
    if (tpcitsTime > 0) {
      const uint32_t clOccPosITSTPC = static_cast<uint32_t>(tpcitsTime * mOccupancyBinsPerTF / mTimeBinsPerTF);
      if (clOccPosITSTPC < mITSTPCTrackOccupanyCombinedTime.size()) {
        ++mITSTPCTrackOccupanyCombinedTime[clOccPosITSTPC];
      }
    }
  }

  auto fillDebug = [this](o2::utils::TreeStreamRedirector* streamer) {
    if (streamer) {
      *streamer << "occupancy"
                << "tfCounter=" << mTFCount
                << "clusterOcc=" << mClusterOccupancy
                << "tpcTrackTimeOcc=" << mITSTPCTrackOccupanyTPCTime
                << "itstpcTrackTimeOcc=" << mITSTPCTrackOccupanyCombinedTime
                << "\n";
    }
  };

  fillDebug(mDBGOutTPC.get());
  fillDebug(mDBGOutITSTPC.get());
}

void TPCRefitterSpec::process(o2::globaltracking::RecoContainer& recoData)
{
  auto prop = o2::base::Propagator::Instance();

  mITSTracksArray = recoData.getITSTracks();
  mTPCTracksArray = recoData.getTPCTracks();
  mITSTPCTracksArray = recoData.getTPCITSTracks();
  mCosmics = recoData.getCosmicTracks();

  mTPCTrackClusIdx = recoData.getTPCTracksClusterRefs();
  mTPCClusterIdxStruct = &recoData.inputsTPCclusters->clusterIndex;
  mTPCRefitterShMap = recoData.clusterShMapTPC;
  mTPCRefitterOccMap = recoData.occupancyMapTPC;

  LOGP(info, "Processing TF {} with {} its, {} tpc, {} its-tpc tracks and {} comsmics", mTFCount, mITSTracksArray.size(), mTPCTracksArray.size(), mITSTPCTracksArray.size(), mCosmics.size());
  if (mUseMC) { // extract MC tracks
    const o2::steer::DigitizationContext* digCont = nullptr;
    if (!mcReader.initFromDigitContext("collisioncontext.root")) {
      throw std::invalid_argument("initialization of MCKinematicsReader failed");
    }
    digCont = mcReader.getDigitizationContext();
    mIntRecs = digCont->getEventRecords();
    mTPCTrkLabels = recoData.getTPCTracksMCLabels();
  }

  mTPCRefitter = std::make_unique<o2::gpu::GPUO2InterfaceRefit>(mTPCClusterIdxStruct, &mTPCCorrMapsLoader, prop->getNominalBz(), mTPCTrackClusIdx.data(), 0, mTPCRefitterShMap.data(), mTPCRefitterOccMap.data(), mTPCRefitterOccMap.size(), nullptr, prop);
  mTPCRefitter->setTrackReferenceX(900); // disable propagation after refit by setting reference to value > 500

  mVdriftTB = mTPCVDriftHelper.getVDriftObject().getVDrift() * o2::tpc::ParameterElectronics::Instance().ZbinWidth; // VDrift expressed in cm/TimeBin
  mTPCTBBias = mTPCVDriftHelper.getVDriftObject().getTimeOffset() / (8 * o2::constants::lhc::LHCBunchSpacingMUS);

  auto dumpClusters = [this] {
    static int tf = 0;
    const auto* corrMap = this->mTPCCorrMapsLoader.getCorrMap();
    for (int sector = 0; sector < 36; sector++) {
      float alp = ((sector % 18) * 20 + 10) * TMath::DegToRad();
      float sn = TMath::Sin(alp), cs = TMath::Cos(alp);
      for (int row = 0; row < 152; row++) {
        for (int ic = 0; ic < this->mTPCClusterIdxStruct->nClusters[sector][row]; ic++) {
          const auto cl = this->mTPCClusterIdxStruct->clusters[sector][row][ic];
          float x, y, z, xG, yG;
          corrMap->TransformIdeal(sector, row, cl.getPad(), cl.getTime(), x, y, z, 0);
          o2::math_utils::detail::rotateZ(x, y, xG, yG, sn, cs);
          LOGP(debug, "tf:{} s:{} r:{} p:{} t:{} qm:{} qt:{} f:{} x:{} y:{} z:{}", tf, sector, row, cl.getPad(), cl.getTime(), cl.getQmax(), cl.getQtot(), cl.getFlags(), x, y, z);
          (*mDBGOutCl) << "tpccl"
                       << "tf=" << tf << "sect=" << sector << "row=" << row << "pad=" << cl.getPad() << "time=" << cl.getTime() << "qmax=" << cl.getQmax() << "qtot=" << cl.getQtot()
                       << "sigT=" << cl.getSigmaTime() << "sigP=" << cl.getSigmaPad()
                       << "flags=" << cl.getFlags()
                       << "x=" << x << "y=" << y << "z=" << z << "xg=" << xG << "yg=" << yG
                       << "\n";
        }
      }
    }
    tf++;
  };

  if (mDBGOutCl) {
    dumpClusters();
  }

  if ((mStudyType & StudyType::TPC) == StudyType::TPC) {
    for (size_t itr = 0; itr < mTPCTracksArray.size(); itr++) {
      processTPCTrack(mTPCTracksArray[itr], mUseMC ? mTPCTrkLabels[itr] : o2::MCCompLabel{}, mDBGOutTPC.get());
    }
  }

  if ((mStudyType & StudyType::ITSTPC) == StudyType::ITSTPC) {
    for (const auto& tpcitsTrack : mITSTPCTracksArray) {
      const auto idxTPC = tpcitsTrack.getRefTPC().getIndex();
      const auto idxITS = tpcitsTrack.getRefITS().getIndex();
      if (idxITS >= mITSTracksArray.size()) {
        LOGP(fatal, "ITS index {} out of array size {}", idxITS, mITSTracksArray.size());
      }
      if (idxTPC >= mTPCTracksArray.size()) {
        LOGP(fatal, "TPC index {} out of array size {}", idxTPC, mTPCTracksArray.size());
      }
      processTPCTrack(mTPCTracksArray[idxTPC], mUseMC ? mTPCTrkLabels[idxTPC] : o2::MCCompLabel{}, mDBGOutITSTPC.get(), &mITSTracksArray[idxITS], &tpcitsTrack);
    }
  }

  if (mCosmics.size() > 0) {
    LOGP(info, "Procssing {} cosmics", mCosmics.size());
    processCosmics(recoData);
  }
}

void TPCRefitterSpec::endOfStream(EndOfStreamContext& ec)
{
  mDBGOutTPC.reset();
  mDBGOutITSTPC.reset();
  mDBGOutTPCTF.reset();
  mDBGOutITSTPCTF.reset();
  mDBGOutCosmics.reset();
  mDBGOutCl.reset();
}

void TPCRefitterSpec::finaliseCCDB(ConcreteDataMatcher& matcher, void* obj)
{
  if (o2::base::GRPGeomHelper::instance().finaliseCCDB(matcher, obj)) {
    return;
  }
  if (mTPCVDriftHelper.accountCCDBInputs(matcher, obj)) {
    return;
  }
  if (mTPCCorrMapsLoader.accountCCDBInputs(matcher, obj)) {
    return;
  }
}

bool TPCRefitterSpec::getDCAs(const o2::track::TrackPar& track, float& dcar, float& dcaz)
{
  auto propagator = o2::base::Propagator::Instance();
  o2::gpu::gpustd::array<float, 2> dca;
  const o2::math_utils::Point3D<float> refPoint{0, 0, 0};
  o2::track::TrackPar propTrack(track);
  const auto ok = propagator->propagateToDCABxByBz(refPoint, propTrack, 2., o2::base::Propagator::MatCorrType::USEMatCorrLUT, &dca);
  dcar = dca[0];
  dcaz = dca[1];
  if (!ok) {
    dcar = 9998.;
    dcaz = 9998.;
  }
  return ok;
}

bool TPCRefitterSpec::processTPCTrack(o2::tpc::TrackTPC tr, o2::MCCompLabel lbl, o2::utils::TreeStreamRedirector* streamer, const o2::its::TrackITS* its, const o2::dataformats::TrackTPCITS* itstpc, bool outward, float time0custom)
{
  auto prop = o2::base::Propagator::Instance();
  static long counter = -1;

  struct ClusterData {
    std::vector<int> occCl;
    std::vector<short> clSector, clRow;
    std::vector<float> clX, clY, clZ, clXI, clYI, clZI; // *I are the uncorrected cluster positions
    std::vector<tpc::ClusterNative> clNative;
  } clData;
  float dcar, dcaz, dcarRef, dcazRef;

  // auto tr = mTPCTracksArray[itr]; // create track copy
  if (tr.hasBothSidesClusters()) {
    return false;
  }

  bool sampleTsallis = false;
  bool sampleMB = false;
  float tsallisWeight = 0;
  if (mDoSampling) {
    std::uniform_real_distribution<> distr(0., 1.);
    if (o2::math_utils::Tsallis::downsampleTsallisCharged(tr.getPt(), mSamplingFactor, mSqrt, tsallisWeight, distr(mGenerator))) {
      sampleTsallis = true;
    }
    if (distr(mGenerator) < mSamplingFactor) {
      sampleMB = true;
    }

    if (!sampleMB && !sampleTsallis) {
      return false;
    }
  }
  //=========================================================================
  // create refitted copy
  auto trackRefit = [&tr, this](o2::track::TrackParCov& trc, float t, float chi2refit, bool outward = false) -> bool {
    int retVal = mUseGPUModel ? this->mTPCRefitter->RefitTrackAsGPU(trc, tr.getClusterRef(), t, &chi2refit, outward, true)
                              : this->mTPCRefitter->RefitTrackAsTrackParCov(trc, tr.getClusterRef(), t, &chi2refit, outward, true);
    if (retVal < 0) {
      LOGP(warn, "Refit failed ({}) with time={}: track#{}[{}]", retVal, t, counter, trc.asString());
      return false;
    }
    return true;
  };

  auto trackProp = [&tr, prop, this](o2::track::TrackParCov& trc) -> bool {
    if (!trc.rotate(tr.getAlpha())) {
      LOGP(warn, "Rotation to original track alpha {} failed, track#{}[{}]", tr.getAlpha(), counter, trc.asString());
      return false;
    }
    float xtgt = this->mXRef;
    if (mUseR && !trc.getXatLabR(this->mXRef, xtgt, prop->getNominalBz(), o2::track::DirInward)) {
      xtgt = 0;
      return false;
    }
    if (!prop->PropagateToXBxByBz(trc, xtgt)) {
      LOGP(warn, "Propagation to X={} failed, track#{}[{}]", xtgt, counter, trc.asString());
      return false;
    }
    return true;
  };

  // auto prepClus = [this, &tr, &clSector, &clRow, &clX, &clY, &clZ, &clXI, &clYI, &clZI, &clNative](float t) { // extract cluster info
  auto prepClus = [this, &tr, &clData](float t) { // extract cluster info
    int count = tr.getNClusters();
    const auto* corrMap = this->mTPCCorrMapsLoader.getCorrMap();
    const o2::tpc::ClusterNative* cl = nullptr;
    for (int ic = count; ic--;) {
      uint8_t sector, row;
      uint32_t clusterIndex;
      o2::tpc::TrackTPC::getClusterReference(mTPCTrackClusIdx, ic, sector, row, clusterIndex, tr.getClusterRef());
      unsigned int absoluteIndex = mTPCClusterIdxStruct->clusterOffset[sector][row] + clusterIndex;
      cl = &mTPCClusterIdxStruct->clusters[sector][row][clusterIndex];
      uint8_t clflags = cl->getFlags();
      if (mTPCRefitterShMap[absoluteIndex] & GPUCA_NAMESPACE::gpu::GPUTPCGMMergedTrackHit::flagShared) {
        clflags |= 0x10;
      }
      clData.clSector.emplace_back(sector);
      clData.clRow.emplace_back(row);
      auto& clCopy = clData.clNative.emplace_back(*cl);
      clCopy.setFlags(clflags);

      float x, y, z;
      // ideal transformation without distortions
      corrMap->TransformIdeal(sector, row, cl->getPad(), cl->getTime(), x, y, z, t); // nominal time of the track
      clData.clXI.emplace_back(x);
      clData.clYI.emplace_back(y);
      clData.clZI.emplace_back(z);

      // transformation without distortions
      mTPCCorrMapsLoader.Transform(sector, row, cl->getPad(), cl->getTime(), x, y, z, t); // nominal time of the track
      clData.clX.emplace_back(x);
      clData.clY.emplace_back(y);
      clData.clZ.emplace_back(z);

      // occupancy estimator
      const auto tpcTime = cl->getTime() + mTimeBinsPerDrift;
      const uint32_t clOccPosTPC = static_cast<uint32_t>(tpcTime * mOccupancyBinsPerTF / mTimeBinsPerTF);
      clData.occCl.emplace_back((clOccPosTPC < mClusterOccupancy.size()) ? mClusterOccupancy[clOccPosTPC] : -1);
    }
  };

  //=========================================================================

  auto trf = tr.getOuterParam(); // we refit inward original track
  float chi2refit = 0;
  float time0 = tr.getTime0();
  if (time0custom > 0) {
    time0 = time0custom;
  }
  if (mDoRefit) {
    if (!trackRefit(trf, time0, chi2refit) || !trackProp(trf)) {
      return false;
    }
  }

  // propagate original track
  if (!trackProp(tr)) {
    return false;
  }

  if (mWriteTrackClusters) {
    prepClus(time0); // original clusters
  }

  if (mEnableDCA) {
    dcar = dcaz = dcarRef = dcazRef = 9999.f;
    if ((trf.getPt() > mDCAMinPt) && (tr.getNClusters() > mDCAMinNCl)) {
      getDCAs(trf, dcarRef, dcazRef);
      getDCAs(tr, dcar, dcaz);
    }
  }

  counter++;
  // store results
  if (streamer) {
    (*streamer) << "tpc"
                << "counter=" << counter
                << "tfCounter=" << mTFCount
                << "tpc=" << tr;

    if (mDoRefit) {
      (*streamer) << "tpc"
                  << "tpcRF=" << trf
                  << "time0=" << time0
                  << "chi2refit=" << chi2refit;
    }

    if (mDoSampling) {
      (*streamer) << "tpc"
                  << "tsallisWeight=" << tsallisWeight
                  << "sampleTsallis=" << sampleTsallis
                  << "sampleMB=" << sampleMB;
    }

    if (mWriteTrackClusters) {
      (*streamer) << "tpc"
                  << "clSector=" << clData.clSector
                  << "clRow=" << clData.clRow;

      if ((mWriteTrackClusters & 0x1) == 0x1) {
        (*streamer) << "tpc"
                    << "cl=" << clData.clNative;
      }

      if ((mWriteTrackClusters & 0x2) == 0x2) {
        (*streamer) << "tpc"
                    << "clX=" << clData.clX
                    << "clY=" << clData.clY
                    << "clZ=" << clData.clZ;
      }

      if ((mWriteTrackClusters & 0x4) == 0x4) {
        (*streamer) << "tpc"
                    << "clXI=" << clData.clXI  // ideal (uncorrected) cluster positions
                    << "clYI=" << clData.clYI  // ideal (uncorrected) cluster positions
                    << "clZI=" << clData.clZI; // ideal (uncorrected) cluster positions
      }

      if ((mWriteTrackClusters & 0x8) == 0x8) {
        (*streamer) << "tpc"
                    << "clOcc=" << clData.occCl;
      }
    }

    if (its) {
      (*streamer) << "tpc"
                  << "its=" << *its;
    }
    if (itstpc) {
      (*streamer) << "tpc"
                  << "itstpc=" << *itstpc;
    }

    if (mEnableDCA) {
      (*streamer) << "tpc"
                  << "dcar=" << dcar
                  << "dcaz=" << dcaz
                  << "dcarRef=" << dcarRef
                  << "dcazRef=" << dcazRef;
    }

    (*streamer) << "tpc"
                << "\n";
  }

  float dz = 0;

  if (mUseMC) { // impose MC time in TPC timebin and refit inward after resetted covariance
                // extract MC truth
    const o2::MCTrack* mcTrack = nullptr;
    if (!lbl.isValid() || !(mcTrack = mcReader.getTrack(lbl))) {
      return false;
    }
    long bc = mIntRecs[lbl.getEventID()].toLong(); // bunch crossing of the interaction
    float bcTB = bc / 8. + mTPCTBBias;             // the same in TPC timebins, accounting for the TPC time bias
                                                   // create MC truth track in O2 format
    std::array<float, 3> xyz{(float)mcTrack->GetStartVertexCoordinatesX(), (float)mcTrack->GetStartVertexCoordinatesY(), (float)mcTrack->GetStartVertexCoordinatesZ()},
      pxyz{(float)mcTrack->GetStartVertexMomentumX(), (float)mcTrack->GetStartVertexMomentumY(), (float)mcTrack->GetStartVertexMomentumZ()};
    TParticlePDG* pPDG = TDatabasePDG::Instance()->GetParticle(mcTrack->GetPdgCode());
    if (!pPDG) {
      return false;
    }
    o2::track::TrackPar mctrO2(xyz, pxyz, TMath::Nint(pPDG->Charge() / 3), false);
    //
    // propagate it to the alpha/X of the reconstructed track
    if (!mctrO2.rotate(tr.getAlpha()) || !prop->PropagateToXBxByBz(mctrO2, tr.getX())) {
      return false;
    }
    // now create a properly refitted track with correct time and distortions correction
    {
      auto trfm = tr.getOuterParam(); // we refit inward
                                      // impose MC time in TPC timebin and refit inward after resetted covariance
      float chi2refit = 0;
      if (!trackRefit(trfm, bcTB, chi2refit) || !trfm.rotate(tr.getAlpha()) || !prop->PropagateToXBxByBz(trfm, tr.getX())) {
        LOGP(warn, "Failed to propagate MC-time refitted track#{} [{}] to X/alpha of original track [{}]", counter, trfm.asString(), tr.asString());
        return false;
      }
      // estimate Z shift in case of no-distortions
      dz = (tr.getTime0() - bcTB) * mVdriftTB;
      if (tr.hasCSideClustersOnly()) {
        dz = -dz;
      }
      //
      prepClus(bcTB); // clusters for MC time
      if (streamer) {
        (*streamer) << "tpcMC"
                    << "counter=" << counter
                    << "movTrackRef=" << trfm
                    << "mcTrack=" << mctrO2
                    << "imposedTB=" << bcTB
                    << "chi2refit=" << chi2refit
                    << "dz=" << dz
                    << "clX=" << clData.clX
                    << "clY=" << clData.clY
                    << "clZ=" << clData.clZ
                    << "\n";
      }
    }
    return false;
  }

  return true;
}

void TPCRefitterSpec::processCosmics(o2::globaltracking::RecoContainer& recoData)
{
  auto tof = recoData.getTOFClusters();
  const auto& par = o2::tpc::ParameterElectronics::Instance();
  const auto invBinWidth = 1.f / par.ZbinWidth;

  for (const auto& cosmic : mCosmics) {
    //
    const auto& gidtop = cosmic.getRefTop();
    const auto& gidbot = cosmic.getRefBottom();

    // LOGP(info, "Sources: {} - {}", o2::dataformats::GlobalTrackID::getSourceName(gidtop.getSource()), o2::dataformats::GlobalTrackID::getSourceName(gidbot.getSource()));

    std::array<GTrackID, GTrackID::NSources> contributorsGID[2] = {recoData.getSingleDetectorRefs(cosmic.getRefTop()), recoData.getSingleDetectorRefs(cosmic.getRefBottom())};
    const auto trackTime = cosmic.getTimeMUS().getTimeStamp() * invBinWidth;

    // check if track has TPC & TOF for top and bottom part
    // loop over both parts
    for (const auto& comsmicInfo : contributorsGID) {
      auto& tpcGlobal = comsmicInfo[GTrackID::TPC];
      auto& tofGlobal = comsmicInfo[GTrackID::TOF];
      if (tpcGlobal.isIndexSet() && tofGlobal.isIndexSet()) {
        const auto itrTPC = tpcGlobal.getIndex();
        const auto itrTOF = tofGlobal.getIndex();
        const auto& tofCl = tof[itrTOF];
        const auto tofTime = tofCl.getTime() * 1e-6 * invBinWidth;       // ps -> us -> time bins
        const auto tofTimeRaw = tofCl.getTimeRaw() * 1e-6 * invBinWidth; // ps -> us -> time bins
        const auto& trackTPC = mTPCTracksArray[itrTPC];
        // LOGP(info, "Cosmic time: {}, TOF time: {}, TOF time raw: {}, TPC time: {}", trackTime, tofTime, tofTimeRaw, trackTPC.getTime0());
        processTPCTrack(trackTPC, mUseMC ? mTPCTrkLabels[itrTPC] : o2::MCCompLabel{}, mDBGOutCosmics.get(), nullptr, nullptr, false, tofTime);
      }
    }
  }
}

DataProcessorSpec getTPCRefitterSpec(GTrackID::mask_t srcTracks, GTrackID::mask_t srcClusters, bool useMC, const o2::tpc::CorrectionMapsLoaderGloOpts& sclOpts, bool requestCosmics)
{
  std::vector<OutputSpec> outputs;
  Options opts{
    {"target-x", VariantType::Float, 83.f, {"Try to propagate to this radius"}},
    {"dump-clusters", VariantType::Bool, false, {"dump all clusters"}},
    {"write-track-clusters", VariantType::Int, 3, {"Bitmask write clusters associated to the track, full native cluster (0x1), corrected (0x2) and uncorrected (0x4) positions, (0x8) occupancy info"}},
    {"tf-start", VariantType::Int, 0, {"1st TF to process"}},
    {"tf-end", VariantType::Int, 999999999, {"last TF to process"}},
    {"use-gpu-fitter", VariantType::Bool, false, {"use GPU track model for refit instead of TrackParCov"}},
    {"do-refit", VariantType::Bool, false, {"do refitting of TPC track"}},
    {"use-r-as-x", VariantType::Bool, false, {"Use radius instead of target sector X"}},
    {"enable-dcas", VariantType::Bool, false, {"Propagate to DCA and add it to the tree"}},
    {"dcaMinPt", VariantType::Float, 1.f, {"Min pT of tracks propagated to DCA"}},
    {"dcaMinNCl", VariantType::Int, 80, {"Min number of clusters for tracks propagated to DCA"}},
    {"sqrts", VariantType::Float, 13600.f, {"Centre of mass energy used for downsampling"}},
    {"do-sampling", VariantType::Bool, false, {"Perform sampling, min. bias and on Tsallis function, using 'sampling-factor'"}},
    {"sampling-factor", VariantType::Float, 0.1f, {"Sampling factor in case sample-unbinned-tsallis is used"}},
    {"study-type", VariantType::Int, 1, {"Bitmask of study type: 0x1 = TPC only, 0x2 = TPC + ITS, 0x4 = Cosmics"}},
    {"writer-type", VariantType::Int, 1, {"Bitmask of writer type: 0x1 = per track streamer, 0x2 = per TF vectors"}},
    {"occupancy-bins-per-drift", VariantType::UInt32, 31u, {"number of bin for occupancy histogram per drift time (500tb)"}},
  };
  auto dataRequest = std::make_shared<DataRequest>();

  dataRequest->requestTracks(srcTracks, useMC);
  dataRequest->requestClusters(srcClusters, useMC);
  if (requestCosmics) {
    dataRequest->requestCoscmicTracks(useMC);
  }
  auto ggRequest = std::make_shared<o2::base::GRPGeomRequest>(false,                             // orbitResetTime
                                                              true,                              // GRPECS=true
                                                              false,                             // GRPLHCIF
                                                              true,                              // GRPMagField
                                                              true,                              // askMatLUT
                                                              o2::base::GRPGeomRequest::Aligned, // geometry
                                                              dataRequest->inputs,
                                                              true);
  o2::tpc::VDriftHelper::requestCCDBInputs(dataRequest->inputs);
  o2::tpc::CorrectionMapsLoader::requestCCDBInputs(dataRequest->inputs, opts, sclOpts);

  return DataProcessorSpec{
    "tpc-refitter",
    dataRequest->inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<TPCRefitterSpec>(dataRequest, ggRequest, sclOpts, srcTracks, useMC)},
    opts};
}

} // namespace o2::trackstudy
