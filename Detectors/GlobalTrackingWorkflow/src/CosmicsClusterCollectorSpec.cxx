// Copyright 2019-2026 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   CosmicsClusterCollectorSpec.cxx
/// @brief  Collects the raw clusters of matched cosmics (attached + road around the legs) for offline refits
///
/// The async reconstruction stores neither TPC tracks nor TPC clusters, so the leg references of TrackCosmics cannot be resolved
/// offline. For every cosmic this device stores the TPC tracks of both legs, the raw TPC clusters attached to them, all raw TPC clusters
/// in a road around each leg (gaps, split pieces, delta electrons; clusters attached to other tracks are flagged), the ITS / TOF / TRD
/// hits of the legs' matched tracks and along roads in these detectors, and per TF the quantities of the TPC transformation.
///
/// Road: each leg's helix is propagated through all pad rows of all sectors on its TPC side, in the frame of its own time0 (the frame
/// in which its clusters are consistent). Points on the other side of the leg's closest approach to the beam line belong to the other
/// leg and are skipped. The predicted real (y, z) is mapped to nominal coordinates with the inverse correction, and clusters with
/// nominal coordinates within the corridor width of the predicted point (distance perpendicular to the track) are taken. The other TPC
/// side is searched only when the time of the cosmic is known (z-continuity time of legs on opposite sides with a small error, or a TOF
/// time); a CE-crossing leg's own frame already covers both sides. Parts of a leg nearly parallel to the pad rows (|snp| >= 0.8 in the
/// sector frame, e.g. around a closest approach to the beam line inside the TPC, where the track runs along one pad row) cannot be
/// reached row by row: there the helix is followed in steps of path length and the rows within the road width of each point are searched.
///
/// TOF tag: all TOF clusters close to the outward extrapolation of the legs within the cosmic's time window are candidates; the top /
/// bottom pair whose time difference matches the muon's flight along the helix (HitTOFFlight) gives the cosmic its TOF time. The same
/// search in the impossible order (bottom hit first) only finds accidental pairs and is kept as QA of the flag's background.
///
/// Roads in the other detectors (--road-detectors): TRD tracklets close to the outward extrapolation of the legs and ITS clusters close
/// to the trajectory near the beam line, within the time window of the cosmic (the TOF time if there is one, else the matcher's time,
/// which is precise for TPC-only legs on opposite sides; otherwise the time window of the legs, with a correspondingly loose z cut).
/// They are flagged as found on the road; hits of the legs' matched global tracks are flagged as matched. In the ITS the road keeps per
/// half of the cosmic and layer the cluster closest to the trajectory, in the inner barrel (dense with collision clusters near the beam
/// line) the 5 closest, best first. Cosmics sharing >= 30 % of their TPC clusters with a better one (a leg split into two TPC tracks) are
/// flagged as duplicates, nothing is removed.
///
/// Polish: a cosmic with a TOF time and TPC-only legs is refitted at that time (as in the matcher: muon mass, energy loss along the
/// flight); one-side legs then have their real z, hence the right material, which the matcher's TPC time cannot always give.
///
/// Besides the matcher's and the polished track, only raw detector data are stored (TPC ClusterNative, ITS compact clusters + patterns,
/// raw TOF time, TRD tracklet words); calibrations, the cluster dictionary and the geometry are applied offline.
///
/// --debug-tree writes cosmics_collector_debug.root for test runs: tree "cosmics" with one entry per cosmic, its matching and timing
/// quantities and vectors per detector: cl* every stored TPC cluster transformed (local x, y, z in the frame used by the road, zCos in the
/// common frame of the cosmic's time, global gx, gy), road* the predicted track points of each leg (same frames; roadFrame 0 / 1 own
/// / cosmic's time frame row by row, 2 / 3 the same along the low-angle walk), its* the
/// local (with the ITS road also global) coordinates of the ITS clusters, tof* the raw and calibrated TOF time and the cluster position,
/// trd* the road tracklets and their residuals. E.g. Draw("clGy:clGx", "scoreTOF >= 0") (TOF-tagged; tTOF can be negative).

#include <vector>
#include <array>
#include <limits>
#include <unordered_set>
#include <algorithm>
#include <numeric>
#include <iterator>
#include <cmath>
#include "TStopwatch.h"
#include "Framework/Task.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/DeviceSpec.h"
#include "GlobalTrackingWorkflow/CosmicsClusterCollectorSpec.h"
#include "GlobalTrackingWorkflow/CosmicsMatchingSpec.h"
#include "DataFormatsGlobalTracking/RecoContainer.h"
#include "DataFormatsGlobalTracking/TrackCosmicsExtended.h"
#include "ReconstructionDataFormats/TrackCosmics.h"
#include "DataFormatsTPC/TrackTPC.h"
#include "DataFormatsTPC/ClusterNative.h"
#include "DataFormatsITS/TrackITS.h"
#include "DataFormatsITSMFT/CompCluster.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "DataFormatsITSMFT/TopologyDictionary.h"
#include "DataFormatsITSMFT/ClusterPattern.h"
#include "ITSMFTBase/SegmentationAlpide.h"
#include "DataFormatsTOF/Cluster.h"
#include "DataFormatsTRD/TrackTRD.h"
#include "DataFormatsTRD/Tracklet64.h"
#include "DataFormatsTRD/TriggerRecord.h"
#include "DataFormatsTRD/CalibratedTracklet.h"
#include "DataFormatsTRD/Constants.h"
#include "DataFormatsITSMFT/DPLAlpideParam.h"
#include "ITSBase/GeometryTGeo.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "CommonConstants/LHCConstants.h"
#include "CommonDataFormat/TFIDInfo.h"
#include "CommonDataFormat/InteractionRecord.h"
#include "DetectorsBase/GRPGeomHelper.h"
#include "DetectorsBase/Propagator.h"
#include "GPUO2InterfaceRefit.h"
#include "GlobalTracking/MatchCosmicsParams.h"
#include "DataFormatsTPC/WorkflowHelper.h"
#include "ReconstructionDataFormats/Vertex.h"
#include "DetectorsBase/TFIDInfoHelper.h"
#include "TPCBase/ParameterElectronics.h"
#include "MathUtils/Utils.h"
#include "MathUtils/Primitive2D.h"
#include "TPCFastTransformPOD.h"
#include "CommonUtils/TreeStreamRedirector.h"

using namespace o2::framework;
using GTrackID = o2::dataformats::GlobalTrackID;
using TPCGeo = o2::gpu::TPCFastTransformGeoPOD;
using DetID = o2::detectors::DetID;

namespace o2::globaltracking
{

namespace
{
constexpr float TanSector = 0.17632698f; // tan(10 deg): half opening of a sector

/// z outside the drift volume of one TPC side by more than margin
bool outsideDriftVolume(bool sideA, float z, float margin)
{
  const float zLength = TPCGeo::getTPCzLength();
  return sideA ? (z < -margin || z > zLength + margin) : (z > margin || z < -zLength - margin);
}

/// TPC side of a track with clusters on one side only (+1 A, -1 C: its z moves with the time assumed for its clusters), 0 otherwise
int tpcSide(const o2::tpc::TrackTPC& trk)
{
  return trk.hasASideClustersOnly() ? 1 : (trk.hasCSideClustersOnly() ? -1 : 0);
}
} // namespace

class CosmicsClusterCollectorSpec : public Task
{
 public:
  CosmicsClusterCollectorSpec(std::shared_ptr<DataRequest> dr, std::shared_ptr<o2::base::GRPGeomRequest> gr, bool useMC, DetID::mask_t roadDets) : mDataRequest(dr), mGGCCDBRequest(gr), mRoadDets(roadDets), mUseMC(useMC) {}
  ~CosmicsClusterCollectorSpec() override = default;
  void init(InitContext& ic) final;
  void run(ProcessingContext& pc) final;
  void endOfStream(EndOfStreamContext& ec) final;
  void finaliseCCDB(ConcreteDataMatcher& matcher, void* obj) final;

 private:
  /// which side of the closest approach to the beam line (transverse) a point is on, relative to the leg's own clusters
  struct LegBranch {
    bool isLine = false; ///< straight line (no field / very high pT) instead of a circle
    float centerX = 0.f; ///< circle centre
    float centerY = 0.f;
    float pcaX = 0.f; ///< point of closest approach to the beam line
    float pcaY = 0.f;
    float dirX = 0.f; ///< direction of the straight line
    float dirY = 0.f;
    int sign = 0; ///< side of the leg's clusters; 0: the leg spans both sides, accept everything
    int side(float x, float y) const
    {
      const float orientation = isLine ? (x - pcaX) * dirX + (y - pcaY) * dirY : (pcaX - centerX) * (y - centerY) - (pcaY - centerY) * (x - centerX);
      return orientation > 0.f ? 1 : -1;
    }
    bool accept(float x, float y) const { return sign == 0 || side(x, y) == sign; }
    void init(const o2::track::TrackPar& inner, const o2::track::TrackPar& outer, float bz);
  };

  /// time of the cosmic in TPC time bins
  struct CosmicTime {
    bool known = false; ///< known well enough to search the other TPC side of one-side legs
    float tb = 0.f;     ///< time [TB]
    float errTB = 0.f;  ///< its error [TB]
  };

  void updateTimeDependentParams(ProcessingContext& pc);
  void buildUsedMap(const RecoContainer& data, std::vector<std::pair<float, float>>& time0Windows);
  void addTPCAttached(const RecoContainer& data, const o2::tpc::TrackTPC& trk, std::vector<o2::dataformats::CosmicTPCCluster>& out, std::unordered_set<uint32_t>& taken) const;
  /// time frame of a TPC road: the leg's z is shifted by dz and its clusters are transformed with vertexTime
  struct RoadFrame {
    float vertexTime; ///< vertex time used for the transformation [TB]
    float dz;         ///< shift of the leg's z into this frame
    float zTolerance; ///< extra z tolerance from the error of the vertex time
    int sectorMin;
    int sectorMax;
    uint8_t flag;
  };
  void addTPCCorridor(const RecoContainer& data, const o2::tpc::TrackTPC& trk, const CosmicTime& cosmicTime, std::vector<o2::dataformats::CosmicTPCCluster>& out, std::unordered_set<uint32_t>& taken) const;
  void walkLowAngleRoad(const o2::tpc::ClusterNativeAccess& clusters, const o2::track::TrackPar& start, bool innerPart, float rMid, const LegBranch& branch, const RoadFrame& frame,
                        std::vector<o2::dataformats::CosmicTPCCluster>& out, std::unordered_set<uint32_t>& taken) const;
  void searchRow(const o2::tpc::ClusterNativeAccess& clusters, int sector, int row, float y, float z, float snp, float tgl, float vertexTime, float zTolerance, uint8_t flag,
                 float dxRow, std::vector<o2::dataformats::CosmicTPCCluster>& out, std::unordered_set<uint32_t>& taken) const;
  void addDebugRoadPoint(int sector, float x, float y, float z, float snp, float tgl, int row, float vertexTime, int frameCode) const;
  struct ITSPattRequest {
    int cosmic;  ///< entry of the cosmic in the output
    int cluster; ///< entry of the cluster in its clITS
    int index;   ///< index of the cluster in the TF
  };
  void addITS(const RecoContainer& data, GTrackID gid, uint8_t leg, std::vector<o2::dataformats::CosmicITSCluster>& out, int icosm, std::vector<ITSPattRequest>& requests, std::unordered_set<int>& matched) const;
  void fillITSPatterns(const RecoContainer& data, std::vector<ITSPattRequest>& requests, std::vector<o2::dataformats::TrackCosmicsExtended>& cosmics) const;
  void addTOF(const RecoContainer& data, GTrackID gid, uint8_t leg, std::vector<o2::dataformats::CosmicTOFCluster>& out, std::unordered_set<int>& matched) const;
  void addTRD(const RecoContainer& data, GTrackID gid, uint8_t leg, std::vector<o2::dataformats::CosmicTRDTracklet>& out, std::unordered_set<int>& matched) const;
  std::pair<float, float> timeWindowMUS(const CosmicTime& cosmicTime) const;
  bool predictOutward(const o2::tpc::TrackTPC& leg, const CosmicTime& cosmicTime, int sector, float x, float& y, float& z) const;
  void roadTOF(const RecoContainer& data, const o2::tpc::TrackTPC* const* legs, const CosmicTime& cosmicTime, int icosm, std::vector<o2::dataformats::CosmicTOFCluster>& out, const std::unordered_set<int>& matched, float& timeTOFMUS, float& scoreTOFPair, float& scoreTOFReversed) const;
  void polish(const o2::tpc::TrackTPC* const* legs, float timeMUS, o2::gpu::GPUO2InterfaceRefit& refitter, o2::dataformats::TrackCosmicsExtended& out) const;
  void roadTRD(const RecoContainer& data, const o2::tpc::TrackTPC* const* legs, const CosmicTime& cosmicTime, int icosm, std::vector<o2::dataformats::CosmicTRDTracklet>& out, const std::unordered_set<int>& matched) const;
  void roadITS(const RecoContainer& data, const o2::dataformats::TrackCosmics& cosm, const CosmicTime& cosmicTime, int legsSide, int icosm, std::vector<o2::dataformats::CosmicITSCluster>& out, std::vector<ITSPattRequest>& requests,
               const std::unordered_set<int>& matched) const;
  void cacheITSChipCentres();
  void writeDebug(const o2::dataformats::TrackCosmicsExtended& cosm, int icosm) const;
  void flagDuplicates(std::vector<o2::dataformats::TrackCosmicsExtended>& cosmics) const;
  void writeDebugTOF(const o2::tof::Cluster& c, int icosm, int leg, uint8_t flags) const;

  std::shared_ptr<DataRequest> mDataRequest;
  std::shared_ptr<o2::base::GRPGeomRequest> mGGCCDBRequest;
  const o2::gpu::TPCFastTransformPOD* mCorrMap = nullptr;
  const o2::itsmft::TopologyDictionary* mITSDict = nullptr;
  std::vector<bool> mUsed;                                     ///< TPC clusters attached to any TPC track in this TF
  o2::InteractionRecord mTFStart{};                            ///< first BC of the TF
  float mCorridor = 1.f;                                       ///< road half-width [cm]
  float mMaxAbsTimeErr = 0.5f;                                 ///< max. time error of the cosmic [mus] to search the other TPC side of a one-side leg
  size_t mMaxCosmicsPerTF = 100;                               ///< cosmics processed per TF at most (protection against fake-dominated settings)
  DetID::mask_t mRoadDets{};                                   ///< detectors searched along the road besides the TPC
  float mRoadTOF = 5.f;                                        ///< road half-width at the TOF [cm]
  float mTOFFlightTol = 2.f;                                   ///< max. deviation of the top/bottom TOF time difference from the flight time [ns]
  float mTOFTimeErr = 0.1f;                                    ///< error of the TOF time of a cosmic for the later roads [mus] (covers TPC vs TOF offsets)
  float mRoadTRD = 5.f;                                        ///< road half-width at the TRD [cm] (in z plus the pad length)
  float mRoadITS = 1.5f;                                       ///< road half-width in the ITS [cm]
  std::vector<o2::math_utils::Point3D<float>> mITSChipCentres; ///< global positions of the ITS chip centres (aligned geometry)
  float mTPCTBinMUS = 0.2f;                                    ///< TPC time bin [mus]
  float mBz = 0.f;
  bool mUseMC = false;
  mutable bool mStaggeredWarned = false;
  mutable bool mNoDictWarned = false;
  mutable bool mTRDMismatchWarned = false;
  std::unique_ptr<o2::utils::TreeStreamRedirector> mDebugOut; ///< debug tree "cosmics", one entry per cosmic (--debug-tree)
  int mDbgTF = 0;                                             ///< context of the debug output: TF counter
  int mDbgCosmic = 0;                                         ///< entry of the cosmic in the TF
  int mDbgLeg = 0;                                            ///< leg (0 bottom, 1 top)
  float mDbgTauC = 0.f;                                       ///< time of the cosmic [TB]: common frame (zCos) of the debug output
  /// debug quantities found while processing a cosmic (road points, TOF and TRD hits), written with its entry
  struct DebugCosmic {
    std::vector<int> roadLeg;
    std::vector<int> roadFrame;
    std::vector<int> roadSector;
    std::vector<int> roadRow;
    std::vector<float> roadX;
    std::vector<float> roadY;
    std::vector<float> roadZ;
    std::vector<float> roadZCos;
    std::vector<float> roadGx;
    std::vector<float> roadGy;
    std::vector<float> roadSnp;
    std::vector<float> roadTgl;
    std::vector<int> tofLeg;
    std::vector<int> tofChannel;
    std::vector<int> tofFlags;
    std::vector<double> tofTimeRaw;
    std::vector<double> tofTime;
    std::vector<float> tofTot;
    std::vector<float> tofX;
    std::vector<float> tofY;
    std::vector<float> tofZ;
    std::vector<float> tofGx;
    std::vector<float> tofGy;
    std::vector<int> trdLeg;
    std::vector<int> trdLayer;
    std::vector<float> trdX;
    std::vector<float> trdY;
    std::vector<float> trdZ;
    std::vector<float> trdDy;
    std::vector<float> trdDz;
    std::vector<float> trdTrigMUS;
  };
  mutable std::vector<DebugCosmic> mDbgCosmics; ///< per cosmic of the TF
  size_t mNCosmics = 0;                         ///< cosmics processed
  size_t mNClAttached = 0;                      ///< attached TPC clusters stored
  size_t mNClCorridor = 0;                      ///< road TPC clusters stored
  TStopwatch mTimer;
};

void CosmicsClusterCollectorSpec::init(InitContext& ic)
{
  mTimer.Stop();
  mTimer.Reset();
  o2::base::GRPGeomHelper::instance().setRequest(mGGCCDBRequest);
  mCorridor = ic.options().get<float>("corridor-width");
  mMaxAbsTimeErr = ic.options().get<float>("max-abs-time-err");
  mMaxCosmicsPerTF = std::max(0, ic.options().get<int>("max-cosmics-per-tf"));
  mRoadTOF = ic.options().get<float>("tof-road-width");
  mTOFFlightTol = ic.options().get<float>("tof-flight-tolerance");
  mTOFTimeErr = ic.options().get<float>("tof-time-error");
  mRoadTRD = ic.options().get<float>("trd-road-width");
  mRoadITS = ic.options().get<float>("its-road-width");
  if (ic.options().get<bool>("debug-tree")) {
    const auto timesliceId = ic.services().get<const o2::framework::DeviceSpec>().inputTimesliceId;
    const std::string name = timesliceId == 0 ? "cosmics_collector_debug.root" : fmt::format("cosmics_collector_debug_{}.root", timesliceId);
    mDebugOut = std::make_unique<o2::utils::TreeStreamRedirector>(name.c_str(), "recreate");
  }
}

void CosmicsClusterCollectorSpec::run(ProcessingContext& pc)
{
  mTimer.Start(false);
  RecoContainer recoData;
  recoData.collectData(pc, *mDataRequest.get());
  updateTimeDependentParams(pc);

  o2::dataformats::TFIDInfo tfID;
  o2::base::TFIDInfoHelper::fillTFIDInfo(pc, tfID);
  mTFStart = {0, tfID.firstTForbit};
  o2::dataformats::CosmicsTFInfo tfInfo;
  tfInfo.vDrift = mCorrMap->getVDrift();
  tfInfo.t0 = mCorrMap->getT0();

  std::vector<o2::dataformats::TrackCosmicsExtended> cosmicsOut;
  const auto cosmics = recoData.getCosmicTracks();
  const size_t nCosmics = std::min(cosmics.size(), mMaxCosmicsPerTF);
  if (nCosmics < cosmics.size()) {
    LOGP(warning, "{} cosmics in TF {}, only the first {} are written (max-cosmics-per-tf)", cosmics.size(), tfID.tfCounter, nCosmics);
  }
  if (nCosmics) {
    // only TPC tracks whose time0 is within two drift times of a leg's time0 can share clusters with the roads
    const float maxDistTB = 2.2f * TPCGeo::getTPCzLength() / mCorrMap->getVDrift();
    std::vector<std::pair<float, float>> time0Windows;
    for (size_t ic = 0; ic < nCosmics; ic++) {
      for (const auto leg : {cosmics[ic].getRefBottom(), cosmics[ic].getRefTop()}) {
        const auto refs = recoData.getSingleDetectorRefs(leg);
        if (refs[GTrackID::TPC].isIndexSet()) {
          const float time0 = recoData.getTPCTrack(refs[GTrackID::TPC]).getTime0();
          time0Windows.emplace_back(time0 - maxDistTB, time0 + maxDistTB);
        }
      }
    }
    buildUsedMap(recoData, time0Windows);
  }
  std::vector<ITSPattRequest> pattRequests;                  // ITS clusters whose patterns are not in the dictionary
  std::unique_ptr<o2::gpu::GPUO2InterfaceRefit> tpcRefitter; // polish of the cosmics with a TOF time
  if (nCosmics && mRoadDets[DetID::TOF] && recoData.inputsTPCclusters) {
    const auto& clusterShMap = recoData.clusterShMapTPC;
    const auto& occupancyMap = recoData.occupancyMapTPC;
    tpcRefitter = std::make_unique<o2::gpu::GPUO2InterfaceRefit>(&recoData.inputsTPCclusters->clusterIndex, mCorrMap, mBz, recoData.getTPCTracksClusterRefs().data(), 0,
                                                                 clusterShMap.data(), occupancyMap.data(), occupancyMap.size(), nullptr, o2::base::Propagator::Instance());
  }
  mDbgTF = tfID.tfCounter;
  if (mDebugOut) {
    mDbgCosmics.assign(nCosmics, DebugCosmic{});
  }
  for (size_t ic = 0; ic < nCosmics; ic++) {
    const auto& cosm = cosmics[ic];
    auto& out = cosmicsOut.emplace_back();
    out.cosmic = cosm;
    if (mUseMC) {
      out.label = recoData.getCosmicTrackMCLabel(ic);
    }
    const GTrackID legs[2] = {cosm.getRefBottom(), cosm.getRefTop()};
    const o2::tpc::TrackTPC* tpcLegs[2] = {nullptr, nullptr};
    std::vector<o2::dataformats::CosmicTPCCluster>* tpcCl[2] = {&out.clTPCBottom, &out.clTPCTop};
    std::unordered_set<int> matchedITS; // hits of the legs' matched tracks, not searched again on the road
    std::unordered_set<int> matchedTOF;
    std::unordered_set<int> matchedTRD;
    for (uint8_t leg = 0; leg < 2; leg++) {
      auto refs = recoData.getSingleDetectorRefs(legs[leg]);
      if (refs[GTrackID::TPC].isIndexSet()) {
        tpcLegs[leg] = &recoData.getTPCTrack(refs[GTrackID::TPC]);
        (leg == 0 ? out.tpcBottom : out.tpcTop) = *tpcLegs[leg];
      }
      if (refs[GTrackID::ITS].isIndexSet()) {
        addITS(recoData, refs[GTrackID::ITS], leg, out.clITS, ic, pattRequests, matchedITS);
      }
      if (refs[GTrackID::TOF].isIndexSet()) {
        addTOF(recoData, refs[GTrackID::TOF], leg, out.clTOF, matchedTOF);
        if (mDebugOut) {
          writeDebugTOF(recoData.getTOFClusters()[refs[GTrackID::TOF].getIndex()], ic, leg, o2::dataformats::HitMatched);
        }
      }
      if (refs[GTrackID::TRD].isIndexSet()) {
        addTRD(recoData, refs[GTrackID::TRD], leg, out.trdTracklets, matchedTRD);
      }
    }
    CosmicTime cosmicTime;
    cosmicTime.tb = cosm.getTimeMUS().getTimeStamp() / mTPCTBinMUS;
    cosmicTime.errTB = cosm.getTimeMUS().getTimeStampError() / mTPCTBinMUS;
    cosmicTime.known = cosm.getTimeMUS().getTimeStampError() < mMaxAbsTimeErr;
    mDbgCosmic = ic;
    // TOF road first: a top/bottom hit pair matching the muon's flight gives the cosmic's time to ~ns (also for one-side legs on the same
    // side, whose brackets leave it open by tens of mus); the TPC corridor of the other side and the TRD / ITS roads then use that time
    if (mRoadDets[DetID::TOF]) {
      roadTOF(recoData, tpcLegs, cosmicTime, ic, out.clTOF, matchedTOF, out.timeTOFMUS, out.scoreTOFPair, out.scoreTOFReversed);
    }
    if (out.hasTOFTime() && tpcRefitter && legs[0].getSource() == GTrackID::TPC && legs[1].getSource() == GTrackID::TPC) {
      polish(tpcLegs, out.timeTOFMUS, *tpcRefitter, out);
    }
    CosmicTime roadTime = cosmicTime;
    if (out.hasTOFTime()) {
      roadTime.tb = out.timeTOFMUS / mTPCTBinMUS;
      roadTime.errTB = mTOFTimeErr / mTPCTBinMUS;
      roadTime.known = true;
    }
    mDbgTauC = roadTime.tb;
    std::unordered_set<uint32_t> taken;
    for (int leg = 0; leg < 2; leg++) { // attached clusters of both legs first, so that a road never takes the other leg's clusters
      if (tpcLegs[leg]) {
        addTPCAttached(recoData, *tpcLegs[leg], *tpcCl[leg], taken);
        mNClAttached += tpcCl[leg]->size();
      }
    }
    for (int leg = 0; leg < 2; leg++) {
      if (tpcLegs[leg]) {
        const size_t nAttached = tpcCl[leg]->size();
        mDbgLeg = leg;
        addTPCCorridor(recoData, *tpcLegs[leg], roadTime, *tpcCl[leg], taken);
        mNClCorridor += tpcCl[leg]->size() - nAttached;
      }
    }
    if (mRoadDets[DetID::TRD]) {
      roadTRD(recoData, tpcLegs, roadTime, ic, out.trdTracklets, matchedTRD);
    }
    if (mRoadDets[DetID::ITS]) {
      // the refitted cosmic's z moves with the time only if both legs are TPC-only on one side (an ITS part fixes it absolutely)
      int legsSide = 0;
      if (tpcLegs[0] && tpcLegs[1] && legs[0].getSource() == GTrackID::TPC && legs[1].getSource() == GTrackID::TPC && tpcSide(*tpcLegs[0]) == tpcSide(*tpcLegs[1])) {
        legsSide = tpcSide(*tpcLegs[0]);
      }
      roadITS(recoData, cosm, roadTime, legsSide, ic, out.clITS, pattRequests, matchedITS);
    }
  }
  if (!pattRequests.empty()) {
    fillITSPatterns(recoData, pattRequests, cosmicsOut);
  }
  flagDuplicates(cosmicsOut);
  if (mDebugOut) {
    for (size_t ic = 0; ic < cosmicsOut.size(); ic++) {
      writeDebug(cosmicsOut[ic], ic);
    }
  }
  mNCosmics += cosmicsOut.size();
  LOGP(info, "Collected clusters for {} cosmics in TF {}", cosmicsOut.size(), tfID.tfCounter);
  pc.outputs().snapshot(Output{"GLO", "COSMFULL", 0}, cosmicsOut);
  pc.outputs().snapshot(Output{"GLO", "COSMFULLTF", 0}, tfInfo);
  pc.outputs().snapshot(Output{"GLO", "COSMFULLTFID", 0}, tfID);
  mTimer.Stop();
}

void CosmicsClusterCollectorSpec::updateTimeDependentParams(ProcessingContext& pc)
{
  o2::base::GRPGeomHelper::instance().checkUpdates(pc);
  mCorrMap = &o2::gpu::TPCFastTransformPOD::get(pc.inputs().get<const char*>("corrMap"));
  mTPCTBinMUS = o2::tpc::ParameterElectronics::Instance().ZbinWidth;
  mBz = o2::base::Propagator::Instance()->getNominalBz();
  if (mRoadDets[DetID::ITS] && mITSChipCentres.empty()) {
    cacheITSChipCentres();
  }
}

void CosmicsClusterCollectorSpec::cacheITSChipCentres()
{
  auto geom = o2::its::GeometryTGeo::Instance();
  geom->fillMatrixCache(o2::math_utils::bit2Mask(o2::math_utils::TransformType::L2G));
  mITSChipCentres.resize(geom->getNumberOfChips());
  for (int chip = 0; chip < geom->getNumberOfChips(); chip++) {
    mITSChipCentres[chip] = geom->getMatrixL2G(chip) * o2::math_utils::Point3D<float>(0.f, 0.f, 0.f);
  }
}

void CosmicsClusterCollectorSpec::buildUsedMap(const RecoContainer& data, std::vector<std::pair<float, float>>& time0Windows)
{
  // merge the windows, then flag the clusters of the TPC tracks whose time0 falls into one of them
  std::sort(time0Windows.begin(), time0Windows.end());
  std::vector<std::pair<float, float>> merged;
  for (const auto& window : time0Windows) {
    if (!merged.empty() && window.first <= merged.back().second) {
      merged.back().second = std::max(merged.back().second, window.second);
    } else {
      merged.push_back(window);
    }
  }
  const auto& clusters = data.getTPCClusters();
  const auto tracks = data.getTPCTracks();
  const auto refs = data.getTPCTracksClusterRefs();
  mUsed.assign(clusters.nClustersTotal, false);
  for (const auto& trk : tracks) {
    const float time0 = trk.getTime0();
    const auto window = std::upper_bound(merged.begin(), merged.end(), time0, [](float t, const std::pair<float, float>& w) { return t < w.first; });
    if (window == merged.begin() || time0 > std::prev(window)->second) {
      continue;
    }
    for (int j = 0; j < trk.getNClusterReferences(); j++) {
      uint8_t sector = 0;
      uint8_t row = 0;
      uint32_t clIdx = 0;
      trk.getClusterReference(refs, j, sector, row, clIdx);
      mUsed[clusters.clusterOffset[sector][row] + clIdx] = true;
    }
  }
}

void CosmicsClusterCollectorSpec::addTPCAttached(const RecoContainer& data, const o2::tpc::TrackTPC& trk, std::vector<o2::dataformats::CosmicTPCCluster>& out, std::unordered_set<uint32_t>& taken) const
{
  const auto& clusters = data.getTPCClusters();
  const auto refs = data.getTPCTracksClusterRefs();
  for (int j = 0; j < trk.getNClusterReferences(); j++) {
    uint8_t sector = 0;
    uint8_t row = 0;
    uint32_t clIdx = 0;
    trk.getClusterReference(refs, j, sector, row, clIdx);
    if (!taken.insert(clusters.clusterOffset[sector][row] + clIdx).second) {
      continue;
    }
    auto& cl = out.emplace_back();
    cl.cl = clusters.clusters[sector][row][clIdx];
    cl.sector = sector;
    cl.row = row;
    cl.flags = o2::dataformats::CosmicTPCCluster::Attached | o2::dataformats::CosmicTPCCluster::Used;
  }
}

void CosmicsClusterCollectorSpec::LegBranch::init(const o2::track::TrackPar& inner, const o2::track::TrackPar& outer, float bz)
{
  if (std::abs(inner.getCurvature(bz)) < 1e-5f) { // radius > 1 km: straight line
    isLine = true;
    const auto point = inner.getXYZGlo();
    const float phi = inner.getPhi();
    dirX = std::cos(phi);
    dirY = std::sin(phi);
    const float proj = point.X() * dirX + point.Y() * dirY;
    pcaX = point.X() - proj * dirX;
    pcaY = point.Y() - proj * dirY;
  } else {
    o2::math_utils::CircleXYf_t circle;
    float sinAlpha = 0.f;
    float cosAlpha = 0.f;
    inner.getCircleParams(bz, circle, sinAlpha, cosAlpha);
    centerX = circle.xC;
    centerY = circle.yC;
    const float centerDist = std::sqrt(centerX * centerX + centerY * centerY);
    if (centerDist < 1e-3f) { // circle around the beam line: no closest approach
      sign = 0;
      return;
    }
    pcaX = centerX * (1.f - circle.rC / centerDist);
    pcaY = centerY * (1.f - circle.rC / centerDist);
  }
  // side of the leg from its end farther from the closest approach; a leg with ends on both sides (both > 10 cm away) spans the
  // closest approach and is accepted on both sides
  const auto pointIn = inner.getXYZGlo();
  const auto pointOut = outer.getXYZGlo();
  const float distIn2 = (pointIn.X() - pcaX) * (pointIn.X() - pcaX) + (pointIn.Y() - pcaY) * (pointIn.Y() - pcaY);
  const float distOut2 = (pointOut.X() - pcaX) * (pointOut.X() - pcaX) + (pointOut.Y() - pcaY) * (pointOut.Y() - pcaY);
  const int sideIn = side(pointIn.X(), pointIn.Y());
  const int sideOut = side(pointOut.X(), pointOut.Y());
  constexpr float MinDist2 = 10.f * 10.f;
  if (sideIn == sideOut) {
    sign = sideIn;
  } else if (std::min(distIn2, distOut2) < MinDist2) {
    sign = distIn2 > distOut2 ? sideIn : sideOut;
  } else {
    sign = 0;
  }
}

void CosmicsClusterCollectorSpec::addTPCCorridor(const RecoContainer& data, const o2::tpc::TrackTPC& trk, const CosmicTime& cosmicTime, std::vector<o2::dataformats::CosmicTPCCluster>& out, std::unordered_set<uint32_t>& taken) const
{
  constexpr int NSectorsA = TPCGeo::getNumberOfSectorsA();
  const float zLength = TPCGeo::getTPCzLength();
  const float vDrift = mCorrMap->getVDrift();
  const auto& clusters = data.getTPCClusters();
  const float time0Leg = trk.getTime0();
  const int legSide = tpcSide(trk);

  RoadFrame frames[2];
  int nFrames = 0;
  if (legSide == 0) { // CE-crossing leg: its time0 is absolute
    frames[nFrames++] = {time0Leg, 0.f, 0.f, 0, 2 * NSectorsA, 0};
  } else {
    frames[nFrames++] = {time0Leg, 0.f, 0.f, legSide > 0 ? 0 : NSectorsA, legSide > 0 ? NSectorsA : 2 * NSectorsA, 0};
    if (cosmicTime.known) { // z of a one-side leg moves by side * vD * (t - time0) when its clusters are transformed with the time t
      frames[nFrames++] = {cosmicTime.tb, legSide * (cosmicTime.tb - time0Leg) * vDrift, cosmicTime.errTB * vDrift,
                           legSide > 0 ? NSectorsA : 0, legSide > 0 ? 2 * NSectorsA : NSectorsA, o2::dataformats::CosmicTPCCluster::AbsTime};
    }
  }

  const o2::track::TrackPar refPar[2] = {trk, trk.getParamOut()}; // rows below rMid: inner parameter, above: outer
  LegBranch branch;
  branch.init(refPar[0], refPar[1], mBz);
  const auto pointIn = refPar[0].getXYZGlo();
  const auto pointOut = refPar[1].getXYZGlo();
  const float rIn = std::hypot(pointIn.X(), pointIn.Y());
  const float rOut = std::hypot(pointOut.X(), pointOut.Y());
  const float rMid = 0.5f * (rIn + rOut);

  for (int iFrame = 0; iFrame < nFrames; iFrame++) {
    const auto& frame = frames[iFrame];
    for (int sector = frame.sectorMin; sector < frame.sectorMax; sector++) {
      const float alpha = o2::math_utils::sector2Angle(sector % NSectorsA);
      const float sinAlpha = std::sin(alpha);
      const float cosAlpha = std::cos(alpha);
      const bool sideA = sector < NSectorsA;
      for (int iPar = 0; iPar < 2; iPar++) {
        auto par = refPar[iPar];
        par.setZ(par.getZ() + frame.dz);
        if (!par.rotateParam(alpha)) { // the leg points away from this sector frame: same helix, opposite direction
          par.invertParam();
          if (!par.rotateParam(alpha)) {
            continue;
          }
        }
        for (int row = 0; row < o2::tpc::constants::MAXGLOBALPADROW; row++) {
          const float xRow = TPCGeo::getRowInfoX(row);
          if ((xRow < rMid) != (iPar == 0)) {
            continue;
          }
          auto parRow = par;
          if (!parRow.propagateParamTo(xRow, mBz)) {
            continue;
          }
          // coarse acceptance at the nominal x of the row before the more expensive move to its real x
          constexpr float CoarseMargin = 10.f; // [cm] covers the shift of y and z between the nominal and the real x
          if (std::abs(parRow.getY()) > xRow * TanSector + mCorridor + CoarseMargin || outsideDriftVolume(sideA, parRow.getZ(), mCorridor + frame.zTolerance + CoarseMargin)) {
            continue;
          }
          // the corrected clusters of this row lie at the real x of the row, x + dx(y, z), not at its nominal x
          float xReal = xRow;
          const float zInside = sideA ? std::clamp(parRow.getZ(), 0.f, zLength) : std::clamp(parRow.getZ(), -zLength, 0.f);
          mCorrMap->InverseTransformYZtoX(sector, row, parRow.getY(), zInside, xReal);
          if (!parRow.propagateParamTo(xReal, mBz)) {
            continue;
          }
          const float y = parRow.getY();
          const float z = parRow.getZ();
          if (std::abs(y) > xRow * TanSector + mCorridor) {
            continue;
          }
          if (outsideDriftVolume(sideA, z, mCorridor + frame.zTolerance)) {
            continue;
          }
          if (!branch.accept(xRow * cosAlpha - y * sinAlpha, xRow * sinAlpha + y * cosAlpha)) {
            continue;
          }
          if (mDebugOut) {
            addDebugRoadPoint(sector, xReal, y, z, parRow.getSnp(), parRow.getTgl(), row, frame.vertexTime, frame.flag != 0);
          }
          searchRow(clusters, sector, row, y, z, parRow.getSnp(), parRow.getTgl(), frame.vertexTime, frame.zTolerance, frame.flag, 0.f, out, taken);
        }
      }
    }
    // parts of the leg nearly parallel to the pad rows, which the propagation row by row cannot reach (e.g. a closest approach to the
    // beam line inside the TPC: in the sector of the closest approach the track runs along one pad row); a leg whose inner end is not
    // inside its outer end (all its clusters along one pad row) is walked from the inner end alone
    const bool splitAtRMid = rOut > rIn;
    for (int iPar = 0; iPar < (splitAtRMid ? 2 : 1); iPar++) {
      auto par = refPar[iPar];
      par.setZ(par.getZ() + frame.dz);
      walkLowAngleRoad(clusters, par, iPar == 0, splitAtRMid ? rMid : std::numeric_limits<float>::max(), branch, frame, out, taken);
    }
  }
}

void CosmicsClusterCollectorSpec::walkLowAngleRoad(const o2::tpc::ClusterNativeAccess& clusters, const o2::track::TrackPar& start, bool innerPart, float rMid, const LegBranch& branch,
                                                   const RoadFrame& frame, std::vector<o2::dataformats::CosmicTPCCluster>& out, std::unordered_set<uint32_t>& taken) const
{
  constexpr int NSectorsA = TPCGeo::getNumberOfSectorsA();
  constexpr float CosSector = 0.98480775f; // cos(10 deg)
  constexpr float Step = 0.5f;             // [cm] path length per step, below the road width along the pad row
  constexpr float MaxPath = 600.f;         // [cm] protection; the walk normally ends at the TPC boundary or the closest approach
  constexpr float MaxEntryPath = 20.f;     // [cm] a start outside the leg's part (up to 10 cm past the closest approach, see LegBranch) is stepped into
  constexpr float MinSnp = 0.8f;           // below, the row crossings are well defined and found row by row
  constexpr float XMargin = 10.f;          // [cm] covers the shift between the nominal and the real x of a row (CoarseMargin of the row loop)
  constexpr int NRows = o2::tpc::constants::MAXGLOBALPADROW;
  const float zLength = TPCGeo::getTPCzLength();
  const float rLow = TPCGeo::getRowInfoX(0) - mCorridor;
  const float rHigh = TPCGeo::getRowInfoX(NRows - 1) / CosSector + mCorridor;
  std::array<float, NSectorsA> sinSector{};
  std::array<float, NSectorsA> cosSector{};
  for (int sector = 0; sector < NSectorsA; sector++) {
    const float alpha = o2::math_utils::sector2Angle(sector);
    sinSector[sector] = std::sin(alpha);
    cosSector[sector] = std::cos(alpha);
  }

  bool startInside = false; // the start point lies in the leg's part
  for (int direction = 0; direction < 2; direction++) {
    auto par = start;
    if (direction == 1) {
      par.invertParam();
    }
    bool entered = direction == 1 && startInside; // the walk has reached the leg's part
    // the start point is processed in the first direction only
    for (float path = direction == 0 ? 0.f : Step; path < MaxPath; path += Step) {
      // step along the helix in the frame of its direction (snp = 0), where the propagation is always defined
      if (path > 0.f && (!par.rotateParam(par.getPhi()) || !par.propagateParamTo(par.getX() + Step, mBz))) {
        break;
      }
      const auto pos = par.getXYZGlo();
      const float r = std::hypot(pos.X(), pos.Y());
      if (r > rHigh || std::abs(pos.Z()) > zLength + mCorridor + frame.zTolerance) {
        break;
      }
      // outside the leg's part: past its closest approach to the beam line, beyond rMid or inside the inner radius; the start of the walk
      // can lie there (a leg's end a few cm past the closest approach, an inner end inside the nominal inner radius)
      if (r < rLow || (r < rMid) != innerPart || !branch.accept(pos.X(), pos.Y())) {
        if (entered || path > MaxEntryPath) {
          break;
        }
        continue;
      }
      entered = true;
      startInside = startInside || path == 0.f;
      const float phiDir = par.getPhi();
      const int sectorPos = o2::math_utils::angle2Sector(std::atan2(pos.Y(), pos.X()));
      for (int dSector = -1; dSector <= 1; dSector++) { // the road can reach into the neighbouring sectors
        const int sectorInSide = (sectorPos + dSector + NSectorsA) % NSectorsA;
        for (int sector = sectorInSide; sector < 2 * NSectorsA; sector += NSectorsA) {
          if (sector < frame.sectorMin || sector >= frame.sectorMax) {
            continue;
          }
          const bool sideA = sector < NSectorsA;
          if (outsideDriftVolume(sideA, pos.Z(), mCorridor + frame.zTolerance)) {
            continue;
          }
          const float alpha = o2::math_utils::sector2Angle(sectorInSide);
          const float sinAlpha = sinSector[sectorInSide];
          const float cosAlpha = cosSector[sectorInSide];
          const float x = pos.X() * cosAlpha + pos.Y() * sinAlpha;
          const float y = -pos.X() * sinAlpha + pos.Y() * cosAlpha;
          float snp = std::sin(phiDir - alpha);
          float tgl = par.getTgl();
          if (std::cos(phiDir - alpha) < 0.f) { // the same line with its direction along +x of the sector frame, as in the row search
            snp = -snp;
            tgl = -tgl;
          }
          if (std::abs(snp) < MinSnp || std::abs(y) > x * TanSector + mCorridor) {
            continue;
          }
          const float zInside = sideA ? std::clamp(pos.Z(), 0.f, zLength) : std::clamp(pos.Z(), -zLength, 0.f);
          // rows whose real x lies within the road width of the point: the first by bisection on the nominal x, which grows with the row
          int rowFirst = 0;
          int rowEnd = NRows;
          while (rowFirst < rowEnd) {
            const int rowMid = (rowFirst + rowEnd) / 2;
            if (TPCGeo::getRowInfoX(rowMid) < x - mCorridor - XMargin) {
              rowFirst = rowMid + 1;
            } else {
              rowEnd = rowMid;
            }
          }
          for (int row = rowFirst; row < NRows && TPCGeo::getRowInfoX(row) <= x + mCorridor + XMargin; row++) {
            float xReal = TPCGeo::getRowInfoX(row);
            mCorrMap->InverseTransformYZtoX(sector, row, y, zInside, xReal);
            const float dxRow = xReal - x;
            if (std::abs(dxRow) > mCorridor) {
              continue;
            }
            if (mDebugOut) {
              addDebugRoadPoint(sector, x, y, pos.Z(), snp, tgl, row, frame.vertexTime, 2 + (frame.flag != 0));
            }
            searchRow(clusters, sector, row, y, pos.Z(), snp, tgl, frame.vertexTime, frame.zTolerance, frame.flag, dxRow, out, taken);
          }
        }
      }
    }
  }
}

void CosmicsClusterCollectorSpec::searchRow(const o2::tpc::ClusterNativeAccess& clusters, int sector, int row, float y, float z, float snp, float tgl, float vertexTime, float zTolerance, uint8_t flag,
                                            float dxRow, std::vector<o2::dataformats::CosmicTPCCluster>& out, std::unordered_set<uint32_t>& taken) const
{
  // (y, z) is the predicted point, dxRow the real x of the row minus the x of that point (0 when the point lies on the row)
  const float zLength = TPCGeo::getTPCzLength();
  const float vDrift = mCorrMap->getVDrift();
  const float t0 = mCorrMap->getT0();
  // nominal (measured) coordinates of the predicted real point; the correction is evaluated inside the drift volume
  const float zInside = sector < TPCGeo::getNumberOfSectorsA() ? std::clamp(z, 0.f, zLength) : std::clamp(z, -zLength, 0.f);
  float yNominal = 0.f;
  float zNominal = 0.f;
  mCorrMap->InverseTransformYZtoNominalYZ(sector, row, y, zInside, yNominal, zNominal);
  zNominal += z - zInside;
  float padPred = 0.f;
  float driftLengthPred = 0.f;
  TPCGeo::convLocalToPadDriftLength(sector, row, yNominal, zNominal, padPred, driftLengthPred);
  const float timePred = driftLengthPred / vDrift + t0 + vertexTime;
  // the road is a cylinder around the track; its section with the pad-row plane is an ellipse with half-axes W / cos(phi) in y and
  // W * sqrt(cos^2(phi) + tgl^2) / cos(phi) in z
  const float cosPhi = std::sqrt((1.f - snp) * (1.f + snp));
  const float cosPhiWindow = std::max(cosPhi, 0.1f); // limits the window for tracks nearly parallel to the pad row
  const float norm = 1.f / std::sqrt(1.f + tgl * tgl);
  const float dirX = cosPhi * norm; // track direction
  const float dirY = snp * norm;
  const float dirZ = tgl * norm;
  const float corridor2 = mCorridor * mCorridor;
  const float windowPad = mCorridor / (cosPhiWindow * TPCGeo::getRowInfoPadWidth(row));
  const float windowTime = (mCorridor * std::sqrt(cosPhiWindow * cosPhiWindow + tgl * tgl) / cosPhiWindow + zTolerance) / vDrift;
  const auto* rowClusters = clusters.clusters[sector][row];
  const uint32_t rowOffset = clusters.clusterOffset[sector][row];
  for (uint32_t k = 0; k < clusters.nClusters[sector][row]; k++) {
    const auto& c = rowClusters[k];
    const float time = c.getTime();
    const float pad = c.getPad();
    if (std::abs(time - timePred) > windowTime || std::abs(pad - padPred) > windowPad) {
      continue;
    }
    float yCluster = 0.f;
    float zCluster = 0.f;
    TPCGeo::convPadDriftLengthToLocal(sector, row, pad, (time - t0 - vertexTime) * vDrift, yCluster, zCluster);
    const float dy = yCluster - yNominal;
    float dz = zCluster - zNominal;
    if (zTolerance > 0.f) { // the vertex time of this frame is known within zTolerance / vD: allow that shift along z
      dz = std::abs(dz) > zTolerance ? dz - std::copysign(zTolerance, dz) : 0.f;
    }
    const float proj = dxRow * dirX + dy * dirY + dz * dirZ;
    if (dxRow * dxRow + dy * dy + dz * dz - proj * proj > corridor2) { // distance perpendicular to the track
      continue;
    }
    if (!taken.insert(rowOffset + k).second) {
      continue;
    }
    auto& cl = out.emplace_back();
    cl.cl = c;
    cl.sector = sector;
    cl.row = row;
    cl.flags = o2::dataformats::CosmicTPCCluster::Corridor | flag | (mUsed[rowOffset + k] ? o2::dataformats::CosmicTPCCluster::Used : 0);
  }
}

void CosmicsClusterCollectorSpec::addDebugRoadPoint(int sector, float x, float y, float z, float snp, float tgl, int row, float vertexTime, int frameCode) const
{
  // predicted point, z in the frame of the road and in the common frame of the cosmic's time
  const float alpha = o2::math_utils::sector2Angle(sector % TPCGeo::getNumberOfSectorsA());
  const float sinAlpha = std::sin(alpha);
  const float cosAlpha = std::cos(alpha);
  const bool sideA = sector < TPCGeo::getNumberOfSectorsA();
  auto& dbg = mDbgCosmics[mDbgCosmic];
  dbg.roadLeg.push_back(mDbgLeg);
  dbg.roadFrame.push_back(frameCode);
  dbg.roadSector.push_back(sector);
  dbg.roadRow.push_back(row);
  dbg.roadX.push_back(x);
  dbg.roadY.push_back(y);
  dbg.roadZ.push_back(z);
  dbg.roadZCos.push_back(z + (sideA ? 1.f : -1.f) * (mDbgTauC - vertexTime) * mCorrMap->getVDrift());
  dbg.roadGx.push_back(x * cosAlpha - y * sinAlpha);
  dbg.roadGy.push_back(x * sinAlpha + y * cosAlpha);
  dbg.roadSnp.push_back(snp);
  dbg.roadTgl.push_back(tgl);
}

void CosmicsClusterCollectorSpec::addITS(const RecoContainer& data, GTrackID gid, uint8_t leg, std::vector<o2::dataformats::CosmicITSCluster>& out, int icosm, std::vector<ITSPattRequest>& requests, std::unordered_set<int>& matched) const
{
  if (gid.getSource() != GTrackID::ITS) { // ITS-AB tracklets are not stored
    return;
  }
  if (data.getITSPerLayer()) {
    if (!mStaggeredWarned) {
      LOGP(warning, "Staggered ITS clusters are not supported, ITS clusters of cosmics are not stored");
      mStaggeredWarned = true;
    }
    return;
  }
  const auto& trk = data.getITSTrack(gid);
  const auto refs = data.getITSTracksClusterRefs();
  const auto clusters = data.getITSClusters();
  const auto rofs = data.getITSClustersROFRecords();
  for (int i = 0; i < trk.getNumberOfClusters(); i++) {
    const int idx = refs[trk.getFirstClusterEntry() + i];
    const auto& c = clusters[idx];
    auto& cl = out.emplace_back();
    cl.chipID = c.getSensorID();
    cl.row = c.getRow();
    cl.col = c.getCol();
    cl.pattID = c.getPatternID();
    cl.leg = leg;
    cl.flags = o2::dataformats::HitMatched;
    matched.insert(idx);
    if (c.getPatternID() == o2::itsmft::CompCluster::InvalidPatternID || (mITSDict && mITSDict->isGroup(c.getPatternID()))) {
      requests.push_back({icosm, int(out.size()) - 1, idx}); // the pattern is in the TF's pattern stream, copied by fillITSPatterns
    }
    auto rof = std::upper_bound(rofs.begin(), rofs.end(), idx, [](int v, const o2::itsmft::ROFRecord& r) { return v < r.getFirstEntry(); });
    if (rof != rofs.begin()) {
      cl.rofBC = std::prev(rof)->getBCData().differenceInBC(mTFStart);
    }
  }
}

void CosmicsClusterCollectorSpec::addTOF(const RecoContainer& data, GTrackID gid, uint8_t leg, std::vector<o2::dataformats::CosmicTOFCluster>& out, std::unordered_set<int>& matched) const
{
  const auto& c = data.getTOFClusters()[gid.getIndex()];
  auto& cl = out.emplace_back();
  cl.timeRaw = c.getTimeRaw();
  cl.tot = c.getTot();
  cl.channel = c.getMainContributingChannel();
  cl.leg = leg;
  cl.flags = o2::dataformats::HitMatched;
  matched.insert(gid.getIndex());
}

std::pair<float, float> CosmicsClusterCollectorSpec::timeWindowMUS(const CosmicTime& cosmicTime) const
{
  // a time fixed by z continuity is given with a 1 sigma error, otherwise the error is the half-width of the legs' time-bracket overlap
  const float timeMUS = cosmicTime.tb * mTPCTBinMUS;
  const float errMUS = cosmicTime.errTB * mTPCTBinMUS;
  const float halfWidth = cosmicTime.known ? 5.f * errMUS + 0.2f : errMUS;
  return {timeMUS - halfWidth, timeMUS + halfWidth};
}

bool CosmicsClusterCollectorSpec::predictOutward(const o2::tpc::TrackTPC& leg, const CosmicTime& cosmicTime, int sector, float x, float& y, float& z) const
{
  // outward continuation of a leg in the frame of a sector; z in the frame of the cosmic's time (a one-side TPC-only leg has z relative to
  // its time0)
  o2::track::TrackPar par = leg.getParamOut();
  if (!par.rotateParam(o2::math_utils::sector2Angle(sector % TPCGeo::getNumberOfSectorsA())) || !par.propagateParamTo(x, mBz)) {
    return false;
  }
  y = par.getY();
  z = par.getZ() + tpcSide(leg) * (cosmicTime.tb - leg.getTime0()) * mCorrMap->getVDrift();
  return true;
}

void CosmicsClusterCollectorSpec::roadTOF(const RecoContainer& data, const o2::tpc::TrackTPC* const* legs, const CosmicTime& cosmicTime, int icosm, std::vector<o2::dataformats::CosmicTOFCluster>& out, const std::unordered_set<int>& matched, float& timeTOFMUS, float& scoreTOFPair, float& scoreTOFReversed) const
{
  // TOF clusters along the legs' outward continuations (the road in PbPb also contains hits of collision tracks). Preferred: the top/bottom
  // pair whose time difference matches the muon's flight between them; its mean time fixes the cosmic's time, also when the legs' brackets
  // leave it open by tens of mus (one-side legs on the same side), and the z of one-side legs is shifted to it. Otherwise per leg the
  // cluster closest to its continuation. The same search in the impossible order (bottom hit first) only finds accidental pairs: its best
  // score is kept as QA of the flag's background.
  constexpr float MaxFlightMUS = 0.1f; // flight time of the muon between the TPC and the TOF, slow tails
  constexpr float CmPerNS = 29.9792458f;
  const auto window = timeWindowMUS(cosmicTime);
  const float zTimeTol = 0.5f * (window.second - window.first) / mTPCTBinMUS * mCorrMap->getVDrift(); // z uncertainty of one-side legs
  const float vDriftPerMUS = mCorrMap->getVDrift() / mTPCTBinMUS;
  const float cosmicTimeMUS = cosmicTime.tb * mTPCTBinMUS;
  int legSide[2] = {0, 0};
  float curvature = 0.f; // |1/R| of the muon [1/cm] for the flight path between the TOF hits
  int nLegs = 0;
  for (int leg = 0; leg < 2; leg++) {
    if (legs[leg]) {
      legSide[leg] = tpcSide(*legs[leg]);
      curvature += std::abs(legs[leg]->getCurvature(mBz));
      nLegs++;
    }
  }
  curvature = nLegs ? curvature / nLegs : 0.f;
  struct Candidate {
    int index;
    bool matched;  // hit of the leg's matched global track (already stored)
    double timeNS; // since the start of the TF
    float dy;
    float dzAtCosmicTime;
    float gx;
    float gy;
    float gz;
  };
  std::vector<Candidate> candidates[2];
  const auto clusters = data.getTOFClusters();
  int best[2] = {-1, -1};
  float bestScore[2] = {1.f, 1.f};
  for (int i = 0; i < (int)clusters.size(); i++) {
    const auto& c = clusters[i];
    const float timeMUS = c.getTime() * 1e-6f; // [ps] since the start of the TF
    if (timeMUS < window.first - MaxFlightMUS || timeMUS > window.second + MaxFlightMUS) {
      continue;
    }
    const bool isMatched = matched.count(i) > 0;
    const float alpha = o2::math_utils::sector2Angle(c.getSector());
    const float sinAlpha = std::sin(alpha);
    const float cosAlpha = std::cos(alpha);
    for (int leg = 0; leg < 2; leg++) {
      float y = 0.f;
      float z = 0.f;
      if (!legs[leg] || !predictOutward(*legs[leg], cosmicTime, c.getSector(), c.getX(), y, z)) {
        continue;
      }
      const float normY = (c.getY() - y) / mRoadTOF;
      const float normZ = (c.getZ() - z) / (mRoadTOF + zTimeTol);
      const float score = std::max(normY * normY, normZ * normZ);
      if (score >= 1.f) {
        continue;
      }
      if (!isMatched && score < bestScore[leg]) {
        bestScore[leg] = score;
        best[leg] = i;
      }
      candidates[leg].push_back(Candidate{i, isMatched, c.getTime() * 1e-3, c.getY() - y, c.getZ() - z, c.getX() * cosAlpha - c.getY() * sinAlpha,
                                          c.getX() * sinAlpha + c.getY() * cosAlpha, c.getZ()});
    }
  }
  float bestPairScore = -1.f;
  float bestReversedScore = -1.f;
  int bestPair[2] = {-1, -1};
  for (const auto& c0 : candidates[0]) {
    for (const auto& c1 : candidates[1]) {
      if (c0.index == c1.index) {
        continue;
      }
      // top / bottom by the hits' height: a muon goes down (for near-horizontal cosmics both orders are possible, but rare)
      const auto& top = c0.gy > c1.gy ? c0 : c1;
      const auto& bottom = c0.gy > c1.gy ? c1 : c0;
      // flight path along the helix: arc in the transverse plane from the chord, then the dip
      const float chordXY = std::hypot(top.gx - bottom.gx, top.gy - bottom.gy);
      const float halfAngleSin = 0.5f * curvature * chordXY;
      const float arcXY = halfAngleSin > 1e-4f && halfAngleSin < 1.f ? 2.f * std::asin(halfAngleSin) / curvature : chordXY;
      const float length = std::hypot(arcXY, top.gz - bottom.gz);
      const float timeDiff = float(top.timeNS - bottom.timeNS);
      const float flightDev = timeDiff + length / CmPerNS;       // the muon crosses the top TOF first
      const float reversedDev = timeDiff - length / CmPerNS;     // bottom hit first: impossible for a muon, only accidental pairs
      const bool reversed = std::abs(flightDev) > mTOFFlightTol; // the two cannot both pass (L/c ~ 25 ns >> tolerance)
      if (reversed && std::abs(reversedDev) > mTOFFlightTol) {
        continue;
      }
      const double pairTimeNS = 0.5 * (c0.timeNS + c1.timeNS);
      const float shiftMUS = float(pairTimeNS * 1e-3) - cosmicTimeMUS;
      const float dz0 = c0.dzAtCosmicTime - legSide[0] * shiftMUS * vDriftPerMUS;
      const float dz1 = c1.dzAtCosmicTime - legSide[1] * shiftMUS * vDriftPerMUS;
      if (std::abs(dz0) > mRoadTOF || std::abs(dz1) > mRoadTOF) {
        continue;
      }
      const float timeDev = reversed ? reversedDev : flightDev;
      const float score = (c0.dy * c0.dy + c1.dy * c1.dy + dz0 * dz0 + dz1 * dz1) / (mRoadTOF * mRoadTOF) + timeDev * timeDev / (mTOFFlightTol * mTOFFlightTol);
      if (reversed) {
        if (bestReversedScore < 0.f || score < bestReversedScore) {
          bestReversedScore = score;
        }
        continue;
      }
      if (bestPairScore < 0.f || score < bestPairScore) {
        bestPairScore = score;
        bestPair[0] = c0.index;
        bestPair[1] = c1.index;
        timeTOFMUS = float(pairTimeNS * 1e-3);
      }
    }
  }
  scoreTOFPair = bestPairScore;
  scoreTOFReversed = bestReversedScore;
  uint8_t flags = o2::dataformats::HitRoad;
  if (bestPairScore >= 0.f) {
    best[0] = bestPair[0];
    best[1] = bestPair[1];
    flags |= o2::dataformats::HitTOFFlight;
  } else if (best[0] >= 0 && best[0] == best[1]) { // one TOF cluster in the roads of both legs: keep it for the closer one
    best[bestScore[0] <= bestScore[1] ? 1 : 0] = -1;
  }
  for (int leg = 0; leg < 2; leg++) {
    if (best[leg] < 0) {
      continue;
    }
    const auto& c = clusters[best[leg]];
    if (matched.count(best[leg])) { // the leg's matched hit, already stored: flag it as part of the flight pair
      for (auto& cl : out) {
        if (cl.leg == leg && cl.channel == c.getMainContributingChannel() && cl.timeRaw == c.getTimeRaw()) {
          cl.flags |= o2::dataformats::HitTOFFlight;
        }
      }
      if (mDebugOut) {
        auto& dbg = mDbgCosmics[icosm];
        for (size_t j = 0; j < dbg.tofFlags.size(); j++) {
          if (dbg.tofLeg[j] == leg && dbg.tofChannel[j] == c.getMainContributingChannel() && dbg.tofTimeRaw[j] == c.getTimeRaw()) {
            dbg.tofFlags[j] |= o2::dataformats::HitTOFFlight;
          }
        }
      }
      continue;
    }
    auto& cl = out.emplace_back();
    cl.timeRaw = c.getTimeRaw();
    cl.tot = c.getTot();
    cl.channel = c.getMainContributingChannel();
    cl.leg = leg;
    cl.flags = flags;
    if (mDebugOut) {
      writeDebugTOF(c, icosm, leg, flags);
    }
  }
}

void CosmicsClusterCollectorSpec::polish(const o2::tpc::TrackTPC* const* legs, float timeMUS, o2::gpu::GPUO2InterfaceRefit& refitter, o2::dataformats::TrackCosmicsExtended& out) const
{
  // refit of a cosmic with TPC-only legs at its TOF time, as MatchCosmics::refitWinners: the bottom leg inward, then to the closest approach
  // to the beam line; the top leg inward, then to the same point; the two halves combined. Muon mass, energy loss along the muon's flight
  // (top to bottom): going inward along the bottom leg and up to the closest approach is against the flight (gain), the top leg inward
  // and down to the closest approach is with it (loss). With a precise time the one-side legs get their real z, hence also the right
  // material, which the matcher's time (the legs' bracket overlap for one-side legs on the same side) cannot give.
  constexpr int ELossGain = 1;
  constexpr int ELossLoss = -1;
  const auto& matchParams = o2::globaltracking::MatchCosmicsParams::Instance(); // propagation settings as in the matcher
  const auto prop = o2::base::Propagator::Instance();
  const float timeTB = timeMUS / mTPCTBinMUS;
  o2::track::TrackParCov bottom = legs[0]->getParamOut();
  bottom.setPID(o2::track::PID::Muon, true);
  bottom.resetCovariance();
  if (std::abs(mBz) <= 0.01f) {
    // B = 0: both legs carry the same conventional q/pt; as the matcher, flip the bottom one so that it is right after the inversion
    bottom.setQ2Pt(-o2::track::kMostProbablePt);
  }
  if (refitter.RefitTrackAsTrackParCov(bottom, legs[0]->getClusterRef(), timeTB, nullptr, false, false, ELossGain) < 0) {
    return;
  }
  bottom.invert();
  const o2::dataformats::VertexBase origin;
  if (!prop->propagateToDCABxByBz(origin, bottom, matchParams.maxStep, matchParams.matCorr, nullptr, nullptr, ELossGain)) {
    return;
  }
  o2::track::TrackParCov top = legs[1]->getParamOut();
  top.setPID(o2::track::PID::Muon, true);
  if (refitter.RefitTrackAsTrackParCov(top, legs[1]->getClusterRef(), timeTB, nullptr, false, true, ELossLoss) < 0) {
    return;
  }
  if (!top.rotate(bottom.getAlpha()) || !prop->PropagateToXBxByBz(top, bottom.getX(), matchParams.maxSnp, matchParams.maxStep, matchParams.matCorr, nullptr, ELossLoss)) {
    return;
  }
  o2::track::TrackParCov::MatrixDSym5 cov5;
  const float chi2Match = bottom.getPredictedChi2(top, cov5);
  if (!bottom.update(top, cov5)) {
    return;
  }
  out.polished = bottom;
  out.chi2MatchPolished = chi2Match;
}

void CosmicsClusterCollectorSpec::roadTRD(const RecoContainer& data, const o2::tpc::TrackTPC* const* legs, const CosmicTime& cosmicTime, int icosm, std::vector<o2::dataformats::CosmicTRDTracklet>& out, const std::unordered_set<int>& matched) const
{
  // per leg and layer the TRD tracklet closest to the leg's outward continuation, from triggers whose readout window can contain the cosmic
  constexpr float ReadoutWindowMUS = 3.f; // a cosmic leaves tracklets only if it passes within the readout window after a trigger
  constexpr float PadLength = 10.f;       // [cm] longest TRD pads: the tracklet z is the pad-row centre
  constexpr int NLayers = o2::trd::constants::NLAYER;
  const auto window = timeWindowMUS(cosmicTime);
  const float zTimeTol = 0.5f * (window.second - window.first) / mTPCTBinMUS * mCorrMap->getVDrift();
  const auto tracklets = data.getTRDTracklets();
  const auto calibrated = data.getTRDCalibratedTracklets();
  if (calibrated.size() != tracklets.size()) {
    if (!mTRDMismatchWarned) {
      LOGP(warning, "{} calibrated vs {} raw TRD tracklets: TRD road of cosmics skipped", calibrated.size(), tracklets.size());
      mTRDMismatchWarned = true;
    }
    return;
  }
  struct Candidate {
    int tracklet = -1;
    int trigBC = 0;
    float score = 1.f;
  };
  Candidate best[2][NLayers];
  for (const auto& trig : data.getTRDTriggerRecords()) {
    const int trigBC = trig.getBCData().differenceInBC(mTFStart);
    const float trigMUS = trigBC * o2::constants::lhc::LHCBunchSpacingMUS;
    if (trigMUS < window.first - ReadoutWindowMUS || trigMUS > window.second) {
      continue;
    }
    for (int it = trig.getFirstTracklet(); it < trig.getFirstTracklet() + trig.getNumberOfTracklets(); it++) {
      if (matched.count(it)) {
        continue;
      }
      const int detector = tracklets[it].getDetector();
      const int sector = detector / o2::trd::constants::NCHAMBERPERSEC;
      const int layer = detector % NLayers;
      const auto& point = calibrated[it];
      for (int leg = 0; leg < 2; leg++) {
        float y = 0.f;
        float z = 0.f;
        if (!legs[leg] || !predictOutward(*legs[leg], cosmicTime, sector, point.getX(), y, z)) {
          continue;
        }
        const float normY = (point.getY() - y) / mRoadTRD;
        const float normZ = (point.getZ() - z) / (mRoadTRD + PadLength + zTimeTol);
        const float score = std::max(normY * normY, normZ * normZ);
        if (score < best[leg][layer].score) {
          best[leg][layer] = {it, trigBC, score};
        }
      }
    }
  }
  for (int leg = 0; leg < 2; leg++) {
    for (int layer = 0; layer < NLayers; layer++) {
      const auto& cand = best[leg][layer];
      if (cand.tracklet < 0) {
        continue;
      }
      auto& tr = out.emplace_back();
      tr.word = tracklets[cand.tracklet].getTrackletWord();
      tr.trigBC = cand.trigBC;
      tr.layer = layer;
      tr.leg = leg;
      tr.flags = o2::dataformats::HitRoad;
      if (mDebugOut) {
        const auto& point = calibrated[cand.tracklet];
        const int sector = tracklets[cand.tracklet].getDetector() / o2::trd::constants::NCHAMBERPERSEC;
        float y = 0.f;
        float z = 0.f;
        predictOutward(*legs[leg], cosmicTime, sector, point.getX(), y, z); // succeeded in the search
        auto& dbg = mDbgCosmics[icosm];
        dbg.trdLeg.push_back(leg);
        dbg.trdLayer.push_back(layer);
        dbg.trdX.push_back(point.getX());
        dbg.trdY.push_back(point.getY());
        dbg.trdZ.push_back(point.getZ());
        dbg.trdDy.push_back(point.getY() - y);
        dbg.trdDz.push_back(point.getZ() - z);
        dbg.trdTrigMUS.push_back(cand.trigBC * float(o2::constants::lhc::LHCBunchSpacingMUS));
      }
    }
  }
}

void CosmicsClusterCollectorSpec::roadITS(const RecoContainer& data, const o2::dataformats::TrackCosmics& cosm, const CosmicTime& cosmicTime, int legsSide, int icosm, std::vector<o2::dataformats::CosmicITSCluster>& out,
                                          std::vector<ITSPattRequest>& requests, const std::unordered_set<int>& matched) const
{
  if (data.getITSPerLayer() || mITSChipCentres.empty()) {
    return;
  }
  // trajectory near the beam line from the combined cosmic, sampled every cm along its direction (closest approach at local x = 0)
  constexpr float MaxRadius = 45.f;    // [cm] outer ITS barrel + margin
  constexpr float ChipHalfDiag = 1.7f; // [cm] half diagonal of an ITS chip
  o2::track::TrackPar par = cosm;
  if (!par.rotateParam(par.getPhi())) {
    return;
  }
  std::vector<o2::math_utils::Point3D<float>> points;
  for (float x = -MaxRadius - 5.f; x <= MaxRadius + 5.f; x += 1.f) {
    bool ok = false;
    const auto point = par.getXYZGloAt(x, mBz, ok);
    if (ok) {
      points.push_back(point);
    }
  }
  if (points.size() < 2) {
    return;
  }
  // closest segment in the transverse plane: distance, z of the trajectory there
  auto closest = [&points](float x, float y, float& dist2, float& zTraj) {
    dist2 = 1e10f;
    for (size_t i = 0; i + 1 < points.size(); i++) {
      const float segX = points[i + 1].X() - points[i].X();
      const float segY = points[i + 1].Y() - points[i].Y();
      const float len2 = segX * segX + segY * segY;
      const float frac = len2 > 0.f ? std::clamp(((x - points[i].X()) * segX + (y - points[i].Y()) * segY) / len2, 0.f, 1.f) : 0.f;
      const float dx = x - (points[i].X() + frac * segX);
      const float dy = y - (points[i].Y() + frac * segY);
      if (dx * dx + dy * dy < dist2) {
        dist2 = dx * dx + dy * dy;
        zTraj = points[i].Z() + frac * (points[i + 1].Z() - points[i].Z());
      }
    }
  };
  float minR2 = 1e10f;
  float pcaY = 0.f;
  for (const auto& point : points) {
    const float r2 = point.X() * point.X() + point.Y() * point.Y();
    if (r2 < minR2) {
      minR2 = r2;
      pcaY = point.Y();
    }
  }
  if (minR2 > MaxRadius * MaxRadius) { // the cosmic does not cross the ITS
    return;
  }
  std::vector<bool> candidateChip(mITSChipCentres.size(), false);
  bool anyChip = false;
  const float chipCut2 = (mRoadITS + ChipHalfDiag) * (mRoadITS + ChipHalfDiag);
  for (size_t chip = 0; chip < mITSChipCentres.size(); chip++) {
    float dist2 = 0.f;
    float zTraj = 0.f;
    closest(mITSChipCentres[chip].X(), mITSChipCentres[chip].Y(), dist2, zTraj);
    if (dist2 < chipCut2) {
      candidateChip[chip] = true;
      anyChip = true;
    }
  }
  if (!anyChip) {
    return;
  }
  const auto window = timeWindowMUS(cosmicTime);
  const float zTimeTol = 0.5f * (window.second - window.first) / mTPCTBinMUS * mCorrMap->getVDrift(); // z of the cosmic is in the frame of its time
  // the refitted cosmic has the z of its TPC time; with legs on one TPC side and a road time from the TOF its z moves by side * vD * dt
  const float zShift = legsSide * (cosmicTime.tb - cosm.getTimeMUS().getTimeStamp() / mTPCTBinMUS) * mCorrMap->getVDrift();
  const float rofLengthMUS = o2::itsmft::DPLAlpideParam<DetID::ITS>::Instance().roFrameLengthInBC * o2::constants::lhc::LHCBunchSpacingMUS;
  // per half of the cosmic and layer the ITS clusters closest to the trajectory, best first: one in the outer barrel, several in the inner
  // barrel, where the road near the beam line is dense with collision clusters and the TPC prediction (~mm) does not single out the hit
  constexpr int NLayers = 7;
  constexpr int NLayersIB = 3;
  constexpr int MaxHitsIB = 5;
  struct Candidate {
    int index = -1;
    int rofBC = 0;
    float score = 1.f;
  };
  Candidate best[2][NLayers][MaxHitsIB];
  const auto clusters = data.getITSClusters();
  auto geom = o2::its::GeometryTGeo::Instance();
  for (const auto& rof : data.getITSClustersROFRecords()) {
    const int rofBC = rof.getBCData().differenceInBC(mTFStart);
    const float rofMUS = rofBC * o2::constants::lhc::LHCBunchSpacingMUS;
    if (rofMUS + rofLengthMUS < window.first || rofMUS > window.second) {
      continue;
    }
    for (int idx = rof.getFirstEntry(); idx < rof.getFirstEntry() + rof.getNEntries(); idx++) {
      const auto& c = clusters[idx];
      if (!candidateChip[c.getSensorID()] || matched.count(idx)) {
        continue;
      }
      o2::math_utils::Point3D<float> local;
      if (mITSDict && c.getPatternID() != o2::itsmft::CompCluster::InvalidPatternID && !mITSDict->isGroup(c.getPatternID())) {
        local = mITSDict->getClusterCoordinates<float>(c);
      } else { // anchor pixel: good enough for a cm road
        o2::itsmft::SegmentationAlpide::detectorToLocalUnchecked(c.getRow(), c.getCol(), local);
      }
      const auto global = geom->getMatrixL2G(c.getSensorID()) * local;
      float dist2 = 0.f;
      float zTraj = 0.f;
      closest(global.X(), global.Y(), dist2, zTraj);
      const float normZ = (global.Z() - zTraj - zShift) / (mRoadITS + zTimeTol);
      const float score = std::max(dist2 / (mRoadITS * mRoadITS), normZ * normZ);
      const int half = global.Y() < pcaY ? 0 : 1; // bottom / top half of the cosmic
      const int layer = geom->getLayer(c.getSensorID());
      if (layer < 0 || layer >= NLayers) {
        continue;
      }
      auto* candidates = best[half][layer];
      int slot = (layer < NLayersIB ? MaxHitsIB : 1) - 1;
      if (!(score < candidates[slot].score)) { // also rejects a NaN score
        continue;
      }
      for (; slot > 0 && score < candidates[slot - 1].score; slot--) {
        candidates[slot] = candidates[slot - 1];
      }
      candidates[slot] = {idx, rofBC, score};
    }
  }
  for (int half = 0; half < 2; half++) {
    for (int layer = 0; layer < NLayers; layer++) {
      for (const auto& candidate : best[half][layer]) {
        if (candidate.index < 0) {
          break;
        }
        const auto& c = clusters[candidate.index];
        auto& cl = out.emplace_back();
        cl.chipID = c.getSensorID();
        cl.row = c.getRow();
        cl.col = c.getCol();
        cl.pattID = c.getPatternID();
        cl.rofBC = candidate.rofBC;
        cl.leg = half;
        cl.flags = o2::dataformats::HitRoad;
        if (c.getPatternID() == o2::itsmft::CompCluster::InvalidPatternID || (mITSDict && mITSDict->isGroup(c.getPatternID()))) {
          requests.push_back({icosm, int(out.size()) - 1, candidate.index});
        }
      }
    }
  }
}

void CosmicsClusterCollectorSpec::fillITSPatterns(const RecoContainer& data, std::vector<ITSPattRequest>& requests, std::vector<o2::dataformats::TrackCosmicsExtended>& cosmics) const
{
  // the TF's pattern stream holds, in cluster order, the patterns of the clusters with an invalid or a group pattern ID
  if (!mITSDict) {
    if (!mNoDictWarned) {
      LOGP(warning, "No ITS cluster dictionary: patterns of ITS clusters of cosmics are not stored");
      mNoDictWarned = true;
    }
    return;
  }
  std::sort(requests.begin(), requests.end(), [](const ITSPattRequest& a, const ITSPattRequest& b) { return a.index < b.index; });
  const auto clusters = data.getITSClusters();
  auto pattIt = data.getITSClustersPatterns().begin();
  size_t ir = 0;
  for (int k = 0; k < (int)clusters.size() && ir < requests.size(); k++) {
    const auto pattID = clusters[k].getPatternID();
    if (pattID != o2::itsmft::CompCluster::InvalidPatternID && !mITSDict->isGroup(pattID)) {
      continue;
    }
    const auto start = pattIt;
    o2::itsmft::ClusterPattern::skipPattern(pattIt);
    for (; ir < requests.size() && requests[ir].index == k; ir++) {
      auto& cosm = cosmics[requests[ir].cosmic];
      cosm.clITS[requests[ir].cluster].pattEntry = cosm.itsPatterns.size();
      cosm.itsPatterns.insert(cosm.itsPatterns.end(), start, pattIt);
    }
  }
}

void CosmicsClusterCollectorSpec::addTRD(const RecoContainer& data, GTrackID gid, uint8_t leg, std::vector<o2::dataformats::CosmicTRDTracklet>& out, std::unordered_set<int>& matched) const
{
  const auto& trk = data.getTrack<o2::trd::TrackTRD>(gid); // TPC-TRD or ITS-TPC-TRD track
  const auto tracklets = data.getTRDTracklets();
  const auto trigs = data.getTRDTriggerRecords();
  for (int layer = 0; layer < 6; layer++) {
    const int it = trk.getTrackletIndex(layer);
    if (it < 0) {
      continue;
    }
    auto& tr = out.emplace_back();
    tr.word = tracklets[it].getTrackletWord();
    tr.layer = layer;
    tr.leg = leg;
    tr.flags = o2::dataformats::HitMatched;
    matched.insert(it);
    auto trig = std::upper_bound(trigs.begin(), trigs.end(), it, [](int v, const o2::trd::TriggerRecord& t) { return v < t.getFirstTracklet(); });
    if (trig != trigs.begin()) {
      tr.trigBC = std::prev(trig)->getBCData().differenceInBC(mTFStart);
    }
  }
}

void CosmicsClusterCollectorSpec::flagDuplicates(std::vector<o2::dataformats::TrackCosmicsExtended>& cosmics) const
{
  // the same muon can be matched twice, e.g. when a leg is split into two TPC tracks: the roads then collect largely the same clusters
  // (PbPb 567939: 10 of 47 cosmic pairs in a TF share 56-99 % of the clusters of the smaller one, all others none). The best one (TOF time,
  // then more attached clusters) stays, the others point to it.
  constexpr float MinSharedFraction = 0.3f;
  const size_t n = cosmics.size();
  if (n < 2) {
    return;
  }
  std::vector<std::vector<uint64_t>> keys(n); // sorted cluster addresses: sector, row and the packed time and pad of the raw cluster
  std::vector<int> nAttached(n, 0);
  for (size_t ic = 0; ic < n; ic++) {
    for (const auto* legClusters : {&cosmics[ic].clTPCBottom, &cosmics[ic].clTPCTop}) {
      for (const auto& c : *legClusters) {
        keys[ic].push_back((uint64_t(c.sector) << 56) | (uint64_t(c.row) << 48) | (uint64_t(c.cl.timeFlagsPacked) << 16) | c.cl.padPacked);
        nAttached[ic] += c.isAttached();
      }
    }
    std::sort(keys[ic].begin(), keys[ic].end());
  }
  std::vector<int> order(n);
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
    const bool timedA = cosmics[a].hasTOFTime();
    const bool timedB = cosmics[b].hasTOFTime();
    return timedA != timedB ? timedA : nAttached[a] > nAttached[b];
  });
  std::vector<int> kept;
  for (int ic : order) {
    for (int ik : kept) {
      std::vector<uint64_t> shared;
      std::set_intersection(keys[ic].begin(), keys[ic].end(), keys[ik].begin(), keys[ik].end(), std::back_inserter(shared));
      const size_t nMin = std::min(keys[ic].size(), keys[ik].size());
      if (nMin > 0 && shared.size() >= MinSharedFraction * nMin) {
        cosmics[ic].duplicateOf = ik;
        break;
      }
    }
    if (cosmics[ic].duplicateOf < 0) {
      kept.push_back(ic);
    }
  }
}

void CosmicsClusterCollectorSpec::writeDebug(const o2::dataformats::TrackCosmicsExtended& cosm, int icosm) const
{
  constexpr int NSectorsA = TPCGeo::getNumberOfSectorsA();
  // the time the other-side corridor used (and the common frame zCos): the TOF time if the TOF road found one, else the TPC time
  const float timeCosmic = (cosm.hasTOFTime() ? cosm.timeTOFMUS : cosm.cosmic.getTimeMUS().getTimeStamp()) / mTPCTBinMUS;
  const bool absTimeKnown = cosm.cosmic.getTimeMUS().getTimeStampError() < mMaxAbsTimeErr;
  int nAttached[2] = {0, 0};
  int nRoad[2] = {0, 0};
  int side[2] = {-2, -2};
  std::vector<int> clLeg;
  std::vector<int> clFlags;
  std::vector<int> clSector;
  std::vector<int> clRow;
  std::vector<float> clPad;
  std::vector<float> clTime;
  std::vector<float> clQMax;
  std::vector<float> clQTot;
  std::vector<float> clX;
  std::vector<float> clY;
  std::vector<float> clZ;
  std::vector<float> clZCos;
  std::vector<float> clGx;
  std::vector<float> clGy;
  const o2::tpc::TrackTPC* legs[2] = {&cosm.tpcBottom, &cosm.tpcTop};
  const std::vector<o2::dataformats::CosmicTPCCluster>* legClusters[2] = {&cosm.clTPCBottom, &cosm.clTPCTop};
  for (int leg = 0; leg < 2; leg++) {
    if (legs[leg]->getNClusters() == 0) { // no TPC part
      continue;
    }
    side[leg] = tpcSide(*legs[leg]);
    for (const auto& c : *legClusters[leg]) {
      // transform with the vertex time used by the road: the leg's time0, or the cosmic's time for clusters found on the other side
      const float vertexTime = c.isAbsTime() ? timeCosmic : legs[leg]->getTime0();
      float x = 0.f;
      float y = 0.f;
      float z = 0.f;
      mCorrMap->Transform(c.sector, c.row, c.cl.getPad(), c.cl.getTime(), x, y, z, vertexTime);
      float xCos = 0.f;
      float yCos = 0.f;
      float zCos = 0.f;
      mCorrMap->Transform(c.sector, c.row, c.cl.getPad(), c.cl.getTime(), xCos, yCos, zCos, timeCosmic);
      const float alpha = o2::math_utils::sector2Angle(c.sector % NSectorsA);
      const float sinAlpha = std::sin(alpha);
      const float cosAlpha = std::cos(alpha);
      clLeg.push_back(leg);
      clFlags.push_back(c.flags);
      clSector.push_back(c.sector);
      clRow.push_back(c.row);
      clPad.push_back(c.cl.getPad());
      clTime.push_back(c.cl.getTime());
      clQMax.push_back(c.cl.getQmax());
      clQTot.push_back(c.cl.getQtot());
      clX.push_back(x);
      clY.push_back(y);
      clZ.push_back(z);
      clZCos.push_back(zCos);
      clGx.push_back(x * cosAlpha - y * sinAlpha);
      clGy.push_back(x * sinAlpha + y * cosAlpha);
      (c.isAttached() ? nAttached : nRoad)[leg]++;
    }
  }
  std::vector<int> itsLeg;
  std::vector<int> itsFlags;
  std::vector<int> itsChip;
  std::vector<int> itsRow;
  std::vector<int> itsCol;
  std::vector<int> itsPattID;
  std::vector<int> itsHasPatt;
  std::vector<int> itsRofBC;
  std::vector<float> itsGx;
  std::vector<float> itsGy;
  std::vector<float> itsGz;
  std::vector<float> itsXLoc;
  std::vector<float> itsZLoc;
  for (const auto& c : cosm.clITS) { // local coordinates on the chip from the dictionary or the stored pattern
    const o2::itsmft::CompClusterExt compCluster(c.row, c.col, c.pattID, c.chipID);
    o2::math_utils::Point3D<float> local;
    if (c.pattEntry >= 0) {
      o2::itsmft::ClusterPattern pattern;
      auto pattIt = cosm.itsPatterns.begin() + c.pattEntry;
      pattern.acquirePattern(pattIt);
      local = o2::itsmft::TopologyDictionary::getClusterCoordinates<float>(compCluster, pattern, c.pattID != o2::itsmft::CompCluster::InvalidPatternID);
    } else if (mITSDict && c.pattID != o2::itsmft::CompCluster::InvalidPatternID) {
      local = mITSDict->getClusterCoordinates<float>(compCluster);
    } else { // pattern not available: pixel centre
      o2::itsmft::SegmentationAlpide::detectorToLocalUnchecked(c.row, c.col, local);
    }
    o2::math_utils::Point3D<float> global(0.f, 0.f, 0.f);
    if (!mITSChipCentres.empty()) { // geometry loaded for the ITS road
      global = o2::its::GeometryTGeo::Instance()->getMatrixL2G(c.chipID) * local;
    }
    itsLeg.push_back(c.leg);
    itsFlags.push_back(c.flags);
    itsChip.push_back(c.chipID);
    itsRow.push_back(c.row);
    itsCol.push_back(c.col);
    itsPattID.push_back(c.pattID);
    itsHasPatt.push_back(c.pattEntry >= 0);
    itsRofBC.push_back(c.rofBC);
    itsGx.push_back(global.X());
    itsGy.push_back(global.Y());
    itsGz.push_back(global.Z());
    itsXLoc.push_back(local.X());
    itsZLoc.push_back(local.Z());
  }
  const auto& dbg = mDbgCosmics[icosm];
  const auto& time = cosm.cosmic.getTimeMUS();
  (*mDebugOut) << "cosmics"
               << "tf=" << mDbgTF << "cosm=" << icosm << "t=" << time.getTimeStamp() << "tErr=" << time.getTimeStampError() << "absTime=" << int(absTimeKnown)
               << "chi2Match=" << cosm.cosmic.getChi2Match() << "chi2Refit=" << cosm.cosmic.getChi2Refit() << "q2pt=" << cosm.cosmic.getQ2Pt()
               << "tgl=" << cosm.cosmic.getTgl() << "side0=" << side[0] << "side1=" << side[1] << "nAtt0=" << nAttached[0] << "nAtt1=" << nAttached[1]
               << "nRoad0=" << nRoad[0] << "nRoad1=" << nRoad[1] << "nITS=" << int(cosm.clITS.size()) << "nTOF=" << int(cosm.clTOF.size())
               << "nTRD=" << int(cosm.trdTracklets.size()) << "tTOF=" << cosm.timeTOFMUS << "scoreTOF=" << cosm.scoreTOFPair << "scoreTOFRev=" << cosm.scoreTOFReversed << "dupOf=" << cosm.duplicateOf << "polChi2Match=" << cosm.chi2MatchPolished << "polQ2Pt=" << cosm.polished.getQ2Pt() << "polZ=" << cosm.polished.getZ()
               << "mcEvent=" << (cosm.label.isSet() ? cosm.label.getEventID() : -1) << "mcTrack=" << (cosm.label.isSet() ? cosm.label.getTrackID() : -1)
               << "clLeg=" << clLeg << "clFlags=" << clFlags << "clSector=" << clSector << "clRow=" << clRow << "clPad=" << clPad << "clTime=" << clTime
               << "clQMax=" << clQMax << "clQTot=" << clQTot << "clX=" << clX << "clY=" << clY << "clZ=" << clZ << "clZCos=" << clZCos << "clGx=" << clGx
               << "clGy=" << clGy << "roadLeg=" << dbg.roadLeg << "roadFrame=" << dbg.roadFrame << "roadSector=" << dbg.roadSector << "roadRow=" << dbg.roadRow
               << "roadX=" << dbg.roadX << "roadY=" << dbg.roadY << "roadZ=" << dbg.roadZ << "roadZCos=" << dbg.roadZCos << "roadGx=" << dbg.roadGx
               << "roadGy=" << dbg.roadGy << "roadSnp=" << dbg.roadSnp << "roadTgl=" << dbg.roadTgl << "itsLeg=" << itsLeg << "itsFlags=" << itsFlags
               << "itsChip=" << itsChip << "itsRow=" << itsRow << "itsCol=" << itsCol << "itsPattID=" << itsPattID << "itsHasPatt=" << itsHasPatt
               << "itsRofBC=" << itsRofBC << "itsGx=" << itsGx << "itsGy=" << itsGy << "itsGz=" << itsGz << "itsXLoc=" << itsXLoc << "itsZLoc=" << itsZLoc
               << "tofLeg=" << dbg.tofLeg << "tofChannel=" << dbg.tofChannel << "tofFlags=" << dbg.tofFlags << "tofTimeRaw=" << dbg.tofTimeRaw
               << "tofTime=" << dbg.tofTime << "tofTot=" << dbg.tofTot << "tofX=" << dbg.tofX << "tofY=" << dbg.tofY << "tofZ=" << dbg.tofZ << "tofGx=" << dbg.tofGx
               << "tofGy=" << dbg.tofGy << "trdLeg=" << dbg.trdLeg << "trdLayer=" << dbg.trdLayer << "trdX=" << dbg.trdX << "trdY=" << dbg.trdY << "trdZ=" << dbg.trdZ
               << "trdDy=" << dbg.trdDy << "trdDz=" << dbg.trdDz << "trdTrigMUS=" << dbg.trdTrigMUS << "\n";
}

void CosmicsClusterCollectorSpec::writeDebugTOF(const o2::tof::Cluster& c, int icosm, int leg, uint8_t flags) const
{
  // the TOF cluster position is in the frame of its sector
  const float alpha = o2::math_utils::sector2Angle(c.getSector());
  const float sinAlpha = std::sin(alpha);
  const float cosAlpha = std::cos(alpha);
  auto& dbg = mDbgCosmics[icosm];
  dbg.tofLeg.push_back(leg);
  dbg.tofChannel.push_back(c.getMainContributingChannel());
  dbg.tofFlags.push_back(flags);
  dbg.tofTimeRaw.push_back(c.getTimeRaw());
  dbg.tofTime.push_back(c.getTime());
  dbg.tofTot.push_back(c.getTot());
  dbg.tofX.push_back(c.getX());
  dbg.tofY.push_back(c.getY());
  dbg.tofZ.push_back(c.getZ());
  dbg.tofGx.push_back(c.getX() * cosAlpha - c.getY() * sinAlpha);
  dbg.tofGy.push_back(c.getX() * sinAlpha + c.getY() * cosAlpha);
}

void CosmicsClusterCollectorSpec::finaliseCCDB(ConcreteDataMatcher& matcher, void* obj)
{
  if (o2::base::GRPGeomHelper::instance().finaliseCCDB(matcher, obj)) {
    return;
  }
  if (matcher == ConcreteDataMatcher("ITS", "CLUSDICT", 0)) {
    mITSDict = (const o2::itsmft::TopologyDictionary*)obj;
    return;
  }
}

void CosmicsClusterCollectorSpec::endOfStream(EndOfStreamContext& ec)
{
  mDebugOut.reset();
  LOGP(info, "Cosmics cluster collector: {} cosmics, {} attached and {} road TPC clusters; Cpu: {:.3e} Real: {:.3e} s in {} slots",
       mNCosmics, mNClAttached, mNClCorridor, mTimer.CpuTime(), mTimer.RealTime(), mTimer.Counter() - 1);
}

DataProcessorSpec getCosmicsClusterCollectorSpec(GTrackID::mask_t src, bool useMC, bool itsStag, DetID::mask_t roadDets)
{
  if (itsStag && roadDets[DetID::ITS]) {
    LOGP(warning, "Staggered ITS clusters are not supported: ITS road of cosmics disabled");
    roadDets &= ~DetID::getMask(DetID::ITS);
  }
  auto dataRequest = std::make_shared<DataRequest>();
  dataRequest->setITSPerLayer(itsStag);
  dataRequest->requestTracks(src, false);
  dataRequest->requestTracks(getLegITSSources(src), false); // the ITS clusters of legs with an ITS part are read through their ITS track
  dataRequest->requestClusters(src, false);
  if (roadDets[DetID::ITS]) {
    dataRequest->requestITSClusters(false);
  }
  if (roadDets[DetID::TOF]) {
    dataRequest->requestTOFClusters(false);
  }
  if (roadDets[DetID::TRD]) {
    dataRequest->requestTRDTracklets(false);
  }
  dataRequest->requestCoscmicTracks(useMC);
  auto ggRequest = std::make_shared<o2::base::GRPGeomRequest>(false,                                                                                     // orbitResetTime
                                                              false,                                                                                     // GRPECS
                                                              false,                                                                                     // GRPLHCIF
                                                              true,                                                                                      // GRPMagField
                                                              roadDets[DetID::TOF],                                                                      // askMatLUT: polish refit of TOF-timed cosmics
                                                              roadDets[DetID::ITS] ? o2::base::GRPGeomRequest::Aligned : o2::base::GRPGeomRequest::None, // ITS road: chip positions
                                                              dataRequest->inputs,
                                                              true);
  dataRequest->inputs.emplace_back("corrMap", o2::header::gDataOriginTPC, "TPCCORRMAP", 0, Lifetime::Timeframe);

  std::vector<OutputSpec> outputs;
  outputs.emplace_back("GLO", "COSMFULL", 0, Lifetime::Timeframe);
  outputs.emplace_back("GLO", "COSMFULLTF", 0, Lifetime::Timeframe);
  outputs.emplace_back("GLO", "COSMFULLTFID", 0, Lifetime::Timeframe);

  return DataProcessorSpec{
    "cosmics-cluster-collector",
    dataRequest->inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<CosmicsClusterCollectorSpec>(dataRequest, ggRequest, useMC, roadDets)},
    Options{
      {"corridor-width", VariantType::Float, 1.f, {"half-width of the road around each leg [cm]"}},
      {"max-abs-time-err", VariantType::Float, 0.5f, {"max. time error of a cosmic [mus] to search the other TPC side of its one-side legs"}},
      {"max-cosmics-per-tf", VariantType::Int, 100, {"write at most this many cosmics per TF (with their clusters), the others are dropped"}},
      {"tof-road-width", VariantType::Float, 5.f, {"half-width of the road at the TOF [cm]"}},
      {"tof-flight-tolerance", VariantType::Float, 2.f, {"max. deviation of the top/bottom TOF time difference from the muon's flight time [ns]"}},
      {"tof-time-error", VariantType::Float, 0.1f, {"error of a cosmic's TOF time for the TPC other-side corridor and the TRD / ITS roads [mus]"}},
      {"trd-road-width", VariantType::Float, 5.f, {"half-width of the road at the TRD [cm] (in z plus the pad length)"}},
      {"its-road-width", VariantType::Float, 1.5f, {"half-width of the road in the ITS [cm]"}},
      {"debug-tree", VariantType::Bool, false, {"write cosmics_collector_debug.root with transformed clusters and road points (test runs)"}}}};
}

} // namespace o2::globaltracking
