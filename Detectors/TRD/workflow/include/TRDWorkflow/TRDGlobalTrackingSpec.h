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

/// @file   TRDGlobalTrackingSpec.h

#ifndef O2_TRD_GLOBALTRACKING
#define O2_TRD_GLOBALTRACKING

#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"
#include "TStopwatch.h"
#include "TRDBase/GeometryFlat.h"
#include "GPUO2Interface.h"
#include "GPUTRDTracker.h"
#include "ReconstructionDataFormats/GlobalTrackID.h"
#include "DataFormatsGlobalTracking/RecoContainer.h"
#include "DataFormatsTRD/TrackTRD.h"
#include "DataFormatsTRD/TrackTriggerRecord.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/ConstMCTruthContainer.h"
#include <memory>
#include "DetectorsBase/GRPGeomHelper.h"

#include "GPUO2InterfaceRefit.h"
#include "TPCFastTransform.h"
#include "TRDBase/RecoParam.h"
#include "DataFormatsTPC/TrackTPC.h"
#include "DataFormatsITS/TrackITS.h"
#include "DataFormatsITSMFT/TrkClusRef.h"
#include "DataFormatsITSMFT/TopologyDictionary.h"

namespace o2
{
namespace trd
{

class TRDGlobalTracking : public o2::framework::Task
{
 public:
  TRDGlobalTracking(bool useMC, std::shared_ptr<o2::globaltracking::DataRequest> dataRequest, std::shared_ptr<o2::base::GRPGeomRequest> gr, o2::dataformats::GlobalTrackID::mask_t src, bool trigRecFilterActive, bool strict) : mUseMC(useMC), mDataRequest(dataRequest), mGGCCDBRequest(gr), mTrkMask(src), mTrigRecFilter(trigRecFilterActive), mStrict(strict) {}
  ~TRDGlobalTracking() override = default;
  void init(o2::framework::InitContext& ic) final;
  void fillMCTruthInfo(const TrackTRD& trk, o2::MCCompLabel lblSeed, std::vector<o2::MCCompLabel>& lblContainerTrd, std::vector<o2::MCCompLabel>& lblContainerMatch, const o2::dataformats::MCTruthContainer<o2::MCCompLabel>* trkltLabels) const;
  void fillTrackTriggerRecord(const std::vector<TrackTRD>& tracks, std::vector<TrackTriggerRecord>& trigRec, const gsl::span<const o2::trd::TriggerRecord>& trackletTrigRec) const;
  void run(o2::framework::ProcessingContext& pc) final;
  void finaliseCCDB(o2::framework::ConcreteDataMatcher& matcher, void* obj) final;
  bool refitITSTPCTRDTrack(TrackTRD& trk, float timeTRD, o2::globaltracking::RecoContainer* recoCont);
  bool refitTPCTRDTrack(TrackTRD& trk, float timeTRD, o2::globaltracking::RecoContainer* recoCont);
  bool refitTRDTrack(TrackTRD& trk, float& chi2, bool inwards);
  void endOfStream(o2::framework::EndOfStreamContext& ec) final;

 private:
  void updateTimeDependentParams(o2::framework::ProcessingContext& pc);

  o2::gpu::GPUTRDTracker* mTracker{nullptr};          ///< TRD tracking engine
  o2::gpu::GPUReconstruction* mRec{nullptr};          ///< GPU reconstruction pointer, handles memory for the tracker
  o2::gpu::GPUChainTracking* mChainTracking{nullptr}; ///< TRD tracker is run in the tracking chain
  std::unique_ptr<GeometryFlat> mFlatGeo{nullptr};    ///< flat TRD geometry
  bool mUseMC{false};                                 ///< MC flag
  float mTPCTBinMUS{.2f};                             ///< width of a TPC time bin in us
  float mTPCTBinMUSInv{1.f / mTPCTBinMUS};            ///< inverse width of a TPC time bin in 1/us
  float mTPCVdrift{2.58f};                            ///< TPC drift velocity (for shifting TPC tracks along Z)
  std::shared_ptr<o2::globaltracking::DataRequest> mDataRequest; ///< seeding input (TPC-only, ITS-TPC or both)
  std::shared_ptr<o2::base::GRPGeomRequest> mGGCCDBRequest;
  o2::dataformats::GlobalTrackID::mask_t mTrkMask;               ///< seeding track sources (TPC, ITS-TPC)
  bool mTrigRecFilter{false};                                    ///< if true, TRD trigger records without matching ITS IR are filtered out
  bool mStrict{false};                                           ///< preliminary matching in strict mode
  TStopwatch mTimer;
  // temporary members -> should go into processor (GPUTRDTracker or additional refit processor?)
  std::unique_ptr<o2::gpu::GPUO2InterfaceRefit> mTPCRefitter;         ///< TPC refitter used for TPC tracks refit during the reconstruction
  const o2::tpc::ClusterNativeAccess* mTPCClusterIdxStruct = nullptr; ///< struct holding the TPC cluster indices
  std::unique_ptr<o2::gpu::TPCFastTransform> mTPCTransform;           ///< TPC cluster transformation
  RecoParam mRecoParam;                                               ///< parameters required for TRD reconstruction
  gsl::span<const Tracklet64> mTrackletsRaw;                          ///< array of raw tracklets needed for TRD refit
  gsl::span<const CalibratedTracklet> mTrackletsCalib;                ///< array of calibrated tracklets needed for TRD refit
  gsl::span<const o2::tpc::TrackTPC> mTPCTracksArray;                 ///< input TPC tracks used for refit
  gsl::span<const o2::its::TrackITS> mITSTracksArray;                 ///< input ITS tracks used for refit
  gsl::span<const o2::itsmft::TrkClusRef> mITSABRefsArray;            ///< input ITS-TPC Afterburner ITS tracklets references
  gsl::span<const int> mITSTrackClusIdx;                              ///< input ITS track cluster indices span
  gsl::span<const int> mITSABTrackClusIdx;                            ///< input ITSAB track cluster indices span
  std::vector<o2::BaseCluster<float>> mITSClustersArray;              ///< ITS clusters created in run() method from compact clusters
  const o2::itsmft::TopologyDictionary* mITSDict = nullptr;           ///< cluster patterns dictionary
};

/// create a processor spec
framework::DataProcessorSpec getTRDGlobalTrackingSpec(bool useMC, o2::dataformats::GlobalTrackID::mask_t src, bool trigRecFilterActive, bool strict = false);

} // namespace trd
} // namespace o2

#endif /* O2_TRD_TRACKLETREADER */
