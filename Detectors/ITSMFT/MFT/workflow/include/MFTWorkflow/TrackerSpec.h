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

/// @file   TrackerSpec.h

#ifndef O2_MFT_TRACKERDPL_H_
#define O2_MFT_TRACKERDPL_H_

#include "MFTTracking/Tracker.h"
#include "DetectorsBase/GRPGeomHelper.h"
#include "ITSMFTBase/DPLAlpideParam.h"

#include "Framework/DataProcessorSpec.h"
#include "MFTTracking/TrackCA.h"
#include "Framework/Task.h"
#include "DataFormatsParameters/GRPObject.h"
#include "DataFormatsITSMFT/TopologyDictionary.h"
#include "TStopwatch.h"

namespace o2
{
namespace mft
{
using o2::mft::TrackLTF;

class TrackerDPL : public o2::framework::Task
{

 public:
  TrackerDPL(std::shared_ptr<o2::base::GRPGeomRequest> gr, bool useMC, int nThreads = 1) : mGGCCDBRequest(gr), mUseMC(useMC), mNThreads(nThreads) {}
  ~TrackerDPL() override = default;
  void init(framework::InitContext& ic) final;
  void run(framework::ProcessingContext& pc) final;
  void endOfStream(framework::EndOfStreamContext& ec) final;
  void finaliseCCDB(framework::ConcreteDataMatcher& matcher, void* obj) final;

 private:
  void updateTimeDependentParams(framework::ProcessingContext& pc);
  ///< MFT readout mode
  bool mMFTTriggered = false; ///< MFT readout is triggered

  ///< set MFT ROFrame duration in microseconds
  void setMFTROFrameLengthMUS(float fums);
  ///< set MFT ROFrame duration in BC (continuous mode only)
  void setMFTROFrameLengthInBC(int nbc);
  int mMFTROFrameLengthInBC = 0;       ///< MFT RO frame in BC (for MFT cont. mode only)
  float mMFTROFrameLengthMUS = -1.;    ///< MFT RO frame in \mus
  float mMFTROFrameLengthMUSInv = -1.; ///< MFT RO frame in \mus inverse
  bool mUseMC = false;
  bool mFieldOn = true;
  int mNThreads = 4;
  std::shared_ptr<o2::base::GRPGeomRequest> mGGCCDBRequest;
  const o2::itsmft::TopologyDictionary* mDict = nullptr;
  std::unique_ptr<o2::parameters::GRPObject> mGRP = nullptr;
  std::vector<std::unique_ptr<o2::mft::Tracker<TrackLTF>>> mTrackerVec;
  std::vector<std::unique_ptr<o2::mft::Tracker<TrackLTFL>>> mTrackerLVec;

  enum TimerIDs { SWTot,
                  SWLoadData,
                  SWFindMFTTracks,
                  SWFitTracks,
                  SWComputeLabels,
                  NStopWatches };
  static constexpr std::string_view TimerName[] = {"TotalProcessing",
                                                   "LoadData",
                                                   "FindTracks",
                                                   "FitTracks",
                                                   "ComputeLabels"};
  TStopwatch mTimer[NStopWatches];

  ROFFilter createIRFrameFilter(gsl::span<const o2::dataformats::IRFrame> irframes)
  {
    return [this, irframes](const ROFRecord& rof) {
      InteractionRecord rofStart{rof.getBCData()};
      InteractionRecord rofEnd = rofStart + mMFTROFrameLengthInBC - 1;
      IRFrame ref(rofStart, rofEnd);
      for (const auto& ir : irframes) {
        if (ir.info > 0) {
          auto overlap = ref.getOverlap(ir);
          if (overlap.isValid()) {
            return true;
          }
        }
      }
      return false;
    };
  }
};

/// create a processor spec
o2::framework::DataProcessorSpec getTrackerSpec(bool useMC, int nThreads);

} // namespace mft
} // namespace o2

#endif /* O2_MFT_TRACKERDPL */
