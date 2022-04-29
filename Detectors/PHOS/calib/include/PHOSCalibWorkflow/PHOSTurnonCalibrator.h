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

#ifndef O2_CALIBRATION_PHOSTURNON_CALIBRATOR_H
#define O2_CALIBRATION_PHOSTURNON_CALIBRATOR_H

/// @file   PHOSTurnonCalibrator.h
/// @brief  Device to calculate PHOS turn-on curve and bad map

#include "Framework/Task.h"
#include "Framework/WorkflowSpec.h"
#include "DetectorsCalibration/TimeSlotCalibration.h"
#include "DataFormatsPHOS/Cluster.h"
#include "DataFormatsPHOS/Cell.h"
#include "DataFormatsPHOS/TriggerRecord.h"
#include "DataFormatsPHOS/TriggerMap.h"
#include "PHOSCalibWorkflow/TurnOnHistos.h"

using namespace o2::framework;

namespace o2
{
namespace phos
{

class PHOSTurnonSlot
{
 public:
  static constexpr short NCHANNELS = 3136; ///< Number of trigger channels

  PHOSTurnonSlot(bool useCCDB, std::string path);
  PHOSTurnonSlot(const PHOSTurnonSlot& other);

  ~PHOSTurnonSlot() = default;

  void print() const;
  void fill(const gsl::span<const Cell>& cells, const gsl::span<const TriggerRecord>& trs,
            const gsl::span<const Cluster>& clusters, const gsl::span<const TriggerRecord>& cluTR);
  void fill(const gsl::span<const Cluster>& /*cells*/){}; //not used
  void merge(const PHOSTurnonSlot* /*prev*/) {}           //not used
  void clear();

  TurnOnHistos& getCollectedHistos() { return *mTurnOnHistos; }

  void setRunStartTime(long tf) { mRunStartTime = tf; }

 private:
  void scanClusters(const gsl::span<const Cell>& cells, const TriggerRecord& celltr,
                    const gsl::span<const Cluster>& clusters, const TriggerRecord& clutr);

 private:
  bool mUseCCDB = false;
  long mRunStartTime = 0;                                 /// start time of the run (sec)
  std::string mCCDBPath{"http://alice-ccdb.cern.ch"};     ///< CCDB server path
  std::bitset<NCHANNELS> mFiredTiles;                     //! Container for bad trigger cells, 1 means bad sell
  std::bitset<NCHANNELS> mNoisyTiles;                     //! Container for bad trigger cells, 1 means bad sell
  std::unique_ptr<TurnOnHistos> mTurnOnHistos;            //! Collection of histos to fill

  ClassDefNV(PHOSTurnonSlot, 1);
};

//==========================================================================================
class PHOSTurnonCalibrator final : public o2::calibration::TimeSlotCalibration<o2::phos::Cluster, o2::phos::PHOSTurnonSlot>
{
  using Slot = o2::calibration::TimeSlot<o2::phos::PHOSTurnonSlot>;

 public:
  PHOSTurnonCalibrator() = default;

  bool hasEnoughData(const Slot& slot) const final { return true; } //no need to merge Slots
  void initOutput() final {}
  void finalizeSlot(Slot& slot) final;
  Slot& emplaceNewSlot(bool front, TFType tstart, TFType tend) final;
  bool process(uint64_t tf, const gsl::span<const Cell>& cells, const gsl::span<const TriggerRecord>& trs,
               const gsl::span<const Cluster>& clusters, const gsl::span<const TriggerRecord>& cluTR);

  TriggerMap& getCalibration() { return *mTriggerMap; }
  void endOfStream();

 private:
  bool calculateCalibrations();

 private:
  bool mUseCCDB = false;
  long mRunStartTime = 0;                                 /// start time of the run (sec)
  std::string mCCDBPath{"http://alice-ccdb.cern.ch"};     /// CCDB path to retrieve current CCDB objects for comparison
  std::unique_ptr<TurnOnHistos> mTurnOnHistos;            //! Collection of histos to fill
  std::unique_ptr<TriggerMap> mTriggerMap;

  ClassDefOverride(PHOSTurnonCalibrator, 1);
};

o2::framework::DataProcessorSpec getPHOSTunronCalibDeviceSpec(bool useCCDB, std::string path);
} // namespace phos
} // namespace o2

#endif
