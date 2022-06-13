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

#ifndef O2_TRD_TRAPSIMULATORWORKFLOW_SRC_TRDTRAPSIMULATORSPEC_H_
#define O2_TRD_TRAPSIMULATORWORKFLOW_SRC_TRDTRAPSIMULATORSPEC_H_

#include <vector>
#include <array>
#include <string>

#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"
#include "TRDBase/Calibrations.h"
#include "TRDSimulation/TrapSimulator.h"
#include "DataFormatsTRD/Tracklet64.h"
#include "DataFormatsTRD/Constants.h"

namespace o2
{
namespace trd
{

class TrapConfig;

class TRDDPLTrapSimulatorTask : public o2::framework::Task
{

 public:
  TRDDPLTrapSimulatorTask(bool useMC) : mUseMC(useMC) {}

  void init(o2::framework::InitContext& ic) override;
  void run(o2::framework::ProcessingContext& pc) override;


 private:
  TrapConfig* mTrapConfig{nullptr};
  int mRunNumber{297595}; // run number to anchor simulation to.
  bool mUseFloatingPointForQ{false};
  bool mEnableOnlineGainCorrection{false};
  bool mUseMC{false}; // whether or not to use MC labels
  bool mEnableTrapConfigDump{false};
  bool mInitCcdbObjectsDone{false}; // flag whether one time download of CCDB objects has been done
  int mNumThreads{-1};              // number of threads used for parallel processing
  std::string mTrapConfigName;      // the name of the config to be used.
  std::string mOnlineGainTableName;
  std::unique_ptr<Calibrations> mCalib; // store the calibrations connection to CCDB. Used primarily for the gaintables in line above.

  void initTrapConfig(long timeStamp);
  void setOnlineGainTables();
  void processTRAPchips(int& nTracklets, std::vector<Tracklet64>& trackletsAccum, std::array<TrapSimulator, constants::NMCMHCMAX>& trapSimulators, std::vector<short>& digitCounts, std::vector<int>& digitIndices);
};

o2::framework::DataProcessorSpec getTRDTrapSimulatorSpec(bool useMC);

} // end namespace trd
} // end namespace o2

#endif //O2_TRD_TRAPSIMULATORWORKFLOW_SRC_TRDTRAPSIMULATORSPEC_H_
