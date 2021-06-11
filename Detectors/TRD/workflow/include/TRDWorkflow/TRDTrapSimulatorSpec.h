// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
#include "TRDSimulation/TrapSimulator.h"
#include "TRDSimulation/TrapConfig.h"
#include "DataFormatsTRD/Tracklet64.h"
#include "DataFormatsTRD/Constants.h"
#include <SimulationDataFormat/MCCompLabel.h>
#include <SimulationDataFormat/ConstMCTruthContainer.h>

class Calibrations;

namespace o2
{
namespace trd
{

class TRDDPLTrapSimulatorTask : public o2::framework::Task
{

 public:
  TRDDPLTrapSimulatorTask(bool useMC) : mUseMC(useMC) {}

  void init(o2::framework::InitContext& ic) override;
  void run(o2::framework::ProcessingContext& pc) override;


 private:
  TrapConfig* mTrapConfig = nullptr;
  unsigned long mRunNumber = 297595; //run number to anchor simulation to.
  bool mEnableOnlineGainCorrection{false};
  bool mUseMC{false}; // whether or not to use MC labels
  bool mEnableTrapConfigDump{false};
  int mNumThreads{-1};              // number of threads used for parallel processing
  std::string mTrapConfigName;      // the name of the config to be used.
  std::string mOnlineGainTableName;
  std::unique_ptr<Calibrations> mCalib; // store the calibrations connection to CCDB. Used primarily for the gaintables in line above.

  TrapConfig* getTrapConfig();
  void loadTrapConfig();
  void loadDefaultTrapConfig();
  void setOnlineGainTables();
  void processTRAPchips(int& nTracklets, std::vector<Tracklet64>& trackletsAccum, std::array<TrapSimulator, constants::NMCMHCMAX>& trapSimulators, std::vector<short>& digitCounts, std::vector<int>& digitIndices);
};

o2::framework::DataProcessorSpec getTRDTrapSimulatorSpec(bool useMC);

} // end namespace trd
} // end namespace o2

#endif //O2_TRD_TRAPSIMULATORWORKFLOW_SRC_TRDTRAPSIMULATORSPEC_H_
