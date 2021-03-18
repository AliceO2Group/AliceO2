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
  std::array<TrapSimulator, 128> mTrapSimulator; //the up to 128 trap simulators for a single chamber
  TrapConfig* mTrapConfig = nullptr;
  unsigned long mRunNumber = 297595; //run number to anchor simulation to.
  int mShowTrackletStats = 1;        // show some statistics for each run
  bool mEnableOnlineGainCorrection{false};
  bool mUseMC{false}; // whether or not to use MC labels
  bool mEnableTrapConfigDump{false};
  std::string mTrapConfigName;      // the name of the config to be used.
  std::string mOnlineGainTableName;
  std::unique_ptr<Calibrations> mCalib; // store the calibrations connection to CCDB. Used primarily for the gaintables in line above.
  std::chrono::duration<double> mTrapSimTime{0}; // timer for the actual processing in the TRAP chips

  TrapConfig* getTrapConfig();
  void loadTrapConfig();
  void loadDefaultTrapConfig();
  void setOnlineGainTables();
  void processTRAPchips(int currDetector, int& nTrackletsInTrigRec, std::vector<Tracklet64>& trapTrackletsAccum, o2::dataformats::MCTruthContainer<o2::MCCompLabel>& lblTracklets, const o2::dataformats::ConstMCTruthContainer<o2::MCCompLabel>* lblDigits);
};

o2::framework::DataProcessorSpec getTRDTrapSimulatorSpec(bool useMC);

} // end namespace trd
} // end namespace o2

#endif //O2_TRD_TRAPSIMULATORWORKFLOW_SRC_TRDTRAPSIMULATORSPEC_H_
