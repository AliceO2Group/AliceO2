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
#include <iostream>

#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"
#include "TRDBase/FeeParam.h"
#include "TRDSimulation/TrapSimulator.h"
#include "DataFormatsTRD/TriggerRecord.h"
#include "DataFormatsTRD/LinkRecord.h"
#include "DataFormatsTRD/Tracklet64.h"
#include "DataFormatsTRD/RawData.h"
#include "TRDSimulation/TrapConfig.h"
#include "CCDB/BasicCCDBManager.h"

class Calibrations;

namespace o2
{
namespace trd
{

class TRDDPLTrapSimulatorTask : public o2::framework::Task
{

 public:
  TRDDPLTrapSimulatorTask() = default;

  void init(o2::framework::InitContext& ic) override;
  void run(o2::framework::ProcessingContext& pc) override;


 private:
  std::array<TrapSimulator, 128> mTrapSimulator; //the up to 128 trap simulators for a single chamber
  TrapConfig* mTrapConfig = nullptr;
  //  std::unique_ptr<TrapConfigHandler> mTrapConfigHandler;
  unsigned long mRunNumber = 297595; //run number to anchor simulation to.
  bool mDriveFromConfig{false};     // option to disable using the trapconfig to drive the simulation
  int mPrintTrackletOptions = 0;    // print the trap chips adc vs timebin to the screen, ascii art
  int mDrawTrackletOptions = 0;     //draw the tracklets 1 per file
  int mShowTrackletStats = 1;       // show some statistics for each run
  bool mPrintOutTrapConfig{false};
  bool mDebugRejectedTracklets{false};
  bool mEnableOnlineGainCorrection{false};
  bool mEnableTrapConfigDump{false};
  std::string mTrapConfigName;      // the name of the config to be used.
  std::string mTrapConfigBaseName = "TRD_test/TrapConfig/";
  std::unique_ptr<CalOnlineGainTables> mGainTable; //this will probably not be used in run3.
  std::string mOnlineGainTableName;
  std::unique_ptr<Calibrations> mCalib; // store the calibrations connection to CCDB. Used primarily for the gaintables in line above.


  //various timers so we can see how long things take
  std::chrono::duration<double> mTrapSimAccumulatedTime{0}; ///< full timer
  std::chrono::duration<double> mDigitLoopTime{0};          ///< full timer
  std::chrono::duration<double> mTrapLoopTime{0};           ///< full timer
  std::chrono::duration<double> moldelse{0};                ///< full timer
  std::chrono::duration<double> mTrackletTime{0};           ///< full timer
  std::chrono::duration<double> mSortingTime{0};            ///< full timer

  TrapConfig* getTrapConfig();
  void loadTrapConfig();
  void loadDefaultTrapConfig();
  void setOnlineGainTables();
};

o2::framework::DataProcessorSpec getTRDTrapSimulatorSpec();

} // end namespace trd
} // end namespace o2

#endif //O2_TRD_TRAPSIMULATORWORKFLOW_SRC_TRDTRAPSIMULATORSPEC_H_
