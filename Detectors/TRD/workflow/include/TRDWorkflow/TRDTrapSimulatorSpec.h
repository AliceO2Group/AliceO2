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

#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"
#include "TRDBase/FeeParam.h"
#include "TRDBase/Tracklet.h"
#include "TRDSimulation/TrapSimulator.h"

#include "TRDSimulation/TrapConfig.h"
#include "TRDSimulation/TrapConfigHandler.h"
#include "CCDB/BasicCCDBManager.h"
#include <iostream>

namespace o2
{
namespace trd
{

class TRDDPLTrapSimulatorTask : public framework::Task
{

 public:
  TRDDPLTrapSimulatorTask() = default;
  ~TRDDPLTrapSimulatorTask() override = default;
  void init(o2::framework::InitContext& ic) override;
  void run(o2::framework::ProcessingContext& pc) override;

 private:
  std::array<TrapSimulator, 8> mTrapSimulator; //the 8 trap simulators for a given padrow.
  FeeParam* mfeeparam;
  TrapConfig* mTrapConfig;
  std::unique_ptr<TRDGeometry> mGeo;
  //  std::unique_ptr<TrapConfigHandler> mTrapConfigHandler;
  bool mDriveFromConfig{false};     // option to disable using the trapconfig to drive the simulation
  int mPrintTrackletOptions = 0;    // print the tracklets to the screen, ascii art
  int mDrawTrackletOptions = 0;     //draw the tracklets 1 per file
  int mShowTrackletStats = 0;       //the the accumulated total tracklets found
  std::vector<Tracklet> mTracklets; // store of tracklets to then be inserted into a message.
  std::string mTrapConfigName;      // the name of the config to be used.
  std::string mTrapConfigBaseName = "TRD_test/TrapConfig/";
  TrapConfig* getTrapConfig();
  void loadTrapConfig();
};

o2::framework::DataProcessorSpec getTRDTrapSimulatorSpec();

} // end namespace trd
} // end namespace o2

#endif //O2_TRD_TRAPSIMULATORWORKFLOW_SRC_TRDTRAPSIMULATORSPEC_H_
