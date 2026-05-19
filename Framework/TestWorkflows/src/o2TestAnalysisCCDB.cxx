// Copyright 2019-2025 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
///
/// \brief FullTracks is a join of Tracks, TracksCov, and TracksExtra.
/// \author
/// \since

#include "Framework/runDataProcessing.h"
#include "Framework/AnalysisTask.h"
#include "Framework/AnalysisDataModel.h"
#include "DataFormatsTOF/CalibLHCphaseTOF.h"
#include <iostream>

#include <TH2F.h>

using namespace o2;
using namespace o2::framework;
using namespace o2::framework::expressions;

namespace o2::aod
{
namespace tofcalib
{
DECLARE_SOA_CCDB_COLUMN(LHCphase, lhcPhase, o2::dataformats::CalibLHCphaseTOF, "TOF/Calib/LHCphase"); //!
} // namespace tofcalib

DECLARE_SOA_TIMESTAMPED_TABLE(TOFCalibrationObjects, aod::Timestamps, o2::aod::timestamp::Timestamp, 1, "TOFCALIB", //!
                              tofcalib::LHCphase);
} // namespace o2::aod

struct DummyTimestampsTable {
  Produces<aod::Timestamps> timestamps; /// Table with SOR timestamps produced by the task
  Service<o2::framework::ControlService> control;

  void process(Enumeration<0, 1>& e)
  {
    timestamps(1747442464000); // c2b3d801393540b7bddb949d600b199f, ecacb915-3d70-11f0-ac6f-808de0f5250c
    timestamps(1747442764000); // 0262dbd9d50aa79c3d4dcd5ec3ca67c3, ed5471c5-3d70-11f0-b0a3-808de0f524ee
    control->readyToQuit(QuitRequest::Me);
    control->endOfStream();
    std::cout << "Executed " << std::endl;
  }
};

struct SimpleCCDBConsumer {
  ConfigurableCCDBPath<o2::aod::tofcalib::LHCphase> lhcPhasePath;

  void process(o2::aod::TOFCalibrationObjects const& ccdbObjectsForAllTimestamps)
  {
    LOGP(info, "LHCphase CCDB path configurable value: {}", lhcPhasePath.value);
    LOGP(info, "Looking at all the LHCphases associated to the timestamps");
    for (auto& object : ccdbObjectsForAllTimestamps) {
      std::cout << object.lhcPhase().getStartValidity() << " " << object.lhcPhase().getEndValidity() << std::endl;
    }
  }
};

struct AnotherCCDBConsumer {
  ConfigurableCCDBPath<o2::aod::tofcalib::LHCphase> lhcPhasePath;

  void process(o2::aod::TOFCalibrationObjects const& ccdbObjectsForAllTimestamps)
  {
    LOGP(info, "AnotherCCDBConsumer LHCphase CCDB path configurable value: {}", lhcPhasePath.value);
    for (auto& object : ccdbObjectsForAllTimestamps) {
      std::cout << object.lhcPhase().getStartValidity() << " " << object.lhcPhase().getEndValidity() << std::endl;
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  return WorkflowSpec{
    adaptAnalysisTask<DummyTimestampsTable>(cfgc),
    adaptAnalysisTask<SimpleCCDBConsumer>(cfgc, TaskName{"simple-ccdb-consumer"}),
    adaptAnalysisTask<AnotherCCDBConsumer>(cfgc, TaskName{"another-ccdb-consumer"}),
  };
}
