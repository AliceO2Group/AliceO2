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

#include "../Framework/Core/src/ArrowSupport.h"
#include "Framework/AnalysisTask.h"
#include "Monitoring/Monitoring.h"
#include "Framework/CommonDataProcessors.h"
#include "SimulationDataFormat/MCTrack.h"
#include "Steer/MCKinematicsReader.h"

#include "Framework/runDataProcessing.h"

using namespace o2::framework;
using namespace o2::steer;

struct O2simKinePublisher {
  Configurable<std::string> kineFileName{"kineFileName", "o2sim", "name of the _Kine.root file (without '_Kine.root')"};
  Configurable<int> aggregate{"aggregate-timeframe", 300, "Number of events to put in a timeframe"};

  int nEvents = 0;
  int eventCounter = 0;
  int tfCounter = 0;
  std::shared_ptr<MCKinematicsReader> mcKinReader = std::make_shared<MCKinematicsReader>();

  void init(o2::framework::InitContext& /*ic*/)
  {
    if (mcKinReader->initFromKinematics((std::string)kineFileName)) {
      nEvents = mcKinReader->getNEvents(0);
    } else {
      LOGP(fatal, "Cannot open kine file {}", (std::string)kineFileName);
    }
  }

  void run(o2::framework::ProcessingContext& pc)
  {
    for (auto i = 0; i < std::min((int)aggregate, nEvents - eventCounter); ++i) {
      auto mcevent = mcKinReader->getMCEventHeader(0, eventCounter);
      auto mctracks = mcKinReader->getTracks(0, eventCounter);
      pc.outputs().snapshot(Output{"MC", "MCHEADER", 0, Lifetime::Timeframe}, mcevent);
      pc.outputs().snapshot(Output{"MC", "MCTRACKS", 0, Lifetime::Timeframe}, mctracks);
      ++eventCounter;
    }
    // report number of TFs injected for the rate limiter to work
    pc.services().get<o2::monitoring::Monitoring>().send(o2::monitoring::Metric{(uint64_t)tfCounter, "df-sent"}.addTag(o2::monitoring::tags::Key::Subsystem, o2::monitoring::tags::Value::DPL));
    ++tfCounter;
    if (eventCounter >= nEvents) {
      pc.services().get<ControlService>().endOfStream();
      pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  auto spec = adaptAnalysisTask<O2simKinePublisher>(cfgc);
  spec.outputs.emplace_back("MC", "MCHEADER", 0, Lifetime::Timeframe);
  spec.outputs.emplace_back("MC", "MCTRACKS", 0, Lifetime::Timeframe);
  spec.requiredServices.push_back(o2::framework::ArrowSupport::arrowBackendSpec());
  spec.algorithm = CommonDataProcessors::wrapWithRateLimiting(spec.algorithm);
  return {spec};
}
