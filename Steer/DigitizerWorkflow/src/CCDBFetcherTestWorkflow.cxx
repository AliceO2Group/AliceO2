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

#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/DataRefUtils.h"
#include "Framework/Lifetime.h"
#include "Framework/Task.h"
#include "Framework/Logger.h"
#include "Framework/CCDBParamSpec.h"
#include "DetectorsRaw/HBFUtils.h"
#include "Framework/CallbacksPolicy.h"
#include "Framework/WorkflowSpec.h"

using namespace o2::framework;

std::vector<std::string> objects{"GLO/Calib/MeanVertex"};
std::vector<unsigned long> times{1657152944347};

void ReadObjectList()
{
  std::ifstream file("ccdb-objects.dat");
  if (file.is_open()) {
    objects.clear();
    std::string line;
    while (std::getline(file, line)) {
      objects.push_back(line);
    }
    file.close();
  } else {
    std::cerr << "Failed to open the file... using default times" << std::endl;
  }
}

void ReadTimesList()
{
  // extract times
  std::ifstream file("ccdb-times.dat");
  if (file.is_open()) {
    std::string line;
    times.clear();
    while (std::getline(file, line)) {
      times.push_back(std::atol(line.c_str()));
    }
    file.close();
  } else {
    std::cerr << "Failed to open the times file ... using default times" << std::endl;
  }
}

// workflow options
void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  // put here options for the workflow
}

// customization to inject time information in the data source device
void customize(std::vector<o2::framework::CallbacksPolicy>& policies)
{
  // we customize the time information sent in DPL headers
  policies.push_back(o2::framework::CallbacksPolicy{
    [](o2::framework::DeviceSpec const& spec, o2::framework::ConfigContext const& context) -> bool {
      return true;
    },
    [](o2::framework::CallbackService& service, o2::framework::InitContext& context) {
      // simple linear enumeration from already updated HBFUtils (set via config key values)
      service.set<o2::framework::CallbackService::Id::NewTimeslice>(
        [](o2::header::DataHeader& dh, o2::framework::DataProcessingHeader& dph) {
          static int counter = 0;
          const auto& hbfu = o2::raw::HBFUtils::Instance();
          const auto offset = int64_t(hbfu.getFirstIRofTF({0, hbfu.orbitFirstSampled}).orbit);
          const auto increment = int64_t(hbfu.nHBFPerTF);
          const auto startTime = hbfu.startTime;
          const auto orbitFirst = hbfu.orbitFirst;
          dh.firstTForbit = offset + increment * dh.tfCounter;
          LOG(info) << "Setting firstTForbit to " << dh.firstTForbit;
          dh.runNumber = hbfu.runNumber;
          LOG(info) << "Setting runNumber to " << dh.runNumber;
          dph.creation = times[counter]; // ; we are taking the times from the timerecord
          counter++;
        });
    }} // end of struct
  );
}

#include "Framework/runDataProcessing.h"

struct Consumer : public o2::framework::Task {
  void run(ProcessingContext& ctx) final
  {
    auto& inputs = ctx.inputs();
    auto msg = inputs.get<unsigned long>("datainput");
    LOG(info) << "Doing compute with conditions for time " << msg;
  }
  void finaliseCCDB(ConcreteDataMatcher&, void*) final
  {
    LOG(error) << "Deserialization callback invoked";
  }
};

struct Producer : public o2::framework::Task {
  void run(ProcessingContext& ctx) final
  {
    static int counter = 0;
    LOG(info) << "Run function of Producer";
    ctx.outputs().snapshot(Output{"TST", "A1", 0}, times[counter]);
    counter++;
    if (counter == times.size()) {
      ctx.services().get<ControlService>().endOfStream();
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const& configcontext)
{
  std::vector<o2::framework::DataProcessorSpec> workflow;

  ReadObjectList();
  ReadTimesList();

  // putting Producer
  workflow.emplace_back(
    o2::framework::DataProcessorSpec{
      "Producer",
      {},
      {OutputSpec{"TST", "A1", 0, Lifetime::Timeframe}},
      adaptFromTask<Producer>(),
      Options{}});

  o2::framework::Inputs inputs;
  inputs.emplace_back(InputSpec{"datainput", "TST", "A1", 0, Lifetime::Timeframe});
  // now put all conditions
  int condcounter = 0;
  for (auto& obj : objects) {
    std::string name("cond");
    name += std::to_string(condcounter);
    condcounter++;
    char descr[9]; // Data description field
    if (name.length() >= 9) {
      std::string lastNineChars = name.substr(name.length() - 9);
      std::strcpy(descr, lastNineChars.c_str());
    } else {
      std::strcpy(descr, name.c_str());
    }
    inputs.emplace_back(InputSpec{name, "TST", descr, 0, Lifetime::Condition, ccdbParamSpec(obj)});
  }

  // putting Consumer
  workflow.emplace_back(
    o2::framework::DataProcessorSpec{
      "Consumer",
      inputs,
      {},
      adaptFromTask<Consumer>(),
      Options{}});

  return workflow;
}
