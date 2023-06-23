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
#include "Framework/CallbacksPolicy.h"
#include "Framework/WorkflowSpec.h"
#include <iostream>

using namespace o2::framework;

std::vector<std::string> objects{"GLO/Calib/MeanVertex"};
std::vector<unsigned long> times{1657152944347};
int gRunNumber = 30000;
int gOrbitsPerTF = 32;

void ReadObjectList(std::string const& filename)
{
  std::ifstream file(filename.c_str());
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

void ReadTimesList(std::string const& filename)
{
  // extract times
  std::ifstream file(filename.c_str());
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
  workflowOptions.push_back(ConfigParamSpec{"run-number", o2::framework::VariantType::Int, 30000, {"run number"}});
  workflowOptions.push_back(ConfigParamSpec{"tforbits", o2::framework::VariantType::Int, 32, {"orbits per tf"}});
  workflowOptions.push_back(ConfigParamSpec{"objects", o2::framework::VariantType::String, "ccdb-object.dat", {"file with rows of object path to fetch"}});
  workflowOptions.push_back(ConfigParamSpec{"times", o2::framework::VariantType::String, "ccdb-times.dat", {"file with times to use"}});
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
          const auto offset = int64_t(0);
          const auto increment = int64_t(gOrbitsPerTF);
          dh.firstTForbit = offset + increment * dh.tfCounter;
          LOG(info) << "Setting firstTForbit to " << dh.firstTForbit;
          dh.runNumber = gRunNumber; // hbfu.runNumber;
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
    static int counter = 1;
    auto& inputs = ctx.inputs();
    auto msg = inputs.get<unsigned long>("datainput");
    LOG(info) << "Doing compute with conditions for time " << msg << " (" << counter++ << "/" << times.size() << ")";
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

  ReadObjectList(configcontext.options().get<std::string>("objects"));
  ReadTimesList(configcontext.options().get<std::string>("times"));
  gRunNumber = configcontext.options().get<int>("run-number");
  gOrbitsPerTF = configcontext.options().get<int>("tforbits");

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
