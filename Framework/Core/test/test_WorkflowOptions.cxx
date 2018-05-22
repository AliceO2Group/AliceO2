// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include "Framework/runDataProcessing.h"
#include "Framework/ControlService.h"

using namespace o2::framework;

#define ASSERT_ERROR(condition)                                                                      \
  if ((condition) == false) {                                                                        \
    LOG(ERROR) << R"(Test condition ")" #condition R"(" failed at )" << __FILE__ << ":" << __LINE__; \
  }

// This is how you can define your processing in a declarative way
WorkflowSpec defineDataProcessing(ConfigContext const&) {
  return WorkflowSpec{
    DataProcessorSpec{
      "producer",
      Inputs{},
      {
        OutputSpec{ "TST", "TEST" },
      },
      AlgorithmSpec{
        // define init callback
        [](InitContext& ic) {
          auto configstring = ic.options().get<std::string>("global-config");
          // read back the option from the command line, see CMakeLists.txt
          ASSERT_ERROR(configstring == "require-me");

          return [](ProcessingContext& ctx) {
            // there is nothing to do, simply stop the workflow
            ctx.services().get<ControlService>().readyToQuit(true);
          };
        },
      },
      {
        ConfigParamSpec{ "channel-config",
                         VariantType::String,
                         "name=foo,type=sub,method=connect,address=tcp://localhost:5450,rateLogging=1",
                         { "Out-of-band channel config" } },
        ConfigParamSpec{ "global-config", VariantType::String, { "A global config option for all processor specs" } },
      },
    },
    DataProcessorSpec{
      "consumer",
      Inputs{
        InputSpec{ "in", "TST", "TEST" },
      },
      {},
      AlgorithmSpec{
        // define an init callback
        [](InitContext& ic) {
          // read back the option from the command line, see CMakeLists.txt
          auto configstring = ic.options().get<std::string>("global-config");
          auto anotheroption = ic.options().get<std::string>("local-option");
          ASSERT_ERROR(configstring == "require-me");
          ASSERT_ERROR(anotheroption == "hello-aliceo2");

          return [](ProcessingContext& ctx) {
            // there is nothing to do, simply stop the workflow
            ctx.services().get<ControlService>().readyToQuit(true);
          };
        },
      },
      {
        ConfigParamSpec{ "global-config", VariantType::String, { "A global config option for all processor specs" } },
        ConfigParamSpec{ "local-option", VariantType::String, { "Option only valid for this processor spec" } },
      },
    }
  };
}
