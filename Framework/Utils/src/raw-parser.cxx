// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "Framework/WorkflowSpec.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/DataSpecUtils.h"
#include "Framework/ControlService.h"
#include "Framework/Logger.h"
#include "Framework/ConfigParamSpec.h"
#include "DPLUtils/RawParser.h"
#include "Headers/DataHeader.h"
#include <vector>
#include <sstream>

using namespace o2::framework;

// we need to add workflow options before including Framework/runDataProcessing
void customize(std::vector<ConfigParamSpec>& workflowOptions)
{
  workflowOptions.push_back(
    ConfigParamSpec{
      "input-spec", VariantType::String, "A:FLP/RAWDATA", {"selection string input specs"}});
}

#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(ConfigContext const& config)
{
  WorkflowSpec workflow;
  workflow.emplace_back(DataProcessorSpec{
    "raw-parser",
    select(config.options().get<std::string>("input-spec").c_str()),
    Outputs{},
    AlgorithmSpec{[](InitContext& setup) {
        auto loglevel = setup.options().get<int>("log-level");
        return adaptStateless([loglevel](InputRecord& inputs, DataAllocator& outputs) {
            for (auto& input : inputs) {
              const auto* dh = DataRefUtils::getHeader<o2::header::DataHeader*>(input);
              if (loglevel > 0) {
                // InputSpec implements operator<< so we could print (input.spec) but that is the
                // matcher instead of the matched data
                LOG(INFO) << dh->dataOrigin.as<std::string>() << "/"
                          << dh->dataDescription.as<std::string>() << "/"
                          << dh->subSpecification << " payload size " << dh->payloadSize;
              }

              auto raw = inputs.get<gsl::span<char>>(input.spec->binding.c_str());

              try {
                o2::framework::RawParser parser(raw.data(), raw.size());

                std::stringstream rdhprintout;
                rdhprintout << parser;
                for (auto it = parser.begin(), end = parser.end(); it != end; ++it) {
                  rdhprintout << it << ": payload size " << it.size() << std::endl;
                }
                if (loglevel > 1) {
                  LOG(INFO) << rdhprintout.str();
                }
              } catch (const std::runtime_error& e) {
                LOG(ERROR) << "can not create raw parser form input data";
                o2::header::hexDump("payload", input.payload, dh->payloadSize, 64);
                LOG(ERROR) << e.what();
              }
            }
          }); }},
    Options{
      {"log-level", VariantType::Int, 1, {"Logging level [0-2]"}}}});
  return workflow;
}
