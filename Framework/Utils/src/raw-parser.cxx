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
    AlgorithmSpec{[](InitContext& setup) { return adaptStateless([](InputRecord& inputs, DataAllocator& outputs) {
                                             for (auto& input : inputs) {
                                               const auto* dh = DataRefUtils::getHeader<o2::header::DataHeader*>(input);
                                               LOG(INFO) << *(input.spec) << " payload size " << dh->payloadSize;

                                               // there is a bug in InpuRecord::get for vectors of simple types, not catched in
                                               // DataAllocator unit test
                                               //auto data = inputs.get<std::vector<char>>(input.spec->binding.c_str());
                                               //LOG(INFO) << "data size " << data.size();
                                               try {
                                                 o2::framework::RawParser parser(input.payload, dh->payloadSize);

                                                 LOG(INFO) << parser << std::endl;
                                                 for (auto it = parser.begin(), end = parser.end(); it != end; ++it) {
                                                   LOG(INFO) << it << ": block length " << it.length() << std::endl;
                                                 }
                                               } catch (const std::runtime_error& e) {
                                                 LOG(ERROR) << "can not create raw parser form input data";
                                                 o2::header::hexDump("payload", input.payload, dh->payloadSize, 64);
                                                 throw e;
                                               }
                                             }
                                           }); }}});
  return workflow;
}
