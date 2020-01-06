// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include "Framework/ConfigParamSpec.h"
#include "Framework/CompletionPolicy.h"
#include "Framework/DataRefUtils.h"
#include "Headers/DataHeader.h"
#include "Framework/DeviceSpec.h"
#include "Framework/Logger.h"

#include <chrono>
#include <thread>
#include <vector>

#include "Framework/runDataProcessing.h"

using namespace o2::framework;

// This allows defining a workflow where the subSpecification
// for the inputs and the outputs are left unspecified.
// The source will produce one message at the time with random
// subspecification and the receiver will match the any message
// with origin "TST" and description "A1" regardless of the subspec.
WorkflowSpec defineDataProcessing(ConfigContext const& specs)
{
  return WorkflowSpec{
    {"A",
     Inputs{},
     {OutputSpec{{"a1"}, {"TST", "A1"}}},
     AlgorithmSpec{adaptStateless(
       [](DataAllocator& outputs) {
         auto rn = rand() % 5;
         std::this_thread::sleep_for(std::chrono::seconds(rn));
         auto& aData = outputs.make<int>(OutputRef{"a1", static_cast<DataAllocator::SubSpecificationType>(rn)});
         LOGP(info, "A random subspec:{}", rn);
       })}},
    {"B",
     {InputSpec{"x", {"TST", "A1"}}},
     {},
     AlgorithmSpec{adaptStateless(
       [](InputRecord& inputs) {
         DataRef ref = inputs.getByPos(0);
         auto const* header = DataRefUtils::getHeader<o2::header::DataHeader*>(ref);
         LOGP(info, "A random subspec:{}", header->subSpecification);
       })}},
  };
}
