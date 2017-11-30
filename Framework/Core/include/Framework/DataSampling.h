// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef FRAMEWORK_DATASAMPLER_H
#define FRAMEWORK_DATASAMPLER_H

#include <functional>
#include <string>
#include <vector>

#include "Framework/AlgorithmSpec.h"
#include "Framework/DataChunk.h"
#include "Framework/DataProcessorSpec.h"


namespace o2 {
namespace framework {

namespace DataSampling {


//wip: Function that takes DataProcessorSpec (or different specification) of QC Tasks, filtering function, % of data
//wip: and returns DataProcessorSpec vector of DataSamplers.
//wip: It may have different versions of DataSamplers topology, for test purposes.

std::vector<DataProcessorSpec> GenerateDataSamplers(const std::string& configurationSource, const std::vector<std::string>& taskNames);


struct DataSamplerConfiguration {
    double fractionOfDataToSample; //between 0 and 1
    std::function<bool(o2::framework::InputRecord&)> filteringFunction;
};

AlgorithmSpec::ProcessCallback initCallback(InitContext& ctx);
void processCallback(ProcessingContext& ctx, const DataSamplerConfiguration& conf);

} //namespace DataSampling


} //namespace framework
} //namespace o2

#endif //FRAMEWORK_DATASAMPLER_H
