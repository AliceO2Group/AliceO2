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
#include <random>

#include "Framework/AlgorithmSpec.h"
#include "Framework/DataChunk.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/WorkflowSpec.h"


namespace o2 {
namespace framework {

namespace DataSampling {


//wip: Function that takes DataProcessorSpec (or different specification) of QC Tasks, filtering function, % of data
//wip: and returns DataProcessorSpec vector of DataSamplers.
//wip: It may have different versions of DataSamplers topology, for test purposes.

void GenerateDataSamplers(WorkflowSpec& workflow, const std::string& configurationSource, const std::vector<std::string>& taskNames);


struct BernoulliGenerator {
  std::default_random_engine generator;
  std::bernoulli_distribution distribution;
//  double probabilityOfTrue; //between 0 and 1

  BernoulliGenerator(double probabiltyOfTrue) :
      generator(static_cast<unsigned int>(std::chrono::system_clock::now().time_since_epoch().count())),
      distribution(probabiltyOfTrue)
  {};

  bool drawLots() {
    return distribution(generator);
  }
};

AlgorithmSpec::ProcessCallback initCallback(InitContext& ctx);
void processCallback(ProcessingContext& ctx, BernoulliGenerator& bernoulliGenerator);

} //namespace DataSampling

} //namespace framework
} //namespace o2

#endif //FRAMEWORK_DATASAMPLER_H
