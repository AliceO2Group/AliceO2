// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include <random>
#include <boost/optional.hpp>
#include <Configuration/ConfigurationInterface.h>
#include <Configuration/ConfigurationFactory.h>

#include "Framework/DataSampling.h"
#include "Framework/ProcessingContext.h"
#include "FairLogger.h" //for LOG()
#include "Headers/DataHeader.h"

using namespace o2::framework;
using namespace AliceO2::Configuration;

namespace o2 {
namespace framework {

namespace DataSampling {

AlgorithmSpec::ProcessCallback initCallback(InitContext &ctx)
{
  BernoulliGenerator conf(0);
  return [conf](o2::framework::ProcessingContext& pCtx) mutable {
    o2::framework::DataSampling::processCallback(pCtx, conf);
  };
}

//todo: root objects support
//todo: parallel sources support
//todo: optimisation
void processCallback(ProcessingContext& ctx, BernoulliGenerator& bernoulliGenerator)
{
  InputRecord& inputs = ctx.inputs();

  if (bernoulliGenerator.drawLots()){
    for(auto& input : inputs){

      const InputSpec* inputSpec = input.spec;
      o2::Header::DataDescription outputDescription = inputSpec->description;

      //todo: better sampled data flagging
      size_t len = strlen(outputDescription.str);
      if (len < outputDescription.size-2){
        outputDescription.str[len] = '_';
        outputDescription.str[len+1] = 'S';
      }

      OutputSpec outputSpec{inputSpec->origin,
                            outputDescription,
                            0,
                            static_cast<OutputSpec::Lifetime>(inputSpec->lifetime)};

      LOG(DEBUG) << "DataSampler sends data from subspec " << inputSpec->subSpec;

      const auto *inputHeader = o2::Header::get<o2::Header::DataHeader>(input.header);
      auto output = ctx.allocator().make<char>(outputSpec, inputHeader->size());

      //todo: use some std function or adopt(), when it is available for POD data
      const char* input_ptr = input.payload;
      for (char &it : output) {
        it = *input_ptr++;
      }
    }
  }
}


//todo: improve readability
void GenerateDataSamplers(WorkflowSpec& workflow, const std::string& configurationSource, const std::vector<std::string>& taskNames)
{
  auto configFile = ConfigurationFactory::getConfiguration(configurationSource);

  for (auto&& taskName : taskNames) {
    DataProcessorSpec dataSampler;
    std::string taskInputsNames;
    double fractionOfDataToSample = 0;

    try {
      std::string simpleQcTaskDefinition = configFile->getString(taskName + "/taskDefinition").value();
      taskInputsNames = configFile->getString(simpleQcTaskDefinition + "/inputs").value();
      fractionOfDataToSample = configFile->getFloat(simpleQcTaskDefinition + "/fraction").value();

      if ( fractionOfDataToSample <= 0 || fractionOfDataToSample > 1 ) {
        LOG(ERROR) << "QC Task configuration error. In file " << configurationSource << ", value "
                   << simpleQcTaskDefinition + "/fraction" << " is not in range (0,1]. Setting value to 0.";
        fractionOfDataToSample = 0;
      }

    } catch (const boost::bad_optional_access &) {
      LOG(ERROR) << "QC Task configuration error. In file " << configurationSource
                 << ", missing value or values for task " << taskName;
      continue;
    }

    std::vector<std::string> taskInputsSplit;
    boost::split(taskInputsSplit, taskInputsNames, boost::is_any_of(","));

    for (auto &&input : taskInputsSplit) {
      InputSpec inputSpec{std::string(), 0, 0, 0, InputSpec::Timeframe};

      try {
        inputSpec.binding = configFile->getString(input + "/inputName").value();

        std::string origin = configFile->getString(input + "/dataOrigin").value();
        origin.copy(inputSpec.origin.str, (size_t) inputSpec.origin.size);

        std::string description = configFile->getString(input + "/dataDescription").value();
        description.copy(inputSpec.description.str, (size_t) inputSpec.description.size);

      } catch (const boost::bad_optional_access &) {
        LOG(ERROR) << "QC Task configuration error. In file " << configurationSource
                   << " input " << input << " has missing values";
        continue;
      }

      for (auto&& dataProcessor : workflow) {
        for (auto && output : dataProcessor.outputs) {
          if (output.origin == inputSpec.origin && output.description == inputSpec.description) {
            inputSpec.subSpec = output.subSpec;
            dataSampler.inputs.push_back(inputSpec);
          }
        }
      }

      OutputSpec output{
        inputSpec.origin,
        inputSpec.description,
        0,
        static_cast<OutputSpec::Lifetime>(inputSpec.lifetime)
      };

      //todo: find better way of flagging sampled data
      //todo: case - two qcTasks have the same input, but with different % sampled
      size_t len = strlen(output.description.str);
      if (len < output.description.size-2) {
        output.description.str[len] = '_';
        output.description.str[len+1] = 'S';
      }

      dataSampler.outputs.push_back(output);
    }

    BernoulliGenerator bernoulliGenerator(fractionOfDataToSample);

    dataSampler.name = "Dispatcher for " + taskName;
    dataSampler.algorithm = AlgorithmSpec{
      (AlgorithmSpec::ProcessCallback) [bernoulliGenerator](ProcessingContext &ctx) mutable {
//        bernoulliGenerator.fractionOfDataToSample = 1;
        DataSampling::processCallback(ctx, bernoulliGenerator);
      }
    };

    workflow.push_back(dataSampler);
  }
}

} //namespace DataSampling

} //namespace framework
} //namespace o2