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
  DataSamplerConfiguration conf;
  conf.fractionOfDataToSample = 1;
  return [conf](o2::framework::ProcessingContext& pCtx) {
    o2::framework::DataSampling::processCallback(pCtx, conf);
  };
}

//todo: root objects support
//todo: parallel sources support
//todo: optimisation
void processCallback(ProcessingContext& ctx, const DataSamplerConfiguration& conf)
{
  InputRecord& inputs = ctx.inputs();

  //todo: do random generator static for every processor ('static' word is not a solution, since variable would become
  //todo: shared among all DataProcessors)
  unsigned seed = static_cast<unsigned int>(std::chrono::system_clock::now().time_since_epoch().count());
  std::default_random_engine generator(seed);
  std::bernoulli_distribution distribution(conf.fractionOfDataToSample);

  if ( distribution(generator) ){
    for(auto& input : inputs){

      const InputSpec* inputSpec = input.spec;
      //todo: make sure it is a deep copy
      o2::Header::DataDescription outputDescription = inputSpec->description;

      //todo: better sampled data flagging
      size_t len = strlen(outputDescription.str);
      if (len < outputDescription.size-2){
        outputDescription.str[len] = '_';
        outputDescription.str[len+1] = 'S';
      }

      OutputSpec outputSpec{inputSpec->origin,
                            outputDescription,
                            inputSpec->subSpec,
                            static_cast<OutputSpec::Lifetime>(inputSpec->lifetime)};

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

std::vector<DataProcessorSpec> GenerateDataSamplers(const std::string& configurationSource, const std::vector<std::string>& taskNames)
{
  std::vector<DataProcessorSpec> dataSamplers;
  auto configFile = ConfigurationFactory::getConfiguration(configurationSource);

  for (auto&& taskName : taskNames) {

    DataSamplerConfiguration algorithmConfig;
    std::string taskInputsNames;

    try {
      std::string simpleQcTaskDefinition = configFile->getString(taskName + "/taskDefinition").value();
      taskInputsNames = configFile->getString(simpleQcTaskDefinition + "/inputs").value();
      algorithmConfig.fractionOfDataToSample = configFile->getFloat(simpleQcTaskDefinition + "/fraction").value();
      //todo: should I check if 0 < fraction < 1 ?

    } catch (const boost::bad_optional_access &) {
      //todo: warning or error?
      continue;
    }

    std::vector<std::string> taskInputsSplit;
    boost::split(taskInputsSplit, taskInputsNames, boost::is_any_of(","));

    Inputs inputs;
    for (auto &&input : taskInputsSplit) {
      std::cout << input << std::endl;
      InputSpec inputSpec{std::string(), 0, 0, 0, InputSpec::Timeframe};

      try {
        inputSpec.binding = configFile->getString(input + "/inputName").value();

        std::string origin = configFile->getString(input + "/dataOrigin").value();
        origin.copy(inputSpec.origin.str, (size_t) inputSpec.origin.size);

        std::string description = configFile->getString(input + "/dataDescription").value();
        description.copy(inputSpec.description.str, (size_t) inputSpec.description.size);

      } catch (const boost::bad_optional_access &) {
        //todo: warning or error?
        continue;
      }

      inputs.push_back(inputSpec);
    }

    //prepare outputs
    //todo: initialize size, no pushback
    Outputs outputs;
    for (auto &&input : inputs) {
      OutputSpec output{
        input.origin,
        input.description,
        input.subSpec,
        OutputSpec::Timeframe
      };

      //todo: find better way of flagging sampled data
      size_t len = strlen(output.description.str);
      if (len < output.description.size-2){
        output.description.str[len] = '_';
        output.description.str[len+1] = 'S';
      }

      outputs.push_back(output);
    }

    DataProcessorSpec dataSampler;
    dataSampler.inputs = inputs;
    dataSampler.outputs = outputs;
    dataSampler.algorithm = AlgorithmSpec{
      (AlgorithmSpec::ProcessCallback) [algorithmConfig](ProcessingContext &ctx) {
        DataSampling::processCallback(ctx, algorithmConfig);
      }
    };
    dataSampler.name = "DataSampler for " + taskName;

    dataSamplers.push_back(dataSampler);
  }

  return dataSamplers;
}

} //namespace DataSampling

} //namespace framework
} //namespace o2