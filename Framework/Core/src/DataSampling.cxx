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

// todo: configuration of GenerateInfrastructure()
// options: ?
// - enable proxy (local vs global?)
// - enable time pipelining
// - one dispatcher per parallel flow vs one dispatcher per whole flow

void DataSampling::GenerateInfrastructure(WorkflowSpec &workflow,
                                          const std::string &configurationSource,
                                          const std::vector<std::string> &taskNames)
{
  QcTaskConfigurations tasks;
  readQcTasksConfiguration(configurationSource, taskNames, tasks);

  for (auto&& task : tasks) {
    DataProcessorSpec dispatcher{
      "Dispatcher_for_" + task.name,
      Inputs{}, Outputs{},
      AlgorithmSpec{
        [gen=BernoulliGenerator(task.fractionOfDataToSample)](ProcessingContext &ctx) mutable {
          DataSampling::dispatcherCallback(ctx, gen);
        }
      }
    };

    //find all available outputs in workflow that match desired data, and add them as dispatcher inputs
    for (auto desiredData : task.desiredDataSpecs) {
      for (auto&& dataProcessor : workflow) {
        for (auto&& externalOutput : dataProcessor.outputs) {
          if (externalOutput.origin == desiredData.origin &&
              externalOutput.description == desiredData.description) {
            desiredData.lifetime = static_cast<InputSpec::Lifetime>(externalOutput.lifetime);
            desiredData.subSpec = externalOutput.subSpec;
            dispatcher.inputs.push_back(desiredData);
          }
        }
      }
      dispatcher.outputs.emplace_back(createDispatcherOutputSpec(desiredData));
    }

    workflow.push_back(dispatcher);
  }
}

void DataSampling::GenerateInfrastructureParallel(WorkflowSpec &workflow,
                                          const std::string &configurationSource,
                                          const std::vector<std::string> &taskNames)
{
  QcTaskConfigurations tasks;
  readQcTasksConfiguration(configurationSource, taskNames, tasks);

  for (auto&& task : tasks) {
    std::unordered_map<Header::DataHeader::SubSpecificationType, DataProcessorSpec> dispatchers;

    //find all desired data outputs in workflow and create separate dispatcher for each parallel flow
    for (auto&& dataProcessor : workflow) {
      for (auto&& externalOutput : dataProcessor.outputs) {
        for (auto desiredData : task.desiredDataSpecs) {
          //if some output in workflow matches desired data
          if (externalOutput.origin == desiredData.origin &&
              externalOutput.description == desiredData.description) {

            desiredData.subSpec = externalOutput.subSpec;
            desiredData.lifetime = static_cast<InputSpec::Lifetime>(externalOutput.lifetime);

            //find if there is already a dispatcher for this parallel flow
            auto res = dispatchers.find(externalOutput.subSpec);
            if (res != dispatchers.end()){
              //add input to existing dispatcher
              res->second.inputs.push_back(desiredData);
              res->second.outputs.push_back(createDispatcherOutputSpec(desiredData));
            }
            else {
              //create a new dispatcher for this parallel flow
              dispatchers[externalOutput.subSpec] = DataProcessorSpec{
                "Dispatcher" + std::to_string(externalOutput.subSpec) + "_for_" + task.name,
                Inputs{
                  desiredData
                },
                Outputs{
                  createDispatcherOutputSpec(desiredData)
                },
                AlgorithmSpec{
                  [gen=BernoulliGenerator(task.fractionOfDataToSample)](ProcessingContext& ctx) mutable {
                    DataSampling::dispatcherCallback(ctx, gen);
                  }
                }
              };
            }
          }
        }
      }
    }

    for (auto& dispatcher : dispatchers) {
      workflow.push_back(std::move(dispatcher.second));
    }
  }
}

void DataSampling::GenerateInfrastructureTimePipelining(WorkflowSpec &workflow,
                                          const std::string &configurationSource,
                                          const std::vector<std::string> &taskNames)
{
  QcTaskConfigurations tasks;
  readQcTasksConfiguration(configurationSource, taskNames, tasks);

  for (auto&& task : tasks) {
    DataProcessorSpec dispatcher{
      "Dispatcher_for_" + task.name,
      Inputs{}, Outputs{},
      AlgorithmSpec{
        [gen=BernoulliGenerator(task.fractionOfDataToSample)](ProcessingContext &ctx) mutable {
          DataSampling::dispatcherCallback(ctx, gen);
        }
      }
    };

    size_t maxTimeParallelism = 1;
    for (auto desiredData : task.desiredDataSpecs) {
      for (auto&& dataProcessor : workflow) {
        for (auto&& externalOutput : dataProcessor.outputs) {
          if (externalOutput.origin == desiredData.origin &&
              externalOutput.description == desiredData.description) {
            desiredData.lifetime = static_cast<InputSpec::Lifetime>(externalOutput.lifetime);
            desiredData.subSpec = externalOutput.subSpec;
            dispatcher.inputs.push_back(desiredData);
            maxTimeParallelism = std::max(maxTimeParallelism, dataProcessor.maxInputTimeslices);
          }
        }
      }
      dispatcher.outputs.emplace_back(createDispatcherOutputSpec(desiredData));
    }
    dispatcher.maxInputTimeslices = maxTimeParallelism;

    workflow.push_back(dispatcher);
  }
}

AlgorithmSpec::ProcessCallback DataSampling::initCallback(InitContext &ctx)
{
  BernoulliGenerator generator(0);
  return [generator](o2::framework::ProcessingContext& pCtx) mutable {
    o2::framework::DataSampling::dispatcherCallback(pCtx, generator);
  };
}

void DataSampling::dispatcherCallback(ProcessingContext &ctx, BernoulliGenerator &bernoulliGenerator)
{
  InputRecord& inputs = ctx.inputs();

  if (bernoulliGenerator.drawLots()){
    for(auto& input : inputs){

      OutputSpec outputSpec = createDispatcherOutputSpec(*input.spec);

      const auto *inputHeader = o2::Header::get<o2::Header::DataHeader>(input.header);

      if (inputHeader->payloadSerializationMethod == o2::Header::gSerializationMethodInvalid){
        LOG(ERROR) << "DataSampling::dispatcherCallback: input of origin'" << inputHeader->dataOrigin.str
                   << "', description '" << inputHeader->dataDescription.str
                   << "' has gSerializationMethodInvalid.";
      }
      else if (inputHeader->payloadSerializationMethod == o2::Header::gSerializationMethodROOT){
        ctx.allocator().adopt(outputSpec, DataRefUtils::as<TObject>(input).release());
      }
      else{ //POD
        //todo: use API for that when it is available
        ctx.allocator().adoptChunk(outputSpec, const_cast<char*>(input.payload), inputHeader->size(),
                                   &o2::Header::Stack::freefn, nullptr);
      }

      LOG(DEBUG) << "DataSampler sends data from subspec " << input.spec->subSpec;
    }
  }
}

OutputSpec DataSampling::createDispatcherOutputSpec(const InputSpec &dispatcherInput)
{
  OutputSpec dispatcherOutput{
        dispatcherInput.origin,
        dispatcherInput.description,
        0,
        static_cast<OutputSpec::Lifetime>(dispatcherInput.lifetime)
      };

  size_t len = strlen(dispatcherOutput.description.str);
  if (len < dispatcherOutput.description.size-2) {
        dispatcherOutput.description.str[len] = '_';
        dispatcherOutput.description.str[len+1] = 'S';
      }

  return dispatcherOutput;
}

void DataSampling::readQcTasksConfiguration(const std::string &configurationSource,
                                     const std::vector<std::string> &taskNames,
                                     std::vector<QcTaskConfiguration>& tasks)
{
  std::unique_ptr<ConfigurationInterface> configFile = ConfigurationFactory::getConfiguration(configurationSource);

  for (auto&& taskName : taskNames) {

    QcTaskConfiguration task;
    task.name = taskName;

    std::string taskInputsNames;
    try {
      std::string simpleQcTaskDefinition = configFile->getString(taskName + "/taskDefinition").value();
      taskInputsNames = configFile->getString(simpleQcTaskDefinition + "/inputs").value();
      task.fractionOfDataToSample = configFile->getFloat(simpleQcTaskDefinition + "/fraction").value();

      if ( task.fractionOfDataToSample <= 0 || task.fractionOfDataToSample > 1 ) {
        LOG(ERROR) << "QC Task configuration error. In file " << configurationSource << ", value "
                   << simpleQcTaskDefinition + "/fraction" << " is not in range (0,1]. Setting value to 0.";
        task.fractionOfDataToSample = 0;
      }

    } catch (const boost::bad_optional_access &) {
      LOG(ERROR) << "QC Task configuration error. In file " << configurationSource
                 << ", missing value or values for task " << taskName;
      continue;
    }

    std::vector<std::string> taskInputsSplit;
    boost::split(taskInputsSplit, taskInputsNames, boost::is_any_of(","));

    for (auto&& input : taskInputsSplit) {

      InputSpec desiredData;
      try {
        desiredData.binding = configFile->getString(input + "/inputName").value();

        std::string origin = configFile->getString(input + "/dataOrigin").value();
        origin.copy(desiredData.origin.str, (size_t) desiredData.origin.size);

        std::string description = configFile->getString(input + "/dataDescription").value();
        description.copy(desiredData.description.str, (size_t) desiredData.description.size);

      } catch (const boost::bad_optional_access &) {
        LOG(ERROR) << "QC Task configuration error. In file " << configurationSource
                   << " input " << input << " has missing values";
        continue;
      }
      task.desiredDataSpecs.push_back(desiredData);
    }

    if (task.desiredDataSpecs.empty()) {
      LOG(ERROR) << "QC Task configuration error. In file " << configurationSource
                 << " task " << taskName << " has no valid inputs";
      continue;
    }
    tasks.push_back(task);
  }
}


} //namespace framework
} //namespace o2