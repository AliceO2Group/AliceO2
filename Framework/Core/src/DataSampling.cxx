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

//ideas:
//make sure if it supports 'vectors' of data
//how about not using dispatcher, when qc needs 100% data ? instead, connect it directly
//how about giving some information, if some desired outputs weren't found?

auto DataSampling::getEdgeMatcher(const SubSpecificationType &subSpec)
{
  return subSpec == -1 ?
         [](const OutputSpec& externalOutput, const InputSpec& desiredData, const DataSampling::SubSpecificationType&) {
           return externalOutput.origin == desiredData.origin &&
                  externalOutput.description == desiredData.description;
         }
                       :
         [](const OutputSpec& externalOutput, const InputSpec& desiredData, const DataSampling::SubSpecificationType& desiredSubSpec) {
           return externalOutput.origin == desiredData.origin &&
                  externalOutput.description == desiredData.description &&
                  externalOutput.subSpec == desiredSubSpec;
         };
}

void DataSampling::GenerateInfrastructure(WorkflowSpec &workflow, const std::string &configurationSource)
{
  //todo:
  // proxy

  QcTaskConfigurations tasks = readQcTasksConfiguration(configurationSource);
  InfrastructureConfig infrastructureCfg = readInfrastructureConfiguration(configurationSource);

  for (auto&& task : tasks) {

    std::unordered_map<SubSpecificationType, DataProcessorSpec> dispatchers;
    auto areEdgesMatching = getEdgeMatcher(task.subSpec);

    // Find all available outputs in workflow that match desired data. Create dispatchers that take them as inputs
    // and provide them filtered as outputs.
    for (auto&& desiredData : task.desiredDataSpecs) {
      for (auto&& dataProcessor : workflow) {
        for (auto&& externalOutput : dataProcessor.outputs) {
          if (areEdgesMatching(externalOutput, desiredData, task.subSpec)) {

            InputSpec newInput{
              desiredData.binding,
              desiredData.origin,
              desiredData.description,
              externalOutput.subSpec,
              static_cast<InputSpec::Lifetime>(externalOutput.lifetime),
            };

            // if parallel dispatchers are not enabled, then edges will be added to the only one dispatcher.
            // in other case, new dispatcher will be created for every parallel flow.
            SubSpecificationType dispatcherSubSpec = infrastructureCfg.enableParallelDispatchers ?
                                                     externalOutput.subSpec : 0;

            auto res = dispatchers.find(dispatcherSubSpec);
            if (res != dispatchers.end()) {
              res->second.inputs.push_back(newInput);
              OutputSpec newOutput = createDispatcherOutputSpec(newInput);
              if (infrastructureCfg.enableParallelDispatchers ||
                  std::find(res->second.outputs.begin(), res->second.outputs.end(), newOutput) == res->second.outputs.end()){
                res->second.outputs.push_back(newOutput);
              }
              if (infrastructureCfg.enableTimePipeliningDispatchers &&
                  res->second.maxInputTimeslices < dataProcessor.maxInputTimeslices) {
                res->second.maxInputTimeslices = dataProcessor.maxInputTimeslices;
              }
            } else {
              dispatchers[dispatcherSubSpec] = DataProcessorSpec{
                "Dispatcher" + std::to_string(dispatcherSubSpec) + "_for_" + task.name,
                Inputs{
                  newInput
                },
                Outputs{
                  createDispatcherOutputSpec(newInput)
                },
                AlgorithmSpec{
                  [gen = BernoulliGenerator(task.fractionOfDataToSample)](ProcessingContext &ctx) mutable {
                    DataSampling::dispatcherCallback(ctx, gen);
                  }
                }
              };
              if (infrastructureCfg.enableTimePipeliningDispatchers) {
                dispatchers[dispatcherSubSpec].maxInputTimeslices = dataProcessor.maxInputTimeslices;
              }
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

DataSampling::QcTaskConfigurations DataSampling::readQcTasksConfiguration(const std::string &configurationSource)
{
  std::vector<QcTaskConfiguration> tasks;
  std::unique_ptr<ConfigurationInterface> configFile = ConfigurationFactory::getConfiguration(configurationSource);

  std::vector<std::string> taskNames;
  try {
    std::string taskNamesString = configFile->getString("DataSampling/tasksList").value();
    boost::split(taskNames, taskNamesString, boost::is_any_of(","));
  } catch (const boost::bad_optional_access &) {
    LOG(ERROR) << "QC Task configuration error. In file " << configurationSource
               << ", wrong or missing value DataSampling/tasksList";
    return QcTaskConfigurations();
  }

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
      //FIXME: I do not like '-1' meaning 'all' - not 100% sure if it's safe to compare with '-1' later
      task.subSpec = static_cast<header::DataHeader::SubSpecificationType>(
        configFile->getInt(simpleQcTaskDefinition + "/subSpec").value_or(-1));

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

  return tasks;
}

DataSampling::InfrastructureConfig DataSampling::readInfrastructureConfiguration(const std::string &configurationSource)
{
  InfrastructureConfig cfg;
  std::unique_ptr<ConfigurationInterface> configFile = ConfigurationFactory::getConfiguration(configurationSource);

  cfg.enableTimePipeliningDispatchers = static_cast<bool>(
    configFile->getInt("DataSampling/enableTimePipeliningDispatchers").get_value_or(0));
  cfg.enableParallelDispatchers= static_cast<bool>(
    configFile->getInt("DataSampling/enableParallelDispatchers").get_value_or(0));
  cfg.enableProxy = static_cast<bool>(
    configFile->getInt("DataSampling/enableProxy").get_value_or(0));

  return cfg;
}


} //namespace framework
} //namespace o2