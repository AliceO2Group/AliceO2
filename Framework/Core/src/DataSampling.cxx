// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file DataSampling.cxx
/// \brief Implementation of O2 Data Sampling, v0.1
///
/// \author Piotr Konopka, piotr.jan.konopka@cern.ch

#include <random>
#include <boost/optional.hpp>
#include <Configuration/ConfigurationInterface.h>
#include <Configuration/ConfigurationFactory.h>
#include <Framework/ExternalFairMQDeviceProxy.h>

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

/// Returns appropriate comparator, dependent on whether all subSpecs are required or only one.
auto DataSampling::getEdgeMatcher(const QcTaskConfiguration &taskCfg){

  return taskCfg.subSpec == -1 ?
         [](const OutputSpec& externalOutput, const InputSpec& desiredData, const SubSpecificationType&) {
           return externalOutput.origin == desiredData.origin &&
                  externalOutput.description == desiredData.description;
         }
                               :
         [](const OutputSpec& externalOutput, const InputSpec& desiredData, const SubSpecificationType& desiredSubSpec) {
           return externalOutput.origin == desiredData.origin &&
                  externalOutput.description == desiredData.description &&
                  externalOutput.subSpec == desiredSubSpec;
         };
}

/// Returns appropriate dispatcher input/output creator, dependent on given configuration.
auto DataSampling::getEdgeCreator(const QcTaskConfiguration & taskCfg, const InfrastructureConfig &infrastructureCfg){

  return taskCfg.fairMqOutputChannelConfig.empty() ?
           (infrastructureCfg.enableParallelDispatchers ?
             [](DataProcessorSpec &dispatcher, const InputSpec &newInput){
               dispatcher.inputs.push_back(newInput);
               dispatcher.outputs.push_back(createDispatcherOutputSpec(newInput));
             }
                                                        :
             [](DataProcessorSpec &dispatcher, const InputSpec &newInput){
               dispatcher.inputs.push_back(newInput);
               OutputSpec newOutput = createDispatcherOutputSpec(newInput);
               if (std::find(dispatcher.outputs.begin(), dispatcher.outputs.end(), newOutput)
                   == dispatcher.outputs.end()){
                 dispatcher.outputs.push_back(newOutput);
               }
             })
                                                   :
           [](DataProcessorSpec &dispatcher, const InputSpec &newInput){
             dispatcher.inputs.push_back(newInput);
           };
}

/// Returns appropriate dispatcher initializer, dependent on whether dispatcher should send data to DPL or FairMQ device.
auto DataSampling::getDispatcherCreator(const QcTaskConfiguration & taskCfg){

  return taskCfg.fairMqOutputChannelConfig.empty() ?
         // create dispatcher with DPL output
         [](const SubSpecificationType &dispatcherSubSpec, const QcTaskConfiguration &task, const InputSpec &input) {
           return DataProcessorSpec{
             "Dispatcher" + std::to_string(dispatcherSubSpec) + "_for_" + task.name,
             Inputs{
               input
             },
             Outputs{
               createDispatcherOutputSpec(input)
             },
             AlgorithmSpec{
               [gen = BernoulliGenerator(task.fractionOfDataToSample)](ProcessingContext &ctx) mutable {
                 DataSampling::dispatcherCallback(ctx, gen);
               }
             }
           };
         }
                                                  :
         // create dispatcher with FairMQ output
         [](const SubSpecificationType& dispatcherSubSpec, const QcTaskConfiguration &task, const InputSpec &newInput)
         {
           //todo: throw an exception when 'name=' not found?
           size_t n_begin = task.fairMqOutputChannelConfig.find("name=") + sizeof("name=") - 1;
           size_t n_end = task.fairMqOutputChannelConfig.find_first_of(',', n_begin);
           std::string channel = task.fairMqOutputChannelConfig.substr(n_begin, n_end - n_begin);

           return DataProcessorSpec{
             "Dispatcher" + std::to_string(dispatcherSubSpec) + "_for_" + task.name,
             Inputs{
               newInput
             },
             Outputs{},
             AlgorithmSpec{
               [fraction=task.fractionOfDataToSample, channel](InitContext &ctx) {
                 return dispatcherInitCallbackFairMQ(ctx, channel, fraction);
               }
             }, {
               ConfigParamSpec{
                 "channel-config", VariantType::String,
                 task.fairMqOutputChannelConfig.c_str(), {"Out-of-band channel config"}}
             }
           };
         };
}

void DataSampling::GenerateInfrastructure(WorkflowSpec &workflow, const std::string &configurationSource) {

  QcTaskConfigurations tasks = readQcTasksConfiguration(configurationSource);
  InfrastructureConfig infrastructureCfg = readInfrastructureConfiguration(configurationSource);

  for (auto&& task : tasks) {

    // if necessary, create FairMQ -> DPL proxies and after that, look for their outputs in workflow.
    // (it is surely not an optimal way to do that, but this is only a temporary feature)
    for (auto&& fairMqProxy : task.desiredFairMqData) {
      workflow.emplace_back(specifyExternalFairMQDeviceProxy(
        ("FairMQ_proxy_for_" + task.name).c_str(),
        Outputs{fairMqProxy.outputSpec},
        fairMqProxy.channelConfig.c_str(),
        fairMqProxy.converterType == "o2DataModelAdaptor" ? o2DataModelAdaptor(fairMqProxy.outputSpec, 0, 1)
                                                          : incrementalConverter(fairMqProxy.outputSpec, 0, 1)
      ));
    }

    std::unordered_map<SubSpecificationType, DataProcessorSpec> dispatchers;

    // some lambda functions to make the later code cleaner and hide its configuration
    auto areEdgesMatching = getEdgeMatcher(task);
    auto createDispatcherSpec = getDispatcherCreator(task);
    auto addEdgesToDispatcher = getEdgeCreator(task, infrastructureCfg);

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

              addEdgesToDispatcher(res->second, newInput);

              if (infrastructureCfg.enableTimePipeliningDispatchers &&
                  res->second.maxInputTimeslices < dataProcessor.maxInputTimeslices) {
                res->second.maxInputTimeslices = dataProcessor.maxInputTimeslices;
              }
            } else {
              dispatchers[dispatcherSubSpec] = createDispatcherSpec(dispatcherSubSpec, task, newInput);

              //todo: can fairmq dispatchers be time pipelined?
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

AlgorithmSpec::ProcessCallback DataSampling::dispatcherInitCallback(InitContext &ctx) {

  BernoulliGenerator generator(0);
  return [generator](o2::framework::ProcessingContext& pCtx) mutable {
    o2::framework::DataSampling::dispatcherCallback(pCtx, generator);
  };
}

void DataSampling::dispatcherCallback(ProcessingContext &ctx, BernoulliGenerator &bernoulliGenerator) {

  InputRecord& inputs = ctx.inputs();

  if (bernoulliGenerator.drawLots()){
    for(auto& input : inputs){

      OutputSpec outputSpec = createDispatcherOutputSpec(*input.spec);

      const auto *inputHeader = Header::get<Header::DataHeader>(input.header);

      if (inputHeader->payloadSerializationMethod == Header::gSerializationMethodInvalid){
        LOG(ERROR) << "DataSampling::dispatcherCallback: input of origin'" << inputHeader->dataOrigin.str
                   << "', description '" << inputHeader->dataDescription.str
                   << "' has gSerializationMethodInvalid.";
      }
      else if (inputHeader->payloadSerializationMethod == Header::gSerializationMethodROOT){
        ctx.allocator().adopt(outputSpec, DataRefUtils::as<TObject>(input).release());
      }
      else{ //POD
        //todo: use API for that when it is available
        ctx.allocator().adoptChunk(outputSpec, const_cast<char*>(input.payload), inputHeader->size(),
                                   &Header::Stack::freefn, nullptr);
      }

      LOG(DEBUG) << "DataSampler sends data from subspec " << input.spec->subSpec;
    }
  }
}

AlgorithmSpec::ProcessCallback DataSampling::dispatcherInitCallbackFairMQ(InitContext &ctx, const std::string &channel,
                                                                          double fraction) {
  auto device = ctx.services().get<RawDeviceService>().device();
  auto gen = BernoulliGenerator(fraction);
//  std::string channel = ctx.options().get<std::string>("name");

  return [gen, device, channel](o2::framework::ProcessingContext& pCtx) mutable {
    o2::framework::DataSampling::dispatcherCallbackFairMQ(pCtx, gen, device, channel);
  };
}

void DataSampling::dispatcherCallbackFairMQ(ProcessingContext &ctx, BernoulliGenerator &bernoulliGenerator,
                                            FairMQDevice *device, const std::string &channel) {

  InputRecord& inputs = ctx.inputs();

  //FIXME: send all inputs inside one fairMQparts message?
  if (bernoulliGenerator.drawLots()){
    for(auto& input : inputs){

      OutputSpec outputSpec = createDispatcherOutputSpec(*input.spec);

      const auto *inputHeader = Header::get<Header::DataHeader>(input.header);

//      if (inputHeader->payloadSerializationMethod == Header::gSerializationMethodInvalid){
//        LOG(ERROR) << "DataSampling::dispatcherCallback: input of origin'" << inputHeader->dataOrigin.str
//                   << "', description '" << inputHeader->dataDescription.str
//                   << "' has gSerializationMethodInvalid.";
//      }
//      else if (inputHeader->payloadSerializationMethod == Header::gSerializationMethodROOT){
////        ctx.allocator().adopt(outputSpec, DataRefUtils::as<TObject>(input).release());
//        //todo: send root objects with fairmq
//      }
//      else{ //POD

        //FIXME: how to describe what data is this? Header + message?
        const auto *header = o2::Header::get<header::DataHeader>(input.header);

        char *p = new char[header->payloadSize];
        memcpy(p, input.payload, header->payloadSize);
        FairMQMessagePtr msg(
          device->NewMessage(
            p, header->payloadSize,
            [](void *data, void *hint) { delete[] reinterpret_cast<char *>(data); },
            p)
        );
      int bytesSent = device->Send(msg, channel);
      LOG(DEBUG) << "Bytes sent: " << bytesSent;
//      }
    }
  }
}

/// Creates dispatcher output specification basing on input specification of the same data. Basically, it adds '_S' at
/// the end of description, which makes data stream distinctive from the main flow (which is not sampled).
OutputSpec DataSampling::createDispatcherOutputSpec(const InputSpec &dispatcherInput) {

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

/// Reads QC Tasks configuration from given filepath. Uses Configuration dependency and handles most of its exceptions.
/// When some obligatory value is missing, it shows ERROR in logs, but continues to read another QC tasks.
DataSampling::QcTaskConfigurations DataSampling::readQcTasksConfiguration(const std::string &configurationSource) {

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
      //if there is a channelConfig specified, then user wants output in raw FairMQ layer, not DPL
      task.fairMqOutputChannelConfig = configFile->getString(simpleQcTaskDefinition + "/channelConfig").value_or("");

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

      // for temporary feature
      if (configFile->getInt(input + "/spawnConverter").value_or(0)){
        FairMqInput fairMqInput{
          OutputSpec{
            desiredData.origin,
            desiredData.description,
            task.subSpec == -1 ? 0 : task.subSpec,
          },
          configFile->getString(input + "/channelConfig").value_or(""),
          configFile->getString(input + "/converterType").value_or("incrementalConverter")
        };
        task.desiredFairMqData.push_back(fairMqInput);
      }
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

/// Reads general Data Sampling infrastructure configuration.
DataSampling::InfrastructureConfig DataSampling::readInfrastructureConfiguration(const std::string &configurationSource) {

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