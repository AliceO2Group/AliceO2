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

#include "Framework/ExternalFairMQDeviceProxy.h"
#include "Framework/DataSampling.h"
#include "Framework/ProcessingContext.h"
#include "Headers/DataHeader.h"
#include "FairLogger.h"

using namespace o2::framework;
using namespace o2::framework::DataSamplingConfig;
using namespace AliceO2::Configuration;

namespace o2 {
namespace framework {

// ideas:
// make sure if it supports 'vectors' of data
// how about not using dispatcher, when qc needs 100% data ? instead, connect it directly

/// Returns appropriate comparator, dependent on whether all subSpecs are required or only one.
auto DataSampling::getEdgeMatcher(const QcTaskConfiguration& taskCfg)
{
  return taskCfg.subSpec == -1 ?
         [](const OutputSpec& externalOutput, const InputSpec& desiredData, const SubSpecificationType&) {
           return externalOutput.origin == desiredData.origin &&
                  externalOutput.description == desiredData.description;
         }
                               :
         [](const OutputSpec& externalOutput, const InputSpec& desiredData,
            const SubSpecificationType& desiredSubSpec) {
           return externalOutput.origin == desiredData.origin &&
                  externalOutput.description == desiredData.description &&
                  externalOutput.subSpec == desiredSubSpec;
         };
}

/// Returns appropriate dispatcher initializer, dependent on whether dispatcher should send data to DPL or FairMQ
/// device.
auto DataSampling::getDispatcherCreator(const QcTaskConfiguration& taskCfg)
{
  // consider: use ROOT ?
  return taskCfg.fairMqOutputChannelConfig.empty() ?
         [](SubSpecificationType dispatcherSubSpec, const QcTaskConfiguration& task, const InfrastructureConfig& cfg) {
           return std::unique_ptr<Dispatcher>(new DispatcherDPL(dispatcherSubSpec, task, cfg));
         }
                                                   :
         [](SubSpecificationType dispatcherSubSpec, const QcTaskConfiguration& task, const InfrastructureConfig& cfg) {
           return std::unique_ptr<Dispatcher>(new DispatcherFairMQ(dispatcherSubSpec, task, cfg));
         };
}

void DataSampling::GenerateInfrastructure(WorkflowSpec& workflow, const std::string& configurationSource)
{
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

    std::vector<std::unique_ptr<Dispatcher>> dispatchers;

    // some lambda functions to make the later code cleaner and hide its configuration
    auto areEdgesMatching = getEdgeMatcher(task);
    auto createDispatcher = getDispatcherCreator(task);

    // Find all available outputs in workflow that match desired data. Create dispatchers that take them as inputs
    // and provide them filtered as outputs.
    for (auto&& desiredData : task.desiredDataSpecs) {
      bool wasDataFound = false;
      for (auto&& dataProcessor : workflow) {
        for (auto&& externalOutput : dataProcessor.outputs) {
          if (areEdgesMatching(externalOutput, desiredData, task.subSpec)) {

            wasDataFound = true;

            // if parallel dispatchers are not enabled, then edges will be added to the only one dispatcher.
            // in other case, new dispatcher will be created for every parallel flow.
            SubSpecificationType dispatcherSubSpec = infrastructureCfg.enableParallelDispatchers ?
                                                     externalOutput.subSpec : 0;

            auto res = std::find_if(dispatchers.begin(), dispatchers.end(),
                                    [dispatcherSubSpec](const auto& d) {
                                      return d->getSubSpec() == dispatcherSubSpec;
                                    });

            if (res != dispatchers.end()) {
              (*res)->addSource(dataProcessor, externalOutput, desiredData.binding);
            } else {
              auto newDispatcher = createDispatcher(dispatcherSubSpec, task, infrastructureCfg);
              newDispatcher->addSource(dataProcessor, externalOutput, desiredData.binding);
              dispatchers.push_back(std::move(newDispatcher));
            }
          }
        }
      }
      if (!wasDataFound) {
        LOG(ERROR) << "Data '" << desiredData.binding << "' for QC task '" << task.name
                   << "' not found in given workflow";
      }
    }
    for (auto& dispatcher : dispatchers) {
      workflow.push_back(std::move(dispatcher->getDataProcessorSpec()));
    }
  }
}

/// Reads QC Tasks configuration from given filepath. Uses Configuration dependency and handles most of its exceptions.
/// When some obligatory value is missing, it shows ERROR in logs, but continues to read another QC tasks.
QcTaskConfigurations DataSampling::readQcTasksConfiguration(const std::string& configurationSource)
{
  QcTaskConfigurations tasks;
  std::unique_ptr<ConfigurationInterface> configFile = ConfigurationFactory::getConfiguration(configurationSource);

  std::vector<std::string> taskNames;
  try {
    std::string taskNamesString = configFile->getString("DataSampling/tasksList").value();
    boost::split(taskNames, taskNamesString, boost::is_any_of(","));
  } catch (const boost::bad_optional_access&) {
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
      if (task.fractionOfDataToSample <= 0 || task.fractionOfDataToSample > 1) {
        LOG(ERROR) << "QC Task configuration error. In file " << configurationSource << ", value "
                   << simpleQcTaskDefinition + "/fraction" << " is not in range (0,1]. Setting value to 0.";
        task.fractionOfDataToSample = 0;
      }
      //if there is a channelConfig specified, then user wants output in raw FairMQ layer, not DPL
      task.fairMqOutputChannelConfig = configFile->getString(simpleQcTaskDefinition + "/channelConfig").value_or("");

      //FIXME: I do not like '-1' meaning 'all' - not 100% sure if it's safe to compare with '-1' later
      task.subSpec = static_cast<header::DataHeader::SubSpecificationType>(
        configFile->getInt(simpleQcTaskDefinition + "/subSpec").value_or(-1));

    } catch (const boost::bad_optional_access&) {
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

      } catch (const boost::bad_optional_access&) {
        LOG(ERROR) << "QC Task configuration error. In file " << configurationSource
                   << " input " << input << " has missing values";
        continue;
      }
      task.desiredDataSpecs.push_back(desiredData);

      // for temporary feature
      if (configFile->getInt(input + "/spawnConverter").value_or(0)) {
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
InfrastructureConfig DataSampling::readInfrastructureConfiguration(const std::string& configurationSource)
{

  InfrastructureConfig cfg;
  std::unique_ptr<ConfigurationInterface> configFile = ConfigurationFactory::getConfiguration(configurationSource);

  cfg.enableTimePipeliningDispatchers = static_cast<bool>(
    configFile->getInt("DataSampling/enableTimePipeliningDispatchers").get_value_or(0));
  cfg.enableParallelDispatchers = static_cast<bool>(
    configFile->getInt("DataSampling/enableParallelDispatchers").get_value_or(0));
  cfg.enableProxy = static_cast<bool>(
    configFile->getInt("DataSampling/enableProxy").get_value_or(0));

  return cfg;
}


} // namespace framework
} // namespace o2