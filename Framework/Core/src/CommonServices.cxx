// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include "Framework/CommonServices.h"
#include "Framework/ParallelContext.h"
#include "Framework/TextControlService.h"
#include "Framework/CallbackService.h"
#include "Framework/TimesliceIndex.h"
#include "Framework/ServiceRegistry.h"
#include "Framework/DeviceSpec.h"
#include "Framework/LocalRootFileService.h"
#include "Framework/DataRelayer.h"

#include <Configuration/ConfigurationInterface.h>
#include <Configuration/ConfigurationFactory.h>
#include <Monitoring/MonitoringFactory.h>
#include <InfoLogger/InfoLogger.hxx>

#include <options/FairMQProgOptions.h>

using AliceO2::InfoLogger::InfoLogger;
using AliceO2::InfoLogger::InfoLoggerContext;
using o2::configuration::ConfigurationFactory;
using o2::configuration::ConfigurationInterface;
using o2::monitoring::Monitoring;
using o2::monitoring::MonitoringFactory;

namespace o2::framework
{
o2::framework::ServiceSpec CommonServices::monitoringSpec()
{
  return ServiceSpec{"monitoring",
                     [](ServiceRegistry&, DeviceState&, fair::mq::ProgOptions& options) -> ServiceHandle {
                       void* service = MonitoringFactory::Get(options.GetPropertyAsString("monitoring-backend")).release();
                       return ServiceHandle{TypeIdHelpers::uniqueId<Monitoring>(), service};
                     },
                     noConfiguration(),
                     ServiceKind::Serial};
}

o2::framework::ServiceSpec CommonServices::infologgerContextSpec()
{
  return ServiceSpec{"infologger-contex",
                     simpleServiceInit<InfoLoggerContext, InfoLoggerContext>(),
                     noConfiguration(),
                     ServiceKind::Serial};
}

// Creates the sink for FairLogger / InfoLogger integration
auto createInfoLoggerSinkHelper(InfoLogger* logger, InfoLoggerContext* ctx)
{
  return [logger,
          ctx](const std::string& content, const fair::LogMetaData& metadata) {
    // translate FMQ metadata
    InfoLogger::InfoLogger::Severity severity = InfoLogger::Severity::Undefined;
    int level = InfoLogger::undefinedMessageOption.level;

    if (metadata.severity_name == fair::Logger::SeverityName(fair::Severity::nolog)) {
      // discard
      return;
    } else if (metadata.severity_name == fair::Logger::SeverityName(fair::Severity::fatal)) {
      severity = InfoLogger::Severity::Fatal;
    } else if (metadata.severity_name == fair::Logger::SeverityName(fair::Severity::error)) {
      severity = InfoLogger::Severity::Error;
    } else if (metadata.severity_name == fair::Logger::SeverityName(fair::Severity::warn)) {
      severity = InfoLogger::Severity::Warning;
    } else if (metadata.severity_name == fair::Logger::SeverityName(fair::Severity::state)) {
      severity = InfoLogger::Severity::Info;
      level = 10;
    } else if (metadata.severity_name == fair::Logger::SeverityName(fair::Severity::info)) {
      severity = InfoLogger::Severity::Info;
    } else if (metadata.severity_name == fair::Logger::SeverityName(fair::Severity::debug)) {
      severity = InfoLogger::Severity::Debug;
    } else if (metadata.severity_name == fair::Logger::SeverityName(fair::Severity::debug1)) {
      severity = InfoLogger::Severity::Debug;
      level = 10;
    } else if (metadata.severity_name == fair::Logger::SeverityName(fair::Severity::debug2)) {
      severity = InfoLogger::Severity::Debug;
      level = 20;
    } else if (metadata.severity_name == fair::Logger::SeverityName(fair::Severity::debug3)) {
      severity = InfoLogger::Severity::Debug;
      level = 30;
    } else if (metadata.severity_name == fair::Logger::SeverityName(fair::Severity::debug4)) {
      severity = InfoLogger::Severity::Debug;
      level = 40;
    } else if (metadata.severity_name == fair::Logger::SeverityName(fair::Severity::trace)) {
      severity = InfoLogger::Severity::Debug;
      level = 50;
    }

    InfoLogger::InfoLoggerMessageOption opt = {
      severity,
      level,
      InfoLogger::undefinedMessageOption.errorCode,
      metadata.file.c_str(),
      atoi(metadata.line.c_str())};

    if (logger) {
      logger->log(opt, *ctx, "DPL: %s", content.c_str());
    }
  };
};

o2::framework::ServiceSpec CommonServices::infologgerSpec()
{
  return ServiceSpec{"infologger",
                     [](ServiceRegistry& services, DeviceState&, fair::mq::ProgOptions& options) -> ServiceHandle {
                       auto infoLoggerMode = options.GetPropertyAsString("infologger-mode");
                       if (infoLoggerMode != "") {
                         setenv("INFOLOGGER_MODE", infoLoggerMode.c_str(), 1);
                       }
                       auto infoLoggerService = new InfoLogger;
                       auto infoLoggerContext = &services.get<InfoLoggerContext>();

                       auto infoLoggerSeverity = options.GetPropertyAsString("infologger-severity");
                       if (infoLoggerSeverity != "") {
                         fair::Logger::AddCustomSink("infologger", infoLoggerSeverity, createInfoLoggerSinkHelper(infoLoggerService, infoLoggerContext));
                       }
                       return ServiceHandle{TypeIdHelpers::uniqueId<InfoLogger>(), infoLoggerService};
                     },
                     noConfiguration(),
                     ServiceKind::Serial};
}

o2::framework::ServiceSpec CommonServices::configurationSpec()
{
  return ServiceSpec{
    "configuration",
    [](ServiceRegistry& services, DeviceState&, fair::mq::ProgOptions& options) -> ServiceHandle {
      auto backend = options.GetPropertyAsString("configuration");
      if (backend == "command-line") {
        return ServiceHandle{0, nullptr};
      }
      return ServiceHandle{TypeIdHelpers::uniqueId<ConfigurationInterface>(),
                           ConfigurationFactory::getConfiguration(backend).release()};
    },
    noConfiguration(),
    ServiceKind::Serial};
}

o2::framework::ServiceSpec CommonServices::controlSpec()
{
  return ServiceSpec{
    "control",
    [](ServiceRegistry& services, DeviceState& state, fair::mq::ProgOptions& options) -> ServiceHandle {
      return ServiceHandle{TypeIdHelpers::uniqueId<ControlService>(),
                           new TextControlService(services, state)};
    },
    noConfiguration(),
    ServiceKind::Serial};
}

o2::framework::ServiceSpec CommonServices::rootFileSpec()
{
  return ServiceSpec{
    "localrootfile",
    simpleServiceInit<LocalRootFileService, LocalRootFileService>(),
    noConfiguration(),
    ServiceKind::Serial};
}

o2::framework::ServiceSpec CommonServices::parallelSpec()
{
  return ServiceSpec{
    "parallel",
    [](ServiceRegistry& services, DeviceState&, fair::mq::ProgOptions& options) -> ServiceHandle {
      auto& spec = services.get<DeviceSpec const>();
      return ServiceHandle{TypeIdHelpers::uniqueId<ParallelContext>(),
                           new ParallelContext(spec.rank, spec.nSlots)};
    },
    noConfiguration(),
    ServiceKind::Serial};
}

o2::framework::ServiceSpec CommonServices::timesliceIndex()
{
  return ServiceSpec{
    "timesliceindex",
    simpleServiceInit<TimesliceIndex, TimesliceIndex>(),
    noConfiguration(),
    ServiceKind::Serial};
}

o2::framework::ServiceSpec CommonServices::callbacksSpec()
{
  return ServiceSpec{
    "callbacks",
    simpleServiceInit<CallbackService, CallbackService>(),
    noConfiguration(),
    ServiceKind::Serial};
}

o2::framework::ServiceSpec CommonServices::dataRelayer()
{
  return ServiceSpec{
    "datarelayer",
    [](ServiceRegistry& services, DeviceState&, fair::mq::ProgOptions& options) -> ServiceHandle {
      auto& spec = services.get<DeviceSpec const>();
      return ServiceHandle{TypeIdHelpers::uniqueId<DataRelayer>(),
                           new DataRelayer(spec.completionPolicy,
                                           spec.inputs,
                                           services.get<Monitoring>(),
                                           services.get<TimesliceIndex>())};
    },
    noConfiguration(),
    ServiceKind::Serial};
}

std::vector<ServiceSpec> CommonServices::defaultServices()
{
  return {
    timesliceIndex(),
    monitoringSpec(),
    infologgerContextSpec(),
    infologgerSpec(),
    configurationSpec(),
    controlSpec(),
    rootFileSpec(),
    parallelSpec(),
    callbacksSpec(),
    dataRelayer()};
}

} // namespace o2::framework
