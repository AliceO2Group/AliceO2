// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file Dispatcher.cxx
/// \brief Implementation of Dispatcher for O2 Data Sampling
///
/// \author Piotr Konopka, piotr.jan.konopka@cern.ch

#include "DataSampling/Dispatcher.h"
#include "DataSampling/DataSamplingPolicy.h"
#include "DataSampling/DataSamplingHeader.h"
#include "Framework/DataProcessingHeader.h"
#include "Framework/DataSpecUtils.h"
#include "Framework/Logger.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/Monitoring.h"
#include "Framework/DataRefUtils.h"
#include "Framework/FairMQDeviceProxy.h"
#include "Framework/DataProcessingHelpers.h"
#include "Framework/DataRelayer.h"

#include <Configuration/ConfigurationInterface.h>
#include <Configuration/ConfigurationFactory.h>

using namespace o2::configuration;
using namespace o2::monitoring;
using namespace o2::framework;

namespace o2::utilities
{

Dispatcher::Dispatcher(std::string name, const std::string reconfigurationSource)
  : mName(std::move(name)), mReconfigurationSource(reconfigurationSource)
{
}

Dispatcher::~Dispatcher() = default;

void Dispatcher::init(InitContext& ctx)
{
  LOG(debug) << "Reading Data Sampling Policies...";
  boost::property_tree::ptree policiesTree;

  if (mReconfigurationSource.empty() == false) {
    std::unique_ptr<ConfigurationInterface> cfg = ConfigurationFactory::getConfiguration(mReconfigurationSource);
    policiesTree = cfg->getRecursive("dataSamplingPolicies");
    mPolicies.clear();
  } else if (ctx.options().isSet("sampling-config-ptree")) {
    policiesTree = ctx.options().get<boost::property_tree::ptree>("sampling-config-ptree");
    mPolicies.clear();
  } else {
    ; // we use policies declared during workflow init.
  }

  for (auto&& policyConfig : policiesTree) {
    // we don't want the Dispatcher to exit due to one faulty Policy
    try {
      mPolicies.emplace_back(std::make_shared<DataSamplingPolicy>(DataSamplingPolicy::fromConfiguration(policyConfig.second)));
    } catch (std::exception& ex) {
      LOG(warn) << "Could not load the Data Sampling Policy '"
                << policyConfig.second.get_optional<std::string>("id").value_or("") << "', because: " << ex.what();
    } catch (...) {
      LOG(warn) << "Could not load the Data Sampling Policy '"
                << policyConfig.second.get_optional<std::string>("id").value_or("") << "'";
    }
  }

  auto& spec = ctx.services().get<const DeviceSpec>();
  mDeviceID.runtimeInit(spec.id.substr(0, DataSamplingHeader::deviceIDTypeSize).c_str());
}

void Dispatcher::run(ProcessingContext& ctx)
{
  // todo: consider matching (and deciding) in completion policy to save some time
  //  it is not trivial though, we would have to share state with the customize() method,
  //  which is not possible atm.

  for (auto inputIt = ctx.inputs().begin(); inputIt != ctx.inputs().end(); inputIt++) {

    const DataRef& firstPart = inputIt.getByPos(0);
    if (firstPart.header == nullptr) {
      continue;
    }
    const auto* firstInputHeader = DataRefUtils::getHeader<header::DataHeader*>(firstPart);
    ConcreteDataMatcher inputMatcher{firstInputHeader->dataOrigin, firstInputHeader->dataDescription, firstInputHeader->subSpecification};

    for (auto& policy : mPolicies) {
      // fixme: in principle matching could be broken by having query "TST/RAWDATA/0" and having parts with just
      //  the first subspec == 0, but others could be different. However, we trust that DPL does necessary checks
      //  during workflow validation and when passing messages (e.g. query "TST/RAWDATA/0" should not match
      //  a "TST/RAWDATA/*" output.
      if (auto route = policy->match(inputMatcher); route != nullptr && policy->decide(firstPart)) {
        auto routeAsConcreteDataType = DataSpecUtils::asConcreteDataTypeMatcher(*route);
        auto dsheader = prepareDataSamplingHeader(*policy);
        for (const auto& part : inputIt) {
          if (part.header != nullptr) {
            // We copy every header which is not DataHeader or DataProcessingHeader,
            // so that custom data-dependent headers are passed forward,
            // and we add a DataSamplingHeader.
            header::Stack headerStack{
              std::move(extractAdditionalHeaders(part.header)),
              dsheader};
            const auto* partInputHeader = DataRefUtils::getHeader<header::DataHeader*>(part);

            Output output{
              routeAsConcreteDataType.origin,
              routeAsConcreteDataType.description,
              partInputHeader->subSpecification,
              part.spec->lifetime,
              std::move(headerStack)};
            send(ctx.outputs(), part, output);
          }
        }
      }
    }
  }

  if (ctx.inputs().isValid("timer-stats")) {
    reportStats(ctx.services().get<Monitoring>());
  }
  auto& proxy = ctx.services().get<FairMQDeviceProxy>();
  auto& relayer = ctx.services().get<DataRelayer>();
  auto timeslice = relayer.getOldestPossibleOutput().timeslice.value;
  DataProcessingHelpers::broadcastOldestPossibleTimeslice(proxy, timeslice);
}

void Dispatcher::reportStats(Monitoring& monitoring) const
{
  uint64_t dispatcherTotalEvaluatedMessages = 0;
  uint64_t dispatcherTotalAcceptedMessages = 0;

  for (const auto& policy : mPolicies) {
    dispatcherTotalEvaluatedMessages += policy->getTotalEvaluatedMessages();
    dispatcherTotalAcceptedMessages += policy->getTotalAcceptedMessages();
  }

  monitoring.send(Metric{dispatcherTotalEvaluatedMessages, "Dispatcher_messages_evaluated", Verbosity::Prod}.addTag(tags::Key::Subsystem, tags::Value::DataSampling));
  monitoring.send(Metric{dispatcherTotalAcceptedMessages, "Dispatcher_messages_passed", Verbosity::Prod}.addTag(tags::Key::Subsystem, tags::Value::DataSampling));
}

DataSamplingHeader Dispatcher::prepareDataSamplingHeader(const DataSamplingPolicy& policy)
{
  uint64_t sampleTime = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count());

  return {
    sampleTime,
    policy.getTotalAcceptedMessages(),
    policy.getTotalEvaluatedMessages(),
    mDeviceID};
}

header::Stack Dispatcher::extractAdditionalHeaders(const char* inputHeaderStack) const
{
  header::Stack headerStack;

  const auto* first = header::BaseHeader::get(reinterpret_cast<const std::byte*>(inputHeaderStack));
  for (const auto* current = first; current != nullptr; current = current->next()) {
    if (current->description != header::DataHeader::sHeaderType &&
        current->description != DataProcessingHeader::sHeaderType) {
      headerStack = std::move(header::Stack{std::move(headerStack), *current});
    }
  }

  return headerStack;
}

void Dispatcher::send(DataAllocator& dataAllocator, const DataRef& inputData, const Output& output) const
{
  const auto* inputHeader = DataRefUtils::getHeader<header::DataHeader*>(inputData);
  dataAllocator.snapshot(output, inputData.payload, DataRefUtils::getPayloadSize(inputData), inputHeader->payloadSerializationMethod);
}

void Dispatcher::registerPolicy(std::unique_ptr<DataSamplingPolicy>&& policy)
{
  mPolicies.emplace_back(std::move(policy));
}

const std::string& Dispatcher::getName()
{
  return mName;
}

Inputs Dispatcher::getInputSpecs()
{
  Inputs declaredInputs;

  // Add data inputs. Avoid duplicates and inputs which include others (e.g. "TST/DATA" includes "TST/DATA/1".
  for (const auto& policy : mPolicies) {
    for (const auto& path : policy->getPathMap()) {
      auto& potentiallyNewInput = path.first;

      // The idea is that we remove all existing inputs which are covered by the potentially new input.
      // If there are none which are broader than the new one, then we add it.
      auto newInputIsBroader = [&potentiallyNewInput](const InputSpec& other) {
        return DataSpecUtils::includes(potentiallyNewInput, other);
      };
      declaredInputs.erase(std::remove_if(declaredInputs.begin(), declaredInputs.end(), newInputIsBroader), declaredInputs.end());

      auto declaredInputIsBroader = [&potentiallyNewInput](const InputSpec& other) {
        return DataSpecUtils::includes(other, potentiallyNewInput);
      };
      if (std::none_of(declaredInputs.begin(), declaredInputs.end(), declaredInputIsBroader)) {
        declaredInputs.push_back(potentiallyNewInput);
      }
    }
  }

  // add timer input
  header::DataDescription timerDescription;
  timerDescription.runtimeInit(("TIMER-" + mName).substr(0, 16).c_str());
  declaredInputs.emplace_back(InputSpec{"timer-stats", "DS", timerDescription, 0, Lifetime::Timer});

  return declaredInputs;
}

Outputs Dispatcher::getOutputSpecs()
{
  Outputs declaredOutputs;
  for (const auto& policy : mPolicies) {
    for (const auto& [_policyInput, policyOutput] : policy->getPathMap()) {
      (void)_policyInput;
      // In principle Data Sampling Policies should have different outputs.
      // We may add a check to be very gentle with users.
      declaredOutputs.push_back(policyOutput);
    }
  }
  return declaredOutputs;
}
framework::Options Dispatcher::getOptions()
{
  return {{"period-timer-stats", framework::VariantType::Int, 10 * 1000000, {"Dispatcher's stats timer period"}}};
}

size_t Dispatcher::numberOfPolicies()
{
  return mPolicies.size();
}

} // namespace o2::utilities
