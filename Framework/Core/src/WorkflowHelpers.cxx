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
#include "WorkflowHelpers.h"
#include "Framework/AlgorithmSpec.h"
#include "Framework/AODReaderHelpers.h"
#include "Framework/ChannelMatching.h"
#include "Framework/ConfigParamsHelper.h"
#include "Framework/CommonDataProcessors.h"
#include "Framework/ConfigContext.h"
#include "Framework/DeviceSpec.h"
#include "Framework/DataSpecUtils.h"
#include "Framework/DataAllocator.h"
#include "Framework/ControlService.h"
#include "Framework/RawDeviceService.h"
#include "Framework/StringHelpers.h"
#include "Framework/CommonMessageBackends.h"
#include "Framework/ChannelSpecHelpers.h"
#include "Framework/ExternalFairMQDeviceProxy.h"
#include "Framework/Plugins.h"
#include "Framework/DataTakingContext.h"
#include "ArrowSupport.h"

#include "Headers/DataHeader.h"
#include <algorithm>
#include <list>
#include <set>
#include <utility>
#include <vector>
#include <climits>
#include <thread>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"

namespace o2::framework
{

std::ostream& operator<<(std::ostream& out, TopoIndexInfo const& info)
{
  out << "(" << info.index << ", " << info.layer << ")";
  return out;
}

std::vector<TopoIndexInfo>
  WorkflowHelpers::topologicalSort(size_t nodeCount,
                                   int const* edgeIn,
                                   int const* edgeOut,
                                   size_t byteStride,
                                   size_t edgesCount)
{
  size_t stride = byteStride / sizeof(int);
  using EdgeIndex = int;
  // Create the index which will be returned.
  std::vector<TopoIndexInfo> index(nodeCount);
  for (auto wi = 0; wi < nodeCount; ++wi) {
    index[wi] = {wi, 0};
  }
  std::vector<EdgeIndex> remainingEdgesIndex(edgesCount);
  for (EdgeIndex ei = 0; ei < edgesCount; ++ei) {
    remainingEdgesIndex[ei] = ei;
  }

  // Create a vector where at each position we have true
  // if the vector has dependencies, false otherwise
  std::vector<bool> nodeDeps(nodeCount, false);
  for (EdgeIndex ei = 0; ei < edgesCount; ++ei) {
    nodeDeps[*(edgeOut + ei * stride)] = true;
  }

  // We start with all those which do not have any dependencies
  // They are layer 0.
  std::list<TopoIndexInfo> L;
  for (auto ii = 0; ii < index.size(); ++ii) {
    if (nodeDeps[ii] == false) {
      L.push_back({ii, 0});
    }
  }

  // The final result.
  std::vector<TopoIndexInfo> S;
  // The set of vertices which can be reached by the current node
  std::set<TopoIndexInfo> nextVertex;
  // The set of edges which are not related to the current node.
  std::vector<EdgeIndex> nextEdges;
  while (!L.empty()) {
    auto node = L.front();
    S.push_back(node);
    L.pop_front();
    nextVertex.clear();
    nextEdges.clear();

    // After this, nextVertex will contain all the vertices
    // which have the current node as incoming.
    // nextEdges will contain all the edges which are not related
    // to the current node.
    for (auto& ei : remainingEdgesIndex) {
      if (*(edgeIn + ei * stride) == node.index) {
        nextVertex.insert({*(edgeOut + ei * stride), node.layer + 1});
      } else {
        nextEdges.push_back(ei);
      }
    }
    remainingEdgesIndex.swap(nextEdges);

    // Of all the vertices which have node as incoming,
    // check if there is any other incoming node.
    std::set<TopoIndexInfo> hasPredecessors;
    for (auto& ei : remainingEdgesIndex) {
      for (auto& m : nextVertex) {
        if (m.index == *(edgeOut + ei * stride)) {
          hasPredecessors.insert({m.index, m.layer});
        }
      }
    }
    std::vector<TopoIndexInfo> withPredecessor;
    std::set_difference(nextVertex.begin(), nextVertex.end(),
                        hasPredecessors.begin(), hasPredecessors.end(),
                        std::back_inserter(withPredecessor));
    std::copy(withPredecessor.begin(), withPredecessor.end(), std::back_inserter(L));
  }
  return S;
}

void WorkflowHelpers::addMissingOutputsToReader(std::vector<OutputSpec> const& providedOutputs,
                                                std::vector<InputSpec> const& requestedInputs,
                                                DataProcessorSpec& publisher)
{
  auto matchingOutputFor = [](InputSpec const& requested) {
    return [&requested](OutputSpec const& provided) {
      return DataSpecUtils::match(requested, provided);
    };
  };
  for (InputSpec const& requested : requestedInputs) {
    auto provided = std::find_if(providedOutputs.begin(),
                                 providedOutputs.end(),
                                 matchingOutputFor(requested));

    if (provided != providedOutputs.end()) {
      continue;
    }

    auto inList = std::find_if(publisher.outputs.begin(),
                               publisher.outputs.end(),
                               matchingOutputFor(requested));
    if (inList != publisher.outputs.end()) {
      continue;
    }

    auto concrete = DataSpecUtils::asConcreteDataMatcher(requested);
    publisher.outputs.emplace_back(OutputSpec{concrete.origin, concrete.description, concrete.subSpec, requested.lifetime, requested.metadata});
  }
}

void WorkflowHelpers::addMissingOutputsToSpawner(std::vector<OutputSpec> const& providedSpecials,
                                                 std::vector<InputSpec> const& requestedSpecials,
                                                 std::vector<InputSpec>& requestedAODs,
                                                 DataProcessorSpec& publisher)
{
  for (auto& input : requestedSpecials) {
    if (std::any_of(providedSpecials.begin(), providedSpecials.end(), [&input](auto const& x) {
          return DataSpecUtils::match(input, x);
        })) {
      continue;
    }
    auto concrete = DataSpecUtils::asConcreteDataMatcher(input);
    publisher.outputs.emplace_back(OutputSpec{concrete.origin, concrete.description, concrete.subSpec});
    for (auto& i : input.metadata) {
      if ((i.type == VariantType::String) && (i.name.find("input:") != std::string::npos)) {
        auto spec = DataSpecUtils::fromMetadataString(i.defaultValue.get<std::string>());
        auto j = std::find(publisher.inputs.begin(), publisher.inputs.end(), spec);
        if (j == publisher.inputs.end()) {
          publisher.inputs.push_back(spec);
        }
        DataSpecUtils::updateInputList(requestedAODs, std::move(spec));
      }
    }
  }
}

void WorkflowHelpers::addMissingOutputsToBuilder(std::vector<InputSpec> const& requestedSpecials,
                                                 std::vector<InputSpec>& requestedAODs,
                                                 std::vector<InputSpec>& requestedDYNs,
                                                 DataProcessorSpec& publisher)
{
  for (auto& input : requestedSpecials) {
    auto concrete = DataSpecUtils::asConcreteDataMatcher(input);
    publisher.outputs.emplace_back(OutputSpec{concrete.origin, concrete.description, concrete.subSpec});
    for (auto& i : input.metadata) {
      if ((i.type == VariantType::String) && (i.name.find("input:") != std::string::npos)) {
        auto spec = DataSpecUtils::fromMetadataString(i.defaultValue.get<std::string>());
        auto j = std::find_if(publisher.inputs.begin(), publisher.inputs.end(), [&](auto x) { return x.binding == spec.binding; });
        if (j == publisher.inputs.end()) {
          publisher.inputs.push_back(spec);
        }
        if (DataSpecUtils::partialMatch(spec, header::DataOrigin{"AOD"})) {
          DataSpecUtils::updateInputList(requestedAODs, std::move(spec));
        } else if (DataSpecUtils::partialMatch(spec, header::DataOrigin{"DYN"})) {
          DataSpecUtils::updateInputList(requestedDYNs, std::move(spec));
        }
      }
    }
  }
}

// get the default value for condition-backend
std::string defaultConditionBackend()
{
  static bool explicitBackend = getenv("DPL_CONDITION_BACKEND");
  static DeploymentMode deploymentMode = CommonServices::getDeploymentMode();
  if (explicitBackend) {
    return getenv("DPL_CONDITION_BACKEND");
  } else if (deploymentMode == DeploymentMode::OnlineDDS || deploymentMode == DeploymentMode::OnlineECS) {
    return "http://o2-ccdb.internal";
  } else {
    return "http://alice-ccdb.cern.ch";
  }
}

// get the default value for condition query rate
int64_t defaultConditionQueryRate()
{
  return getenv("DPL_CONDITION_QUERY_RATE") ? std::stoll(getenv("DPL_CONDITION_QUERY_RATE")) : 0;
}

void WorkflowHelpers::injectServiceDevices(WorkflowSpec& workflow, ConfigContext const& ctx)
{
  auto fakeCallback = AlgorithmSpec{[](InitContext& ic) {
    LOG(info) << "This is not a real device, merely a placeholder for external inputs";
    LOG(info) << "To be hidden / removed at some point.";
    // mark this dummy process as ready-to-quit
    ic.services().get<ControlService>().readyToQuit(QuitRequest::Me);

    return [](ProcessingContext& pc) {
      // this callback is never called since there is no expiring input
      pc.services().get<RawDeviceService>().waitFor(2000);
    };
  }};

  DataProcessorSpec ccdbBackend{
    .name = "internal-dpl-ccdb-backend",
    .outputs = {},
    .options = {{"condition-backend", VariantType::String, defaultConditionBackend(), {"URL for CCDB"}},
                {"condition-not-before", VariantType::Int64, 0ll, {"do not fetch from CCDB objects created before provide timestamp"}},
                {"condition-not-after", VariantType::Int64, 3385078236000ll, {"do not fetch from CCDB objects created after the timestamp"}},
                {"condition-remap", VariantType::String, "", {"remap condition path in CCDB based on the provided string."}},
                {"condition-tf-per-query", VariantType::Int64, defaultConditionQueryRate(), {"check condition validity per requested number of TFs, fetch only once if <0"}},
                {"condition-time-tolerance", VariantType::Int64, 5000ll, {"prefer creation time if its difference to orbit-derived time exceeds threshold (ms), impose if <0"}},
                {"orbit-offset-enumeration", VariantType::Int64, 0ll, {"initial value for the orbit"}},
                {"orbit-multiplier-enumeration", VariantType::Int64, 0ll, {"multiplier to get the orbit from the counter"}},
                {"start-value-enumeration", VariantType::Int64, 0ll, {"initial value for the enumeration"}},
                {"end-value-enumeration", VariantType::Int64, -1ll, {"final value for the enumeration"}},
                {"step-value-enumeration", VariantType::Int64, 1ll, {"step between one value and the other"}}},
  };
  DataProcessorSpec transientStore{"internal-dpl-transient-store",
                                   {},
                                   {},
                                   AlgorithmSpec::dummyAlgorithm()};
  DataProcessorSpec qaStore{"internal-dpl-qa-store",
                            {},
                            {},
                            AlgorithmSpec::dummyAlgorithm()};
  DataProcessorSpec timer{"internal-dpl-clock",
                          {},
                          {},
                          AlgorithmSpec::dummyAlgorithm()};

  // In case InputSpec of origin AOD are
  // requested but not available as part of the workflow,
  // we insert in the configuration something which
  // reads them from file.
  //
  // FIXME: source branch is DataOrigin, for the moment. We should
  //        make it configurable via ConfigParamsOptions
  auto aodLifetime = Lifetime::Enumeration;
  if (ctx.options().get<int64_t>("aod-memory-rate-limit")) {
    aodLifetime = Lifetime::Signal;
  }

  DataProcessorSpec aodReader{
    "internal-dpl-aod-reader",
    {InputSpec{"enumeration",
               "DPL",
               "ENUM",
               static_cast<DataAllocator::SubSpecificationType>(compile_time_hash("internal-dpl-aod-reader")),
               aodLifetime}},
    {},
    AlgorithmSpec::dummyAlgorithm(),
    {ConfigParamSpec{"aod-file", VariantType::String, {"Input AOD file"}},
     ConfigParamSpec{"aod-reader-json", VariantType::String, {"json configuration file"}},
     ConfigParamSpec{"aod-parent-access-level", VariantType::String, {"Allow parent file access up to specified level. Default: no (0)"}},
     ConfigParamSpec{"aod-parent-base-path-replacement", VariantType::String, {R"(Replace base path of parent files. Syntax: FROM;TO. E.g. "alien:///path/in/alien;/local/path". Enclose in "" on the command line.)"}},
     ConfigParamSpec{"time-limit", VariantType::Int64, 0ll, {"Maximum run time limit in seconds"}},
     ConfigParamSpec{"orbit-offset-enumeration", VariantType::Int64, 0ll, {"initial value for the orbit"}},
     ConfigParamSpec{"orbit-multiplier-enumeration", VariantType::Int64, 0ll, {"multiplier to get the orbit from the counter"}},
     ConfigParamSpec{"start-value-enumeration", VariantType::Int64, 0ll, {"initial value for the enumeration"}},
     ConfigParamSpec{"end-value-enumeration", VariantType::Int64, -1ll, {"final value for the enumeration"}},
     ConfigParamSpec{"step-value-enumeration", VariantType::Int64, 1ll, {"step between one value and the other"}}},
  };

  // AOD reader can be rate limited
  int rateLimitingIPCID = std::stoi(ctx.options().get<std::string>("timeframes-rate-limit-ipcid"));
  std::string rateLimitingChannelConfigInput;
  std::string rateLimitingChannelConfigOutput;
  bool internalRateLimiting = false;

  // In case we have rate-limiting requested, any device without an input will get one on the special
  // "DPL/RATE" message.
  if (rateLimitingIPCID >= 0) {
    rateLimitingChannelConfigInput = fmt::format("name=metric-feedback,type=pull,method=connect,address=ipc://{}metric-feedback-{},transport=shmem,rateLogging=0",
                                                 ChannelSpecHelpers::defaultIPCFolder(), rateLimitingIPCID);
    rateLimitingChannelConfigOutput = fmt::format("name=metric-feedback,type=push,method=bind,address=ipc://{}metric-feedback-{},transport=shmem,rateLogging=0",
                                                  ChannelSpecHelpers::defaultIPCFolder(), rateLimitingIPCID);
    internalRateLimiting = true;
    aodReader.options.emplace_back(ConfigParamSpec{"channel-config", VariantType::String, rateLimitingChannelConfigInput, {"how many timeframes can be in flight at the same time"}});
  }

  std::vector<InputSpec> requestedAODs;
  std::vector<OutputSpec> providedAODs;
  std::vector<InputSpec> requestedDYNs;
  std::vector<OutputSpec> providedDYNs;
  std::vector<InputSpec> requestedIDXs;

  std::vector<InputSpec> requestedCCDBs;
  std::vector<OutputSpec> providedCCDBs;
  std::vector<OutputSpec> providedOutputObjHist;

  std::vector<OutputTaskInfo> outTskMap;
  std::vector<OutputObjectInfo> outObjHistMap;

  for (size_t wi = 0; wi < workflow.size(); ++wi) {
    auto& processor = workflow[wi];
    auto name = processor.name;
    auto hash = compile_time_hash(name.c_str());
    outTskMap.push_back({hash, name});

    std::string prefix = "internal-dpl-";
    if (processor.inputs.empty() && processor.name.compare(0, prefix.size(), prefix) != 0) {
      processor.inputs.push_back(InputSpec{"enumeration", "DPL", "ENUM", static_cast<DataAllocator::SubSpecificationType>(compile_time_hash(processor.name.c_str())), Lifetime::Enumeration});
      ConfigParamsHelper::addOptionIfMissing(processor.options, ConfigParamSpec{"orbit-offset-enumeration", VariantType::Int64, 0ll, {"1st injected orbit"}});
      ConfigParamsHelper::addOptionIfMissing(processor.options, ConfigParamSpec{"orbit-multiplier-enumeration", VariantType::Int64, 0ll, {"orbits/TForbit"}});
      processor.options.push_back(ConfigParamSpec{"start-value-enumeration", VariantType::Int64, 0ll, {"initial value for the enumeration"}});
      processor.options.push_back(ConfigParamSpec{"end-value-enumeration", VariantType::Int64, -1ll, {"final value for the enumeration"}});
      processor.options.push_back(ConfigParamSpec{"step-value-enumeration", VariantType::Int64, 1ll, {"step between one value and the other"}});
    }
    bool hasTimeframeInputs = false;
    for (auto& input : processor.inputs) {
      if (input.lifetime == Lifetime::Timeframe) {
        hasTimeframeInputs = true;
        break;
      }
    }
    bool hasTimeframeOutputs = false;
    for (auto& output : processor.outputs) {
      if (output.lifetime == Lifetime::Timeframe) {
        hasTimeframeOutputs = true;
        break;
      }
    }
    // A timeframeSink consumes timeframes without creating new
    // timeframe data.
    bool timeframeSink = hasTimeframeInputs && !hasTimeframeOutputs;
    if (std::stoi(ctx.options().get<std::string>("timeframes-rate-limit-ipcid")) != -1) {
      if (timeframeSink && processor.name != "internal-dpl-injected-dummy-sink") {
        processor.outputs.push_back(OutputSpec{{"dpl-summary"}, ConcreteDataMatcher{"DPL", "SUMMARY", static_cast<DataAllocator::SubSpecificationType>(compile_time_hash(processor.name.c_str()))}});
      }
    }
    bool hasConditionOption = false;
    for (size_t ii = 0; ii < processor.inputs.size(); ++ii) {
      auto& input = processor.inputs[ii];
      switch (input.lifetime) {
        case Lifetime::Timer: {
          auto concrete = DataSpecUtils::asConcreteDataMatcher(input);
          auto hasOption = std::any_of(processor.options.begin(), processor.options.end(), [&input](auto const& option) { return (option.name == "period-" + input.binding); });
          if (hasOption == false) {
            processor.options.push_back(ConfigParamSpec{"period-" + input.binding, VariantType::Int, 1000, {"period of the timer in milliseconds"}});
          }
          timer.outputs.emplace_back(OutputSpec{concrete.origin, concrete.description, concrete.subSpec, Lifetime::Timer});
        } break;
        case Lifetime::Signal: {
          auto concrete = DataSpecUtils::asConcreteDataMatcher(input);
          timer.outputs.emplace_back(OutputSpec{concrete.origin, concrete.description, concrete.subSpec, Lifetime::Signal});
        } break;
        case Lifetime::Enumeration: {
          auto concrete = DataSpecUtils::asConcreteDataMatcher(input);
          timer.outputs.emplace_back(OutputSpec{concrete.origin, concrete.description, concrete.subSpec, Lifetime::Enumeration});
        } break;
        case Lifetime::Condition: {
          for (auto& option : processor.options) {
            if (option.name == "condition-backend") {
              hasConditionOption = true;
              break;
            }
          }
          if (hasConditionOption == false) {
            processor.options.emplace_back(ConfigParamSpec{"condition-backend", VariantType::String, defaultConditionBackend(), {"URL for CCDB"}});
            processor.options.emplace_back(ConfigParamSpec{"condition-timestamp", VariantType::Int64, 0ll, {"Force timestamp for CCDB lookup"}});
            hasConditionOption = true;
          }
          requestedCCDBs.emplace_back(input);
        } break;
        case Lifetime::OutOfBand: {
          auto concrete = DataSpecUtils::asConcreteDataMatcher(input);
          auto hasOption = std::any_of(processor.options.begin(), processor.options.end(), [&input](auto const& option) { return (option.name == "out-of-band-channel-name-" + input.binding); });
          if (hasOption == false) {
            processor.options.push_back(ConfigParamSpec{"out-of-band-channel-name-" + input.binding, VariantType::String, "out-of-band", {"channel to listen for out of band data"}});
          }
          timer.outputs.emplace_back(OutputSpec{concrete.origin, concrete.description, concrete.subSpec, Lifetime::Enumeration});
        } break;
        case Lifetime::QA:
        case Lifetime::Transient:
        case Lifetime::Timeframe:
        case Lifetime::Optional:
          break;
      }
      if (DataSpecUtils::partialMatch(input, header::DataOrigin{"AOD"})) {
        DataSpecUtils::updateInputList(requestedAODs, InputSpec{input});
      }
      if (DataSpecUtils::partialMatch(input, header::DataOrigin{"DYN"})) {
        DataSpecUtils::updateInputList(requestedDYNs, InputSpec{input});
      }
      if (DataSpecUtils::partialMatch(input, header::DataOrigin{"IDX"})) {
        DataSpecUtils::updateInputList(requestedIDXs, InputSpec{input});
      }
    }

    std::stable_sort(timer.outputs.begin(), timer.outputs.end(), [](OutputSpec const& a, OutputSpec const& b) { return *DataSpecUtils::getOptionalSubSpec(a) < *DataSpecUtils::getOptionalSubSpec(b); });

    for (auto& output : processor.outputs) {
      if (DataSpecUtils::partialMatch(output, header::DataOrigin{"AOD"})) {
        providedAODs.emplace_back(output);
      } else if (DataSpecUtils::partialMatch(output, header::DataOrigin{"DYN"})) {
        providedDYNs.emplace_back(output);
      } else if (DataSpecUtils::partialMatch(output, header::DataOrigin{"ATSK"})) {
        providedOutputObjHist.emplace_back(output);
        auto it = std::find_if(outObjHistMap.begin(), outObjHistMap.end(), [&](auto&& x) { return x.id == hash; });
        if (it == outObjHistMap.end()) {
          outObjHistMap.push_back({hash, {output.binding.value}});
        } else {
          it->bindings.push_back(output.binding.value);
        }
      }
      if (output.lifetime == Lifetime::Condition) {
        providedCCDBs.push_back(output);
      }
    }
  }

  auto inputSpecLessThan = [](InputSpec const& lhs, InputSpec const& rhs) { return DataSpecUtils::describe(lhs) < DataSpecUtils::describe(rhs); };
  auto outputSpecLessThan = [](OutputSpec const& lhs, OutputSpec const& rhs) { return DataSpecUtils::describe(lhs) < DataSpecUtils::describe(rhs); };
  std::sort(requestedDYNs.begin(), requestedDYNs.end(), inputSpecLessThan);
  std::sort(providedDYNs.begin(), providedDYNs.end(), outputSpecLessThan);
  std::vector<InputSpec> spawnerInputs;
  for (auto& input : requestedDYNs) {
    if (std::none_of(providedDYNs.begin(), providedDYNs.end(), [&input](auto const& x) { return DataSpecUtils::match(input, x); })) {
      spawnerInputs.emplace_back(input);
    }
  }

  DataProcessorSpec aodSpawner{
    "internal-dpl-aod-spawner",
    {},
    {},
    readers::AODReaderHelpers::aodSpawnerCallback(spawnerInputs),
    {}};

  DataProcessorSpec indexBuilder{
    "internal-dpl-aod-index-builder",
    {},
    {},
    readers::AODReaderHelpers::indexBuilderCallback(requestedIDXs),
    {}};

  addMissingOutputsToBuilder(requestedIDXs, requestedAODs, requestedDYNs, indexBuilder);
  addMissingOutputsToSpawner({}, spawnerInputs, requestedAODs, aodSpawner);

  addMissingOutputsToReader(providedAODs, requestedAODs, aodReader);
  addMissingOutputsToReader(providedCCDBs, requestedCCDBs, ccdbBackend);

  std::vector<DataProcessorSpec> extraSpecs;

  if (transientStore.outputs.empty() == false) {
    extraSpecs.push_back(transientStore);
  }
  if (qaStore.outputs.empty() == false) {
    extraSpecs.push_back(qaStore);
  }

  if (aodSpawner.outputs.empty() == false) {
    extraSpecs.push_back(timePipeline(aodSpawner, ctx.options().get<int64_t>("spawners")));
  }

  if (indexBuilder.outputs.empty() == false) {
    extraSpecs.push_back(indexBuilder);
  }

  // add the reader
  if (aodReader.outputs.empty() == false) {

    auto&& algo = PluginManager::loadAlgorithmFromPlugin("O2FrameworkAnalysisSupport", "ROOTFileReader");
    if (internalRateLimiting) {
      aodReader.algorithm = CommonDataProcessors::wrapWithRateLimiting(algo);
    } else {
      aodReader.algorithm = algo;
    }
    aodReader.outputs.emplace_back(OutputSpec{"TFN", "TFNumber"});
    aodReader.outputs.emplace_back(OutputSpec{"TFF", "TFFilename"});
    extraSpecs.push_back(timePipeline(aodReader, ctx.options().get<int64_t>("readers")));
    auto concrete = DataSpecUtils::asConcreteDataMatcher(aodReader.inputs[0]);
    timer.outputs.emplace_back(OutputSpec{concrete.origin, concrete.description, concrete.subSpec, Lifetime::Enumeration});
  }

  ConcreteDataMatcher dstf{"FLP", "DISTSUBTIMEFRAME", 0xccdb};
  if (ccdbBackend.outputs.empty() == false) {
    ccdbBackend.outputs.push_back(OutputSpec{"CTP", "OrbitReset", 0});
    InputSpec matcher{"dstf", "FLP", "DISTSUBTIMEFRAME", 0xccdb};
    bool providesDISTSTF = false;
    // Check if any of the provided outputs is a DISTSTF
    // Check if any of the requested inputs is for a 0xccdb message
    for (auto& dp : workflow) {
      for (auto& output : dp.outputs) {
        if (DataSpecUtils::match(matcher, output)) {
          providesDISTSTF = true;
          dstf = DataSpecUtils::asConcreteDataMatcher(output);
          break;
        }
      }
      if (providesDISTSTF) {
        break;
      }
    }
    // * If there are AOD outputs we use TFNumber as the CCDB clock
    // * If one device provides a DISTSTF we use that as the CCDB clock
    // * If one of the devices provides a timer we use that as the CCDB clock
    // * If none of the above apply add to the first data processor
    //   which has no inputs apart from enumerations the responsibility
    //   to provide the DISTSUBTIMEFRAME.
    if (aodReader.outputs.empty() == false) {
      ccdbBackend.inputs.push_back(InputSpec{"tfn", "TFN", "TFNumber"});
    } else if (providesDISTSTF) {
      ccdbBackend.inputs.push_back(InputSpec{"tfn", dstf, Lifetime::Timeframe});
    } else {
      // We find the first device which has either just enumerations or
      // just timers, and we add the DISTSUBTIMEFRAME to it.
      // Notice how we do so in a stable manner by sorting the devices
      // by name.
      int enumCandidate = -1;
      int timerCandidate = -1;
      for (size_t wi = 0; wi < workflow.size(); wi++) {
        auto& dp = workflow[wi];
        if (dp.inputs.size() != 1) {
          continue;
        }
        auto lifetime = dp.inputs[0].lifetime;
        if (lifetime == Lifetime::Enumeration && (enumCandidate == -1 || workflow[enumCandidate].name > dp.name)) {
          enumCandidate = wi;
        }
        if (lifetime == Lifetime::Timer && (timerCandidate == -1 || workflow[timerCandidate].name > dp.name)) {
          timerCandidate = wi;
        }
      }
      if (enumCandidate != -1) {
        auto& dp = workflow[enumCandidate];
        dp.outputs.push_back(OutputSpec{{"ccdb-diststf"}, dstf, Lifetime::Timeframe});
        ccdbBackend.inputs.push_back(InputSpec{"tfn", dstf, Lifetime::Timeframe});
      } else if (timerCandidate != -1) {
        auto& dp = workflow[timerCandidate];
        dstf = DataSpecUtils::asConcreteDataMatcher(dp.outputs[0]);
        ccdbBackend.inputs.push_back(InputSpec{{"tfn"}, dstf, Lifetime::Timeframe});
      }
    }

    // Load the CCDB backend from the plugin
    ccdbBackend.algorithm = PluginManager::loadAlgorithmFromPlugin("O2FrameworkCCDBSupport", "CCDBFetcherPlugin");
    extraSpecs.push_back(ccdbBackend);
  } else {
    // If there is no CCDB requested, but we still ask for a FLP/DISTSUBTIMEFRAME/0xccdb
    // we add to the first data processor which has no inputs (apart from
    // enumerations / timers) the responsibility to provide the DISTSUBTIMEFRAME
    bool requiresDISTSUBTIMEFRAME = false;
    for (auto& dp : workflow) {
      for (auto& input : dp.inputs) {
        if (DataSpecUtils::match(input, dstf)) {
          requiresDISTSUBTIMEFRAME = true;
          break;
        }
      }
    }
    if (requiresDISTSUBTIMEFRAME) {
      // We find the first device which has either just enumerations or
      // just timers, and we add the DISTSUBTIMEFRAME to it.
      // Notice how we do so in a stable manner by sorting the devices
      // by name.
      int enumCandidate = -1;
      int timerCandidate = -1;
      for (size_t wi = 0; wi < workflow.size(); wi++) {
        auto& dp = workflow[wi];
        if (dp.inputs.size() != 1) {
          continue;
        }
        auto lifetime = dp.inputs[0].lifetime;
        if (lifetime == Lifetime::Enumeration && (enumCandidate == -1 || workflow[enumCandidate].name > dp.name)) {
          enumCandidate = wi;
        }
        if (lifetime == Lifetime::Timer && (timerCandidate == -1 || workflow[timerCandidate].name > dp.name)) {
          timerCandidate = wi;
        }
      }
      if (enumCandidate != -1) {
        auto& dp = workflow[enumCandidate];
        dp.outputs.push_back(OutputSpec{{"ccdb-diststf"}, dstf, Lifetime::Timeframe});
        ccdbBackend.inputs.push_back(InputSpec{"tfn", dstf, Lifetime::Timeframe});
      } else if (timerCandidate != -1) {
        auto& dp = workflow[timerCandidate];
        dstf = DataSpecUtils::asConcreteDataMatcher(dp.outputs[0]);
        ccdbBackend.inputs.push_back(InputSpec{{"tfn"}, dstf, Lifetime::Timeframe});
      }
    }
  }

  // add the timer
  if (timer.outputs.empty() == false) {
    extraSpecs.push_back(timer);
  }

  // This is to inject a file sink so that any dangling ATSK object is written
  // to a ROOT file.
  if (providedOutputObjHist.empty() == false) {
    auto rootSink = CommonDataProcessors::getOutputObjHistSink(outObjHistMap, outTskMap);
    extraSpecs.push_back(rootSink);
  }

  workflow.insert(workflow.end(), extraSpecs.begin(), extraSpecs.end());
  extraSpecs.clear();

  /// Analyze all ouputs
  auto [outputsInputs, isDangling] = analyzeOutputs(workflow);

  // create DataOutputDescriptor
  std::shared_ptr<DataOutputDirector> dod = getDataOutputDirector(ctx.options(), outputsInputs, isDangling);

  // select outputs of type AOD which need to be saved
  // ATTENTION: if there are dangling outputs the getGlobalAODSink
  // has to be created in any case!
  std::vector<InputSpec> outputsInputsAOD;
  auto isAOD = [](InputSpec const& spec) {
    return (DataSpecUtils::partialMatch(spec, header::DataOrigin("AOD")) ||
            DataSpecUtils::partialMatch(spec, header::DataOrigin("DYN")) ||
            DataSpecUtils::partialMatch(spec, header::DataOrigin("AMD")));
  };
  for (auto ii = 0u; ii < outputsInputs.size(); ii++) {
    if (isAOD(outputsInputs[ii])) {
      auto ds = dod->getDataOutputDescriptors(outputsInputs[ii]);
      if (ds.size() > 0 || isDangling[ii]) {
        outputsInputsAOD.emplace_back(outputsInputs[ii]);
      }
    }
  }

  // file sink for any AOD output
  if (outputsInputsAOD.size() > 0) {
    // add TFNumber and TFFilename as input to the writer
    outputsInputsAOD.emplace_back(InputSpec{"tfn", "TFN", "TFNumber"});
    outputsInputsAOD.emplace_back(InputSpec{"tff", "TFF", "TFFilename"});
    auto fileSink = CommonDataProcessors::getGlobalAODSink(dod, outputsInputsAOD);
    extraSpecs.push_back(fileSink);

    auto it = std::find_if(outputsInputs.begin(), outputsInputs.end(), [](InputSpec& spec) -> bool {
      return DataSpecUtils::partialMatch(spec, o2::header::DataOrigin("TFN"));
    });
    size_t ii = std::distance(outputsInputs.begin(), it);
    isDangling[ii] = false;
  }

  workflow.insert(workflow.end(), extraSpecs.begin(), extraSpecs.end());
  extraSpecs.clear();

  // Select dangling outputs which are not of type AOD
  std::vector<InputSpec> redirectedOutputsInputs;
  for (auto ii = 0u; ii < outputsInputs.size(); ii++) {
    if (ctx.options().get<std::string>("forwarding-policy") == "none") {
      continue;
    }
    // We forward to the output proxy all the inputs only if they are dangling
    // or if the forwarding policy is "proxy".
    if (!isDangling[ii] && (ctx.options().get<std::string>("forwarding-policy") != "all")) {
      continue;
    }
    // AODs are skipped in any case.
    if (isAOD(outputsInputs[ii])) {
      continue;
    }
    redirectedOutputsInputs.emplace_back(outputsInputs[ii]);
  }

  std::vector<InputSpec> unmatched;
  auto forwardingDestination = ctx.options().get<std::string>("forwarding-destination");
  if (redirectedOutputsInputs.size() > 0 && forwardingDestination == "file") {
    auto fileSink = CommonDataProcessors::getGlobalFileSink(redirectedOutputsInputs, unmatched);
    if (unmatched.size() != redirectedOutputsInputs.size()) {
      extraSpecs.push_back(fileSink);
    }
  } else if (redirectedOutputsInputs.size() > 0 && forwardingDestination == "fairmq") {
    auto fairMQSink = CommonDataProcessors::getGlobalFairMQSink(redirectedOutputsInputs);
    extraSpecs.push_back(fairMQSink);
  } else if (forwardingDestination != "drop") {
    throw runtime_error_f("Unknown forwarding destination %s", forwardingDestination.c_str());
  }
  if (unmatched.size() > 0 || redirectedOutputsInputs.size() > 0) {
    std::vector<InputSpec> ignored = unmatched;
    ignored.insert(ignored.end(), redirectedOutputsInputs.begin(), redirectedOutputsInputs.end());
    for (auto& ignoredInput : ignored) {
      if (ignoredInput.lifetime == Lifetime::OutOfBand) {
        // FIXME: Use Lifetime::Dangling when fully working?
        ignoredInput.lifetime = Lifetime::Timeframe;
      }
    }

    extraSpecs.push_back(CommonDataProcessors::getDummySink(ignored, rateLimitingChannelConfigOutput));
  }

  workflow.insert(workflow.end(), extraSpecs.begin(), extraSpecs.end());
  extraSpecs.clear();
}

void WorkflowHelpers::adjustTopology(WorkflowSpec& workflow, ConfigContext const&)
{
  unsigned int distSTFCount = 0;
  for (auto& spec : workflow) {
    auto& inputs = spec.inputs;
    bool allSporadic = true;
    bool hasTimer = false;
    bool hasSporadic = false;
    bool hasOptionals = false;
    for (auto& input : inputs) {
      if (input.lifetime == Lifetime::Optional) {
        hasOptionals = true;
      }
    }
    for (auto& input : inputs) {
      // Any InputSpec that is DPL/DISTSUBTIMEFRAME/0 will actually be replaced by one
      // which looks like DPL/DISTSUBTIMEFRAME/<incremental number> for devices that
      // have Optional inputs as well.
      // This is done to avoid the race condition where the DISTSUBTIMEFRAME/0 gets
      // forwarded before actual RAWDATA arrives.
      if (DataSpecUtils::match(input, ConcreteDataTypeMatcher{"FLP", "DISTSUBTIMEFRAME"}) &&
          !DataSpecUtils::match(input, ConcreteDataMatcher{"FLP", "DISTSUBTIMEFRAME", 0})) {
        LOGP(error,
             "Only FLP/DISTSUBTIMEFRAME/0 is supported as input "
             "provided by the user. Please replace {} with FLP/DISTSUBTIMEFRAME/0 in {}.",
             DataSpecUtils::describe(input), input.binding);
      }
      if (hasOptionals && DataSpecUtils::match(input, ConcreteDataMatcher{"FLP", "DISTSUBTIMEFRAME", 0})) {
        // The first one remains unchanged, therefore we use the postincrement
        DataSpecUtils::updateMatchingSubspec(input, distSTFCount++);
        continue;
      }
      // Timers are sporadic only when they are not
      // alone.
      if (input.lifetime == Lifetime::Timer) {
        hasTimer = true;
        continue;
      }
      if (input.lifetime == Lifetime::Sporadic) {
        hasSporadic = true;
      } else {
        allSporadic = false;
      }
    }

    LOGP(debug, "WorkflowHelpers::adjustTopology: spec {} hasTimer {} hasSporadic {} allSporadic {}", spec.name, hasTimer, hasSporadic, allSporadic);

    // If they are not all sporadic (excluding timers)
    // we leave things as they are.
    if (allSporadic == false) {
      continue;
    }
    // A timer alone is not sporadic.
    if (hasSporadic == false) {
      continue;
    }
    /// If we get here all the inputs are sporadic and
    /// there is at least one sporadic input apart from
    /// the timers.
    for (auto& output : spec.outputs) {
      if (output.lifetime == Lifetime::Timeframe) {
        output.lifetime = Lifetime::Sporadic;
      }
    }
  }

  if (distSTFCount > 0) {
    bool found = false;
    for (auto& spec : workflow) {
      for (auto& output : spec.outputs) {
        if (DataSpecUtils::match(output, ConcreteDataMatcher{"FLP", "DISTSUBTIMEFRAME", 0})) {
          found = true;
          break;
        }
      }
      if (found) {
        for (unsigned int i = 1; i < distSTFCount; ++i) {
          spec.outputs.emplace_back(OutputSpec{ConcreteDataMatcher{"FLP", "DISTSUBTIMEFRAME", i}, Lifetime::Timeframe});
        }
        break;
      }
    }
  }
}

void WorkflowHelpers::constructGraph(const WorkflowSpec& workflow,
                                     std::vector<DeviceConnectionEdge>& logicalEdges,
                                     std::vector<OutputSpec>& outputs,
                                     std::vector<LogicalForwardInfo>& forwardedInputsInfo)
{
  assert(!workflow.empty());

  // This is the state. Oif is the iterator I use for the searches.
  std::vector<LogicalOutputInfo> availableOutputsInfo;
  auto const& constOutputs = outputs; // const version of the outputs
  // Forwards is a local cache to avoid adding forwards before time.
  std::vector<LogicalOutputInfo> forwards;

  // Notice that availableOutputsInfo MUST be updated first, since it relies on
  // the size of outputs to be the one before the update.
  auto enumerateAvailableOutputs = [&workflow, &outputs, &availableOutputsInfo]() {
    for (size_t wi = 0; wi < workflow.size(); ++wi) {
      auto& producer = workflow[wi];

      for (size_t oi = 0; oi < producer.outputs.size(); ++oi) {
        auto& out = producer.outputs[oi];
        auto uniqueOutputId = outputs.size();
        availableOutputsInfo.emplace_back(LogicalOutputInfo{wi, uniqueOutputId, false});
        outputs.push_back(out);
      }
    }
  };

  auto errorDueToMissingOutputFor = [&workflow, &constOutputs](size_t ci, size_t ii) {
    auto input = workflow[ci].inputs[ii];
    std::ostringstream str;
    str << "No matching output found for "
        << DataSpecUtils::describe(input) << " as requested by data processor \"" << workflow[ci].name << "\". Candidates:\n";

    for (auto& output : constOutputs) {
      str << "-" << DataSpecUtils::describe(output) << "\n";
    }

    throw std::runtime_error(str.str());
  };

  // This is the outer loop
  //
  // Here we iterate over dataprocessor items in workflow and we consider them
  // as consumer, since we are interested in their inputs.
  // Notice also we need to search for all the matching inputs, since
  // we could have more than one source that matches (e.g. in the
  // case of a time merger).
  // Once consumed, an output is not actually used anymore, however
  // we append it as a forward.
  // Finally, If a device has n-way pipelining, we need to create one node per
  // parallel pipeline and add an edge for each.
  enumerateAvailableOutputs();

  std::vector<bool> matches(constOutputs.size());
  for (size_t consumer = 0; consumer < workflow.size(); ++consumer) {
    for (size_t input = 0; input < workflow[consumer].inputs.size(); ++input) {
      forwards.clear();
      for (size_t i = 0; i < constOutputs.size(); i++) {
        matches[i] = DataSpecUtils::match(workflow[consumer].inputs[input], constOutputs[i]);
      }

      for (size_t i = 0; i < availableOutputsInfo.size(); i++) {
        // Notice that if the output is actually a forward, we need to store that information so that when we add it at device level we know which output channel we need to connect it too.
        if (!matches[availableOutputsInfo[i].outputGlobalIndex]) {
          continue;
        }
        auto* oif = &availableOutputsInfo[i];
        if (oif->forward) {
          forwardedInputsInfo.emplace_back(LogicalForwardInfo{consumer, input, oif->outputGlobalIndex});
        }
        auto producer = oif->specIndex;
        auto uniqueOutputId = oif->outputGlobalIndex;
        for (size_t tpi = 0; tpi < workflow[consumer].maxInputTimeslices; ++tpi) {
          for (size_t ptpi = 0; ptpi < workflow[producer].maxInputTimeslices; ++ptpi) {
            logicalEdges.emplace_back(DeviceConnectionEdge{producer, consumer, tpi, ptpi, uniqueOutputId, input, oif->forward});
          }
        }
        forwards.push_back(LogicalOutputInfo{consumer, uniqueOutputId, true});
        // We have consumed the input, therefore we remove it from the list. We will insert the forwarded inputs only at the end of the iteration.
        oif->enabled = false;
      }
      if (forwards.empty()) {
        errorDueToMissingOutputFor(consumer, input);
      }
      availableOutputsInfo.erase(std::remove_if(availableOutputsInfo.begin(), availableOutputsInfo.end(), [](auto& info) { return info.enabled == false; }), availableOutputsInfo.end());
      for (auto& forward : forwards) {
        availableOutputsInfo.push_back(forward);
      }
    }
  }
}

std::vector<EdgeAction>
  WorkflowHelpers::computeOutEdgeActions(
    const std::vector<DeviceConnectionEdge>& edges,
    const std::vector<size_t>& index)
{
  DeviceConnectionEdge last{ULONG_MAX, ULONG_MAX, ULONG_MAX, ULONG_MAX, ULONG_MAX, ULONG_MAX};

  assert(edges.size() == index.size());
  std::vector<EdgeAction> actions(edges.size(), EdgeAction{false, false});
  for (size_t i : index) {
    auto& edge = edges[i];
    auto& action = actions[i];
    action.requiresNewDevice = last.producer != edge.producer || last.producerTimeIndex != edge.producerTimeIndex;
    action.requiresNewChannel = last.consumer != edge.consumer || last.producer != edge.producer || last.timeIndex != edge.timeIndex || last.producerTimeIndex != edge.producerTimeIndex;
    last = edge;
  }
  return actions;
}

std::vector<EdgeAction>
  WorkflowHelpers::computeInEdgeActions(
    const std::vector<DeviceConnectionEdge>& edges,
    const std::vector<size_t>& index)
{
  DeviceConnectionEdge last{ULONG_MAX, ULONG_MAX, ULONG_MAX, ULONG_MAX, ULONG_MAX, ULONG_MAX};

  assert(edges.size() == index.size());
  std::vector<EdgeAction> actions(edges.size(), EdgeAction{false, false});
  for (size_t i : index) {
    auto& edge = edges[i];
    auto& action = actions[i];
    // Calculate which actions need to be taken for this edge.
    action.requiresNewDevice = last.consumer != edge.consumer || last.timeIndex != edge.timeIndex;
    action.requiresNewChannel =
      last.consumer != edge.consumer || last.timeIndex != edge.timeIndex || last.producer != edge.producer || last.producerTimeIndex != edge.producerTimeIndex;

    last = edge;
  }
  return actions;
}

void WorkflowHelpers::sortEdges(std::vector<size_t>& inEdgeIndex,
                                std::vector<size_t>& outEdgeIndex,
                                const std::vector<DeviceConnectionEdge>& edges)
{
  inEdgeIndex.resize(edges.size());
  outEdgeIndex.resize(edges.size());
  std::iota(inEdgeIndex.begin(), inEdgeIndex.end(), 0);
  std::iota(outEdgeIndex.begin(), outEdgeIndex.end(), 0);

  // Two indexes, one to bind the outputs, the other
  // one to connect the inputs. The
  auto outSorter = [&edges](size_t i, size_t j) {
    auto& a = edges[i];
    auto& b = edges[j];
    return std::tie(a.producer, a.producerTimeIndex, a.timeIndex, a.consumer) < std::tie(b.producer, b.producerTimeIndex, b.timeIndex, b.consumer);
  };
  auto inSorter = [&edges](size_t i, size_t j) {
    auto& a = edges[i];
    auto& b = edges[j];
    return std::tie(a.consumer, a.timeIndex, a.producer, a.producerTimeIndex) < std::tie(b.consumer, b.timeIndex, b.producer, b.producerTimeIndex);
  };

  std::sort(inEdgeIndex.begin(), inEdgeIndex.end(), inSorter);
  std::sort(outEdgeIndex.begin(), outEdgeIndex.end(), outSorter);
}

WorkflowParsingState WorkflowHelpers::verifyWorkflow(const o2::framework::WorkflowSpec& workflow)
{
  if (workflow.empty()) {
    return WorkflowParsingState::Empty;
  }
  std::set<std::string> validNames;
  std::vector<OutputSpec> availableOutputs;
  std::vector<InputSpec> requiredInputs;

  // An index many to one index to go from a given input to the
  // associated spec
  std::map<size_t, size_t> inputToSpec;
  // A one to one index to go from a given output to the Spec emitting it
  std::map<size_t, size_t> outputToSpec;

  std::ostringstream ss;

  for (auto& spec : workflow) {
    if (spec.name.empty()) {
      throw std::runtime_error("Invalid DataProcessorSpec name");
    }
    if (strpbrk(spec.name.data(), ",;:\"'$") != nullptr) {
      throw std::runtime_error("Cannot use any of ,;:\"'$ as DataProcessor name");
    }
    if (validNames.find(spec.name) != validNames.end()) {
      throw std::runtime_error("Name " + spec.name + " is used twice.");
    }
    validNames.insert(spec.name);
    for (auto& option : spec.options) {
      if (option.defaultValue.type() != VariantType::Empty &&
          option.type != option.defaultValue.type()) {
        ss << "Mismatch between declared option type (" << (int)option.type << ") and default value type (" << (int)option.defaultValue.type()
           << ") for " << option.name << " in DataProcessorSpec of "
           << spec.name;
        throw std::runtime_error(ss.str());
      }
    }
    for (size_t ii = 0; ii < spec.inputs.size(); ++ii) {
      InputSpec const& input = spec.inputs[ii];
      if (DataSpecUtils::validate(input) == false) {
        ss << "In spec " << spec.name << " input specification "
           << ii << " requires binding, description and origin"
                    " to be fully specified";
        throw std::runtime_error(ss.str());
      }
    }
  }
  return WorkflowParsingState::Valid;
}

using UnifiedDataSpecType = std::variant<InputSpec, OutputSpec>;
struct DataMatcherId {
  size_t workflowId;
  size_t id;
};

std::shared_ptr<DataOutputDirector> WorkflowHelpers::getDataOutputDirector(ConfigParamRegistry const& options, std::vector<InputSpec> const& OutputsInputs, std::vector<bool> const& isDangling)
{
  std::shared_ptr<DataOutputDirector> dod = std::make_shared<DataOutputDirector>();

  // analyze options and take actions accordingly
  // default values
  std::string rdn, resdir("./");
  std::string fnb, fnbase("AnalysisResults_trees");
  float mfs, maxfilesize(-1.);
  std::string fmo, filemode("RECREATE");
  int ntfm, ntfmerge = 1;

  // values from json
  if (options.isSet("aod-writer-json")) {
    auto fnjson = options.get<std::string>("aod-writer-json");
    if (!fnjson.empty()) {
      std::tie(rdn, fnb, fmo, mfs, ntfm) = dod->readJson(fnjson);
      if (!rdn.empty()) {
        resdir = rdn;
      }
      if (!fnb.empty()) {
        fnbase = fnb;
      }
      if (!fmo.empty()) {
        filemode = fmo;
      }
      if (mfs > 0.) {
        maxfilesize = mfs;
      }
      if (ntfm > 0) {
        ntfmerge = ntfm;
      }
    }
  }

  // values from command line options, information from json is overwritten
  if (options.isSet("aod-writer-resdir")) {
    rdn = options.get<std::string>("aod-writer-resdir");
    if (!rdn.empty()) {
      resdir = rdn;
    }
  }
  if (options.isSet("aod-writer-resfile")) {
    fnb = options.get<std::string>("aod-writer-resfile");
    if (!fnb.empty()) {
      fnbase = fnb;
    }
  }
  if (options.isSet("aod-writer-resmode")) {
    fmo = options.get<std::string>("aod-writer-resmode");
    if (!fmo.empty()) {
      filemode = fmo;
    }
  }
  if (options.isSet("aod-writer-maxfilesize")) {
    mfs = options.get<float>("aod-writer-maxfilesize");
    if (mfs > 0) {
      maxfilesize = mfs;
    }
  }
  if (options.isSet("aod-writer-ntfmerge")) {
    ntfm = options.get<int>("aod-writer-ntfmerge");
    if (ntfm > 0) {
      ntfmerge = ntfm;
    }
  }
  // parse the keepString
  auto isAOD = [](InputSpec const& spec) { return DataSpecUtils::partialMatch(spec, header::DataOrigin("AOD")); };
  if (options.isSet("aod-writer-keep")) {
    auto keepString = options.get<std::string>("aod-writer-keep");
    if (!keepString.empty()) {

      dod->reset();
      std::string d("dangling");
      if (d.find(keepString) == 0) {

        // use the dangling outputs
        std::vector<InputSpec> danglingOutputs;
        for (auto ii = 0u; ii < OutputsInputs.size(); ii++) {
          if (isAOD(OutputsInputs[ii]) && isDangling[ii]) {
            danglingOutputs.emplace_back(OutputsInputs[ii]);
          }
        }
        dod->readSpecs(danglingOutputs);

      } else {

        // use the keep string
        dod->readString(keepString);
      }
    }
  }
  dod->setResultDir(resdir);
  dod->setFilenameBase(fnbase);
  dod->setFileMode(filemode);
  dod->setMaximumFileSize(maxfilesize);
  dod->setNumberTimeFramesToMerge(ntfmerge);

  return dod;
}

std::tuple<std::vector<InputSpec>, std::vector<bool>> WorkflowHelpers::analyzeOutputs(WorkflowSpec const& workflow)
{
  // compute total number of input/output
  size_t totalInputs = 0;
  size_t totalOutputs = 0;
  for (auto& spec : workflow) {
    totalInputs += spec.inputs.size();
    totalOutputs += spec.outputs.size();
  }

  std::vector<DataMatcherId> inputs;
  std::vector<DataMatcherId> outputs;
  inputs.reserve(totalInputs);
  outputs.reserve(totalOutputs);

  std::vector<InputSpec> results;
  std::vector<bool> isDangling;
  results.reserve(totalOutputs);
  isDangling.reserve(totalOutputs);

  /// Prepare an index to do the iterations quickly.
  for (size_t wi = 0, we = workflow.size(); wi != we; ++wi) {
    auto& spec = workflow[wi];
    for (size_t ii = 0, ie = spec.inputs.size(); ii != ie; ++ii) {
      inputs.emplace_back(DataMatcherId{wi, ii});
    }
    for (size_t oi = 0, oe = spec.outputs.size(); oi != oe; ++oi) {
      outputs.emplace_back(DataMatcherId{wi, oi});
    }
  }

  for (size_t oi = 0, oe = outputs.size(); oi != oe; ++oi) {
    auto& output = outputs[oi];
    auto& outputSpec = workflow[output.workflowId].outputs[output.id];

    // is dangling output?
    bool matched = false;
    for (size_t ii = 0, ie = inputs.size(); ii != ie; ++ii) {
      auto& input = inputs[ii];
      // Inputs of the same workflow cannot match outputs
      if (output.workflowId == input.workflowId) {
        continue;
      }
      auto& inputSpec = workflow[input.workflowId].inputs[input.id];
      if (DataSpecUtils::match(inputSpec, outputSpec)) {
        matched = true;
        break;
      }
    }

    auto input = DataSpecUtils::matchingInput(outputSpec);
    char buf[64];
    input.binding = (snprintf(buf, 63, "output_%zu_%zu", output.workflowId, output.id), buf);

    // make sure that entries are unique
    if (std::find(results.begin(), results.end(), input) == results.end()) {
      results.emplace_back(input);
      isDangling.emplace_back(matched == false);
    }
  }

  // make sure that results is unique
  return std::make_tuple(results, isDangling);
}

std::vector<InputSpec> WorkflowHelpers::computeDanglingOutputs(WorkflowSpec const& workflow)
{

  auto [outputsInputs, isDangling] = analyzeOutputs(workflow);

  std::vector<InputSpec> results;
  for (auto ii = 0u; ii < outputsInputs.size(); ii++) {
    if (isDangling[ii]) {
      results.emplace_back(outputsInputs[ii]);
    }
  }

  return results;
}

#pragma diagnostic pop
} // namespace o2::framework
