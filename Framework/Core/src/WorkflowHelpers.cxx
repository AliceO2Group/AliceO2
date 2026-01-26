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
#include "Framework/AnalysisSupportHelpers.h"
#include "Framework/AlgorithmSpec.h"
#include "Framework/ConfigParamSpec.h"
#include "Framework/ConfigParamsHelper.h"
#include "Framework/CommonDataProcessors.h"
#include "Framework/ConfigContext.h"
#include "Framework/DeviceSpec.h"
#include "Framework/DataSpecUtils.h"
#include "Framework/DataSpecViews.h"
#include "Framework/DataAllocator.h"
#include "Framework/RawDeviceService.h"
#include "Framework/StringHelpers.h"
#include "Framework/ChannelSpecHelpers.h"
#include "Framework/PluginManager.h"
#include "Framework/DataTakingContext.h"
#include "Framework/DefaultsHelpers.h"
#include "Framework/Signpost.h"
#include "Framework/ServiceRegistryHelpers.h"

#include "Framework/Variant.h"
#include "Headers/DataHeader.h"
#include <algorithm>
#include <list>
#include <set>
#include <utility>
#include <vector>
#include <climits>
#include <numeric>

O2_DECLARE_DYNAMIC_LOG(workflow_helpers);

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
  for (auto wi = 0; static_cast<size_t>(wi) < nodeCount; ++wi) {
    index[wi] = {wi, 0};
  }
  std::vector<EdgeIndex> remainingEdgesIndex(edgesCount);
  for (EdgeIndex ei = 0; static_cast<size_t>(ei) < edgesCount; ++ei) {
    remainingEdgesIndex[ei] = ei;
  }

  // Create a vector where at each position we have true
  // if the vector has dependencies, false otherwise
  std::vector<bool> nodeDeps(nodeCount, false);
  for (EdgeIndex ei = 0; static_cast<size_t>(ei) < edgesCount; ++ei) {
    nodeDeps[*(edgeOut + ei * stride)] = true;
  }

  // We start with all those which do not have any dependencies
  // They are layer 0.
  std::list<TopoIndexInfo> L;
  for (auto ii = 0; static_cast<size_t>(ii) < index.size(); ++ii) {
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
    for (auto const& ei : remainingEdgesIndex) {
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
    for (auto const& ei : remainingEdgesIndex) {
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

// get the default value for condition-backend
std::string defaultConditionBackend()
{
  static bool explicitBackend = getenv("DPL_CONDITION_BACKEND");
  static DeploymentMode deploymentMode = DefaultsHelpers::deploymentMode();
  if (explicitBackend) {
    return getenv("DPL_CONDITION_BACKEND");
  } else if (deploymentMode == DeploymentMode::OnlineDDS || deploymentMode == DeploymentMode::OnlineECS) {
    return "http://o2-ccdb.internal";
  } else {
    return "http://alice-ccdb.cern.ch";
  }
}

// get the default value for condition query rate
int defaultConditionQueryRate()
{
  return getenv("DPL_CONDITION_QUERY_RATE") ? std::stoi(getenv("DPL_CONDITION_QUERY_RATE")) : 0;
}

// get the default value for condition query rate multiplier
int defaultConditionQueryRateMultiplier()
{
  return getenv("DPL_CONDITION_QUERY_RATE_MULTIPLIER") ? std::stoi(getenv("DPL_CONDITION_QUERY_RATE_MULTIPLIER")) : 1;
}

void WorkflowHelpers::injectServiceDevices(WorkflowSpec& workflow, ConfigContext& ctx)
{
  int rateLimitingIPCID = std::stoi(ctx.options().get<std::string>("timeframes-rate-limit-ipcid"));
  DataProcessorSpec ccdbBackend{
    .name = "internal-dpl-ccdb-backend",
    .outputs = {},
    .options = {{"condition-backend", VariantType::String, defaultConditionBackend(), {"URL for CCDB"}},
                {"condition-not-before", VariantType::Int64, 0ll, {"do not fetch from CCDB objects created before provide timestamp"}},
                {"condition-not-after", VariantType::Int64, 3385078236000ll, {"do not fetch from CCDB objects created after the timestamp"}},
                {"condition-remap", VariantType::String, "", {"remap condition path in CCDB based on the provided string."}},
                {"condition-tf-per-query", VariantType::Int, defaultConditionQueryRate(), {"check condition validity per requested number of TFs, fetch only once if <=0"}},
                {"condition-tf-per-query-multiplier", VariantType::Int, defaultConditionQueryRateMultiplier(), {"check conditions once per this amount of nominal checks (>0) or on module of TFcounter (<0)"}},
                {"condition-use-slice-for-prescaling", VariantType::Int, 0, {"use TFslice instead of TFcounter to control validation frequency. If > query rate, do not allow TFCounter excursion exceeding it"}},
                {"condition-time-tolerance", VariantType::Int64, 5000ll, {"prefer creation time if its difference to orbit-derived time exceeds threshold (ms), impose if <0"}},
                {"orbit-offset-enumeration", VariantType::Int64, 0ll, {"initial value for the orbit"}},
                {"orbit-multiplier-enumeration", VariantType::Int64, 0ll, {"multiplier to get the orbit from the counter"}},
                {"start-value-enumeration", VariantType::Int64, 0ll, {"initial value for the enumeration"}},
                {"end-value-enumeration", VariantType::Int64, -1ll, {"final value for the enumeration"}},
                {"step-value-enumeration", VariantType::Int64, 1ll, {"step between one value and the other"}}},
  };
  DataProcessorSpec analysisCCDBBackend{
    .name = "internal-dpl-aod-ccdb",
    .inputs = {},
    .outputs = {},
    .algorithm = AlgorithmSpec::dummyAlgorithm(),
    .options = {{"condition-backend", VariantType::String, defaultConditionBackend(), {"URL for CCDB"}},
                {"condition-not-before", VariantType::Int64, 0ll, {"do not fetch from CCDB objects created before provide timestamp"}},
                {"condition-not-after", VariantType::Int64, 3385078236000ll, {"do not fetch from CCDB objects created after the timestamp"}},
                {"condition-remap", VariantType::String, "", {"remap condition path in CCDB based on the provided string."}},
                {"condition-tf-per-query", VariantType::Int, defaultConditionQueryRate(), {"check condition validity per requested number of TFs, fetch only once if <=0"}},
                {"condition-tf-per-query-multiplier", VariantType::Int, defaultConditionQueryRateMultiplier(), {"check conditions once per this amount of nominal checks (>0) or on module of TFcounter (<0)"}},
                {"condition-use-slice-for-prescaling", VariantType::Int, 0, {"use TFslice instead of TFcounter to control validation frequency. If > query rate, do not allow TFCounter excursion exceeding it"}},
                {"condition-time-tolerance", VariantType::Int64, 5000ll, {"prefer creation time if its difference to orbit-derived time exceeds threshold (ms), impose if <0"}},
                {"start-value-enumeration", VariantType::Int64, 0ll, {"initial value for the enumeration"}},
                {"end-value-enumeration", VariantType::Int64, -1ll, {"final value for the enumeration"}},
                {"step-value-enumeration", VariantType::Int64, 1ll, {"step between one value and the other"}}}};
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

  DataProcessorSpec aodReader{
    .name = "internal-dpl-aod-reader",
    .inputs = {InputSpec{"enumeration",
                         "DPL",
                         "ENUM",
                         static_cast<DataAllocator::SubSpecificationType>(compile_time_hash("internal-dpl-aod-reader")),
                         aodLifetime}},
    .algorithm = AlgorithmSpec::dummyAlgorithm(),
    .options = {ConfigParamSpec{"aod-file-private", VariantType::String, ctx.options().get<std::string>("aod-file"), {"AOD file"}},
                ConfigParamSpec{"aod-max-io-rate", VariantType::Float, 0.f, {"Maximum I/O rate in MB/s"}},
                ConfigParamSpec{"aod-reader-json", VariantType::String, {"json configuration file"}},
                ConfigParamSpec{"time-limit", VariantType::Int64, 0ll, {"Maximum run time limit in seconds"}},
                ConfigParamSpec{"orbit-offset-enumeration", VariantType::Int64, 0ll, {"initial value for the orbit"}},
                ConfigParamSpec{"orbit-multiplier-enumeration", VariantType::Int64, 0ll, {"multiplier to get the orbit from the counter"}},
                ConfigParamSpec{"start-value-enumeration", VariantType::Int64, 0ll, {"initial value for the enumeration"}},
                ConfigParamSpec{"end-value-enumeration", VariantType::Int64, -1ll, {"final value for the enumeration"}},
                ConfigParamSpec{"step-value-enumeration", VariantType::Int64, 1ll, {"step between one value and the other"}}},
    .requiredServices = CommonServices::defaultServices("O2FrameworkAnalysisSupport:RunSummary")};

  ctx.services().registerService(ServiceRegistryHelpers::handleForService<DanglingEdgesContext>(new DanglingEdgesContext));
  auto& dec = ctx.services().get<DanglingEdgesContext>();

  std::vector<InputSpec> requestedCCDBs;
  std::vector<OutputSpec> providedCCDBs;

  for (size_t wi = 0; wi < workflow.size(); ++wi) {
    auto& processor = workflow[wi];
    auto name = processor.name;
    uint32_t hash = runtime_hash(name.c_str());
    dec.outTskMap.push_back({hash, name});

    std::string prefix = "internal-dpl-";
    if (processor.inputs.empty() && processor.name.compare(0, prefix.size(), prefix) != 0) {
      processor.inputs.push_back(InputSpec{"enumeration", "DPL", "ENUM", static_cast<DataAllocator::SubSpecificationType>(runtime_hash(processor.name.c_str())), Lifetime::Enumeration});
      ConfigParamsHelper::addOptionIfMissing(processor.options, ConfigParamSpec{"orbit-offset-enumeration", VariantType::Int64, 0ll, {"1st injected orbit"}});
      ConfigParamsHelper::addOptionIfMissing(processor.options, ConfigParamSpec{"orbit-multiplier-enumeration", VariantType::Int64, 0ll, {"orbits/TForbit"}});
      processor.options.push_back(ConfigParamSpec{"start-value-enumeration", VariantType::Int64, 0ll, {"initial value for the enumeration"}});
      processor.options.push_back(ConfigParamSpec{"end-value-enumeration", VariantType::Int64, -1ll, {"final value for the enumeration"}});
      processor.options.push_back(ConfigParamSpec{"step-value-enumeration", VariantType::Int64, 1ll, {"step between one value and the other"}});
    }
    bool hasTimeframeInputs = std::ranges::any_of(processor.inputs, [](auto const& input) { return input.lifetime == Lifetime::Timeframe; });
    bool hasTimeframeOutputs = std::ranges::any_of(processor.outputs, [](auto const& output) { return output.lifetime == Lifetime::Timeframe; });

    // A timeframeSink consumes timeframes without creating new
    // timeframe data.
    bool timeframeSink = hasTimeframeInputs && !hasTimeframeOutputs;
    if (rateLimitingIPCID != -1) {
      if (timeframeSink && processor.name.find("internal-dpl-injected-dummy-sink") == std::string::npos) {
        O2_SIGNPOST_ID_GENERATE(sid, workflow_helpers);
        bool hasMatch = false;
        ConcreteDataMatcher summaryMatcher = ConcreteDataMatcher{"DPL", "SUMMARY", static_cast<DataAllocator::SubSpecificationType>(hash)};
        auto summaryOutput = std::ranges::find_if(processor.outputs, [&summaryMatcher](auto const& output) { return DataSpecUtils::match(output, summaryMatcher); });
        if (summaryOutput != processor.outputs.end()) {
          O2_SIGNPOST_EVENT_EMIT(workflow_helpers, sid, "output enumeration", "%{public}s already there in %{public}s",
                                 DataSpecUtils::describe(*summaryOutput).c_str(), processor.name.c_str());
          hasMatch = true;
        }

        if (!hasMatch) {
          O2_SIGNPOST_EVENT_EMIT(workflow_helpers, sid, "output enumeration", "Adding DPL/SUMMARY/%d to %{public}s", hash, processor.name.c_str());
          processor.outputs.push_back(OutputSpec{{"dpl-summary"}, ConcreteDataMatcher{"DPL", "SUMMARY", static_cast<DataAllocator::SubSpecificationType>(hash)}});
        }
      }
    }
    bool hasConditionOption = false;
    for (size_t ii = 0; ii < processor.inputs.size(); ++ii) {
      auto& input = processor.inputs[ii];
      switch (input.lifetime) {
        case Lifetime::Timer: {
          auto concrete = DataSpecUtils::asConcreteDataMatcher(input);
          auto hasOption = std::ranges::any_of(processor.options, [&input](auto const& option) { return (option.name == "period-" + input.binding); });
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
          requestedCCDBs.emplace_back(input);
          if ((hasConditionOption == false) && std::ranges::none_of(processor.options, [](auto const& option) { return (option.name.compare("condition-backend") == 0); })) {
            processor.options.emplace_back(ConfigParamSpec{"condition-backend", VariantType::String, defaultConditionBackend(), {"URL for CCDB"}});
            processor.options.emplace_back(ConfigParamSpec{"condition-timestamp", VariantType::Int64, 0ll, {"Force timestamp for CCDB lookup"}});
            hasConditionOption = true;
          }
        } break;
        case Lifetime::OutOfBand: {
          auto concrete = DataSpecUtils::asConcreteDataMatcher(input);
          auto hasOption = std::ranges::any_of(processor.options, [&input](auto const& option) { return (option.name == "out-of-band-channel-name-" + input.binding); });
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
      if (DataSpecUtils::partialMatch(input, AODOrigins)) {
        DataSpecUtils::updateInputList(dec.requestedAODs, InputSpec{input});
      }
      if (DataSpecUtils::partialMatch(input, header::DataOrigin{"DYN"})) {
        DataSpecUtils::updateInputList(dec.requestedDYNs, InputSpec{input});
      }
      if (DataSpecUtils::partialMatch(input, header::DataOrigin{"IDX"})) {
        DataSpecUtils::updateInputList(dec.requestedIDXs, InputSpec{input});
      }
      if (DataSpecUtils::partialMatch(input, header::DataOrigin{"ATIM"})) {
        DataSpecUtils::updateInputList(dec.requestedTIMs, InputSpec{input});
      }
    }

    std::ranges::stable_sort(timer.outputs, [](OutputSpec const& a, OutputSpec const& b) { return *DataSpecUtils::getOptionalSubSpec(a) < *DataSpecUtils::getOptionalSubSpec(b); });

    for (auto& output : processor.outputs) {
      if (DataSpecUtils::partialMatch(output, AODOrigins)) {
        dec.providedAODs.emplace_back(output);
      } else if (DataSpecUtils::partialMatch(output, header::DataOrigin{"DYN"})) {
        dec.providedDYNs.emplace_back(output);
      } else if (DataSpecUtils::partialMatch(output, header::DataOrigin{"ATIM"})) {
        dec.providedTIMs.emplace_back(output);
      } else if (DataSpecUtils::partialMatch(output, header::DataOrigin{"ATSK"})) {
        dec.providedOutputObjHist.emplace_back(output);
        auto it = std::ranges::find_if(dec.outObjHistMap, [&](auto&& x) { return x.id == hash; });
        if (it == dec.outObjHistMap.end()) {
          dec.outObjHistMap.push_back({hash, {output.binding.value}});
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
  std::ranges::sort(dec.requestedDYNs, inputSpecLessThan);
  std::ranges::sort(dec.requestedTIMs, inputSpecLessThan);
  std::ranges::sort(dec.providedDYNs, outputSpecLessThan);
  std::ranges::sort(dec.providedTIMs, outputSpecLessThan);

  DataProcessorSpec indexBuilder{
    "internal-dpl-aod-index-builder",
    {},
    {},
    AlgorithmSpec::dummyAlgorithm(), // real algorithm will be set in adjustTopology
    {}};
  AnalysisSupportHelpers::addMissingOutputsToBuilder(dec.requestedIDXs, dec.requestedAODs, dec.requestedDYNs, indexBuilder);

  dec.requestedTIMs | views::filter_not_matching(dec.providedTIMs) | sinks::append_to{dec.analysisCCDBInputs};
  DeploymentMode deploymentMode = DefaultsHelpers::deploymentMode();
  if (deploymentMode != DeploymentMode::OnlineDDS && deploymentMode != DeploymentMode::OnlineECS) {
    AnalysisSupportHelpers::addMissingOutputsToBuilder(dec.analysisCCDBInputs, dec.requestedAODs, dec.requestedTIMs, analysisCCDBBackend);
  }

  dec.requestedDYNs | views::filter_not_matching(dec.providedDYNs) | sinks::append_to{dec.spawnerInputs};

  DataProcessorSpec aodSpawner{
    "internal-dpl-aod-spawner",
    {},
    {},
    AlgorithmSpec::dummyAlgorithm(), // real algorithm will be set in adjustTopology
    {}};
  AnalysisSupportHelpers::addMissingOutputsToSpawner({}, dec.spawnerInputs, dec.requestedAODs, aodSpawner);
  AnalysisSupportHelpers::addMissingOutputsToReader(dec.providedAODs, dec.requestedAODs, aodReader);

  std::ranges::sort(requestedCCDBs, inputSpecLessThan);
  std::ranges::sort(providedCCDBs, outputSpecLessThan);
  AnalysisSupportHelpers::addMissingOutputsToReader(providedCCDBs, requestedCCDBs, ccdbBackend);

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
    auto mctracks2aod = std::ranges::find_if(workflow, [](auto const& x) { return x.name == "mctracks-to-aod"; });
    if (mctracks2aod == workflow.end()) {
      // add normal reader
      aodReader.outputs.emplace_back(OutputSpec{"TFN", "TFNumber"});
      aodReader.outputs.emplace_back(OutputSpec{"TFF", "TFFilename"});
    } else {
      // AODs are being injected on-the-fly, add error-handler reader
      aodReader.algorithm = AlgorithmSpec{
        adaptStateful(
          [](DeviceSpec const& spec) {
            LOGP(warn, "Workflow with injected AODs has unsatisfied inputs:");
            for (auto const& output : spec.outputs) {
              LOGP(warn, "  {}", DataSpecUtils::describe(output.matcher));
            }
            LOGP(fatal, "Stopping.");
            // to ensure the output type for adaptStateful
            return adaptStateless([](DataAllocator&) {});
          })};
    }
    auto concrete = DataSpecUtils::asConcreteDataMatcher(aodReader.inputs[0]);
    timer.outputs.emplace_back(concrete.origin, concrete.description, concrete.subSpec, Lifetime::Enumeration);
    extraSpecs.push_back(timePipeline(aodReader, ctx.options().get<int64_t>("readers")));
  }

  InputSpec matcher{"dstf", "FLP", "DISTSUBTIMEFRAME", 0xccdb};
  auto& dstf = std::get<ConcreteDataMatcher>(matcher.matcher);
  // Check if any of the provided outputs is a DISTSTF
  // Check if any of the requested inputs is for a 0xccdb message
  bool providesDISTSTF = std::ranges::any_of(workflow,
                                             [&matcher](auto const& dp) {
                                               return std::any_of(dp.outputs.begin(), dp.outputs.end(), [&matcher](auto const& output) {
                                                 return DataSpecUtils::match(matcher, output);
                                               });
                                             });

  // If there is no CCDB requested, but we still ask for a FLP/DISTSUBTIMEFRAME/0xccdb
  // we add to the first data processor which has no inputs (apart from
  // enumerations / timers) the responsibility to provide the DISTSUBTIMEFRAME
  bool requiresDISTSUBTIMEFRAME = std::ranges::any_of(workflow,
                                                      [&dstf](auto const& dp) {
                                                        return std::any_of(dp.inputs.begin(), dp.inputs.end(), [&dstf](auto const& input) {
                                                          return DataSpecUtils::match(input, dstf);
                                                        });
                                                      });

  // We find the first device which has either just enumerations or
  // just timers, and we will add the DISTSUBTIMEFRAME to it.
  // Notice how we do so in a stable manner by sorting the devices
  // by name.
  int enumCandidate = -1;
  int timerCandidate = -1;
  for (auto wi = 0U; wi < workflow.size(); ++wi) {
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

  // * If there are AOD outputs we use TFNumber as the CCDB clock
  // * If one device provides a DISTSTF we use that as the CCDB clock
  // * If one of the devices provides a timer we use that as the CCDB clock
  // * If none of the above apply, add to the first data processor
  //   which has no inputs apart from enumerations the responsibility
  //   to provide the DISTSUBTIMEFRAME.
  if (ccdbBackend.outputs.empty() == false) {
    if (aodReader.outputs.empty() == false) {
      // fetcher clock follows AOD source (TFNumber)
      ccdbBackend.inputs.push_back(InputSpec{"tfn", "TFN", "TFNumber"});
    } else if (providesDISTSTF) {
      // fetcher clock follows DSTF/ccdb source (DISTSUBTIMEFRAME)
      ccdbBackend.inputs.push_back(InputSpec{"tfn", dstf, Lifetime::Timeframe});
    } else {
      if (enumCandidate != -1) {
        // add DSTF/ccdb source to the enumeration-driven source explicitly
        // fetcher clock is provided by enumeration-driven source (DISTSUBTIMEFRAME)
        DataSpecUtils::updateOutputList(workflow[enumCandidate].outputs, OutputSpec{{"ccdb-diststf"}, dstf, Lifetime::Timeframe});
        ccdbBackend.inputs.push_back(InputSpec{"tfn", dstf, Lifetime::Timeframe});
      } else if (timerCandidate != -1) {
        // fetcher clock is proived by timer source
        auto timer_dstf = DataSpecUtils::asConcreteDataMatcher(workflow[timerCandidate].outputs[0]);
        ccdbBackend.inputs.push_back(InputSpec{"tfn", timer_dstf, Lifetime::Timeframe});
      }
    }

    ccdbBackend.outputs.push_back(OutputSpec{"CTP", "OrbitReset", 0});
    // Load the CCDB backend from the plugin
    ccdbBackend.algorithm = PluginManager::loadAlgorithmFromPlugin("O2FrameworkCCDBSupport", "CCDBFetcherPlugin", ctx);
    extraSpecs.push_back(ccdbBackend);
  } else if (requiresDISTSUBTIMEFRAME && enumCandidate != -1) {
    // add DSTF/ccdb source to the enumeration-driven source explicitly if it is required in the workflow
    DataSpecUtils::updateOutputList(workflow[enumCandidate].outputs, OutputSpec{{"ccdb-diststf"}, dstf, Lifetime::Timeframe});
  }

  // add the Analysys CCDB backend which reads CCDB objects using a provided table
  if (analysisCCDBBackend.outputs.empty() == false) {
    extraSpecs.push_back(analysisCCDBBackend);
  }

  // add the timer
  if (timer.outputs.empty() == false) {
    extraSpecs.push_back(timer);
  }

  // This is to inject a file sink so that any dangling ATSK object is written
  // to a ROOT file.
  if (dec.providedOutputObjHist.empty() == false) {
    auto rootSink = AnalysisSupportHelpers::getOutputObjHistSink(ctx);
    extraSpecs.push_back(rootSink);
  }

  workflow.insert(workflow.end(), extraSpecs.begin(), extraSpecs.end());
  extraSpecs.clear();

  injectAODWriter(workflow, ctx);

  // Select dangling outputs which are not of type AOD
  std::vector<InputSpec> redirectedOutputsInputs;
  for (auto ii = 0u; ii < dec.outputsInputs.size(); ii++) {
    if (ctx.options().get<std::string>("forwarding-policy") == "none") {
      continue;
    }
    // We forward to the output proxy all the inputs only if they are dangling
    // or if the forwarding policy is "proxy".
    if (!dec.isDangling[ii] && (ctx.options().get<std::string>("forwarding-policy") != "all")) {
      continue;
    }
    // AODs are skipped in any case.
    if (DataSpecUtils::partialMatch(dec.outputsInputs[ii], extendedAODOrigins)) {
      continue;
    }
    redirectedOutputsInputs.emplace_back(dec.outputsInputs[ii]);
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
      ignoredInput.lifetime = Lifetime::Sporadic;
    }

    // Use the new dummy sink when the AOD reader is there
    O2_SIGNPOST_ID_GENERATE(sid, workflow_helpers);
    if (aodReader.outputs.empty() == false) {
      O2_SIGNPOST_EVENT_EMIT(workflow_helpers, sid, "injectServiceDevices", "Injecting scheduled dummy sink");
      extraSpecs.push_back(CommonDataProcessors::getScheduledDummySink(ignored));
    } else {
      O2_SIGNPOST_EVENT_EMIT(workflow_helpers, sid, "injectServiceDevices", "Injecting rate limited dummy sink");
      std::string rateLimitingChannelConfigOutput;
      if (rateLimitingIPCID != -1) {
        rateLimitingChannelConfigOutput = fmt::format("name=metric-feedback,type=push,method=bind,address=ipc://{}metric-feedback-{},transport=shmem,rateLogging=0", ChannelSpecHelpers::defaultIPCFolder(), rateLimitingIPCID);
      }
      extraSpecs.push_back(CommonDataProcessors::getDummySink(ignored, rateLimitingChannelConfigOutput));
    }
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
    for (auto& spec : workflow) {
      if (std::ranges::any_of(spec.outputs, [](auto const& output) { return DataSpecUtils::match(output, ConcreteDataMatcher{"FLP", "DISTSUBTIMEFRAME", 0}); })) {
        for (unsigned int i = 1; i < distSTFCount; ++i) {
          spec.outputs.emplace_back(OutputSpec{ConcreteDataMatcher{"FLP", "DISTSUBTIMEFRAME", i}, Lifetime::Timeframe});
        }
        break;
      }
    }
  }
}

void WorkflowHelpers::injectAODWriter(WorkflowSpec& workflow, ConfigContext const& ctx)
{
  auto& dec = ctx.services().get<DanglingEdgesContext>();
  /// Analyze all ouputs
  std::tie(dec.outputsInputs, dec.isDangling) = analyzeOutputs(workflow);

  // create DataOutputDescriptor
  std::shared_ptr<DataOutputDirector> dod = AnalysisSupportHelpers::getDataOutputDirector(ctx);

  // select outputs of type AOD which need to be saved
  dec.outputsInputsAOD.clear();
  for (auto ii = 0u; ii < dec.outputsInputs.size(); ii++) {
    if (DataSpecUtils::partialMatch(dec.outputsInputs[ii], extendedAODOrigins)) {
      auto ds = dod->getDataOutputDescriptors(dec.outputsInputs[ii]);
      if (ds.size() > 0 || dec.isDangling[ii]) {
        dec.outputsInputsAOD.emplace_back(dec.outputsInputs[ii]);
      }
    }
  }

  // file sink for any AOD output
  if (dec.outputsInputsAOD.size() > 0) {
    // add TFNumber and TFFilename as input to the writer
    DataSpecUtils::updateInputList(dec.outputsInputsAOD, InputSpec{"tfn", "TFN", "TFNumber"});
    DataSpecUtils::updateInputList(dec.outputsInputsAOD, InputSpec{"tff", "TFF", "TFFilename"});
    auto fileSink = AnalysisSupportHelpers::getGlobalAODSink(ctx);
    workflow.push_back(fileSink);

    auto it = std::find_if(dec.outputsInputs.begin(), dec.outputsInputs.end(), [](InputSpec const& spec) -> bool {
      return DataSpecUtils::partialMatch(spec, o2::header::DataOrigin("TFN"));
    });
    dec.isDangling[std::distance(dec.outputsInputs.begin(), it)] = false;
  }
}

void WorkflowHelpers::constructGraph(const WorkflowSpec& workflow,
                                     std::vector<DeviceConnectionEdge>& logicalEdges,
                                     std::vector<OutputSpec>& outputs,
                                     std::vector<LogicalForwardInfo>& forwardedInputsInfo)
{
  // In case the workflow is empty, we do not have anything to do.
  if (workflow.empty()) {
    return;
  }

  // This is the state. Oif is the iterator I use for the searches.
  std::vector<LogicalOutputInfo> availableOutputsInfo;
  auto const& constOutputs = outputs; // const version of the outputs
  // Forwards is a local cache to avoid adding forwards before time.
  std::vector<LogicalOutputInfo> forwards;

  // Notice that availableOutputsInfo MUST be updated first, since it relies on
  // the size of outputs to be the one before the update.
  auto enumerateAvailableOutputs = [&workflow, &outputs, &availableOutputsInfo]() {
    O2_SIGNPOST_ID_GENERATE(sid, workflow_helpers);
    for (size_t wi = 0; wi < workflow.size(); ++wi) {
      auto& producer = workflow[wi];
      if (producer.outputs.empty()) {
        O2_SIGNPOST_EVENT_EMIT(workflow_helpers, sid, "output enumeration", "No outputs for [%zu] %{public}s", wi, producer.name.c_str());
      }
      O2_SIGNPOST_START(workflow_helpers, sid, "output enumeration", "Enumerating outputs for producer [%zu] %{}s public", wi, producer.name.c_str());

      for (size_t oi = 0; oi < producer.outputs.size(); ++oi) {
        auto& out = producer.outputs[oi];
        auto uniqueOutputId = outputs.size();
        availableOutputsInfo.emplace_back(LogicalOutputInfo{wi, uniqueOutputId, false});
        O2_SIGNPOST_EVENT_EMIT(workflow_helpers, sid, "output enumeration", "- [%zu, %zu] %{public}s",
                               oi, uniqueOutputId, DataSpecUtils::describe(out).c_str());
        outputs.push_back(out);
      }
      O2_SIGNPOST_END(workflow_helpers, sid, "output enumeration", "");
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
    O2_SIGNPOST_ID_GENERATE(sid, workflow_helpers);
    O2_SIGNPOST_START(workflow_helpers, sid, "input matching", "Matching inputs of consumer [%zu] %{}s public", consumer, workflow[consumer].name.c_str());
    for (size_t input = 0; input < workflow[consumer].inputs.size(); ++input) {
      forwards.clear();
      for (size_t i = 0; i < constOutputs.size(); i++) {
        matches[i] = DataSpecUtils::match(workflow[consumer].inputs[input], constOutputs[i]);
        if (matches[i]) {
          O2_SIGNPOST_EVENT_EMIT(workflow_helpers, sid, "output", "Input %{public}s matches %{public}s",
                                 DataSpecUtils::describe(workflow[consumer].inputs[input]).c_str(),
                                 DataSpecUtils::describe(constOutputs[i]).c_str());
        }
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
            O2_SIGNPOST_EVENT_EMIT(workflow_helpers, sid, "output", "Adding edge between %{public}s and %{public}s", workflow[consumer].name.c_str(),
                                   workflow[producer].name.c_str());
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
    O2_SIGNPOST_END(workflow_helpers, sid, "input matching", "");
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
                    " to be fully specified (found "
           << input.binding << ":" << DataSpecUtils::describe(input) << ")";
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
    if (std::ranges::find(results, input) == results.end()) {
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

bool validateLifetime(std::ostream& errors,
                      DataProcessorSpec const& producer, OutputSpec const& output, DataProcessorPoliciesInfo const& producerPolicies,
                      DataProcessorSpec const& consumer, InputSpec const& input, DataProcessorPoliciesInfo const& consumerPolicies)
{
  // In case the completion policy is consume-any, we do not need to check anything.
  if (consumerPolicies.completionPolicyName == "consume-any") {
    return true;
  }
  if (input.lifetime == Lifetime::Timeframe && output.lifetime == Lifetime::Sporadic) {
    errors << fmt::format("Input {} of {} has lifetime Timeframe, but output {} of {} has lifetime Sporadic\n",
                          DataSpecUtils::describe(input).c_str(), consumer.name,
                          DataSpecUtils::describe(output).c_str(), producer.name);
    return false;
  }
  return true;
}

bool validateExpendable(std::ostream& errors,
                        DataProcessorSpec const& producer, OutputSpec const& output, DataProcessorPoliciesInfo const& producerPolicies,
                        DataProcessorSpec const& consumer, InputSpec const& input, DataProcessorPoliciesInfo const& consumerPolicies)
{
  auto isExpendable = [](DataProcessorLabel const& label) {
    return label.value == "expendable";
  };
  auto isResilient = [](DataProcessorLabel const& label) {
    return label.value == "expendable" || label.value == "resilient";
  };
  bool producerExpendable = std::find_if(producer.labels.begin(), producer.labels.end(), isExpendable) != producer.labels.end();
  bool consumerCritical = std::find_if(consumer.labels.begin(), consumer.labels.end(), isResilient) == consumer.labels.end();
  if (producerExpendable && consumerCritical) {
    errors << fmt::format("Critical consumer {} depends on expendable producer {}\n",
                          consumer.name,
                          producer.name);
    return false;
  }
  return true;
}

using Validator = std::function<bool(std::ostream& errors,
                                     DataProcessorSpec const& producer, OutputSpec const& output, DataProcessorPoliciesInfo const& producerPolicies,
                                     DataProcessorSpec const& consumer, InputSpec const& input, DataProcessorPoliciesInfo const& consumerPolicies)>;

void WorkflowHelpers::validateEdges(WorkflowSpec const& workflow,
                                    std::vector<DataProcessorPoliciesInfo> const& policies,
                                    std::vector<DeviceConnectionEdge> const& edges,
                                    std::vector<OutputSpec> const& outputs)
{
  static bool disableLifetimeCheck = getenv("DPL_WORKAROUND_DO_NOT_CHECK_FOR_CORRECT_WORKFLOW_LIFETIMES") && atoi(getenv("DPL_WORKAROUND_DO_NOT_CHECK_FOR_CORRECT_WORKFLOW_LIFETIMES"));
  std::vector<Validator> defaultValidators = {validateExpendable};
  if (!disableLifetimeCheck) {
    defaultValidators.emplace_back(validateLifetime);
  }
  std::stringstream errors;
  // Iterate over all the edges.
  // Get the input lifetime and the output lifetime.
  // Output lifetime must be Timeframe if the input lifetime is Timeframe.
  bool hasErrors = false;
  for (auto& edge : edges) {
    DataProcessorSpec const& producer = workflow[edge.producer];
    DataProcessorSpec const& consumer = workflow[edge.consumer];
    DataProcessorPoliciesInfo const& producerPolicies = policies[edge.producer];
    DataProcessorPoliciesInfo const& consumerPolicies = policies[edge.consumer];
    OutputSpec const& output = outputs[edge.outputGlobalIndex];
    InputSpec const& input = consumer.inputs[edge.consumerInputIndex];
    for (auto& validator : defaultValidators) {
      hasErrors |= !validator(errors, producer, output, producerPolicies, consumer, input, consumerPolicies);
    }
  }
  if (hasErrors) {
    throw std::runtime_error(errors.str());
  }
}

} // namespace o2::framework
