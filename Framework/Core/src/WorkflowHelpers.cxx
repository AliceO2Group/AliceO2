// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include "WorkflowHelpers.h"
#include "Framework/AlgorithmSpec.h"
#include "Framework/AODReaderHelpers.h"
#include "Framework/ChannelMatching.h"
#include "Framework/CommonDataProcessors.h"
#include "Framework/ConfigContext.h"
#include "Framework/DeviceSpec.h"
#include "Framework/DataSpecUtils.h"
#include "Framework/DataAllocator.h"
#include "Framework/ControlService.h"
#include "Framework/RawDeviceService.h"
#include "Framework/StringHelpers.h"
#include "Framework/CommonMessageBackends.h"
#include "Framework/ExternalFairMQDeviceProxy.h"
#include "Framework/Plugins.h"

#include "Headers/DataHeader.h"
#include <algorithm>
#include <list>
#include <set>
#include <utility>
#include <vector>
#include <climits>
#include <thread>

namespace o2
{
namespace framework
{

std::ostream& operator<<(std::ostream& out, TopoIndexInfo const& info)
{
  out << "(" << info.index << ", " << info.layer << ")";
  return out;
}

enum OutputType : char {
  UNKNOWN = 0,
  DANGLING = 1,
  ANALYSIS = 2,
};

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

void addMissingOutputsToReader(std::vector<OutputSpec> const& providedOutputs,
                               std::vector<InputSpec> requestedInputs,
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
    publisher.outputs.emplace_back(OutputSpec{concrete.origin, concrete.description, concrete.subSpec});
  }
}

void addMissingOutputsToSpawner(std::vector<InputSpec>&& requestedDYNs,
                                std::vector<InputSpec>& requestedAODs,
                                DataProcessorSpec& publisher)
{
  for (auto& input : requestedDYNs) {
    publisher.inputs.emplace_back(InputSpec{input.binding, header::DataOrigin{"AOD"}, DataSpecUtils::asConcreteDataMatcher(input).description});
    requestedAODs.emplace_back(InputSpec{input.binding, header::DataOrigin{"AOD"}, DataSpecUtils::asConcreteDataMatcher(input).description});
    auto concrete = DataSpecUtils::asConcreteDataMatcher(input);
    publisher.outputs.emplace_back(OutputSpec{concrete.origin, concrete.description, concrete.subSpec});
  }
}

void addMissingOutputsToBuilder(std::vector<InputSpec>&& requestedIDXs,
                                std::vector<InputSpec>& requestedAODs,
                                DataProcessorSpec& publisher)
{
  auto inputSpecFromString = [](std::string s) {
    std::regex word_regex("(\\w+)");
    auto words = std::sregex_iterator(s.begin(), s.end(), word_regex);
    if (std::distance(words, std::sregex_iterator()) != 3) {
      throw runtime_error_f("Malformed spec: %s", s.c_str());
    }
    std::vector<std::string> data;
    for (auto i = words; i != std::sregex_iterator(); ++i) {
      data.emplace_back(i->str());
    }
    char origin[4];
    char description[16];
    std::memcpy(&origin, data[1].c_str(), 4);
    std::memcpy(&description, data[2].c_str(), 16);
    return InputSpec{data[0], header::DataOrigin{origin}, header::DataDescription{description}};
  };

  for (auto& input : requestedIDXs) {
    auto concrete = DataSpecUtils::asConcreteDataMatcher(input);
    publisher.outputs.emplace_back(OutputSpec{concrete.origin, concrete.description, concrete.subSpec});
    for (auto& i : input.metadata) {
      auto spec = inputSpecFromString(i.defaultValue.get<std::string>());
      auto j = std::find_if(publisher.inputs.begin(), publisher.inputs.end(), [&](auto x) { return x.binding == spec.binding; });
      if (j == publisher.inputs.end()) {
        publisher.inputs.push_back(spec);
      }
      requestedAODs.push_back(spec);
    }
  }
}

void WorkflowHelpers::injectServiceDevices(WorkflowSpec& workflow, ConfigContext const& ctx)
{
  auto fakeCallback = AlgorithmSpec{[](InitContext& ic) {
    LOG(INFO) << "This is not a real device, merely a placeholder for external inputs";
    LOG(INFO) << "To be hidden / removed at some point.";
    // mark this dummy process as ready-to-quit
    ic.services().get<ControlService>().readyToQuit(QuitRequest::Me);

    return [](ProcessingContext& pc) {
      // this callback is never called since there is no expiring input
      pc.services().get<RawDeviceService>().waitFor(2000);
    };
  }};

  DataProcessorSpec ccdbBackend{
    "internal-dpl-ccdb-backend",
    {},
    {},
    AlgorithmSpec::dummyAlgorithm()};
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
  auto readerServices = CommonServices::defaultServices();
  readerServices.push_back(CommonMessageBackends::rateLimitingSpec());

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
     ConfigParamSpec{"time-limit", VariantType::Int64, 0ll, {"Maximum run time limit in seconds"}},
     ConfigParamSpec{"start-value-enumeration", VariantType::Int64, 0ll, {"initial value for the enumeration"}},
     ConfigParamSpec{"end-value-enumeration", VariantType::Int64, -1ll, {"final value for the enumeration"}},
     ConfigParamSpec{"step-value-enumeration", VariantType::Int64, 1ll, {"step between one value and the other"}}},
    readerServices};

  std::vector<InputSpec> requestedAODs;
  std::vector<OutputSpec> providedAODs;
  std::vector<InputSpec> requestedDYNs;
  std::vector<InputSpec> requestedIDXs;

  std::vector<InputSpec> requestedCCDBs;
  std::vector<OutputSpec> providedCCDBs;
  std::vector<OutputSpec> providedOutputObjHist;

  outputTasks outTskMap;
  outputObjects outObjHistMap;

  for (size_t wi = 0; wi < workflow.size(); ++wi) {
    auto& processor = workflow[wi];
    auto name = processor.name;
    auto hash = compile_time_hash(name.c_str());
    outTskMap.push_back({hash, name});

    std::string prefix = "internal-dpl-";
    if (processor.inputs.empty() && processor.name.compare(0, prefix.size(), prefix) != 0) {
      processor.inputs.push_back(InputSpec{"enumeration", "DPL", "ENUM", static_cast<DataAllocator::SubSpecificationType>(compile_time_hash(processor.name.c_str())), Lifetime::Enumeration});
      processor.options.push_back(ConfigParamSpec{"start-value-enumeration", VariantType::Int64, 0ll, {"initial value for the enumeration"}});
      processor.options.push_back(ConfigParamSpec{"end-value-enumeration", VariantType::Int64, -1ll, {"final value for the enumeration"}});
      processor.options.push_back(ConfigParamSpec{"step-value-enumeration", VariantType::Int64, 1ll, {"step between one value and the other"}});
    }
    bool hasConditionOption = false;
    for (size_t ii = 0; ii < processor.inputs.size(); ++ii) {
      auto& input = processor.inputs[ii];
      switch (input.lifetime) {
        case Lifetime::Timer: {
          auto concrete = DataSpecUtils::asConcreteDataMatcher(input);
          auto hasOption = std::any_of(processor.options.begin(), processor.options.end(), [&input](auto const& option) { return (option.name == "period-" + input.binding); });
          if (hasOption == false) {
            processor.options.push_back(ConfigParamSpec{"period-" + input.binding, VariantType::Int, 1000, {"period of the timer"}});
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
          if (hasConditionOption == false) {
            processor.options.emplace_back(ConfigParamSpec{"condition-backend", VariantType::String, "http://localhost:8080", {"URL for CCDB"}});
            processor.options.emplace_back(ConfigParamSpec{"condition-timestamp", VariantType::String, "", {"Force timestamp for CCDB lookup"}});
            hasConditionOption = true;
          }
          requestedCCDBs.emplace_back(input);
        } break;
        case Lifetime::QA:
        case Lifetime::Transient:
        case Lifetime::Timeframe:
        case Lifetime::Optional:
          break;
      }
      if (DataSpecUtils::partialMatch(input, header::DataOrigin{"AOD"})) {
        requestedAODs.emplace_back(input);
      }
      if (DataSpecUtils::partialMatch(input, header::DataOrigin{"RN2"})) {
        requestedAODs.emplace_back(input);
      }
      if (DataSpecUtils::partialMatch(input, header::DataOrigin{"DYN"})) {
        if (std::find_if(requestedDYNs.begin(), requestedDYNs.end(), [&](InputSpec const& spec) { return input.binding == spec.binding; }) == requestedDYNs.end()) {
          requestedDYNs.emplace_back(input);
        }
      }
      if (DataSpecUtils::partialMatch(input, header::DataOrigin{"IDX"})) {
        if (std::find_if(requestedIDXs.begin(), requestedIDXs.end(), [&](InputSpec const& spec) { return input.binding == spec.binding; }) == requestedIDXs.end()) {
          requestedIDXs.emplace_back(input);
        }
      }
    }

    std::stable_sort(timer.outputs.begin(), timer.outputs.end(), [](OutputSpec const& a, OutputSpec const& b) { return *DataSpecUtils::getOptionalSubSpec(a) < *DataSpecUtils::getOptionalSubSpec(b); });

    for (size_t oi = 0; oi < processor.outputs.size(); ++oi) {
      auto& output = processor.outputs[oi];
      if (DataSpecUtils::partialMatch(output, header::DataOrigin{"AOD"})) {
        providedAODs.emplace_back(output);
      } else if (DataSpecUtils::partialMatch(output, header::DataOrigin{"RN2"})) {
        providedAODs.emplace_back(output);
      } else if (DataSpecUtils::partialMatch(output, header::DataOrigin{"ATSK"})) {
        providedOutputObjHist.emplace_back(output);
        auto it = std::find_if(outObjHistMap.begin(), outObjHistMap.end(), [&](auto&& x) { return x.first == hash; });
        if (it == outObjHistMap.end()) {
          outObjHistMap.push_back({hash, {output.binding.value}});
        } else {
          it->second.push_back(output.binding.value);
        }
      }
      if (output.lifetime == Lifetime::Condition) {
        providedCCDBs.push_back(output);
      }
    }
  }
  auto sortingEquals = [](InputSpec const& a, InputSpec const& b) { return DataSpecUtils::describe(a) == DataSpecUtils::describe(b); };
  std::sort(requestedDYNs.begin(), requestedDYNs.end(), sortingEquals);
  auto last = std::unique(requestedDYNs.begin(), requestedDYNs.end());
  requestedDYNs.erase(last, requestedDYNs.end());

  std::sort(requestedIDXs.begin(), requestedIDXs.end(), sortingEquals);
  last = std::unique(requestedIDXs.begin(), requestedIDXs.end());
  requestedIDXs.erase(last, requestedIDXs.end());

  DataProcessorSpec aodSpawner{
    "internal-dpl-aod-spawner",
    {},
    {},
    readers::AODReaderHelpers::aodSpawnerCallback(requestedDYNs),
    {}};

  DataProcessorSpec indexBuilder{
    "internal-dpl-aod-index-builder",
    {},
    {},
    readers::AODReaderHelpers::indexBuilderCallback(requestedIDXs),
    {}};

  addMissingOutputsToSpawner(std::move(requestedDYNs), requestedAODs, aodSpawner);
  addMissingOutputsToBuilder(std::move(requestedIDXs), requestedAODs, indexBuilder);

  addMissingOutputsToReader(providedAODs, requestedAODs, aodReader);
  addMissingOutputsToReader(providedCCDBs, requestedCCDBs, ccdbBackend);

  std::vector<DataProcessorSpec> extraSpecs;

  if (ccdbBackend.outputs.empty() == false) {
    extraSpecs.push_back(ccdbBackend);
  }
  if (transientStore.outputs.empty() == false) {
    extraSpecs.push_back(transientStore);
  }
  if (qaStore.outputs.empty() == false) {
    extraSpecs.push_back(qaStore);
  }

  if (aodSpawner.outputs.empty() == false) {
    extraSpecs.push_back(aodSpawner);
  }

  if (indexBuilder.outputs.empty() == false) {
    extraSpecs.push_back(indexBuilder);
  }

  // add the reader
  if (aodReader.outputs.empty() == false) {
    uv_lib_t supportLib;
    int result = 0;
#ifdef __APPLE__
    result = uv_dlopen("libO2FrameworkAnalysisSupport.dylib", &supportLib);
#else
    result = uv_dlopen("libO2FrameworkAnalysisSupport.so", &supportLib);
#endif
    if (result == -1) {
      LOG(FATAL) << uv_dlerror(&supportLib);
      return;
    }
    void* callback = nullptr;
    DPLPluginHandle* (*dpl_plugin_callback)(DPLPluginHandle*);

    result = uv_dlsym(&supportLib, "dpl_plugin_callback", (void**)&dpl_plugin_callback);
    if (result == -1) {
      LOG(FATAL) << uv_dlerror(&supportLib);
      return;
    }
    if (dpl_plugin_callback == nullptr) {
      LOG(FATAL) << "Could not find the AnalysisSupport plugin.";
      return;
    }
    DPLPluginHandle* pluginInstance = dpl_plugin_callback(nullptr);
    AlgorithmPlugin* creator = PluginManager::getByName<AlgorithmPlugin>(pluginInstance, "ROOTFileReader");
    aodReader.algorithm = creator->create();
    aodReader.outputs.emplace_back(OutputSpec{"TFN", "TFNumber"});
    extraSpecs.push_back(timePipeline(aodReader, ctx.options().get<int64_t>("readers")));
    auto concrete = DataSpecUtils::asConcreteDataMatcher(aodReader.inputs[0]);
    timer.outputs.emplace_back(OutputSpec{concrete.origin, concrete.description, concrete.subSpec, Lifetime::Enumeration});
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
  //  outputTypes = isAOD*2 + isdangling*1 + 0
  auto [outputsInputs, outputTypes] = analyzeOutputs(workflow);

  // create DataOutputDescriptor
  std::shared_ptr<DataOutputDirector> dod = getDataOutputDirector(ctx.options(), outputsInputs, outputTypes);

  // select outputs of type AOD which need to be saved
  // ATTENTION: if there are dangling outputs the getGlobalAODSink
  // has to be created in any case!
  std::vector<InputSpec> outputsInputsAOD;
  for (auto ii = 0u; ii < outputsInputs.size(); ii++) {
    if ((outputTypes[ii] & ANALYSIS) == ANALYSIS) {
      auto ds = dod->getDataOutputDescriptors(outputsInputs[ii]);
      if (ds.size() > 0 || (outputTypes[ii] & DANGLING) == DANGLING) {
        outputsInputsAOD.emplace_back(outputsInputs[ii]);
      }
    }
  }

  // file sink for any AOD output
  if (outputsInputsAOD.size() > 0) {
    // add TFNumber as input to the writer
    outputsInputsAOD.emplace_back(InputSpec{"tfn", "TFN", "TFNumber"});
    auto fileSink = CommonDataProcessors::getGlobalAODSink(dod, outputsInputsAOD);
    extraSpecs.push_back(fileSink);
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
    if (!(outputTypes[ii] & DANGLING) && (ctx.options().get<std::string>("forwarding-policy") != "all")) {
      continue;
    }
    // AODs are skipped in any case.
    if ((outputTypes[ii] & ANALYSIS)) {
      continue;
    }
    redirectedOutputsInputs.emplace_back(outputsInputs[ii]);
  }

  std::vector<InputSpec> unmatched;
  if (redirectedOutputsInputs.size() > 0 && ctx.options().get<std::string>("forwarding-destination") == "file") {
    auto fileSink = CommonDataProcessors::getGlobalFileSink(redirectedOutputsInputs, unmatched);
    if (unmatched.size() != redirectedOutputsInputs.size()) {
      extraSpecs.push_back(fileSink);
    }
  } else if (redirectedOutputsInputs.size() > 0 && ctx.options().get<std::string>("forwarding-destination") == "fairmq") {
    auto fairMQSink = CommonDataProcessors::getGlobalFairMQSink(redirectedOutputsInputs);
    extraSpecs.push_back(fairMQSink);
  }
  if (unmatched.size() > 0) {
    extraSpecs.push_back(CommonDataProcessors::getDummySink(unmatched));
  }

  workflow.insert(workflow.end(), extraSpecs.begin(), extraSpecs.end());
  extraSpecs.clear();
}

void WorkflowHelpers::constructGraph(const WorkflowSpec& workflow,
                                     std::vector<DeviceConnectionEdge>& logicalEdges,
                                     std::vector<OutputSpec>& outputs,
                                     std::vector<LogicalForwardInfo>& forwardedInputsInfo)
{
  assert(!workflow.empty());

  // This is the state. Oif is the iterator I use for the searches.
  std::list<LogicalOutputInfo> availableOutputsInfo;
  auto const& constOutputs = outputs; // const version of the outputs
  decltype(availableOutputsInfo.begin()) oif;
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

  // Notice that if the output is actually a forward, we need to store that
  // information so that when we add it at device level we know which output
  // channel we need to connect it too.
  auto hasMatchingOutputFor = [&workflow, &constOutputs,
                               &availableOutputsInfo, &oif,
                               &forwardedInputsInfo](size_t ci, size_t ii) {
    assert(ci < workflow.size());
    assert(ii < workflow[ci].inputs.size());
    auto& input = workflow[ci].inputs[ii];
    auto matcher = [&input, &constOutputs](const LogicalOutputInfo& outputInfo) -> bool {
      auto& output = constOutputs[outputInfo.outputGlobalIndex];
      return DataSpecUtils::match(input, output);
    };
    oif = std::find_if(availableOutputsInfo.begin(),
                       availableOutputsInfo.end(),
                       matcher);
    if (oif != availableOutputsInfo.end() && oif->forward) {
      LogicalForwardInfo forward;
      forward.consumer = ci;
      forward.inputLocalIndex = ii;
      forward.outputGlobalIndex = oif->outputGlobalIndex;
      forwardedInputsInfo.emplace_back(LogicalForwardInfo{ci, ii, oif->outputGlobalIndex});
    }
    return oif != availableOutputsInfo.end();
  };

  // We have consumed the input, therefore we remove it from the list.
  // We will insert the forwarded inputs only at the end of the iteration.
  auto findNextOutputFor = [&availableOutputsInfo, &constOutputs, &oif, &workflow](
                             size_t ci, size_t& ii) {
    auto& input = workflow[ci].inputs[ii];
    auto matcher = [&input, &constOutputs](const LogicalOutputInfo& outputInfo) -> bool {
      auto& output = constOutputs[outputInfo.outputGlobalIndex];
      return DataSpecUtils::match(input, output);
    };
    oif = availableOutputsInfo.erase(oif);
    oif = std::find_if(oif, availableOutputsInfo.end(), matcher);
    return oif;
  };

  auto numberOfInputsFor = [&workflow](size_t ci) {
    auto& consumer = workflow[ci];
    return consumer.inputs.size();
  };

  auto maxInputTimeslicesFor = [&workflow](size_t pi) {
    auto& processor = workflow[pi];
    return processor.maxInputTimeslices;
  };

  // Trivial, but they make reading easier..
  auto getOutputAssociatedProducer = [&oif]() {
    return oif->specIndex;
  };

  // Trivial, but they make reading easier..
  auto getAssociateOutput = [&oif]() {
    return oif->outputGlobalIndex;
  };

  auto isForward = [&oif]() {
    return oif->forward;
  };

  // Trivial but makes reasing easier in the outer loop.
  auto createEdge = [&logicalEdges](size_t producer,
                                    size_t consumer,
                                    size_t tpi,
                                    size_t ptpi,
                                    size_t uniqueOutputId,
                                    size_t matchingInputInConsumer,
                                    bool doForward) {
    logicalEdges.emplace_back(
      DeviceConnectionEdge{producer,
                           consumer,
                           tpi,
                           ptpi,
                           uniqueOutputId,
                           matchingInputInConsumer,
                           doForward});
  };

  auto errorDueToMissingOutputFor = [&workflow, &constOutputs](size_t ci, size_t ii) {
    auto input = workflow[ci].inputs[ii];
    std::ostringstream str;
    str << "No matching output found for "
        << DataSpecUtils::describe(input) << ". Candidates:\n";

    for (auto& output : constOutputs) {
      str << "-" << DataSpecUtils::describe(output) << "\n";
    }

    throw std::runtime_error(str.str());
  };

  // Whenever we have a set of forwards, we need to append it
  // the the global list of outputs, so that they can be matched
  // and we need to add a ForwardRoute for the current consumer
  // because it is the one who will actually do the forwarding.
  auto appendForwardsToPossibleOutputs = [&availableOutputsInfo, &forwards]() {
    for (auto& forward : forwards) {
      availableOutputsInfo.push_back(forward);
    }
  };

  // Given we create a forward every time we match and input and an
  // output, having no forwards means we did not find any matching.
  auto noMatchingOutputFound = [&forwards]() {
    return forwards.empty();
  };

  // Forwards is basically a cache to record
  auto newEdgeBetweenDevices = [&forwards]() {
    forwards.clear();
  };

  auto forwardOutputFrom = [&forwards](size_t consumer, size_t uniqueOutputId) {
    forwards.push_back(LogicalOutputInfo{consumer, uniqueOutputId, true});
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

  for (size_t consumer = 0; consumer < workflow.size(); ++consumer) {
    for (size_t input = 0; input < numberOfInputsFor(consumer); ++input) {
      newEdgeBetweenDevices();

      while (hasMatchingOutputFor(consumer, input)) {
        auto producer = getOutputAssociatedProducer();
        auto uniqueOutputId = getAssociateOutput();
        for (size_t tpi = 0; tpi < maxInputTimeslicesFor(consumer); ++tpi) {
          for (size_t ptpi = 0; ptpi < maxInputTimeslicesFor(producer); ++ptpi) {
            createEdge(producer, consumer, tpi, ptpi, uniqueOutputId, input, isForward());
          }
          forwardOutputFrom(consumer, uniqueOutputId);
        }
        findNextOutputFor(consumer, input);
      }
      if (noMatchingOutputFound()) {
        errorDueToMissingOutputFor(consumer, input);
      }
      appendForwardsToPossibleOutputs();
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
    if (validNames.find(spec.name) != validNames.end()) {
      throw std::runtime_error("Name " + spec.name + " is used twice.");
    }
    validNames.insert(spec.name);
    for (auto& option : spec.options) {
      if (option.defaultValue.type() != VariantType::Empty &&
          option.type != option.defaultValue.type()) {
        ss << "Mismatch between declared option type and default value type"
           << " for " << option.name << " in DataProcessorSpec of "
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

std::shared_ptr<DataOutputDirector> WorkflowHelpers::getDataOutputDirector(ConfigParamRegistry const& options, std::vector<InputSpec> const& OutputsInputs, std::vector<unsigned char> const& outputTypes)
{
  std::shared_ptr<DataOutputDirector> dod = std::make_shared<DataOutputDirector>();

  // analyze options and take actions accordingly
  // default values
  std::string fnb, fnbase("AnalysisResults_trees");
  std::string fmo, filemode("RECREATE");
  int ntfm, ntfmerge = 1;

  // values from json
  if (options.isSet("aod-writer-json")) {
    auto fnjson = options.get<std::string>("aod-writer-json");
    if (!fnjson.empty()) {
      std::tie(fnb, fmo, ntfm) = dod->readJson(fnjson);
      if (!fnb.empty()) {
        fnbase = fnb;
      }
      if (!fmo.empty()) {
        filemode = fmo;
      }
      if (ntfm > 0) {
        ntfmerge = ntfm;
      }
    }
  }

  // values from command line options, information from json is overwritten
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
  if (options.isSet("aod-writer-ntfmerge")) {
    ntfm = options.get<int>("aod-writer-ntfmerge");
    if (ntfm > 0) {
      ntfmerge = ntfm;
    }
  }
  // parse the keepString
  if (options.isSet("aod-writer-keep")) {
    auto keepString = options.get<std::string>("aod-writer-keep");
    if (!keepString.empty()) {

      dod->reset();
      std::string d("dangling");
      if (d.find(keepString) == 0) {

        // use the dangling outputs
        std::vector<InputSpec> danglingOutputs;
        for (auto ii = 0; ii < OutputsInputs.size(); ii++) {
          if ((outputTypes[ii] & 2) == 2 && (outputTypes[ii] & 1) == 1) {
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
  dod->setFilenameBase(fnbase);
  dod->setFileMode(filemode);
  dod->setNumberTimeFramesToMerge(ntfmerge);

  return dod;
}

std::tuple<std::vector<InputSpec>, std::vector<unsigned char>> WorkflowHelpers::analyzeOutputs(WorkflowSpec const& workflow)
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
  std::vector<unsigned char> outputTypes;
  results.reserve(totalOutputs);
  outputTypes.reserve(totalOutputs);

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

    // compute output type
    unsigned char outputType = UNKNOWN;

    // is AOD?
    if (DataSpecUtils::partialMatch(outputSpec, header::DataOrigin("AOD"))) {
      outputType |= ANALYSIS;
    }
    // is RN2?
    if (DataSpecUtils::partialMatch(outputSpec, header::DataOrigin("RN2"))) {
      outputType |= ANALYSIS;
    }

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
    if (!matched) {
      outputType |= DANGLING;
    }

    // update results and outputTypes
    auto input = DataSpecUtils::matchingInput(outputSpec);
    char buf[64];
    input.binding = (snprintf(buf, 63, "output_%zu_%zu", output.workflowId, output.id), buf);

    // make sure that entries are unique
    if (std::find(results.begin(), results.end(), input) == results.end()) {
      results.emplace_back(input);
      outputTypes.emplace_back(outputType);
    }
  }

  // make sure that results is unique

  return std::make_tuple(results, outputTypes);
}

std::vector<InputSpec> WorkflowHelpers::computeDanglingOutputs(WorkflowSpec const& workflow)
{

  auto [outputsInputs, outputTypes] = analyzeOutputs(workflow);

  std::vector<InputSpec> results;
  for (auto ii = 0u; ii < outputsInputs.size(); ii++) {
    if (outputTypes[ii] & DANGLING) {
      results.emplace_back(outputsInputs[ii]);
    }
  }

  return results;
}

} // namespace framework
} // namespace o2
