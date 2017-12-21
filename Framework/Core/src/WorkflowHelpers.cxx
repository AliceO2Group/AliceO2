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
#include "Framework/ChannelMatching.h"
#include "Framework/DeviceSpec.h"
#include <algorithm>
#include <list>
#include <set>
#include <utility>
#include <vector>
#include <climits>

namespace o2 {
namespace framework {

std::vector<size_t>
WorkflowHelpers::topologicalSort(size_t nodeCount,
                                 const size_t *edgeIn,
                                 const size_t *edgeOut,
                                 size_t stride,
                                 size_t edgesCount) {
  using NodeIndex = size_t;
  using EdgeIndex = size_t;
  // Create the index which will be returned.
  std::vector<NodeIndex> index(nodeCount);
  for (NodeIndex wi = 0; wi < nodeCount; ++wi) {
    index[wi] = wi;
  }
  // Temporary vector holding vertices to be processed
  std::vector<EdgeIndex> remainingEdgesIndex(edgesCount);
  for (EdgeIndex ei = 0; ei < edgesCount; ++ei) {
    remainingEdgesIndex[ei] = ei;
  }

  // Create a vector where at each position we have true
  // if the vector has dependencies, false otherwise
  std::vector<bool> nodeDeps(nodeCount, false);
  for (EdgeIndex ei = 0; ei < edgesCount; ++ei) {
    nodeDeps[*(edgeOut+ei*stride)] = true;
  }

  std::list<NodeIndex> L;
  std::vector<NodeIndex> S;
  std::set<NodeIndex> nextVertex;
  std::vector<EdgeIndex> nextEdges;

  for (size_t ii = 0, ie = index.size(); ii < ie; ++ii) {
    if (nodeDeps[ii] == false) {
      L.push_back(ii);
    }
  }
  while (!L.empty()) {
    auto node = L.front();
    S.push_back(node);
    L.pop_front();
    nextVertex.clear();
    nextEdges.clear();

    for (EdgeIndex ei = 0, ee = remainingEdgesIndex.size(); ei < ee; ++ei) {
      if (*(edgeIn+ei*stride) == node) {
        nextVertex.insert(*(edgeOut+ei*stride));
      } else {
        nextEdges.push_back(remainingEdgesIndex[ei]);
      }
    }
    remainingEdgesIndex.swap(nextEdges);

    std::set<NodeIndex> hasPredecessors;
    for (auto &ei : remainingEdgesIndex) {
      for (auto &m : nextVertex) {
        if (m == *(edgeOut+ei*stride)) {
          hasPredecessors.insert(m);
        }
      }
    }
    std::vector<NodeIndex> withPredecessor;
    std::set_difference(nextVertex.begin(), nextVertex.end(),
                        hasPredecessors.begin(), hasPredecessors.end(),
                        std::back_inserter(withPredecessor));
    std::copy(withPredecessor.begin(), withPredecessor.end(), std::back_inserter(L));
  }
  return S;
}

void
WorkflowHelpers::constructGraph(const WorkflowSpec &workflow,
                                std::vector<DeviceConnectionEdge> &logicalEdges,
                                std::vector<OutputSpec> &outputs,
                                std::vector<LogicalForwardInfo> &forwardedInputsInfo) {
  assert(!workflow.empty());
  // This is the state. Oif is the iterator I use for the searches.
  std::list<LogicalOutputInfo> availableOutputsInfo;
  auto const &constOutputs = outputs; // const version of the outputs
  decltype(availableOutputsInfo.begin()) oif;
  // Forwards is a local cache to avoid adding forwards before time.
  std::vector<LogicalOutputInfo> forwards;

  // Notice that availableOutputsInfo MUST be updated first, since it relies on
  // the size of outputs to be the one before the update.
  auto enumerateAvailableOutputs = [&workflow,&outputs,&availableOutputsInfo]() {
    for (size_t wi = 0; wi < workflow.size(); ++wi) {
      auto &producer = workflow[wi];

      for (size_t oi = 0; oi < producer.outputs.size(); ++oi) {
        auto &out = producer.outputs[oi];
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
    auto &input = workflow[ci].inputs[ii];
    auto matcher = [&input, &constOutputs](const LogicalOutputInfo &outputInfo) -> bool {
      auto &output = constOutputs[outputInfo.outputGlobalIndex];
      return matchDataSpec2Channel(input, outputSpec2LogicalChannel(output));
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
  auto findNextOutputFor = [&availableOutputsInfo,&constOutputs,&oif,&workflow](
        size_t ci, size_t &ii) {
    auto &input = workflow[ci].inputs[ii];
    auto matcher = [&input, &constOutputs](const LogicalOutputInfo &outputInfo) -> bool {
      auto &output = constOutputs[outputInfo.outputGlobalIndex];
      return matchDataSpec2Channel(input, outputSpec2LogicalChannel(output));
    };
    oif = availableOutputsInfo.erase(oif);
    oif = std::find_if(oif, availableOutputsInfo.end(), matcher);
    return oif;
  };

  auto numberOfInputsFor = [&workflow](size_t ci) {
    auto &consumer = workflow[ci];
    return consumer.inputs.size();
  };

  auto maxInputTimeslicesFor = [&workflow](size_t pi) {
    auto &processor = workflow[pi];
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

  auto errorDueToMissingOutputFor = [&workflow](size_t ci, size_t ii) {
    auto input = workflow[ci].inputs[ii];
    std::ostringstream str;
    str << "No matching output found for " << input.origin.str << " "
        << input.description.str << " "
        << input.subSpec << "\n";
    throw std::runtime_error(str.str());
  };


  // Whenever we have a set of forwards, we need to append it
  // the the global list of outputs, so that they can be matched
  // and we need to add a ForwardRoute for the current consumer
  // because it is the one who will actually do the forwarding.
  auto appendForwardsToPossibleOutputs = [&availableOutputsInfo, &forwards]() {
    for(auto &forward : forwards) {
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
                               const std::vector<DeviceConnectionEdge> &edges,
                               const std::vector<size_t> &index
                               ) {
  DeviceConnectionEdge last{ULONG_MAX,ULONG_MAX,ULONG_MAX,ULONG_MAX,ULONG_MAX,ULONG_MAX};

  assert(edges.size() == index.size());
  std::vector<EdgeAction> actions(edges.size(), EdgeAction{false, false});
  for (size_t i : index) {
    auto &edge = edges[i];
    auto &action = actions[i];
    action.requiresNewDevice = last.producer != edge.producer
                             || last.producerTimeIndex != edge.producerTimeIndex;
    action.requiresNewChannel = last.consumer != edge.consumer
                             || last.producer != edge.producer
                             || last.timeIndex != edge.timeIndex
                             || last.producerTimeIndex != edge.producerTimeIndex;
    last = edge;
  }
  return actions;
}

std::vector<EdgeAction>
WorkflowHelpers::computeInEdgeActions(
                               const std::vector<DeviceConnectionEdge> &edges,
                               const std::vector<size_t> &index
                               ) {
  DeviceConnectionEdge last{ULONG_MAX,ULONG_MAX,ULONG_MAX,ULONG_MAX,ULONG_MAX,ULONG_MAX};

  assert(edges.size() == index.size());
  std::vector<EdgeAction> actions(edges.size(), EdgeAction{false, false});
  for (size_t i : index) {
    auto &edge = edges[i];
    auto &action = actions[i];
    // Calculate which actions need to be taken for this edge.
    action.requiresNewDevice = last.consumer != edge.consumer
                             || last.timeIndex != edge.timeIndex;
    action.requiresNewChannel =
                              last.consumer != edge.consumer
                              || last.timeIndex != edge.timeIndex
                              || last.producer != edge.producer
                              || last.producerTimeIndex != edge.producerTimeIndex;

    last = edge;
  }
  return actions;
}

void
WorkflowHelpers::sortEdges(std::vector<size_t> &inEdgeIndex,
                           std::vector<size_t> &outEdgeIndex,
                           const std::vector<DeviceConnectionEdge> &edges) {
  inEdgeIndex.resize(edges.size());
  outEdgeIndex.resize(edges.size());
  std::iota(inEdgeIndex.begin(), inEdgeIndex.end(), 0);
  std::iota(outEdgeIndex.begin(), outEdgeIndex.end(), 0);

  // Two indexes, one to bind the outputs, the other
  // one to connect the inputs. The 
  auto outSorter = [&edges](size_t i, size_t j) {
    auto &a = edges[i];
    auto &b = edges[j];
    return std::tie(a.producer, a.producerTimeIndex, a.timeIndex, a.consumer)
          < std::tie(b.producer, b.producerTimeIndex, b.timeIndex, b.consumer);
  };
  auto inSorter = [&edges](size_t i, size_t j) {
    auto &a = edges[i];
    auto &b = edges[j];
    return std::tie(a.consumer, a.timeIndex, a.producer, a.producerTimeIndex)
          < std::tie(b.consumer, b.timeIndex, b.producer, b.producerTimeIndex);
  };

  std::sort(inEdgeIndex.begin(), inEdgeIndex.end(), inSorter);
  std::sort(outEdgeIndex.begin(), outEdgeIndex.end(), outSorter);
}

void
WorkflowHelpers::verifyWorkflow(const o2::framework::WorkflowSpec &workflow) {
  if (workflow.empty()){
    throw std::runtime_error("Empty workflow!");
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

  for (auto &spec : workflow)
  {
    if (spec.name.empty())
      throw std::runtime_error("Invalid DataProcessorSpec name");
    if (validNames.find(spec.name) != validNames.end())
      throw std::runtime_error("Name " + spec.name + " is used twice.");
    for (auto &option : spec.options) {
      if (option.defaultValue.type() != VariantType::Empty &&
          option.type != option.defaultValue.type()) {
        ss << "Mismatch between declared option type and default value type"
           << " for " << option.name << " in DataProcessorSpec of "
           << spec.name;
        throw std::runtime_error(ss.str());
      }
    }
    for (size_t ii = 0; ii < spec.inputs.size(); ++ii) {
      InputSpec const &input = spec.inputs[ii];
      if (input.binding.empty()
          || input.description == o2::header::DataDescription("")
          || input.origin == o2::header::DataOrigin("")) {
        ss << "In spec " << spec.name << " input specification " 
           << ii << " requires binding, description and origin"
           " to be fully specified";
        throw std::runtime_error(ss.str());
      }
    }
  }
}

} // namespace framwork
} // namespace o2
