// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef FRAMEWORK_WORKFLOWHELPERS_H
#define FRAMEWORK_WORKFLOWHELPERS_H

#include "Framework/InputSpec.h"
#include "Framework/OutputSpec.h"
#include "Framework/ForwardRoute.h"
#include "Framework/WorkflowSpec.h"

#include <cstddef>
#include <vector>

namespace o2
{
namespace framework
{

// Structure to hold information which was derived 
// for output channels.
struct LogicalOutputInfo {
  size_t specIndex;
  size_t outputGlobalIndex;
  bool forward;
};

// We use this to keep track of the forwards which should
// be added to each device.
// @a consumer is the data processor id the information refers to
// (so all the devices which are incarnation of that data processor should
//  have the forward).
// @a inputGlobalIndex is pointer to a unique id for the input the forward
//    refers to.
struct LogicalForwardInfo {
  size_t consumer;
  size_t inputLocalIndex;
  size_t outputGlobalIndex;
};

enum struct ConnectionKind {
  Out = 0,
  Forward = 1,
  In = 2,
  Unknown = 3
};

struct DeviceConnectionEdge {
  // the index in the workflow of the producer DataProcessorSpec
  size_t producer;
  // the index in the workflow of the consumer DataProcessorSpec
  size_t consumer;
  // The timeindex for the consumer
  size_t timeIndex;
  // The timeindex of the producer
  size_t producerTimeIndex;
  // An absolute id for the output
  size_t outputGlobalIndex;
  // A DataProcessor relative id for the input
  size_t consumerInputIndex;
  // Whether this is the result of a forwarding operation or not
  bool isForward;
  enum ConnectionKind kind;
};


// Unique identifier for a connection
struct DeviceConnectionId {
  size_t producer;
  size_t consumer;
  size_t timeIndex;
  size_t producerTimeIndex;
  size_t port;

  bool operator<(const DeviceConnectionId &rhs) const {
    return std::tie(producer, consumer, timeIndex, producerTimeIndex) <
           std::tie(rhs.producer, rhs.consumer, rhs.timeIndex, rhs.producerTimeIndex);
  }
};

// A device is uniquely identified by its DataProcessorSpec and
// the timeslice it consumes.
struct DeviceId {
  size_t processorIndex;
  size_t timeslice;
  size_t deviceIndex;

  bool operator<(const DeviceId &rhs) const {
    return std::tie(processorIndex, timeslice) <
           std::tie(rhs.processorIndex, rhs.timeslice);
  }
};

struct EdgeAction {
  bool requiresNewDevice = false;
  bool requiresNewChannel = false;
};

/// A set of internal helper classes to manipulate a Workflow
struct WorkflowHelpers {
  /// Topological sort for a graph of @a nodeCount nodes.
  ///  
  /// @a edgeIn pointer to the index of the input node for the first edge
  /// @a edgeOut pointer to the index of the out node for the first edge
  /// @a stride distance (in size_t) between the first and the second element the array
  ///    holding the edges
  /// @return an index vector for the @a nodeCount nodes, where the order is a topological
  /// sort according to the information provided in edges.
  static std::vector<size_t> topologicalSort(size_t nodeCount,
                                             const size_t *edgeIn,
                                             const size_t *edgeOut,
                                             size_t stride,
                                             size_t edgesCount
      );

  // Helper method to verify that a given workflow is actually valid e.g. that
  // it contains no empty labels.
  static void verifyWorkflow(const WorkflowSpec &workflow);

  // Depending on the workflow and the dangling inputs inside it, inject "fake"
  // devices to mark the fact we might need some extra action to make sure
  // dangling inputs are satisfied.
  static void injectServiceDevices(WorkflowSpec& workflow);

  static void constructGraph(const WorkflowSpec &workflow,
                             std::vector<DeviceConnectionEdge> &logicalEdges,
                             std::vector<OutputSpec> &outputs,
                             std::vector<LogicalForwardInfo> &availableForwardsInfo);

  // FIXME: this is an implementation detail for compute edge action, 
  //        actually. It should be moved to the cxx. Comes handy for testing things though..
  static void sortEdges(std::vector<size_t> &inEdgeIndex,
                        std::vector<size_t> &outEdgeIndex,
                        const std::vector<DeviceConnectionEdge> &edges);


  static std::vector<EdgeAction> computeOutEdgeActions(
                                 const std::vector<DeviceConnectionEdge> &edges,
                                 const std::vector<size_t> &index);

  static std::vector<EdgeAction> computeInEdgeActions(
                                 const std::vector<DeviceConnectionEdge> &edges,
                                 const std::vector<size_t> &index);
};

}
}

#endif // FRAMEWORK_WORKFLOWHELPERS_H
