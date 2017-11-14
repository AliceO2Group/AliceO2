// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef FRAMEWORK_DEVICESPECHELPERS_H
#define FRAMEWORK_DEVICESPECHELPERS_H

#include "Framework/WorkflowSpec.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/ChannelSpec.h"
#include "Framework/DeviceControl.h"
#include "Framework/DeviceExecution.h"
#include "Framework/DeviceSpec.h"
#include "Framework/AlgorithmSpec.h"
#include "Framework/ConfigParamSpec.h"
#include "Framework/OutputRoute.h"
#include "WorkflowHelpers.h"
#include <boost/program_options.hpp>

#include <vector>
#include <string>
#include <map>

namespace o2 {
namespace framework {

struct DeviceSpecHelpers {
  /// Helper to convert from an abstract dataflow specification, @a workflow,
  /// to an actual set of devices which will have to run.
  static void dataProcessorSpecs2DeviceSpecs(
      const WorkflowSpec &workflow,
      std::vector<DeviceSpec> &devices);

  /// Helper to prepare the arguments which will be used to 
  /// start the various devices.
  static void prepareArguments(
      int argc,
      char **argv,
      bool defaultQuiet,
      bool defaultStopped,
      const std::vector<DeviceSpec> &deviceSpecs,
      std::vector<DeviceExecution> &deviceExecutions,
      std::vector<DeviceControl> &deviceControls);

  /// This takes the list of preprocessed edges of a graph
  /// and creates Devices and Channels which are related 
  /// to the outgoing edges i.e. those which refer
  /// to the act of producing data.
  static void processOutEdgeActions(
      std::vector<DeviceSpec> &devices,
      std::vector<DeviceId> &deviceIndex,
      std::vector<DeviceConnectionId> &connections,
      unsigned short &nextPort,
      const std::vector<size_t> &outEdgeIndex,
      const std::vector<DeviceConnectionEdge> &logicalEdges,
      const std::vector<EdgeAction> &actions,
      const WorkflowSpec &workflow,
      const std::vector<OutputSpec> &outputs
  );

  /// This takes the list of preprocessed edges of a graph
  /// and creates Devices and Channels which are related 
  /// to the incoming edges i.e. those which refer to
  /// the act of consuming data.
  static void processInEdgeActions(
        std::vector<DeviceSpec> &devices,
        std::vector<DeviceId> &deviceIndex,
        unsigned short &nextPort,
        const std::vector<DeviceConnectionId> &connections,
        const std::vector<size_t> &inEdgeIndex,
        const std::vector<DeviceConnectionEdge> &logicalEdges,
        const std::vector<EdgeAction> &actions,
        const WorkflowSpec &workflow,
        const std::vector<LogicalForwardInfo> &availableForwardsInfo
  );

  /// return a description of all options to be forwarded to the device
  /// by default
  static boost::program_options::options_description getForwardedDeviceOptions();
};

}
}
#endif // FRAMEWORK_DEVICESPECHELPERS_H
