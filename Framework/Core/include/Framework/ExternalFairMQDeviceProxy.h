// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef FRAMEWORK_RAWDEVICESOURCE_H
#define FRAMEWORK_RAWDEVICESOURCE_H

#include "Framework/DataProcessorSpec.h"
#include "Framework/OutputSpec.h"
#include "Framework/DataAllocator.h"
#include <fairmq/FairMQParts.h>
#include <vector>
#include <functional>

namespace o2
{
namespace framework
{
using ChannelRetreiver = std::function<std::string(OutputSpec const&)>;
using InjectorFunction = std::function<void(FairMQDevice& device, FairMQParts& inputs, ChannelRetreiver)>;

void sendOnChannel(FairMQDevice& device, o2::header::Stack&& headerStack, FairMQMessagePtr&& payloadMessage, OutputSpec const& spec, ChannelRetreiver& channelRetreiver);

/// Helper function which takes a set of inputs coming from a device,
/// massages them so that they are valid DPL messages using @param spec as header
/// and sends them to the downstream components.
InjectorFunction incrementalConverter(OutputSpec const& spec, uint64_t startTime, uint64_t step);

/// This is to be used for sources which already have an O2 Data Model /
/// (header, payload) structure for their output. At the moment what this /
/// does is to add a DataProcessingHeader. In the future, it will hopefully
/// not be required. Notice that as a requirement the tuple (origin, data
/// description, data subspecification) must be unique for each message in a given
/// multipart ensemble.
InjectorFunction o2DataModelAdaptor(OutputSpec const& spec, uint64_t startTime, uint64_t step);

/// This is to be used when the input data is already formatted like DPL
/// expects it, i.e. with the DataProcessingHeader in the header stack and
/// with the tuple (origin, description, data subspecification, timestamp)
/// must be unique for each message.
InjectorFunction dplModelAdaptor(OutputSpec const& spec);

/// The default connection method for the custom source
static auto gDefaultConverter = incrementalConverter(OutputSpec{"TST", "TEST", 0}, 0, 1);

/// Create a DataProcessorSpec which can be used to inject
/// messages in the DPL.
/// @param label is the label of the DataProcessorSpec associated.
/// @param outputs is the type of messages which this source produces.
/// @param channelConfig is string to be passed to fairmq to create the device.
///        notice that the name of the device will be the same as the label.
/// @param converter is a lambda to be invoked to convert @a inputs into
///        messages of the DPL. By default @a incrementalConverter is used
///        which attaches to each @input FairMQPart a DataProcessingHeader
///        with an incremental number as start time.
DataProcessorSpec specifyExternalFairMQDeviceProxy(char const* label,
                                                   std::vector<OutputSpec> const& outputs,
                                                   const char* channelConfig,
                                                   InjectorFunction converter);
} // namespace framework
} // namespace o2

#endif // FRAMEWORK_RAWDEVICESOURCE_H
