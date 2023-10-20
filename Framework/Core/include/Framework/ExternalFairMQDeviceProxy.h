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
#ifndef FRAMEWORK_RAWDEVICESOURCE_H
#define FRAMEWORK_RAWDEVICESOURCE_H

#include "Framework/DataProcessorSpec.h"
#include "Framework/OutputSpec.h"
#include <fairmq/FwdDecls.h>
#include <vector>
#include <functional>

namespace o2::framework
{

/// A callback function to retrieve the fair::mq::Channel name to be used for sending
/// messages of the specified OutputSpec
using ChannelRetriever = std::function<std::string(OutputSpec const&, DataProcessingHeader::StartTime)>;
/// The callback which actually does the heavy lifting of converting the input data into
/// DPL messages. The callback is invoked with the following parameters:
/// @param timingInfo is the timing information of the current timeslice
/// @param services is the service registry
/// @param inputs is the list of input messages
/// @param channelRetriever is a callback to retrieve the fair::mq::Channel name to be used for
///        sending the messages
/// @param newTimesliceId is the timeslice ID of the current timeslice
/// @return true if any message were sent, false otherwise
using InjectorFunction = std::function<bool(TimingInfo&, ServiceRegistryRef const& services, fair::mq::Parts& inputs, ChannelRetriever, size_t newTimesliceId, bool& stop)>;
using ChannelSelector = std::function<std::string(InputSpec const& input, const std::unordered_map<std::string, std::vector<fair::mq::Channel>>& channels)>;

struct InputChannelSpec;
struct OutputChannelSpec;

/// helper method to format a configuration string for an external channel
std::string formatExternalChannelConfiguration(InputChannelSpec const&);

/// helper method to format a configuration string for an external channel
std::string formatExternalChannelConfiguration(OutputChannelSpec const&);

/// send header/payload O2 message for an OutputSpec, a channel retriever callback is required to
/// get the associated fair::mq::Channel
/// FIXME: can in principle drop the OutputSpec parameter and take the DataHeader
void sendOnChannel(fair::mq::Device& device, o2::header::Stack&& headerStack, fair::mq::MessagePtr&& payloadMessage, OutputSpec const& spec, ChannelRetriever& channelRetriever);

void sendOnChannel(fair::mq::Device& device, fair::mq::Parts& messages, std::string const& channel, size_t timeSlice);

/// append a header/payload part to multipart message for aggregate sending, a channel retriever
/// callback is required to get the associated fair::mq::Channel
void appendForSending(fair::mq::Device& device, o2::header::Stack&& headerStack, size_t timeSliceID, fair::mq::MessagePtr&& payloadMessage, OutputSpec const& spec, fair::mq::Parts& messageCache, ChannelRetriever& channelRetriever);

/// Helper function which takes a set of inputs coming from a device,
/// massages them so that they are valid DPL messages using @param spec as header
/// and sends them to the downstream components.
InjectorFunction incrementalConverter(OutputSpec const& spec, o2::header::SerializationMethod method, uint64_t startTime, uint64_t step);

/// This is to be used for sources which already have an O2 Data Model /
/// (header, payload) structure for their output. At the moment what this /
/// does is to add a DataProcessingHeader. In the future, it will hopefully
/// not be required. Notice that as a requirement the tuple (origin, data
/// description, data subspecification) must be unique for each message in a given
/// multipart ensemble.
InjectorFunction o2DataModelAdaptor(OutputSpec const& spec, uint64_t startTime, uint64_t step);

/// @struct DPLModelAdapterConfig
/// Configuration object for dplModelAdaptor
struct DPLModelAdapterConfig {
  /// throw runtime error if an input message is not matched by filter rules
  bool throwOnUnmatchedInputs = true;
  /// do all kinds of consistency checks
  bool paranoid = false;
  /// blindly forward on one channel
  bool blindForward = false;
};

/// This is to be used when the input data is already formatted like DPL
/// expects it, i.e. with the DataProcessingHeader in the header stack
/// The list of specs is used as a filter list, all incoming data matching an entry
/// in the list will be send through the corresponding channel
InjectorFunction dplModelAdaptor(std::vector<OutputSpec> const& specs = {{header::gDataOriginAny, header::gDataDescriptionAny}},
                                 DPLModelAdapterConfig config = DPLModelAdapterConfig{});

/// legacy function
inline InjectorFunction dplModelAdaptor(std::vector<OutputSpec> const& specs, bool throwOnUnmatchedInputs)
{
  return dplModelAdaptor(specs, DPLModelAdapterConfig{throwOnUnmatchedInputs});
}

/// The default connection method for the custom source
static auto gDefaultConverter = incrementalConverter(OutputSpec{"TST", "TEST", 0}, header::gSerializationMethodROOT, 0, 1);

/// Default way to select an output channel for multi-output proxy.
std::string defaultOutputProxyChannelSelector(InputSpec const& input, const std::unordered_map<std::string, std::vector<fair::mq::Channel>>& channels);

/// Create a DataProcessorSpec which can be used to inject
/// messages in the DPL.
/// @param label is the label of the DataProcessorSpec associated and name of the input channel.
/// @param outputs is the type of messages which this source produces.
/// @param channelConfig is string to be passed to fairmq to create the device.
///        notice that the name of the device will be added as the name of the channel if the
///        name tag is not yet in the configuration
/// @param converter is a lambda to be invoked to convert @a inputs into
///        messages of the DPL. By default @a incrementalConverter is used
///        which attaches to each @input FairMQPart a DataProcessingHeader
///        with an incremental number as start time.
DataProcessorSpec specifyExternalFairMQDeviceProxy(char const* label,
                                                   std::vector<OutputSpec> const& outputs,
                                                   const char* defaultChannelConfig,
                                                   InjectorFunction converter,
                                                   uint64_t minSHM = 0,
                                                   bool sendTFcounter = false,
                                                   bool doInjectMissingData = false,
                                                   unsigned int doPrintSizes = 0);

DataProcessorSpec specifyFairMQDeviceOutputProxy(char const* label,
                                                 Inputs const& inputSpecs,
                                                 const char* defaultChannelConfig);
/// Create a DataProcessorSpec for a DPL processor with an out-of-band channel to relay DPL
/// workflow data to an external fair::mq::Device channel.
///
/// The output configuration is determined by one or multiple entries of the fair::mq::Device
/// command line option '--channel-config' in the format
///    --channel-config "name=channel-name;..."
/// A default string is build from the provided parameter.
///
/// The target of each input data matcher is specified as the binding identifier and matched to the
/// configured output channels
/// @param label is the label of the DataProcessorSpec associated.
/// @param inputSpecs the list of inputs to read from, the binding of the spec must
///        correspond to an output channel
/// @param defaultChannelConfig is the default configuration of the out-of-band channel
///        the string is passed to fairmq to create the device channel and can be adjusted
///        by command line option '--channel-config'
///        notice that the name of the device will be added as the name of the channel if the
///        name tag is not yet in the configuration
DataProcessorSpec specifyFairMQDeviceMultiOutputProxy(char const* label,
                                                      Inputs const& inputSpecs,
                                                      const char* defaultChannelConfig,
                                                      ChannelSelector channelSelector = defaultOutputProxyChannelSelector);

} // namespace o2

#endif // FRAMEWORK_RAWDEVICESOURCE_H
