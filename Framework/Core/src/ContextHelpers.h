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
#ifndef O2_FRAMEWORK_CONTEXTHELPERS_H_
#define O2_FRAMEWORK_CONTEXTHELPERS_H_

#include "Framework/StreamContext.h"
#include "Framework/DataProcessingContext.h"
#include "Framework/ServiceSpec.h"

namespace o2::framework
{
struct ContextHelpers {
  static void bindStreamService(DataProcessorContext& dpContext, StreamContext& stream, ServiceSpec const& spec, void* service);
  static void bindProcessorService(DataProcessorContext& dpContext, ServiceSpec const& spec, void* service);
};

void ContextHelpers::bindStreamService(DataProcessorContext& dpContext, StreamContext& context, ServiceSpec const& spec, void* service)
{
  assert(spec.preDangling == nullptr);
  assert(spec.postDangling == nullptr);
  assert(spec.postDispatching == nullptr);
  assert(spec.postForwarding == nullptr);
  assert(spec.stop == nullptr);
  assert(spec.exit == nullptr);
  assert(spec.domainInfoUpdated == nullptr);
  assert(spec.preSendingMessages == nullptr);
  assert(spec.postRenderGUI == nullptr);
  // Notice that this will mean that stream services will execute the start
  // callback once per stream, not once per dataprocessor.
  if (spec.start) {
    context.preStartStreamHandles.push_back(ServiceStartStreamHandle{spec, spec.start, service});
  }
  if (spec.preProcessing) {
    context.preProcessingHandles.push_back(ServiceProcessingHandle{spec, spec.preProcessing, service});
  }
  if (spec.postProcessing) {
    context.postProcessingHandles.push_back(ServiceProcessingHandle{spec, spec.postProcessing, service});
  }
  // We need to call the preEOS also on a per stream basis, not only on a per
  // data processor basis.
  if (spec.preEOS) {
    dpContext.preEOSHandles.push_back(ServiceEOSHandle{spec, spec.preEOS, service});
  }
  if (spec.postEOS) {
    dpContext.postEOSHandles.push_back(ServiceEOSHandle{spec, spec.postEOS, service});
  }
}

void ContextHelpers::bindProcessorService(DataProcessorContext& dataProcessorContext, ServiceSpec const& spec, void* service)
{
  if (spec.preProcessing) {
    dataProcessorContext.preProcessingHandlers.push_back(ServiceProcessingHandle{spec, spec.preProcessing, service});
  }
  if (spec.postProcessing) {
    dataProcessorContext.postProcessingHandlers.push_back(ServiceProcessingHandle{spec, spec.postProcessing, service});
  }
  if (spec.preDangling) {
    dataProcessorContext.preDanglingHandles.push_back(ServiceDanglingHandle{spec, spec.preDangling, service});
  }
  if (spec.postDangling) {
    dataProcessorContext.postDanglingHandles.push_back(ServiceDanglingHandle{spec, spec.postDangling, service});
  }
  if (spec.preEOS) {
    dataProcessorContext.preEOSHandles.push_back(ServiceEOSHandle{spec, spec.preEOS, service});
  }
  if (spec.postEOS) {
    dataProcessorContext.postEOSHandles.push_back(ServiceEOSHandle{spec, spec.postEOS, service});
  }
  if (spec.postDispatching) {
    dataProcessorContext.postDispatchingHandles.push_back(ServiceDispatchingHandle{spec, spec.postDispatching, service});
  }
  if (spec.postForwarding) {
    dataProcessorContext.postForwardingHandles.push_back(ServiceForwardingHandle{spec, spec.postForwarding, service});
  }
  if (spec.start) {
    dataProcessorContext.preStartHandles.push_back(ServiceStartHandle{spec, spec.start, service});
  }
  if (spec.stop) {
    dataProcessorContext.postStopHandles.push_back(ServiceStopHandle{spec, spec.stop, service});
  }
  if (spec.exit) {
    dataProcessorContext.preExitHandles.push_back(ServiceExitHandle{spec, spec.exit, service});
  }
  if (spec.domainInfoUpdated) {
    dataProcessorContext.domainInfoHandles.push_back(ServiceDomainInfoHandle{spec, spec.domainInfoUpdated, service});
  }
  if (spec.preSendingMessages) {
    dataProcessorContext.preSendingMessagesHandles.push_back(ServicePreSendingMessagesHandle{spec, spec.preSendingMessages, service});
  }
  if (spec.preLoop) {
    dataProcessorContext.preLoopHandles.push_back(ServicePreLoopHandle{spec, spec.preLoop, service});
  }
}

} // namespace o2::framework
#endif // O2_FRAMEWORK_CONTEXTHELPERS_H_
