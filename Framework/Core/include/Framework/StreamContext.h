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
#ifndef O2_FRAMEWORK_STREAMCONTEXT_H_
#define O2_FRAMEWORK_STREAMCONTEXT_H_

#include "Framework/ServiceHandle.h"
#include "ProcessingContext.h"
#include "ServiceSpec.h"
#include <functional>

namespace o2::framework
{

struct ProcessingContext;

/// This context exists only for a given stream,
/// it can therefore be used for e.g. callbacks
/// which need to happen on a per stream basis
/// or for caching information which is specific
/// to a given stream (like wether or not we processed
/// something in the current iteration).
struct StreamContext {
  constexpr static ServiceKind service_kind = ServiceKind::Stream;

  // Invoked once per stream before we start processing
  // They are all guaranteed to be invoked before the PreRun
  // function terminates.
  // Notice this will mean that a StreamService which has
  // a start callback might be created upfront.
  void preStartStreamCallbacks(ServiceRegistryRef);

  void preProcessingCallbacks(ProcessingContext& pcx);
  void postProcessingCallbacks(ProcessingContext& pcx);

  /// Invoke callbacks to be executed before every EOS user callback invokation
  void preEOSCallbacks(EndOfStreamContext& eosContext);
  /// Invoke callbacks to be executed after every EOS user callback invokation
  void postEOSCallbacks(EndOfStreamContext& eosContext);

  /// Callbacks for services to be executed before every process method invokation
  std::vector<ServiceProcessingHandle> preProcessingHandles;
  /// Callbacks for services to be executed after every process method invokation
  std::vector<ServiceProcessingHandle> postProcessingHandles;

  /// Callbacks for services to be executed before every EOS user callback invokation
  std::vector<ServiceEOSHandle> preEOSHandles;
  /// Callbacks for services to be executed after every EOS user callback invokation
  std::vector<ServiceEOSHandle> postEOSHandles;

  // Callbacks for services to be executed before a stream starts processing
  // Notice that in such a case all the services will be created upfront, so
  // the callback will be called for all of them.
  std::vector<ServiceStartStreamHandle> preStartStreamHandles;
};

} // namespace o2::framework

#endif // O2_FRAMEWORK_DATAPROCESSINGCONTEXT_H_
