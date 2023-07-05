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
#ifndef FRAMEWORK_ALGORITHMSPEC_H
#define FRAMEWORK_ALGORITHMSPEC_H

#include "Framework/ProcessingContext.h"
#include "Framework/ErrorContext.h"
#include "Framework/InitContext.h"

#include "Framework/FunctionalHelpers.h"

#include <functional>

namespace o2::framework
{

/// This is the class holding the actual algorithm to be used. Notice that the
/// InitCallback  can  be  used  to  define stateful  data  as  it  returns  a
/// ProcessCallback  which will  be invoked  to  do the  data processing.  For
/// example if you  want to have some geometry available  at process time, but
/// of course you do not want to initialise it at every creation, you can do:
///
///
///     AlgorithmSpec{InitCallback{
///       static Geometry geo; // this will last like the whole job
///       return [&geo] {
///           /// Do something with the geometry
///           ///
///       }
///     }
///     }
///
/// FIXME:  we  should  probably  return   also  a  function  to  handle  EXIT
/// transition...
struct AlgorithmSpec {
  using ProcessCallback = std::function<void(ProcessingContext&)>;
  using InitCallback = std::function<ProcessCallback(InitContext&)>;
  using ErrorCallback = std::function<void(ErrorContext&)>;
  using InitErrorCallback = std::function<void(InitErrorContext&)>;

  static AlgorithmSpec dummyAlgorithm()
  {
    return AlgorithmSpec{ProcessCallback{nullptr}};
  }

  static ErrorCallback& emptyErrorCallback()
  {
    static ErrorCallback callback = nullptr;
    return callback;
  }

  static InitErrorCallback& emptyInitErrorCallback()
  {
    static InitErrorCallback callback = nullptr;
    return callback;
  }

  AlgorithmSpec() = default;

  AlgorithmSpec(AlgorithmSpec&&) = default;
  AlgorithmSpec(const AlgorithmSpec&) = default;
  AlgorithmSpec(AlgorithmSpec&) = default;
  AlgorithmSpec& operator=(const AlgorithmSpec&) = default;

  AlgorithmSpec(ProcessCallback process, ErrorCallback& error = emptyErrorCallback())
    : onInit{nullptr},
      onProcess{process},
      onError{error},
      onInitError{nullptr}
  {
  }

  AlgorithmSpec(InitCallback init, ErrorCallback& error = emptyErrorCallback(), InitErrorCallback& initError = emptyInitErrorCallback())
    : onInit{init},
      onProcess{nullptr},
      onError{error},
      onInitError{initError}
  {
  }

  InitCallback onInit = nullptr;
  ProcessCallback onProcess = nullptr;
  ErrorCallback onError = nullptr;
  InitErrorCallback onInitError = nullptr;
};

/// Helper class for an algorithm which is loaded as a plugin.
struct AlgorithmPlugin {
  virtual AlgorithmSpec create() = 0;
};

template <typename T, typename S = std::void_t<>>
struct ContextElementTraits {
  static decltype(auto) get(ProcessingContext& ctx)
  {
    return ctx.services().get<T>();
  }
  static decltype(auto) get(InitContext& ctx)
  {
    return ctx.services().get<T>();
  }
};

template <>
struct ContextElementTraits<ConfigParamRegistry const> {
  static ConfigParamRegistry const& get(InitContext& ctx)
  {
    return ctx.options();
  }
};

template <typename S>
struct ContextElementTraits<ConfigParamRegistry, S> {
  static ConfigParamRegistry const& get(InitContext& ctx)
  {
    static_assert(always_static_assert_v<S>, "Should be ConfigParamRegistry const&");
    return ctx.options();
  }
};

template <>
struct ContextElementTraits<InputRecord> {
  static InputRecord& get(ProcessingContext& ctx)
  {
    return ctx.inputs();
  }
};

template <>
struct ContextElementTraits<DataAllocator> {
  static DataAllocator& get(ProcessingContext& ctx)
  {
    return ctx.outputs();
  }
};

template <>
struct ContextElementTraits<ProcessingContext> {
  static ProcessingContext& get(ProcessingContext& ctx)
  {
    return ctx;
  }
};

template <typename... CONTEXTELEMENT>
AlgorithmSpec::ProcessCallback adaptStatelessF(std::function<void(CONTEXTELEMENT&...)> callback)
{
  return [callback](ProcessingContext& ctx) {
    return callback(ContextElementTraits<CONTEXTELEMENT>::get(ctx)...);
  };
}

template <typename... CONTEXTELEMENT>
AlgorithmSpec::InitCallback adaptStatefulF(std::function<AlgorithmSpec::ProcessCallback(CONTEXTELEMENT&...)> callback)
{
  return [callback](InitContext& ctx) {
    return callback(ContextElementTraits<CONTEXTELEMENT>::get(ctx)...);
  };
}

template <typename R, typename... ARGS>
AlgorithmSpec::ProcessCallback adaptStatelessP(R (*callback)(ARGS...))
{
  std::function<R(ARGS...)> f = callback;
  return adaptStatelessF(f);
}

/// This helper allows us to create a process callback without
/// having to use a context object from which the services and the
/// inputs hang, but it simply uses templates magic to extract them
/// from the context itself and pass them by reference. So instead of
/// writing:
///
/// AlgorithmSpec{[](ProcessingContext& ctx) {
///   ctx.inputs().get<int>("someInt"); // do something with the inputs
/// }
/// }
///
/// you can simply do:
///
/// AlgorithmSpec{[](InputRecord& inputs){
///   inputs.get<int>("someInt");
/// }}
///
/// Notice you can specify in any order any of InputRecord, DataAllocator,
/// ConfigParamRegistry or any of the services which are usually hanging
/// from the ServiceRegistry, e.g. ControlService.
template <typename LAMBDA>
AlgorithmSpec::ProcessCallback adaptStateless(LAMBDA l)
{
  // MAGIC: this makes the lambda decay into a function / method pointer
  return adaptStatelessF(FFL(l));
}

template <typename LAMBDA>
AlgorithmSpec::InitCallback adaptStateful(LAMBDA l)
{
  return adaptStatefulF(FFL(l));
}

} // namespace o2::framework

#endif // FRAMEWORK_ALGORITHMSPEC_H
