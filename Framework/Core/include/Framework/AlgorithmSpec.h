// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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

namespace o2
{
namespace framework
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
  static ErrorCallback& emptyErrorCallback()
  {
    static ErrorCallback callback = nullptr;
    return callback;
  }

  AlgorithmSpec()
    : onInit{nullptr},
      onProcess{nullptr},
      onError{nullptr}
  {
  }

  AlgorithmSpec(AlgorithmSpec&&) = default;
  AlgorithmSpec(const AlgorithmSpec&) = default;
  AlgorithmSpec(AlgorithmSpec&) = default;
  AlgorithmSpec& operator=(const AlgorithmSpec&) = default;

  AlgorithmSpec(ProcessCallback process, ErrorCallback& error = emptyErrorCallback())
    : onInit{nullptr},
      onProcess{process},
      onError{error}
  {
  }

  AlgorithmSpec(InitCallback init, ErrorCallback& error = emptyErrorCallback())
    : onInit{init},
      onProcess{nullptr},
      onError{error}
  {
  }

  InitCallback onInit = nullptr;
  ProcessCallback onProcess = nullptr;
  ErrorCallback onError = nullptr;
};

template <typename T>
struct ContextElementTraits {
  static T& get(ProcessingContext& ctx)
  {
    return ctx.services().get<T>();
  }
  static T& get(InitContext& ctx)
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

} // namespace framework
} // namespace o2

#endif // FRAMEWORK_ALGORITHMSPEC_H
