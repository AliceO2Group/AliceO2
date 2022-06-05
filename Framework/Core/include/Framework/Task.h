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
#ifndef O2_FRAMEWORK_TASK_H_
#define O2_FRAMEWORK_TASK_H_

#include "Framework/AlgorithmSpec.h"
#include "Framework/CallbackService.h"
#include "Framework/EndOfStreamContext.h"
#include <utility>
#include <memory>

namespace o2::framework
{

/// Check if the class task has EndOfStream
template <typename T>
class has_endOfStream
{
  typedef char one;
  struct two {
    char x[2];
  };

  template <typename C>
  static one test(decltype(&C::endOfStream));
  template <typename C>
  static two test(...);

 public:
  enum { value = sizeof(test<T>(nullptr)) == sizeof(char) };
};

/// Check if the class task has EndOfStream
template <typename T>
class has_finaliseCCDB
{
  typedef char one;
  struct two {
    char x[2];
  };

  template <typename C>
  static one test(decltype(&C::finaliseCCDB));
  template <typename C>
  static two test(...);

 public:
  enum { value = sizeof(test<T>(nullptr)) == sizeof(char) };
};

/// Check if the class task has Stop
template <typename T>
class has_stop
{
  typedef char one;
  struct two {
    char x[2];
  };

  template <typename C>
  static one test(decltype(&C::stop));
  template <typename C>
  static two test(...);

 public:
  enum { value = sizeof(test<T>(nullptr)) == sizeof(char) };
};

/// A more familiar task API for the DPL.
/// This allows you to define your own tasks as subclasses
/// of o2::framework::Task and to pass them in the specification
/// using:
///
/// adaptTask<YourDerivedTask>(constructor args, ...);
///
/// The appropriate AlgorithmSpec invoking `Task::init(...)` at
/// startup and `Task::run(...)` will be created.
class Task
{
 public:
  virtual ~Task();
  /// The method which is called once to initialise the task.
  /// Derived classes can use this to save extra state.
  virtual void init(InitContext& context) {}
  /// This is invoked whenever a new InputRecord is demeed to
  /// be complete.
  virtual void run(ProcessingContext& context) = 0;

  /// This is invoked whenever we have an EndOfStream event
  virtual void endOfStream(EndOfStreamContext& context) {}

  /// This is invoked whenever a new CCDB object associated to
  /// a given ConcreteDataMatcher is deserialised
  virtual void finaliseCCDB(ConcreteDataMatcher& matcher, void* obj)
  {
    LOGP(debug, "CCDB deserialization invoked");
  }

  /// This is invoked on stop
  virtual void stop() {}
};

/// Adaptor to make an AlgorithmSpec from a o2::framework::Task
///
template <typename T, typename... Args>
AlgorithmSpec adaptFromTask(Args&&... args)
{
  return AlgorithmSpec::InitCallback{[=](InitContext& ic) {
    auto task = std::make_shared<T>(args...);
    if constexpr (has_endOfStream<T>::value) {
      auto& callbacks = ic.services().get<CallbackService>();
      callbacks.set(CallbackService::Id::EndOfStream, [task](EndOfStreamContext& eosContext) {
        task->endOfStream(eosContext);
      });
    }
    if constexpr (has_finaliseCCDB<T>::value) {
      auto& callbacks = ic.services().get<CallbackService>();
      callbacks.set(CallbackService::Id::CCDBDeserialised, [task](ConcreteDataMatcher& matcher, void* obj) {
        task->finaliseCCDB(matcher, obj);
      });
    }
    if constexpr (has_stop<T>::value) {
      auto& callbacks = ic.services().get<CallbackService>();
      callbacks.set(CallbackService::Id::Stop, [task]() {
        task->stop();
      });
    }
    task->init(ic);
    return [task](ProcessingContext& pc) {
      task->run(pc);
    };
  }};
}

template <typename T>
AlgorithmSpec adoptTask(std::shared_ptr<T> task)
{
  return AlgorithmSpec::InitCallback{[task](InitContext& ic) {
    if constexpr (has_endOfStream<T>::value) {
      auto& callbacks = ic.services().get<CallbackService>();
      callbacks.set(CallbackService::Id::EndOfStream, [task](EndOfStreamContext& eosContext) {
        task->endOfStream(eosContext);
      });
    }
    if constexpr (has_finaliseCCDB<T>::value) {
      auto& callbacks = ic.services().get<CallbackService>();
      callbacks.set(CallbackService::Id::CCDBDeserialised, [task](ConcreteDataMatcher& matcher, void* obj) {
        task->finaliseCCDB(matcher, obj);
      });
    }
    if constexpr (has_stop<T>::value) {
      auto& callbacks = ic.services().get<CallbackService>();
      callbacks.set(CallbackService::Id::Stop, [task]() {
        task->stop();
      });
    }
    task->init(ic);
    return [&task](ProcessingContext& pc) {
      task->run(pc);
    };
  }};
}
} // namespace o2::framework
#endif // O2_FRAMEWORK_TASK_H_
