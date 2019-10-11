// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef O2_FRAMEWORK_TASK_H_
#define O2_FRAMEWORK_TASK_H_

#include "Framework/AlgorithmSpec.h"
#include <utility>
#include <memory>

namespace o2::framework
{

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
};

/// Adaptor to make an AlgorithmSpec from a o2::framework::Task
///
template <typename T, typename... Args>
AlgorithmSpec adaptFromTask(Args&&... args)
{
  auto task = std::make_shared<T>(std::forward<Args>(args)...);
  return AlgorithmSpec::InitCallback{[task](InitContext& ic) {
    task->init(ic);
    return [task](ProcessingContext& pc) {
      task->run(pc);
    };
  }};
}
} // namespace o2::framework
#endif // O2_FRAMEWORK_TASK_H_
