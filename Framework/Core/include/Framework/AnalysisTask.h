// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef FRAMEWORK_ANALYSIS_TASK_H_
#define FRAMEWORK_ANALYSIS_TASK_H_

#include "Framework/DataProcessorSpec.h"
#include "Framework/AlgorithmSpec.h"
#include <arrow/table.h>
#include <type_traits>
#include <utility>
#include <memory>

namespace o2
{
namespace framework
{

/// A more familiar task API for the DPL analysis framework.
/// This allows you to define your own tasks as subclasses
/// of o2::framework::AnalysisTask and to pass them in the specification
/// using:
///
/// adaptAnalysisTask<YourDerivedTask>(constructor args, ...);
///
/// The appropriate AlgorithmSpec invoking `AnalysisTask::init(...)` at
/// startup and `AnalysisTask::run(...)` will be created.
///
class AnalysisTask
{
 public:
  virtual ~AnalysisTask() = default;
  /// The method which is called once to initialise the task.
  /// Derived classes can use this to save extra state.
  virtual void init(InitContext& context) {}
  /// This is invoked whenever a new InputRecord is demeed to
  /// be complete.
  virtual void run(ProcessingContext& context) = 0;

  /// This will be invoked and passed the tracks table for
  /// each message.
  virtual void processTracks(std::shared_ptr<arrow::Table> tracks) {}
};

/// Adaptor to make an AlgorithmSpec from a o2::framework::Task
///
template <typename T, typename... Args>
DataProcessorSpec adaptAnalysisTask(std::string name, Args&&... args)
{
  auto task = std::make_shared<T>(std::forward<Args>(args)...);
  auto algo = AlgorithmSpec::InitCallback{ [task](InitContext& ic) {
    task->init(ic);
    return [task](ProcessingContext& pc) {
      task->run(pc);
      if constexpr (std::is_member_function_pointer<decltype(&T::processTracks)>::value) {
        auto tracks = pc.inputs().get<TableConsumer>("tracks");
        task->processTracks(tracks->asArrowTable());
      }
    };
  } };
  std::vector<InputSpec> inputs;

  if constexpr (std::is_member_function_pointer<decltype(&T::processTracks)>::value) {
    inputs.emplace_back(InputSpec{ "tracks", "AOD", "TRACKPAR" });
  }

  DataProcessorSpec spec{
    name,
    // FIXME: For the moment we hardcode this. We could build
    // this list from the list of methods actually implemented in the
    // task itself.
    inputs,
    // FIXME: Placeholeder for results. We should make it configurable
    // from the task.
    Outputs{ OutputSpec{ "ASIS", "RESULTS", 0 } },
    algo
  };
  return spec;
}

} // namespace framework
} // namespace o2
#endif // FRAMEWORK_ANALYSISTASK_H_
