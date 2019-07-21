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

#include "Framework/ASoA.h"
#include "Framework/AlgorithmSpec.h"
#include "Framework/AnalysisDataModel.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/Kernels.h"
#include "Framework/Logger.h"
#include "Framework/StructToTuple.h"
#include "Framework/Traits.h"

#include <arrow/compute/context.h>
#include <arrow/compute/kernel.h>
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
struct AnalysisTask {
  template <typename T>
  struct Produces {
    static_assert(always_static_assert_v<T>, "Type must be a o2::soa::Table");
  };

  template <typename... C>
  struct Produces<soa::Table<C...>> {
    void operator()(typename C::type...)
    {
      assert(false);
    }
  };
};

// Helper struct which builds a DataProcessorSpec from
// the contents of an AnalysisTask...
struct AnalysisDataProcessorBuilder {
  template <typename Arg>
  static void appendInputWithMetadata(std::vector<InputSpec>& inputs)
  {
    using metadata = typename aod::MetadataTrait<std::decay_t<Arg>>::metadata;
    static_assert(std::is_same_v<metadata, void> == false,
                  "Could not find metadata. Did you register your type?");
    inputs.push_back({metadata::label(), "RN2", metadata::description()});
  }

  template <typename R, typename C, typename... Args>
  static void inputsFromArgs(R (C::*)(Args...), std::vector<InputSpec>& inputs)
  {
    (appendInputWithMetadata<Args>(inputs), ...);
  }

  template <typename R, typename C, typename Grouping, typename... Args>
  static auto signatures(InputRecord& record, R (C::*)(Grouping, Args...))
  {
    return std::declval<std::tuple<Grouping, Args...>>();
  }

  template <typename R, typename C, typename Grouping, typename... Args>
  static auto bindGroupingTable(InputRecord& record, R (C::*)(Grouping, Args...))
  {
    using metadata = typename aod::MetadataTrait<std::decay_t<Grouping>>::metadata;
    return typename metadata::table_t(record.get<TableConsumer>(metadata::label())->asArrowTable());
  }

  template <typename R, typename C>
  static auto bindGroupingTable(InputRecord& record, R (C::*)())
  {
    static_assert(always_static_assert_v<C>, "Your task process method needs at least one argument");
    return o2::soa::Table<>{ nullptr };
  }

  template <typename R, typename C, typename Grouping, typename... Args>
  static auto bindAssociatedTables(InputRecord& record, R (C::*)(Grouping, Args...))
  {
    using metadata = typename aod::MetadataTrait<std::decay_t<Grouping>>::metadata;
    return std::make_tuple(typename aod::MetadataTrait<std::decay_t<Args>>::metadata::table_t(record.get<TableConsumer>(aod::MetadataTrait<std::decay_t<Args>>::metadata::label())->asArrowTable())...);
  }

  template <typename R, typename C>
  static auto bindAssociatedTables(InputRecord& record, R (C::*)())
  {
    static_assert(always_static_assert_v<C>, "Your task process method needs at least one argument");
    return std::tuple<>{};
  }

  template <typename Task, typename R, typename C, typename Grouping, typename... Associated>
  static void invokeProcess(Task& task, InputRecord& inputs, R (C::*)(Grouping, Associated...))
  {
    auto groupingTable = AnalysisDataProcessorBuilder::bindGroupingTable(inputs, &C::process);
    auto associatedTables = AnalysisDataProcessorBuilder::bindAssociatedTables(inputs, &C::process);

    if constexpr (sizeof...(Associated) == 0) {
      // No extra tables: we need to either iterate over the contents of
      // grouping or pass the whole grouping table, depending on whether Grouping
      // is a o2::soa::Table or a o2::soa::RowView
      if constexpr (is_specialization<std::decay_t<Grouping>, o2::soa::Table>::value) {
        task.process(groupingTable);
      } else if constexpr (is_specialization<std::decay_t<Grouping>, o2::soa::RowView>::value) {
        for (auto& groupedElement : groupingTable) {
          task.process(groupedElement);
        }
      } else {
        static_assert(always_static_assert_v<Grouping>,
                      "The first argument of the process method of your task must be either"
                      " a o2::soa::Table or a o2::soa::RowView");
      }
    } else if constexpr (sizeof...(Associated) == 1) {
      // One extra table provided: if the first argument itself is a table,
      // then we simply pass it over to the process method and let the user do the
      // double looping (e.g. if they want to do some association between different
      // physics quantities.
      //
      // If the first argument is a single element, we consider that the user
      // wants to do a loop first on the table associated to the first element,
      // then to the subgroup of the second table which is associated to the
      // first one. E.g.:
      //
      // MyTask::process(Collision const& collision, Tracks const& tracks)
      //
      // Will iterate on all the tracks for the provided collision.
      if constexpr (is_specialization<std::decay_t<Grouping>, o2::soa::Table>::value) {
        static_assert((is_specialization<Associated, o2::soa::Table>::value && ...),
                      "You cannot have a soa::RowView iterator as an argument after the "
                      " first argument of type soa::Table which is found as in the "
                      " prototype of the task process method.");
        task.process(groupingTable, std::get<0>(associatedTables));
      } else if constexpr (is_specialization<std::decay_t<Grouping>, o2::soa::RowView>::value) {
        using AssociatedType = std::tuple_element_t<0, std::tuple<Associated...>>;
        if constexpr (is_specialization<std::decay_t<AssociatedType>, o2::soa::RowView>::value) {
          auto groupedTable = std::get<0>(associatedTables);
          size_t currentGrouping = 0;
          Grouping groupingElement = groupingTable.begin();
          for (auto& groupedElement : groupedTable) {
            // FIXME: this only works for collisions for now...
            auto groupingIndex = groupedElement.collisionId(); // Fine for the moment.
            // We find the associated collision, assuming they are sorted.
            while (groupingIndex > currentGrouping) {
              // This const_cast is done because I do not want people to be
              // able to move the iterator in the user code.
              ++const_cast<std::decay_t<Grouping>&>(groupingElement);
              ++const_cast<std::decay_t<AssociatedType>&>(groupedElement);
            }
            task.process(groupingElement, groupedElement);
          }
        } else if constexpr (is_specialization<std::decay_t<AssociatedType>, o2::soa::Table>::value) {
          auto allGroupedTable = std::get<0>(associatedTables);
          using groupingMetadata = typename aod::MetadataTrait<std::decay_t<Grouping>>::metadata;
          arrow::compute::FunctionContext ctx;
          std::vector<arrow::compute::Datum> groupsCollection;
          auto indexColumnName = std::string("fID4") + groupingMetadata::label();
          auto result = o2::framework::sliceByColumn(&ctx, indexColumnName,
                                                     allGroupedTable.asArrowTable(), &groupsCollection);
          if (result.ok() == false) {
            LOGF(ERROR, "Error while splitting second collection");
            return;
          }
          size_t currentGrouping = 0;
          auto groupingElement = groupingTable.begin();

          // FIXME: this assumes every groupingElement has a group associated,
          // which migh not be the case.
          for (auto& groupedDatum : groupsCollection) {
            auto groupedElementsTable = arrow::util::get<std::shared_ptr<arrow::Table>>(groupedDatum.value);
            task.process(groupingElement, AssociatedType{ groupedElementsTable });
            ++const_cast<std::decay_t<Grouping>&>(groupingElement);
          }
        } else {
          static_assert(always_static_assert_v<AssociatedType>, "I do not know how to iterate on this");
        }
      } else {
        static_assert(always_static_assert_v<Grouping>,
                      "Only grouping by Collision is supported for now");
      }
    } else {
      static_assert(always_static_assert_v<Grouping, Associated...>,
                    "Unable to find a way to iterate on the provided set of arguments. Probably unimplemented");
    }
  }
};

template <typename T>
struct OutputAppender {
  template <typename ANY>
  static bool appendOutput(std::vector<OutputSpec> &outputs, ANY&) {
    return false;
  };
};

template <typename TABLE>
struct OutputAppender<AnalysisTask::Produces<TABLE>> {
  static bool appendOutput(std::vector<OutputSpec> &outputs, AnalysisTask::Produces<TABLE> &what) {
    using metadata = typename aod::MetadataTrait<std::decay_t<TABLE>>::metadata;
    outputs.emplace_back(OutputSpec{ OutputLabel{ metadata::label() }, metadata::origin(), metadata::description() });
    return true;
  };
};

// SFINAE test
template <typename T>
class has_process
{
  typedef char one;
  struct two {
    char x[2];
  };

  template <typename C>
  static one test(decltype(&C::process));
  template <typename C>
  static two test(...);

 public:
  enum { value = sizeof(test<T>(nullptr)) == sizeof(char) };
};

template <typename T>
class has_run
{
  typedef char one;
  struct two {
    char x[2];
  };

  template <typename C>
  static one test(decltype(&C::run));
  template <typename C>
  static two test(...);

 public:
  enum { value = sizeof(test<T>(nullptr)) == sizeof(char) };
};

template <typename T>
class has_init
{
  typedef char one;
  struct two {
    char x[2];
  };

  template <typename C>
  static one test(decltype(&C::init));
  template <typename C>
  static two test(...);

 public:
  enum { value = sizeof(test<T>(nullptr)) == sizeof(char) };
};

/// Adaptor to make an AlgorithmSpec from a o2::framework::Task
///
template <typename T, typename... Args>
DataProcessorSpec adaptAnalysisTask(std::string name, Args&&... args)
{
  auto task = std::make_shared<T>(std::forward<Args>(args)...);

  std::vector<OutputSpec> outputs;
  auto tupledTask = o2::framework::to_tuple(*task);
  std::apply([&outputs](auto ...x){ return (OutputAppender<decltype(x)>::appendOutput(outputs, x) || ...);}, tupledTask);
  static_assert(has_process<T>::value || has_run<T>::value || has_init<T>::value,
                "At least one of process(...), T::run(...), init(...) must be defined");

  std::vector<InputSpec> inputs;
  if constexpr (has_process<T>::value) {
    AnalysisDataProcessorBuilder::inputsFromArgs(&T::process, inputs);
  }

  auto algo = AlgorithmSpec::InitCallback{ [task](InitContext& ic) {
    if constexpr (has_init<T>::value) {
      task->init(ic);
    }
    return [task](ProcessingContext& pc) {
      if constexpr (has_run<T>::value) {
        task->run(pc);
      }
      if constexpr (has_process<T>::value) {
        AnalysisDataProcessorBuilder::invokeProcess(*task, pc.inputs(), &T::process);
      }
    };
  } };

  DataProcessorSpec spec{
    name,
    // FIXME: For the moment we hardcode this. We could build
    // this list from the list of methods actually implemented in the
    // task itself.
    inputs,
    outputs,
    algo
  };
  return spec;
}

} // namespace framework
} // namespace o2
#endif // FRAMEWORK_ANALYSISTASK_H_
