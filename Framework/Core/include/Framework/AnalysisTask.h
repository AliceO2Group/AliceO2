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
#include "Framework/CallbackService.h"
#include "Framework/ControlService.h"
#include "Framework/ConfigParamSpec.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/Expressions.h"
#include "Framework/EndOfStreamContext.h"
#include "Framework/Kernels.h"
#include "Framework/Logger.h"
#include "Framework/HistogramRegistry.h"
#include "Framework/StructToTuple.h"
#include "Framework/FunctionalHelpers.h"
#include "Framework/Traits.h"
#include "Framework/VariantHelpers.h"
#include "Framework/OutputObjHeader.h"

#include <arrow/compute/context.h>
#include <arrow/compute/kernel.h>
#include <arrow/table.h>
#include <gandiva/node.h>
#include <type_traits>
#include <utility>
#include <memory>
#if (defined(__GNUC__) && (__GNUC___ > 8 || (__GNUC__ == 8 && __GNUC_MINOR__ >= 1))) || defined(__clang__)
#include <charconv>
#else
#include <sstream>
#include <iomanip>
#endif
namespace o2::framework
{

/// A more familiar task API for the DPL analysis framework.
/// This allows you to define your own tasks as subclasses
/// of o2::framework::AnalysisTask and to pass them in the specification
/// using:
///
/// adaptAnalysisTask<YourDerivedTask>(constructor args, ...);
///
// FIXME: for the moment this needs to stay outside AnalysisTask
//        because we cannot inherit from it due to a C++17 bug
//        in GCC 7.3. We need to move to 7.4+
template <typename T>
struct WritingCursor {
  static_assert(always_static_assert_v<T>, "Type must be a o2::soa::Table");
};

template <typename T>
struct Produces {
  static_assert(always_static_assert_v<T>, "Type must be a o2::soa::Table");
};

/// Helper class actually implementing the cursor which can write to
/// a table. The provided template arguments are if type Column and
/// therefore refer only to the persisted columns.
template <typename... PC>
struct WritingCursor<soa::Table<PC...>> {
  using persistent_table_t = soa::Table<PC...>;
  using cursor_t = decltype(std::declval<TableBuilder>().cursor<persistent_table_t>());

  template <typename... T>
  void operator()(T... args)
  {
    static_assert(sizeof...(PC) == sizeof...(T), "Argument number mismatch");
    ++mCount;
    cursor(0, extract(args)...);
  }

  /// Last index inserted in the table
  int64_t lastIndex()
  {
    return mCount;
  }

  bool resetCursor(TableBuilder& builder)
  {
    mBuilder = &builder;
    cursor = std::move(FFL(builder.cursor<persistent_table_t>()));
    mCount = -1;
    return true;
  }

  /// reserve @a size rows when filling, so that we do not
  /// spend time reallocating the buffers.
  void reserve(int64_t size)
  {
    mBuilder->reserve(typename persistent_table_t::columns{}, size);
  }

  decltype(FFL(std::declval<cursor_t>())) cursor;

 private:
  template <typename T>
  static decltype(auto) extract(T const& arg)
  {
    if constexpr (soa::is_soa_iterator_t<T>::value) {
      return arg.globalIndex();
    } else {
      static_assert(!framework::has_type_v<T, framework::pack<PC...>>, "Argument type mismatch");
      return arg;
    }
  }

  /// The table builder which actually performs the
  /// construction of the table. We keep it around to be
  /// able to do all-columns methods like reserve.
  TableBuilder* mBuilder = nullptr;
  int64_t mCount = -1;
};

/// This helper class allow you to declare things which will be crated by a
/// give analysis task. Notice how the actual cursor is implemented by the
/// means of the WritingCursor helper class, from which produces actually
/// derives.
template <typename... C>
struct Produces<soa::Table<C...>> : WritingCursor<typename soa::FilterPersistentColumns<soa::Table<C...>>::persistent_table_t> {
  using table_t = soa::Table<C...>;
  using metadata = typename aod::MetadataTrait<table_t>::metadata;

  // @return the associated OutputSpec
  OutputSpec const spec()
  {
    return OutputSpec{OutputLabel{metadata::label()}, metadata::origin(), metadata::description()};
  }

  OutputRef ref()
  {
    return OutputRef{metadata::label(), 0};
  }
};

/// This helper class allows you to declare things which will be created by a
/// given analysis task. Currently wrapped objects are limited to be TNamed
/// descendants. Objects will be written to a ROOT file at the end of the
/// workflow, in directories, corresponding to the task they were declared in.
/// Each object has associated handling policy, which is used by the framework
/// to determine the target file, e.g. analysis result, QA or control histogram,
/// etc.
template <typename T>
struct OutputObj {
  using obj_t = T;

  OutputObj(T const& t, OutputObjHandlingPolicy policy_ = OutputObjHandlingPolicy::AnalysisObject)
    : object(std::make_shared<T>(t)),
      label(t.GetName()),
      policy{policy_},
      mTaskHash{0}
  {
  }

  OutputObj(std::string const& label_, OutputObjHandlingPolicy policy_ = OutputObjHandlingPolicy::AnalysisObject)
    : object(nullptr),
      label(label_),
      policy{policy_},
      mTaskHash{0}
  {
  }

  void setObject(T const& t)
  {
    object = std::make_shared<T>(t);
    object->SetName(label.c_str());
  }

  void setObject(T&& t)
  {
    object = std::make_shared<T>(t);
    object->SetName(label.c_str());
  }

  void setObject(T* t)
  {
    object.reset(t);
    object->SetName(label.c_str());
  }

  void setHash(uint32_t hash)
  {
    mTaskHash = hash;
  }

  /// @return the associated OutputSpec
  OutputSpec const spec()
  {
    static_assert(std::is_base_of_v<TNamed, T>, "You need a TNamed derived class to use OutputObj");
    header::DataDescription desc{};
    auto lhash = compile_time_hash(label.c_str());
    memset(desc.str, '_', 16);
#if (defined(__GNUC__) && (__GNUC___ > 8 || (__GNUC__ == 8 && __GNUC_MINOR__ >= 1))) || defined(__clang__)
    char buf[12];
    std::to_chars(buf, buf + 2, lhash, 16);
    std::to_chars(buf + 2, buf + 4, mTaskHash, 16);
    std::to_chars(buf + 4, buf + 12, reinterpret_cast<uint64_t>(this), 16);
    std::memcpy(desc.str, buf, 12);
#else
    std::stringstream s;
    s << std::hex << lhash << mTaskHash << reinterpret_cast<uint64_t>(this);
    std::memcpy(desc.str, s.str().c_str(), 12);
#endif

    return OutputSpec{OutputLabel{label}, "ATSK", desc, 0};
  }

  T* operator->()
  {
    return object.get();
  }

  T& operator*()
  {
    return *object.get();
  }

  OutputRef ref()
  {
    return OutputRef{std::string{label}, 0,
                     o2::header::Stack{OutputObjHeader{policy, mTaskHash}}};
  }

  std::shared_ptr<T> object;
  std::string label;
  OutputObjHandlingPolicy policy;
  uint32_t mTaskHash;
};

/// This helper allows you to create a configurable option associated to a task.
/// Internally it will be bound to a ConfigParamSpec.
template <typename T>
struct Configurable {
  Configurable(std::string const& name, T defaultValue, std::string const& help)
    : name(name), value(defaultValue), help(help)
  {
  }
  using type = T;
  std::string name;
  T value;
  std::string help;
  operator T()
  {
    return value;
  }
};

struct AnalysisTask {
};

// Helper struct which builds a DataProcessorSpec from
// the contents of an AnalysisTask...

template <typename... C>
gandiva::SchemaPtr createSchemaFromColumns(framework::pack<C...>)
{
  return std::make_shared<arrow::Schema>(std::vector<std::shared_ptr<arrow::Field>>{
    std::make_shared<arrow::Field>(C::mLabel, expressions::concreteArrowType(expressions::selectArrowType<typename C::type>()))...});
}

struct AnalysisDataProcessorBuilder {
  template <typename Arg>
  static void doAppendInputWithMetadata(std::vector<InputSpec>& inputs)
  {
    using metadata = typename aod::MetadataTrait<std::decay_t<Arg>>::metadata;
    static_assert(std::is_same_v<metadata, void> == false,
                  "Could not find metadata. Did you register your type?");
    inputs.push_back({metadata::label(), "AOD", metadata::description()});
  }

  template <typename... Args>
  static void doAppendInputWithMetadata(framework::pack<Args...>, std::vector<InputSpec>& inputs)
  {
    (doAppendInputWithMetadata<Args>(inputs), ...);
  }

  template <typename T, size_t At>
  static void appendSomethingWithMetadata(std::vector<InputSpec>& inputs, std::vector<ExpressionInfo>& eInfos)
  {
    using dT = std::decay_t<T>;
    if constexpr (framework::is_specialization<dT, soa::Filtered>::value) {
      eInfos.push_back({At, createSchemaFromColumns(typename dT::table_t::persistent_columns_t{}), nullptr});
    } else if constexpr (soa::is_soa_iterator_t<dT>::value) {
      if (std::is_same_v<typename dT::policy_t, soa::FilteredIndexPolicy>) {
        eInfos.push_back({At, createSchemaFromColumns(typename dT::table_t::persistent_columns_t{}), nullptr});
      }
    }
    doAppendInputWithMetadata(soa::make_originals_from_type<dT>(), inputs);
  }

  template <typename R, typename C, typename... Args>
  static void inputsFromArgs(R (C::*)(Args...), std::vector<InputSpec>& inputs, std::vector<ExpressionInfo>& eInfos)
  {
    (appendSomethingWithMetadata<Args, has_type_at<Args>(pack<Args...>{})>(inputs, eInfos), ...);
  }

  template <typename R, typename C, typename Grouping, typename... Args>
  static auto signatures(InputRecord&, R (C::*)(Grouping, Args...))
  {
    return std::declval<std::tuple<Grouping, Args...>>();
  }

  template <typename R, typename C, typename Grouping, typename... Args>
  static auto bindGroupingTable(InputRecord& record, R (C::*)(Grouping, Args...), std::vector<ExpressionInfo> const& infos)
  {
    return extractSomethingFromRecord<Grouping, 0>(record, infos);
  }

  template <typename R, typename C>
  static auto bindGroupingTable(InputRecord&, R (C::*)(), std::vector<ExpressionInfo> const&)
  {
    static_assert(always_static_assert_v<C>, "Your task process method needs at least one argument");
    return o2::soa::Table<>{nullptr};
  }

  template <typename T>
  static auto extractTableFromRecord(InputRecord& record)
  {
    if constexpr (soa::is_type_with_metadata_v<aod::MetadataTrait<T>>) {
      return record.get<TableConsumer>(aod::MetadataTrait<T>::metadata::label())->asArrowTable();
    } else if constexpr (soa::is_type_with_originals_v<T>) {
      return extractFromRecord<T>(record, typename T::originals{});
    }
    O2_BUILTIN_UNREACHABLE();
  }

  template <typename T, typename... Os>
  static auto extractFromRecord(InputRecord& record, pack<Os...> const&)
  {
    if constexpr (soa::is_soa_iterator_t<T>::value) {
      return typename T::parent_t{{extractTableFromRecord<Os>(record)...}};
    } else {
      return T{{extractTableFromRecord<Os>(record)...}};
    }
  }

  template <typename T, typename... Os>
  static auto extractFilteredFromRecord(InputRecord& record, ExpressionInfo const& info, pack<Os...> const&)
  {
    if constexpr (soa::is_soa_iterator_t<T>::value) {
      return soa::Filtered<typename T::table_t>(std::vector<std::shared_ptr<arrow::Table>>{extractTableFromRecord<Os>(record)...}, info.tree);
    } else {
      return T(std::vector<std::shared_ptr<arrow::Table>>{extractTableFromRecord<Os>(record)...}, info.tree);
    }
  }

  template <typename T, size_t At>
  static auto extractSomethingFromRecord(InputRecord& record, std::vector<ExpressionInfo> const infos)
  {
    using decayed = std::decay_t<T>;
    if constexpr (is_specialization<decayed, soa::Filtered>::value) {
      for (auto& info : infos) {
        if (info.index == At)
          return extractFilteredFromRecord<decayed>(record, info, soa::make_originals_from_type<decayed>());
      }
    } else if constexpr (soa::is_soa_iterator_t<decayed>::value) {
      if constexpr (std::is_same_v<typename decayed::policy_t, soa::FilteredIndexPolicy>) {
        for (auto& info : infos) {
          if (info.index == At)
            return extractFilteredFromRecord<decayed>(record, info, soa::make_originals_from_type<decayed>());
        }
      } else {
        return extractFromRecord<decayed>(record, soa::make_originals_from_type<decayed>());
      }
    } else {
      return extractFromRecord<decayed>(record, soa::make_originals_from_type<decayed>());
    }
    O2_BUILTIN_UNREACHABLE();
  }

  template <typename R, typename C, typename Grouping, typename... Args>
  static auto bindAssociatedTables(InputRecord& record, R (C::*)(Grouping, Args...), std::vector<ExpressionInfo> const infos)
  {
    return std::make_tuple(extractSomethingFromRecord<Args, has_type_at<Args>(pack<Args...>{}) + 1u>(record, infos)...);
  }

  template <typename R, typename C>
  static auto bindAssociatedTables(InputRecord&, R (C::*)(), std::vector<ExpressionInfo> const)
  {
    static_assert(always_static_assert_v<C>, "Your task process method needs at least one argument");
    return std::tuple<>{};
  }

  template <typename Task, typename R, typename C, typename Grouping, typename... Associated>
  static void invokeProcess(Task& task, InputRecord& inputs, R (C::*)(Grouping, Associated...), std::vector<ExpressionInfo> const& infos)
  {
    auto groupingTable = AnalysisDataProcessorBuilder::bindGroupingTable(inputs, &C::process, infos);
    auto associatedTables = AnalysisDataProcessorBuilder::bindAssociatedTables(inputs, &C::process, infos);

    if constexpr (sizeof...(Associated) == 0) {
      // No extra tables: we need to either iterate over the contents of
      // grouping or pass the whole grouping table, depending on whether Grouping
      // is a o2::soa::Table or a o2::soa::RowView
      if constexpr (soa::is_soa_table_t<std::decay_t<Grouping>>::value) {
        task.process(groupingTable);
      } else if constexpr (soa::is_soa_iterator_t<std::decay_t<Grouping>>::value) {
        for (auto& groupedElement : groupingTable) {
          task.process(groupedElement);
        }
      } else if constexpr (soa::is_soa_join_t<std::decay_t<Grouping>>::value) {
        task.process(groupingTable);
      } else if constexpr (soa::is_soa_filtered_t<std::decay_t<Grouping>>::value) {
        task.process(groupingTable);
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
      if constexpr (soa::is_soa_table_t<std::decay_t<Grouping>>::value) {
        static_assert(((soa::is_soa_iterator_t<std::decay_t<Associated>>::value == false) && ...),
                      "You cannot have a soa::RowView iterator as an argument after the "
                      " first argument of type soa::Table which is found as in the "
                      " prototype of the task process method.");
        auto& associated = std::get<0>(associatedTables);
        associated.bindExternalIndices(&groupingTable);
        groupingTable.bindExternalIndices(&associated);
        task.process(groupingTable, associated);
      } else if constexpr (soa::is_soa_iterator_t<std::decay_t<Grouping>>::value) {
        using AssociatedType = std::tuple_element_t<0, std::tuple<Associated...>>;
        if constexpr (soa::is_soa_iterator_t<std::decay_t<AssociatedType>>::value) {
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
        } else if constexpr (soa::is_soa_table_like_t<std::decay_t<AssociatedType>>::value) {
          auto allGroupedTable = std::get<0>(associatedTables);
          using groupingMetadata = typename aod::MetadataTrait<std::decay_t<Grouping>>::metadata;
          arrow::compute::FunctionContext ctx;
          std::vector<arrow::compute::Datum> groupsCollection;
          auto indexColumnName = std::string("f") + groupingMetadata::label() + "ID";
          std::vector<uint64_t> offsets;
          auto result = o2::framework::sliceByColumn(&ctx, indexColumnName,
                                                     allGroupedTable.asArrowTable(), &groupsCollection, &offsets);
          if (result.ok() == false) {
            LOGF(ERROR, "Error while splitting second collection");
            return;
          }
          size_t currentGrouping = 0;
          auto groupingElement = groupingTable.begin();

          // FIXME: this assumes every groupingElement has a group associated,
          // which migh not be the case.
          size_t oi = 0;
          if constexpr (soa::is_soa_table_t<std::decay_t<AssociatedType>>::value) {
            for (auto& groupedDatum : groupsCollection) {
              auto groupedElementsTable = arrow::util::get<std::shared_ptr<arrow::Table>>(groupedDatum.value);
              std::decay_t<AssociatedType> typedTable{groupedElementsTable, offsets[oi]};
              typedTable.bindExternalIndices(&groupingTable);
              task.process(groupingElement, typedTable);
              ++const_cast<std::decay_t<Grouping>&>(groupingElement);
              ++oi;
            }
          } else if constexpr (soa::is_soa_filtered_t<std::decay_t<AssociatedType>>::value) {
            auto& fullSelection = allGroupedTable.getSelectedRows();
            offsets.push_back(allGroupedTable.tableSize());

            auto current_start = fullSelection.begin();

            for (auto& groupedDatum : groupsCollection) {
              auto groupedElementsTable = arrow::util::get<std::shared_ptr<arrow::Table>>(groupedDatum.value);

              // for each grouping element we need to slice the selection vector
              auto start_iterator = std::lower_bound(current_start, fullSelection.end(), offsets[oi]);
              auto stop_iterator = std::lower_bound(start_iterator, fullSelection.end(), offsets[oi + 1]);
              current_start = stop_iterator;
              soa::SelectionVector slicedSelection{start_iterator, stop_iterator};
              std::transform(slicedSelection.begin(), slicedSelection.end(), slicedSelection.begin(), [&](int64_t index) { return index - static_cast<int64_t>(offsets[oi]); });

              std::decay_t<AssociatedType> typedTable{{groupedElementsTable}, std::move(slicedSelection), offsets[oi]};
              typedTable.bindExternalIndices(&groupingTable);
              task.process(groupingElement, typedTable);
              ++const_cast<std::decay_t<Grouping>&>(groupingElement);
              ++oi;
            }
          } else if constexpr (soa::is_soa_join_t<std::decay_t<AssociatedType>>::value || soa::is_soa_concat_t<std::decay_t<AssociatedType>>::value) {
            for (auto& groupedDatum : groupsCollection) {
              auto groupedElementsTable = arrow::util::get<std::shared_ptr<arrow::Table>>(groupedDatum.value);
              // Set the refererred table.
              // FIXME: we should be able to do this upfront for all the tables.
              std::decay_t<AssociatedType> typedTable{{groupedElementsTable}, offsets[oi]};
              typedTable.bindExternalIndices(&groupingTable);
              task.process(groupingElement, typedTable);
              ++const_cast<std::decay_t<Grouping>&>(groupingElement);
              ++oi;
            }
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
struct FilterManager {
  template <typename ANY>
  static bool createExpressionTrees(ANY&, std::vector<ExpressionInfo>&)
  {
    return false;
  }
};

template <>
struct FilterManager<expressions::Filter> {
  static bool createExpressionTrees(expressions::Filter const& filter, std::vector<ExpressionInfo>& expressionInfos)
  {
    updateExpressionInfos(filter, expressionInfos);
    return true;
  }
};

template <typename T>
struct OutputManager {
  template <typename ANY>
  static bool appendOutput(std::vector<OutputSpec>&, ANY&, uint32_t)
  {
    return false;
  }

  template <typename ANY>
  static bool prepare(ProcessingContext&, ANY&)
  {
    return false;
  }

  template <typename ANY>
  static bool postRun(EndOfStreamContext&, ANY&)
  {
    return true;
  }

  template <typename ANY>
  static bool finalize(ProcessingContext&, ANY&)
  {
    return true;
  }
};

template <typename TABLE>
struct OutputManager<Produces<TABLE>> {
  static bool appendOutput(std::vector<OutputSpec>& outputs, Produces<TABLE>& what, uint32_t)
  {
    outputs.emplace_back(what.spec());
    return true;
  }
  static bool prepare(ProcessingContext& context, Produces<TABLE>& what)
  {
    what.resetCursor(context.outputs().make<TableBuilder>(what.ref()));
    return true;
  }
  static bool finalize(ProcessingContext&, Produces<TABLE>&)
  {
    return true;
  }
  static bool postRun(EndOfStreamContext&, Produces<TABLE>&)
  {
    return true;
  }
};

template <>
struct OutputManager<HistogramRegistry> {
  static bool appendOutput(std::vector<OutputSpec>& outputs, HistogramRegistry& what, uint32_t)
  {
    outputs.emplace_back(what.spec());
    return true;
  }
  static bool prepare(ProcessingContext&, HistogramRegistry&)
  {
    return true;
  }

  static bool finalize(ProcessingContext&, HistogramRegistry&)
  {
    return true;
  }

  static bool postRun(EndOfStreamContext&, HistogramRegistry&)
  {
    return true;
  }
};

template <typename T>
struct OutputManager<OutputObj<T>> {
  static bool appendOutput(std::vector<OutputSpec>& outputs, OutputObj<T>& what, uint32_t hash)
  {
    what.setHash(hash);
    outputs.emplace_back(what.spec());
    return true;
  }
  static bool prepare(ProcessingContext&, OutputObj<T>&)
  {
    return true;
  }

  static bool finalize(ProcessingContext&, OutputObj<T>&)
  {
    return true;
  }

  static bool postRun(EndOfStreamContext& context, OutputObj<T>& what)
  {
    context.outputs().snapshot(what.ref(), *what);
    return true;
  }
};

template <typename T>
struct OptionManager {
  template <typename ANY>
  static bool appendOption(std::vector<ConfigParamSpec>&, ANY&)
  {
    return false;
  }

  template <typename ANY>
  static bool prepare(InitContext&, ANY&)
  {
    return false;
  }
};

template <typename T>
struct OptionManager<Configurable<T>> {
  static bool appendOption(std::vector<ConfigParamSpec>& options, Configurable<T>& what)
  {
    options.emplace_back(ConfigParamSpec{what.name, variant_trait_v<typename std::decay<T>::type>, what.value, {what.help}});
    return true;
  }

  static bool prepare(InitContext& context, Configurable<T>& what)
  {
    what.value = context.options().get<T>(what.name.c_str());
    return true;
  }
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
DataProcessorSpec adaptAnalysisTask(char const* name, Args&&... args)
{
  TH1::AddDirectory(false);
  auto task = std::make_shared<T>(std::forward<Args>(args)...);
  auto hash = compile_time_hash(name);

  std::vector<OutputSpec> outputs;
  std::vector<ConfigParamSpec> options;

  auto tupledTask = o2::framework::to_tuple_refs(*task.get());
  static_assert(has_process<T>::value || has_run<T>::value || has_init<T>::value,
                "At least one of process(...), T::run(...), init(...) must be defined");

  std::vector<InputSpec> inputs;
  std::vector<ExpressionInfo> expressionInfos;

  if constexpr (has_process<T>::value) {
    // this pushes (I,schemaPtr,nullptr) into expressionInfos for arguments that are Filtered/filtered_iterators
    AnalysisDataProcessorBuilder::inputsFromArgs(&T::process, inputs, expressionInfos);
    // here the FilterManager will prepare the gandiva trees matched to schemas and put the pointers into expressionInfos
    std::apply([&expressionInfos](auto&... x) {
      return (FilterManager<std::decay_t<decltype(x)>>::createExpressionTrees(x, expressionInfos), ...);
    },
               tupledTask);
  }

  std::apply([&outputs, &hash](auto&... x) { return (OutputManager<std::decay_t<decltype(x)>>::appendOutput(outputs, x, hash), ...); }, tupledTask);
  std::apply([&options, &hash](auto&... x) { return (OptionManager<std::decay_t<decltype(x)>>::appendOption(options, x), ...); }, tupledTask);

  auto algo = AlgorithmSpec::InitCallback{[task, expressionInfos](InitContext& ic) {
    auto tupledTask = o2::framework::to_tuple_refs(*task.get());
    std::apply([&ic](auto&&... x) { return (OptionManager<std::decay_t<decltype(x)>>::prepare(ic, x), ...); }, tupledTask);

    auto& callbacks = ic.services().get<CallbackService>();
    auto endofdatacb = [task](EndOfStreamContext& eosContext) {
      auto tupledTask = o2::framework::to_tuple_refs(*task.get());
      std::apply([&eosContext](auto&&... x) { return (OutputManager<std::decay_t<decltype(x)>>::postRun(eosContext, x), ...); }, tupledTask);
      eosContext.services().get<ControlService>().readyToQuit(QuitRequest::Me);
    };
    callbacks.set(CallbackService::Id::EndOfStream, endofdatacb);

    if constexpr (has_init<T>::value) {
      task->init(ic);
    }
    return [task, expressionInfos](ProcessingContext& pc) {
      auto tupledTask = o2::framework::to_tuple_refs(*task.get());
      std::apply([&pc](auto&&... x) { return (OutputManager<std::decay_t<decltype(x)>>::prepare(pc, x), ...); }, tupledTask);
      if constexpr (has_run<T>::value) {
        task->run(pc);
      }
      if constexpr (has_process<T>::value) {
        AnalysisDataProcessorBuilder::invokeProcess(*(task.get()), pc.inputs(), &T::process, expressionInfos);
      }
      std::apply([&pc](auto&&... x) { return (OutputManager<std::decay_t<decltype(x)>>::finalize(pc, x), ...); }, tupledTask);
    };
  }};

  DataProcessorSpec spec{
    name,
    // FIXME: For the moment we hardcode this. We could build
    // this list from the list of methods actually implemented in the
    // task itself.
    inputs,
    outputs,
    algo,
    options};
  return spec;
}

} // namespace o2::framework
#endif // FRAMEWORK_ANALYSISTASK_H_
