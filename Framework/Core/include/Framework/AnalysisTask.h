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

#include "Framework/Kernels.h"
#include "Framework/AlgorithmSpec.h"
#include "Framework/AnalysisDataModel.h"
#include "Framework/CallbackService.h"
#include "Framework/ControlService.h"
#include "Framework/ConfigParamSpec.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/Expressions.h"
#include "../src/ExpressionHelpers.h"
#include "Framework/EndOfStreamContext.h"
#include "Framework/Logger.h"
#include "Framework/HistogramRegistry.h"
#include "Framework/StructToTuple.h"
#include "Framework/FunctionalHelpers.h"
#include "Framework/Traits.h"
#include "Framework/VariantHelpers.h"
#include "Framework/OutputObjHeader.h"
#include "Framework/RootConfigParamHelpers.h"

#include <arrow/compute/kernel.h>
#include <arrow/table.h>
#include <gandiva/node.h>
#include <type_traits>
#include <utility>
#include <memory>
#include <sstream>
#include <iomanip>
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

/// This helper class allows you to declare things which will be created by a
/// given analysis task. Notice how the actual cursor is implemented by the
/// means of the WritingCursor helper class, from which produces actually
/// derives.
template <typename... C>
struct Produces<soa::Table<C...>> : WritingCursor<typename soa::FilterPersistentColumns<soa::Table<C...>>::persistent_table_t> {
  using table_t = soa::Table<C...>;
  using metadata = typename aod::MetadataTrait<table_t>::metadata;

  // @return the associated OutputSpec
  OutputSpec const spec()
  {
    return OutputSpec{OutputLabel{metadata::tableLabel()}, metadata::origin(), metadata::description()};
  }

  OutputRef ref()
  {
    return OutputRef{metadata::tableLabel(), 0};
  }
};

/// Base template for table transformation declarations
template <typename T>
struct TransformTable {
  using metadata = typename aod::MetadataTrait<T>::metadata;
  using originals = typename metadata::originals;

  template <typename O>
  InputSpec const base_spec()
  {
    using o_metadata = typename aod::MetadataTrait<O>::metadata;
    return InputSpec{
      o_metadata::tableLabel(),
      header::DataOrigin{o_metadata::origin()},
      header::DataDescription{o_metadata::description()}};
  }

  template <typename... Os>
  std::vector<InputSpec> const base_specs_impl(framework::pack<Os...>)
  {
    return {base_spec<Os>()...};
  }

  std::vector<InputSpec> const base_specs()
  {
    return base_specs_impl(originals{});
  }

  OutputSpec const spec() const
  {
    return OutputSpec{OutputLabel{metadata::tableLabel()}, metadata::origin(), metadata::description()};
  }

  OutputRef ref() const
  {
    return OutputRef{metadata::tableLabel(), 0};
  }

  T* operator->()
  {
    return table.get();
  }
  T& operator*()
  {
    return *table.get();
  }

  auto asArrowTable()
  {
    return table->asArrowTable();
  }
  std::shared_ptr<T> table = nullptr;
};

/// This helper struct allows you to declare extended tables which should be
/// created by the task (as opposed to those pre-defined by data model)
template <typename T>
struct Spawns : TransformTable<T> {
  using metadata = typename TransformTable<T>::metadata;
  using originals = typename metadata::originals;
  using expression_pack_t = typename metadata::expression_pack_t;

  constexpr auto pack()
  {
    return expression_pack_t{};
  }
};

/// Policy to control index building
/// Exclusive index: each entry in a row has a valid index
struct IndexExclusive {
  /// Generic builder for in index table
  template <typename... Cs, typename Key, typename T1, typename... T>
  static auto indexBuilder(framework::pack<Cs...>, Key const&, std::tuple<T1, T...> tables)
  {
    static_assert(sizeof...(Cs) == sizeof...(T) + 1, "Number of columns does not coincide with number of supplied tables");
    using tables_t = framework::pack<T...>;
    using first_t = T1;
    auto tail = tuple_tail(tables);
    TableBuilder builder;
    auto cursor = framework::FFL(builder.cursor<o2::soa::Table<Cs...>>());

    std::array<int32_t, sizeof...(T)> values;
    iterator_tuple_t<std::decay_t<T>...> iterators = std::apply(
      [](auto&&... x) {
        return std::make_tuple(x.begin()...);
      },
      tail);

    using rest_it_t = decltype(pack_from_tuple(iterators));

    int32_t idx = -1;
    auto setValue = [&](auto& x) -> bool {
      using type = std::decay_t<decltype(x)>;
      constexpr auto position = framework::has_type_at<type>(rest_it_t{});

      lowerBound<Key>(idx, x);
      if (x == soa::RowViewSentinel{static_cast<uint64_t>(x.mMaxRow)}) {
        return false;
      } else if (x.template getId<Key>() != idx) {
        return false;
      } else {
        values[position] = x.globalIndex();
        ++x;
        return true;
      }
    };

    auto first = std::get<first_t>(tables);
    for (auto& row : first) {
      idx = row.template getId<Key>();

      if (std::apply(
            [](auto&... x) {
              return ((x == soa::RowViewSentinel{static_cast<uint64_t>(x.mMaxRow)}) && ...);
            },
            iterators)) {
        break;
      }

      auto result = std::apply(
        [&](auto&... x) {
          std::array<bool, sizeof...(T)> results{setValue(x)...};
          return (results[framework::has_type_at<std::decay_t<decltype(x)>>(rest_it_t{})] && ...);
        },
        iterators);

      if (result) {
        cursor(0, row.globalIndex(), values[framework::has_type_at<T>(tables_t{})]...);
      }
    }
    return builder.finalize();
  }
};
/// Sparse index: values in a row can be (-1), index table is isomorphic (joinable)
/// to T1
struct IndexSparse {
  template <typename... Cs, typename Key, typename T1, typename... T>
  static auto indexBuilder(framework::pack<Cs...>, Key const&, std::tuple<T1, T...> tables)
  {
    static_assert(sizeof...(Cs) == sizeof...(T) + 1, "Number of columns does not coincide with number of supplied tables");
    using tables_t = framework::pack<T...>;
    using first_t = T1;
    auto tail = tuple_tail(tables);
    TableBuilder builder;
    auto cursor = framework::FFL(builder.cursor<o2::soa::Table<Cs...>>());

    std::array<int32_t, sizeof...(T)> values;

    iterator_tuple_t<std::decay_t<T>...> iterators = std::apply(
      [](auto&&... x) {
        return std::make_tuple(x.begin()...);
      },
      tail);

    using rest_it_t = decltype(pack_from_tuple(iterators));

    int32_t idx = -1;
    auto setValue = [&](auto& x) -> bool {
      using type = std::decay_t<decltype(x)>;
      constexpr auto position = framework::has_type_at<type>(rest_it_t{});

      lowerBound<Key>(idx, x);
      if (x == soa::RowViewSentinel{static_cast<uint64_t>(x.mMaxRow)}) {
        values[position] = -1;
        return false;
      } else if (x.template getId<Key>() != idx) {
        values[position] = -1;
        return false;
      } else {
        values[position] = x.globalIndex();
        ++x;
        return true;
      }
    };

    auto first = std::get<first_t>(tables);
    for (auto& row : first) {
      idx = row.template getId<Key>();
      std::apply(
        [&](auto&... x) {
          (setValue(x), ...);
        },
        iterators);

      cursor(0, row.globalIndex(), values[framework::has_type_at<T>(tables_t{})]...);
    }
    return builder.finalize();
  }
};

/// This helper struct allows you to declare index tables to be created in a task
template <typename T, typename IP = IndexSparse>
struct Builds : TransformTable<T> {
  using metadata = typename TransformTable<T>::metadata;
  using originals = typename metadata::originals;
  using Key = typename T::indexing_t;
  using H = typename T::first_t;
  using Ts = typename T::rest_t;
  using index_pack_t = typename metadata::index_pack_t;

  constexpr auto pack()
  {
    return index_pack_t{};
  }

  template <typename... Cs, typename Key, typename T1, typename... Ts>
  auto build(framework::pack<Cs...>, Key const& key, std::tuple<T1, Ts...> tables)
  {
    this->table = std::make_shared<T>(IP::indexBuilder(framework::pack<Cs...>{}, key, tables));
    return (this->table != nullptr);
  }
};

template <typename T>
using BuildsExclusive = Builds<T, IndexExclusive>;

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

  OutputObj(T&& t, OutputObjHandlingPolicy policy_ = OutputObjHandlingPolicy::AnalysisObject)
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
    header::DataDescription desc{};
    auto lhash = compile_time_hash(label.c_str());
    std::memset(desc.str, '_', 16);
    std::stringstream s;
    s << std::hex << lhash;
    s << std::hex << mTaskHash;
    s << std::hex << reinterpret_cast<uint64_t>(this);
    std::memcpy(desc.str, s.str().c_str(), 12);
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

/// This helper allows you to fetch a Sevice from the context or
/// by using some singleton. This hopefully will hide the Singleton and
/// We will be able to retrieve it in a more thread safe manner later on.
template <typename T>
struct Service {
  T* service;

  T* operator->() const
  {
    return service;
  }
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
  T const* operator->() const
  {
    return &value;
  }
};

template <typename T>
struct Partition {
  Partition(expressions::Node&& node)
  {
    auto filter = expressions::Filter(std::move(node));
    auto schema = o2::soa::createSchemaFromColumns(typename T::persistent_columns_t{});
    expressions::Operations ops = createOperations(filter);
    if (isSchemaCompatible(schema, ops)) {
      mTree = createExpressionTree(ops, schema);
    } else {
      throw std::runtime_error("Partition filter does not match declared table type");
    }
  }

  void setTable(const T& table)
  {
    if constexpr (soa::is_soa_filtered_t<std::decay_t<T>>::value) {
      mFiltered.reset(new o2::soa::Filtered<T>{{table}, mTree});
    } else {
      mFiltered.reset(new o2::soa::Filtered<T>{{table.asArrowTable()}, mTree});
    }
  }

  template <typename... Ts>
  void bindExternalIndices(Ts*... tables)
  {
    if (mFiltered != nullptr) {
      mFiltered->bindExternalIndices(tables...);
    }
  }

  template <typename T2>
  void getBoundToExternalIndices(T2& table)
  {
    if (mFiltered != nullptr) {
      table.bindExternalIndices(mFiltered.get());
    }
  }

  gandiva::NodePtr mTree;
  std::unique_ptr<o2::soa::Filtered<T>> mFiltered;

  using filtered_iterator = typename o2::soa::Filtered<T>::iterator;
  using filtered_const_iterator = typename o2::soa::Filtered<T>::const_iterator;
  inline filtered_iterator begin()
  {
    return mFiltered->begin();
  }
  inline o2::soa::RowViewSentinel end()
  {
    return mFiltered->end();
  }
  inline filtered_const_iterator begin() const
  {
    return mFiltered->begin();
  }
  inline o2::soa::RowViewSentinel end() const
  {
    return mFiltered->end();
  }

  int64_t size() const
  {
    return mFiltered->size();
  }
};

template <typename ANY>
struct PartitionManager {
  template <typename... T2s>
  static void setPartition(ANY&, T2s&...)
  {
  }

  template <typename... Ts>
  static void bindExternalIndices(ANY&, Ts*...)
  {
  }

  template <typename... Ts>
  static void getBoundToExternalIndices(ANY&, Ts&...)
  {
  }
};

template <typename T>
struct PartitionManager<Partition<T>> {
  template <typename T2>
  static void doSetPartition(Partition<T>& partition, T2& table)
  {
    if constexpr (std::is_same_v<T, T2>) {
      partition.setTable(table);
    }
  }

  template <typename... T2s>
  static void setPartition(Partition<T>& partition, T2s&... tables)
  {
    (doSetPartition(partition, tables), ...);
  }

  template <typename... Ts>
  static void bindExternalIndices(Partition<T>& partition, Ts*... tables)
  {
    partition.bindExternalIndices(tables...);
  }

  template <typename... Ts>
  static void getBoundToExternalIndices(Partition<T>& partition, Ts&... tables)
  {
    partition.getBoundToExternalIndices(tables...);
  }
};

struct AnalysisTask {
};

// Helper struct which builds a DataProcessorSpec from
// the contents of an AnalysisTask...

struct AnalysisDataProcessorBuilder {
  template <typename Arg>
  static void doAppendInputWithMetadata(std::vector<InputSpec>& inputs)
  {
    using metadata = typename aod::MetadataTrait<std::decay_t<Arg>>::metadata;
    static_assert(std::is_same_v<metadata, void> == false,
                  "Could not find metadata. Did you register your type?");
    inputs.push_back({metadata::tableLabel(), metadata::origin(), metadata::description()});
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
      eInfos.push_back({At, o2::soa::createSchemaFromColumns(typename dT::table_t::persistent_columns_t{}), nullptr});
    } else if constexpr (soa::is_soa_iterator_t<dT>::value) {
      if constexpr (std::is_same_v<typename dT::policy_t, soa::FilteredIndexPolicy>) {
        eInfos.push_back({At, o2::soa::createSchemaFromColumns(typename dT::table_t::persistent_columns_t{}), nullptr});
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
      return record.get<TableConsumer>(aod::MetadataTrait<T>::metadata::tableLabel())->asArrowTable();
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
      return typename T::parent_t(std::vector<std::shared_ptr<arrow::Table>>{extractTableFromRecord<Os>(record)...}, info.tree);
    } else {
      return T(std::vector<std::shared_ptr<arrow::Table>>{extractTableFromRecord<Os>(record)...}, info.tree);
    }
  }

  template <typename T, size_t At>
  static auto extractSomethingFromRecord(InputRecord& record, std::vector<ExpressionInfo> const infos)
  {
    using decayed = std::decay_t<T>;

    if constexpr (soa::is_soa_filtered_t<decayed>::value) {
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

  template <typename T, typename C>
  using is_external_index_to_t = std::is_same<typename C::binding_t, T>;

  template <typename G, typename... A>
  struct GroupSlicer {
    using grouping_t = std::decay_t<G>;
    GroupSlicer(G& gt, std::tuple<A...>& at)
      : max{gt.size()},
        mBegin{GroupSlicerIterator(gt, at)}
    {
    }

    struct GroupSlicerSentinel {
      int64_t position;
    };

    struct GroupSlicerIterator {
      using associated_pack_t = framework::pack<A...>;

      GroupSlicerIterator() = default;
      GroupSlicerIterator(GroupSlicerIterator const&) = default;
      GroupSlicerIterator(GroupSlicerIterator&&) = default;
      GroupSlicerIterator& operator=(GroupSlicerIterator const&) = default;
      GroupSlicerIterator& operator=(GroupSlicerIterator&&) = default;

      auto getLabelFromType()
      {
        if constexpr (soa::is_soa_index_table_t<std::decay_t<G>>::value) {
          using T = typename std::decay_t<G>::first_t;
          if constexpr (soa::is_type_with_originals_v<std::decay_t<T>>) {
            using O = typename framework::pack_element_t<0, typename std::decay_t<G>::originals>;
            using groupingMetadata = typename aod::MetadataTrait<O>::metadata;
            return std::string("f") + groupingMetadata::tableLabel() + "ID";
          } else {
            using groupingMetadata = typename aod::MetadataTrait<T>::metadata;
            return std::string("f") + groupingMetadata::tableLabel() + "ID";
          }
        } else if constexpr (soa::is_type_with_originals_v<std::decay_t<G>>) {
          using T = typename framework::pack_element_t<0, typename std::decay_t<G>::originals>;
          using groupingMetadata = typename aod::MetadataTrait<T>::metadata;
          return std::string("f") + groupingMetadata::tableLabel() + "ID";
        } else {
          using groupingMetadata = typename aod::MetadataTrait<std::decay_t<G>>::metadata;
          return std::string("f") + groupingMetadata::tableLabel() + "ID";
        }
      }

      GroupSlicerIterator(G& gt, std::tuple<A...>& at)
        : mAt{&at},
          mGroupingElement{gt.begin()},
          position{0}
      {
        if constexpr (soa::is_soa_filtered_t<std::decay_t<G>>::value) {
          groupSelection = &gt.getSelectedRows();
        }
        auto indexColumnName = getLabelFromType();
        /// prepare slices and offsets for all associated tables that have index
        /// to grouping table
        ///
        auto splitter = [&](auto&& x) {
          using xt = std::decay_t<decltype(x)>;
          constexpr auto index = framework::has_type_at<std::decay_t<decltype(x)>>(associated_pack_t{});
          if (hasIndexTo<std::decay_t<G>>(typename xt::persistent_columns_t{})) {
            auto result = o2::framework::sliceByColumn(indexColumnName.c_str(),
                                                       x.asArrowTable(),
                                                       static_cast<int32_t>(gt.size()),
                                                       &groups[index],
                                                       &offsets[index]);
            if (result.ok() == false) {
              throw std::runtime_error("Cannot split collection");
            }
            if (groups[index].size() != gt.tableSize()) {
              throw std::runtime_error(fmt::format("Splitting collection resulted in different group number ({}) than there is rows in the grouping table ({}).", groups[index].size(), gt.tableSize()));
            };
          }
        };

        std::apply(
          [&](auto&&... x) -> void {
            (splitter(x), ...);
          },
          at);
        /// extract selections from filtered associated tables
        auto extractor = [&](auto&& x) {
          using xt = std::decay_t<decltype(x)>;
          if constexpr (soa::is_soa_filtered_t<xt>::value) {
            constexpr auto index = framework::has_type_at<std::decay_t<decltype(x)>>(associated_pack_t{});
            selections[index] = &x.getSelectedRows();
            starts[index] = selections[index]->begin();
            offsets[index].push_back(std::get<xt>(at).tableSize());
          }
        };
        std::apply(
          [&](auto&&... x) -> void {
            (extractor(x), ...);
          },
          at);
      }

      template <typename B, typename... C>
      constexpr bool hasIndexTo(framework::pack<C...>&&)
      {
        return (isIndexTo<B, C>() || ...);
      }

      template <typename B, typename C>
      constexpr bool isIndexTo()
      {
        if constexpr (soa::is_type_with_binding_v<C>) {
          if constexpr (soa::is_soa_index_table_t<std::decay_t<B>>::value) {
            using T = typename std::decay_t<B>::first_t;
            if constexpr (soa::is_type_with_originals_v<std::decay_t<T>>) {
              using TT = typename framework::pack_element_t<0, typename std::decay_t<T>::originals>;
              return std::is_same_v<typename C::binding_t, TT>;
            } else {
              using TT = std::decay_t<T>;
              return std::is_same_v<typename C::binding_t, TT>;
            }
          } else {
            if constexpr (soa::is_type_with_originals_v<std::decay_t<B>>) {
              using TT = typename framework::pack_element_t<0, typename std::decay_t<B>::originals>;
              return std::is_same_v<typename C::binding_t, TT>;
            } else {
              using TT = std::decay_t<B>;
              return std::is_same_v<typename C::binding_t, TT>;
            }
          }
        }
        return false;
      }

      GroupSlicerIterator operator++()
      {
        ++position;
        ++mGroupingElement;
        return *this;
      }

      bool operator==(GroupSlicerSentinel const& other)
      {
        return O2_BUILTIN_UNLIKELY(position == other.position);
      }

      bool operator!=(GroupSlicerSentinel const& other)
      {
        return O2_BUILTIN_LIKELY(position != other.position);
      }

      auto& groupingElement()
      {
        return mGroupingElement;
      }

      GroupSlicerIterator& operator*()
      {
        return *this;
      }

      auto associatedTables()
      {
        return std::make_tuple(prepareArgument<A>()...);
      }

      template <typename A1>
      auto prepareArgument()
      {
        constexpr auto index = framework::has_type_at<A1>(associated_pack_t{});
        if (hasIndexTo<G>(typename std::decay_t<A1>::persistent_columns_t{})) {
          auto pos = position;
          if constexpr (soa::is_soa_filtered_t<std::decay_t<G>>::value) {
            pos = groupSelection[position];
          }
          if constexpr (soa::is_soa_filtered_t<std::decay_t<A1>>::value) {
            auto groupedElementsTable = arrow::util::get<std::shared_ptr<arrow::Table>>(((groups[index])[pos]).value);

            // for each grouping element we need to slice the selection vector
            auto start_iterator = std::lower_bound(starts[index], selections[index]->end(), (offsets[index])[pos]);
            auto stop_iterator = std::lower_bound(start_iterator, selections[index]->end(), (offsets[index])[pos + 1]);
            starts[index] = stop_iterator;
            soa::SelectionVector slicedSelection{start_iterator, stop_iterator};
            std::transform(slicedSelection.begin(), slicedSelection.end(), slicedSelection.begin(),
                           [&](int64_t idx) {
                             return idx - static_cast<int64_t>((offsets[index])[pos]);
                           });

            std::decay_t<A1> typedTable{{groupedElementsTable}, std::move(slicedSelection), (offsets[index])[pos]};
            return typedTable;
          } else {
            auto groupedElementsTable = arrow::util::get<std::shared_ptr<arrow::Table>>(((groups[index])[pos]).value);
            std::decay_t<A1> typedTable{{groupedElementsTable}, (offsets[index])[pos]};
            return typedTable;
          }
        } else {
          return std::get<A1>(*mAt);
        }
        O2_BUILTIN_UNREACHABLE();
      }

      std::tuple<A...>* mAt;
      typename grouping_t::iterator mGroupingElement;
      uint64_t position = 0;
      soa::SelectionVector* groupSelection = nullptr;

      std::array<std::vector<arrow::Datum>, sizeof...(A)> groups;
      std::array<std::vector<uint64_t>, sizeof...(A)> offsets;
      std::array<soa::SelectionVector const*, sizeof...(A)> selections;
      std::array<soa::SelectionVector::const_iterator, sizeof...(A)> starts;
    };

    GroupSlicerIterator& begin()
    {
      return mBegin;
    }

    GroupSlicerSentinel end()
    {
      return GroupSlicerSentinel{max};
    }
    int64_t max;
    GroupSlicerIterator mBegin;
  };

  template <typename Task, typename R, typename C, typename Grouping, typename... Associated>
  static void invokeProcess(Task& task, InputRecord& inputs, R (C::*)(Grouping, Associated...), std::vector<ExpressionInfo> const& infos)
  {
    auto tupledTask = o2::framework::to_tuple_refs(task);
    using G = std::decay_t<Grouping>;
    auto groupingTable = AnalysisDataProcessorBuilder::bindGroupingTable(inputs, &C::process, infos);

    // set filtered tables for partitions with grouping
    std::apply([&groupingTable](auto&... x) {
      (PartitionManager<std::decay_t<decltype(x)>>::setPartition(x, groupingTable), ...);
    },
               tupledTask);

    if constexpr (sizeof...(Associated) == 0) {
      // single argument to process
      std::apply([&groupingTable](auto&... x) {
        (PartitionManager<std::decay_t<decltype(x)>>::bindExternalIndices(x, &groupingTable), ...);
        (PartitionManager<std::decay_t<decltype(x)>>::getBoundToExternalIndices(x, groupingTable), ...);
      },
                 tupledTask);
      if constexpr (soa::is_soa_iterator_t<G>::value) {
        for (auto& element : groupingTable) {
          task.process(*element);
        }
      } else {
        static_assert(soa::is_soa_table_like_t<G>::value,
                      "Single argument of process() should be a table-like or an iterator");
        task.process(groupingTable);
      }
    } else {
      // multiple arguments to process
      static_assert(((soa::is_soa_iterator_t<std::decay_t<Associated>>::value == false) && ...),
                    "Associated arguments of process() should not be iterators");
      auto associatedTables = AnalysisDataProcessorBuilder::bindAssociatedTables(inputs, &C::process, infos);
      auto binder = [&](auto&& x) {
        x.bindExternalIndices(&groupingTable, &std::get<std::decay_t<Associated>>(associatedTables)...);
        std::apply([&x](auto&... t) {
          (PartitionManager<std::decay_t<decltype(t)>>::setPartition(t, x), ...);
          (PartitionManager<std::decay_t<decltype(t)>>::bindExternalIndices(t, &x), ...);
          (PartitionManager<std::decay_t<decltype(t)>>::getBoundToExternalIndices(t, x), ...);
        },
                   tupledTask);
      };
      groupingTable.bindExternalIndices(&std::get<std::decay_t<Associated>>(associatedTables)...);

      // always pre-bind full tables to support index hierarchy
      std::apply(
        [&](auto&&... x) {
          (binder(x), ...);
        },
        associatedTables);

      if constexpr (soa::is_soa_iterator_t<std::decay_t<G>>::value) {
        // grouping case
        auto slicer = GroupSlicer(groupingTable, associatedTables);
        for (auto& slice : slicer) {
          auto associatedSlices = slice.associatedTables();

          std::apply(
            [&](auto&&... x) {
              (binder(x), ...);
            },
            associatedSlices);

          // bind partitions and grouping table
          std::apply([&groupingTable](auto&... x) {
            (PartitionManager<std::decay_t<decltype(x)>>::bindExternalIndices(x, &groupingTable), ...);
            (PartitionManager<std::decay_t<decltype(x)>>::getBoundToExternalIndices(x, groupingTable), ...);
          },
                     tupledTask);

          invokeProcessWithArgs(task, slice.groupingElement(), associatedSlices);
        }
      } else {
        // non-grouping case

        // bind partitions and grouping table
        std::apply([&groupingTable](auto&... x) {
          (PartitionManager<std::decay_t<decltype(x)>>::bindExternalIndices(x, &groupingTable), ...);
          (PartitionManager<std::decay_t<decltype(x)>>::getBoundToExternalIndices(x, groupingTable), ...);
        },
                   tupledTask);

        invokeProcessWithArgs(task, groupingTable, associatedTables);
      }
    }
  }

  template <typename T, typename G, typename... A>
  static void invokeProcessWithArgs(T& task, G g, std::tuple<A...>& at)
  {
    task.process(g, std::get<A>(at)...);
  }
};

template <typename ANY>
struct FilterManager {
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

/// SFINAE placeholder
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

/// Produces specialization
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

/// HistogramRegistry specialization
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

/// OutputObj specialization
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

/// Spawns specializations
template <typename O>
static auto extractOriginal(ProcessingContext& pc)
{
  return pc.inputs().get<TableConsumer>(aod::MetadataTrait<O>::metadata::tableLabel())->asArrowTable();
}

template <typename... Os>
static std::vector<std::shared_ptr<arrow::Table>> extractOriginals(framework::pack<Os...>, ProcessingContext& pc)
{
  return {extractOriginal<Os>(pc)...};
}

template <typename T>
struct OutputManager<Spawns<T>> {
  static bool appendOutput(std::vector<OutputSpec>& outputs, Spawns<T>& what, uint32_t)
  {
    outputs.emplace_back(what.spec());
    return true;
  }

  static bool prepare(ProcessingContext& pc, Spawns<T>& what)
  {
    using metadata = typename std::decay_t<decltype(what)>::metadata;
    auto original_table = soa::ArrowHelpers::joinTables(extractOriginals(typename metadata::originals{}, pc));
    what.table = std::make_shared<T>(o2::soa::spawner(what.pack(), original_table.get()));
    return true;
  }

  static bool finalize(ProcessingContext&, Spawns<T>&)
  {
    return true;
  }

  static bool postRun(EndOfStreamContext& eosc, Spawns<T>& what)
  {
    using metadata = typename std::decay_t<decltype(what)>::metadata;
    eosc.outputs().adopt(Output{metadata::origin(), metadata::description()}, what.asArrowTable());
    return true;
  }
};

/// Builds specialization
template <typename O>
static auto extractTypedOriginal(ProcessingContext& pc)
{
  ///FIXME: this should be done in invokeProcess() as some of the originals may be compound tables
  return O{pc.inputs().get<TableConsumer>(aod::MetadataTrait<O>::metadata::tableLabel())->asArrowTable()};
}

template <typename... Os>
static auto extractOriginalsTuple(framework::pack<Os...>, ProcessingContext& pc)
{
  return std::make_tuple(extractTypedOriginal<Os>(pc)...);
}

template <typename T, typename P>
struct OutputManager<Builds<T, P>> {
  static bool appendOutput(std::vector<OutputSpec>& outputs, Builds<T, P>& what, uint32_t)
  {
    outputs.emplace_back(what.spec());
    return true;
  }

  static bool prepare(ProcessingContext& pc, Builds<T, P>& what)
  {
    using metadata = typename std::decay_t<decltype(what)>::metadata;
    return what.build(typename metadata::index_pack_t{},
                      extractTypedOriginal<typename metadata::Key>(pc),
                      extractOriginalsTuple(typename metadata::originals{}, pc));
  }

  static bool finalize(ProcessingContext&, Builds<T, P>&)
  {
    return true;
  }

  static bool postRun(EndOfStreamContext& eosc, Builds<T, P>& what)
  {
    using metadata = typename std::decay_t<decltype(what)>::metadata;
    eosc.outputs().adopt(Output{metadata::origin(), metadata::description()}, what.asArrowTable());
    return true;
  }
};

template <typename T>
class has_instance
{
  typedef char one;
  struct two {
    char x[2];
  };

  template <typename C>
  static one test(decltype(&C::instance));
  template <typename C>
  static two test(...);

 public:
  enum { value = sizeof(test<T>(nullptr)) == sizeof(char) };
};

template <typename T>
struct ServiceManager {
  template <typename ANY>
  static bool prepare(InitContext&, ANY&)
  {
    return false;
  }
};

template <typename T>
struct ServiceManager<Service<T>> {
  static bool prepare(InitContext& context, Service<T>& service)
  {
    if constexpr (has_instance<T>::value) {
      service.service = &(T::instance()); // Sigh...
      return true;
    } else {
      service.service = context.services().get<T>();
      return true;
    }
    return false;
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
    if constexpr (variant_trait_v<typename std::decay<T>::type> != VariantType::Unknown) {
      options.emplace_back(ConfigParamSpec{what.name, variant_trait_v<typename std::decay<T>::type>, what.value, {what.help}});
    } else {
      auto specs = RootConfigParamHelpers::asConfigParamSpecs<T>(what.name, what.value);
      options.insert(options.end(), specs.begin(), specs.end());
    }
    return true;
  }

  static bool prepare(InitContext& context, Configurable<T>& what)
  {
    if constexpr (variant_trait_v<typename std::decay<T>::type> != VariantType::Unknown) {
      what.value = context.options().get<T>(what.name.c_str());
    } else {
      auto pt = context.options().get<boost::property_tree::ptree>(what.name.c_str());
      what.value = RootConfigParamHelpers::as<T>(pt);
    }
    return true;
  }
};

/// Manager template to facilitate extended tables spawning
template <typename T>
struct SpawnManager {
  static bool requestInputs(std::vector<InputSpec>&, T const&) { return false; }
};

template <typename TABLE>
struct SpawnManager<Spawns<TABLE>> {
  static bool requestInputs(std::vector<InputSpec>& inputs, Spawns<TABLE>& spawns)
  {
    auto base_specs = spawns.base_specs();
    for (auto& base_spec : base_specs) {
      if (std::find_if(inputs.begin(), inputs.end(), [&](InputSpec const& spec) { return base_spec.binding == spec.binding; }) == inputs.end()) {
        inputs.emplace_back(base_spec);
      }
    }
    return true;
  }
};

/// Manager template for building index tables
template <typename T>
struct IndexManager {
  static bool requestInputs(std::vector<InputSpec>&, T const&) { return false; };
};

template <typename IDX, typename P>
struct IndexManager<Builds<IDX, P>> {
  static bool requestInputs(std::vector<InputSpec>& inputs, Builds<IDX, P>& builds)
  {
    auto base_specs = builds.base_specs();
    for (auto& base_spec : base_specs) {
      if (std::find_if(inputs.begin(), inputs.end(), [&](InputSpec const& spec) { return base_spec.binding == spec.binding; }) == inputs.end()) {
        inputs.emplace_back(base_spec);
      }
    }
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
  //request base tables for spawnable extended tables
  std::apply([&inputs](auto&... x) {
    return (SpawnManager<std::decay_t<decltype(x)>>::requestInputs(inputs, x), ...);
  },
             tupledTask);

  //request base tables for indices to be built
  std::apply([&inputs](auto&... x) {
    return (IndexManager<std::decay_t<decltype(x)>>::requestInputs(inputs, x), ...);
  },
             tupledTask);

  std::apply([&outputs, &hash](auto&... x) { return (OutputManager<std::decay_t<decltype(x)>>::appendOutput(outputs, x, hash), ...); }, tupledTask);
  std::apply([&options, &hash](auto&... x) { return (OptionManager<std::decay_t<decltype(x)>>::appendOption(options, x), ...); }, tupledTask);

  auto algo = AlgorithmSpec::InitCallback{[task, expressionInfos](InitContext& ic) {
    auto tupledTask = o2::framework::to_tuple_refs(*task.get());
    std::apply([&ic](auto&&... x) { return (OptionManager<std::decay_t<decltype(x)>>::prepare(ic, x), ...); }, tupledTask);
    std::apply([&ic](auto&&... x) { return (ServiceManager<std::decay_t<decltype(x)>>::prepare(ic, x), ...); }, tupledTask);

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
