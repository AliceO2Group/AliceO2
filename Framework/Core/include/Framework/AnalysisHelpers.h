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
#ifndef o2_framework_AnalysisHelpers_H_DEFINED
#define o2_framework_AnalysisHelpers_H_DEFINED

#include "Framework/Traits.h"
#include "Framework/TableBuilder.h"
#include "Framework/AnalysisDataModel.h"
#include "Framework/OutputSpec.h"
#include "Framework/OutputRef.h"
#include "Framework/InputSpec.h"
#include "Framework/OutputObjHeader.h"
#include "Framework/StringHelpers.h"
#include "Framework/Output.h"
#include "Framework/IndexBuilderHelpers.h"
#include "Framework/Plugins.h"

#include <string>
namespace o2::framework
{
class TableConsumer;

template <typename T>
struct WritingCursor {
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

  void setLabel(const char* label)
  {
    mBuilder->setLabel(label);
  }

  /// reserve @a size rows when filling, so that we do not
  /// spend time reallocating the buffers.
  void reserve(int64_t size)
  {
    mBuilder->reserve(typename persistent_table_t::column_types{}, size);
  }

  decltype(FFL(std::declval<cursor_t>())) cursor;

 private:
  template <typename T>
  static decltype(auto) extract(T const& arg)
  {
    if constexpr (soa::is_soa_iterator_v<T>) {
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
template <typename T>
struct Produces : WritingCursor<typename soa::PackToTable<typename T::table_t::persistent_columns_t>::table> {
  using table_t = T;
  using metadata = typename aod::MetadataTrait<T>::metadata;

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

template <template <typename...> class T, typename... C>
struct Produces<T<C...>> : WritingCursor<typename soa::PackToTable<typename T<C...>::table_t::persistent_columns_t>::table> {
  using table_t = T<C...>;
  using metadata = typename aod::MetadataTrait<table_t>::metadata;

  // @return the associated OutputSpec
  OutputSpec const spec()
  {
    return OutputSpec{OutputLabel{metadata::tableLabel()}, metadata::origin(), metadata::description(), metadata::version()};
  }

  OutputRef ref()
  {
    return OutputRef{metadata::tableLabel(), metadata::version()};
  }
};

/// Helper template for table transformations
template <typename METADATA>
struct TableTransform {
  using SOURCES = typename METADATA::sources;
  using ORIGINALS = typename METADATA::originals;

  using metadata = METADATA;
  using sources = SOURCES;

  constexpr auto sources_pack() const
  {
    return SOURCES{};
  }

  constexpr auto originals_pack() const
  {
    return ORIGINALS{};
  }

  template <typename Oi>
  constexpr auto base_spec() const
  {
    using o_metadata = typename aod::MetadataTrait<Oi>::metadata;
    return InputSpec{
      o_metadata::tableLabel(),
      header::DataOrigin{o_metadata::origin()},
      header::DataDescription{o_metadata::description()},
      o_metadata::version()};
  }

  template <typename... Os>
  std::vector<InputSpec> base_specs_impl(framework::pack<Os...>) const
  {
    return {base_spec<Os>()...};
  }

  std::vector<InputSpec> base_specs() const
  {
    return base_specs_impl(sources_pack());
  }

  constexpr auto spec() const
  {
    return OutputSpec{OutputLabel{METADATA::tableLabel()}, METADATA::origin(), METADATA::description(), METADATA::version()};
  }

  constexpr auto output() const
  {
    return Output{METADATA::origin(), METADATA::description(), METADATA::version()};
  }

  constexpr auto ref() const
  {
    return OutputRef{METADATA::tableLabel(), METADATA::version()};
  }
};

/// This helper struct allows you to declare extended tables which should be
/// created by the task (as opposed to those pre-defined by data model)
template <typename T>
struct Spawns : TableTransform<typename aod::MetadataTrait<framework::pack_head_t<typename T::originals>>::metadata> {
  using extension_t = framework::pack_head_t<typename T::originals>;
  using base_table_t = typename aod::MetadataTrait<extension_t>::metadata::base_table_t;
  using expression_pack_t = typename aod::MetadataTrait<extension_t>::metadata::expression_pack_t;

  constexpr auto pack()
  {
    return expression_pack_t{};
  }

  typename T::table_t* operator->()
  {
    return table.get();
  }
  typename T::table_t const& operator*() const
  {
    return *table;
  }

  auto asArrowTable()
  {
    return extension->asArrowTable();
  }
  std::shared_ptr<typename T::table_t> table = nullptr;
  std::shared_ptr<extension_t> extension = nullptr;
};

/// Policy to control index building
/// Exclusive index: each entry in a row has a valid index
/// Sparse index: values in a row can be (-1), index table is isomorphic (joinable)
/// to T1
struct Exclusive {
};
struct Sparse {
};

namespace
{
template <typename T, typename Key>
inline std::shared_ptr<arrow::ChunkedArray> getIndexToKey(arrow::Table* table)
{
  using IC = framework::pack_element_t<framework::has_type_at_conditional<soa::is_binding_compatible, Key>(typename T::external_index_columns_t{}), typename T::external_index_columns_t>;
  return table->column(framework::has_type_at<IC>(typename T::persistent_columns_t{}));
}

template <typename C>
struct ColumnTrait {
  static_assert(framework::is_base_of_template_v<o2::soa::Column, C>, "Not a column type!");
  using column_t = C;

  static constexpr auto listSize()
  {
    if constexpr (std::is_same_v<typename C::type, std::vector<int>>) {
      return -1;
    } else if constexpr (std::is_same_v<int[2], typename C::type>) {
      return 2;
    } else {
      return 1;
    }
  }

  template <typename T, typename Key>
  static std::shared_ptr<SelfIndexColumnBuilder> makeColumnBuilder(arrow::Table* table, arrow::MemoryPool* pool)
  {
    if constexpr (!std::is_same_v<T, Key>) {
      return std::make_shared<IndexColumnBuilder>(getIndexToKey<T, Key>(table), C::columnLabel(), listSize(), pool);
    } else {
      return std::make_shared<SelfIndexColumnBuilder>(C::columnLabel(), pool);
    }
  }
};

template <typename Key, typename C>
struct Reduction {
  using type = typename std::conditional<soa::is_binding_compatible_v<Key, typename C::binding_t>(), SelfIndexColumnBuilder, IndexColumnBuilder>::type;
};
} // namespace

template <typename Kind>
struct IndexBuilder {
  template <typename Key, typename C1, typename... Cs, typename T1, typename... Ts>
  static auto indexBuilder(const char* label, std::vector<std::shared_ptr<arrow::Table>>&& tables, framework::pack<C1, Cs...>, framework::pack<T1, Ts...>)
  {
    auto pool = arrow::default_memory_pool();
    SelfIndexColumnBuilder self{C1::columnLabel(), pool};
    std::unique_ptr<ChunkedArrayIterator> keyIndex = nullptr;
    int64_t counter = 0;
    if constexpr (!std::is_same_v<T1, Key>) {
      keyIndex = std::make_unique<ChunkedArrayIterator>(getIndexToKey<T1, Key>(tables[0].get()));
    }

    std::array<std::shared_ptr<framework::SelfIndexColumnBuilder>, sizeof...(Cs)> columnBuilders{ColumnTrait<Cs>::template makeColumnBuilder<framework::pack_element_t<framework::has_type_at_v<Cs>(framework::pack<Cs...>{}), framework::pack<Ts...>>, Key>(
      tables[framework::has_type_at_v<Cs>(framework::pack<Cs...>{}) + 1].get(),
      pool)...};
    std::array<bool, sizeof...(Cs)> finds;

    for (counter = 0; counter < tables[0]->num_rows(); ++counter) {
      auto idx = -1;
      if constexpr (std::is_same_v<T1, Key>) {
        idx = counter;
      } else {
        idx = keyIndex->valueAt(counter);
      }
      finds = {std::static_pointer_cast<typename Reduction<Key, Cs>::type>(columnBuilders[framework::has_type_at_v<Cs>(framework::pack<Cs...>{})])->template find<Cs>(idx)...};
      if constexpr (std::is_same_v<Kind, Sparse>) {
        (std::static_pointer_cast<typename Reduction<Key, Cs>::type>(columnBuilders[framework::has_type_at_v<Cs>(framework::pack<Cs...>{})])->template fill<Cs>(idx), ...);
        self.fill<C1>(counter);
      } else if constexpr (std::is_same_v<Kind, Exclusive>) {
        if (std::none_of(finds.begin(), finds.end(), [](bool const x) { return x == false; })) {
          (std::static_pointer_cast<typename Reduction<Key, Cs>::type>(columnBuilders[framework::has_type_at_v<Cs>(framework::pack<Cs...>{})])->template fill<Cs>(idx), ...);
          self.fill<C1>(counter);
        }
      }
    }

    return makeArrowTable(label,
                          {self.template result<C1>(), std::static_pointer_cast<typename Reduction<Key, Cs>::type>(columnBuilders[framework::has_type_at_v<Cs>(framework::pack<Cs...>{})])->template result<Cs>()...},
                          {self.field(), std::static_pointer_cast<typename Reduction<Key, Cs>::type>(columnBuilders[framework::has_type_at_v<Cs>(framework::pack<Cs...>{})])->field()...});
  }

  template <typename IDX, typename Key, typename T1, typename... T>
  static auto makeIndex(Key const& key, std::tuple<T1, T...>&& tables)
  {
    auto t = IDX{indexBuilder(o2::aod::MetadataTrait<IDX>::metadata::tableLabel(),
                              typename o2::aod::MetadataTrait<IDX>::metadata::index_pack_t{},
                              key,
                              std::make_tuple(std::decay_t<T1>{{std::get<T1>(tables)}}, std::decay_t<T>{{std::get<T>(tables)}}...))};
    t.bindExternalIndices(&key, &std::get<T1>(tables), &std::get<T>(tables)...);
    return t;
  }
};

/// This helper struct allows you to declare index tables to be created in a task
template <typename T>
struct Builds : TableTransform<typename aod::MetadataTrait<T>::metadata> {
  using IP = std::conditional_t<aod::MetadataTrait<T>::metadata::exclusive, IndexBuilder<Exclusive>, IndexBuilder<Sparse>>;
  using Key = typename T::indexing_t;
  using H = typename T::first_t;
  using Ts = typename T::rest_t;
  using index_pack_t = typename aod::MetadataTrait<T>::metadata::index_pack_t;

  T* operator->()
  {
    return table.get();
  }
  T const& operator*() const
  {
    return *table;
  }

  auto asArrowTable()
  {
    return table->asArrowTable();
  }
  std::shared_ptr<T> table = nullptr;

  constexpr auto pack()
  {
    return index_pack_t{};
  }

  template <typename Key, typename... Cs, typename... Ts>
  auto build(framework::pack<Cs...>, framework::pack<Ts...>, std::vector<std::shared_ptr<arrow::Table>>&& tables)
  {
    this->table = std::make_shared<T>(IP::template indexBuilder<Key>(aod::MetadataTrait<T>::metadata::tableLabel(), std::forward<std::vector<std::shared_ptr<arrow::Table>>>(tables), framework::pack<Cs...>{}, framework::pack<Ts...>{}));
    return (this->table != nullptr);
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

  OutputObj(T&& t, OutputObjHandlingPolicy policy_ = OutputObjHandlingPolicy::AnalysisObject, OutputObjSourceType sourceType_ = OutputObjSourceType::OutputObjSource)
    : object(std::make_shared<T>(t)),
      label(t.GetName()),
      policy{policy_},
      sourceType{sourceType_},
      mTaskHash{0}
  {
  }

  OutputObj(std::string const& label_, OutputObjHandlingPolicy policy_ = OutputObjHandlingPolicy::AnalysisObject, OutputObjSourceType sourceType_ = OutputObjSourceType::OutputObjSource)
    : object(nullptr),
      label(label_),
      policy{policy_},
      sourceType{sourceType_},
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

  void setObject(std::shared_ptr<T> t)
  {
    object = t;
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
    return OutputSpec{OutputLabel{label}, "ATSK", desc, 0, Lifetime::QA};
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
                     o2::header::Stack{OutputObjHeader{policy, sourceType, mTaskHash}}};
  }

  std::shared_ptr<T> object;
  std::string label;
  OutputObjHandlingPolicy policy;
  OutputObjSourceType sourceType;
  uint32_t mTaskHash;
};

/// This helper allows you to fetch a Sevice from the context or
/// by using some singleton. This hopefully will hide the Singleton and
/// We will be able to retrieve it in a more thread safe manner later on.
template <typename T>
struct Service {
  T* service;

  decltype(auto) operator->() const
  {
    if constexpr (is_base_of_template_v<LoadableServicePlugin, T>) {
      return service->get();
    } else {
      return service;
    }
  }
};

template <typename T>
auto getTableFromFilter(const T& table, soa::SelectionVector&& selection)
{
  if constexpr (soa::is_soa_filtered_v<std::decay_t<T>>) {
    return std::make_unique<o2::soa::Filtered<T>>(std::vector{table}, std::forward<soa::SelectionVector>(selection));
  } else {
    return std::make_unique<o2::soa::Filtered<T>>(std::vector{table.asArrowTable()}, std::forward<soa::SelectionVector>(selection));
  }
}

template <typename T>
struct Partition {
  Partition(expressions::Node&& filter_) : filter{std::forward<expressions::Node>(filter_)}
  {
  }

  Partition(expressions::Node&& filter_, T const& table)
    : filter{std::forward<expressions::Node>(filter_)}
  {
    setTable(table);
  }

  void intializeCaches(std::shared_ptr<arrow::Schema> const& schema)
  {
    if (tree == nullptr) {
      expressions::Operations ops = createOperations(filter);
      if (isSchemaCompatible(schema, ops)) {
        tree = createExpressionTree(ops, schema);
      } else {
        throw std::runtime_error("Partition filter does not match declared table type");
      }
    }
    if (gfilter == nullptr) {
      gfilter = framework::expressions::createFilter(schema, framework::expressions::makeCondition(tree));
    }
  }

  void inline bindTable(T const& table)
  {
    setTable(table);
  }

  void setTable(T const& table)
  {
    intializeCaches(table.asArrowTable()->schema());
    if (dataframeChanged) {
      mFiltered = getTableFromFilter(table, soa::selectionToVector(framework::expressions::createSelection(table.asArrowTable(), gfilter)));
      dataframeChanged = false;
    }
  }

  template <typename... Ts>
  void bindExternalIndices(Ts*... tables)
  {
    if (mFiltered != nullptr) {
      mFiltered->bindExternalIndices(tables...);
    }
  }

  void bindInternalIndices()
  {
    if (mFiltered != nullptr) {
      mFiltered->bindInternalIndices();
    }
  }

  template <typename E>
  void bindInternalIndicesTo(E* ptr)
  {
    if (mFiltered != nullptr) {
      mFiltered->bindInternalIndicesTo(ptr);
    }
  }

  void updatePlaceholders(InitContext& context)
  {
    expressions::updatePlaceholders(filter, context);
  }

  o2::soa::Filtered<T>* operator->()
  {
    return mFiltered.get();
  }

  expressions::Filter filter;
  std::unique_ptr<o2::soa::Filtered<T>> mFiltered = nullptr;
  gandiva::NodePtr tree = nullptr;
  gandiva::FilterPtr gfilter = nullptr;
  bool dataframeChanged = true;

  using iterator = typename o2::soa::Filtered<T>::iterator;
  using const_iterator = typename o2::soa::Filtered<T>::const_iterator;
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
} // namespace o2::framework

namespace o2::soa
{
/// On-the-fly adding of expression columns
template <typename T, typename... Cs>
auto Extend(T const& table)
{
  static_assert((soa::is_type_spawnable_v<Cs> && ...), "You can only extend a table with expression columns");
  using output_t = Join<T, soa::Table<Cs...>>;
  return output_t{{o2::framework::spawner(framework::pack<Cs...>{}, {table.asArrowTable()}, "dynamicExtension"), table.asArrowTable()}, 0};
}

/// Template function to attach dynamic columns on-the-fly (e.g. inside
/// process() function). Dynamic columns need to be compatible with the table.
template <typename T, typename... Cs>
auto Attach(T const& table)
{
  static_assert((framework::is_base_of_template_v<o2::soa::DynamicColumn, Cs> && ...), "You can only attach dynamic columns");
  using output_t = Join<T, o2::soa::Table<Cs...>>;
  return output_t{{table.asArrowTable()}, table.offset()};
}
} // namespace o2::soa

#endif // o2_framework_AnalysisHelpers_H_DEFINED
