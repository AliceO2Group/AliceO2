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

#include "ConfigParamSpec.h"
#include "Framework/ASoA.h"
#include "Framework/DataAllocator.h"
#include "Framework/IndexBuilderHelpers.h"
#include "Framework/InputSpec.h"
#include "Framework/Output.h"
#include "Framework/OutputObjHeader.h"
#include "Framework/OutputRef.h"
#include "Framework/OutputSpec.h"
#include "Framework/Plugins.h"
#include "Framework/StringHelpers.h"
#include "Framework/TableBuilder.h"
#include "Framework/Traits.h"

#include <string>
namespace o2::framework
{
std::string serializeProjectors(std::vector<framework::expressions::Projector>& projectors);
std::string serializeSchema(std::shared_ptr<arrow::Schema>& schema);
}  // namespace o2::framework

namespace o2::soa
{
template <TableRef R>
constexpr auto tableRef2ConfigParamSpec()
{
  return o2::framework::ConfigParamSpec{
    std::string{"input:"} + o2::aod::label<R>(),
    framework::VariantType::String,
    aod::sourceSpec<R>(),
    {"\"\""}};
}

namespace
{
template <soa::with_sources T>
inline constexpr auto getSources()
{
  return []<size_t N, std::array<soa::TableRef, N> refs>() {
    return []<size_t... Is>(std::index_sequence<Is...>) {
      return std::vector{soa::tableRef2ConfigParamSpec<refs[Is]>()...};
    }(std::make_index_sequence<N>());
  }.template operator()<T::sources.size(), T::sources>();
}

template <soa::with_ccdb_urls T>
inline constexpr auto getCCDBUrls()
{
  std::vector<framework::ConfigParamSpec> result;
  for (size_t i = 0; i < T::ccdb_urls.size(); ++i) {
    result.push_back({std::string{"ccdb:"} + std::string{T::ccdb_bindings[i]},
                      framework::VariantType::String,
                      T::ccdb_urls[i],
                      {"\"\""}});
  }
  return result;
}

template <soa::with_sources T>
constexpr auto getInputMetadata() -> std::vector<framework::ConfigParamSpec>
{
  std::vector<framework::ConfigParamSpec> inputMetadata;
  auto inputSources = getSources<T>();
  std::sort(inputSources.begin(), inputSources.end(), [](framework::ConfigParamSpec const& a, framework::ConfigParamSpec const& b) { return a.name < b.name; });
  auto last = std::unique(inputSources.begin(), inputSources.end(), [](framework::ConfigParamSpec const& a, framework::ConfigParamSpec const& b) { return a.name == b.name; });
  inputSources.erase(last, inputSources.end());
  inputMetadata.insert(inputMetadata.end(), inputSources.begin(), inputSources.end());
  return inputMetadata;
}

template <typename T>
  requires(!soa::with_sources<T>)
constexpr auto getInputMetadata() -> std::vector<framework::ConfigParamSpec>
{
  return {};
}

template <soa::with_ccdb_urls T>
constexpr auto getCCDBMetadata() -> std::vector<framework::ConfigParamSpec>
{
  std::vector<framework::ConfigParamSpec> results = getCCDBUrls<T>();
  std::sort(results.begin(), results.end(), [](framework::ConfigParamSpec const& a, framework::ConfigParamSpec const& b) { return a.name < b.name; });
  auto last = std::unique(results.begin(), results.end(), [](framework::ConfigParamSpec const& a, framework::ConfigParamSpec const& b) { return a.name == b.name; });
  results.erase(last, results.end());
  return results;
}

template <typename T>
constexpr auto getCCDBMetadata() -> std::vector<framework::ConfigParamSpec>
{
  return {};
}

template <soa::with_expression_pack T>
constexpr auto getExpressionMetadata() -> std::vector<framework::ConfigParamSpec>
{
  using expression_pack_t = T::expression_pack_t;

  auto projectors = []<typename... C>(framework::pack<C...>) -> std::vector<framework::expressions::Projector> {
    std::vector<framework::expressions::Projector> result;
    (result.emplace_back(std::move(C::Projector())), ...);
    return result;
  }(expression_pack_t{});

  auto schema = std::make_shared<arrow::Schema>(o2::soa::createFieldsFromColumns(expression_pack_t{}));

  auto json = framework::serializeProjectors(projectors);
  return {framework::ConfigParamSpec{"projectors", framework::VariantType::String, json, {"\"\""}},
          framework::ConfigParamSpec{"schema", framework::VariantType::String, framework::serializeSchema(schema), {"\"\""}}};
}

template <typename T>
  requires(!soa::with_expression_pack<T>)
constexpr auto getExpressionMetadata() -> std::vector<framework::ConfigParamSpec>
{
  return {};
}

}  // namespace

template <TableRef R>
constexpr auto tableRef2InputSpec()
{
  std::vector<framework::ConfigParamSpec> metadata;
  auto m = getInputMetadata<typename o2::aod::MetadataTrait<o2::aod::Hash<R.desc_hash>>::metadata>();
  metadata.insert(metadata.end(), m.begin(), m.end());
  auto ccdbMetadata = getCCDBMetadata<typename o2::aod::MetadataTrait<o2::aod::Hash<R.desc_hash>>::metadata>();
  metadata.insert(metadata.end(), ccdbMetadata.begin(), ccdbMetadata.end());
  auto p = getExpressionMetadata<typename o2::aod::MetadataTrait<o2::aod::Hash<R.desc_hash>>::metadata>();
  metadata.insert(metadata.end(), p.begin(), p.end());

  return framework::InputSpec{
    o2::aod::label<R>(),
    o2::aod::origin<R>(),
    o2::aod::description(o2::aod::signature<R>()),
    R.version,
    framework::Lifetime::Timeframe,
    metadata};
}

template <TableRef R>
constexpr auto tableRef2OutputSpec()
{
  return framework::OutputSpec{
    framework::OutputLabel{o2::aod::label<R>()},
    o2::aod::origin<R>(),
    o2::aod::description(o2::aod::signature<R>()),
    R.version};
}

template <TableRef R>
constexpr auto tableRef2Output()
{
  return framework::Output{
    o2::aod::origin<R>(),
    o2::aod::description(o2::aod::signature<R>()),
    R.version};
}

template <TableRef R>
constexpr auto tableRef2OutputRef()
{
  return framework::OutputRef{
    o2::aod::label<R>(),
    R.version};
}
}  // namespace o2::soa

namespace o2::framework
{
class TableConsumer;

/// Helper class actually implementing the cursor which can write to
/// a table. The provided template arguments are if type Column and
/// therefore refer only to the persisted columns.
template <typename T>
concept is_producable = soa::has_metadata<aod::MetadataTrait<T>> || soa::has_metadata<aod::MetadataTrait<typename T::parent_t>>;

template <typename T>
concept is_enumerated_iterator = requires(T t) { t.globalIndex(); };

template <is_producable T>
struct WritingCursor {
 public:
  using persistent_table_t = decltype([]() { if constexpr (soa::is_iterator<T>) { return typename T::parent_t{nullptr}; } else { return T{nullptr}; } }());
  using cursor_t = decltype(std::declval<TableBuilder>().cursor<persistent_table_t>());

  template <typename... Ts>
  void operator()(Ts&&... args)
    requires(sizeof...(Ts) == framework::pack_size(typename persistent_table_t::persistent_columns_t{}))
  {
    ++mCount;
    cursor(0, extract(args)...);
  }

  /// Last index inserted in the table
  int64_t lastIndex()
  {
    return mCount;
  }

  bool resetCursor(LifetimeHolder<TableBuilder> builder)
  {
    mBuilder = std::move(builder);
    cursor = std::move(FFL(mBuilder->cursor<persistent_table_t>()));
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

  void release()
  {
    mBuilder.release();
  }

  decltype(FFL(std::declval<cursor_t>())) cursor;

 private:
  static decltype(auto) extract(is_enumerated_iterator auto const& arg)
  {
    return arg.globalIndex();
  }

  template <typename A>
    requires(!is_enumerated_iterator<A>)
  static decltype(auto) extract(A&& arg)
  {
    return arg;
  }

  /// The table builder which actually performs the
  /// construction of the table. We keep it around to be
  /// able to do all-columns methods like reserve.
  LifetimeHolder<TableBuilder> mBuilder = nullptr;
  int64_t mCount = -1;
};

/// Helper to define output for a Table
template <soa::is_table T>
consteval auto typeWithRef() -> T
{
}

template <soa::is_iterator T>
consteval auto typeWithRef() -> typename T::parent_t
{
}

template <typename T>
  requires soa::is_table<T> || soa::is_iterator<T>
struct OutputForTable {
  using table_t = decltype(typeWithRef<T>());
  using metadata = aod::MetadataTrait<o2::aod::Hash<table_t::ref.desc_hash>>::metadata;

  static OutputSpec const spec()
  {
    return OutputSpec{OutputLabel{aod::label<table_t::ref>()}, o2::aod::origin<table_t::ref>(), o2::aod::description(o2::aod::signature<table_t::ref>()), table_t::ref.version};
  }

  static OutputRef ref()
  {
    return OutputRef{aod::label<table_t::ref>(), table_t::ref.version};
  }
};

/// This helper class allows you to declare things which will be created by a
/// given analysis task. Notice how the actual cursor is implemented by the
/// means of the WritingCursor helper class, from which produces actually
/// derives.
template <is_producable T>
struct Produces : WritingCursor<T> {
};

template <typename T>
concept is_produces = requires(T t) { typename T::cursor_t; typename T::persistent_table_t; &T::cursor; };

/// Use this to group together produces. Useful to separate them logically
/// or simply to stay within the 100 elements per Task limit.
/// Use as:
///
/// struct MySetOfProduces : ProducesGroup {
/// } products;
///
/// Notice the label MySetOfProduces is just a mnemonic and can be omitted.
struct ProducesGroup {
};

template <typename T>
concept is_produces_group = std::derived_from<T, ProducesGroup>;

/// Helper template for table transformations
template <soa::is_metadata M, soa::TableRef Ref>
struct TableTransform {
  using metadata = M;
  constexpr static auto sources = M::sources;

  template <soa::TableRef R>
  static constexpr auto base_spec()
  {
    return soa::tableRef2InputSpec<R>();
  }

  static auto base_specs()
  {
    return []<size_t... Is>(std::index_sequence<Is...>) -> std::vector<InputSpec> {
      return {base_spec<sources[Is]>()...};
    }(std::make_index_sequence<sources.size()>{});
  }

  constexpr auto spec() const
  {
    return soa::tableRef2OutputSpec<Ref>();
  }

  constexpr auto output() const
  {
    return soa::tableRef2Output<Ref>();
  }

  constexpr auto ref() const
  {
    return soa::tableRef2OutputRef<Ref>();
  }
};

/// This helper struct allows you to declare extended tables which should be
/// created by the task (as opposed to those pre-defined by data model)
template <typename T>
concept is_spawnable = soa::has_metadata<aod::MetadataTrait<o2::aod::Hash<T::ref.desc_hash>>> && soa::has_extension<typename aod::MetadataTrait<o2::aod::Hash<T::ref.desc_hash>>::metadata>;

template <typename T>
concept is_dynamically_spawnable = soa::has_metadata<aod::MetadataTrait<o2::aod::Hash<T::ref.desc_hash>>> && soa::has_configurable_extension<typename aod::MetadataTrait<o2::aod::Hash<T::ref.desc_hash>>::metadata>;

template <is_spawnable T>
constexpr auto transformBase()
{
  using metadata = typename aod::MetadataTrait<o2::aod::Hash<T::ref.desc_hash>>::metadata;
  return TableTransform<metadata, metadata::extension_table_t::ref>{};
}

template <is_spawnable T>
struct Spawns : decltype(transformBase<T>()) {
  using spawnable_t = T;
  using metadata = decltype(transformBase<T>())::metadata;
  using extension_t = typename metadata::extension_table_t;
  using base_table_t = typename metadata::base_table_t;
  using expression_pack_t = typename metadata::expression_pack_t;
  static constexpr size_t N = framework::pack_size(expression_pack_t{});

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
  std::array<o2::framework::expressions::Projector, N> projectors = []<typename... C>(framework::pack<C...>) -> std::array<expressions::Projector, sizeof...(C)>
  {
    return {{std::move(C::Projector())...}};
  }
  (expression_pack_t{});
  std::shared_ptr<gandiva::Projector> projector = nullptr;
  std::shared_ptr<arrow::Schema> schema = std::make_shared<arrow::Schema>(o2::soa::createFieldsFromColumns(expression_pack_t{}));
};

template <typename T>
concept is_spawns = requires(T t) {
  typename T::metadata;
  requires std::same_as<decltype(t.pack()), typename T::expression_pack_t>;
  requires std::same_as<decltype(t.projector), std::shared_ptr<gandiva::Projector>>;
};

/// This helper struct allows you to declare extended tables with dynamically-supplied
/// expressions to be created by the task
/// The actual expressions have to be set in init() for the configurable expression
/// columns, used to define the table

template <is_dynamically_spawnable T, bool DELAYED = false>
struct Defines : decltype(transformBase<T>()) {
  static constexpr bool delayed = DELAYED;
  using spawnable_t = T;
  using metadata = decltype(transformBase<T>())::metadata;
  using extension_t = typename metadata::extension_table_t;
  using base_table_t = typename metadata::base_table_t;
  using placeholders_pack_t = typename metadata::placeholders_pack_t;
  static constexpr size_t N = framework::pack_size(placeholders_pack_t{});

  constexpr auto pack()
  {
    return placeholders_pack_t{};
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

  std::array<o2::framework::expressions::Projector, N> projectors;
  std::shared_ptr<gandiva::Projector> projector = nullptr;
  std::shared_ptr<arrow::Schema> schema = std::make_shared<arrow::Schema>(o2::soa::createFieldsFromColumns(placeholders_pack_t{}));
  std::shared_ptr<arrow::Schema> inputSchema = nullptr;

  bool needRecompilation = false;

  void recompile()
  {
    projector = framework::expressions::createProjectorHelper(N, projectors.data(), inputSchema, schema->fields());
  }
};

template <is_dynamically_spawnable T>
using DefinesDelayed = Defines<T, true>;

template <typename T>
concept is_defines = requires(T t) {
  typename T::metadata;
  requires std::same_as<decltype(t.pack()), typename T::placeholders_pack_t>;
  requires std::same_as<decltype(t.projector), std::shared_ptr<gandiva::Projector>>;
  requires std::same_as<decltype(t.needRecompilation), bool>;
  &T::recompile;
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
  using IC = framework::pack_element_t<framework::has_type_at_conditional_v<soa::is_binding_compatible, Key>(typename T::external_index_columns_t{}), typename T::external_index_columns_t>;
  return table->column(framework::has_type_at_v<IC>(typename T::persistent_columns_t{}));
}

template <soa::is_column C>
struct ColumnTrait {
  using column_t = C;

  static consteval auto listSize()
  {
    if constexpr (std::same_as<typename C::type, std::vector<int>>) {
      return -1;
    } else if constexpr (std::same_as<int[2], typename C::type>) {
      return 2;
    } else {
      return 1;
    }
  }

  template <typename T, typename Key>
  static std::shared_ptr<SelfIndexColumnBuilder> makeColumnBuilder(arrow::Table* table, arrow::MemoryPool* pool)
  {
    if constexpr (!std::same_as<T, Key>) {
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

template <typename Key, typename C>
using reduced_t = Reduction<Key, C>::type;
}  // namespace

template <typename Kind>
struct IndexBuilder {
  template <typename Key, size_t N, std::array<soa::TableRef, N> refs, typename C1, typename... Cs>
  static auto indexBuilder(const char* label, std::vector<std::shared_ptr<arrow::Table>>&& tables, framework::pack<C1, Cs...>)
  {
    auto pool = arrow::default_memory_pool();
    SelfIndexColumnBuilder self{C1::columnLabel(), pool};
    std::unique_ptr<ChunkedArrayIterator> keyIndex = nullptr;
    if constexpr (!Key::template hasOriginal<refs[0]>()) {
      keyIndex = std::make_unique<ChunkedArrayIterator>(tables[0]->column(o2::aod::MetadataTrait<o2::aod::Hash<refs[0].desc_hash>>::metadata::template getIndexPosToKey<Key>()));
    }

    auto sq = std::make_index_sequence<sizeof...(Cs)>();

    auto columnBuilders = [&tables, &pool ]<size_t... Is>(std::index_sequence<Is...>) -> std::array<std::shared_ptr<framework::SelfIndexColumnBuilder>, sizeof...(Cs)>
    {
      return {[](arrow::Table* table, arrow::MemoryPool* pool) {
        using T = framework::pack_element_t<Is, framework::pack<Cs...>>;
        if constexpr (!Key::template hasOriginal<refs[Is + 1]>()) {
          constexpr auto pos = o2::aod::MetadataTrait<o2::aod::Hash<refs[Is + 1].desc_hash>>::metadata::template getIndexPosToKey<Key>();
          return std::make_shared<IndexColumnBuilder>(table->column(pos), T::columnLabel(), ColumnTrait<T>::listSize(), pool);
        } else {
          return std::make_shared<SelfIndexColumnBuilder>(T::columnLabel(), pool);
        }
      }(tables[Is + 1].get(), pool)...};
    }
    (sq);

    std::array<bool, sizeof...(Cs)> finds;

    for (int64_t counter = 0; counter < tables[0]->num_rows(); ++counter) {
      int64_t idx = -1;
      if constexpr (Key::template hasOriginal<refs[0]>()) {
        idx = counter;
      } else {
        idx = keyIndex->valueAt(counter);
      }
      finds = [&idx, &columnBuilders]<size_t... Is>(std::index_sequence<Is...>) {
        return std::array{
          [&idx, &columnBuilders]() {
            using T = typename framework::pack_element_t<Is, framework::pack<Cs...>>;
            return std::static_pointer_cast<reduced_t<Key, T>>(columnBuilders[Is])->template find<T>(idx);
          }()...};
      }(sq);
      if constexpr (std::same_as<Kind, Sparse>) {
        [&idx, &columnBuilders]<size_t... Is>(std::index_sequence<Is...>) {
          ([&idx, &columnBuilders]() {
            using T = typename framework::pack_element_t<Is, framework::pack<Cs...>>;
            return std::static_pointer_cast<reduced_t<Key, T>>(columnBuilders[Is])->template fill<T>(idx); }(), ...);
        }(sq);
        self.fill<C1>(counter);
      } else if constexpr (std::same_as<Kind, Exclusive>) {
        if (std::none_of(finds.begin(), finds.end(), [](bool const x) { return x == false; })) {
          [&idx, &columnBuilders]<size_t... Is>(std::index_sequence<Is...>) {
            ([&idx, &columnBuilders]() {
              using T = typename framework::pack_element_t<Is, framework::pack<Cs...>>;
              return std::static_pointer_cast<reduced_t<Key, T>>(columnBuilders[Is])->template fill<T>(idx);
            }(),
             ...);
          }(sq);
          self.fill<C1>(counter);
        }
      }
    }

    return [&label, &columnBuilders, &self]<size_t... Is>(std::index_sequence<Is...>) {
      return makeArrowTable(label,
                            {self.template result<C1>(), [&columnBuilders]() {
                               using T = typename framework::pack_element_t<Is, framework::pack<Cs...>>;
                               return std::static_pointer_cast<reduced_t<Key, T>>(columnBuilders[Is])->template result<T>();
                             }()...},
                            {self.field(), [&columnBuilders]() {
                               using T = typename framework::pack_element_t<Is, framework::pack<Cs...>>;
                               return std::static_pointer_cast<reduced_t<Key, T>>(columnBuilders[Is])->field();
                             }()...});
    }(sq);
  }
};

/// This helper struct allows you to declare index tables to be created in a task

template <soa::is_index_table T>
constexpr auto transformBase()
{
  using metadata = typename aod::MetadataTrait<o2::aod::Hash<T::ref.desc_hash>>::metadata;
  return TableTransform<metadata, T::ref>{};
}

template <soa::is_index_table T>
struct Builds : decltype(transformBase<T>()) {
  using buildable_t = T;
  using metadata = decltype(transformBase<T>())::metadata;
  using IP = std::conditional_t<metadata::exclusive, IndexBuilder<Exclusive>, IndexBuilder<Sparse>>;
  using Key = metadata::Key;
  using H = typename T::first_t;
  using Ts = typename T::rest_t;
  using index_pack_t = metadata::index_pack_t;

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

  template <typename Key, typename... Cs>
  auto build(framework::pack<Cs...>, std::vector<std::shared_ptr<arrow::Table>>&& tables)
  {
    this->table = std::make_shared<T>(IP::template indexBuilder<Key, metadata::sources.size(), metadata::sources>(o2::aod::label<T::ref>(), std::forward<std::vector<std::shared_ptr<arrow::Table>>>(tables), framework::pack<Cs...>{}));
    return (this->table != nullptr);
  }
};

template <typename T>
concept is_builds = requires(T t) {
  typename T::metadata;
  typename T::Key;
  requires std::same_as<decltype(t.pack()), typename T::index_pack_t>;
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
    auto lhash = runtime_hash(label.c_str());
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

  OutputRef ref(uint16_t index, uint16_t max)
  {
    return OutputRef{std::string{label}, 0,
                     o2::header::Stack{OutputObjHeader{policy, sourceType, mTaskHash, index, max}}};
  }

  std::shared_ptr<T> object;
  std::string label;
  OutputObjHandlingPolicy policy;
  OutputObjSourceType sourceType;
  uint32_t mTaskHash;
};

template <typename T>
concept is_outputobj = requires(T t) {
  &T::setHash;
  &T::spec;
  &T::ref;
  requires std::same_as<decltype(t.operator->()), typename T::obj_t*>;
  requires std::same_as<decltype(t.object), std::shared_ptr<typename T::obj_t>>;
};

/// This helper allows you to fetch a Sevice from the context or
/// by using some singleton. This hopefully will hide the Singleton and
/// We will be able to retrieve it in a more thread safe manner later on.
template <typename T>
struct Service {
  using service_t = T;
  T* service;

  decltype(auto) operator->() const
  {
    if constexpr (base_of_template<LoadableServicePlugin, T>) {
      return service->get();
    } else {
      return service;
    }
  }
};

template <typename T>
concept is_service = requires(T t) {
  requires std::same_as<decltype(t.service), typename T::service_t*>;
  &T::operator->;
};

auto getTableFromFilter(soa::is_filtered_table auto const& table, soa::SelectionVector&& selection)
{
  return std::make_unique<o2::soa::Filtered<std::decay_t<decltype(table)>>>(std::vector{table}, std::forward<soa::SelectionVector>(selection));
}

auto getTableFromFilter(soa::is_not_filtered_table auto const& table, soa::SelectionVector&& selection)
{
  return std::make_unique<o2::soa::Filtered<std::decay_t<decltype(table)>>>(std::vector{table.asArrowTable()}, std::forward<soa::SelectionVector>(selection));
}

void initializePartitionCaches(std::set<uint32_t> const& hashes, std::shared_ptr<arrow::Schema> const& schema, expressions::Filter const& filter, gandiva::NodePtr& tree, gandiva::FilterPtr& gfilter);

template <typename T>
struct Partition {
  using content_t = T;
  Partition(expressions::Node&& filter_) : filter{std::forward<expressions::Node>(filter_)}
  {
  }

  Partition(expressions::Node&& filter_, T const& table)
    : filter{std::forward<expressions::Node>(filter_)}
  {
    setTable(table);
  }

  void intializeCaches(std::set<uint32_t> const& hashes, std::shared_ptr<arrow::Schema> const& schema)
  {
    initializePartitionCaches(hashes, schema, filter, tree, gfilter);
  }

  void bindTable(T const& table)
  {
    intializeCaches(T::table_t::hashes(), table.asArrowTable()->schema());
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

  [[nodiscard]] std::shared_ptr<arrow::Table> asArrowTable() const
  {
    return mFiltered->asArrowTable();
  }

  o2::soa::Filtered<T>* operator->()
  {
    return mFiltered.get();
  }

  template <typename T1>
  [[nodiscard]] auto rawSliceBy(o2::framework::Preslice<T1> const& container, int value) const
  {
    return mFiltered->rawSliceBy(container, value);
  }

  [[nodiscard]] auto sliceByCached(framework::expressions::BindingNode const& node, int value, o2::framework::SliceCache& cache) const
  {
    return mFiltered->sliceByCached(node, value, cache);
  }

  [[nodiscard]] auto sliceByCachedUnsorted(framework::expressions::BindingNode const& node, int value, o2::framework::SliceCache& cache) const
  {
    return mFiltered->sliceByCachedUnsorted(node, value, cache);
  }

  template <typename T1, typename Policy, bool OPT>
  [[nodiscard]] auto sliceBy(o2::framework::PresliceBase<T1, Policy, OPT> const& container, int value) const
  {
    return mFiltered->sliceBy(container, value);
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

template <typename T>
concept is_partition = requires(T t) {
  &T::updatePlaceholders;
  requires std::same_as<decltype(t.filter), expressions::Filter>;
  requires std::same_as<decltype(t.mFiltered), std::unique_ptr<o2::soa::Filtered<typename T::content_t>>>;
};
}  // namespace o2::framework

namespace o2::soa
{
/// On-the-fly adding of expression columns
template <soa::is_table T, soa::is_spawnable_column... Cs>
auto Extend(T const& table)
{
  using output_t = Join<T, soa::Table<o2::aod::Hash<"JOIN"_h>, o2::aod::Hash<"JOIN/0"_h>, o2::aod::Hash<"JOIN"_h>, Cs...>>;
  static std::array<framework::expressions::Projector, sizeof...(Cs)> projectors{{std::move(Cs::Projector())...}};
  static std::shared_ptr<gandiva::Projector> projector = nullptr;
  static auto schema = std::make_shared<arrow::Schema>(o2::soa::createFieldsFromColumns(framework::pack<Cs...>{}));
  return output_t{{o2::framework::spawner(framework::pack<Cs...>{}, {table.asArrowTable()}, "dynamicExtension", projectors.data(), projector, schema), table.asArrowTable()}, 0};
}

/// Template function to attach dynamic columns on-the-fly (e.g. inside
/// process() function). Dynamic columns need to be compatible with the table.
template <soa::is_table T, soa::is_dynamic_column... Cs>
auto Attach(T const& table)
{
  using output_t = Join<T, o2::soa::Table<o2::aod::Hash<"JOIN"_h>, o2::aod::Hash<"JOIN/0"_h>, o2::aod::Hash<"JOIN"_h>, Cs...>>;
  return output_t{{table.asArrowTable()}, table.offset()};
}
}  // namespace o2::soa

#endif  // o2_framework_AnalysisHelpers_H_DEFINED
