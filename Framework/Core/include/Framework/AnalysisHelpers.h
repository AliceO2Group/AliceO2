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
#include "Framework/Logger.h"
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
/// Structure to contain mapping between matchers and process functions.
/// Process function is identified by hash, each matcher has associated
/// argument position for that process function; single argument can have
/// many matchers associated due to complicated joins
struct InputInfo {
  uint32_t hash;
  std::vector<std::pair<int, ConcreteDataMatcher>> matchers;
};
} // namespace o2::framework

namespace o2::soa
{
struct IndexRecord {
  std::string label;
  framework::ConcreteDataMatcher matcher;
  std::string columnLabel;
  IndexKind kind;
  int pos;
  std::shared_ptr<arrow::DataType> type = [](IndexKind kind) -> std::shared_ptr<arrow::DataType> {
    switch (kind) {
      case IndexKind::IdxSingle:
      case IndexKind::IdxSelf:
        return arrow::int32();
      case IndexKind::IdxSlice:
        return arrow::fixed_size_list(arrow::int32(), 2);
      case IndexKind::IdxArray:
        return arrow::list(arrow::int32());
      default:
        return {nullptr};
    }
  }(kind);

  auto operator==(IndexRecord const& other) const
  {
    return (this->label == other.label) && (this->columnLabel == other.columnLabel) && (this->kind == other.kind) && (this->pos == other.pos);
  }

  std::shared_ptr<arrow::Field> field() const
  {
    return std::make_shared<arrow::Field>(columnLabel, type);
  }
};

struct IndexBuilder {
  static std::vector<framework::IndexColumnBuilder> makeBuilders(std::vector<std::shared_ptr<arrow::Table>>&& tables, std::vector<soa::IndexRecord> const& records);
  static void resetBuilders(std::vector<framework::IndexColumnBuilder>& builders, std::vector<std::shared_ptr<arrow::Table>>&& tables);

  static std::shared_ptr<arrow::Table> materialize(std::vector<framework::IndexColumnBuilder>& builders, std::vector<std::shared_ptr<arrow::Table>>&& tables, std::vector<soa::IndexRecord> const& records, std::shared_ptr<arrow::Schema> const& schema, bool exclusive);
};
} // namespace o2::soa

namespace o2::framework
{
void wrongOriginReplacement(std::string_view replacement);
std::shared_ptr<arrow::Table> makeEmptyTableImpl(const char* name, std::shared_ptr<arrow::Schema>& schema);

template <soa::is_table T>
auto makeEmptyTable(const char* name)
{
  auto schema = std::make_shared<arrow::Schema>(soa::createFieldsFromColumns(typename T::table_t::persistent_columns_t{}));
  return makeEmptyTableImpl(name, schema);
}

template <soa::TableRef R>
  requires(soa::not_void<typename aod::MetadataTrait<aod::Hash<R.desc_hash>>::metadata>)
auto makeEmptyTable()
{
  auto schema = std::make_shared<arrow::Schema>(soa::createFieldsFromColumns(typename aod::MetadataTrait<aod::Hash<R.desc_hash>>::metadata::persistent_columns_t{}));
  return makeEmptyTableImpl(o2::aod::label<R>(), schema);
}

template <typename... Cs>
auto makeEmptyTable(const char* name, framework::pack<Cs...> p)
{
  auto schema = std::make_shared<arrow::Schema>(soa::createFieldsFromColumns(p));
  return makeEmptyTableImpl(name, schema);
}

template <aod::is_aod_hash D>
  requires(soa::not_void<typename aod::MetadataTrait<D>::metadata>)
auto makeEmptyTable(const char* name)
{
  auto schema = std::make_shared<arrow::Schema>(soa::createFieldsFromColumns(typename aod::MetadataTrait<D>::metadata::persistent_columns_t{}));
  return makeEmptyTableImpl(name, schema);
}

std::shared_ptr<arrow::Table> spawnerHelper(std::shared_ptr<arrow::Table> const& fullTable, std::shared_ptr<arrow::Schema> newSchema, size_t nColumns,
                                            expressions::Projector* projectors, const char* name, std::shared_ptr<gandiva::Projector>& projector);

std::shared_ptr<arrow::Table> spawnerHelper(std::shared_ptr<arrow::Table> const& fullTable, std::shared_ptr<arrow::Schema> newSchema,
                                            const char* name, size_t nColumns,
                                            const std::shared_ptr<gandiva::Projector>& projector);

/// Expression-based column generator to materialize columns
template <aod::is_aod_hash D>
  requires(soa::has_extension<typename o2::aod::MetadataTrait<D>::metadata>)
auto spawner(std::shared_ptr<arrow::Table> const& fullTable, const char* name, o2::framework::expressions::Projector* projectors, std::shared_ptr<gandiva::Projector>& projector, std::shared_ptr<arrow::Schema> const& schema)
{
  if (fullTable->num_rows() == 0) {
    return makeEmptyTable<D>(name);
  }
  constexpr auto Ncol = []<typename M>() {
    if constexpr (soa::has_configurable_extension<M>) {
      return framework::pack_size(typename M::placeholders_pack_t{});
    } else {
      return framework::pack_size(typename M::expression_pack_t{});
    }
  }.template operator()<typename o2::aod::MetadataTrait<D>::metadata>();
  return spawnerHelper(fullTable, schema, Ncol, projectors, name, projector);
}

template <typename... C>
auto spawner(framework::pack<C...>, std::vector<std::shared_ptr<arrow::Table>>&& tables, const char* name, expressions::Projector* projectors, std::shared_ptr<gandiva::Projector>& projector, std::shared_ptr<arrow::Schema> const& schema)
{
  std::array<const char*, 1> labels{"original"};
  auto fullTable = soa::ArrowHelpers::joinTables(std::move(tables), std::span<const char* const>{labels});
  if (fullTable->num_rows() == 0) {
    return makeEmptyTable(name, framework::pack<C...>{});
  }
  return spawnerHelper(fullTable, schema, sizeof...(C), projectors, name, projector);
}

std::string serializeProjectors(std::vector<framework::expressions::Projector>& projectors);
std::string serializeSchema(std::shared_ptr<arrow::Schema> schema);
std::string serializeIndexRecords(std::vector<o2::soa::IndexRecord>& irs);
std::vector<std::shared_ptr<arrow::Table>> extractSources(ProcessingContext& pc, std::vector<std::string> const& labels);

struct Spawner {
  std::string binding;
  std::vector<std::string> labels;
  std::vector<framework::ConcreteDataMatcher> matchers;
  std::vector<std::shared_ptr<gandiva::Expression>> expressions;
  std::shared_ptr<gandiva::Projector> projector = nullptr;
  std::shared_ptr<arrow::Schema> schema = nullptr;
  std::shared_ptr<arrow::Schema> inputSchema = nullptr;

  header::DataOrigin origin;
  header::DataDescription description;
  header::DataHeader::SubSpecificationType version;

  std::shared_ptr<arrow::Table> materialize(ProcessingContext& pc) const;
};

struct Builder {
  bool exclusive;
  std::vector<std::string> labels;
  std::vector<framework::ConcreteDataMatcher> matchers;
  std::vector<o2::soa::IndexRecord> records;
  std::shared_ptr<arrow::Schema> outputSchema;
  header::DataOrigin origin;
  header::DataDescription description;
  header::DataHeader::SubSpecificationType version;

  std::shared_ptr<std::vector<framework::IndexColumnBuilder>> builders = nullptr;

  std::shared_ptr<arrow::Table> materialize(ProcessingContext& pc);
};

ConfigParamSpec replaceOrigin(ConfigParamSpec& source, std::string const& originStr);
ConcreteDataMatcher replaceOrigin(ConcreteDataMatcher& matcher, const header::DataOrigin& newOrigin);
} // namespace o2::framework

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

template <TableRef R>
constexpr auto tableRef2Schema()
{
  return o2::framework::ConfigParamSpec{
    std::string{"input-schema:"} + o2::aod::label<R>(),
    framework::VariantType::String,
    framework::serializeSchema(o2::aod::MetadataTrait<o2::aod::Hash<R.desc_hash>>::metadata::getSchema()),
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

template <soa::with_sources T>
inline constexpr auto getSourceSchemas()
{
  return []<size_t N, std::array<soa::TableRef, N> refs>() {
    return []<size_t... Is>(std::index_sequence<Is...>) {
      return std::vector{soa::tableRef2Schema<refs[Is]>()...};
    }(std::make_index_sequence<N>());
  }.template operator()<T::sources.size(), T::sources>();
}

template <soa::with_sources_generator T, aod::is_origin_hash O = o2::aod::Hash<"AOD"_h>>
inline constexpr auto getSources()
{
  return []<size_t N, std::array<soa::TableRef, N> refs>() {
    return []<size_t... Is>(std::index_sequence<Is...>) {
      return std::vector{soa::tableRef2ConfigParamSpec<refs[Is]>()...};
    }(std::make_index_sequence<N>());
  }.template operator()<T::N, T::template generateSources<O>()>();
}

template <soa::with_sources_generator T, aod::is_origin_hash O = o2::aod::Hash<"AOD"_h>>
inline constexpr auto getSourceSchemas()
{
  return []<size_t N, std::array<soa::TableRef, N> refs>() {
    return []<size_t... Is>(std::index_sequence<Is...>) {
      return std::vector{soa::tableRef2Schema<refs[Is]>()...};
    }(std::make_index_sequence<N>());
  }.template operator()<T::N, T::template generateSources<O>()>();
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

template <typename T>
  requires(std::same_as<T, int>)
consteval IndexKind getIndexKind()
{
  return IndexKind::IdxSingle;
}

template <typename T>
  requires(std::is_bounded_array_v<T>)
consteval IndexKind getIndexKind()
{
  return IndexKind::IdxSlice;
}

template <typename T>
  requires(framework::is_specialization_v<T, std::vector>)
consteval IndexKind getIndexKind()
{
  return IndexKind::IdxArray;
}

template <soa::with_index_pack T>
inline constexpr auto getIndexMapping(header::DataOrigin newOrigin = header::DataOrigin{"AOD"})
{
  std::vector<IndexRecord> idx;
  using indices = T::index_pack_t;
  using Key = T::Key;
  [&idx, &newOrigin]<size_t... Is>(std::index_sequence<Is...>) mutable {
    constexpr auto refs = T::generateSources();
    ([&idx, &newOrigin]<TableRef ref, typename C>() mutable {
      constexpr auto pos = o2::aod::MetadataTrait<o2::aod::Hash<ref.desc_hash>>::metadata::template getIndexPosToKey<Key>();
      auto matcher = o2::aod::matcher<ref>();
      if ((ref.origin_hash == "AOD"_h) && (newOrigin != header::DataOrigin{"AOD"})) {
        matcher = replaceOrigin(matcher, newOrigin);
      }
      if constexpr (pos == -1) {
        idx.emplace_back(o2::aod::label<ref>(), matcher, C::columnLabel(), IndexKind::IdxSelf, pos);
      } else {
        idx.emplace_back(o2::aod::label<ref>(), matcher, C::columnLabel(), getIndexKind<typename C::type>(), pos);
      }
    }.template operator()<refs[Is], typename framework::pack_element_t<Is, indices>>(),
     ...);
  }(std::make_index_sequence<framework::pack_size(indices{})>());
  ;
  return idx;
}

template <soa::with_sources_generator T, aod::is_origin_hash O = o2::aod::Hash<"AOD"_h>>
constexpr auto getInputMetadata() -> std::vector<framework::ConfigParamSpec>
{
  std::vector<framework::ConfigParamSpec> inputMetadata;

  auto inputSources = getSources<T, O>();
  std::sort(inputSources.begin(), inputSources.end(), [](framework::ConfigParamSpec const& a, framework::ConfigParamSpec const& b) { return a.name < b.name; });
  auto last = std::unique(inputSources.begin(), inputSources.end(), [](framework::ConfigParamSpec const& a, framework::ConfigParamSpec const& b) { return a.name == b.name; });
  inputSources.erase(last, inputSources.end());
  inputMetadata.insert(inputMetadata.end(), inputSources.begin(), inputSources.end());

  auto inputSchemas = getSourceSchemas<T, O>();
  std::sort(inputSchemas.begin(), inputSchemas.end(), [](framework::ConfigParamSpec const& a, framework::ConfigParamSpec const& b) { return a.name < b.name; });
  last = std::unique(inputSchemas.begin(), inputSchemas.end(), [](framework::ConfigParamSpec const& a, framework::ConfigParamSpec const& b) { return a.name == b.name; });
  inputSchemas.erase(last, inputSchemas.end());
  inputMetadata.insert(inputMetadata.end(), inputSchemas.begin(), inputSchemas.end());

  return inputMetadata;
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

  auto inputSchemas = getSourceSchemas<T>();
  std::sort(inputSchemas.begin(), inputSchemas.end(), [](framework::ConfigParamSpec const& a, framework::ConfigParamSpec const& b) { return a.name < b.name; });
  last = std::unique(inputSchemas.begin(), inputSchemas.end(), [](framework::ConfigParamSpec const& a, framework::ConfigParamSpec const& b) { return a.name == b.name; });
  inputSchemas.erase(last, inputSchemas.end());
  inputMetadata.insert(inputMetadata.end(), inputSchemas.begin(), inputSchemas.end());

  return inputMetadata;
}

template <typename T>
  requires(!(soa::with_sources<T> || soa::with_sources_generator<T>))
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

  auto json = framework::serializeProjectors(projectors);
  return {framework::ConfigParamSpec{"projectors", framework::VariantType::String, json, {"\"\""}}};
}

template <typename T>
  requires(!soa::with_expression_pack<T>)
constexpr auto getExpressionMetadata() -> std::vector<framework::ConfigParamSpec>
{
  return {};
}

template <soa::with_index_pack T>
constexpr auto getIndexMetadata(header::DataOrigin newOrigin = header::DataOrigin{"AOD"}) -> std::vector<framework::ConfigParamSpec>
{
  auto map = getIndexMapping<T>(newOrigin);
  return {framework::ConfigParamSpec{"index-records", framework::VariantType::String, framework::serializeIndexRecords(map), {"\"\""}},
          {framework::ConfigParamSpec{"index-exclusive", framework::VariantType::Bool, T::exclusive, {"\"\""}}}};
}

template <typename T>
  requires(!soa::with_index_pack<T>)
constexpr auto getIndexMetadata(header::DataOrigin) -> std::vector<framework::ConfigParamSpec>
{
  return {};
}

} // namespace

template <TableRef R>
constexpr auto tableRef2InputSpec(header::DataOrigin newOrigin = header::DataOrigin{"AOD"})
{
  std::vector<framework::ConfigParamSpec> metadata;
  std::vector<framework::ConfigParamSpec> sources;
  if constexpr (soa::with_sources<typename o2::aod::MetadataTrait<o2::aod::Hash<R.desc_hash>>::metadata>) {
    sources = getInputMetadata<typename o2::aod::MetadataTrait<o2::aod::Hash<R.desc_hash>>::metadata>();
  } else if constexpr (soa::with_sources_generator<typename o2::aod::MetadataTrait<o2::aod::Hash<R.desc_hash>>::metadata>) {
    sources = getInputMetadata<typename o2::aod::MetadataTrait<o2::aod::Hash<R.desc_hash>>::metadata, o2::aod::Hash<R.origin_hash>>();
  }
  if ((R.origin_hash == "AOD"_h) && (newOrigin != header::DataOrigin{"AOD"})) {
    std::ranges::transform(sources, sources.begin(), [originStr = newOrigin.as<std::string>()](framework::ConfigParamSpec& source) {
      return replaceOrigin(source, originStr);
    });
    metadata.emplace_back(framework::ConfigParamSpec{"aod-origin-replaced", framework::VariantType::Bool, true, {"\"\""}});
  }
  metadata.insert(metadata.end(), sources.begin(), sources.end());
  auto ccdbURLs = getCCDBMetadata<typename o2::aod::MetadataTrait<o2::aod::Hash<R.desc_hash>>::metadata>();
  metadata.insert(metadata.end(), ccdbURLs.begin(), ccdbURLs.end());
  auto expressions = getExpressionMetadata<typename o2::aod::MetadataTrait<o2::aod::Hash<R.desc_hash>>::metadata>();
  metadata.insert(metadata.end(), expressions.begin(), expressions.end());
  auto indices = getIndexMetadata<typename o2::aod::MetadataTrait<o2::aod::Hash<R.desc_hash>>::metadata>(newOrigin);
  metadata.insert(metadata.end(), indices.begin(), indices.end());
  if constexpr (!soa::with_ccdb_urls<typename o2::aod::MetadataTrait<o2::aod::Hash<R.desc_hash>>::metadata>) {
    metadata.emplace_back(framework::ConfigParamSpec{"schema", framework::VariantType::String, framework::serializeSchema(o2::aod::MetadataTrait<o2::aod::Hash<R.desc_hash>>::metadata::getSchema()), {"\"\""}});
  }

  return framework::InputSpec{
    o2::aod::label<R>(),
    ((R.origin_hash == "AOD"_h) && (newOrigin != header::DataOrigin{"AOD"})) ? newOrigin : o2::aod::origin<R>(),
    o2::aod::description(o2::aod::signature<R>()),
    R.version,
    framework::Lifetime::Timeframe,
    metadata};
}

template <TableRef R>
constexpr auto tableRef2OutputSpec(header::DataOrigin newOrigin = header::DataOrigin{"AOD"})
{
  std::vector<framework::ConfigParamSpec> metadata;
  using md = typename o2::aod::MetadataTrait<o2::aod::Hash<R.desc_hash>>::metadata;
  if constexpr (soa::with_ccdb_urls<md>) {
    metadata.emplace_back("ccdb:", framework::VariantType::Bool, true, framework::ConfigParamSpec::HelpString{"\"\""});
  } else if constexpr (soa::with_expression_pack<md>) {
    metadata.emplace_back("projectors", framework::VariantType::Bool, true, framework::ConfigParamSpec::HelpString{"\"\""});
  } else if constexpr (soa::with_index_pack<md>) {
    metadata.emplace_back("index-records", framework::VariantType::Bool, true, framework::ConfigParamSpec::HelpString{"\"\""});
  }
  if ((R.origin_hash == "AOD"_h) && (newOrigin != header::DataOrigin{"AOD"})) {
    metadata.push_back(framework::ConfigParamSpec{"aod-origin-replaced", framework::VariantType::Bool, true, {"\"\""}});
  }
  return framework::OutputSpec{
    framework::OutputLabel{o2::aod::label<R>()},
    ((R.origin_hash == "AOD"_h) && (newOrigin != header::DataOrigin{"AOD"})) ? newOrigin : o2::aod::origin<R>(),
    o2::aod::description(o2::aod::signature<R>()),
    R.version,
    framework::Lifetime::Timeframe,
    metadata};
}

template <TableRef R>
constexpr auto tableRef2OutputRef()
{
  return framework::OutputRef{
    o2::aod::label<R>(),
    R.version};
}
} // namespace o2::soa

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
  OutputSpec outputSpec{soa::tableRef2OutputSpec<persistent_table_t::ref>()};
  static OutputSpec updateOutputSpec(header::DataOrigin const& newOrigin = header::DataOrigin{"AOD"})
  {
    return soa::tableRef2OutputSpec<persistent_table_t::ref>(newOrigin);
  }

  template <typename... Ts>
  void operator()(Ts&&... args)
    requires(sizeof...(Ts) == framework::pack_size(typename persistent_table_t::persistent_columns_t{}))
  {
    ++mCount;
    if (mReserved >= 0 && mCount >= mReserved) [[unlikely]] {
      // reserve() switched this cursor to UnsafeAppend, which does not grow its
      // buffers. Writing row mCount (>= the reserved count) would overrun them and
      // silently corrupt the heap, so fail here, naming the offending table and
      // row, rather than crashing later somewhere unrelated.
      LOG(fatal) << "Table '" << outputSpec.binding.value << "': writing row " << mCount
                 << " exceeds reserve(" << mReserved << ").";
    }
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
    // Back to the safe, bounds-checked cursor: no reservation to validate until
    // reserve() is called again for this timeframe.
    mReserved = -1;
    return true;
  }

  void setLabel(const char* label)
  {
    mBuilder->setLabel(label);
  }

  /// reserve @a size rows when filling, so that we do not
  /// spend time reallocating the buffers.
  /// Switches the internal cursor to UnsafeAppend (no capacity check),
  /// which is safe because we just reserved enough space.
  void reserve(int64_t size)
  {
    mBuilder->reserve(typename persistent_table_t::column_types{}, size);
    mReserved = size;
    cursor = std::move(FFL(mBuilder->template unsafeCursor<persistent_table_t>()));
  }

  void release()
  {
    // Called once per timeframe, when the table is finalized. If reserve() was
    // used (switching to UnsafeAppend, which skips per-row bounds checks), make
    // sure we did not write past what we reserved: mCount + 1 is the number of
    // rows actually filled, mReserved the capacity we requested. Overrunning it
    // is silent memory corruption of the arrow buffers, so we fail hard here,
    // before the (corrupt) table is serialized downstream. mReserved < 0 means
    // reserve() was not called and the safe cursor was used: nothing to check.
    if (mReserved >= 0 && mCount + 1 > mReserved) {
      LOG(fatal) << "Table '" << outputSpec.binding.value << "': filled " << (mCount + 1)
                 << " rows after reserve(" << mReserved
                 << "). UnsafeAppend overran the reserved buffer — reserve() must request "
                    "at least as many rows as are filled.";
    }
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
  /// Number of rows reserved via reserve() (which switches to UnsafeAppend);
  /// -1 when reserve() was never called. Used by the destructor to detect an
  /// UnsafeAppend overrun.
  int64_t mReserved = -1;
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

  static constexpr auto spec()
  {
    return soa::tableRef2OutputSpec<table_t::ref>();
  }

  static constexpr auto ref()
  {
    return soa::tableRef2OutputRef<table_t::ref>();
  }
};

/// For the table-producing category of templates
/// * In a multi-origin case the origin is provided by the type
/// * In a rewritten origin case, we need to modify the output designation

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
  constexpr static auto sources = M::template generateSources<o2::aod::Hash<Ref.origin_hash>>();

  OutputSpec outputSpec{soa::tableRef2OutputSpec<Ref>()};
  static OutputSpec updateOutputSpec(header::DataOrigin const& newOrigin = header::DataOrigin{"AOD"})
  {
    return soa::tableRef2OutputSpec<Ref>(newOrigin);
  }

  std::array<InputSpec, sources.size()> requiredInputs = getRequiredInputs();
  static constexpr auto getRequiredInputs(header::DataOrigin const& newOrigin = header::DataOrigin{"AOD"})
  {
    return [&newOrigin]<size_t... Is>(std::index_sequence<Is...>) {
      return std::array{soa::tableRef2InputSpec<sources[Is]>(newOrigin)...};
    }(std::make_index_sequence<sources.size()>());
  }
};

/// This helper struct allows you to declare extended tables which should be
/// created by the task (as opposed to those pre-defined by data model)
template <typename T>
concept is_spawnable = soa::has_metadata<aod::MetadataTrait<o2::aod::Hash<T::originals[T::originals.size() - 1].desc_hash>>> && soa::has_extension<typename aod::MetadataTrait<o2::aod::Hash<T::originals[T::originals.size() - 1].desc_hash>>::metadata>;

template <typename T>
concept is_dynamically_spawnable = soa::has_metadata<aod::MetadataTrait<o2::aod::Hash<T::originals[T::originals.size() - 1].desc_hash>>> && soa::has_configurable_extension<typename aod::MetadataTrait<o2::aod::Hash<T::originals[T::originals.size() - 1].desc_hash>>::metadata>;

template <is_spawnable T>
consteval auto transformBase()
{
  using metadata = typename aod::MetadataTrait<o2::aod::Hash<T::originals[T::originals.size() - 1].desc_hash>>::metadata;
  return TableTransform<metadata, metadata::template extension_table_t_from<o2::aod::Hash<T::originals[T::originals.size() - 1].origin_hash>>::ref>{};
}

/// for the automatic table templates
/// * In a multi-origin case the origin is provided by the type
/// * In a rewritten origin case the output designation needs to be changed through base class
/// * The extraction of the elements happens in AnalysisManagers using the origin information from the base class
template <is_spawnable T>
struct Spawns : decltype(transformBase<T>()) {
  using spawnable_t = T;
  using metadata = decltype(transformBase<T>())::metadata;
  using extension_t = typename metadata::template extension_table_t_from<o2::aod::Hash<T::originals[T::originals.size() - 1].origin_hash>>;
  using expression_pack_t = typename metadata::expression_pack_t;
  static constexpr size_t N = framework::pack_size(expression_pack_t{});

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
  std::array<o2::framework::expressions::Projector, N> projectors = []<typename... C>(framework::pack<C...>)->std::array<expressions::Projector, sizeof...(C)>
  {
    return {{std::move(C::Projector())...}};
  }
  (expression_pack_t{});
  std::shared_ptr<gandiva::Projector> projector = nullptr;
  std::shared_ptr<arrow::Schema> schema = []() {
    auto s = std::make_shared<arrow::Schema>(o2::soa::createFieldsFromColumns(expression_pack_t{}));
    s->WithMetadata(std::make_shared<arrow::KeyValueMetadata>(std::vector{std::string{"label"}}, std::vector{std::string{o2::aod::label<T::ref>()}}));
    return s;
  }();
};

template <typename T>
concept is_spawns = requires(T t) {
  typename T::metadata;
  typename T::expression_pack_t;
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
  using extension_t = typename metadata::template extension_table_t_from<o2::aod::Hash<T::originals[T::originals.size() - 1].origin_hash>>;
  using placeholders_pack_t = typename metadata::placeholders_pack_t;
  static constexpr size_t N = framework::pack_size(placeholders_pack_t{});

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
  std::shared_ptr<arrow::Schema> schema = []() {
    auto s = std::make_shared<arrow::Schema>(o2::soa::createFieldsFromColumns(placeholders_pack_t{}));
    s->WithMetadata(std::make_shared<arrow::KeyValueMetadata>(std::vector{std::string{"label"}}, std::vector{std::string{o2::aod::label<T::ref>()}}));
    return s;
  }();
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
  typename T::placeholders_pack_t;
  requires std::same_as<decltype(t.projector), std::shared_ptr<gandiva::Projector>>;
  requires std::same_as<decltype(t.needRecompilation), bool>;
  &T::recompile;
};

/// Policy to control index building
/// Exclusive index: each entry in a row has a valid index
/// Sparse index: values in a row can be (-1), index table is isomorphic (joinable) to T1
struct Exclusive {
};
struct Sparse {
};

/// This helper struct allows you to declare index tables to be created in a task
template <soa::is_index_table T>
consteval auto transformBase()
{
  using metadata = typename aod::MetadataTrait<o2::aod::Hash<T::ref.desc_hash>>::metadata;
  return TableTransform<metadata, T::ref>{};
}

template <soa::is_index_table T>
struct Builds : decltype(transformBase<T>()) {
  using buildable_t = T;
  using metadata = decltype(transformBase<T>())::metadata;
  using Key = metadata::Key;
  using H = typename T::first_t;
  using Ts = typename T::rest_t;
  using index_pack_t = metadata::index_pack_t;

  std::shared_ptr<arrow::Schema> outputSchema = []() { return std::make_shared<arrow::Schema>(soa::createFieldsFromColumns(index_pack_t{}))->WithMetadata(std::make_shared<arrow::KeyValueMetadata>(std::vector{std::string{"label"}}, std::vector{std::string{o2::aod::label<T::ref>()}})); }();

  std::vector<soa::IndexRecord> map = soa::getIndexMapping<metadata>();

  std::vector<framework::IndexColumnBuilder> builders;

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

  static consteval auto pack()
  {
    return index_pack_t{};
  }

  auto build(std::vector<std::shared_ptr<arrow::Table>>&& tables)
  {
    this->table = std::make_shared<T>(soa::IndexBuilder::materialize(builders, std::forward<std::vector<std::shared_ptr<arrow::Table>>>(tables), map, outputSchema, metadata::exclusive));
    return (this->table != nullptr);
  }
};

template <typename T>
concept is_builds = requires(T t) {
  typename T::metadata;
  typename T::Key;
  requires std::same_as<decltype(t.map), std::vector<soa::IndexRecord>>;
};

/// a task with rewritten origin, if running together with a task with the default, will
/// have a different name and thus its output would be routed separately

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

/// Partition ties directly to the argument type
/// in a case with several origins in subscriptions it will get the correct input, as the type contains the origin
/// in a case with rewritten origin the type stays the same, so the association stays correct
/// FIXME: currently partition has to rerun the selection each time the invokeProcess is called
///        the real reason is to provide grouped parts for the process functions that request it
///        better solution would be to "slice" the selection, as is already done in GroupSlicer
///        for the same purpose, instead of reapplying the filtering
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
} // namespace o2::framework

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
} // namespace o2::soa

#endif // o2_framework_AnalysisHelpers_H_DEFINED
