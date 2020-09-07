// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
#include <ROOT/RDataFrame.hxx>
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

template <typename T>
struct Produces {
  static_assert(always_static_assert_v<T>, "Type must be a o2::soa::Table");
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

template <typename T>
struct Partition {
  Partition(expressions::Node&& filter_) : filter{std::move(filter_)}
  {
  }

  void setTable(const T& table)
  {
    auto schema = table.asArrowTable()->schema();
    expressions::Operations ops = createOperations(filter);
    if (isSchemaCompatible(schema, ops)) {
      mTree = createExpressionTree(ops, schema);
    } else {
      throw std::runtime_error("Partition filter does not match declared table type");
    }

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

  void updatePlaceholders(InitContext& context)
  {
    expressions::updatePlaceholders(filter, context);
  }

  expressions::Filter filter;
  gandiva::NodePtr mTree = nullptr;
  std::unique_ptr<o2::soa::Filtered<T>> mFiltered = nullptr;

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

namespace o2
{
namespace analysis
{

/// Do a single loop on all the entries of the @a input table
ROOT::RDataFrame doSingleLoopOn(std::unique_ptr<framework::TableConsumer>& input);

/// Do a double loop on all the entries with the same value for the \a grouping
/// of the @a input table, where the entries for the outer index are prefixed
/// with `<name>_` while the entries for the inner loop are prefixed with
/// `<name>bar_`.
ROOT::RDataFrame doSelfCombinationsWith(std::unique_ptr<framework::TableConsumer>& input,
                                        std::string name = "p",
                                        std::string grouping = "eventID");

} // namespace analysis
} // namespace o2

#endif // o2_framework_AnalysisHelpers_H_DEFINED
