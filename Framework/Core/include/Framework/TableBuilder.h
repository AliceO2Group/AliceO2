// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_FRAMEWORK_TABLEBUILDER_H_
#define O2_FRAMEWORK_TABLEBUILDER_H_

#include "Framework/ASoA.h"
#include "Framework/StructToTuple.h"
#include "Framework/FunctionalHelpers.h"
#include "Framework/VariantHelpers.h"

// Apparently needs to be on top of the arrow includes.
#include <sstream>

#include <arrow/stl.h>
#include <arrow/type_traits.h>
#include <arrow/table.h>
#include <arrow/builder.h>

#include <functional>
#include <vector>
#include <string>
#include <memory>
#include <tuple>

namespace arrow
{
class ArrayBuilder;
class Table;
class Array;
} // namespace arrow

namespace o2
{
namespace framework
{
namespace detail
{
// This is needed by Arrow 0.12.0 which dropped
//
//      using ArrowType = ArrowType_;
//
// from ARROW_STL_CONVERSION
template <typename T>
struct ConversionTraits {
};

#define O2_ARROW_STL_CONVERSION(c_type, ArrowType_) \
  template <>                                       \
  struct ConversionTraits<c_type> {                 \
    using ArrowType = ::arrow::ArrowType_;          \
  };

// FIXME: for now we use Int8 to store booleans
O2_ARROW_STL_CONVERSION(bool, Int8Type)
O2_ARROW_STL_CONVERSION(int8_t, Int8Type)
O2_ARROW_STL_CONVERSION(int16_t, Int16Type)
O2_ARROW_STL_CONVERSION(int32_t, Int32Type)
O2_ARROW_STL_CONVERSION(long long, Int64Type)
O2_ARROW_STL_CONVERSION(long, Int64Type)
O2_ARROW_STL_CONVERSION(uint8_t, UInt8Type)
O2_ARROW_STL_CONVERSION(uint16_t, UInt16Type)
O2_ARROW_STL_CONVERSION(uint32_t, UInt32Type)
O2_ARROW_STL_CONVERSION(long long unsigned, UInt64Type)
O2_ARROW_STL_CONVERSION(long unsigned, UInt64Type)
O2_ARROW_STL_CONVERSION(float, FloatType)
O2_ARROW_STL_CONVERSION(double, DoubleType)
O2_ARROW_STL_CONVERSION(std::string, StringType)
} // namespace detail

struct BuilderUtils {
  template <typename BuilderType, typename T>
  static arrow::Status append(BuilderType& builder, T value)
  {
    return builder->Append(value);
  }

  /// Appender for the pointer case.
  /// Assumes that the pointer actually points to a buffer
  /// which contains the correct number of elements.
  template <typename BuilderType, typename T>
  static arrow::Status append(BuilderType& builder, T* data)
  {
    return builder->Append(reinterpret_cast<const uint8_t*>(data));
  }
  /// Appender for the array case.
  template <typename BuilderType, typename T, int N>
  static arrow::Status append(BuilderType& builder, T (&data)[N])
  {
    return builder->Append(reinterpret_cast<const uint8_t*>(data));
  }

  /// Appender for the array case.
  template <typename BuilderType, typename T, int N>
  static arrow::Status append(BuilderType& builder, std::array<T, N> const& data)
  {
    return builder->Append(reinterpret_cast<const uint8_t*>(data.data()));
  }

  template <typename BuilderType, typename T>
  static void unsafeAppend(BuilderType& builder, T value)
  {
    return builder->UnsafeAppend(value);
  }

  template <typename BuilderType, typename PTR>
  static arrow::Status bulkAppend(BuilderType& builder, size_t bulkSize, const PTR ptr)
  {
    return builder->AppendValues(ptr, bulkSize, nullptr);
  }

  template <typename BuilderType, typename ITERATOR>
  static arrow::Status append(BuilderType& builder, std::pair<ITERATOR, ITERATOR> ip)
  {
    using ArrowType = typename detail::ConversionTraits<typename ITERATOR::value_type>::ArrowType;
    using ValueBuilderType = typename arrow::TypeTraits<ArrowType>::BuilderType;
    // FIXME: for the moment we do not fill things.
    auto status = builder->Append();
    auto valueBuilder = reinterpret_cast<ValueBuilderType*>(builder->value_builder());
    return status & valueBuilder->AppendValues(&*ip.first, std::distance(ip.first, ip.second));
  }

  // Lists do not have UnsafeAppend so we need to use the slow path in any case.
  template <typename BuilderType, typename ITERATOR>
  static void unsafeAppend(BuilderType& builder, std::pair<ITERATOR, ITERATOR> ip)
  {
    using ArrowType = typename detail::ConversionTraits<typename ITERATOR::value_type>::ArrowType;
    using ValueBuilderType = typename arrow::TypeTraits<ArrowType>::BuilderType;
    // FIXME: for the moment we do not fill things.
    auto status = builder->Append();
    auto valueBuilder = reinterpret_cast<ValueBuilderType*>(builder->value_builder());
    status &= valueBuilder->AppendValues(&*ip.first, std::distance(ip.first, ip.second));
    if (!status.ok()) {
      throw std::runtime_error("Unable to append values");
    }
    return;
  }
};

template <typename T>
struct BuilderMaker {
  using FillType = T;
  using ArrowType = typename detail::ConversionTraits<T>::ArrowType;
  using BuilderType = typename arrow::TypeTraits<ArrowType>::BuilderType;

  static std::unique_ptr<BuilderType> make(arrow::MemoryPool* pool)
  {
    return std::make_unique<BuilderType>(pool);
  }

  static std::shared_ptr<arrow::DataType> make_datatype()
  {
    return arrow::TypeTraits<ArrowType>::type_singleton();
  }

  static arrow::Status append(BuilderType& builder, T value)
  {
    return builder.Append(value);
  }

  template <int N>
  static arrow::Status append(BuilderType& builder, std::array<T, N>& value)
  {
    return builder.Append(value);
  }
};

template <typename ITERATOR>
struct BuilderMaker<std::pair<ITERATOR, ITERATOR>> {
  using FillType = std::pair<ITERATOR, ITERATOR>;
  using ArrowType = arrow::ListType;
  using ValueType = typename detail::ConversionTraits<typename ITERATOR::value_type>::ArrowType;
  using BuilderType = arrow::ListBuilder;
  using ValueBuilder = typename arrow::TypeTraits<ValueType>::BuilderType;

  static std::unique_ptr<BuilderType> make(arrow::MemoryPool* pool)
  {
    auto valueBuilder = std::make_shared<ValueBuilder>(pool);
    return std::make_unique<arrow::ListBuilder>(pool, valueBuilder);
  }

  static std::shared_ptr<arrow::DataType> make_datatype()
  {
    return arrow::list(arrow::TypeTraits<ValueType>::type_singleton());
  }
};

template <typename T, int N>
struct BuilderMaker<T[N]> {
  using FillType = T*;
  using BuilderType = arrow::FixedSizeBinaryBuilder;
  using ArrowType = arrow::FixedSizeBinaryType;

  static std::unique_ptr<BuilderType> make(arrow::MemoryPool* pool)
  {
    return std::make_unique<BuilderType>(arrow::fixed_size_binary(sizeof(T[N])), pool);
  }

  static std::shared_ptr<arrow::DataType> make_datatype()
  {
    return arrow::fixed_size_binary(sizeof(T[N]));
  }
};

template <typename... ARGS>
auto make_builders()
{
  return std::make_tuple(std::make_unique<ARGS>()...);
}

template <typename T>
struct BuilderTraits {
  using ArrowType = typename detail::ConversionTraits<T>::ArrowType;
  using BuilderType = typename arrow::TypeTraits<ArrowType>::BuilderType;
};

// Support for building tables where each entry is an iterator pair.
// We map them to an arrow::list for now.
template <typename ITERATOR>
struct BuilderTraits<std::pair<ITERATOR, ITERATOR>> {
  using ArrowType = arrow::ListType;
  using BuilderType = arrow::ListBuilder;
};

// Support for building array columns
// FIXME: move to use FixedSizeList<T> once we move to 0.16.1
template <typename T, int N>
struct BuilderTraits<T[N]> {
  using ArrowType = arrow::FixedSizeBinaryType;
  using BuilderType = arrow::FixedSizeBinaryBuilder;
};

struct TableBuilderHelpers {
  template <typename... ARGS>
  static auto makeFields(std::vector<std::string> const& names)
  {
    std::vector<std::shared_ptr<arrow::DataType>> types{BuilderMaker<ARGS>::make_datatype()...};
    std::vector<std::shared_ptr<arrow::Field>> result;
    for (size_t i = 0; i < names.size(); ++i) {
      result.emplace_back(std::make_shared<arrow::Field>(names[i], types[i], true, nullptr));
    }
    return std::move(result);
  }

  /// Invokes the append method for each entry in the tuple
  template <std::size_t... Is, typename BUILDERS, typename VALUES>
  static bool append(BUILDERS& builders, std::index_sequence<Is...>, VALUES&& values)
  {
    return (BuilderUtils::append(std::get<Is>(builders), std::get<Is>(values)).ok() && ...);
  }

  /// Invokes the UnsafeAppend method for each entry in the tuple
  /// For this to be used, one should make sure the number of entries
  /// is known a-priori.
  template <std::size_t... Is, typename BUILDERS, typename VALUES>
  static void unsafeAppend(BUILDERS& builders, std::index_sequence<Is...>, VALUES&& values)
  {
    (BuilderUtils::unsafeAppend(std::get<Is>(builders), std::get<Is>(values)), ...);
  }

  template <std::size_t... Is, typename BUILDERS, typename PTRS>
  static bool bulkAppend(BUILDERS& builders, size_t bulkSize, std::index_sequence<Is...>, PTRS ptrs)
  {
    return (BuilderUtils::bulkAppend(std::get<Is>(builders), bulkSize, std::get<Is>(ptrs)).ok() && ...);
  }

  /// Invokes the append method for each entry in the tuple
  template <typename BUILDERS, std::size_t... Is>
  static bool finalize(std::vector<std::shared_ptr<arrow::Array>>& arrays, BUILDERS& builders, std::index_sequence<Is...> seq)
  {
    return (std::get<Is>(builders)->Finish(&arrays[Is]).ok() && ...);
  }

  template <typename BUILDERS, std::size_t... Is>
  static bool reserveAll(BUILDERS& builders, size_t s, std::index_sequence<Is...>)
  {
    return (std::get<Is>(builders)->Reserve(s).ok() && ...);
  }
};

template <typename... ARGS>
auto tuple_to_pack(std::tuple<ARGS...>&&)
{
  return framework::pack<ARGS...>{};
}

/// Detect if this is a fixed size array
/// FIXME: Notice that C++20 provides a method with the same name
/// so we should move to it when we switch.
template <class T>
struct is_bounded_array : std::false_type {
};

template <class T, std::size_t N>
struct is_bounded_array<T[N]> : std::true_type {
};

template <class T, std::size_t N>
struct is_bounded_array<std::array<T, N>> : std::true_type {
};

/// Helper class which creates a lambda suitable for building
/// an arrow table from a tuple. This can be used, for example
/// to build an arrow::Table from a TDataFrame.
class TableBuilder
{
  template <typename... ARGS>
  using BuildersTuple = typename std::tuple<std::unique_ptr<typename BuilderTraits<ARGS>::BuilderType>...>;

  /// Get the builders, assumning they were created with a given pack
  ///  of basic types
  template <typename... ARGS>
  auto getBuilders(o2::framework::pack<ARGS...> pack)
  {
    return (BuildersTuple<ARGS...>*)mBuilders;
  }

  template <typename... ARGS>
  void validate(std::vector<std::string> const& columnNames)
  {
    constexpr int nColumns = sizeof...(ARGS);
    if (nColumns != columnNames.size()) {
      throw std::runtime_error("Mismatching number of column types and names");
    }
    if (mBuilders != nullptr) {
      throw std::runtime_error("TableBuilder::persist can only be invoked once per instance");
    }
  }

  template <typename... ARGS>
  auto makeBuilders(std::vector<std::string> const& columnNames, size_t nRows)
  {
    mSchema = std::make_shared<arrow::Schema>(TableBuilderHelpers::makeFields<ARGS...>(columnNames));

    auto builders = new BuildersTuple<ARGS...>(BuilderMaker<ARGS>::make(mMemoryPool)...);
    if (nRows != -1) {
      auto seq = std::make_index_sequence<sizeof...(ARGS)>{};
      TableBuilderHelpers::reserveAll(*builders, nRows, seq);
    }
    mBuilders = builders; // We store the builders
  }

  template <typename... ARGS>
  auto makeFinalizer()
  {
    mFinalizer = [schema = mSchema, &arrays = mArrays, builders = (BuildersTuple<ARGS...>*)mBuilders]() -> std::shared_ptr<arrow::Table> {
      auto status = TableBuilderHelpers::finalize(arrays, *builders, std::make_index_sequence<sizeof...(ARGS)>{});
      if (status == false) {
        throw std::runtime_error("Unable to finalize");
      }
      return arrow::Table::Make(schema, arrays);
    };
  }

 public:
  TableBuilder(arrow::MemoryPool* pool = arrow::default_memory_pool())
    : mBuilders{nullptr},
      mMemoryPool{pool}
  {
  }

  /// Creates a lambda which is suitable to persist things
  /// in an arrow::Table
  template <typename... ARGS>
  auto persist(std::vector<std::string> const& columnNames)
  {
    using args_pack_t = framework::pack<ARGS...>;
    if constexpr (sizeof...(ARGS) == 1 &&
                  is_bounded_array<pack_element_t<0, args_pack_t>>::value == false &&
                  std::is_arithmetic_v<pack_element_t<0, args_pack_t>> == false) {
      using objType_t = pack_element_t<0, framework::pack<ARGS...>>;
      using argsPack_t = decltype(tuple_to_pack(framework::to_tuple(std::declval<objType_t>())));
      auto persister = persistTuple(argsPack_t{}, columnNames);
      return [persister = persister](unsigned int slot, objType_t const& obj) -> void {
        auto t = to_tuple(obj);
        persister(slot, t);
      };
    } else if constexpr (sizeof...(ARGS) == 1 &&
                         is_bounded_array<pack_element_t<0, args_pack_t>>::value == true) {
      using objType_t = pack_element_t<0, framework::pack<ARGS...>>;
      auto persister = persistTuple(framework::pack<objType_t>{}, columnNames);
      // Callback used to fill the builders
      return [persister = persister](unsigned int slot, typename BuilderMaker<objType_t>::FillType const& arg) -> void {
        persister(slot, std::forward_as_tuple(arg));
      };
    } else if constexpr (sizeof...(ARGS) >= 1) {
      auto persister = persistTuple(framework::pack<ARGS...>{}, columnNames);
      // Callback used to fill the builders
      return [persister = persister](unsigned int slot, typename BuilderMaker<ARGS>::FillType... args) -> void {
        persister(slot, std::forward_as_tuple(args...));
      };
    } else {
      static_assert(o2::framework::always_static_assert_v<ARGS...>, "Unmanaged case");
    }
  }

  /// Same a the above, but use a tuple to persist stuff.
  template <typename... ARGS>
  auto persistTuple(framework::pack<ARGS...>, std::vector<std::string> const& columnNames)
  {
    constexpr int nColumns = sizeof...(ARGS);
    validate<ARGS...>(columnNames);
    mArrays.resize(nColumns);
    makeBuilders<ARGS...>(columnNames, 1000);
    makeFinalizer<ARGS...>();

    // Callback used to fill the builders
    using FillTuple = std::tuple<typename BuilderMaker<ARGS>::FillType...>;
    return [builders = (BuildersTuple<ARGS...>*)mBuilders](unsigned int slot, FillTuple const& t) -> void {
      auto status = TableBuilderHelpers::append(*builders, std::index_sequence_for<ARGS...>{}, t);
      if (status == false) {
        throw std::runtime_error("Unable to append");
      }
    };
  }

  // Same as above, but starting from a o2::soa::Table, which has all the
  // information already available.
  template <typename T>
  auto cursor()
  {
    using persistent_filter = soa::FilterPersistentColumns<T>;
    using persistent_columns_pack = typename persistent_filter::persistent_columns_pack;
    constexpr auto persistent_size = pack_size(persistent_columns_pack{});
    return cursorHelper<typename persistent_filter::persistent_table_t>(std::make_index_sequence<persistent_size>());
  }

  template <typename T, typename E>
  auto cursor()
  {
    using persistent_filter = soa::FilterPersistentColumns<T>;
    using persistent_columns_pack = typename persistent_filter::persistent_columns_pack;
    constexpr auto persistent_size = pack_size(persistent_columns_pack{});
    return cursorHelper<typename persistent_filter::persistent_table_t, E>(std::make_index_sequence<persistent_size>());
  }

  template <typename... ARGS>
  auto preallocatedPersist(std::vector<std::string> const& columnNames, int nRows)
  {
    constexpr int nColumns = sizeof...(ARGS);
    validate<ARGS...>(columnNames);
    mArrays.resize(nColumns);
    makeBuilders<ARGS...>(columnNames, nRows);
    makeFinalizer<ARGS...>();

    // Callback used to fill the builders
    return [builders = (BuildersTuple<ARGS...>*)mBuilders](unsigned int slot, typename BuilderMaker<ARGS>::FillType... args) -> void {
      TableBuilderHelpers::unsafeAppend(*builders, std::index_sequence_for<ARGS...>{}, std::forward_as_tuple(args...));
    };
  }

  template <typename... ARGS>
  auto bulkPersist(std::vector<std::string> const& columnNames, size_t nRows)
  {
    constexpr int nColumns = sizeof...(ARGS);
    validate<ARGS...>(columnNames);
    mArrays.resize(nColumns);
    makeBuilders<ARGS...>(columnNames, nRows);
    makeFinalizer<ARGS...>();

    return [builders = (BuildersTuple<ARGS...>*)mBuilders](unsigned int slot, size_t batchSize, typename BuilderMaker<ARGS>::FillType const*... args) -> void {
      TableBuilderHelpers::bulkAppend(*builders, batchSize, std::index_sequence_for<ARGS...>{}, std::forward_as_tuple(args...));
    };
  }

  /// Reserve method to expand the columns as needed.
  template <typename... ARGS>
  auto reserve(o2::framework::pack<ARGS...> pack, int s)
  {
    visitBuilders(pack, overloaded{[s](auto& builder) { return builder.Reserve(s).ok(); }});
  }

  /// Invoke the appropriate visitor on the various builders
  template <typename... ARGS, typename V>
  auto visitBuilders(o2::framework::pack<ARGS...> pack, V&& visitor)
  {
    auto builders = getBuilders(pack);
    auto visitAll = [visitor](std::unique_ptr<typename BuilderTraits<ARGS>::BuilderType>&... args) { (visitor(*args), ...); };
    return std::apply(visitAll, *builders);
  }


  /// Actually creates the arrow::Table from the builders
  std::shared_ptr<arrow::Table> finalize();

 private:
  /// Helper which actually creates the insertion cursor. Notice that the
  /// template argument T is a o2::soa::Table which contains only the
  /// persistent columns.
  template <typename T, size_t... Is>
  auto cursorHelper(std::index_sequence<Is...> s)
  {
    std::vector<std::string> columnNames{pack_element_t<Is, typename T::columns>::label()...};
    return this->template persist<typename pack_element_t<Is, typename T::columns>::type...>(columnNames);
  }

  template <typename T, typename E, size_t... Is>
  auto cursorHelper(std::index_sequence<Is...> s)
  {
    std::vector<std::string> columnNames{pack_element_t<Is, typename T::columns>::label()...};
    return this->template persist<E>(columnNames);
  }

  std::function<void(void)> mFinalizer;
  void* mBuilders;
  arrow::MemoryPool* mMemoryPool;
  std::shared_ptr<arrow::Schema> mSchema;
  std::vector<std::shared_ptr<arrow::Array>> mArrays;
};

} // namespace framework
} // namespace o2
#endif // FRAMEWORK_TABLEBUILDER_H
