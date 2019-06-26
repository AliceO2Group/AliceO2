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

O2_ARROW_STL_CONVERSION(bool, BooleanType)
O2_ARROW_STL_CONVERSION(int8_t, Int8Type)
O2_ARROW_STL_CONVERSION(int16_t, Int16Type)
O2_ARROW_STL_CONVERSION(int32_t, Int32Type)
O2_ARROW_STL_CONVERSION(int64_t, Int64Type)
O2_ARROW_STL_CONVERSION(uint8_t, UInt8Type)
O2_ARROW_STL_CONVERSION(uint16_t, UInt16Type)
O2_ARROW_STL_CONVERSION(uint32_t, UInt32Type)
O2_ARROW_STL_CONVERSION(uint64_t, UInt64Type)
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

/// Helper class which creates a lambda suitable for building
/// an arrow table from a tuple. This can be used, for example
/// to build an arrow::Table from a TDataFrame.
class TableBuilder
{
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
    using BuildersTuple = typename std::tuple<std::unique_ptr<typename BuilderTraits<ARGS>::BuilderType>...>;
    mSchema = std::make_shared<arrow::Schema>(TableBuilderHelpers::makeFields<ARGS...>(columnNames));

    BuildersTuple* builders = new BuildersTuple(BuilderMaker<ARGS>::make(mMemoryPool)...);
    if (nRows != -1) {
      auto seq = std::make_index_sequence<sizeof...(ARGS)>{};
      TableBuilderHelpers::reserveAll(*builders, nRows, seq);
    }
    mBuilders = builders; // We store the builders
  }

  template <typename... ARGS>
  auto makeFinalizer()
  {
    using BuildersTuple = typename std::tuple<std::unique_ptr<typename BuilderTraits<ARGS>::BuilderType>...>;
    mFinalizer = [schema = mSchema, &arrays = mArrays, builders = (BuildersTuple*)mBuilders]() -> std::shared_ptr<arrow::Table> {
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
    using BuildersTuple = typename std::tuple<std::unique_ptr<typename BuilderTraits<ARGS>::BuilderType>...>;
    constexpr int nColumns = sizeof...(ARGS);
    validate<ARGS...>(columnNames);
    mArrays.resize(nColumns);
    makeBuilders<ARGS...>(columnNames, 1000);
    makeFinalizer<ARGS...>();

    // Callback used to fill the builders
    return [builders = (BuildersTuple*)mBuilders](unsigned int slot, typename BuilderMaker<ARGS>::FillType... args) -> void {
      auto status = TableBuilderHelpers::append(*builders, std::index_sequence_for<ARGS...>{}, std::forward_as_tuple(args...));
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
    constexpr auto tuple_size = std::tuple_size_v<typename T::columns>;
    return cursorHelper<T>(std::make_index_sequence<tuple_size>());
  }

  template <typename... ARGS>
  auto preallocatedPersist(std::vector<std::string> const& columnNames, int nRows)
  {
    using BuildersTuple = typename std::tuple<std::unique_ptr<typename BuilderTraits<ARGS>::BuilderType>...>;
    constexpr int nColumns = sizeof...(ARGS);
    validate<ARGS...>(columnNames);
    mArrays.resize(nColumns);
    makeBuilders<ARGS...>(columnNames, nRows);
    makeFinalizer<ARGS...>();

    // Callback used to fill the builders
    return [builders = (BuildersTuple*)mBuilders](unsigned int slot, typename BuilderMaker<ARGS>::FillType... args) -> void {
      TableBuilderHelpers::unsafeAppend(*builders, std::index_sequence_for<ARGS...>{}, std::forward_as_tuple(args...));
    };
  }

  template <typename... ARGS>
  auto bulkPersist(std::vector<std::string> const& columnNames, size_t nRows)
  {
    using BuildersTuple = typename std::tuple<std::unique_ptr<typename BuilderTraits<ARGS>::BuilderType>...>;
    constexpr int nColumns = sizeof...(ARGS);
    validate<ARGS...>(columnNames);
    mArrays.resize(nColumns);
    makeBuilders<ARGS...>(columnNames, nRows);
    makeFinalizer<ARGS...>();

    return [builders = (BuildersTuple*)mBuilders](unsigned int slot, size_t batchSize, typename BuilderMaker<ARGS>::FillType const*... args) -> void {
      TableBuilderHelpers::bulkAppend(*builders, batchSize, std::index_sequence_for<ARGS...>{}, std::forward_as_tuple(args...));
    };
  }

  /// Actually creates the arrow::Table from the builders
  std::shared_ptr<arrow::Table> finalize();

 private:
  template <typename T, size_t... Is>
  auto cursorHelper(std::index_sequence<Is...> s)
  {
    std::vector<std::string> columnNames{std::tuple_element_t<Is, typename T::columns>::label()...};
    return this->template persist<typename std::tuple_element_t<Is, typename T::columns>::type...>(columnNames);
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
