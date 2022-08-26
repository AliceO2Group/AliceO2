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

#ifndef O2_FRAMEWORK_TABLEBUILDER_H_
#define O2_FRAMEWORK_TABLEBUILDER_H_

#include "Framework/ASoA.h"
#include "Framework/StructToTuple.h"
#include "Framework/FunctionalHelpers.h"
#include "Framework/VariantHelpers.h"
#include "Framework/RuntimeError.h"
#include "arrow/type_traits.h"

// Apparently needs to be on top of the arrow includes.
#include <sstream>

#include <arrow/chunked_array.h>
#include <arrow/status.h>
#include <arrow/memory_pool.h>
#include <arrow/stl.h>
#include <arrow/type_traits.h>
#include <arrow/table.h>
#include <arrow/builder.h>

#include <vector>
#include <string>
#include <memory>
#include <tuple>
#include <type_traits>

namespace arrow
{
class ArrayBuilder;
class Table;
class Array;
} // namespace arrow

template <typename T>
struct BulkInfo {
  const T ptr;
  size_t size;
};

namespace o2::framework
{
namespace detail
{
/// FIXME: adapt type conversion to arrow 1.0
// This is needed by Arrow 0.12.0 which dropped
//
//      using ArrowType = ArrowType_;
//
// from ARROW_STL_CONVERSION
template <typename T>
struct ConversionTraits {
};

template <typename T, int N>
struct ConversionTraits<T (&)[N]> {
  using ArrowType = ::arrow::FixedSizeListType;
};

template <typename T, int N>
struct ConversionTraits<T[N]> {
  using ArrowType = ::arrow::FixedSizeListType;
};

template <typename T>
struct ConversionTraits<std::vector<T>> {
  using ArrowType = ::arrow::ListType;
};

#define O2_ARROW_STL_CONVERSION(c_type, ArrowType_) \
  template <>                                       \
  struct ConversionTraits<c_type> {                 \
    using ArrowType = ::arrow::ArrowType_;          \
  };

// FIXME: for now we use Int8 to store booleans
O2_ARROW_STL_CONVERSION(bool, BooleanType)
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

void addLabelToSchema(std::shared_ptr<arrow::Schema>& schema, const char* label);

struct BuilderUtils {
  template <typename T>
  static arrow::Status appendToList(std::unique_ptr<arrow::FixedSizeListBuilder>& builder, T* data, int size = 1)
  {
    using ArrowType = typename detail::ConversionTraits<std::decay_t<T>>::ArrowType;
    using BuilderType = typename arrow::TypeTraits<ArrowType>::BuilderType;
    size_t numElements = static_cast<const arrow::FixedSizeListType*>(builder->type().get())->list_size();

    auto status = builder->AppendValues(size);
    auto ValueBuilder = static_cast<BuilderType*>(builder->value_builder());
    status &= ValueBuilder->AppendValues(data, numElements * size, nullptr);

    return status;
  }

  template <typename HolderType, typename T>
  static arrow::Status append(HolderType& holder, T value)
  {
    return static_cast<typename HolderType::Policy&>(holder).append(holder.builder, value);
  }

  template <typename HolderType>
  static arrow::Status flush(HolderType& holder)
  {
    return static_cast<typename HolderType::Policy&>(holder).flush(holder.builder);
  }

  /// Appender for the pointer case.
  /// Assumes that the pointer actually points to a buffer
  /// which contains the correct number of elements.
  template <typename HolderType, typename T>
  static arrow::Status append(HolderType& holder, T* data)
  {
    if constexpr (std::is_same_v<decltype(holder.builder), std::unique_ptr<arrow::FixedSizeListBuilder>>) {
      return appendToList<T>(holder.builder, data);
    } else {
      return holder.builder->Append(reinterpret_cast<const uint8_t*>(data));
    }
  }
  /// Appender for the array case.
  template <typename HolderType, typename T, int N>
  static arrow::Status append(HolderType& holder, T (&data)[N])
  {
    return holder.builder->Append(reinterpret_cast<const uint8_t*>(data));
  }

  /// Appender for the array case.
  template <typename HolderType, typename T, int N>
  static arrow::Status append(HolderType& holder, std::array<T, N> const& data)
  {
    return holder.builder->Append(reinterpret_cast<const uint8_t*>(data.data()));
  }

  /// Appender for the vector case.
  template <typename HolderType, typename T>
  static arrow::Status append(HolderType& holder, std::vector<T> const& data)
  {
    using ArrowType = typename detail::ConversionTraits<T>::ArrowType;
    using ValueBuilderType = typename arrow::TypeTraits<ArrowType>::BuilderType;
    auto status = holder.builder->Reserve(data.size());
    status &= holder.builder->Append();
    auto vbuilder = static_cast<ValueBuilderType*>(holder.builder->value_builder());
    status &= vbuilder->AppendValues(data.begin(), data.end());

    return status;
  }

  template <typename HolderType, typename T>
  static void unsafeAppend(HolderType& holder, std::vector<T> const& value)
  {
    auto status = append(holder, value);
    if (!status.ok()) {
      throw runtime_error("Unable to append to column");
    }
  }

  template <typename HolderType, typename T>
  static void unsafeAppend(HolderType& holder, T value)
  {
    return holder.builder->UnsafeAppend(value);
  }

  template <typename HolderType, typename T>
  static void unsafeAppend(HolderType& holder, T* value)
  {
    if constexpr (std::is_same_v<decltype(holder.builder), std::unique_ptr<arrow::FixedSizeListBuilder>>) {
      auto status = appendToList<T>(holder.builder, value);
    } else {
      return holder.builder->UnsafeAppend(reinterpret_cast<const uint8_t*>(value));
    }
  }

  template <typename HolderType, typename PTR>
  static arrow::Status bulkAppend(HolderType& holder, size_t bulkSize, const PTR ptr)
  {
    return holder.builder->AppendValues(ptr, bulkSize, nullptr);
  }

  template <typename HolderType, typename PTR>
  static arrow::Status bulkAppendChunked(HolderType& holder, BulkInfo<PTR> info)
  {
    // Appending nullptr is a no-op.
    if (info.ptr == nullptr) {
      return arrow::Status::OK();
    }
    if constexpr (std::is_same_v<decltype(holder.builder), std::unique_ptr<arrow::FixedSizeListBuilder>>) {
      if (appendToList<std::remove_pointer_t<decltype(info.ptr)>>(holder.builder, info.ptr, info.size).ok() == false) {
        throw runtime_error("Unable to append to column");
      } else {
        return arrow::Status::OK();
      }
    } else {
      if (holder.builder->AppendValues(info.ptr, info.size, nullptr).ok() == false) {
        throw runtime_error("Unable to append to column");
      } else {
        return arrow::Status::OK();
      }
    }
  }

  template <typename HolderType, typename ITERATOR>
  static arrow::Status append(HolderType& holder, std::pair<ITERATOR, ITERATOR> ip)
  {
    using ArrowType = typename detail::ConversionTraits<typename ITERATOR::value_type>::ArrowType;
    using ValueBuilderType = typename arrow::TypeTraits<ArrowType>::BuilderType;
    // FIXME: for the moment we do not fill things.
    auto status = holder.builder->Append();
    auto valueBuilder = reinterpret_cast<ValueBuilderType*>(holder.builder->value_builder());
    return status & valueBuilder->AppendValues(&*ip.first, std::distance(ip.first, ip.second));
  }

  // Lists do not have UnsafeAppend so we need to use the slow path in any case.
  template <typename HolderType, typename ITERATOR>
  static void unsafeAppend(HolderType& holder, std::pair<ITERATOR, ITERATOR> ip)
  {
    using ArrowType = typename detail::ConversionTraits<typename ITERATOR::value_type>::ArrowType;
    using ValueBuilderType = typename arrow::TypeTraits<ArrowType>::BuilderType;
    // FIXME: for the moment we do not fill things.
    auto status = holder.builder->Append();
    auto valueBuilder = reinterpret_cast<ValueBuilderType*>(holder.builder->value_builder());
    status &= valueBuilder->AppendValues(&*ip.first, std::distance(ip.first, ip.second));
    if (!status.ok()) {
      throw runtime_error("Unable to append values to valueBuilder!");
    }
    return;
  }
};

template <typename T>
struct BuilderMaker {
  using FillType = T;
  using STLValueType = T;
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

template <>
struct BuilderMaker<bool> {
  using FillType = bool;
  using STLValueType = bool;
  using ArrowType = typename detail::ConversionTraits<bool>::ArrowType;
  using BuilderType = typename arrow::TypeTraits<ArrowType>::BuilderType;

  static std::unique_ptr<BuilderType> make(arrow::MemoryPool* pool)
  {
    return std::make_unique<BuilderType>(pool);
  }

  static std::shared_ptr<arrow::DataType> make_datatype()
  {
    return arrow::TypeTraits<ArrowType>::type_singleton();
  }

  static arrow::Status append(BuilderType& builder, bool value)
  {
    return builder.Append(value);
  }
};

template <typename ITERATOR>
struct BuilderMaker<std::pair<ITERATOR, ITERATOR>> {
  using FillType = std::pair<ITERATOR, ITERATOR>;
  using STLValueType = typename ITERATOR::value_type;
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
struct BuilderMaker<T (&)[N]> {
  using FillType = T*;
  using STLValueType = T;
  using BuilderType = arrow::FixedSizeListBuilder;
  using ArrowType = arrow::FixedSizeListType;
  using ElementType = typename detail::ConversionTraits<T>::ArrowType;

  static std::unique_ptr<BuilderType> make(arrow::MemoryPool* pool)
  {
    std::unique_ptr<arrow::ArrayBuilder> valueBuilder;
    auto status =
      arrow::MakeBuilder(pool, arrow::TypeTraits<ElementType>::type_singleton(), &valueBuilder);
    return std::make_unique<BuilderType>(pool, std::move(valueBuilder), N);
  }

  static std::shared_ptr<arrow::DataType> make_datatype()
  {
    return arrow::fixed_size_list(arrow::TypeTraits<ElementType>::type_singleton(), N);
  }
};

template <typename T, int N>
struct BuilderMaker<T[N]> {
  using FillType = T*;
  using BuilderType = arrow::FixedSizeListBuilder;
  using ArrowType = arrow::FixedSizeListType;
  using ElementType = typename detail::ConversionTraits<T>::ArrowType;

  static std::unique_ptr<BuilderType> make(arrow::MemoryPool* pool)
  {
    std::unique_ptr<arrow::ArrayBuilder> valueBuilder;
    auto status =
      arrow::MakeBuilder(pool, arrow::TypeTraits<ElementType>::type_singleton(), &valueBuilder);
    return std::make_unique<BuilderType>(pool, std::move(valueBuilder), N);
  }

  static std::shared_ptr<arrow::DataType> make_datatype()
  {
    return arrow::fixed_size_list(arrow::TypeTraits<ElementType>::type_singleton(), N);
  }
};

template <typename T>
struct BuilderMaker<std::vector<T>> {
  using FillType = std::vector<T>;
  using BuilderType = arrow::ListBuilder;
  using ArrowType = arrow::ListType;
  using ElementType = typename detail::ConversionTraits<T>::ArrowType;

  static std::unique_ptr<BuilderType> make(arrow::MemoryPool* pool)
  {
    std::unique_ptr<arrow::ArrayBuilder> valueBuilder;
    auto status =
      arrow::MakeBuilder(pool, arrow::TypeTraits<ElementType>::type_singleton(), &valueBuilder);
    return std::make_unique<BuilderType>(pool, std::move(valueBuilder));
  }

  static std::shared_ptr<arrow::DataType> make_datatype()
  {
    return arrow::list(arrow::TypeTraits<ElementType>::type_singleton());
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
  using ArrowType = arrow::FixedSizeListType;
  using BuilderType = arrow::FixedSizeListBuilder;
};

template <typename T>
struct BuilderTraits<std::vector<T>> {
  using ArrowType = arrow::ListType;
  using BuilderType = arrow::ListBuilder;
};

template <typename T>
struct DirectInsertion {
  template <typename BUILDER>
  arrow::Status append(BUILDER& builder, T value)
  {
    return builder->Append(value);
  }

  template <typename BUILDER>
  arrow::Status flush(BUILDER&)
  {
    return arrow::Status::OK();
  }
};

template <typename T>
struct CachedInsertion {
  static constexpr int CHUNK_SIZE = 256;

  template <typename BUILDER>
  arrow::Status append(BUILDER& builder, T value)
  {
    cache[pos % CHUNK_SIZE] = value;
    ++pos;
    if (pos % CHUNK_SIZE == 0) {
      return builder->AppendValues(cache, CHUNK_SIZE, nullptr);
    }
    return arrow::Status::OK();
  }

  template <typename BUILDER>
  arrow::Status flush(BUILDER& builder)
  {
    if (pos % CHUNK_SIZE != 0) {
      return builder->AppendValues(cache, pos % CHUNK_SIZE, nullptr);
    }
    return arrow::Status::OK();
  }
  T cache[CHUNK_SIZE];
  int pos = 0;
};

template <typename T, template <typename U> typename InsertionPolicy>
struct BuilderHolder : InsertionPolicy<T> {
  using Policy = InsertionPolicy<T>;
  using ArrowType = typename detail::ConversionTraits<T>::ArrowType;
  using BuilderType = typename arrow::TypeTraits<ArrowType>::BuilderType;

  BuilderHolder(arrow::MemoryPool* pool)
    : builder{BuilderMaker<T>::make(pool)}
  {
  }

  std::unique_ptr<BuilderType> builder;
};

struct TableBuilderHelpers {

  template <typename... ARGS, size_t NCOLUMNS>
  static std::array<arrow::DataType, NCOLUMNS> makeArrowColumnTypes()
  {
    return {BuilderTraits<ARGS>::make_datatype()...};
  }

  template <typename... ARGS, size_t NCOLUMNS = sizeof...(ARGS)>
  static std::vector<std::shared_ptr<arrow::Field>> makeFields(std::array<char const*, NCOLUMNS> const& names)
  {
    char const* const* names_ptr = names.data();
    return {
      std::make_shared<arrow::Field>(*names_ptr++, BuilderMaker<ARGS>::make_datatype(), true, nullptr)...};
  }

  /// Invokes the append method for each entry in the tuple
  template <std::size_t... Is, typename HOLDERS, typename VALUES>
  static bool append(HOLDERS& holders, std::index_sequence<Is...>, VALUES&& values)
  {
    return (BuilderUtils::append(std::get<Is>(holders), std::get<Is>(values)).ok() && ...);
  }

  /// Invokes the UnsafeAppend method for each entry in the tuple
  /// For this to be used, one should make sure the number of entries
  /// is known a-priori.
  template <std::size_t... Is, typename HOLDERS, typename VALUES>
  static void unsafeAppend(HOLDERS& holders, std::index_sequence<Is...>, VALUES&& values)
  {
    (BuilderUtils::unsafeAppend(std::get<Is>(holders), std::get<Is>(values)), ...);
  }

  template <std::size_t... Is, typename HOLDERS, typename PTRS>
  static bool bulkAppend(HOLDERS& holders, size_t bulkSize, std::index_sequence<Is...>, PTRS ptrs)
  {
    return (BuilderUtils::bulkAppend(std::get<Is>(holders), bulkSize, std::get<Is>(ptrs)).ok() && ...);
  }

  /// Return true if all columns are done.
  template <std::size_t... Is, typename BUILDERS, typename INFOS>
  static bool bulkAppendChunked(BUILDERS& builders, std::index_sequence<Is...>, INFOS infos)
  {
    return (BuilderUtils::bulkAppendChunked(std::get<Is>(builders), std::get<Is>(infos)).ok() && ...);
  }

  /// Invokes the append method for each entry in the tuple
  template <typename HOLDERS, std::size_t... Is>
  static bool finalize(std::vector<std::shared_ptr<arrow::Array>>& arrays, HOLDERS& holders, std::index_sequence<Is...> seq)
  {
    return (finalize(arrays[Is], std::get<Is>(holders)) && ...);
  }

  template <typename HOLDER>
  static bool finalize(std::shared_ptr<arrow::Array>& array, HOLDER& holder)
  {
    return BuilderUtils::flush(holder).ok() && holder.builder->Finish(&array).ok();
  }

  template <typename HOLDERS, std::size_t... Is>
  static bool reserveAll(HOLDERS& holders, size_t s, std::index_sequence<Is...>)
  {
    return (std::get<Is>(holders).builder->Reserve(s).ok() && ...);
  }

  template <typename HOLDER>
  static HOLDER&& reserveAll(HOLDER&& holder, size_t s)
  {
    if (s != -1) {
      holder.builder->Reserve(s).ok();
    }
    return std::move(holder);
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

template <typename T>
struct HolderTrait {
  using Holder = BuilderHolder<T, DirectInsertion>;
};

template <>
struct HolderTrait<int8_t> {
  using Holder = BuilderHolder<int8_t, CachedInsertion>;
};

template <>
struct HolderTrait<uint8_t> {
  using Holder = BuilderHolder<uint8_t, CachedInsertion>;
};

template <>
struct HolderTrait<uint16_t> {
  using Holder = BuilderHolder<uint16_t, CachedInsertion>;
};

template <>
struct HolderTrait<int16_t> {
  using Holder = BuilderHolder<int16_t, CachedInsertion>;
};

template <>
struct HolderTrait<int> {
  using Holder = BuilderHolder<int, CachedInsertion>;
};

template <>
struct HolderTrait<float> {
  using Holder = BuilderHolder<float, CachedInsertion>;
};

template <>
struct HolderTrait<double> {
  using Holder = BuilderHolder<double, CachedInsertion>;
};

template <>
struct HolderTrait<unsigned int> {
  using Holder = BuilderHolder<unsigned int, CachedInsertion>;
};

template <>
struct HolderTrait<uint64_t> {
  using Holder = BuilderHolder<uint64_t, CachedInsertion>;
};

template <>
struct HolderTrait<int64_t> {
  using Holder = BuilderHolder<int64_t, CachedInsertion>;
};

/// Helper function to convert a brace-initialisable struct to
/// a tuple.
template <class T>
auto constexpr to_tuple(T&& object) noexcept
{
  using type = std::decay_t<T>;
  if constexpr (is_braces_constructible<type, any_type, any_type, any_type, any_type>{}) {
    auto&& [p0, p1, p2, p3] = object;
    return std::make_tuple(p0, p1, p2, p3);
  } else if constexpr (is_braces_constructible<type, any_type, any_type, any_type>{}) {
    auto&& [p0, p1, p2] = object;
    return std::make_tuple(p0, p1, p2);
  } else if constexpr (is_braces_constructible<type, any_type, any_type>{}) {
    auto&& [p0, p1] = object;
    return std::make_tuple(p0, p1);
  } else if constexpr (is_braces_constructible<type, any_type>{}) {
    auto&& [p0] = object;
    return std::make_tuple(p0);
  } else {
    return std::make_tuple();
  }
}

/// Helper class which creates a lambda suitable for building
/// an arrow table from a tuple. This can be used, for example
/// to build an arrow::Table from a TDataFrame.
class TableBuilder
{
  static void throwError(RuntimeErrorRef const& ref);

  template <typename... ARGS>
  using HoldersTuple = typename std::tuple<typename HolderTrait<ARGS>::Holder...>;

  /// Get the builders, assumning they were created with a given pack
  ///  of basic types
  template <typename... ARGS>
  auto getBuilders(o2::framework::pack<ARGS...>)
  {
    return (HoldersTuple<ARGS...>*)mHolders;
  }

  void validate() const;

  template <typename... ARGS, size_t I = sizeof...(ARGS)>
  auto makeBuilders(std::array<char const*, I> const& columnNames, size_t nRows)
  {
    mSchema = std::make_shared<arrow::Schema>(TableBuilderHelpers::makeFields<ARGS...>(columnNames));
    mHolders = new HoldersTuple<ARGS...>(TableBuilderHelpers::reserveAll(typename HolderTrait<ARGS>::Holder(mMemoryPool), nRows)...);
  }

  template <typename... ARGS>
  auto makeFinalizer()
  {
    mFinalizer = [](std::shared_ptr<arrow::Schema> schema, std::vector<std::shared_ptr<arrow::Array>>& arrays, void* holders) -> bool {
      return TableBuilderHelpers::finalize(arrays, *(HoldersTuple<ARGS...>*)holders, std::make_index_sequence<sizeof...(ARGS)>{});
    };
  }

 public:
  template <typename... ARGS>
  static constexpr int countColumns()
  {
    using args_pack_t = framework::pack<ARGS...>;
    if constexpr (sizeof...(ARGS) == 1 &&
                  is_bounded_array<pack_element_t<0, args_pack_t>>::value == false &&
                  std::is_arithmetic_v<pack_element_t<0, args_pack_t>> == false &&
                  framework::is_base_of_template_v<std::vector, pack_element_t<0, args_pack_t>> == false) {
      using objType_t = pack_element_t<0, framework::pack<ARGS...>>;
      using argsPack_t = decltype(tuple_to_pack(framework::to_tuple(std::declval<objType_t>())));
      return framework::pack_size(argsPack_t{});
    } else if constexpr (sizeof...(ARGS) == 1 &&
                         (is_bounded_array<pack_element_t<0, args_pack_t>>::value == true ||
                          framework::is_base_of_template_v<std::vector, pack_element_t<0, args_pack_t>> == true)) {
      using objType_t = pack_element_t<0, framework::pack<ARGS...>>;
      using argsPack_t = framework::pack<objType_t>;
      return framework::pack_size(argsPack_t{});
    } else if constexpr (sizeof...(ARGS) >= 1) {
      return sizeof...(ARGS);
    } else {
      static_assert(o2::framework::always_static_assert_v<ARGS...>, "Unmanaged case");
    }
  }
  void setLabel(const char* label);

  TableBuilder(arrow::MemoryPool* pool = arrow::default_memory_pool())
    : mHolders{nullptr},
      mMemoryPool{pool}
  {
  }

  /// Creates a lambda which is suitable to persist things
  /// in an arrow::Table
  template <typename... ARGS, size_t NCOLUMNS = countColumns<ARGS...>()>
  auto persist(std::array<char const*, NCOLUMNS> const& columnNames)
  {
    using args_pack_t = framework::pack<ARGS...>;
    if constexpr (sizeof...(ARGS) == 1 &&
                  is_bounded_array<pack_element_t<0, args_pack_t>>::value == false &&
                  std::is_arithmetic_v<pack_element_t<0, args_pack_t>> == false &&
                  framework::is_base_of_template_v<std::vector, pack_element_t<0, args_pack_t>> == false) {
      using objType_t = pack_element_t<0, framework::pack<ARGS...>>;
      using argsPack_t = decltype(tuple_to_pack(framework::to_tuple(std::declval<objType_t>())));
      auto persister = persistTuple(argsPack_t{}, columnNames);
      return [persister = persister](unsigned int slot, objType_t const& obj) -> void {
        auto t = to_tuple(obj);
        persister(slot, t);
      };
    } else if constexpr (sizeof...(ARGS) == 1 &&
                         (is_bounded_array<pack_element_t<0, args_pack_t>>::value == true ||
                          framework::is_base_of_template_v<std::vector, pack_element_t<0, args_pack_t>> == true)) {
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
  auto persistTuple(framework::pack<ARGS...>, std::array<char const*, sizeof...(ARGS)> const& columnNames)
  {
    constexpr int nColumns = sizeof...(ARGS);
    validate();
    mArrays.resize(nColumns);
    makeBuilders<ARGS...>(columnNames, 1000);
    makeFinalizer<ARGS...>();

    // Callback used to fill the builders
    using FillTuple = std::tuple<typename BuilderMaker<ARGS>::FillType...>;
    return [holders = mHolders](unsigned int slot, FillTuple const& t) -> void {
      auto status = TableBuilderHelpers::append(*(HoldersTuple<ARGS...>*)holders, std::index_sequence_for<ARGS...>{}, t);
      if (status == false) {
        throwError(runtime_error("Unable to append"));
      }
    };
  }

  // Same as above, but starting from a o2::soa::Table, which has all the
  // information already available.
  template <typename T>
  auto cursor()
  {
    return cursorHelper(typename T::table_t::persistent_columns_t{});
  }

  template <typename T, typename E>
  auto cursor()
  {
    return cursorHelper2<E>(typename T::table_t::persistent_columns_t{});
  }

  template <typename... ARGS, size_t NCOLUMNS = sizeof...(ARGS)>
  auto preallocatedPersist(std::array<char const*, NCOLUMNS> const& columnNames, int nRows)
  {
    constexpr size_t nColumns = NCOLUMNS;
    validate();
    mArrays.resize(nColumns);
    makeBuilders<ARGS...>(columnNames, nRows);
    makeFinalizer<ARGS...>();

    // Callback used to fill the builders
    return [holders = mHolders](unsigned int slot, typename BuilderMaker<ARGS>::FillType... args) -> void {
      TableBuilderHelpers::unsafeAppend(*(HoldersTuple<ARGS...>*)holders, std::index_sequence_for<ARGS...>{}, std::forward_as_tuple(args...));
    };
  }

  template <typename... ARGS, size_t NCOLUMNS = sizeof...(ARGS)>
  auto bulkPersist(std::array<char const*, NCOLUMNS> const& columnNames, size_t nRows)
  {
    validate();
    //  Should not be called more than once
    mArrays.resize(NCOLUMNS);
    makeBuilders<ARGS...>(columnNames, nRows);
    makeFinalizer<ARGS...>();

    return [holders = mHolders](unsigned int slot, size_t batchSize, typename BuilderMaker<ARGS>::FillType const*... args) -> void {
      TableBuilderHelpers::bulkAppend(*(HoldersTuple<ARGS...>*)holders, batchSize, std::index_sequence_for<ARGS...>{}, std::forward_as_tuple(args...));
    };
  }

  template <typename... ARGS, size_t NCOLUMNS = sizeof...(ARGS)>
  auto bulkPersistChunked(std::array<char const*, NCOLUMNS> const& columnNames, size_t nRows)
  {
    validate();
    mArrays.resize(NCOLUMNS);
    makeBuilders<ARGS...>(columnNames, nRows);
    makeFinalizer<ARGS...>();

    return [holders = mHolders](unsigned int slot, BulkInfo<typename BuilderMaker<ARGS>::STLValueType const*>... args) -> bool {
      return TableBuilderHelpers::bulkAppendChunked(*(HoldersTuple<ARGS...>*)holders, std::index_sequence_for<ARGS...>{}, std::forward_as_tuple(args...));
    };
  }

  /// Reserve method to expand the columns as needed.
  template <typename... ARGS>
  auto reserve(o2::framework::pack<ARGS...> pack, int s)
  {
    visitBuilders(pack, [s](auto& holder) { return holder.builder->Reserve(s).ok(); });
  }

  /// Invoke the appropriate visitor on the various builders
  template <typename... ARGS, typename V>
  auto visitBuilders(o2::framework::pack<ARGS...> pack, V&& visitor)
  {
    auto builders = getBuilders(pack);
    return std::apply(overloaded{
                        [visitor](typename HolderTrait<ARGS>::Holder&... args) { (visitor(args), ...); }},
                      *builders);
  }

  /// Actually creates the arrow::Table from the builders
  std::shared_ptr<arrow::Table> finalize();

 private:
  /// Helper which actually creates the insertion cursor. Notice that the
  /// template argument T is a o2::soa::Table which contains only the
  /// persistent columns.
  template <typename... Cs>
  auto cursorHelper(framework::pack<Cs...>)
  {
    return this->template persist<typename Cs::type...>({Cs::columnLabel()...});
  }

  template <typename E, typename... Cs>
  auto cursorHelper2(framework::pack<Cs...>)
  {
    return this->template persist<E>({Cs::columnLabel()...});
  }

  bool (*mFinalizer)(std::shared_ptr<arrow::Schema> schema, std::vector<std::shared_ptr<arrow::Array>>& arrays, void* holders);
  void* mHolders;
  arrow::MemoryPool* mMemoryPool;
  std::shared_ptr<arrow::Schema> mSchema;
  std::vector<std::shared_ptr<arrow::Array>> mArrays;
};

template <typename T>
auto makeEmptyTable(const char* name)
{
  TableBuilder b;
  [[maybe_unused]] auto writer = b.cursor<T>();
  b.setLabel(name);
  return b.finalize();
}

std::shared_ptr<arrow::Table> spawnerHelper(std::shared_ptr<arrow::Table> fullTable, std::shared_ptr<arrow::Schema> newSchema, size_t nColumns,
                                            expressions::Projector* projectors, std::vector<std::shared_ptr<arrow::Field>> const& fields, const char* name);

/// Expression-based column generator to materialize columns
template <typename... C>
auto spawner(framework::pack<C...> columns, std::vector<std::shared_ptr<arrow::Table>>&& tables, const char* name)
{
  auto fullTable = soa::ArrowHelpers::joinTables(std::move(tables));
  if (fullTable->num_rows() == 0) {
    return makeEmptyTable<soa::Table<C...>>(name);
  }
  static auto fields = o2::soa::createFieldsFromColumns(columns);
  static auto new_schema = std::make_shared<arrow::Schema>(fields);
  std::array<expressions::Projector, sizeof...(C)> projectors{{std::move(C::Projector())...}};
  return spawnerHelper(fullTable, new_schema, sizeof...(C), projectors.data(), fields, name);
}

/// Helper to get a tuple tail
template <typename Head, typename... Tail>
std::tuple<Tail...> tuple_tail(std::tuple<Head, Tail...>& t)
{
  return apply([](auto const&, auto&... tail) { return std::tie(tail...); }, t);
}

/// Helpers to get type pack from tuple
template <typename... T>
constexpr auto pack_from_tuple(std::tuple<T...> const&)
{
  return framework::pack<T...>{};
}

/// Binary search for an index column
template <typename Key, typename T>
void lowerBound(int32_t value, T& start)
{
  static_assert(soa::is_soa_iterator_v<T>, "Argument needs to be a Table::iterator");
  int step;
  auto count = start.size() - start.globalIndex();

  while (count > 0) {
    step = count / 2;
    start.moveByIndex(step);
    if (start.template getId<Key>() <= value) {
      count -= step + 1;
    } else {
      start.moveByIndex(-step);
      count = step;
    }
  }
}

template <typename... T>
using iterator_tuple_t = std::tuple<typename T::iterator...>;
} // namespace o2::framework
#endif // FRAMEWORK_TABLEBUILDER_H
