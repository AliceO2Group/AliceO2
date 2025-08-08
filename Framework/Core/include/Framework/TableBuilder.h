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
#include <concepts>

namespace arrow
{
class ArrayBuilder;
class Table;
class Array;
} // namespace arrow

extern template class arrow::NumericBuilder<arrow::UInt8Type>;
extern template class arrow::NumericBuilder<arrow::UInt32Type>;
extern template class arrow::NumericBuilder<arrow::FloatType>;
extern template class arrow::NumericBuilder<arrow::Int32Type>;
extern template class arrow::NumericBuilder<arrow::Int8Type>;

namespace o2::framework
{
namespace detail
{
/// FIXME: adapt type conversion to new arrow
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

template <typename T, int N>
struct ConversionTraits<std::array<T, N>> {
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
O2_ARROW_STL_CONVERSION(std::span<std::byte>, BinaryViewType)
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

template <>
struct BuilderMaker<std::span<std::byte>> {
  using FillType = std::span<std::byte>;
  using STLValueType = std::span<std::byte>;
  using ArrowType = typename detail::ConversionTraits<std::span<std::byte>>::ArrowType;
  using BuilderType = typename arrow::TypeTraits<ArrowType>::BuilderType;

  static std::unique_ptr<BuilderType> make(arrow::MemoryPool* pool)
  {
    return std::make_unique<BuilderType>(pool);
  }

  static std::shared_ptr<arrow::DataType> make_datatype()
  {
    return arrow::TypeTraits<ArrowType>::type_singleton();
  }

  static arrow::Status append(BuilderType& builder, std::span<std::byte> value)
  {
    return builder.Append((char*)value.data(), (int64_t)value.size());
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

template <typename T, int N>
struct BuilderMaker<std::array<T, N>> {
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
    requires std::same_as<std::span<std::byte>, T>
  arrow::Status append(BUILDER& builder, T value)
  {
    return builder->Append((char*)value.data(), (int64_t)value.size());
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

template <typename T>
struct InsertionTrait {
  static consteval DirectInsertion<T> policy();
  using Policy = decltype(policy());
};

template <size_t I, typename T>
struct BuilderHolder : InsertionTrait<T>::Policy {
  static constexpr size_t index = I;
  using Policy = typename InsertionTrait<T>::Policy;
  using ArrowType = typename detail::ConversionTraits<T>::ArrowType;
  using BuilderType = typename arrow::TypeTraits<ArrowType>::BuilderType;

  BuilderHolder(arrow::MemoryPool* pool, size_t nRows = 0)
    : builder{BuilderMaker<T>::make(pool)}
  {
    if (nRows > 0) {
      auto s = builder->Reserve(nRows);
      if (!s.ok()) {
        throw runtime_error_f("Unable to reserve %ll rows", nRows);
      }
    }
  }

  std::unique_ptr<BuilderType> builder;
};

struct TableBuilderHelpers {
  template <typename... ARGS, size_t NCOLUMNS>
  static std::array<arrow::DataType, NCOLUMNS> makeArrowColumnTypes()
  {
    return {BuilderTraits<ARGS>::make_datatype()...};
  }

  /// Invokes the append method for each entry in the tuple
  template <typename... Ts, typename VALUES>
  static bool append(std::tuple<Ts...>& holders, VALUES&& values)
  {
    return (BuilderUtils::append(std::get<Ts::index>(holders), std::get<Ts::index>(values)).ok() && ...);
  }

  /// Invokes the UnsafeAppend method for each entry in the tuple
  /// For this to be used, one should make sure the number of entries
  /// is known a-priori.
  template <typename... Ts, typename VALUES>
  static void unsafeAppend(std::tuple<Ts...>& holders, VALUES&& values)
  {
    (BuilderUtils::unsafeAppend(std::get<Ts::index>(holders), std::get<Ts::index>(values)), ...);
  }

  /// Invokes the append method for each entry in the tuple
  template <typename... Ts>
  static bool finalize(std::vector<std::shared_ptr<arrow::Array>>& arrays, std::tuple<Ts...>& holders)
  {
    return (finalize(arrays[Ts::index], std::get<Ts::index>(holders)) && ...);
  }

  template <typename HOLDER>
  static bool finalize(std::shared_ptr<arrow::Array>& array, HOLDER& holder)
  {
    return BuilderUtils::flush(holder).ok() && holder.builder->Finish(&array).ok();
  }
};

template <typename... ARGS>
constexpr auto tuple_to_pack(std::tuple<ARGS...>&&)
{
  return framework::pack<ARGS...>{};
}

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

template <typename... ARGS>
constexpr auto makeHolderTypes()
{
  return []<std::size_t... Is>(std::index_sequence<Is...>) {
    return std::tuple(BuilderHolder<Is, ARGS>(arrow::default_memory_pool())...);
  }(std::make_index_sequence<sizeof...(ARGS)>{});
}

template <typename... ARGS>
auto makeHolders(arrow::MemoryPool* pool, size_t nRows)
{
  return [pool, nRows]<std::size_t... Is>(std::index_sequence<Is...>) {
    return new std::tuple(BuilderHolder<Is, ARGS>(pool, nRows)...);
  }(std::make_index_sequence<sizeof...(ARGS)>{});
}

template <typename... ARGS>
using IndexedHoldersTuple = decltype(makeHolderTypes<ARGS...>());

template <typename T>
concept ShouldNotDeconstruct = std::is_bounded_array_v<T> || std::is_arithmetic_v<T> || framework::is_base_of_template_v<std::vector, T> || std::same_as<std::span<std::byte>, T>;

/// Helper class which creates a lambda suitable for building
/// an arrow table from a tuple. This can be used, for example
/// to build an arrow::Table from a TDataFrame.
class TableBuilder
{
  static void throwError(RuntimeErrorRef const& ref);

  template <typename... ARGS>
  using HoldersTuple = typename std::tuple<BuilderHolder<0, ARGS>...>;

  template <typename... ARGS>
  using HoldersTupleIndexed = decltype(makeHolderTypes<ARGS...>());

  /// Get the builders, assumning they were created with a given pack
  ///  of basic types
  template <typename... ARGS>
  auto getBuilders(o2::framework::pack<ARGS...>)
  {
    return (HoldersTupleIndexed<ARGS...>*)mHolders;
  }

  void validate() const;

  template <typename... ARGS, size_t I = sizeof...(ARGS)>
  auto makeBuilders(std::array<char const*, I> const& columnNames, size_t nRows)
  {
    char const* const* names_ptr = columnNames.data();
    mSchema = std::make_shared<arrow::Schema>(
      std::vector<std::shared_ptr<arrow::Field>>({std::make_shared<arrow::Field>(*names_ptr++, BuilderMaker<ARGS>::make_datatype(), true, nullptr)...}));

    mHolders = makeHolders<ARGS...>(mMemoryPool, nRows);
    mFinalizer = [](std::vector<std::shared_ptr<arrow::Array>>& arrays, void* holders) -> bool {
      return TableBuilderHelpers::finalize(arrays, *(HoldersTupleIndexed<ARGS...>*)holders);
    };
    mDestructor = [](void* holders) mutable -> void {
      delete (HoldersTupleIndexed<ARGS...>*)holders;
    };
  }

 public:
  template <typename ARG0, typename... ARGS>
    requires(sizeof...(ARGS) == 0) && (!ShouldNotDeconstruct<ARG0>)
  static constexpr int countColumns()
  {
    using argsPack_t = decltype(tuple_to_pack(framework::to_tuple(std::declval<ARG0>())));
    return framework::pack_size(argsPack_t{});
  }

  template <typename ARG0, typename... ARGS>
    requires(sizeof...(ARGS) > 0) || ShouldNotDeconstruct<ARG0>
  static constexpr int countColumns()
  {
    return 1 + sizeof...(ARGS);
  }

  void setLabel(const char* label);

  TableBuilder(arrow::MemoryPool* pool = arrow::default_memory_pool())
    : mHolders{nullptr},
      mMemoryPool{pool}
  {
  }

  ~TableBuilder()
  {
    mDestructor(mHolders);
  }

  /// Creates a lambda which is suitable to persist things
  /// in an arrow::Table
  template <typename ARG0, typename... ARGS>
    requires(sizeof...(ARGS) > 0) || ShouldNotDeconstruct<ARG0>
  auto persist(std::array<char const*, sizeof...(ARGS) + 1> const& columnNames)
  {
    auto persister = persistTuple(framework::pack<ARG0, ARGS...>{}, columnNames);
    // Callback used to fill the builders
    return [persister = persister](unsigned int slot, typename BuilderMaker<ARG0>::FillType const& arg, typename BuilderMaker<ARGS>::FillType... args) -> void {
      persister(slot, std::forward_as_tuple(arg, args...));
    };
  }

  // Special case for a single parameter to handle the serialization of struct
  // which can be decomposed
  template <typename ARG0, typename... ARGS>
    requires(sizeof...(ARGS) == 0) && (!ShouldNotDeconstruct<ARG0>)
  auto persist(std::array<char const*, countColumns<ARG0, ARGS...>()> const& columnNames)
  {
    using argsPack_t = decltype(tuple_to_pack(framework::to_tuple(std::declval<ARG0>())));
    auto persister = persistTuple(argsPack_t{}, columnNames);
    return [persister = persister](unsigned int slot, ARG0 const& obj) -> void {
      auto t = to_tuple(obj);
      persister(slot, t);
    };
  }

  /// Same a the above, but use a tuple to persist stuff.
  template <typename... ARGS>
  auto persistTuple(framework::pack<ARGS...>, std::array<char const*, sizeof...(ARGS)> const& columnNames)
  {
    constexpr int nColumns = sizeof...(ARGS);
    validate();
    mArrays.resize(nColumns);
    makeBuilders<ARGS...>(columnNames, 10);

    // Callback used to fill the builders
    using FillTuple = std::tuple<typename BuilderMaker<ARGS>::FillType...>;
    return [holders = mHolders](unsigned int /*slot*/, FillTuple const& t) -> void {
      auto status = TableBuilderHelpers::append(*(HoldersTupleIndexed<ARGS...>*)holders, t);
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
    return [this]<typename... Cs>(pack<Cs...>) {
      return this->template persist<typename Cs::type...>({Cs::columnLabel()...});
    }(typename T::table_t::persistent_columns_t{});
  }

  template <typename... Cs>
  auto cursor(framework::pack<Cs...>)
  {
    return this->template persist<typename Cs::type...>({Cs::columnLabel()...});
  }

  template <typename T, typename E>
  auto cursor()
  {
    return [this]<typename... Cs>(pack<Cs...>) {
      return this->template persist<E>({Cs::columnLabel()...});
    }(typename T::table_t::persistent_columns_t{});
  }

  /// Reserve method to expand the columns as needed.
  template <typename... Ts>
  auto reserveArrays(std::tuple<Ts...>& holders, int s)
  {
    return (std::get<Ts::index>(holders).builder->Reserve(s).ok() && ...);
  }

  template <typename... ARGS>
  auto reserve(o2::framework::pack<ARGS...>&&, int s)
  {
    reserveArrays(*(HoldersTupleIndexed<ARGS...>*)mHolders, s);
  }

  /// Actually creates the arrow::Table from the builders
  void extracted(bool& status);
  std::shared_ptr<arrow::Table> finalize();

 private:
  bool (*mFinalizer)(std::vector<std::shared_ptr<arrow::Array>>& arrays, void* holders);
  void (*mDestructor)(void* holders);
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

template <soa::TableRef R>
auto makeEmptyTable()
{
  TableBuilder b;
  [[maybe_unused]] auto writer = b.cursor(typename aod::MetadataTrait<aod::Hash<R.desc_hash>>::metadata::persistent_columns_t{});
  b.setLabel(aod::label<R>());
  return b.finalize();
}

template <typename... Cs>
auto makeEmptyTable(const char* name, framework::pack<Cs...> p)
{
  TableBuilder b;
  [[maybe_unused]] auto writer = b.cursor(p);
  b.setLabel(name);
  return b.finalize();
}

std::shared_ptr<arrow::Table> spawnerHelper(std::shared_ptr<arrow::Table> const& fullTable, std::shared_ptr<arrow::Schema> newSchema, size_t nColumns,
                                            expressions::Projector* projectors, const char* name, std::shared_ptr<gandiva::Projector>& projector);

/// Expression-based column generator to materialize columns
template <aod::is_aod_hash D>
  requires(soa::has_configurable_extension<typename o2::aod::MetadataTrait<D>::metadata>)
auto spawner(std::shared_ptr<arrow::Table> const& fullTable, const char* name, o2::framework::expressions::Projector* projectors, std::shared_ptr<gandiva::Projector>& projector, std::shared_ptr<arrow::Schema> const& schema)
{
  using placeholders_pack_t = typename o2::aod::MetadataTrait<D>::metadata::placeholders_pack_t;
  if (fullTable->num_rows() == 0) {
    return makeEmptyTable(name, placeholders_pack_t{});
  }
  return spawnerHelper(fullTable, schema, framework::pack_size(placeholders_pack_t{}), projectors, name, projector);
}

template <aod::is_aod_hash D>
  requires(soa::has_configurable_extension<typename o2::aod::MetadataTrait<D>::metadata>)
auto spawner(std::vector<std::shared_ptr<arrow::Table>>&& tables, const char* name, o2::framework::expressions::Projector* projectors, std::shared_ptr<gandiva::Projector>& projector, std::shared_ptr<arrow::Schema> const& schema)
{
  auto fullTable = soa::ArrowHelpers::joinTables(std::move(tables), std::span{o2::aod::MetadataTrait<D>::metadata::base_table_t::originalLabels});
  return spawner<D>(fullTable, name, projectors, projector, schema);
}

template <aod::is_aod_hash D>
  requires(soa::has_extension<typename o2::aod::MetadataTrait<D>::metadata> && !soa::has_configurable_extension<typename o2::aod::MetadataTrait<D>::metadata>)
auto spawner(std::shared_ptr<arrow::Table> const& fullTable, const char* name, expressions::Projector* projectors, std::shared_ptr<gandiva::Projector>& projector, std::shared_ptr<arrow::Schema> const& schema)
{
  using expression_pack_t = typename o2::aod::MetadataTrait<D>::metadata::expression_pack_t;
  if (fullTable->num_rows() == 0) {
    return makeEmptyTable(name, expression_pack_t{});
  }
  return spawnerHelper(fullTable, schema, framework::pack_size(expression_pack_t{}), projectors, name, projector);
}

template <aod::is_aod_hash D>
  requires(soa::has_extension<typename o2::aod::MetadataTrait<D>::metadata> && !soa::has_configurable_extension<typename o2::aod::MetadataTrait<D>::metadata>)
auto spawner(std::vector<std::shared_ptr<arrow::Table>>&& tables, const char* name, expressions::Projector* projectors, std::shared_ptr<gandiva::Projector>& projector, std::shared_ptr<arrow::Schema> const& schema)
{
  auto fullTable = soa::ArrowHelpers::joinTables(std::move(tables), std::span{o2::aod::MetadataTrait<D>::metadata::base_table_t::originalLabels});
  return spawner<D>(fullTable, name, projectors, projector, schema);
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

template <typename... T>
using iterator_tuple_t = std::tuple<typename T::iterator...>;
} // namespace o2::framework
#endif // FRAMEWORK_TABLEBUILDER_H
