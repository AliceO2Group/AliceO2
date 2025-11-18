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

#ifndef O2_FRAMEWORK_ARROWTYPES_H
#define O2_FRAMEWORK_ARROWTYPES_H
#include "Framework/Traits.h"
#include "arrow/type_fwd.h"
#include <span>

namespace o2::soa
{
template <typename T>
struct arrow_array_for {
};
template <>
struct arrow_array_for<bool> {
  using type = arrow::BooleanArray;
};
template <>
struct arrow_array_for<int8_t> {
  using type = arrow::Int8Array;
};
template <>
struct arrow_array_for<uint8_t> {
  using type = arrow::UInt8Array;
};
template <>
struct arrow_array_for<int16_t> {
  using type = arrow::Int16Array;
};
template <>
struct arrow_array_for<uint16_t> {
  using type = arrow::UInt16Array;
};
template <>
struct arrow_array_for<int32_t> {
  using type = arrow::Int32Array;
};
template <>
struct arrow_array_for<int64_t> {
  using type = arrow::Int64Array;
};
template <>
struct arrow_array_for<uint32_t> {
  using type = arrow::UInt32Array;
};
template <>
struct arrow_array_for<uint64_t> {
  using type = arrow::UInt64Array;
};
template <>
struct arrow_array_for<float> {
  using type = arrow::FloatArray;
};
template <>
struct arrow_array_for<double> {
  using type = arrow::DoubleArray;
};
template <>
struct arrow_array_for<std::span<std::byte>> {
  using type = arrow::BinaryViewArray;
};
template <int N>
struct arrow_array_for<float[N]> {
  using type = arrow::FixedSizeListArray;
  using value_type = float;
};
template <int N>
struct arrow_array_for<int[N]> {
  using type = arrow::FixedSizeListArray;
  using value_type = int;
};
template <int N>
struct arrow_array_for<short[N]> {
  using type = arrow::FixedSizeListArray;
  using value_type = short;
};
template <int N>
struct arrow_array_for<double[N]> {
  using type = arrow::FixedSizeListArray;
  using value_type = double;
};
template <int N>
struct arrow_array_for<int8_t[N]> {
  using type = arrow::FixedSizeListArray;
  using value_type = int8_t;
};

#define ARROW_VECTOR_FOR(_type_)                \
  template <>                                   \
  struct arrow_array_for<std::vector<_type_>> { \
    using type = arrow::ListArray;              \
    using value_type = _type_;                  \
  };

ARROW_VECTOR_FOR(uint8_t);
ARROW_VECTOR_FOR(uint16_t);
ARROW_VECTOR_FOR(uint32_t);
ARROW_VECTOR_FOR(uint64_t);

ARROW_VECTOR_FOR(int8_t);
ARROW_VECTOR_FOR(int16_t);
ARROW_VECTOR_FOR(int32_t);
ARROW_VECTOR_FOR(int64_t);

ARROW_VECTOR_FOR(float);
ARROW_VECTOR_FOR(double);

template <typename T>
using arrow_array_for_t = typename arrow_array_for<T>::type;
template <typename T>
using value_for_t = typename arrow_array_for<T>::value_type;

template <class Array>
using array_element_t = std::decay_t<decltype(std::declval<Array>()[0])>;

template <typename T>
std::shared_ptr<arrow::DataType> asArrowDataType(int list_size = 1)
{
  auto typeGenerator = [](std::shared_ptr<arrow::DataType> const& type, int list_size) -> std::shared_ptr<arrow::DataType> {
    switch (list_size) {
      case -1:
        return arrow::list(type);
      case 1:
        return std::move(type);
      default:
        return arrow::fixed_size_list(type, list_size);
    }
  };

  if constexpr (std::is_arithmetic_v<T>) {
    if constexpr (std::same_as<T, bool>) {
      return typeGenerator(arrow::boolean(), list_size);
    } else if constexpr (std::same_as<T, uint8_t>) {
      return typeGenerator(arrow::uint8(), list_size);
    } else if constexpr (std::same_as<T, uint16_t>) {
      return typeGenerator(arrow::uint16(), list_size);
    } else if constexpr (std::same_as<T, uint32_t>) {
      return typeGenerator(arrow::uint32(), list_size);
    } else if constexpr (std::same_as<T, uint64_t>) {
      return typeGenerator(arrow::uint64(), list_size);
    } else if constexpr (std::same_as<T, int8_t>) {
      return typeGenerator(arrow::int8(), list_size);
    } else if constexpr (std::same_as<T, int16_t>) {
      return typeGenerator(arrow::int16(), list_size);
    } else if constexpr (std::same_as<T, int32_t>) {
      return typeGenerator(arrow::int32(), list_size);
    } else if constexpr (std::same_as<T, int64_t>) {
      return typeGenerator(arrow::int64(), list_size);
    } else if constexpr (std::same_as<T, float>) {
      return typeGenerator(arrow::float32(), list_size);
    } else if constexpr (std::same_as<T, double>) {
      return typeGenerator(arrow::float64(), list_size);
    }
  } else if constexpr (std::is_bounded_array_v<T>) {
    return asArrowDataType<array_element_t<T>>(std::extent_v<T>);
  } else if constexpr (o2::framework::is_specialization_v<T, std::vector>) {
    return asArrowDataType<typename T::value_type>(-1);
  }
  return nullptr;
}
} // namespace o2::soa
#endif // O2_FRAMEWORK_ARROWTYPES_H
