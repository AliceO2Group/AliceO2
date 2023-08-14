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
#include "arrow/type_fwd.h"

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
} // namespace o2::soa
#endif // O2_FRAMEWORK_ARROWTYPES_H
