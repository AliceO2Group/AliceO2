// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
  using elemtype = bool;
};
template <>
struct arrow_array_for<int8_t> {
  using type = arrow::Int8Array;
  using elemtype = int8_t;
};
template <>
struct arrow_array_for<uint8_t> {
  using type = arrow::UInt8Array;
  using elemtype = uint8_t;
};
template <>
struct arrow_array_for<int16_t> {
  using type = arrow::Int16Array;
  using elemtype = int16_t;
};
template <>
struct arrow_array_for<uint16_t> {
  using type = arrow::UInt16Array;
  using elemtype = uint16_t;
};
template <>
struct arrow_array_for<int32_t> {
  using type = arrow::Int32Array;
  using elemtype = int32_t;
};
template <>
struct arrow_array_for<int64_t> {
  using type = arrow::Int64Array;
  using elemtype = int64_t;
};
template <>
struct arrow_array_for<uint32_t> {
  using type = arrow::UInt32Array;
  using elemtype = uint32_t;
};
template <>
struct arrow_array_for<uint64_t> {
  using type = arrow::UInt64Array;
  using elemtype = uint64_t;
};
template <>
struct arrow_array_for<float> {
  using type = arrow::FloatArray;
  using elemtype = float;
};
template <>
struct arrow_array_for<double> {
  using type = arrow::DoubleArray;
  using elemtype = double;
};
template <int N>
struct arrow_array_for<float[N]> {
  using type = arrow::FixedSizeListArray;
  using elemtype = float;
};
template <int N>
struct arrow_array_for<int[N]> {
  using type = arrow::FixedSizeListArray;
  using elemtype = int;
};
template <int N>
struct arrow_array_for<double[N]> {
  using type = arrow::FixedSizeListArray;
  using elemtype = double;
};

template <typename T>
using arrow_array_for_t = typename arrow_array_for<T>::type;
template <typename T>
using element_for_t = typename arrow_array_for<T>::elemtype;
} // namespace o2::soa
#endif // O2_FRAMEWORK_ARROWTYPES_H
