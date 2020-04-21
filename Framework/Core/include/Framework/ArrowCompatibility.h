// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_FRAMEWORK_ARROW_COMPATIBILITY_H_
#define O2_FRAMEWORK_ARROW_COMPATIBILITY_H_

#include <arrow/table.h>

namespace o2::framework
{

// Compatibility helpers between Arrow 0.14 and 0.15
template <typename T>
static inline std::shared_ptr<arrow::ChunkedArray> getBackendColumnData(T p)
{
  return p->data();
}

template <>
inline std::shared_ptr<arrow::ChunkedArray>
  getBackendColumnData<std::shared_ptr<arrow::ChunkedArray>>(std::shared_ptr<arrow::ChunkedArray> p)
{
  return p;
}

template <typename T>
static inline arrow::ChunkedArray const*
  getBackendColumnPtrData(T const* p)
{
  return p->data().get();
}

template <>
inline arrow::ChunkedArray const*
  getBackendColumnPtrData<arrow::ChunkedArray>(arrow::ChunkedArray const* p)
{
  return p;
}

using BackendColumnType = typename decltype(std::declval<arrow::Table>().column(0))::element_type;

template <typename T>
static inline std::shared_ptr<T> makeBackendColumn(std::shared_ptr<arrow::Field> field, arrow::ArrayVector array)
{
  return std::make_shared<T>(field, array);
}

template <typename T>
static inline std::shared_ptr<T> makeBackendColumn(std::shared_ptr<arrow::Field> field, std::shared_ptr<arrow::Array> array)
{
  return std::make_shared<T>(field, array);
}

template <>
inline std::shared_ptr<arrow::ChunkedArray> makeBackendColumn<arrow::ChunkedArray>(std::shared_ptr<arrow::Field>, arrow::ArrayVector array)
{
  return std::make_shared<arrow::ChunkedArray>(array);
}

template <>
inline std::shared_ptr<arrow::ChunkedArray> makeBackendColumn<arrow::ChunkedArray>(std::shared_ptr<arrow::Field>, std::shared_ptr<arrow::Array> array)
{
  return std::make_shared<arrow::ChunkedArray>(array);
}

} // namespace o2::framework

#endif
