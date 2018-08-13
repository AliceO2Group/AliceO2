// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "Framework/TableBuilder.h"
#include <memory>
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wshadow"
#endif
#include <arrow/builder.h>
#include <arrow/memory_pool.h>
#include <arrow/record_batch.h>
#include <arrow/table.h>
#include <arrow/type_traits.h>
#include <arrow/status.h>
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

using namespace arrow;

namespace
{

// FIXME: Dummy schema, to compile.
template <typename TYPE, typename C_TYPE>
void ArrayFromVector(const std::vector<C_TYPE>& values, std::shared_ptr<arrow::Array>* out)
{
  typename arrow::TypeTraits<TYPE>::BuilderType builder;
  for (size_t i = 0; i < values.size(); ++i) {
    auto status = builder.Append(values[i]);
    assert(status.ok());
  }
  auto status = builder.Finish(out);
  assert(status.ok());
}

} // namespace

namespace o2
{
namespace framework
{

std::shared_ptr<arrow::Table>
  TableBuilder::finalize()
{
  mFinalizer();
  std::vector<std::shared_ptr<arrow::Column>> columns;
  columns.reserve(mArrays.size());
  for (size_t i = 0; i < mSchema->num_fields(); ++i) {
    auto column = std::make_shared<arrow::Column>(mSchema->field(i), mArrays[i]);
    columns.emplace_back(column);
  }
  auto table_ = arrow::Table::Make(mSchema, columns);
  return table_;
}

} // namespace framework
} // namespace o2
