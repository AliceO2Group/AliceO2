// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include "Framework/Kernels.h"
#include "Framework/ArrowTypes.h"

#include "ArrowDebugHelpers.h"

#include <arrow/builder.h>
#include <arrow/status.h>
#include <arrow/type.h>
#include <arrow/util/variant.h>
#include <memory>
#include <iostream>

using namespace arrow;
using namespace arrow::compute;

namespace o2
{
namespace framework
{

HashByColumnKernel::HashByColumnKernel(HashByColumnOptions options)
  : mOptions{options}
{
}

Status HashByColumnKernel::Call(FunctionContext* ctx, Datum const& inputTable, Datum* hashes)
{
  if (inputTable.kind() == Datum::TABLE) {
    auto table = arrow::util::get<std::shared_ptr<arrow::Table>>(inputTable.value);
    auto columnIndex = table->schema()->GetFieldIndex(mOptions.columnName);
    auto chunkedArray = table->column(columnIndex);
    *hashes = std::move(chunkedArray);
    return arrow::Status::OK();
  }
  return Status::Invalid("Input Datum was not a table");
}

} // namespace framework
} // namespace o2
