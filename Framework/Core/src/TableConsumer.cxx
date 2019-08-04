// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "Framework/TableConsumer.h"

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
#include <arrow/io/memory.h>
#include <arrow/ipc/reader.h>
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

using namespace arrow;

namespace o2
{
namespace framework
{

TableConsumer::TableConsumer(const uint8_t* data, int64_t size)
  : mBuffer{std::make_shared<Buffer>(data, size)}
{
}

std::shared_ptr<arrow::Table>
  TableConsumer::asArrowTable()
{
  std::shared_ptr<Table> inTable;
  // In case the buffer is empty, we cannot determine the schema
  // and therefore return an empty table;
  if (mBuffer->size() == 0) {
    std::vector<std::shared_ptr<arrow::Field>> dummyFields{};
    std::vector<std::shared_ptr<arrow::Column>> dummyColumns{};
    auto dummySchema = std::make_shared<arrow::Schema>(dummyFields);
    return arrow::Table::Make(dummySchema, dummyColumns);
  }

  /// Reading back from the stream
  std::shared_ptr<io::InputStream> bufferReader = std::make_shared<io::BufferReader>(mBuffer);
  std::shared_ptr<ipc::RecordBatchReader> batchReader;

  auto readerOk = ipc::RecordBatchStreamReader::Open(bufferReader, &batchReader);
  std::vector<std::shared_ptr<RecordBatch>> batches;
  while (true) {
    std::shared_ptr<RecordBatch> batch;
    auto next = batchReader->ReadNext(&batch);
    if (batch.get() == nullptr) {
      break;
    }
    batches.push_back(batch);
  }

  auto inStatus = Table::FromRecordBatches(batches, &inTable);

  return inTable;
}

} // namespace framework
} // namespace o2
