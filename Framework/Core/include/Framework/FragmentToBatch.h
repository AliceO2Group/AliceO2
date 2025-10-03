// Copyright 2019-2025 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef O2_FRAMEWORK_FRAGMENT_TO_BATCH_H_
#define O2_FRAMEWORK_FRAGMENT_TO_BATCH_H_

#include <arrow/buffer.h>
#include <arrow/io/interfaces.h>
#include <arrow/record_batch.h>
#include <arrow/dataset/file_base.h>
#include <memory>

// =============================================================================
namespace o2::framework
{
class FragmentToBatch
{
 public:
  // The function to be used to create the required stream.
  using StreamerCreator = std::function<std::shared_ptr<arrow::io::OutputStream>(std::shared_ptr<arrow::dataset::FileFragment>, const std::shared_ptr<arrow::ResizableBuffer>& buffer)>;

  FragmentToBatch(StreamerCreator, std::shared_ptr<arrow::dataset::FileFragment>, arrow::MemoryPool* pool = arrow::default_memory_pool());
  void setLabel(const char* label);
  void fill(std::shared_ptr<arrow::Schema> dataSetSchema, std::shared_ptr<arrow::dataset::FileFormat>);
  std::shared_ptr<arrow::RecordBatch> finalize();

  std::shared_ptr<arrow::io::OutputStream> streamer(std::shared_ptr<arrow::ResizableBuffer> buffer)
  {
    return mCreator(mFragment, buffer);
  }

 private:
  std::shared_ptr<arrow::dataset::FileFragment> mFragment;
  arrow::MemoryPool* mArrowMemoryPool = nullptr;
  std::string mTableLabel;
  std::shared_ptr<arrow::RecordBatch> mRecordBatch;
  StreamerCreator mCreator;
};

// -----------------------------------------------------------------------------
} // namespace o2::framework

// =============================================================================
#endif // O2_FRAMEWORK_FRAGMENT_TO_BATCH_H_
