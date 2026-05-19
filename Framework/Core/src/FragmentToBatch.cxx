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
#include "Framework/FragmentToBatch.h"
#include "Framework/Logger.h"
#include "Framework/Signpost.h"

#include <arrow/dataset/file_base.h>
#include <arrow/record_batch.h>
#include <arrow/type.h>
#include <arrow/util/key_value_metadata.h>
#include <TBufferFile.h>

#include <memory>
#include <utility>

O2_DECLARE_DYNAMIC_LOG(tabletree_helpers);

namespace o2::framework
{

FragmentToBatch::FragmentToBatch(StreamerCreator creator, std::shared_ptr<arrow::dataset::FileFragment> fragment, arrow::MemoryPool* pool)
  : mFragment{std::move(fragment)},
    mArrowMemoryPool{pool},
    mCreator{std::move(creator)}
{
}

void FragmentToBatch::setLabel(const char* label)
{
  mTableLabel = label;
}

void FragmentToBatch::fill(std::shared_ptr<arrow::Schema> schema, std::shared_ptr<arrow::dataset::FileFormat> format)
{
  auto options = std::make_shared<arrow::dataset::ScanOptions>();
  options->dataset_schema = schema;
  auto scanner = format->ScanBatchesAsync(options, mFragment);
  auto batch = (*scanner)();
  auto result = batch.result();
  if (!result.ok()) {
    throw std::runtime_error("FragmentToBatch::fill: scan failed: " + result.status().ToString());
  }
  mRecordBatch = *result;
  if (!mRecordBatch) {
    throw std::runtime_error("FragmentToBatch::fill: scan returned null RecordBatch");
  }
  // Notice that up to here the buffer was not yet filled.
}

std::shared_ptr<arrow::RecordBatch> FragmentToBatch::finalize()
{
  return mRecordBatch;
}

} // namespace o2::framework
