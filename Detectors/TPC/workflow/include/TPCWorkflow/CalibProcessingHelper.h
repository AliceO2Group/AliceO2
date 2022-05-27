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

#ifndef O2_TPC_CalibProcessingHelper_H
#define O2_TPC_CalibProcessingHelper_H

#include <memory>

#include "Framework/InputRecord.h"

using o2::tpc::rawreader::RawReaderCRU;

namespace o2
{
namespace tpc
{
namespace rawreader
{
class RawReaderCRU;
}
namespace calib_processing_helper
{

uint64_t processRawData(o2::framework::InputRecord& inputs, std::unique_ptr<RawReaderCRU>& reader, bool useOldSubspec = false, const std::vector<int>& sectors = {}, size_t* nerrors = nullptr, uint32_t syncOffsetReference = 144);
} // namespace calib_processing_helper
} // namespace tpc
} // namespace o2

#endif
