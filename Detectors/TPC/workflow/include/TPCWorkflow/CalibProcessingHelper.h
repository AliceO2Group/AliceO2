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

uint64_t processRawData(o2::framework::InputRecord& inputs, std::unique_ptr<o2::tpc::rawreader::RawReaderCRU>& reader, bool useOldSubspec = false, const std::vector<int>& sectors = {}, size_t* nerrors = nullptr, uint32_t syncOffsetReference = 144, uint32_t decoderType = 1, bool useTrigger = true, bool returnOnNoTrigger = false);

/// absolute BC relative to TF start (firstOrbit)
std::vector<o2::framework::InputSpec> getFilter(o2::framework::InputRecord& inputs);

/// absolute BC relative to TF start (firstOrbit)
int getTriggerBCoffset(o2::framework::InputRecord& inputs, std::vector<o2::framework::InputSpec> filter = {}, bool slowScan = false);

/// absolute BC relative to TF start (firstOrbit)
/// \param data full raw page (incl. RDH)
/// \param size size of raw page
/// \param firstOrbit first orbit of the TF
int getTriggerBCoffset(const char* data, size_t size, uint32_t firstOrbit);

} // namespace calib_processing_helper
} // namespace tpc
} // namespace o2

#endif
