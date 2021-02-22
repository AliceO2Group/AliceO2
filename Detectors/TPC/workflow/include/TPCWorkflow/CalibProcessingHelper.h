// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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

uint64_t processRawData(o2::framework::InputRecord& inputs, std::unique_ptr<RawReaderCRU>& reader, bool useOldSubspec = false, const std::vector<int>& sectors = {});
} // namespace calib_processing_helper
} // namespace tpc
} // namespace o2

#endif
