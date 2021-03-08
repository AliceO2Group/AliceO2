// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_TPC_RawProcessingHelpers_H
#define O2_TPC_RawProcessingHelpers_H

#include <functional>

#include "TPCBase/RDHUtils.h"

namespace o2
{
namespace tpc
{
namespace raw_processing_helpers
{

using ADCCallback = std::function<bool(int cru, int rowInSector, int padInRow, int timeBin, float adcValue)>;

bool processZSdata(const char* data, size_t size, rdh_utils::FEEIDType feeId, uint32_t globalBCoffset, ADCCallback fillADC, bool useTimeBin = false);

} // namespace raw_processing_helpers
} // namespace tpc
} // namespace o2
#endif
