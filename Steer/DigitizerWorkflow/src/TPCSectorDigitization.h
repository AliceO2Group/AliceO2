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

#ifndef STEER_DIGITIZERWORKFLOW_TPCSECTORDIGITIZATION_H_
#define STEER_DIGITIZERWORKFLOW_TPCSECTORDIGITIZATION_H_

#include <functional>
#include <mutex>
#include <vector>
#include <RtypesCore.h>
#include "CommonDataFormat/RangeReference.h"
#include "DataFormatsTPC/Digit.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "TPCSimulation/CommonMode.h"

class TChain;

namespace o2
{
namespace steer
{
class DigitizationContext;
}
namespace tpc
{
class Digitizer;

/// Output buffers of one flush of a sector
struct SectorFlushBuffers {
  std::vector<Digit> digits;
  dataformats::MCTruthContainer<MCCompLabel> labels;
  std::vector<CommonMode> commonMode;
};

/// Digitize all collisions of one sector of a timeframe.
/// After each flush the new output is in buffers and onFlush is called on the calling thread.
/// The function makes no DPL calls. Concurrent calls need one Digitizer with sector seeding and one set of chains
/// per thread, and a common cdbMutex, which serialises the calls that read calibrations through the CDBInterface.
/// \return the digit groups, one per collision part in triggered mode, one for the timeframe in continuous mode
std::vector<dataformats::RangeReference<int, int>> digitizeSector(Digitizer& digitizer, const steer::DigitizationContext& context,
                                                                  const std::vector<TChain*>& simChains, int sector, ULong64_t timeframe,
                                                                  SectorFlushBuffers& buffers, const std::function<void()>& onFlush,
                                                                  std::mutex* cdbMutex = nullptr);

} // namespace tpc
} // namespace o2

#endif
