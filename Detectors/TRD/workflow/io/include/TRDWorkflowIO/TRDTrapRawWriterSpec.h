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

#ifndef O2_TRDTRAPSIMULATORRAWWRITER_H
#define O2_TRDTRAPSIMULATORRAWWRITER_H

#include <fstream>
#include <iostream>

// This is the raw data that is output by the trap trap chip.
// mcmheader word followed by 1 or more traprawtracklet words.
// and a halfchamber header at the beginning of a half chamber.
// halfchamber header + mcmheader uniquely identifies the chip producing the data.

namespace o2
{
namespace framework
{
struct DataProcessorSpec;
}
} // namespace o2

namespace o2
{
namespace trd
{

o2::framework::DataProcessorSpec getTRDTrapRawWriterSpec();

} // end namespace trd
} // end namespace o2

#endif // O2_TRDTRAPSIMULATORRAWWRITER_H
