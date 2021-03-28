// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_TRD_ADCARRAY_H_
#define ALICEO2_TRD_ADCARRAY_H_

#include "DataFormatsTRD/Constants.h"
#include "SimulationDataFormat/MCCompLabel.h"

#include <array>
#include <unordered_map>
#include <vector>

namespace o2
{
namespace trd
{

struct SignalArray {
  double firstTBtime;                               // first TB time
  std::array<float, constants::TIMEBINS> signals{}; // signals
  std::unordered_map<int, int> trackIds;            // tracks Ids associated to the signal
  std::vector<o2::MCCompLabel> labels;              // labels associated to the signal
  bool isDigit = false;                             // flag a signal converted to a digit
  bool isShared = false;                            // flag if converted digit is shared (copied)
                                                    // if that is the case, also the labels have to be copied
};

} // namespace trd
} // namespace o2
#endif
