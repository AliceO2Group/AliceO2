// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

// System includes
#include <fmt/ostream.h>

// O2 includes
#include "DataFormatsAnalysis/FlagReasons.h"

using namespace o2::analysis;

ClassImp(FlagReasons);

void FlagReasons::streamTo(std::ostream& output) const
{
  fmt::print(output, "{:=^38}\n", "| Flags |");
  fmt::print(output, "{:>5} : {:>30}\n", "bit", "reason");
  for (int iReason = 0; iReason< mReasonCollection.size(); ++iReason) {
    fmt::print(output, "{:>5} : {:>30}\n", iReason, mReasonCollection[iReason]);
  }
}
