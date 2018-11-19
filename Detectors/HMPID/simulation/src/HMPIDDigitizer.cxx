// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "HMPIDSimulation/HMPIDDigitizer.h"
#include "HMPIDBase/Digit.h"

using namespace o2::hmpid;

ClassImp(HMPIDDigitizer);

// this will process hits and fill the digit vector with digits which are finalized
void HMPIDDigitizer::process(std::vector<o2::hmpid::HitType> const& hits, std::vector<o2::hmpid::Digit>& digits)
{
  // this is a very simple variant that creates one digit from one hit
  // conversion is done in the digit constructor
  // TODO: introduce cross-talk, pile-up, etc.
  for (auto& hit : hits) {
    digits.emplace_back(hit);
  }
}
