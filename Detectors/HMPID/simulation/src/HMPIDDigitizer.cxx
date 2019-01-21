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

  // for (auto& hit : hits) {
  //   digits.emplace_back(hit);
  // }

  // clear lookup structures
  for (auto& pad : mInvolvedPads) {
    mIndexForPad[pad] = -1;
  }
  mInvolvedPads.clear();

  for (auto& hit : hits) {
    int chamber, pc, px, py;
    float totalQ;
    // retrieves center pad and the total charge
    Digit::getPadAndTotalCharge(hit, chamber, pc, px, py, totalQ);
    LOG(INFO) << "CHAMBER " << chamber;
    LOG(INFO) << "PC " << pc;
    LOG(INFO) << "PX " << px;
    LOG(INFO) << "PY " << py;

    if (px < 0 || py < 0) {
      continue;
    }

    // determine which pads to loop over
    std::array<int, 9> allpads;
    int counter = 0;
    for (int nx = -1; nx <= 1; ++nx) {
      for (int ny = -1; ny <= 1; ++ny) {
        allpads[counter] = Param::Abs(chamber, pc, px + nx, py + ny);
        LOG(INFO) << "ADDING PAD " << allpads[counter];
        counter++;
      }
    }

    LOG(INFO) << "DIGIT ON MAINPAD " <<  Param::Abs(chamber, pc, px, py) << " TOTAL CHARGE " << totalQ;

    for (auto& pad : allpads) {
      auto index = mIndexForPad[pad];
      float fraction = Digit::getFractionalContributionForPad(hit, pad);
      LOG(INFO) << "FRACTION ON PAD " << pad << " IS " << fraction;
      if (index != -1) {
        // digit exists ... reuse
        auto& digit = mDigits[index];
        digit.addCharge(totalQ * fraction);
      } else {
        // create digit ... and register
        mDigits.emplace_back(pad, totalQ * fraction);
        mIndexForPad[pad] = mDigits.size() - 1;
        mInvolvedPads.emplace_back(pad);
      }
    }
  }
}
