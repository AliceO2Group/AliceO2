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

#include "HMPIDSimulation/HMPIDDigitizer.h"
#include "DataFormatsHMP/Digit.h"
#include "DataFormatsHMP/Trigger.h"

#include "Framework/Logger.h"

using namespace o2::hmpid;

ClassImp(HMPIDDigitizer);

float HMPIDDigitizer::getThreshold(o2::hmpid::Digit const& digiti) const
{
  // TODO: implement like in AliRoot some thresholding depending on conditions ...
  return 4.;
}

// applies threshold to digits; removes the ones below a certain charge threshold
void HMPIDDigitizer::zeroSuppress(std::vector<o2::hmpid::Digit> const& digits, std::vector<o2::hmpid::Digit>& newdigits,
                                  o2::dataformats::MCTruthContainer<o2::MCCompLabel> const& labels,
                                  o2::dataformats::MCTruthContainer<o2::MCCompLabel>* newlabels)
{
  int index = 0;
  for (auto& digit : digits) {
    if (digit.getCharge() >= getThreshold(digit)) {
      //     if(digit.getPx() < 80 && digit.getPy() < 48) {
      newdigits.push_back(digit);
      if (newlabels) {
        // copy the labels to the new place with the right new index
        newlabels->addElements(newdigits.size() - 1, labels.getLabels(index));
      }
      //   }
    }
    index++;
  }
}

void HMPIDDigitizer::flush(std::vector<o2::hmpid::Digit>& digits)
{
  // flushing and finalizing digits in the workspace
  zeroSuppress(mDigits, digits, mTmpLabelContainer, mRegisteredLabelContainer);
  reset();
}

void HMPIDDigitizer::reset()
{
  mIndexForPad.clear();
  mInvolvedPads.clear();
  mDigits.clear();
  mTmpLabelContainer.clear();
}

// this will process hits and fill the digit vector with digits which are finalized
void HMPIDDigitizer::process(std::vector<o2::hmpid::HitType> const& hits, std::vector<o2::hmpid::Digit>& digits)
{
  LOG(info) << "Starting HMPPID digitizer process function";

  for (auto& hit : hits) {
    int chamber, pc, px, py;
    float totalQ;
    // retrieves center pad and the total charge
    o2::hmpid::Digit::getPadAndTotalCharge(hit, chamber, pc, px, py, totalQ);

    if (px < 0 || py < 0) {
      continue;
    }

    // determine which pads to loop over
    std::array<uint32_t, 9> allpads;
    int counter = 0;
    for (int nx = -1; nx <= 1; ++nx) {
      for (int ny = -1; ny <= 1; ++ny) {
        if ((px + nx) < 0 || (px + nx) > 79 || (py + ny) < 0 || (py + ny) > 47) {
          // LOG(info) << ">> Pad out the PhotoCathod boundary. Excluded :" << px << " " << py << " :" << nx << "," << ny;
          continue;
        }
        allpads[counter] = o2::hmpid::Digit::abs(chamber, pc, px + nx, py + ny);
        counter++;
      }
    }
    // LOG(info) << "." <<  px << " " << py ;
    for (auto& pad : allpads) {
      auto iter = mIndexForPad.find(pad);
      int index = -1;
      if (iter != mIndexForPad.end()) {
        index = iter->second;
      }
      // auto index = mIndexForPad[pad];
      float fraction = o2::hmpid::Digit::getFractionalContributionForPad(hit, (int)pad);
      // LOG(info) << "FRACTION ON PAD " << pad << " IS " << fraction;
      if (index != -1) {
        // digit exists ... reuse
        auto& digit = mDigits[index];
        digit.addCharge(totalQ * fraction);

        if (mRegisteredLabelContainer) {
          auto labels = mTmpLabelContainer.getLabels(index);
          o2::MCCompLabel newlabel(hit.GetTrackID(), mEventID, mSrcID, false);
          bool newlabelneeded = true;
          for (auto& l : labels) {
            if (l == newlabel) {
              newlabelneeded = false;
              break;
            }
          }
          if (newlabelneeded) {
            mTmpLabelContainer.addElementRandomAccess(index, newlabel);
          }
        }
      } else {
        // create digit ... and register
        //        mDigits.emplace_back(mCurrentTriggerTime, pad, totalQ * fraction);
        mDigits.emplace_back(pad, totalQ * fraction);
        mIndexForPad[pad] = mDigits.size() - 1;
        mInvolvedPads.emplace_back(pad);

        if (mRegisteredLabelContainer) {
          // add label for this digit
          mTmpLabelContainer.addElement(mDigits.size() - 1, o2::MCCompLabel(hit.GetTrackID(), mEventID, mSrcID, false));
        }
      }
    }
  }
}
