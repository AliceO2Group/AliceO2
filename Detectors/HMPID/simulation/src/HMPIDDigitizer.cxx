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
      newdigits.push_back(digit);

      if (newlabels) {
        // copy the labels to the new place with the right new index
        newlabels->addElements(newdigits.size() - 1, labels.getLabels(index));
      }
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
  for (auto& hit : hits) {
    int chamber, pc, px, py;
    float totalQ;
    // retrieves center pad and the total charge
    Digit::getPadAndTotalCharge(hit, chamber, pc, px, py, totalQ);

    if (px < 0 || py < 0) {
      continue;
    }

    // determine which pads to loop over
    std::array<int, 9> allpads;
    int counter = 0;
    for (int nx = -1; nx <= 1; ++nx) {
      for (int ny = -1; ny <= 1; ++ny) {
        allpads[counter] = Param::Abs(chamber, pc, px + nx, py + ny);
        counter++;
      }
    }

    for (auto& pad : allpads) {
      auto iter = mIndexForPad.find(pad);
      int index = -1;
      if (iter != mIndexForPad.end()) {
        index = iter->second;
      }
      // auto index = mIndexForPad[pad];
      float fraction = Digit::getFractionalContributionForPad(hit, pad);
      // LOG(INFO) << "FRACTION ON PAD " << pad << " IS " << fraction;
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
        mDigits.emplace_back(mCurrentTriggerTime, pad, totalQ * fraction);
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
