// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "ZDCSimulation/Digitizer.h"
#include "ZDCSimulation/Digit.h"
#include "ZDCSimulation/Hit.h"
#include <vector>

using namespace o2::zdc;

ClassImp(Digitizer);

float Digitizer::getThreshold(o2::zdc::Digit const& digiti) const
{
  // TODO: implement like in AliRoot some thresholding depending on conditions ...
  return 4.;
}

// applies threshold to digits; removes the ones below a certain charge threshold
void Digitizer::zeroSuppress(std::vector<o2::zdc::Digit> const& digits, std::vector<o2::zdc::Digit>& newdigits,
                             o2::dataformats::MCTruthContainer<o2::MCCompLabel> const& labels,
                             o2::dataformats::MCTruthContainer<o2::MCCompLabel>* newlabels)
{
  int index = 0;
  for (auto& digit : digits) {
    if (digit.getADC() >= getThreshold(digit)) {
      newdigits.push_back(digit);

      if (newlabels) {
        // copy the labels to the new place with the right new index
        newlabels->addElements(newdigits.size() - 1, labels.getLabels(index));
      }
    }
    index++;
  }
}

void Digitizer::flush(std::vector<o2::zdc::Digit>& digits)
{
  // flushing and finalizing digits in the workspace
  zeroSuppress(mDigits, digits, mTmpLabelContainer, mRegisteredLabelContainer);
  reset();
}

void Digitizer::reset()
{
  mDigits.clear();
  mTmpLabelContainer.clear();
}

// this will process hits and fill the digit vector with digits which are finalized
void Digitizer::process(std::vector<o2::zdc::Hit> const& hits, std::vector<o2::zdc::Digit>& digits)
{
  // loop over all hits and produce digits
  for (auto& hit : hits) {

    // for each hit find out sector + detector information
    const auto detID = hit.GetDetectorID();
    const auto secID = hit.getSector();

    const auto channel = toChannel(detID, secID);

    // if digit for this sector does not exist, create one otherwise add to it
    auto& digit = digits[channel];
    digit.setDetInfo(detID, secID);

    // the right photonelectron number to use depends on the sector ID
    if (secID == 0) {
      digit.add(fromPhotoelectronsToACD(hit.getPMCLightYield()));
    } else {
      digit.add(fromPhotoelectronsToACD(hit.getPMQLightYield()));
    }
  }
}
