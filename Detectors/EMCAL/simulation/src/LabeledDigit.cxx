// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "EMCALSimulation/LabeledDigit.h"
#include <iostream>

using namespace o2::emcal;

LabeledDigit::LabeledDigit(Digit digit, o2::emcal::MCLabel label)
  : mDigit(digit)
{
  mLabels.push_back(label);
}

LabeledDigit::LabeledDigit(Short_t tower, Double_t amplitudeGeV, Double_t time, o2::emcal::MCLabel label, ChannelType_t ctype)
  : mDigit(tower, amplitudeGeV, time, ctype)
{
  mLabels.push_back(label);
}

LabeledDigit& LabeledDigit::operator+=(const LabeledDigit& other)
{
  if (canAdd(other)) {
    Int_t a1 = getAmplitudeADC();
    Int_t a2 = other.getAmplitudeADC();
    Double_t r = ((a1 + a2) != 0) ? 1.0 / (a1 + a2) : 0.0;
    mDigit += other.getDigit();

    for (int j = 0; j < mLabels.size(); j++) {
      mLabels.at(j).setAmplitudeFraction(mLabels.at(j).getAmplitudeFraction() * a1 * r);
    }

    for (auto label : other.getLabels()) {
      label.setAmplitudeFraction(label.getAmplitudeFraction() * a2 * r);
      mLabels.push_back(label);
    }
  }
  // Does nothing if the digits are in different towers or have incompatible times.
  return *this;
}

void LabeledDigit::PrintStream(std::ostream& stream) const
{
  stream << "EMCAL LabeledDigit: Tower " << getTower() << ", Time " << getTimeStamp() << ", Amplitude " << getAmplitude() << " GeV, Type " << getType() << ", Labels ( ";
  for (auto label : mLabels) {
    stream << label.getRawValue() << " ";
  }
  stream << ")";
}

std::ostream& operator<<(std::ostream& stream, const LabeledDigit& digi)
{
  digi.PrintStream(stream);
  return stream;
}
