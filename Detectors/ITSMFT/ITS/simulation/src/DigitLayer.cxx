// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See https://alice-o2.web.cern.ch/ for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

//
//  DigitLayer.cxx
//  ALICEO2
//
//  Created by Markus Fasel on 25.03.15.
//
//



#include "ITSSimulation/DigitLayer.h"
#include "ITSSimulation/DigitStave.h"

#include "FairLogger.h"

using o2::ITSMFT::Digit;
using namespace o2::ITS;

DigitLayer::DigitLayer(Int_t layerID, Int_t nstaves) :
  mLayerID(layerID),
  mNStaves(nstaves),
  mStaves(nullptr)
{
  mStaves = new DigitStave *[mNStaves];
  for (int istave = 0; istave < mNStaves; istave++) {
    mStaves[istave] = new DigitStave();
  }
}

DigitLayer::~DigitLayer()
{
  for (int istave = 0; istave < 0; istave++) {
    delete mStaves[istave];
  }
}

void DigitLayer::SetDigit(Digit *digi, Int_t stave, Int_t pixel)
{
  if (stave >= mNStaves) {
    LOG(ERROR) << "Stave index " << stave << " out of range for layer " << mLayerID << ", maximum " << mNStaves <<
               FairLogger::endl;
  } else {
    mStaves[stave]->SetDigit(pixel, digi);
  }
}

Digit *DigitLayer::FindDigit(Int_t stave, Int_t pixel)
{
  if (stave > mNStaves) {
    LOG(ERROR) << "Stave index " << stave << " out of range for layer " << mLayerID << ", maximum " << mNStaves <<
               FairLogger::endl;
    return nullptr;
  }
  return mStaves[stave]->FindDigit(pixel);
}

void DigitLayer::Reset()
{
  for (DigitStave **staveIter = mStaves; staveIter < mStaves + mNStaves; staveIter++) {
    (*staveIter)->Reset();
  }
}

void DigitLayer::FillOutputContainer(TClonesArray *output)
{
  for (DigitStave **staveIter = mStaves; staveIter < mStaves + mNStaves; staveIter++) {
    (*staveIter)->FillOutputContainer(output);
  }
}
