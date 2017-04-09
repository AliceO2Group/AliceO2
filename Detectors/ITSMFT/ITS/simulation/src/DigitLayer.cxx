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
