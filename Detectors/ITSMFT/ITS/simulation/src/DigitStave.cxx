//
//  DigitStave.cxx
//  ALICEO2
//
//  Created by Markus Fasel on 26.03.15.
//
//
#include "ITSSimulation/DigitStave.h"
#include "ITSMFTBase/Digit.h"

#include "FairLogger.h"  // for LOG

#include "TClonesArray.h"

using o2::ITSMFT::Digit;
using namespace o2::ITS;

DigitStave::DigitStave()
{

}

DigitStave::~DigitStave()
= default;

void DigitStave::Reset()
{
  for (auto pixel: mPixels) {
    delete pixel.second;
  }
  mPixels.clear();
}

Digit *DigitStave::FindDigit(Int_t pixel)
{
  Digit *result = nullptr;
  auto digitentry = mPixels.find(pixel);
  if (digitentry != mPixels.end()) {
    result = digitentry->second;
  }
  return result;
}

void DigitStave::SetDigit(int pixel, Digit *digi)
{
  Digit *olddigit = FindDigit(pixel);
  if (olddigit != nullptr) {
    LOG(ERROR) << "Pixel already set previously, replacing and deleting previous content" << FairLogger::endl;
    delete olddigit;
  }
  mPixels.insert(std::pair<int, Digit *>(pixel, digi));
}

void DigitStave::FillOutputContainer(TClonesArray *outputcont)
{
  TClonesArray &clref = *outputcont;
  for (auto digit: mPixels) {
    new(clref[clref.GetEntriesFast()]) Digit(*(digit.second));
  }
}
