//
//  DigitStave.cxx
//  ALICEO2
//
//  Created by Markus Fasel on 26.03.15.
//
//
#include "ITSSimulation/DigitStave.h"
#include "ITSSimulation/Digit.h"

#include "FairLogger.h"  // for LOG

#include "TClonesArray.h"

using namespace AliceO2::ITS;

DigitStave::DigitStave()
{

}

DigitStave::~DigitStave()
{ }

void DigitStave::Reset()
{
  for (auto pixel: fPixels) {
    delete pixel.second;
  }
  fPixels.clear();
}

Digit *DigitStave::FindDigit(Int_t pixel)
{
  Digit *result = nullptr;
  auto digitentry = fPixels.find(pixel);
  if (digitentry != fPixels.end()) {
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
  fPixels.insert(std::pair<int, Digit *>(pixel, digi));
}

void DigitStave::FillOutputContainer(TClonesArray *outputcont)
{
  TClonesArray &clref = *outputcont;
  for (auto digit: fPixels) {
    new(clref[clref.GetEntriesFast()]) Digit(*(digit.second));
  }
}
