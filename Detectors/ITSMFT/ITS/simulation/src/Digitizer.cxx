/// \file Digitizer.cxx
/// \brief Implementation of the ITS digitizer

#include "ITSMFTBase/Digit.h"
#include "ITSMFTBase/SegmentationPixel.h"
#include "ITSMFTSimulation/Point.h"
#include "ITSSimulation/DigitChip.h"
#include "ITSSimulation/Digitizer.h"

#include "FairLogger.h"   // for LOG
#include "TClonesArray.h" // for TClonesArray

ClassImp(o2::ITS::Digitizer)

using o2::ITSMFT::Point;
using o2::ITSMFT::Chip;
using o2::ITSMFT::SimulationAlpide;
using o2::ITSMFT::Digit;
using o2::ITSMFT::SegmentationPixel;

using namespace o2::ITS;

Digitizer::Digitizer() : mGeometry(), mNumOfChips(0), mChips(), mSimulations(), mDigitContainer() {}

Digitizer::~Digitizer() = default;

void Digitizer::init(Bool_t build)
{
  mGeometry.Build(build);
  mNumOfChips = mGeometry.getNumberOfChips();

  mChips.resize(mNumOfChips);
  mSimulations.resize(mNumOfChips);
  mDigitContainer.resize(mNumOfChips);

  SegmentationPixel* seg = (SegmentationPixel*)mGeometry.getSegmentationById(0);
  DigitChip::setNumberOfRows(seg->getNumberOfRows());

  Double_t param[] = {
    50,     // ALPIDE threshold
    -1.315, // ACSFromBGPar0
    0.5018, // ACSFromBGPar1
    1.084   // ACSFromBGPar2
  };
  for (Int_t i = 0; i < mNumOfChips; i++) {
    mChips[i].Init(i, mGeometry.getMatrixSensor(i));
    mSimulations[i].Init(param, seg, &mChips[i]);
  }
}

void Digitizer::process(TClonesArray* points, TClonesArray* digits)
{
  // Convert points to digits
  for (TIter iter = TIter(points).Begin(); iter != TIter::End(); ++iter) {
    Point* point = dynamic_cast<Point*>(*iter);
    Int_t chipID = point->GetDetectorID();
    if (chipID >= mNumOfChips)
      continue;
    mChips[chipID].InsertPoint(point);
  }

  for (Int_t i = 0; i < mNumOfChips; i++) {
    mSimulations[i].DigitiseChip(digits);
  }
}

DigitContainer& Digitizer::process(TClonesArray* points)
{
  mDigitContainer.reset();

  // Convert points to digits
  for (TIter pointiter = TIter(points).Begin(); pointiter != TIter::End(); ++pointiter) {
    Point* point = dynamic_cast<Point*>(*pointiter);

    LOG(DEBUG) << "Processing next point: " << FairLogger::endl;
    LOG(DEBUG) << "=======================" << FairLogger::endl;
    LOG(DEBUG) << *point << FairLogger::endl;

    Double_t x = 0.5 * (point->GetX() + point->GetStartX());
    Double_t y = 0.5 * (point->GetY() + point->GetStartY());
    Double_t z = 0.5 * (point->GetZ() + point->GetStartZ());
    Double_t charge = point->GetEnergyLoss();
    Int_t label = point->GetTrackID();
    Int_t chipID = point->GetDetectorID();

    LOG(DEBUG) << "Creating new digit" << FairLogger::endl;
    const Double_t glo[3] = { x, y, z };
    Double_t loc[3] = { 0., 0., 0. };
    mGeometry.globalToLocal(chipID, glo, loc);
    const SegmentationPixel* seg = (SegmentationPixel*)mGeometry.getSegmentationById(0);
    Int_t ix, iz;
    seg->localToDetector(loc[0], loc[2], ix, iz);
    if ((ix < 0) || (iz < 0)) {
      LOG(DEBUG) << "Out of the chip" << FairLogger::endl;
      continue;
    }
    Digit* digit = mDigitContainer.addDigit(chipID, ix, iz, charge, point->GetTime());
    digit->setLabel(0, label);
  }
  return mDigitContainer;
}
