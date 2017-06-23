/// \file Digitizer.h
/// \brief Implementation of the conversion from points to digits
/// \author bogdan.vulpescu@cern.ch 
/// \date 03/05/2017

#include "FairLogger.h"

#include "MFTSimulation/Digitizer.h"

#include "ITSMFTSimulation/Point.h"

ClassImp(o2::MFT::Digitizer)

using o2::ITSMFT::Point;
using o2::ITSMFT::Chip;
using o2::ITSMFT::SimulationAlpide;
using o2::ITSMFT::Digit;
using o2::ITSMFT::SegmentationPixel;

using namespace o2::MFT;

//_____________________________________________________________________________
Digitizer::Digitizer() : mGeometry(), mNumOfChips(0), mChips(), mSimulations(), mDigitContainer() {}

//_____________________________________________________________________________
Digitizer::~Digitizer() = default;

//_____________________________________________________________________________
void Digitizer::init(Bool_t build)
{

}

//_____________________________________________________________________________
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
  }

}

//_____________________________________________________________________________
DigitContainer& Digitizer::process(TClonesArray* points)
{

  mDigitContainer.reset();
  /*
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
  */
  return mDigitContainer;

}

