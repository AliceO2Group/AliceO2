/// \file AliITSUpgradeDigitizer.cxx
/// \brief Digitizer for the upgrated ITS
#include "Digitizer.h"
#include "Point.h"                // for Point
#include "UpgradeGeometryTGeo.h"  // for UpgradeGeometryTGeo
#include "Digit.h"            // for Digit
#include "DigitContainer.h"   // for DigitContainer

#include "FairLogger.h"           // for LOG
#include "TClonesArray.h"         // for TClonesArray
#include "TCollection.h"          // for TIter

#include <iostream>

ClassImp(AliceO2::ITS::Digitizer)

using namespace AliceO2::ITS;

Digitizer::Digitizer():
TObject(),
fChipContainer(),
fDigitContainer(nullptr),
fGain(1.)
{
  fGeometry = new UpgradeGeometryTGeo(kTRUE, kTRUE);
}

Digitizer::~Digitizer(){
  delete fGeometry;
  if(fDigitContainer) delete fDigitContainer;
}

void Digitizer::Init(){
  fDigitContainer = new DigitContainer(fGeometry);

  for (int i = 0; i < fGeometry->getNumberOfChips(); i++) {
    fChipContainer.push_back(Chip(i, fGeometry));
  }
}

DigitContainer *Digitizer::Process(TClonesArray *points){
  fDigitContainer->Reset();
  ClearChips();

  // Assign points to chips
  for (TIter pointiter = TIter(points).Begin(); pointiter != TIter::End(); ++pointiter) {
    Point *itspoint = dynamic_cast<Point *>(*pointiter);
    if (itspoint->GetDetectorID() > fChipContainer.size()){
      LOG(ERROR) << "Chip ID " << itspoint->GetDetectorID() <<"out of range, max " << fChipContainer.size() << FairLogger::endl;
      continue;
    }
    fChipContainer[itspoint->GetDetectorID()].InsertPoint(itspoint);
  }

  // Convert points to summable digits
  for (TIter pointiter = TIter(points).Begin(); pointiter != TIter::End(); ++pointiter) {
    Point *inputpoint = dynamic_cast<Point *>(*pointiter);
    LOG(DEBUG) << "Processing next point: " << FairLogger::endl;
    LOG(DEBUG) << "=======================" << FairLogger::endl;
    LOG(DEBUG) << *inputpoint << FairLogger::endl;
    Int_t layer = fGeometry->getLayer(inputpoint->GetDetectorID()),
    stave = fGeometry->getStave(inputpoint->GetDetectorID()),
    staveindex = fGeometry->getChipIdInStave(inputpoint->GetDetectorID());
    Double_t charge = inputpoint->GetEnergyLoss();
    Digit * digit = fDigitContainer->FindDigit(layer, stave, staveindex);
    if(digit){
      // For the moment only add the charge to the current digit
      LOG(DEBUG) << "Digit already found" << FairLogger::endl;
      double chargeDigit = digit->GetCharge();
      chargeDigit += charge * fGain;
    } else {
      LOG(DEBUG) << "Creating new digit" << FairLogger::endl;
      // @TODO Impplement handling og pixels within the chip
      digit = new Digit(inputpoint->GetDetectorID(), 0, charge, inputpoint->GetTime());
      fDigitContainer->AddDigit(digit);
    }
  }
  return fDigitContainer;
}

void Digitizer::ClearChips(){
  for (auto chipIter : fChipContainer) {
    chipIter.Clear();
  }
}
