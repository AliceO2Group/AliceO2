/// \file AliITSDigitizer.cxx
/// \brief Digitizer for the upgrated ITS

#include "ITSBase/SegmentationPixel.h"
#include "ITSBase/Digit.h"
#include "ITSSimulation/SimulationAlpide.h"
#include "ITSSimulation/Chip.h"
#include "ITSSimulation/Digitizer.h"
#include "ITSSimulation/Point.h"
#include "ITSSimulation/DigitChip.h"

#include "FairLogger.h"           // for LOG
#include "TClonesArray.h"         // for TClonesArray

ClassImp(AliceO2::ITS::Digitizer)

using namespace AliceO2::ITS;

Digitizer::Digitizer():
fGeometry(),
fNChips(0),
fChips(),
fSimulations(),
fDigitContainer()
{
}

Digitizer::~Digitizer()
{
}

void Digitizer::Init(Bool_t build)
{
  fGeometry.Build(build);
  fNChips=fGeometry.getNumberOfChips();

  fChips.resize(fNChips, Chip(0,&fGeometry));
  fSimulations.resize(fNChips);
  fDigitContainer.Resize(fNChips);
  
  SegmentationPixel *seg =
     (SegmentationPixel*)fGeometry.getSegmentationById(0);
  DigitChip::SetNRows(seg->getNumberOfRows());
  
  Double_t param[]={
    50,     // ALPIDE threshold
    -1.315, // ACSFromBGPar0
    0.5018, // ACSFromBGPar1
    1.084   // ACSFromBGPar2
  };
  for (Int_t i=0; i<fNChips; i++) {
    fChips[i].SetChipIndex(i);
    fSimulations[i].Init(param, seg, &fChips[i]);
  }
}

void Digitizer::Process(TClonesArray *points, TClonesArray *digits)
{

  // Convert points to digits
  for (TIter iter = TIter(points).Begin(); iter != TIter::End(); ++iter) {
    Point *point = dynamic_cast<Point *>(*iter);
    Int_t chipID= point->GetDetectorID();
    if (chipID >= fNChips) continue;
    fChips[chipID].InsertPoint(point); 
  }

  for (Int_t i=0; i<fNChips; i++) {
    fSimulations[i].DigitiseChip(digits);
  }

}

DigitContainer &Digitizer::Process(TClonesArray *points)
{
    fDigitContainer.Reset();

  // Convert points to digits
  for (TIter pointiter = TIter(points).Begin(); pointiter != TIter::End(); ++pointiter) {
    
    Point *point = dynamic_cast<Point *>(*pointiter);

    LOG(DEBUG) << "Processing next point: " << FairLogger::endl;
    LOG(DEBUG) << "=======================" << FairLogger::endl;
    LOG(DEBUG) << *point << FairLogger::endl;

    Double_t x=0.5*(point->GetX() + point->GetStartX());
    Double_t y=0.5*(point->GetY() + point->GetStartY());
    Double_t z=0.5*(point->GetZ() + point->GetStartZ());
    Double_t charge = point->GetEnergyLoss();
    Int_t label = point->GetTrackID();
    Int_t chipID= point->GetDetectorID();
    
    LOG(DEBUG) << "Creating new digit" << FairLogger::endl;
    const Double_t glo[3]= {x, y, z};
    Double_t loc[3]={0.,0.,0.};    
    fGeometry.globalToLocal(chipID,glo,loc);
    const SegmentationPixel *seg =
       (SegmentationPixel*)fGeometry.getSegmentationById(0);
    Int_t ix, iz;
    seg->localToDetector(loc[0],loc[2],ix,iz);
    if ((ix<0) || (iz<0)) {
       LOG(DEBUG) << "Out of the chip" << FairLogger::endl;
       continue; 
    }
    Digit *digit=fDigitContainer.AddDigit(
       chipID, ix, iz, charge, point->GetTime()
    );
    digit->SetLabel(0,label);
  }
  return fDigitContainer;
}
