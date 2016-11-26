/// \file AliITSUpgradeDigitizer.cxx
/// \brief Digitizer for the upgrated ITS
#include "ITSSimulation/Digitizer.h"
#include "ITSSimulation/Point.h"
#include "ITSBase/UpgradeGeometryTGeo.h"
#include "ITSBase/UpgradeSegmentationPixel.h"
#include "ITSBase/Digit.h"
#include "ITSSimulation/DigitChip.h"
#include "ITSSimulation/DigitContainer.h"

#include "FairLogger.h"           // for LOG
#include "TClonesArray.h"         // for TClonesArray

ClassImp(AliceO2::ITS::Digitizer)

using namespace AliceO2::ITS;

Digitizer::Digitizer()
{
  fGeometry = new UpgradeGeometryTGeo(kTRUE, kTRUE);
  fDigitContainer = new DigitContainer(fGeometry->getNumberOfChips());
  const UpgradeSegmentationPixel *seg =
     (UpgradeSegmentationPixel*)fGeometry->getSegmentationById(0);
  DigitChip::SetNRows(seg->getNumberOfRows());
}

Digitizer::~Digitizer()
{
  delete fGeometry;
  if (fDigitContainer) { delete fDigitContainer; }
}

DigitContainer *Digitizer::Process(TClonesArray *points)
{
    fDigitContainer->Reset();

  // Convert points to summable digits
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
    fGeometry->globalToLocal(chipID,glo,loc);
    const UpgradeSegmentationPixel *seg =
       (UpgradeSegmentationPixel*)fGeometry->getSegmentationById(0);
    Int_t ix, iz;
    seg->localToDetector(loc[0],loc[2],ix,iz);
    if ((ix<0) || (iz<0)) {
       LOG(DEBUG) << "Out of the chip" << FairLogger::endl;
       continue; 
    }
    Digit *digit=fDigitContainer->AddDigit(
       chipID, ix, iz, charge, point->GetTime()
    );
    digit->SetLabel(0,label);
  }
  return fDigitContainer;
}

