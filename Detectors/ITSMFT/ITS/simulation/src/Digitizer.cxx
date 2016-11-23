/// \file AliITSUpgradeDigitizer.cxx
/// \brief Digitizer for the upgrated ITS
#include "ITSSimulation/Digitizer.h"
#include "ITSSimulation/Point.h"
#include "ITSSimulation/UpgradeGeometryTGeo.h"
#include "ITSSimulation/UpgradeSegmentationPixel.h"
#include "ITSSimulation/Digit.h"
#include "ITSSimulation/DigitContainer.h"

#include "FairLogger.h"           // for LOG
#include "TClonesArray.h"         // for TClonesArray
#include "TCollection.h"          // for TIter

#include <iostream>

ClassImp(AliceO2::ITS::Digitizer)

using namespace AliceO2::ITS;

Digitizer::Digitizer() :
  fGain(1.)
{
  fGeometry = new UpgradeGeometryTGeo(kTRUE, kTRUE);
  fDigitContainer = new DigitContainer(fGeometry->getNumberOfChips());

  // Default segmentation.
  const double kSensThick = 18e-4;
  const double kPitchX = 20e-4;
  const double kPitchZ = 20e-4;
  const int kNRow = 650;
  const int kNCol = 1500;
  const double kSiThickIB = 150e-4;
  const double kSiThickOB = 150e-4;
  const double kGuardRing = 50e-4; // width of passive area on left/right/top of the sensor
  const double kReadOutEdge = 0.2; // width of the readout edge (passive bottom)
  fSeg = new UpgradeSegmentationPixel(
    0,           // segID (0:9)
    1,           // chips per module
    kNCol,       // ncols (total for module)
    kNRow,       // nrows
    kPitchX,     // default row pitch in cm
    kPitchZ,     // default col pitch in cm
    kSensThick,  // sensor thickness in cm
    -1,          // no special left col between chips
    -1,          // no special right col between chips
    kGuardRing,  // left
    kGuardRing,  // right
    kGuardRing,  // top
    kReadOutEdge // bottom
  );
}

Digitizer::~Digitizer()
{
  delete fGeometry;
  delete fSeg;
  if (fDigitContainer) { delete fDigitContainer; }
}

DigitContainer *Digitizer::Process(TClonesArray *points)
{
  fDigitContainer->Reset();

  // Convert points to summable digits
  for (TIter pointiter = TIter(points).Begin(); pointiter != TIter::End(); ++pointiter) {
    Point *point = dynamic_cast<Point *>(*pointiter);

    Float_t x,z;
    if (point->IsStopped()) {
      x=point->GetX();
      z=point->GetZ();
    } else if (point->IsExiting()) {
      x=0.5*(point->GetX() + point->GetStartX());
      z=0.5*(point->GetZ() + point->GetStartZ());
    } else { continue; }
    
    LOG(DEBUG) << "Processing next point: " << FairLogger::endl;
    LOG(DEBUG) << "=======================" << FairLogger::endl;
    LOG(DEBUG) << *point << FairLogger::endl;
    Double_t charge = point->GetEnergyLoss();
    Int_t label = point->GetTrackID();
    Int_t chipID= point->GetDetectorID();
    
    LOG(DEBUG) << "Creating new digit" << FairLogger::endl;
    const Double_t glo[3]= {point->GetX(), point->GetY(), point->GetZ()};
    Double_t loc[3]={0.,0.,0.};    
    fGeometry->globalToLocal(chipID,glo,loc);
    Int_t ix, iz;
    fSeg->localToDetector(loc[0],loc[2],ix,iz);
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

