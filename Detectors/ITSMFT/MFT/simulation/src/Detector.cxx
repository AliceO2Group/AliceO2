/// \file Detector.cxx
/// \brief Implementation of the Detector class

#include "Detector.h"
#include "GeometryTGeo.h"

#include "DataFormats/simulation/include/DetectorList.h"
#include "TVirtualMC.h"

using namespace AliceO2::MFT;

//_____________________________________________________________________________
Detector::Detector()
  : AliceO2::Base::Detector("MFT", kTRUE, kAliMft),
    mGeometryTGeo(0) 
{

}

//_____________________________________________________________________________
Detector::Detector(const Detector& rhs)
  : AliceO2::Base::Detector(rhs) 
{

}

//_____________________________________________________________________________
Detector::~Detector()
{

}

//_____________________________________________________________________________
Detector& Detector::operator=(const Detector& rhs)
{
  // The standard = operator
  // Inputs:
  //   Detector   &h the sourse of this copy
  // Outputs:
  //   none.
  // Return:
  //  A copy of the sourse hit h

  if (this == &rhs) {
    return *this;
  }

  // base class assignment
  Base::Detector::operator=(rhs);

}

//_____________________________________________________________________________
void Detector::Initialize()
{

  mGeometryTGeo = new GeometryTGeo();

  FairDetector::Initialize();

}

//_____________________________________________________________________________
Bool_t Detector::ProcessHits(FairVolume* vol)
{
  // This method is called from the MC stepping
  if (!(TVirtualMC::GetMC()->TrackCharge())) {
    return kFALSE;
  }

}

