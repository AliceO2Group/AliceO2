/// \file Point.cxx
/// \brief Implementation of the Point class

#include "ITSMFTSimulation/Point.h"

#include <iostream>

ClassImp(o2::ITSMFT::Point)

using std::cout;
using std::endl;
using namespace o2::ITSMFT;
using namespace o2; //::Base;


void Point::Print(const Option_t *opt) const
{
  printf("Det: %5d Track: %6d E.loss: %.3e P: %+.3e %+.3e %+.3e\n"
	 "PosIn: %+.3e %+.3e %+.3e PosOut: %+.3e %+.3e %+.3e\n",
	 GetDetectorID(),GetTrackID(),GetEnergyLoss(),GetPx(),GetPy(),GetPz(),
	 GetStartX(),GetStartY(),GetStartZ(),GetX(),GetY(),GetZ() );
}


