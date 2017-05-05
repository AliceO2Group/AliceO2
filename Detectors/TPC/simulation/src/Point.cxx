
#include "TPCSimulation/Point.h"
#include <iostream>

using std::cout;
using std::endl;

using namespace o2::TPC;

void Point::Print(const Option_t* opt) const
{
  cout << "-I- Point: O2tpc point for track " << GetTrackID()
       << " in detector " << GetDetectorID() << endl;
  cout << "    Position (" << GetX() << ", " << GetY() << ", " << GetZ()
       << ") cm" << endl;
  cout << "    Time " << GetTime() << " ns, n electrons " << GetEnergyLoss() << endl;
}

ClassImp(Point)
ClassImp(LinkableHitGroup)
ClassImp(ElementalHit)
