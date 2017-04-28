#include "TPCEventdisplay/HitDraw.h"
#include "TPCSimulation/Point.h"                // for FairMCPoint

#include "TVector3.h"                   // for TVector3

class TObject;

using namespace o2::TPC;


TVector3 HitDraw::GetVector(TObject* obj)
{
  Point* p = static_cast<Point*>(obj);
  return TVector3(p->GetX(), p->GetY(), p->GetZ());
}


ClassImp(HitDraw)
