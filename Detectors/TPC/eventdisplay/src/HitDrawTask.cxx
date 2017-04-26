/// \file HitDrawTask.cxx
/// \brief Task to draw MC hits in event display
/// \author Jens Wiechula, Jens.Wiechula@ikf.uni-frankfurt.de
//
#include "TPCEventdisplay/HitDrawTask.h"

#include "TPCSimulation/Point.h"                // for FairMCPoint

#include "TVector3.h"                   // for TVector3

class TObject;

using namespace o2::TPC;

HitDrawTask::HitDrawTask()
{
  // TODO Auto-generated constructor stub

}

HitDrawTask::~HitDrawTask()
{
  // TODO Auto-generated destructor stub
}

TVector3 HitDrawTask::GetVector(TObject* obj)
{
  Point* p = static_cast<Point*>(obj);
  return TVector3(p->GetX(), p->GetY(), p->GetZ());
}


ClassImp(HitDrawTask)
