/// \file HitDrawTask.h
/// \brief Task to draw MC hits in event display
/// \author Jens Wiechula, Jens.Wiechula@ikf.uni-frankfurt.de

#ifndef FAIRMCPOINTDRAW_H_
#define FAIRMCPOINTDRAW_H_

#include "FairPointSetDraw.h"           // for FairPointSetDraw

#include "Rtypes.h"                     // for HitDrawTask::Class, etc

class TObject;
class TVector3;

namespace o2 {
namespace TPC {

/// \class HitDrawTask
/// This class is required to draw MC hits in the event display
/// The generic class can not be used, since the hits don't inherit from
/// FairPoint
class HitDrawTask: public FairPointSetDraw
{
  public:
    HitDrawTask();
    HitDrawTask(const char* name, Color_t color ,Style_t mstyle, Int_t iVerbose = 1):FairPointSetDraw(name, color, mstyle, iVerbose) {};
    virtual ~HitDrawTask();

  protected:
    TVector3 GetVector(TObject* obj);

    ClassDef(HitDrawTask,0);
};

};
};
#endif /* FAIRMCPOINTDRAW_H_ */
