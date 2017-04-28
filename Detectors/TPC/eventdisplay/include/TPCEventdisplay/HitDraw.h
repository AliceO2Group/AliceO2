#ifndef ALICEO2_TPC_HITDRAW_H_
#define ALICEO2_TPC_HITDRAW_H_

#include "FairPointSetDraw.h"           // for FairPointSetDraw

class TObject;
class TVector3;

namespace o2 {
namespace TPC {

class HitDraw: public FairPointSetDraw
{
  public:
    HitDraw() = default;
    HitDraw(const char* name, Color_t color ,Style_t mstyle, Int_t iVerbose = 1):FairPointSetDraw(name, color, mstyle, iVerbose) {};
    ~HitDraw() override = default;

  protected:
    TVector3 GetVector(TObject* obj) override;

    ClassDefOverride(HitDraw,0);
};

}
}

#endif 
