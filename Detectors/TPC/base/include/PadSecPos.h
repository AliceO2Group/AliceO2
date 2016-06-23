#ifndef AliceO2_TPC_PadSecPos_H
#define AliceO2_TPC_PadSecPos_H

#include "ROC.h"
#include "PadPos.h"

namespace AliceO2 {
namespace TPC {
  class PadSecPos {
    public:
      ROC  sector() const { return mROC; }
      ROC& sector()       { return mROC; }
      PadPos  padPos() const { return mPad; }
      PadPos& padPos()       { return mPad; }
    private:
      ROC mROC{};
      PadPos mPad{};
  };
}
}
#endif
