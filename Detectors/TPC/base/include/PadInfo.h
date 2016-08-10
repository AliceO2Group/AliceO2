#ifndef AliceO2_TPC_PadInfo_H
#define AliceO2_TPC_PadInfo_H

#include "Defs.h"
#include "PadPos.h"
#include "Point2D.h"
#include "FECInfo.h"

namespace AliceO2 {
namespace TPC {

class PadInfo {
  public:


  private:
    GlobalPadNumber mIndex{};       /// unique pad index in sector
    PadPos          mPadPos{};      /// pad row and pad
    PadCentre       mPadCentre{};   /// pad coordingate as seen for sector A04 in global ALICE coordiantes
    FECInfo         mFECInfo{};     /// FEC mapping information

};

}
}

#endif
