#ifndef AliceO2_TPC_PadInfo_H
#define AliceO2_TPC_PadInfo_H

#include "TPCBase/Defs.h"
#include "TPCBase/PadPos.h"
#include "TPCBase/Point2D.h"
#include "TPCBase/FECInfo.h"

namespace o2 {
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
