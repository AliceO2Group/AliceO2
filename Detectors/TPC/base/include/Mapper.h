#ifndef AliceO2_TPC_Mapper_H
#define AliceO2_TPC_Mapper_H

#include <map>

#include "Defs.h"
#include "PadPos.h"
#include "FECInfo.h"

namespace AliceO2 {
namespace TPC {

class Mapper {
public:

private:
  std::map<PadPos, unsigned short>         mMapPadPosGlobalPad{};  /// mapping pad position to global pad number
  std::map<unsigned short, FECInfo>        mMapGlobalPadFECInfo(); /// map global pad ID to FEC info

};

}
}

#endif
