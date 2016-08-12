#include "TPCbase/DigitPos.h"
#include "TPCbase/Mapper.h"

namespace AliceO2 {
namespace TPC {


PadPos    DigitPos::getGlobalPadPos() const
{
  const Mapper &mapper = Mapper::instance();
  const PadRegionInfo &p=mapper.getPadRegionInfo(mCRU.region());

  return PadPos(mPadPos.getRow()+p.getGlobalRowOffset(), mPadPos.getPad());
}


PadSecPos DigitPos::getPadSecPos() const
{
  return PadSecPos(mCRU.sector(), getGlobalPadPos());
}

}
}
