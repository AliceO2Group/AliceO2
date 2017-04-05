///
/// @file   PadSecPos.h
/// @author Jens Wiechula, Jens.Wiechula@ikf.uni-frankfurt.de
///

/// @brief  Pad and row inside a sector
///
/// This class encapsulates the pad and row inside a sector
/// @see TPCBase/PadSecPos.h
/// @see TPCBase/Sector.h
///
/// origin: TPC
/// @author Jens Wiechula, Jens.Wiechula@ikf.uni-frankfurt.de

#ifndef AliceO2_TPC_PadSecPos_H
#define AliceO2_TPC_PadSecPos_H

#include "TPCBase/Sector.h"
#include "TPCBase/PadPos.h"

namespace o2 {
namespace TPC {
class PadSecPos
{
  public:
    PadSecPos() {};

    PadSecPos(const int sector, const int rowInSector, const int padInRow) : mSector(sector), mPadPos(PadPos(rowInSector, padInRow)) {}
    PadSecPos(const Sector &sec, const PadPos &padPosition) : mSector(sec), mPadPos(padPosition) {}

    Sector getSector() const { return mSector; }

    Sector &getSector() { return mSector; }

    const PadPos& getPadPos() const { return mPadPos; }

    PadPos& getPadPos() { return mPadPos; }

  private:
    Sector mSector{};
    PadPos mPadPos{};
};
}
}
#endif
