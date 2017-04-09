#ifndef AliceO2_TPC_DigitPos_H
#define AliceO2_TPC_DigitPos_H

#include "TPCBase/Defs.h"
#include "TPCBase/CRU.h"
#include "TPCBase/PadPos.h"
#include "TPCBase/PadSecPos.h"

namespace o2 {
namespace TPC {

class DigitPos {
public:
  DigitPos() {}
  DigitPos(CRU c, PadPos pad) : mCRU(c), mPadPos(pad) {}
  const CRU&  getCRU() const { return mCRU; }
  CRU& cru()       { return mCRU; }
  PadPos    getPadPos()       const { return mPadPos; }
  PadPos    getGlobalPadPos() const;
  PadSecPos getPadSecPos()    const;

  PadPos& padPos()       { return mPadPos; }

  bool isValid() const { return mPadPos.isValid(); }

  bool    operator==(const DigitPos& other)  const { return (mCRU==other.mCRU) && (mPadPos==other.mPadPos); }
  bool    operator!=(const DigitPos& other)  const { return (mCRU!=other.mCRU) || (mPadPos!=other.mPadPos); }
  bool    operator< (const DigitPos& other)  const { return (mCRU <other.mCRU) && (mPadPos <other.mPadPos); }

private:
  CRU mCRU{};
  PadPos mPadPos{};          /// Pad position in the local partition coordinates: row starts from 0 for each partition
};

}
}
#endif
