#ifndef AliceO2_TPC_DigitPos_H
#define AliceO2_TPC_DigitPos_H

#include "Defs.h"
#include "CRU.h"
#include "PadPos.h"
#include "PadSecPos.h"

namespace AliceO2 {
namespace TPC {

class DigitPos {
public:
  DigitPos() {}
  DigitPos(CRU c, PadPos pad) : mCRU(c), mPadPos(pad) {}
  const CRU&  getCru() const { return mCRU; }
  CRU& cru()       { return mCRU; }
  PadPos    getPadPos()       const { return mPadPos; }
  PadPos    getGlobalPadPos() const;
  PadSecPos getPadSecPos()    const;

  PadPos& padPos()       { return mPadPos; }

  bool isValid() const { return mPadPos.isValid(); }
private:
  CRU mCRU{};
  PadPos mPadPos{};          /// Pad position in the local partition coordinates: row starts from 0 for each partition
};

}
}
#endif
