#ifndef AliceO2_TPC_PadPos_H
#define AliceO2_TPC_PadPos_H

namespace AliceO2 {
namespace TPC {
  class PadPos {
    public:
//       PadPos() {}
//       PadPos(const unsigned char row, const unsigned char pad) :
      const unsigned char row() const { return mRow; }
      const unsigned char pad() const { return mPad; }
      
      bool operator==(const PadPos& other) { return (mRow==other.mRow) && (mPad==other.mPad); }
      bool operator<(const PadPos& other) {
        if (mRow<other.mRow) return true;
        if (mRow==other.mRow && mPad<other.mPad) return true;
        return false;
      }

    private:
      unsigned char mRow{0};
      unsigned char mPad{0};
  };
}
}
#endif
