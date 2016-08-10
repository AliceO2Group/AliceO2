#ifndef AliceO2_TPC_Sector_H
#define AliceO2_TPC_Sector_H

#include "Defs.h"
//using namespace AliceO2::TPC;

namespace AliceO2 {
namespace TPC {
//   enum RocType {ISector=0, OSector=1};

  class Sector {
    public:
      enum { MaxSector=36 };
      Sector(){}
      Sector(unsigned char sec):mSector(sec%MaxSector){;}
      Sector(const Sector& sec):mSector(sec.mSector){;}
//       Sector(RocType t, Side s, unsigned char r):mSector( (s==Side::A)*18 + (t==RocType::OSector)*18 + r%18 ) {}
//       Sector(Side t) {}

      Sector& operator= (const Sector& other) { mSector=other.mSector; return *this; }

      bool    operator==(const Sector& other) { return mSector==other.mSector; }
      bool    operator!=(const Sector& other) { return mSector!=other.mSector; }
      bool    operator< (const Sector& other)  { return mSector<other.mSector; }
      bool    operator++()                    { mLoop=++mSector>=MaxSector; mSector%=MaxSector; return mLoop; }

      unsigned char getSector() const { return mSector; }

      Side    side()      const { return (mSector<MaxSector/2)? Side::A : Side::C; }
      bool    looped()    const { return mLoop; }

      const double phi()  const { return (mSector%SECTORSPERSIDE)*SECPHIWIDTH+SECPHIWIDTH/2.; }

    private:
      unsigned char mSector{};    /// Sector representation 0-MaxSector
      bool          mLoop{};   /// if operator execution resulted in looping
  };
}
}


#endif
