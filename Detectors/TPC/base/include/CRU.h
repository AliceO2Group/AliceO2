#ifndef AliceO2_TPC_CRU_H
#define AliceO2_TPC_CRU_H

#include "Defs.h"
#include "Sector.h"
#include "ROC.h"
//using namespace AliceO2::TPC;

namespace AliceO2 {
namespace TPC {
//   enum RocType {ICRU=0, OCRU=1};

  class CRU {
    public:
      enum { CRUperPartition=2, CRUperIROC=4, CRUperSector=10, MaxCRU=360 };
      CRU(){}
      CRU(unsigned short cru):mCRU(cru%MaxCRU) {}
      CRU(const CRU& cru):mCRU(cru.mCRU) {}
      CRU(const Sector& sec, const unsigned char partitionNum) : mCRU(sec.getSector()*CRUperSector+partitionNum) {}
//       CRU(RocType t, Side s, unsigned char r):mCRU( (s==Side::A)*18 + (t==RocType::OCRU)*18 + r%18 ) {}
//       CRU(Side t) {}

      CRU& operator= (const CRU& other) { mCRU=other.mCRU; return *this; }
      CRU& operator= (const unsigned short cru) { mCRU=cru; return *this; }

      bool    operator==(const CRU& other)  { return mCRU==other.mCRU; }
      bool    operator!=(const CRU& other)  { return mCRU!=other.mCRU; }
      bool    operator< (const CRU& other)  { return mCRU<other.mCRU; }
      bool    operator++()                  { mLoop=++mCRU>=MaxCRU; mCRU%=MaxCRU; return mLoop; }

      unsigned short number()   const { return mCRU; }
      const ROC roc()           const { return ROC(sector(), rocType()); }
      Side    side()            const { return (mCRU<(MaxCRU/SIDES))?Side::A:Side::C; }
      unsigned char partition() const { return (mCRU%CRUperSector)/CRUperPartition; }
      unsigned char region()    const { return (mCRU%CRUperSector); }
      const Sector  sector()    const { return Sector(mCRU/CRUperSector); }
      const RocType rocType()   const { return mCRU%CRUperSector < CRUperIROC ? RocType::IROC : RocType::OROC; }

      bool    looped()    const { return mLoop; }


    private:
      unsigned short mCRU{};    /// CRU representation 0-MaxCRU
      bool           mLoop{};   /// if operator execution resulted in looping
  };
}
}


#endif
