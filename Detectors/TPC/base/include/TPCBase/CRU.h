// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef AliceO2_TPC_CRU_H
#define AliceO2_TPC_CRU_H

#include "DataFormatsTPC/Defs.h"
#include "TPCBase/Sector.h"
#include "TPCBase/ROC.h"

namespace o2
{
namespace tpc
{
//   enum RocType {ICRU=0, OCRU=1};

class CRU
{
 public:
  enum { CRUperPartition = 2,
         CRUperIROC = 4,
         CRUperSector = 10,
         MaxCRU = 360 };
  CRU() = default;
  CRU(unsigned short cru) : mCRU(cru % MaxCRU) {}
  CRU(const CRU& cru) : mCRU(cru.mCRU) {}
  CRU(const Sector& sec, const unsigned char partitionNum) : mCRU(sec.getSector() * CRUperSector + partitionNum) {}
  //       CRU(RocType t, Side s, unsigned char r):mCRU( (s==Side::A)*18 + (t==RocType::OCRU)*18 + r%18 ) {}
  //       CRU(Side t) {}

  CRU& operator=(const CRU& other)
  {
    mCRU = other.mCRU;
    return *this;
  }
  CRU& operator=(const unsigned short cru)
  {
    mCRU = cru;
    return *this;
  }

  bool operator==(const CRU& other) const { return mCRU == other.mCRU; }
  bool operator!=(const CRU& other) const { return mCRU != other.mCRU; }
  bool operator<(const CRU& other) const { return mCRU < other.mCRU; }
  bool operator++()
  {
    mLoop = ++mCRU >= MaxCRU;
    mCRU %= MaxCRU;
    return mLoop;
  }

  unsigned short number() const { return mCRU; }
  const ROC roc() const { return ROC(sector(), rocType()); }
  Side side() const { return (mCRU < (MaxCRU / SIDES)) ? Side::A : Side::C; }
  unsigned char partition() const { return (mCRU % CRUperSector) / CRUperPartition; }
  unsigned char region() const { return (mCRU % CRUperSector); }
  const Sector sector() const { return Sector(mCRU / CRUperSector); }
  RocType rocType() const { return mCRU % CRUperSector < CRUperIROC ? RocType::IROC : RocType::OROC; }
  GEMstack gemStack() const;

  /// int return operator to use similar as integer
  /// \return cru number
  operator int() const { return int(mCRU); }

  bool looped() const { return mLoop; }

 private:
  unsigned short mCRU{}; /// CRU representation 0-MaxCRU
  bool mLoop{};          /// if operator execution resulted in looping
};

inline GEMstack CRU::gemStack() const
{
  const int reg = int(region());

  if (reg < CRUperIROC)
    return GEMstack::IROCgem;
  else if (reg - CRUperIROC < CRUperPartition)
    return GEMstack::OROC1gem;
  else if (reg - CRUperIROC - CRUperPartition < CRUperPartition)
    return GEMstack::OROC2gem;
  else
    return GEMstack::OROC3gem;
}
} // namespace tpc
} // namespace o2

#endif
