///
/// @file   Sector.h
/// @author Jens Wiechula, Jens.Wiechula@ikf.uni-frankfurt.de
///

/// @brief  Sector type
///
/// This class represents a Sector
/// Sectors are counted
/// from 0-17 (A-Side)
/// and 18-35 (C-Side)
///
/// origin: TPC
/// @author Jens Wiechula, Jens.Wiechula@ikf.uni-frankfurt.de

#ifndef AliceO2_TPC_Sector_H
#define AliceO2_TPC_Sector_H

#include "TPCBase/Defs.h"
//using namespace AliceO2::TPC;

namespace o2 {
namespace TPC {
//   enum RocType {ISector=0, OSector=1};

class Sector
{
  public:
    enum
    {
        MaxSector = 36
    };

    /// constructor
    Sector() {}

    /// construction
    /// @param [in] sec sector number
    Sector(unsigned char sec) : mSector(sec % MaxSector) { ; }

    /// comparison operator
    bool operator==(const Sector &other) const { return mSector == other.mSector; }

    /// unequal operator
    bool operator!=(const Sector &other) const { return mSector != other.mSector; }

    /// smaller operator
    bool operator<(const Sector &other) const { return mSector < other.mSector; }

    /// increment operator
    /// This operator can be used to iterate over all sectors e.g.
    /// Sector sec;
    /// while (++sec) { std::cout << "Sector: " << sec.getSector() << std::endl; }
    bool operator++()
    {
      mLoop = ++mSector >= MaxSector;
      mSector %= MaxSector;
      return mLoop;
    }

    unsigned char getSector() const { return mSector; }

    Side side() const { return (mSector < MaxSector / 2) ? Side::A : Side::C; }

    bool looped() const { return mLoop; }

    const double phi() const { return (mSector % SECTORSPERSIDE) * SECPHIWIDTH + SECPHIWIDTH / 2.; }

  private:
    unsigned char mSector{};    /// Sector representation 0-MaxSector
    bool mLoop{};   /// if operator execution resulted in looping
};
}
}

#endif
