// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

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

#include "TPCBase/Constants.h"
#include "TPCBase/Defs.h"
//using namespace AliceO2::TPC;

namespace o2 {
namespace TPC {
//   enum RocType {ISector=0, OSector=1};

class Sector
{
  public:
    // the number of sectors
    static constexpr int MAXSECTOR=Constants::MAXSECTOR;

    /// constructor
    Sector() {}

    /// construction
    /// @param [in] sec sector number
    Sector(unsigned char sec) : mSector(sec % MAXSECTOR) { ; }

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
      mLoop = ++mSector >= MAXSECTOR;
      mSector %= MAXSECTOR;
      return mLoop;
    }

    /// int return operator to use similar as integer
    /// \return sector number
    operator int() const { return int(mSector); }

    /// assignment operator with int
    Sector& operator=(int sector) { mSector=sector%MAXSECTOR; return *this; }

    unsigned char getSector() const { return mSector; }

    Side side() const { return (mSector < MAXSECTOR / 2) ? Side::A : Side::C; }

    bool looped() const { return mLoop; }

    double phi() const { return (mSector % SECTORSPERSIDE) * SECPHIWIDTH + SECPHIWIDTH / 2.; }

    // helper function to retrieve a TPC sector given cartesian coordinates
    template <typename T>
    static int ToSector(T x, T y, T z) {
      static const T invangle(static_cast<T>(180)/static_cast<T>(M_PI*20.)); // the angle describing one sector
      // force positive angle for conversion
      auto s = (std::atan2(-y,-x) + static_cast<T>(M_PI))*invangle;
      // detect if on C size
      if (z<static_cast<T>(0.)) {
        s += Sector::MAXSECTOR/2;
      }
      return s;
    }

  private:
    unsigned char mSector{};    /// Sector representation 0-MAXSECTOR
    bool mLoop{};   /// if operator execution resulted in looping
};
}
}

#endif
