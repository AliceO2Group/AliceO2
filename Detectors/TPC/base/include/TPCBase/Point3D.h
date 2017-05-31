// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See https://alice-o2.web.cern.ch/ for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

///
/// @file   Point3D.h
/// @author Jens Wiechula, Jens.Wiechula@ikf.uni-frankfurt.de
///

/// @brief  Simple templated 3D point
///
/// origin: TPC
/// @author Jens Wiechula, Jens.Wiechula@ikf.uni-frankfurt.de

#ifndef AliceO2_TPC_Point3D_H
#define AliceO2_TPC_Point3D_H

namespace o2 {
namespace TPC {

template < class T >
class Point3D {
  public:
    Point3D() = default;
    Point3D(const T &x, const T &y, const T&z) : mX(x), mY(y), mZ(z) {}

    const T& getX() const { return mX; }
    const T& getY() const { return mY; }
    const T& getZ() const { return mZ; }

  private:
    T mX{};  /// x-position
    T mY{};  /// y-position
    T mZ{};  /// y-position
};

}
}

#endif
