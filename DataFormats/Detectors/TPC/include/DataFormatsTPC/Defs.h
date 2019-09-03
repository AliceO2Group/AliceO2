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
/// @file   Defs.h
/// @author Jens Wiechula, Jens.Wiechula@ikf.uni-frankfurt.de
///

/// @brief  Global TPC definitions and constants

#ifndef AliceO2_TPC_Defs_H
#define AliceO2_TPC_Defs_H

#include <cmath>

#include "MathUtils/Cartesian2D.h"
#include "MathUtils/Cartesian3D.h"

namespace o2
{
namespace tpc
{

/// TPC readout sidE
enum Side { A = 0,
            C = 1,
            UNDEFINED = 2 };
//   enum class Side {A=0, C=1};
//  Problem with root cint. does not seem to support enum class ...
constexpr unsigned char SECTORSPERSIDE = 18;
constexpr unsigned char SIDES = 2;

constexpr double PI = 3.14159265358979323846;
constexpr double TWOPI = 2. * PI;
constexpr double SECPHIWIDTH = TWOPI / 18.;

/// TPC ROC types
enum RocType { IROC = 0,
               OROC = 1 };
// enum class RocType {IROC=0, OROC=1};

/// TPC GEM stack types
enum GEMstack { IROCgem = 0,
                OROC1gem = 1,
                OROC2gem = 2,
                OROC3gem = 3 };
constexpr unsigned short GEMSTACKSPERSECTOR = 4;

/// Definition of the different pad subsets
enum class PadSubset : char {
  ROC,       ///< ROCs (up to 72)
  Partition, ///< Partitions (up to 36*5)
  Region     ///< Regions (up to 36*10)
};
// default point definitions for PointND, PointNDlocal, PointNDglobal are in
// MathUtils/CartesianND.h

/// Pad centres as 2D float
// For some reason cling does not like the nested using statement, typedef works ...
typedef Point2D<float> PadCentre;
typedef Point2D<float> GlobalPosition2D;
typedef Point2D<float> LocalPosition2D;
typedef Point3D<float> GlobalPosition3D;
typedef Point3D<float> LocalPosition3D;

/// global pad number
typedef unsigned short GlobalPadNumber;

/// global time bin
typedef unsigned int TimeBin;

// GlobalPosition3D LocalToGlobal(const LocalPosition3D pos, const float alpha)
// {
//   const double cs=cos(alpha), sn=sin(alpha);
//   return GlobalPosition3D(pos.X()*cs-pos.Y()*sn,pos.X()*sn+pos.Y()*cs,pos.Z());
// }

// LocalPosition3D GlobalToLocal(const GlobalPosition3D& pos, const float alpha)
// {
//   const double cs=cos(-alpha), sn=sin(-alpha);
//   return LocalPosition3D(pos.X()*cs-pos.Y()*sn,pos.X()*sn+pos.Y()*cs,pos.Z());
// }

/**
 * simple class to allow for range for loops over enums
 * e.g. for (auto &side : Enum<Sides>() ) { cout << side << endl; }
 * taken from http://stackoverflow.com/questions/8498300/allow-for-range-based-for-with-enum-classes
 */

template <typename T>
class Enum
{
 public:
  class Iterator
  {
   public:
    Iterator(int value) : m_value(value) {}

    T operator*() const { return (T)m_value; }

    void operator++() { ++m_value; }

    bool operator!=(Iterator rhs) { return m_value != rhs.m_value; }

   private:
    int m_value;
  };
};

template <typename T>
typename Enum<T>::Iterator begin(Enum<T>)
{
  return typename Enum<T>::Iterator((int)T::First);
}

template <typename T>
typename Enum<T>::Iterator end(Enum<T>)
{
  return typename Enum<T>::Iterator(((int)T::Last) + 1);
}
} // namespace tpc
} // namespace o2

#endif
