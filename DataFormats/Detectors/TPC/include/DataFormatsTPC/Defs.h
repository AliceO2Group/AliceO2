// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
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

#ifndef GPUCA_GPUCODE_DEVICE
#include <cmath>
#endif

#include "GPUROOTCartesianFwd.h"
#ifndef GPUCA_ALIGPUCODE
#include "MathUtils/Cartesian.h"
#endif

namespace o2::tpc
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
constexpr unsigned short GEMSPERSTACK = 4;
constexpr unsigned short GEMSTACKSPERSIDE = GEMSTACKSPERSECTOR * SECTORSPERSIDE;
constexpr unsigned short GEMSTACKS = GEMSTACKSPERSECTOR * SECTORSPERSIDE * SIDES;

/// Definition of the different pad subsets
enum class PadSubset : char {
  ROC,       ///< ROCs (up to 72)
  Partition, ///< Partitions (up to 36*5)
  Region     ///< Regions (up to 36*10)
};

// TPC dE/dx charge types
enum ChargeType {
  Max = 0,
  Tot = 1
};
constexpr unsigned short CHARGETYPES = 2;

/// GEM stack identification
struct StackID {
  int sector{};
  GEMstack type{};

  /// Single number identification for the stacks
  GPUdi() int getIndex() const
  {
    return sector + type * SECTORSPERSIDE * SIDES;
  }
  GPUdi() void setIndex(int index)
  {
    sector = index % (SECTORSPERSIDE * SIDES);
    type = static_cast<GEMstack>((index / (SECTORSPERSIDE * SIDES)) % GEMSTACKSPERSECTOR);
  }
};

/// Statistics type
enum class StatisticsType {
  GausFit,     ///< Use slow gaus fit (better fit stability)
  GausFitFast, ///< Use fast gaus fit (less accurate error treatment)
  MeanStdDev   ///< Use mean and standard deviation
};

enum class PadFlags : unsigned short {
  flagGoodPad = 1 << 0,      ///< flag for a good pad binary 0001
  flagDeadPad = 1 << 1,      ///< flag for a dead pad binary 0010
  flagUnknownPad = 1 << 2,   ///< flag for unknown status binary 0100
  flagSaturatedPad = 1 << 3, ///< flag for saturated status binary 0100
  flagHighPad = 1 << 4,      ///< flag for pad with extremly high IDC value
  flagLowPad = 1 << 5,       ///< flag for pad with extremly low IDC value
  flagSkip = 1 << 6,         ///< flag for defining a pad which is just ignored during the calculation of I1 and IDCDelta
  flagFEC = 1 << 7,          ///< flag for a whole masked FEC
  flagNeighbour = 1 << 8,    ///< flag if n neighbouring pads are outlier
  flagAllNoneGood = flagDeadPad | flagUnknownPad | flagSaturatedPad | flagHighPad | flagLowPad | flagSkip | flagFEC | flagNeighbour,
};

inline PadFlags operator&(PadFlags a, PadFlags b) { return static_cast<PadFlags>(static_cast<int>(a) & static_cast<int>(b)); }
inline PadFlags operator~(PadFlags a) { return static_cast<PadFlags>(~static_cast<int>(a)); }
inline PadFlags operator|(PadFlags a, PadFlags b) { return static_cast<PadFlags>(static_cast<int>(a) | static_cast<int>(b)); }

// default point definitions for PointND, PointNDlocal, PointNDglobal are in
// MathUtils/CartesianND.h

/// Pad centres as 2D float
// For some reason cling does not like the nested using statement, typedef works ...
typedef math_utils::Point2D<float> PadCentre;
typedef math_utils::Point2D<float> GlobalPosition2D;
typedef math_utils::Point2D<float> LocalPosition2D;
typedef math_utils::Point3D<float> GlobalPosition3D;
typedef math_utils::Point3D<float> LocalPosition3D;

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
} // namespace o2::tpc

#endif
