// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUROOTCartesianFwd.h
/// \author David Rohr

#ifndef GPUROOTCARTESIANFWD_H
#define GPUROOTCARTESIANFWD_H

// Standalone forward declarations for Cartesian2D / Cartesian3D / Point2D / Point3D etc.
// To be used on GPU where ROOT is not available.

namespace ROOT
{
namespace Math
{
template <class T, unsigned int D1, unsigned int D2, class R>
class SMatrix;
template <class T, unsigned int D>
class MatRepSym;
template <class T, unsigned int D1, unsigned int D2>
class MatRepStd;
template <class CoordSystem, class Tag>
class PositionVector2D;
template <class CoordSystem, class Tag>
class PositionVector3D;
template <class T>
class Cartesian2D;
template <class T>
class Cartesian3D;
class DefaultCoordinateSystemTag;
} // namespace Math
} // namespace ROOT
template <typename T>
using Point2D = ROOT::Math::PositionVector2D<ROOT::Math::Cartesian2D<T>, ROOT::Math::DefaultCoordinateSystemTag>;
template <typename T>
using Point3D = ROOT::Math::PositionVector3D<ROOT::Math::Cartesian3D<T>, ROOT::Math::DefaultCoordinateSystemTag>;

#endif
