// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   Defs.h
/// @brief Common definitions for 2D coordinates
/// @author Jens Wiechula, Jens.Wiechula@ikf.uni-frankfurt.de

#ifndef ALICEO2_CARTESIAN2D_H
#define ALICEO2_CARTESIAN2D_H

#include "Math/GenVector/DisplacementVector2D.h"
#include "Math/GenVector/PositionVector2D.h"

template <typename T>
using Point2D = ROOT::Math::PositionVector2D<ROOT::Math::Cartesian2D<T>, ROOT::Math::DefaultCoordinateSystemTag>;

template <typename T>
using Vector2D = ROOT::Math::DisplacementVector2D<ROOT::Math::Cartesian2D<T>, ROOT::Math::DefaultCoordinateSystemTag>;

#endif
