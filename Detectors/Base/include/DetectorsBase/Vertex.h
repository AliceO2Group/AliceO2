// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file Vertex
/// \brief Point in the space with its covariance matrix

#ifndef ALICEO2_BASE_VERTEX
#define ALICEO2_BASE_VERTEX

#include <algorithm>
#include "MathUtils/Cartesian3D.h"


namespace o2
{
namespace Base
{

template<typename F> struct Vertex_t {
  Point3D<F> position;
  std::array<F,6> covariance;
};

typedef Vertex_t<float> Vertex;

}
}

#endif
