// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file Geometry.h
/// \brief Base definition of FIT-FDD geometry.
///
/// \author Michal Broz

#ifndef ALICEO2_FDD_GEOMETRY_H_
#define ALICEO2_FDD_GEOMETRY_H_

#include <vector>
#include <TGeoMatrix.h>
#include <TGeoVolume.h>

namespace o2
{
namespace fdd
{
/// FIT-FDD Geometry
class Geometry
{
 public:
  enum EGeoType {
    eUninitilized,
    eDummy,
    eOnlySensitive,
    eFull
  };

  //Default constructor.
  Geometry() { mGeometryType = eUninitilized; };
  // Standard constructor
  Geometry(EGeoType initType);
  Geometry(const Geometry& geom);

 private:
  void buildGeometry();

  int mGeometryType;

  ClassDefNV(Geometry, 1);
};
} // namespace fdd
} // namespace o2

#endif
