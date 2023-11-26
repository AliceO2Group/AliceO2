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

/// \file GeometryTGeo.h
/// \brief Definition of the GeometryTGeo class for ITS3
/// \author felix.schlepper@cern.ch

#ifndef ALICEO2_ITS3_GEOMETRYTGEO_H_
#define ALICEO2_ITS3_GEOMETRYTGEO_H_

#include <array>
#include <string>
#include <vector>
#include "DetectorsBase/GeometryManager.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "DetectorsCommonDataFormats/DetMatrixCache.h"
#include "MathUtils/Utils.h"

namespace o2::its3
{
class GeometryTGeo : public o2::detectors::DetMatrixCache
{
 public:
  using Mat3D = o2::math_utils::Transform3D;
  using DetMatrixCache::getMatrixL2G;
  using DetMatrixCache::getMatrixT2G;
  using DetMatrixCache::getMatrixT2GRot;
  using DetMatrixCache::getMatrixT2L;
  using o2::detectors::DetMatrixCache::fillMatrixCache;

  GeometryTGeo() = default;
  GeometryTGeo(const o2::detectors::DetID& detid) : o2::detectors::DetMatrixCache(detid) {}

  GeometryTGeo(const GeometryTGeo& src) = delete;
  GeometryTGeo(GeometryTGeo&& src) = delete;
  GeometryTGeo& operator=(const GeometryTGeo& geom) = delete;
  ~GeometryTGeo() final = default;

  static GeometryTGeo* Instance()
  {
    // get (create if needed) a unique instance of the object
#ifdef GPUCA_STANDALONE
    return nullptr; // TODO: DR: Obviously wrong, but to make it compile for now
#else
    if (!mInstance) {
      mInstance = std::unique_ptr<GeometryTGeo>(new GeometryTGeo());
    }
    return mInstance.get();
#endif
  }

 private:
#ifndef GPUCA_STANDALONE
  static std::unique_ptr<GeometryTGeo> mInstance; ///< singletone instance
#endif

  ClassDefOverride(GeometryTGeo, 0);
};
} // namespace o2::its3

#endif
