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
#ifndef ALICEO2_FD3_GEOMETRYTGEO_H_
#define ALICEO2_FD3_GEOMETRYTGEO_H_

#include <DetectorsCommonDataFormats/DetMatrixCache.h>

#include "DetectorsBase/GeometryManager.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include <Rtypes.h>
#include <TGeoPhysicalNode.h>
#include <vector>
#include <array>
#include <TGeoMatrix.h>
#include <TGeoVolume.h>
#include <TVirtualMC.h>

namespace o2
{
namespace fd3
{

/// FD3 Geometry type
class GeometryTGeo : public o2::detectors::DetMatrixCache
{
 public:
  GeometryTGeo(bool build = false, int loadTrans = 0);

  void Build(int loadTrans);
  void fillMatrixCache(int mask);
  virtual ~GeometryTGeo();

  static GeometryTGeo* Instance();

  void getGlobalPosition(float& x, float& y, float& z);

  static constexpr o2::detectors::DetID::ID getDetID() { return o2::detectors::DetID::FD3; }

 private:
  static std::unique_ptr<o2::fd3::GeometryTGeo> sInstance;

  ClassDefNV(GeometryTGeo, 1);
};
} // namespace fd3
} // namespace o2
#endif
