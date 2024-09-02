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
#ifndef ALICEO2_FVD_GEOMETRYTGEO_H_
#define ALICEO2_FVD_GEOMETRYTGEO_H_

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
namespace fvd
{

/// FVD Geometry type
class GeometryTGeo : public o2::detectors::DetMatrixCache
{
 public:
  GeometryTGeo();

  void Build() const;
  void fillMatrixCache(int mask);
  virtual ~GeometryTGeo();

  static GeometryTGeo* Instance();

  void getGlobalPosition(float& x, float& y, float& z);
  static constexpr int getNumberOfReadoutChannelsA() { return sNumberOfReadoutChannelsA; };
  static constexpr int getNumberOfReadoutChannelsC() { return sNumberOfReadoutChannelsC; };
  static constexpr int getNumberOfReadoutChannels()  { return sNumberOfReadoutChannelsA + sNumberOfCellsC; }

  static constexpr o2::detectors::DetID::ID getDetID() { return o2::detectors::DetID::FVD; }

  int getCellId(int nmod, int nring, int nsec) const;
  int getCurrentCellId(const TVirtualMC* fMC) const;

 private:

  static constexpr float sDzScintillator = 4; 
  static constexpr float sXGlobal = 0;
  static constexpr float sYGlobal = 0;
  static constexpr float sZGlobalA =  1700. - sDzScintillator/2;
  static constexpr float sZGlobalC = -1950. + sDzScintillator/2;

  static constexpr int sNumberOfCellSectors = 8;
  static constexpr int sNumberOfCellRingsA  = 5; // 3
  static constexpr int sNumberOfCellRingsC  = 6; 
  static constexpr int sNumberOfCellsA = sNumberOfCellRingsA * sNumberOfCellSectors;
  static constexpr int sNumberOfCellsC = sNumberOfCellRingsC * sNumberOfCellSectors;
  static constexpr int sNumberOfReadoutChannelsA = sNumberOfCellsA;
  static constexpr int sNumberOfReadoutChannelsC = sNumberOfCellsC;

  static constexpr float sCellRingRadiiA[sNumberOfCellRingsA + 1] = {3., 14.8, 26.6, 38.4, 50.2, 62.};
  static constexpr float sCellRingRadiiC[sNumberOfCellRingsC + 1] = {3.5, 17., 30.5, 44., 57.5, 71.};

  static std::unique_ptr<o2::fvd::GeometryTGeo> sInstance;

  TGeoVolumeAssembly* buildModuleA() const;
  TGeoVolumeAssembly* buildModuleC() const;

  ClassDefNV(GeometryTGeo, 1);
};
} // namespace fvd
} // namespace o2
#endif
