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

#ifndef ALICEO2_FIT_GEOMETRY_H_
#define ALICEO2_FIT_GEOMETRY_H_
////////////////////////////////////////////////
// Full geomrtry  hits classes for detector: FIT    //
////////////////////////////////////////////////
#include "DetectorsBase/GeometryManager.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "Framework/Logger.h"
#include <Rtypes.h>
#include <TGeoPhysicalNode.h>
#include <TVector3.h>
#include <string>

class TGeoPNEntry;

namespace o2
{
namespace ft0
{
// FIT is not tracking detector, Geometry could be used in future but not now. So just simple settings
class Geometry
{
 public:
  ///
  /// Default constructor.
  /// It must be kept public for root persistency purposes,
  /// but should never be called by the outside world
  Geometry();
  /// Copy constructor.
  ///
  // Geometry(const Geometry& geom) : mMCP{ { 0, 0, 0 }; }

  ///
  TVector3 centerMCP(int imcp) { return mMCP[imcp]; }
  TVector3 tiltMCP(int imcp) { return mAngles[imcp]; }

  static constexpr int Nchannels = 229;                     // number of LUT channels
  static constexpr int Nsensors = 208;                      // number of channels
  static constexpr int NCellsA = 24;                        // number of radiatiors on A side
  static constexpr int NCellsC = 28;                        // number of radiatiors on C side
  static constexpr float ZdetA = 335.5;                     // number of radiatiors on A side
  static constexpr float ZdetC = 82;                        // number of radiatiors on C side
  static constexpr float ChannelWidth = 13.02;              // channel width in ps
  static constexpr float ChannelWidthInverse = 0.076804916; // channel width in ps inverse
  static constexpr o2::detectors::DetID::ID getDetID() { return o2::detectors::DetID::FT0; }
  void setAsideModules();
  void setCsideModules();
  TGeoPNEntry* getPNEntry(int index) const
  {
    /// Get a pointer to the TGeoPNEntry of a chip identified by 'index'
    /// Returns NULL in case of invalid index, missing TGeoManager or invalid symbolic name
    return o2::base::GeometryManager::getPNEntry(getDetID(), index);
  }

 private:
  TVector3 mMCP[52];
  TVector3 mAngles[28];

  ClassDefNV(Geometry, 2);
};
} // namespace ft0
} // namespace o2
#endif
