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
#include "FT0Base/Constants.h"
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

  //  static constexpr int Nchannels = o2::ft0::Constants::sNCHANNELS_PM; // number of PM channels
  static constexpr int Nchannels = 208;                     // number of sensors
  static constexpr int Nsensors = 208;                      // number of sensors
  static constexpr int NCellsA = 24;                        // number of radiatiors on A side
  static constexpr int NCellsC = 28;                        // number of radiatiors on C side
  static constexpr float ZdetA = 335.5;                     // Z position of center volume  on A side
  static constexpr float ZdetC = 82;                        //  Z position of center volume on C side
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

  // NEW:
  // Calculate the coordinates of all the channels.
  void calculateChannelCenter();
  // Get the coordinates of the center of the channel channelId.
  TVector3 getChannelCenter(UInt_t channelId) const { return mChannelCenter[channelId]; }

 private:
  TVector3 mMCP[52];
  TVector3 mAngles[28];
  std::array<TVector3, Nchannels> mChannelCenter; ///< NEW: Center of each channel in FT0 A (96) and C (112) combined.

  // Convert the local ordering of the channels to the official one and apply it to the channel map.
  // localChannelOrder[local channel index] = official channel.
  Int_t localChannelOrder[Nchannels] = {
    58, 56, 59, 57, 54, 52, 55, 53, 50, 49, 51, 48, 47, 45, 46, 44, 43, 42, 41, 40,
    61, 60, 63, 62, 14, 12, 15, 13, 10, 9, 11, 8, 7, 6, 5, 4, 39, 38, 37, 36,
    65, 64, 66, 67, 17, 16, 18, 19, 3, 2, 0, 1, 35, 34, 32, 33,
    68, 69, 70, 71, 20, 21, 22, 23, 24, 27, 25, 26, 29, 31, 28, 30, 94, 95, 92, 93,
    72, 73, 74, 75, 76, 78, 77, 79, 80, 83, 81, 82, 85, 87, 84, 86, 89, 91, 88, 90,
    173, 172, 175, 174, 206, 207, 204, 205, 169, 168, 171, 170, 202, 203, 200, 201,
    117, 116, 119, 118, 142, 143, 140, 141, 114, 112, 115, 113, 137, 139, 136, 138,
    166, 164, 167, 165, 197, 199, 196, 198, 110, 108, 111, 109, 133, 135, 132, 134,
    162, 160, 163, 161, 193, 195, 192, 194, 107, 105, 106, 104, 128, 130, 129, 131,
    159, 157, 158, 156, 188, 190, 189, 191, 99, 98, 97, 96, 120, 121, 122, 123,
    103, 102, 101, 100, 124, 125, 126, 127, 155, 153, 154, 152, 184, 186, 185, 187,
    147, 146, 145, 144, 176, 177, 178, 179, 151, 150, 149, 148, 180, 181, 182, 183};

  ClassDefNV(Geometry, 3);
};
} // namespace ft0
} // namespace o2
#endif
