// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_FIT_GEOMETRY_H_
#define ALICEO2_FIT_GEOMETRY_H_
////////////////////////////////////////////////
// Full geomrtry  hits classes for detector: FIT    //
////////////////////////////////////////////////

#include <Rtypes.h>
#include <TVector3.h>

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

  static constexpr int Nchannels = 229;                     // number od channels
  static constexpr int NCellsA = 24;                        // number of radiatiors on A side
  static constexpr int NCellsC = 28;                        // number of radiatiors on C side
  static constexpr float ZdetA = 335.5;                     // number of radiatiors on A side
  static constexpr float ZdetC = 82;                        // number of radiatiors on C side
  static constexpr float ChannelWidth = 13.02;              // channel width in ps
  static constexpr float ChannelWidthInverse = 0.076804916; // channel width in ps inverse

 private:
  TVector3 mMCP[52];

  ClassDefNV(Geometry, 1);
};
} // namespace ft0
} // namespace o2
#endif
