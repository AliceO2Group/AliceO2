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

/// \file  testTrackResiduals.cxx
/// \brief perform functionality tests related to TPC average space charge distortion correction maps (currently only binning)
///
/// \author  Ole Schmidt, ole.schmidt@cern.ch

#define BOOST_TEST_MODULE Test TrackResidualsTest class
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include "SpacePoints/TrackResiduals.h"

namespace o2::tpc
{

// testing default binning and custom binning
BOOST_AUTO_TEST_CASE(TrackResidualsBinning_test)
{
  TrackResiduals residUniform;
  TrackResiduals residCustom;
  residUniform.initBinning();

  // fill uniform binning in Y/X manually
  std::vector<float> binY2X;
  for (int iBin = 0; iBin <= residUniform.getNY2XBins(); ++iBin) {
    binY2X.push_back(-1. + 2. / residUniform.getNY2XBins() * iBin);
  }
  residCustom.setY2XBinning(binY2X);
  // fill uniform binning in Z/X manually
  std::vector<float> binZ2X;
  for (int iBin = 0; iBin <= residUniform.getNZ2XBins(); ++iBin) {
    binZ2X.push_back(1. / residUniform.getNZ2XBins() * iBin);
  }
  residCustom.setZ2XBinning(binZ2X);
  residCustom.initBinning();

  float x, p, z, xRef, pRef, zRef;
  for (int ix = 0; ix < residUniform.getNXBins(); ++ix) {
    for (int ip = 0; ip < residUniform.getNY2XBins(); ++ip) {
      for (int iz = 0; iz < residUniform.getNZ2XBins(); ++iz) {
        residUniform.getVoxelCoordinates(0, ix, ip, iz, xRef, pRef, zRef);
        residCustom.getVoxelCoordinates(0, ix, ip, iz, x, p, z);
        BOOST_CHECK_SMALL(x - xRef, 1e-6f);
        BOOST_CHECK_SMALL(p - pRef, 1e-6f);
        BOOST_CHECK_SMALL(z - zRef, 1e-6f);
      }
    }
  }
}

} // namespace o2::tpc
