// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file testTPCMapper.cxx
/// \brief This task tests the mapper function
/// \author Andi Mathis, TU München, andreas.mathis@ph.tum.de

#define BOOST_TEST_MODULE Test TPC Mapper
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include "TPCBase/Mapper.h"

namespace o2
{
namespace tpc
{

/// \brief Test the mapping
/// Complex test, in which the pad plane mapping is read in and compared to the outcome of the filtering of the pad
/// coordinates (x,y) through the mapper
/// Tests the most important functions
/// - findDigitPosFromGlobalPosition(GlobalPosition3D)
/// - getPadRegionInfo(Region)
/// - LocalPosition3D
/// - LocalToGlobal(LocalPosition3D, Sector)
BOOST_AUTO_TEST_CASE(Mapper_complex_test1)
{
  Mapper& mapper = Mapper::instance();
  std::vector<std::string> mappingTables{{"/TABLE-IROC.txt", "/TABLE-OROC1.txt", "/TABLE-OROC2.txt", "/TABLE-OROC3.txt"}};
  std::vector<int> nPads{{0, mapper.getPadsInIROC(), mapper.getPadsInOROC1(), mapper.getPadsInOROC2()}};
  std::vector<float> signTest{{-1.f, 1.f}};

  // ===| Input variables |=====================================================
  // pad info
  GlobalPadNumber padIndex;
  unsigned int padRow, pad;
  float xPos, yPos;

  // pad plane info
  unsigned int connector, pin, partion, region;

  // FEC info
  unsigned int fecIndex, fecConnector, fecChannel, sampaChip, sampaChannel;

  for (int i = 0; i < mappingTables.size(); ++i) {
    std::string line;
    const char* aliceO2env = std::getenv("O2_ROOT");
    std::string inputDir = " ";
    if (aliceO2env)
      inputDir = aliceO2env;
    inputDir += "/share/Detectors/TPC/files";

    std::string file = inputDir + mappingTables[i];
    std::ifstream infile(file, std::ifstream::in);
    while (std::getline(infile, line)) {
      std::stringstream streamLine(line);
      streamLine >> padIndex >> padRow >> pad >> xPos >> yPos >> connector >> pin >> partion >> region >> fecIndex >> fecConnector >> fecChannel >> sampaChip >> sampaChannel;

      const float localX = xPos / 10.f;
      const float localY = yPos / 10.f;

      // test for z > 0 and z< 0
      for (int j = 0; j < signTest.size(); ++j) {

        /// Get the coordinates of each pad swap also x to accomodate the mirroring from A to C side
        const GlobalPosition3D pos((-1.f) * signTest[j] * localX, localY, signTest[j] * 10.f);

        /// Transform to pad/row space
        const DigitPos digi = mapper.findDigitPosFromGlobalPosition(pos);

        /// Check whether the transformation was done properly
        BOOST_CHECK(pad == int(digi.getPadPos().getPad()));
        BOOST_CHECK(pad == int(digi.getPadSecPos().getPadPos().getPad()));

        BOOST_CHECK(padRow == int(digi.getPadSecPos().getPadPos().getRow()));

        const CRU cru(digi.getCRU());
        /// \todo check CRU
        BOOST_CHECK(region == int(cru.region()));
        const PadRegionInfo& regionDigi = mapper.getPadRegionInfo(cru.region());
        BOOST_CHECK(partion == int(regionDigi.getRegion()));

        const int rowInSector = digi.getPadPos().getRow() + regionDigi.getGlobalRowOffset();
        BOOST_CHECK(padRow == rowInSector);

        /// Transformation back into xyz coordinates
        const GlobalPadNumber padPos = mapper.globalPadNumber(PadPos(rowInSector, digi.getPadPos().getPad()));
        const PadCentre& padCentre = mapper.padCentre(padPos);

        LocalPosition3D posLoc(padCentre.X(), padCentre.Y(), pos.Z());

        /// As we're in sector 4, we can test the local coordinates by swapping x & y (again taking into account the mirroring between A & C side)
        BOOST_CHECK_CLOSE(posLoc.X(), pos.Y(), 1E-12);
        BOOST_CHECK_CLOSE(posLoc.Y(), signTest[j] * pos.X(), 1E-12);

        GlobalPosition3D posGlob = Mapper::LocalToGlobal(posLoc, cru.sector());

        /// Check whether the global coordinates match
        ///
        /// \todo here there should be no mirroring necessary!
        BOOST_CHECK_CLOSE((-1.f) * signTest[j] * pos.X(), posGlob.X(), 1E-12);
        BOOST_CHECK_CLOSE(pos.Y(), posGlob.Y(), 1E-12);
      }
    }
  }
}
} // namespace tpc
} // namespace o2
