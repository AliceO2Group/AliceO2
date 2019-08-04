// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#define BOOST_TEST_MODULE midBaseMapping
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
// Keep this separate or clang format will sort the include
// thus breaking compilation
#include <boost/test/data/monomorphic.hpp>
#include <boost/test/data/monomorphic/generators/xrange.hpp>
#include <boost/test/data/test_case.hpp>
#include <iostream>
#include <sstream>
#include "MIDBase/Mapping.h"
#include "MIDBase/MpArea.h"

namespace bdata = boost::unit_test::data;

namespace o2
{
namespace mid
{
class MyFixture
{
 public:
  Mapping mapping;
};

bool areEqual(const Mapping::MpStripIndex& strip1, const Mapping::MpStripIndex& strip2, int cathode)
{
  bool sameColStrip = (strip1.column == strip2.column && strip1.strip == strip2.strip);
  if (cathode == 1) {
    return sameColStrip;
  }
  return (sameColStrip && strip1.line == strip2.line);
}

bool findInNeighbours(const Mapping::MpStripIndex& refStrip, const std::vector<Mapping::MpStripIndex>& neighbours,
                      int cathode)
{
  for (auto const& neigh : neighbours) {
    if (areEqual(refStrip, neigh, cathode)) {
      return true;
    }
  }
  std::cerr << "Cannot find " << refStrip.strip << "  line  " << refStrip.line << "  column " << refStrip.column
            << std::endl;
  return false;
}

int findInNeighbours(float xPos, float yPos, int cathode, int deId, bool checkUpDown,
                     const std::vector<Mapping::MpStripIndex>& neighbours, const Mapping& mapping)
{
  int found = 0;
  float delta = checkUpDown ? 0.1 : 0;
  Mapping::MpStripIndex refStrip = mapping.stripByPosition(xPos, yPos + delta, cathode, deId, false);
  if (!refStrip.isValid()) {
    return found;
  }

  if (!findInNeighbours(refStrip, neighbours, cathode)) {
    return -1;
  }
  ++found;

  if (checkUpDown) {
    Mapping::MpStripIndex upRef = mapping.stripByPosition(xPos, yPos - delta, cathode, deId, false);
    if (!areEqual(refStrip, upRef, cathode)) {
      if (!findInNeighbours(refStrip, neighbours, cathode)) {
        return -2;
      }
      ++found;
    }
  }
  return found;
}

bool areNeighboursOk(const Mapping::MpStripIndex& stripIndex, const std::vector<Mapping::MpStripIndex>& neighbours,
                     const Mapping& mapping, int cathode, int deId)
{
  MpArea stripArea =
    mapping.stripByLocation(stripIndex.strip, cathode, stripIndex.line, stripIndex.column, deId, false);

  std::stringstream ss;
  ss << "Strip " << stripIndex.strip << "  cathode " << cathode << "  line  " << stripIndex.line << "  column "
     << stripIndex.column;

  bool isOk = true;

  double xPos[3] = {stripArea.getXmin() - 0.1, stripArea.getCenterX(), stripArea.getXmax() + 0.1};
  double yPos[3] = {stripArea.getYmin() - 0.1, stripArea.getCenterY(), stripArea.getYmax() + 0.1};

  int nMatched = 0;
  bool checkUpDown = false;
  for (int ix = 0; ix < 3; ++ix) {
    for (int iy = 0; iy < 3; ++iy) {
      if ((ix - 1) * (iy - 1) != 0 || ix == iy) {
        continue;
      }
      if (cathode == 0) {
        checkUpDown = (ix == 1) ? false : true;
      } else if (iy != 1) {
        continue;
      }

      int nFound = findInNeighbours(xPos[ix], yPos[iy], cathode, deId, checkUpDown, neighbours, mapping);
      if (nFound < 0) {
        ss << "  err: (" << ix - 1 << ", " << iy - 1 << ")";
        isOk = false;
      } else {
        nMatched += nFound;
      }
    }
  }

  if (nMatched != neighbours.size()) {
    isOk = false;
    ss << "  matched " << nMatched << " != " << neighbours.size();
  }

  if (!isOk) {
    std::cerr << ss.str() << ".  Neighbours: " << std::endl;
    for (auto const& neigh : neighbours) {
      std::cerr << "    strip " << neigh.strip << "  line  " << neigh.line << "  column " << neigh.column << std::endl;
    }
  }

  return isOk;
}

BOOST_DATA_TEST_CASE_F(MyFixture, MID_Mapping_NeighboursNBP, boost::unit_test::data::xrange(72))
{
  int deId = sample;
  for (int icolumn = mapping.getFirstColumn(deId); icolumn < 7; ++icolumn) {
    int firstLineBP = mapping.getFirstBoardBP(icolumn, deId);
    int lastLineBP = mapping.getLastBoardBP(icolumn, deId);
    for (int icathode = 0; icathode < 2; ++icathode) {
      int firstLine = firstLineBP;
      int lastLine = lastLineBP;
      if (icathode == 1) {
        // When generating the neighbours in the non-bending plane
        // we look at the y position as well (given by the "line")
        // In the RPCs "cut", we can find or not a neighbour depending
        // on the line.
        // When we perform the tests here, we search for neighbours basing on the
        // x, y position of the center of the strip.
        // If we want to have consistent results we therefore need to attribute a line
        // close to the center for the non-bending plane
        firstLine = lastLine = (firstLine + lastLine) / 2;
      }
      int nStrips = (icathode == 0) ? 16 : mapping.getNStripsNBP(icolumn, deId);
      for (int iline = firstLine; iline <= lastLine; ++iline) {
        for (int istrip = 0; istrip < nStrips; ++istrip) {
          Mapping::MpStripIndex stripIndex;
          stripIndex.column = icolumn;
          stripIndex.line = iline;
          stripIndex.strip = istrip;
          std::vector<Mapping::MpStripIndex> neighbours = mapping.getNeighbours(stripIndex, icathode, deId);
          BOOST_TEST(areNeighboursOk(stripIndex, neighbours, mapping, icathode, deId));
        } // loop on strips
      }   // loop on lines
    }     // loop on cathode
  }       // loop on column
}

} // namespace mid
} // namespace o2
