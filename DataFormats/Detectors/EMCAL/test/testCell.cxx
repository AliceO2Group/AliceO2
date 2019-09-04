// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#define BOOST_TEST_MODULE Test EMCAL Cell
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include "DataFormatsEMCAL/Cell.h"

#include <algorithm>

namespace o2
{

namespace emcal
{

/// \brief Standard tests for cell class
///
/// - verify that set and get functions return consistent values
BOOST_AUTO_TEST_CASE(Cell_test)
{
  Cell c;

  for (short j = 0; j < 17664; j++) {
    c.setTower(j);
    BOOST_CHECK_EQUAL(c.getTower(), j);
    BOOST_CHECK_EQUAL(c.getTimeStamp(), 0);
    BOOST_CHECK_EQUAL(c.getEnergy(), 0);
    BOOST_CHECK_EQUAL(c.getLowGain(), true);
  }

  c.setTower(0);
  std::vector<int> times = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 50, 100, 200};

  for (auto t : times) {
    c.setTimeStamp(t);
    BOOST_CHECK_EQUAL(c.getTower(), 0);
    BOOST_CHECK_EQUAL(c.getTimeStamp(), t);
    BOOST_CHECK_EQUAL(c.getEnergy(), 0);
    BOOST_CHECK_EQUAL(c.getLowGain(), true);
  }

  c.setTimeStamp(0);
  std::vector<double> energies = {0.5, 1, 2, 5, 10, 20, 40, 60, 100, 150, 200};

  for (auto e : energies) {
    c.setEnergy(e);
    BOOST_CHECK_EQUAL(c.getTower(), 0);
    BOOST_CHECK_EQUAL(c.getTimeStamp(), 0);
    BOOST_CHECK_SMALL(e - c.getEnergy(), 0.02); // Require 20 MeV resolution
    BOOST_CHECK_EQUAL(c.getLowGain(), true);
  }

  c.setEnergy(0);

  c.setLowGain();
  BOOST_CHECK_EQUAL(c.getTower(), 0);
  BOOST_CHECK_EQUAL(c.getTimeStamp(), 0);
  BOOST_CHECK_EQUAL(c.getEnergy(), 0);
  BOOST_CHECK_EQUAL(c.getLowGain(), true);
  BOOST_CHECK_EQUAL(c.getHighGain(), false);
  BOOST_CHECK_EQUAL(c.getLEDMon(), false);
  BOOST_CHECK_EQUAL(c.getTRU(), false);

  c.setHighGain();
  BOOST_CHECK_EQUAL(c.getTower(), 0);
  BOOST_CHECK_EQUAL(c.getTimeStamp(), 0);
  BOOST_CHECK_EQUAL(c.getEnergy(), 0);
  BOOST_CHECK_EQUAL(c.getLowGain(), false);
  BOOST_CHECK_EQUAL(c.getHighGain(), true);
  BOOST_CHECK_EQUAL(c.getLEDMon(), false);
  BOOST_CHECK_EQUAL(c.getTRU(), false);

  c.setLEDMon();
  BOOST_CHECK_EQUAL(c.getTower(), 0);
  BOOST_CHECK_EQUAL(c.getTimeStamp(), 0);
  BOOST_CHECK_EQUAL(c.getEnergy(), 0);
  BOOST_CHECK_EQUAL(c.getLowGain(), false);
  BOOST_CHECK_EQUAL(c.getHighGain(), false);
  BOOST_CHECK_EQUAL(c.getLEDMon(), true);
  BOOST_CHECK_EQUAL(c.getTRU(), false);

  c.setTRU();
  BOOST_CHECK_EQUAL(c.getTower(), 0);
  BOOST_CHECK_EQUAL(c.getTimeStamp(), 0);
  BOOST_CHECK_EQUAL(c.getEnergy(), 0);
  BOOST_CHECK_EQUAL(c.getLowGain(), false);
  BOOST_CHECK_EQUAL(c.getHighGain(), false);
  BOOST_CHECK_EQUAL(c.getLEDMon(), false);
  BOOST_CHECK_EQUAL(c.getTRU(), true);
}

} // namespace emcal

} // namespace o2