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

#define BOOST_TEST_MODULE Test PHOS Cell
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include "DataFormatsPHOS/Cell.h"

#include <algorithm>

namespace o2
{

namespace phos
{

/// \brief Standard tests for cell class
///
/// - verify that set and get functions return consistent values
BOOST_AUTO_TEST_CASE(Cell_test)
{
  Cell c;
  c.setTime(0.);
  for (short j = 1793; j < 14337; j++) {
    c.setAbsId(j);
    BOOST_CHECK_EQUAL(c.getAbsId(), j);
    BOOST_CHECK_EQUAL(c.getTRUId(), j);
    BOOST_CHECK_SMALL(c.getTime() - float(0.), kTimeAccuracy3);
    BOOST_CHECK_SMALL(c.getEnergy() - 0, float(0.1));
    BOOST_CHECK_EQUAL(c.getLowGain(), true);
    BOOST_CHECK_EQUAL(c.getTRU(), false);
  }
  for (short j = 0; j < 3136; j++) { // TRU
    c.setAbsId(14337 + j);
    BOOST_CHECK_EQUAL(c.getAbsId(), 0);
    BOOST_CHECK_EQUAL(c.getTRUId(), 14337 + j);
    BOOST_CHECK_SMALL(c.getTime() - float(0.), kTimeAccuracy3);
    BOOST_CHECK_SMALL(c.getEnergy() - float(0), float(0.1));
    BOOST_CHECK_EQUAL(c.getTRU(), true);
  }

  c.setAbsId(1793);
  std::vector<float> times = {-150.e-9, -10.5e-9, -0.55e-9, 0.35e-9, 2.1e-9, 3.2e-9, 4.e-9, 5.e-9, 6.e-9, 7.e-9, 8.e-9, 9.e-9, 10.e-9, 20.e-9, 50.e-9, 100.e-9, 150.e-9};

  for (float t : times) {
    c.setTime(t);
    BOOST_CHECK_EQUAL(c.getAbsId(), 1793);
    BOOST_CHECK_SMALL(c.getTime() - t, kTimeAccuracy3);
    BOOST_CHECK_SMALL(c.getEnergy() - float(0), float(0.1));
    BOOST_CHECK_EQUAL(c.getLowGain(), true);
  }

  c.setTime(0);
  std::vector<float> energies = {2., 5., 10., 50., 100., 200., 500., 900., 1200., 1600., 2000.};

  for (float e : energies) {
    c.setEnergy(e);
    BOOST_CHECK_EQUAL(c.getAbsId(), 1793);
    BOOST_CHECK_SMALL(c.getTime() - float(0.), kTimeAccuracy3);
    BOOST_CHECK_SMALL(e - c.getEnergy(), float(0.1));
    BOOST_CHECK_EQUAL(c.getLowGain(), true);
  }

  c.setEnergy(0);

  c.setLowGain();
  BOOST_CHECK_EQUAL(c.getAbsId(), 1793);
  BOOST_CHECK_SMALL(c.getTime() - float(0.), kTimeAccuracy3);
  BOOST_CHECK_SMALL(c.getEnergy() - float(0), float(0.1));
  BOOST_CHECK_EQUAL(c.getLowGain(), true);
  BOOST_CHECK_EQUAL(c.getHighGain(), false);
  BOOST_CHECK_EQUAL(c.getTRU(), false);

  c.setHighGain();
  BOOST_CHECK_EQUAL(c.getAbsId(), 1793);
  BOOST_CHECK_SMALL(c.getTime() - float(0.), kTimeAccuracy3);
  BOOST_CHECK_SMALL(c.getEnergy() - float(0), float(0.1));
  BOOST_CHECK_EQUAL(c.getLowGain(), false);
  BOOST_CHECK_EQUAL(c.getHighGain(), true);
  BOOST_CHECK_EQUAL(c.getTRU(), false);
}

} // namespace phos

} // namespace o2
