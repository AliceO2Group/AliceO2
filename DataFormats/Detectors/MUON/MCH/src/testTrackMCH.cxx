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

#include <boost/test/tools/old/interface.hpp>
#define BOOST_TEST_MODULE MCH TrackMCH
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include <boost/test/data/test_case.hpp>
#include <boost/test/data/monomorphic.hpp>

#include "DataFormatsMCH/TrackMCH.h"

BOOST_AUTO_TEST_CASE(TrackIRMatchesTrackTime)
{
  uint16_t bc{10};
  uint32_t orbit{2000};
  uint32_t orbitRef = orbit - 120;
  o2::InteractionRecord ir{bc, orbit};
  o2::InteractionRecord irRef{0, orbitRef};

  o2::mch::TrackMCH track;

  auto bcDiff = ir.differenceInBC(irRef);
  float tMean = o2::constants::lhc::LHCBunchSpacingMUS * bcDiff;
  float tErr = 6.0 * o2::constants::lhc::LHCBunchSpacingMUS;
  track.setTimeMUS(tMean, tErr);

  o2::InteractionRecord trackIR = track.getMeanIR(orbitRef);
  BOOST_CHECK_EQUAL(ir, trackIR);
}

BOOST_AUTO_TEST_CASE(TrackIRMatchesNegativeTrackTime)
{
  uint16_t bc{10};
  uint32_t orbit{2000};
  uint32_t orbitRef = orbit + 1;
  o2::InteractionRecord ir{bc, orbit};
  o2::InteractionRecord irRef{0, orbitRef};

  o2::mch::TrackMCH track;

  auto bcDiff = ir.differenceInBC(irRef);
  float tMean = o2::constants::lhc::LHCBunchSpacingMUS * bcDiff;
  float tErr = 6.0 * o2::constants::lhc::LHCBunchSpacingMUS;
  track.setTimeMUS(tMean, tErr);

  o2::InteractionRecord trackIR = track.getMeanIR(orbitRef);
  BOOST_CHECK_EQUAL(ir, trackIR);
}
