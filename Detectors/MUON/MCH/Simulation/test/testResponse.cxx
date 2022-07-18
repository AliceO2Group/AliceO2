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
/// \file testResponse.cxx
/// \brief This task tests of the Response of the MCH digitization
/// \author Michael Winn, DPhN/IRFU/CEA, michael.winn@cern.ch

#define BOOST_TEST_MODULE Test MCHSimulation Response
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include <memory>
#include <vector>

#include "MCHSimulation/Response.h"

#include "TH1D.h"
#include "TF1.h"

namespace o2
{
namespace mch
{

BOOST_TEST_DECORATOR(*boost::unit_test::disabled())
BOOST_AUTO_TEST_CASE(Response_test)
{
  // check transition between energy and charge
  // check integration via Mathieson
  // check FEE response
  Response r_stat1(Station::Type1);
  Response r_stat2(Station::Type2345);

  // check conversion energy to charge
  float eloss = 1e-6;
  TH1D hTest("hTest", "", 10000, 0, 1000);
  TF1 gaus("gaus", "gaus");
  for (int i = 0; i < 1000000; i++) {
    hTest.Fill(r_stat1.etocharge(eloss));
  }

  hTest.Fit("gaus", "Q");

  float charge_target = 137.15721769834721;
  float charge_precision = charge_target / 100.f;
  BOOST_CHECK_CLOSE(gaus.GetParameter(1), charge_target, charge_precision);
  // TODO, if needed
  float charge_resolution_target = 21.7099991;
  float charge_resolution_precision = charge_resolution_target / 100.f;
  BOOST_CHECK_CLOSE(gaus.GetParameter(2), charge_resolution_target, charge_resolution_precision);

  // test Mathieson integration on Pad
  // total charge to be distributed
  float charge_on_plane = 100.f;
  // borders with some distance to anode
  // smallest pad dimension for  stat.
  float xmin = -0.658334;
  float xmax = -0.0283;
  float ymin = -0.0934209;
  float ymax = 0.326579;

  float expected_chargeonpad_stat1 = 0.29306942 * charge_on_plane;

  float chargeonpad_precision = expected_chargeonpad_stat1 / 10.f;
  float result_chargeonpad_stat1 = r_stat1.chargePadfraction(xmin, xmax, ymin, ymax) * charge_on_plane;
  BOOST_CHECK_CLOSE(result_chargeonpad_stat1, expected_chargeonpad_stat1, chargeonpad_precision);

  // test r_stat2
  xmin = -0.428572;
  xmax = 0.285714;
  ymin = -0.108383;
  ymax = 9.89162;
  float expected_chargeonpad_stat2 = 0.633606318 * charge_on_plane;
  chargeonpad_precision = expected_chargeonpad_stat2 / 10.f;
  float result_chargeonpad_stat2 = r_stat2.chargePadfraction(xmin, xmax, ymin, ymax) * charge_on_plane;
  BOOST_CHECK_CLOSE(result_chargeonpad_stat2, expected_chargeonpad_stat2, chargeonpad_precision);
  // todo some test of chargeCorr? function, not so obvious
  // getAnod, necessary to do?
}

} // namespace mch
} // namespace o2
