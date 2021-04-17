// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include <boost/test/tools/old/interface.hpp>
#define BOOST_TEST_MODULE MCH Digit
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include <boost/test/data/test_case.hpp>
#include <boost/test/data/monomorphic.hpp>

#include "DataFormatsMCH/Digit.h"

int dummyDetId{712};
int dummyPadId{0};
unsigned long dummyADC{0};
uint32_t dummyTime{0};
uint16_t dummyNofSamples{0};

BOOST_AUTO_TEST_CASE(NofSamplesMustFitWithin10Bits)
{
  BOOST_CHECK_THROW(o2::mch::Digit d(dummyDetId, dummyPadId, dummyADC, dummyTime, 1025), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(DefaultConstructorMakesNonSaturatedDigit)
{
  o2::mch::Digit d(dummyDetId, dummyPadId, dummyADC, dummyTime, dummyNofSamples);
  BOOST_CHECK(d.isSaturated() == false);
}

std::vector<uint16_t> nsamples{
  1 << 0,
  1 << 1,
  1 << 2,
  1 << 3,
  1 << 4,
  1 << 5,
  1 << 6,
  1 << 7,
  1 << 8,
  1 << 9};

BOOST_DATA_TEST_CASE(DefaultConstructorNofSamplesIsInvariant,
                     boost::unit_test::data::make(nsamples), nofSamples)
{
  o2::mch::Digit d(dummyDetId, dummyPadId, dummyADC, dummyTime, nofSamples);
  BOOST_CHECK_EQUAL(d.nofSamples(), nofSamples);
}

BOOST_DATA_TEST_CASE(SetSaturatedDoesNotAffectPublicNofSamples,
                     boost::unit_test::data::make(nsamples), nofSamples)
{
  o2::mch::Digit d(dummyDetId, dummyPadId, dummyADC, dummyTime, nofSamples);

  BOOST_TEST_INFO("setting saturation to true");
  d.setSaturated(true);
  BOOST_CHECK_EQUAL(d.nofSamples(), nofSamples);

  BOOST_TEST_INFO("setting saturation to false");
  d.setSaturated(false);
  BOOST_CHECK_EQUAL(d.nofSamples(), nofSamples);
}
