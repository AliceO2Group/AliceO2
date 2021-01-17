// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include <type_traits>
#define BOOST_TEST_MODULE Test DetectorsDCS DataPoints
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include "DetectorsDCS/DataPointCompositeObject.h"
#include "Framework/TypeTraits.h"
#include <vector>
#include <list>
#include <gsl/gsl>
#include <boost/mpl/list.hpp>

typedef boost::mpl::list<o2::dcs::DataPointIdentifier, o2::dcs::DataPointValue, o2::dcs::DataPointCompositeObject> testTypes;

BOOST_AUTO_TEST_CASE_TEMPLATE(DataPointCompositeObjectTypeTraits, T, testTypes)
{
  BOOST_CHECK_EQUAL(std::is_trivially_copyable<T>::value, true);
  BOOST_CHECK_EQUAL(std::is_polymorphic<T>::value, false);
  BOOST_CHECK_EQUAL(std::is_pointer<T>::value, false);
  BOOST_CHECK_EQUAL(o2::framework::is_forced_non_messageable<T>::value, false);
}

BOOST_AUTO_TEST_CASE(DataPointsAreMessageable)
{
  BOOST_CHECK_EQUAL(o2::framework::is_messageable<o2::dcs::DataPointIdentifier>::value, true);
  BOOST_CHECK_EQUAL(o2::framework::is_messageable<o2::dcs::DataPointValue>::value, true);
  BOOST_CHECK_EQUAL(o2::framework::is_messageable<o2::dcs::DataPointCompositeObject>::value, true);
}
