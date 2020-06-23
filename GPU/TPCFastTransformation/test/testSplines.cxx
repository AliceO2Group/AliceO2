// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file testSplines.cxx
/// \author Sergey Gorbunov

#define BOOST_TEST_MODULE Test TPC Fast Transformation
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include "Spline1D.h"
#include "Spline2D.h"

namespace o2
{
namespace gpu
{

/// @brief Basic test if we can create the interface
BOOST_AUTO_TEST_CASE(Spline_test1)
{

  o2::gpu::Spline1D<float> s1;
  int err1 = s1.test(0);
  BOOST_CHECK_MESSAGE(err1 == 0, "test of GPU/TPCFastTransform/Spline1D failed with the error code " << err1);

  o2::gpu::Spline2D<float, 1> s2;
  int err2 = s2.test(0);
  BOOST_CHECK_MESSAGE(err2 == 0, "test of GPU/TPCFastTransform/Spline2D failed with the error code " << err2);
}
} // namespace gpu
} // namespace o2
