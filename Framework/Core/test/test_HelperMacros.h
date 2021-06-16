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
#ifndef FRAMEWORK_TEST_HELPERMACROS
#define FRAMEWORK_TEST_HELPERMACROS

// Expand BOOST_CHECK_EQUAL to have an additional message
#define BOOST_CHECK_EQUAL_MESSAGE(L, R, M) \
  {                                        \
    BOOST_TEST_INFO(M);                    \
    BOOST_CHECK_EQUAL(L, R);               \
  }

#endif // FRAMEWORK_TEST_HELPERMACROS
