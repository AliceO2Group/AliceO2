// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#define BOOST_TEST_MODULE Viewer
#define BOOST_TEST_MAIN

#include <boost/test/unit_test.hpp>
#include <string>

#include "QCViewer/ViewerDevice.h"

BOOST_AUTO_TEST_SUITE(ViewerTestSuite)

BOOST_AUTO_TEST_CASE(createViewerDevice)
{
  const std::string viewerId = "Viewer_1";
  const int numberOfThreads = 1;

  ViewerDevice viewer(viewerId);

  BOOST_TEST(viewer.GetId() == viewerId);
}

BOOST_AUTO_TEST_CASE(establishChannelByViewerDevice)
{
  const std::string viewerId = "Viewer_1";
  const int numberOfThreads = 1;
  ViewerDevice viewer(viewerId);

  BOOST_TEST(viewer.fChannels.size() == 0, "Viewer device has a channel connected at startup");

  viewer.establishChannel("req", "connect", "tcp://localhost:5005", "data");
  BOOST_TEST(viewer.fChannels.size() == 1, "Viewer device did not establish channel");
}

BOOST_AUTO_TEST_SUITE_END()
