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

  ViewerDevice viewer(viewerId, numberOfThreads);

  BOOST_TEST(viewer.GetProperty(ViewerDevice::Id, "default_id") == viewerId);
}

BOOST_AUTO_TEST_CASE(establishChannelByViewerDevice)
{
  const std::string viewerId = "Viewer_1";
  const int numberOfThreads = 1;
  ViewerDevice viewer(viewerId, numberOfThreads);

  BOOST_TEST(viewer.fChannels.size() == 0, "Viewer device has a channel connected at startup");

  viewer.establishChannel("req", "connect", "tcp://localhost:5005", "data");
  BOOST_TEST(viewer.fChannels.size() == 1, "Viewer device did not establish channel");
}

BOOST_AUTO_TEST_SUITE_END()
