#define BOOST_TEST_MODULE MergerDevice
#define BOOST_TEST_MAIN

#include <memory>

#include <boost/test/unit_test.hpp>

#include "QCMerger/MergerDevice.h"

using namespace std;

namespace
{
const char* INPUT_ADDRESS = "tcp://*:5005";
const char* OUTPUT_ADDREDD = "tcp://login01.pro.cyfronet.pl:5004";
const char* MERGER_DEVICE_ID = "TEST_MERGER";
const int NUMBER_OF_IO_THREADS = 1;
const int NUMBER_OF_QC_OBJECTS_FOR_COMPLETE_DATA = 2;
int bufferSize = 10;

shared_ptr<MergerDevice> mrgerDevice;
}

BOOST_AUTO_TEST_SUITE(MergerDeviceTestSuite)

BOOST_AUTO_TEST_CASE(createMergerDevice)
{
  unique_ptr<MergerDevice> mrgerDevice(new MergerDevice(
    unique_ptr<Merger>(new Merger(NUMBER_OF_QC_OBJECTS_FOR_COMPLETE_DATA)), MERGER_DEVICE_ID, NUMBER_OF_IO_THREADS));

  BOOST_CHECK(mrgerDevice->GetProperty(MergerDevice::Id, "default_id") == MERGER_DEVICE_ID);
  BOOST_CHECK(mrgerDevice->GetProperty(MergerDevice::NumIoThreads, 0) == NUMBER_OF_IO_THREADS);
}

BOOST_AUTO_TEST_CASE(establishChannelByMergerDevice)
{
  unique_ptr<MergerDevice> mrgerDevice(new MergerDevice(
    unique_ptr<Merger>(new Merger(NUMBER_OF_QC_OBJECTS_FOR_COMPLETE_DATA)), MERGER_DEVICE_ID, NUMBER_OF_IO_THREADS));

  BOOST_TEST(mrgerDevice->fChannels.size() == 0, "Producer device has a channel connected at startup");

  mrgerDevice->establishChannel("req", "connect", OUTPUT_ADDREDD, "test", bufferSize, bufferSize);
  BOOST_TEST(mrgerDevice->fChannels.size() == 1, "Producer device did not establish channel");
}

BOOST_AUTO_TEST_SUITE_END()
