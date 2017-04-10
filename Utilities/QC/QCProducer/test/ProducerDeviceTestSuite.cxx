#define BOOST_TEST_MODULE ProducerDeviceTest
#define BOOST_TEST_MAIN

#include <memory>

#include <boost/test/unit_test.hpp>

#include "QCProducer/ProducerDevice.h"
#include "QCProducer/TH1Producer.h"

using namespace std;

namespace
{
const int NUMBER_OF_BINS = 10;
const char* NAME = "TEST_NAME";
const char* TITLE = "TEST_TITLE";
const int BUFFER_SIZE = 10;
}

BOOST_AUTO_TEST_SUITE(ProducerTestSuite)

BOOST_AUTO_TEST_CASE(establishChannelByProducerDevice)
{
  shared_ptr<Producer> producer(new TH1Producer(NAME, TITLE, NUMBER_OF_BINS));
  unique_ptr<ProducerDevice> producerDevice(new ProducerDevice("Producer", 1, producer));

  BOOST_TEST(producerDevice->fChannels.size() == 0, "Producer device has a channel connected at startup");

  producerDevice->establishChannel("req", "connect", "tcp://localhost:5005", "data", BUFFER_SIZE);
  BOOST_TEST(producerDevice->fChannels.size() == 1, "Producer device did not establish channel");
}

BOOST_AUTO_TEST_SUITE_END()
