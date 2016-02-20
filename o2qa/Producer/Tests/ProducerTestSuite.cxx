#define BOOST_TEST_MODULE Producer
#define BOOST_TEST_MAIN

#include <boost/test/unit_test.hpp>
#include <string>
#include <memory>
#include <FairMQDevice.h>
#include <TH1.h>
#include <iostream>

#include "Producer/HistogramProducer.h"
#include "Producer/ProducerDevice.h"
#include "fakeit.hpp"

using namespace std;

BOOST_AUTO_TEST_SUITE(ProducerTestSuite)

BOOST_AUTO_TEST_CASE(produceHistogramWithGivenParameters)
{
    string histogramId = "TestId";
    float xLow = -10.0;
    float xUp = 10.0;
    const int expectedNumberOfEntries = 1000;
    const int expectedNumberOfBins = 100;

    unique_ptr<HistogramProducer> histogramProducer(new HistogramProducer(histogramId, xLow, xUp));
    unique_ptr<TH1> histogram(dynamic_cast<TH1*>(histogramProducer->produceData()));

    BOOST_TEST(histogram->GetEntries() == expectedNumberOfEntries, "Invalid number of entries");
    BOOST_TEST(histogram->GetName() == histogramId, "Invalid name of histogram");
    BOOST_TEST(histogram->GetXaxis()->GetXmin() == xLow, "Invalid minimal value for x axis");
    BOOST_TEST(histogram->GetXaxis()->GetXmax() == xUp, "Invalid maximal value for x axis");
    BOOST_TEST(histogram->GetNbinsX() == expectedNumberOfBins, "Invalid number of bins");
}

BOOST_AUTO_TEST_CASE(establishChannelByProducerDevice)
{
	unique_ptr<ProducerDevice> producer(new ProducerDevice("Producer", "Hist", -10.0, 10.0, 1));
	BOOST_TEST(producer->fChannels.size() == 0, "Producer device has a channel connected at startup");

	producer->establishChannel("req", "connect", "tcp://localhost:5005", "data");
	BOOST_TEST(producer->fChannels.size() == 1, "Producer device did not establish channel");
}

BOOST_AUTO_TEST_SUITE_END()
