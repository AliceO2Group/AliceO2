#define BOOST_TEST_MODULE Producer
#define BOOST_TEST_MAIN

#include <boost/test/unit_test.hpp>
#include <string>
#include <memory>
#include <FairMQDevice.h>
#include <TH1.h>

#include "Producer/HistogramProducer.h"
#include "TreeProducer.h"
#include "Producer/ProducerDevice.h"

using namespace std;

BOOST_AUTO_TEST_SUITE(ProducerTestSuite)

BOOST_AUTO_TEST_CASE(produceHistogramWithGivenParameters)
{
  string histogramNamePrefix = "TestId_";
  string histogramTitle = "Gauss_distribution";
  float xLow = -10.0;
  float xUp = 10.0;
  const int expectedNumberOfEntries = 1000;
  const int expectedNumberOfBins = 100;

  unique_ptr<HistogramProducer> histogramProducer(new HistogramProducer(histogramNamePrefix,
                                                                        histogramTitle,
                                                                        xLow,
                                                                        xUp));

  unique_ptr<TH1> histogram(dynamic_cast<TH1*>(histogramProducer->produceData()));

  BOOST_TEST(histogram->GetEntries() == expectedNumberOfEntries, "Invalid number of entries");
  BOOST_TEST(histogram->GetName() == "TestId_0", "Invalid name of histogram");
  BOOST_TEST(histogram->GetTitle() == histogramTitle, "Invalid title of histogram");
  BOOST_TEST(histogram->GetXaxis()->GetXmin() == xLow, "Invalid minimal value for x axis");
  BOOST_TEST(histogram->GetXaxis()->GetXmax() == xUp, "Invalid maximal value for x axis");
  BOOST_TEST(histogram->GetNbinsX() == expectedNumberOfBins, "Invalid number of bins");
}

BOOST_AUTO_TEST_CASE(produceTreeWithGivenParameters)
{
  string treeNamePrefix = "test_tree_prefix";
  string treeTitle = "test_tree_title";
  double numberOfBranches = 2;
  double numberOfEntriesInEachBranch = 1000;

  unique_ptr<TreeProducer> treeProducer(new TreeProducer(treeNamePrefix,
                                                         treeTitle,
                                                         numberOfBranches,
                                                         numberOfEntriesInEachBranch));

  unique_ptr<TTree> tree(dynamic_cast<TTree*>(treeProducer->produceData()));

  BOOST_TEST(tree->GetTitle() == treeTitle, "Invalid title of tree");
  BOOST_TEST(tree->GetEntries() == numberOfBranches * numberOfEntriesInEachBranch, "Invalid number of entries in tree");
}

BOOST_AUTO_TEST_CASE(establishChannelByProducerDevice)
{
  std::shared_ptr<Producer> producer = std::make_shared<HistogramProducer>("HistName_", "HistTitle_", -10.0, 10.0);
	unique_ptr<ProducerDevice> producerDevice(new ProducerDevice("Producer", 1, producer));

	BOOST_TEST(producerDevice->fChannels.size() == 0, "Producer device has a channel connected at startup");

	producerDevice->establishChannel("req", "connect", "tcp://localhost:5005", "data");
	BOOST_TEST(producerDevice->fChannels.size() == 1, "Producer device did not establish channel");
}

BOOST_AUTO_TEST_SUITE_END()
