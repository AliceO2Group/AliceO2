#define BOOST_TEST_MODULE Merger
#define BOOST_TEST_MAIN

#include <boost/test/unit_test.hpp>
#include <string>
#include <memory>
#include <vector>
#include <sstream>
#include <TH1F.h>

#include "Merger/MergerDevice.h"
#include "Merger.h"

BOOST_AUTO_TEST_SUITE(MergerDeviceTestSuite)

BOOST_AUTO_TEST_CASE(createMergerDeviceWithGivenIdAndNumberOfThreads)
{
  std::string mergerId = "Test_merger";
  unsigned short numberOfThreads = 1;

  MergerDevice merger(std::unique_ptr<Merger>(new Merger()),
                                              mergerId,
                                              numberOfThreads);

  BOOST_CHECK(merger.GetProperty(MergerDevice::Id, "default_id") == mergerId);
  BOOST_CHECK(merger.GetProperty(MergerDevice::NumIoThreads, 0) == numberOfThreads);
}

BOOST_AUTO_TEST_CASE(mergeFiveHistogramsWithTheSameTitle)
{
  using namespace std;

  const unsigned numberOfHistogramsToTest = 5;
  Merger* merger = new Merger();
  vector<shared_ptr<TH1F>> histograms;

  for (int i = 0; i < numberOfHistogramsToTest; ++i) {
    ostringstream histogramName;
    histogramName << "test_histogram_" << i;

    histograms.push_back(make_shared<TH1F>(histogramName.str().c_str(), "Gauss_distribution", 100, -10.0, 10.0));
    histograms.at(i)->FillRandom("gaus", 1000);

    TH1* histogram = dynamic_cast<TH1*>(merger->mergeObject(histograms.at(i).get()));

    ostringstream errorMessage;
    errorMessage << "Expected: " << (1000 * (i + 1)) << " entries, received: " << histogram->GetEntries();
    BOOST_TEST(histogram->GetEntries() == (1000 * (i + 1)), errorMessage.str().c_str());

    delete histogram;
  }

  delete merger;
}

BOOST_AUTO_TEST_CASE(mergeFourHistogramsWithTwoDifferentTitles)
{
  using namespace std;

  const unsigned histogramsToMerge = 4;
  const unsigned numberOfHistogramsWithTheSameTitle = 2;
  unsigned histogramNameIndex = 0;
  Merger* merger = new Merger();
  vector<shared_ptr<TH1F>> histograms;

  histograms.push_back(make_shared<TH1F>("test_histogram_0", "first_title", 100, -10.0, 10.0));
  histograms.push_back(make_shared<TH1F>("test_histogram_1", "first_title", 100, -10.0, 10.0));
  histograms.push_back(make_shared<TH1F>("test_histogram_2", "second_title", 100, -10.0, 10.0));
  histograms.push_back(make_shared<TH1F>("test_histogram_3", "second_title", 100, -10.0, 10.0));

  for (int i = 0; i < histogramsToMerge; ++i) {
    histograms.at(i)->FillRandom("gaus", 1000);
    TH1* histogram = dynamic_cast<TH1*>(merger->mergeObject(histograms.at(i).get()));

    if (i < numberOfHistogramsWithTheSameTitle) {
      ostringstream errorMessage;
      errorMessage << "Expected: " << (1000 * (i + 1)) << " entries, received: " << histogram->GetEntries();
      BOOST_TEST(histogram->GetEntries() == (1000 * (i + 1)), errorMessage.str().c_str());
      BOOST_TEST(histogram->GetTitle() == "first_title", "Invalid title of histogram");
    }
    else {
      ostringstream errorMessage;
      errorMessage << "Expected: " << (1000 * (i - 1)) << " entries, received: " << histogram->GetEntries();
      BOOST_TEST(histogram->GetEntries() == (1000 * (i - 1)), errorMessage.str().c_str());
      BOOST_TEST(histogram->GetTitle() == "second_title", "Invalid title of histogram");
    }

    delete histogram;
  }

  delete merger;
}

BOOST_AUTO_TEST_SUITE_END()
