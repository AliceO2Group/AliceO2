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
#include "fakeit.hpp"

BOOST_AUTO_TEST_SUITE(MergerDeviceTestSuite)

BOOST_AUTO_TEST_CASE(createMergerDeviceWithGivenIdAndNumberOfThreads)
{
    std::string mergerId = "Test_merger";
    unsigned short numberOfThreads = 1;

    std::unique_ptr<MergerDevice> merger(new MergerDevice(std::unique_ptr<Merger>(new Merger()), mergerId, numberOfThreads));

    BOOST_CHECK(merger != nullptr);
    BOOST_CHECK(merger->GetProperty(MergerDevice::Id, "default_id") == mergerId);
    BOOST_CHECK(merger->GetProperty(MergerDevice::NumIoThreads, 0) == numberOfThreads);
}

BOOST_AUTO_TEST_CASE(mergeTwoHistograms)
{
	using namespace std;

	const unsigned numberOfHistogramsToTest = 2;
	Merger* merger = new Merger();
	vector<TH1F> histograms;

	for(int i = 0; i < numberOfHistogramsToTest; ++i) {
		histograms.push_back(TH1F("test_histogram", "Gauss distribution", 100, -10.0, 10.0));
		histograms.at(i).FillRandom("gaus", 1000);

		TH1* histogram = dynamic_cast<TH1*>(merger->mergeObject(&(histograms.at(i))));

		ostringstream errorMessage;
		errorMessage << "Expected: " << (1000 * (i + 1)) << " entries, received: " << histogram->GetEntries();
		BOOST_TEST(histogram->GetEntries() == (1000 * (i + 1)), errorMessage.str().c_str());

		delete histogram;
	}

	delete merger;

}

BOOST_AUTO_TEST_SUITE_END()
