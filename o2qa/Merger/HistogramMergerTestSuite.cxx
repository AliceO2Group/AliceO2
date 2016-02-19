#define BOOST_TEST_MODULE Merger
#define BOOST_TEST_MAIN

#include <boost/test/unit_test.hpp>
#include <string>

#include "Merger/HistogramMerger.h"


BOOST_AUTO_TEST_SUITE(HistogramMergerTestSuite)

BOOST_AUTO_TEST_CASE(createMergerWithGivenIdAndNumberOfThreads)
{
    std::string mergerId = "Test_merger";
    unsigned short numberOfThreads = 1;

    HistogramMerger *merger = new HistogramMerger(mergerId, numberOfThreads);

    BOOST_CHECK(merger != nullptr);
    BOOST_CHECK(merger->GetProperty(HistogramMerger::Id, "default_id") == mergerId);
    BOOST_CHECK(merger->GetProperty(HistogramMerger::NumIoThreads, 0) == numberOfThreads);

    delete merger;
}

BOOST_AUTO_TEST_SUITE_END()
