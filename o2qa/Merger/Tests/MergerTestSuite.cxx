#define BOOST_TEST_MODULE Merger
#define BOOST_TEST_MAIN

#include <boost/test/unit_test.hpp>
#include <string>
#include <memory>

#include "Merger/HistogramMerger.h"
#include "fakeit.hpp"

class Dummy {
public:
	virtual int returnThree() {return 3;}
};

BOOST_AUTO_TEST_SUITE(HistogramMergerTestSuite)

BOOST_AUTO_TEST_CASE(createMergerWithGivenIdAndNumberOfThreads)
{
    std::string mergerId = "Test_merger";
    unsigned short numberOfThreads = 1;

    std::unique_ptr<HistogramMerger> merger(new HistogramMerger(mergerId, numberOfThreads));

    BOOST_CHECK(merger != nullptr);
    BOOST_CHECK(merger->GetProperty(HistogramMerger::Id, "default_id") == mergerId);
    BOOST_CHECK(merger->GetProperty(HistogramMerger::NumIoThreads, 0) == numberOfThreads);
}

BOOST_AUTO_TEST_CASE(checkClass)
{
	Dummy dummy = Dummy();
	BOOST_CHECK(dummy.returnThree() == 3);
}

BOOST_AUTO_TEST_CASE(CheckFakeIt)
{
	using namespace fakeit;
	Mock<Dummy> mock;

	When(Method(mock, returnThree)).Return(1);

	Dummy &ref = mock.get();

	BOOST_CHECK(ref.returnThree() == 1);
}

BOOST_AUTO_TEST_SUITE_END()
