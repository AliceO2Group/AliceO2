#define BOOST_TEST_MODULE Merger
#define BOOST_TEST_MAIN

#include <TH1F.h>
#include <boost/test/unit_test.hpp>
#include <memory>
#include <vector>

#include "QCMerger/Merger.h"
#include "QCMerger/MergerDevice.h"

using namespace std;

namespace
{
const int NUMBER_OF_QC_OBJECTS_FOR_COMPLETE_DATA = 2;
const char* HISTOGRAM_NAME = "TEST_NAME";
const char* HISTOGRAM_TITLE = "HISTOGRAM_TITLE";
const char* RANDOM_GENERATION_TYPE = "gaus";
const int NUMBER_OF_BINS = 100;
const int NUMBER_OF_ENTRIES = 100;
const double X_LOW = -10.0;
const double X_UP = 10.0;
}

BOOST_AUTO_TEST_SUITE(MergerTestSuite)

BOOST_AUTO_TEST_CASE(mergeTenHistograms)
{
  const unsigned HISTOGRAMS_TO_TEST = 10;
  unique_ptr<Merger> merger(new Merger(NUMBER_OF_QC_OBJECTS_FOR_COMPLETE_DATA));
  vector<TH1F*> histograms;

  for (int i = 0; i < HISTOGRAMS_TO_TEST; ++i) {
    histograms.push_back(new TH1F(HISTOGRAM_NAME, HISTOGRAM_TITLE, NUMBER_OF_BINS, X_LOW, X_UP));
    histograms.at(i)->FillRandom(RANDOM_GENERATION_TYPE, NUMBER_OF_ENTRIES);
  }

  for (int i = 0; i < HISTOGRAMS_TO_TEST; ++i) {
    TObject* mergedObject = merger->mergeObject(histograms.at(i));

    if (i % NUMBER_OF_QC_OBJECTS_FOR_COMPLETE_DATA != 0) {
      BOOST_TEST(mergedObject->GetName() == HISTOGRAM_NAME, "Wrong name of histogram: " << mergedObject->GetName());
      BOOST_TEST(reinterpret_cast<TH1F*>(mergedObject)->GetEntries() ==
                 (NUMBER_OF_QC_OBJECTS_FOR_COMPLETE_DATA * NUMBER_OF_ENTRIES));

      delete mergedObject;
    } else {
      if (mergedObject != nullptr) {
        BOOST_FAIL("Object should not be merged " << mergedObject->GetName());
      }
    }
  }
}

BOOST_AUTO_TEST_SUITE_END()
