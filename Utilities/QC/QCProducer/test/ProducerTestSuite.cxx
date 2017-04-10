#define BOOST_TEST_MODULE ProducerTest
#define BOOST_TEST_MAIN

#include <memory>
#include <string>

#include <boost/test/unit_test.hpp>

#include <TH1.h>
#include <TH2.h>
#include <TH3.h>
#include <THn.h>
#include <TTree.h>

#include "QCProducer/ProducerDevice.h"
#include "QCProducer/TH1Producer.h"
#include "QCProducer/TH2Producer.h"
#include "QCProducer/TH3Producer.h"
#include "QCProducer/THnProducer.h"
#include "QCProducer/TreeProducer.h"

using namespace std;

namespace
{
const int NUMBER_OF_BINS = 10;
const char* NAME = "TEST_NAME";
const char* TITLE = "TEST_TITLE";

void validateHistogramParameters(TH1* object)
{
  BOOST_TEST(object->GetNbinsX() == NUMBER_OF_BINS,
             string("Invalid number of x bins: ").append(to_string(object->GetNbinsX())));
  BOOST_TEST(object->GetName() == NAME, string("Invalid name of histogram: ").append(object->GetName()));
  BOOST_TEST(object->GetTitle() == TITLE, string("Invalid title of histogram: ").append(object->GetTitle()));
}
}

BOOST_AUTO_TEST_SUITE(ProducerTestSuite)

BOOST_AUTO_TEST_CASE(produceTH1F)
{
  unique_ptr<TH1Producer> histogramProducer(new TH1Producer(NAME, TITLE, NUMBER_OF_BINS));
  unique_ptr<TH1> histogram(dynamic_cast<TH1*>(histogramProducer->produceData()));

  validateHistogramParameters(histogram.get());
}

BOOST_AUTO_TEST_CASE(produceTH2F)
{
  unique_ptr<TH2Producer> histogramProducer(new TH2Producer(NAME, TITLE, NUMBER_OF_BINS));
  unique_ptr<TH2> histogram(dynamic_cast<TH2*>(histogramProducer->produceData()));

  validateHistogramParameters(histogram.get());
  BOOST_TEST(histogram->GetNbinsY() == NUMBER_OF_BINS,
             string("Invalid number of y bins: ").append(to_string(histogram->GetNbinsX())));
}

BOOST_AUTO_TEST_CASE(produceTH3F)
{
  unique_ptr<TH3Producer> histogramProducer(new TH3Producer(NAME, TITLE, NUMBER_OF_BINS));
  unique_ptr<TH3> histogram(dynamic_cast<TH3*>(histogramProducer->produceData()));

  validateHistogramParameters(histogram.get());
  BOOST_TEST(histogram->GetNbinsY() == NUMBER_OF_BINS,
             string("Invalid number of y bins: ").append(to_string(histogram->GetNbinsX())));
  BOOST_TEST(histogram->GetNbinsZ() == NUMBER_OF_BINS,
             string("Invalid number of z bins: ").append(to_string(histogram->GetNbinsX())));
}

BOOST_AUTO_TEST_CASE(produceTHnF)
{
  unique_ptr<THnProducer> histogramProducer(new THnProducer(NAME, TITLE, NUMBER_OF_BINS));
  unique_ptr<THn> histogram(dynamic_cast<THn*>(histogramProducer->produceData()));

  BOOST_TEST(histogram->GetName() == NAME, string("Invalid name of histogram: ").append(histogram->GetName()));
  BOOST_TEST(histogram->GetTitle() == TITLE, string("Invalid title of histogram: ").append(histogram->GetTitle()));
}

BOOST_AUTO_TEST_CASE(produceTTree)
{
  const char* FIRST_BRANCH_NAME = "default_branch_name_0";
  const char* SECOND_BRANCH_NAME = "default_branch_name_1";
  const char* INVALID_BRANCH_NAME = "default_branch_name_2";
  const int NUMBER_OF_BRANCHES = 2;
  const int NUMBER_OF_ENTRIES = 10;

  unique_ptr<TreeProducer> treeProducer(new TreeProducer(NAME, TITLE, NUMBER_OF_BRANCHES, NUMBER_OF_ENTRIES));
  unique_ptr<TTree> tree(dynamic_cast<TTree*>(treeProducer->produceData()));

  BOOST_TEST(tree->GetName() == NAME, string("Invalid name of tree: ").append(tree->GetName()));
  BOOST_TEST(tree->GetTitle() == TITLE, string("Invalid title of tree: ").append(tree->GetTitle()));
  BOOST_TEST(tree->GetBranchStatus(FIRST_BRANCH_NAME) == true, "Invalid name of first branch");
  BOOST_TEST(tree->GetBranchStatus(SECOND_BRANCH_NAME) == true, "Invalid name of second branch");
  BOOST_TEST(tree->GetBranchStatus(INVALID_BRANCH_NAME) == false, "Invalid branch exists");
  BOOST_TEST(tree->GetEntries() == NUMBER_OF_BRANCHES * NUMBER_OF_ENTRIES, "Invalid number of entries in tree");
}

BOOST_AUTO_TEST_SUITE_END()
