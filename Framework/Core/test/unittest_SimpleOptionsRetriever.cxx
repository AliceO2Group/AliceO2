// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#define BOOST_TEST_MODULE Framework SimpleOptionsRetriever
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include "Framework/SimpleOptionsRetriever.h"
#include <boost/test/unit_test.hpp>
#include "Framework/ParamRetriever.h"
#include "Framework/ConfigParamStore.h"

using namespace o2::framework;

BOOST_AUTO_TEST_CASE(TestInsertion)
{
  std::vector<ConfigParamSpec> specs{
    ConfigParamSpec{"foo", VariantType::Int, 1, {"foo"}}};
  boost::property_tree::ptree opt;
  opt.put<int>("foo", 123);
  std::vector<std::unique_ptr<ParamRetriever>> retrievers;
  std::unique_ptr<ParamRetriever> retriever{new SimpleOptionsRetriever(opt, "simple")};
  retrievers.emplace_back(std::move(retriever));

  ConfigParamStore store{specs, std::move(retrievers)};
  store.preload();
  store.activate();

  BOOST_CHECK_EQUAL(store.store().get<int>("foo"), 123);
}
