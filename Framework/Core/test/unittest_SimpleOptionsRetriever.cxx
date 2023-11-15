// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "Framework/SimpleOptionsRetriever.h"
#include "Framework/ParamRetriever.h"
#include "Framework/ConfigParamStore.h"
#include <catch_amalgamated.hpp>

using namespace o2::framework;

TEST_CASE("TestInsertion")
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

  CHECK(store.store().get<int>("foo") == 123);
}
