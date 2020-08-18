// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#define BOOST_TEST_MODULE Test Framework HistogramRegistry
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include "Framework/HistogramRegistry.h"
#include <boost/test/unit_test.hpp>

using namespace o2;
using namespace o2::framework;

HistogramRegistry foo()
{
  return {"r", true, {{"histo", "histo", {"TH1F", 100, 0, 1}}}};
}

BOOST_AUTO_TEST_CASE(HistogramRegistryLookup)
{
  /// Construct a registry object with direct declaration
  HistogramRegistry registry{"registry", true, {{"eta", "#Eta", {"TH1F", 100, -2.0, 2.0}}, {"phi", "#Phi", {"TH1D", 102, 0, 2 * M_PI}}, {"pt", "p_{T}", {"TH1D", 1002, -0.01, 50.1}}}};

  /// Get histograms by name
  BOOST_REQUIRE_EQUAL(registry.get("eta")->GetNbinsX(), 100);
  BOOST_REQUIRE_EQUAL(registry.get("phi")->GetNbinsX(), 102);
  BOOST_REQUIRE_EQUAL(registry.get("pt")->GetNbinsX(), 1002);

  /// Get a pointer to the histogram
  auto histo = registry.get("pt").get();
  BOOST_REQUIRE_EQUAL(histo->GetNbinsX(), 1002);

  /// Get registry object from a function
  auto r = foo();
  auto histo2 = r.get("histo").get();
  BOOST_REQUIRE_EQUAL(histo2->GetNbinsX(), 100);
}
