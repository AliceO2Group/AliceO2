// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#define BOOST_TEST_MODULE Test Framework AnalysisTask
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include "Framework/AnalysisTask.h"
#include "Framework/AnalysisDataModel.h"

#include <boost/test/unit_test.hpp>

#include <random>

using namespace o2;
using namespace o2::framework;

namespace o2::aod
{
namespace track
{
DECLARE_SOA_COLUMN(Foo, foo, float, "fBar");
DECLARE_SOA_COLUMN(Bar, bar, float, "fFoo");
DECLARE_SOA_DYNAMIC_COLUMN(Sum, sum, [](float x, float y) { return x + y; });
} // namespace track
DECLARE_SOA_TABLE(FooBars, "AOD", "FOOBAR",
                  track::Foo, track::Bar,
                  track::Sum<track::Foo, track::Bar>);
} // namespace o2::aod

// FIXME: for the moment we do not derive from AnalysisTask as
// we need GCC 7.4+ to fix a bug.
struct FTask {
  HistogramRegistry registry{"registry",true,{
      {"eta","#Eta",{"TH1F",100,-2.0,2.0}},
      {"phi","#Phi",{"TH1D",102,0,2*M_PI}},
      {"pt","p_{T}",{"TH1D",1002,-0.01,50.1}}
                             }};
  void process(o2::aod::FooBars const& foobar)
  {
     registry.get("eta")->Print();
     registry.get("phi")->Print();
     registry.get("pt")->Print();
  }
};

BOOST_AUTO_TEST_CASE(AdaptorCompilation)
{
  auto task6 = FTask();

  std::default_random_engine e1(1234567891);
  std::uniform_real_distribution<float> uniform_dist(0, 1);

  TableBuilder builder;
  auto rowWriter = builder.cursor<o2::aod::FooBars>();
  for (size_t i = 0; i < 100; ++i) {
    rowWriter(0, uniform_dist(e1), uniform_dist(e1));
  }
  auto table = builder.finalize();

  o2::aod::FooBars tracks{table};
  task6.process(tracks);
}
