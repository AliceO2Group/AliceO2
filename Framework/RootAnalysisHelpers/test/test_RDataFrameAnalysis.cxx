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
// FIXME: for now RDataFrame needs to be the first header, because of incompatibilities
// with our HistogramRegistry
#include <ROOT/RDataFrame.hxx>
#include <ROOT/RArrowDS.hxx>
#include "Framework/runDataProcessing.h"
#include "Framework/AnalysisTask.h"
#include <memory>

using namespace o2::framework;
using namespace o2::aod;

// FIXME: this is just to simplify the test, one can simply use the
// AnalysisDataModel.h and a ROOT file with the data
// FIXME: for now you need to be inside the o2::aod namespace to use the AOD types
namespace o2::aod::test
{
DECLARE_SOA_COLUMN_FULL(X, x, int32_t, "x");
DECLARE_SOA_COLUMN_FULL(Y, y, int32_t, "y");
} // namespace o2::aod::test

DECLARE_SOA_TABLE(Points, "TST", "POINTS", o2::aod::test::X, o2::aod::test::Y);

struct Generator {
  Produces<Points> points;
  void process(o2::framework::Enumeration<0, 1>&)
  {
    points(0, 0);
    points(1, 1);
    points(2, 2);
    points(3, 3);
    points(4, 4);
    points(5, 5);
    points(6, 6);
    points(7, 7);
    points(8, 8);
    points(9, 9);
  }
};

struct RDataframeConsumer {
  Service<ControlService> control;
  void process(Points& points)
  {
    auto source = std::make_unique<ROOT::RDF::RArrowDS>(points.asArrowTable(), std::vector<std::string>{});
    ROOT::RDataFrame rdf(std::move(source));

    if (*rdf.Count() != 10) {
      LOG(error) << "Wrong number of entries for the DataFrame" << *rdf.Count();
    }
    LOG(info) << "DataFrame has " << *rdf.Mean("x") << " mean";

    control->readyToQuit(QuitRequest::All);
  }
};

/// Example of how to use ROOT::RDataFrame using DPL.
WorkflowSpec defineDataProcessing(ConfigContext const& cfg)
{
  return WorkflowSpec{
    adaptAnalysisTask<Generator>(cfg),
    adaptAnalysisTask<RDataframeConsumer>(cfg)};
}
