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

#define BOOST_TEST_MODULE Test InteractionSampler class
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include "SimulationDataFormat/InteractionSampler.h"
#include "CCDB/BasicCCDBManager.h"
#include "DataFormatsParameters/AggregatedRunInfo.h"
#include "DataFormatsParameters/GRPLHCIFData.h"
#include "TFile.h"
#include "TGrid.h"
#include "TRandom.h"
#include <TH1F.h>
#include <cmath>
#include <cstdint>
#include <map>
#include <vector>

namespace o2
{

namespace
{
// Per-crossing counts of a sample of collisions, split into classes of colliding BCs
struct CrossingStats {
  std::vector<double> slots;    // colliding BCs passed over, per class
  std::vector<double> occupied; // BCs with at least one collision, per class
  std::vector<double> multi;    // BCs with at least two collisions, per class
  std::vector<double> colls;    // collisions, per class
};

// Counts the collisions of a sampler per crossing; classOf maps the index of a colliding BC to its class
template <typename Sampler, typename ClassOf>
CrossingStats countCrossings(Sampler& sampler, int nColl, int nClasses, ClassOf classOf)
{
  std::vector<o2::InteractionTimeRecord> samples;
  samples.reserve(nColl);
  sampler.generateCollisionTimes(samples);

  const auto& filling = sampler.getBunchFilling();
  std::vector<int> indexOfBC(o2::constants::lhc::LHCMaxBunches, -1);
  int nColliding = 0;
  for (int bc = 0; bc < o2::constants::lhc::LHCMaxBunches; ++bc) {
    if (filling.testBC(bc)) {
      indexOfBC[bc] = nColliding++;
    }
  }

  std::map<int64_t, int> perCrossing; // key: global index of the colliding BC
  int64_t prevKey = -1;
  for (const auto& ir : samples) {
    BOOST_REQUIRE(ir.bc < o2::constants::lhc::LHCMaxBunches && indexOfBC[ir.bc] >= 0); // only colliding BCs may be sampled
    const int64_t key = int64_t(ir.orbit) * nColliding + indexOfBC[ir.bc];
    BOOST_REQUIRE(key >= prevKey); // records come in time order
    prevKey = key;
    perCrossing[key]++;
  }

  CrossingStats st{std::vector<double>(nClasses), std::vector<double>(nClasses), std::vector<double>(nClasses), std::vector<double>(nClasses)};
  const auto first = perCrossing.begin()->first;
  const auto last = perCrossing.rbegin()->first;
  for (auto key = first; key <= last; ++key) {
    st.slots[classOf(int(key % nColliding))] += 1;
  }
  for (const auto& [key, n] : perCrossing) {
    const int c = classOf(int(key % nColliding));
    st.occupied[c] += 1;
    st.multi[c] += (n > 1);
    st.colls[c] += n;
  }
  return st;
}

// Poisson probability of at least two collisions in a crossing, given at least one
double poissonMultiGivenOccupied(double mu)
{
  const double p0 = std::exp(-mu);
  return (1. - p0 - mu * p0) / (1. - p0);
}
} // namespace

// Non-uniform sampler with uniform scales must reproduce Poisson statistics
BOOST_AUTO_TEST_CASE(NonUniformSamplerUniformScalesIsPoisson)
{
  const int nColl = 500000;
  auto allOneClass = [](int) { return 0; };

  gRandom->SetSeed(12345);
  o2::steer::NonUniformMuInteractionSampler sampler;
  sampler.setMuPerBC(0.2);
  sampler.init();
  // the constructor already sets unit scales; setting them again also exercises the normalisation
  BOOST_REQUIRE(sampler.setBCIntensityScales(std::vector<float>(sampler.getBunchFilling().getNBunches(), 1.f)));
  auto st = countCrossings(sampler, nColl, 1, allOneClass);

  const double mu = sampler.getMuPerBC();
  const double expMulti = poissonMultiGivenOccupied(mu);
  const double expOccupied = 1. - std::exp(-mu);
  // relative statistical precision of the multi fraction is about 0.5% here, of the others well below that
  BOOST_CHECK_CLOSE(st.multi[0] / st.occupied[0], expMulti, 5.);
  BOOST_CHECK_CLOSE(st.occupied[0] / st.slots[0], expOccupied, 2.);
  BOOST_CHECK_CLOSE(st.colls[0] / st.slots[0], mu, 2.);
}

// Non-uniform sampler must give each colliding BC the Poisson statistics of its own mu
BOOST_AUTO_TEST_CASE(NonUniformSamplerPerBCPoisson)
{
  const int nColl = 500000;
  const std::vector<double> scale{0.5, 1.5}; // even and odd colliding BCs
  auto evenOdd = [](int idx) { return idx % 2; };

  gRandom->SetSeed(54321);
  o2::steer::NonUniformMuInteractionSampler sampler;
  sampler.setMuPerBC(0.2);
  sampler.init();
  std::vector<float> scales(sampler.getBunchFilling().getNBunches());
  for (size_t i = 0; i < scales.size(); ++i) {
    scales[i] = scale[evenOdd(i)];
  }
  BOOST_REQUIRE(sampler.setBCIntensityScales(scales));
  auto st = countCrossings(sampler, nColl, 2, evenOdd);

  for (int c = 0; c < 2; ++c) {
    const double muC = sampler.getMuPerBC() * sampler.getBCIntensityScales()[c];
    BOOST_TEST_MESSAGE("class " << c << " mu " << muC << " multi/occupied " << st.multi[c] / st.occupied[c]
                                << " expected " << poissonMultiGivenOccupied(muC));
    BOOST_CHECK_CLOSE(st.multi[c] / st.occupied[c], poissonMultiGivenOccupied(muC), 7.);
    BOOST_CHECK_CLOSE(st.occupied[c] / st.slots[c], 1. - std::exp(-muC), 2.);
    BOOST_CHECK_CLOSE(st.colls[c] / st.slots[c], muC, 2.);
  }
}

// Scales of the wrong length must be rejected and leave the previous scales untouched
BOOST_AUTO_TEST_CASE(NonUniformSamplerRejectsWrongScaleLength)
{
  o2::steer::NonUniformMuInteractionSampler sampler;
  sampler.setMuPerBC(0.2);
  sampler.init();
  const auto nBC = sampler.getBunchFilling().getNBunches();
  BOOST_REQUIRE(sampler.setBCIntensityScales(std::vector<float>(nBC, 1.f)));
  const auto before = sampler.getBCIntensityScales();
  BOOST_CHECK(!sampler.setBCIntensityScales(std::vector<float>(nBC + 1, 2.f)));
  BOOST_CHECK(sampler.getBCIntensityScales() == before);
}

BOOST_AUTO_TEST_CASE(NonUniformSampler)
{
  auto run_number = 559827;
  TGrid::Connect("alien");
  if (gGrid) {
    auto runInfo = o2::parameters::AggregatedRunInfo::buildAggregatedRunInfo(o2::ccdb::BasicCCDBManager::instance(), run_number);

    o2::steer::NonUniformMuInteractionSampler sampler;
    sampler.setBunchFilling(runInfo.grpLHC->getBunchFilling());

    // the test distribution provided by Igor Altsybeev
    auto distr_file = TFile::Open("alien:///alice/cern.ch/user/s/swenzel/AliceO2_TestData/NBcVTX_559827/hBcTVX_data_PbPb_24ar_559827.root");

    //
    if (distr_file && !distr_file->IsZombie()) {
      auto hist = distr_file->Get<TH1F>("hBcTVX");
      if (hist) {
        sampler.init();
        sampler.setBCIntensityScales(*hist);

        // sample into a vector of a certain size
        std::vector<o2::InteractionTimeRecord> samples;

        int N = 100000;
        samples.resize(N);

        sampler.generateCollisionTimes(samples);

        // fill an output histogram
        auto output_hist = (TH1F*)hist->Clone("h2"); // make a full copy
        output_hist->Reset();

        for (const auto& sample : samples) {
          output_hist->Fill(sample.bc);
        }

        // Write out
        auto fout = TFile::Open("NBCVTX_out.root", "RECREATE");
        fout->WriteObject(output_hist, "NBcVTX");
        fout->Close();

        // compare mean values of original and newly sampled hist
        BOOST_CHECK_CLOSE(hist->GetMean(), output_hist->GetMean(), 0.5);
      }
    }
  }
}

} // namespace o2
