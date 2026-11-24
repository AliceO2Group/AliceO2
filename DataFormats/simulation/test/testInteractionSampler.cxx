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
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <set>
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

// Collisions per occupied crossing of a sampler, in time order
struct CrossingSequence {
  std::vector<int> mult; // collisions per occupied crossing
  int64_t slots = 0;     // colliding BCs passed over
  std::set<float> times; // distinct collision times within the BC
};

template <typename Sampler>
CrossingSequence sequenceOfCrossings(Sampler& sampler, int nColl)
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

  CrossingSequence seq;
  int64_t firstKey = -1, prevKey = -1;
  for (const auto& ir : samples) {
    BOOST_REQUIRE(ir.bc < o2::constants::lhc::LHCMaxBunches && indexOfBC[ir.bc] >= 0);
    const int64_t key = int64_t(ir.orbit) * nColliding + indexOfBC[ir.bc];
    BOOST_REQUIRE(key >= prevKey);
    if (key != prevKey) {
      seq.mult.push_back(0);
    }
    if (firstKey < 0) {
      firstKey = key;
    }
    seq.mult.back()++;
    seq.times.insert(ir.timeInBCNS);
    prevKey = key;
  }
  seq.slots = prevKey - firstKey + 1;
  return seq;
}

// Poisson probability of k collisions in a crossing, given at least one
double poissonGivenOccupied(double mu, int k)
{
  return std::exp(-mu) * std::pow(mu, k) / std::tgamma(k + 1.) / (1. - std::exp(-mu));
}

// Checks that the uniform sampler gives Poisson crossing multiplicities without periodic patterns
void checkUniformSamplerIsPoisson(double muBC, unsigned int seed, const std::vector<double>& tolPercent)
{
  const int nColl = 500000;
  gRandom->SetSeed(seed);
  o2::steer::InteractionSampler sampler;
  sampler.setMuPerBC(muBC);
  sampler.init();
  auto seq = sequenceOfCrossings(sampler, nColl);
  const double mu = sampler.getMuPerBC();
  const double nOcc = seq.mult.size();

  std::vector<double> nK(4, 0.);
  std::set<int> residues; // positions modulo 1000 of the crossings with at least two collisions
  for (size_t i = 0; i < seq.mult.size(); ++i) {
    nK[std::min(seq.mult[i], 3)] += 1;
    if (seq.mult[i] > 1) {
      residues.insert(int(i % 1000));
    }
  }
  for (int k = 1; k <= 3; ++k) {
    const double expK = k < 3 ? poissonGivenOccupied(mu, k) : 1. - poissonGivenOccupied(mu, 1) - poissonGivenOccupied(mu, 2);
    BOOST_TEST_MESSAGE("mu " << mu << " k" << (k < 3 ? "=" : ">=") << k << " observed " << nK[k] << " expected " << expK * nOcc);
    BOOST_CHECK(nK[k] > 0);
    BOOST_CHECK_CLOSE(nK[k] / nOcc, expK, tolPercent[k - 1]);
  }
  BOOST_CHECK_CLOSE(nOcc / seq.slots, 1. - std::exp(-mu), 1.);
  BOOST_TEST_MESSAGE("mu " << mu << " residues mod 1000 of k>=2 crossings " << residues.size()
                           << ", distinct times in BC " << seq.times.size());
  // pre-generated rings of 1000 values repeat the same multiplicities and times every 1000 draws
  BOOST_CHECK(residues.size() > 900);
  BOOST_CHECK(seq.times.size() > 10000);
}
} // namespace

// Uniform sampler at small mu must reproduce Poisson statistics, including triple collisions
BOOST_AUTO_TEST_CASE(UniformSamplerIsPoissonSmallMu)
{
  checkUniformSamplerIsPoisson(0.05, 13579, {0.5, 5., 25.});
}

// Uniform sampler at large mu must reproduce Poisson statistics
BOOST_AUTO_TEST_CASE(UniformSamplerIsPoissonLargeMu)
{
  checkUniformSamplerIsPoisson(0.3, 24680, {0.5, 3., 7.});
}

// Zero-truncated Poisson draw must reproduce its distribution, mean and variance from very small to large mu
BOOST_AUTO_TEST_CASE(ZeroTruncatedPoissonDraw)
{
  gRandom->SetSeed(4321);
  const int n = 1000000;
  for (double mu : {2e-4, 2e-3, 0.1, 2., 6.5, 20., 29.9, 30.1, 100.}) {
    std::map<int, int> counts;
    double sum = 0., sum2 = 0.;
    for (int i = 0; i < n; ++i) {
      const int k = o2::steer::InteractionSampler::drawZeroTruncatedPoisson(mu);
      BOOST_REQUIRE(k >= 1);
      counts[k]++;
      sum += k;
      sum2 += double(k) * k;
    }
    const double expMean = mu / -std::expm1(-mu);
    const double expVar = expMean * (1. + mu - expMean);
    const double mean = sum / n;
    const double var = sum2 / n - mean * mean;
    BOOST_TEST_MESSAGE("mu " << mu << " mean " << mean << " expected " << expMean << " variance " << var << " expected " << expVar);
    BOOST_CHECK(std::abs(mean - expMean) < 5. * std::sqrt(expVar / n) + 1e-12);
    double m4 = 0.; // fourth central moment, for the standard error of the variance
    for (const auto& [k, c] : counts) {
      m4 += c * std::pow(k - mean, 4) / n;
    }
    BOOST_CHECK(std::abs(var - expVar) < 5. * std::sqrt(std::max(m4 - var * var, 0.) / n) + 1e-9);
    double p = mu * std::exp(-mu) / -std::expm1(-mu);
    for (int k = 1; k <= 4; ++k) {
      const double expected = n * p;
      const double tolerance = 5. * std::sqrt(expected) + 1.;
      BOOST_TEST_MESSAGE("mu " << mu << " k=" << k << " observed " << counts[k] << " expected " << expected);
      BOOST_CHECK(std::abs(counts[k] - expected) < tolerance);
      p *= mu / (k + 1);
    }
  }
}

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
