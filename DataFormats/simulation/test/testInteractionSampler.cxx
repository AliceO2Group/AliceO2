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
#include <TH1F.h>

namespace o2
{

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
