// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include "Framework/DataRefUtils.h"
#include "Framework/ServiceRegistry.h"
#include "Framework/runDataProcessing.h"
#include <Monitoring/Monitoring.h>
#include "Headers/DataHeader.h"
// FIXME: this should not be needed as the framework should be able to
//        decode TClonesArray by itself.
#include "Framework/TMessageSerializer.h"
#include "Framework/Logger.h"
#include <TClonesArray.h>
#include <TH1F.h>
#include <TString.h>

#include <chrono>

using namespace o2::framework;
using DataHeader = o2::header::DataHeader;
using DataOrigin = o2::header::DataOrigin;
using DataDescription = o2::header::DataDescription;

// This is how you can define your processing in a declarative way
WorkflowSpec defineDataProcessing(ConfigContext const&) {
  return WorkflowSpec{
    { "producer",
      {},
      { OutputSpec{ "TST", "HISTOS" },
        OutputSpec{ "TST", "STRING" } },
      AlgorithmSpec{
        [](ProcessingContext& ctx) {
          std::this_thread::sleep_for(std::chrono::seconds(1));
          // Create an histogram
          auto& singleHisto = ctx.outputs().make<TH1F>(Output{ "TST", "HISTOS", 0 }, "h1", "test", 100, -10., 10.);
          auto& aString = ctx.outputs().make<TObjString>(Output{ "TST", "STRING", 0 }, "fao");
          singleHisto.FillRandom("gaus", 1000);
          Double_t stats[4];
          singleHisto.GetStats(stats);
          LOG(INFO) << "sumw" << stats[0] << "\n"
                    << "sumw2" << stats[1] << "\n"
                    << "sumwx" << stats[2] << "\n"
                    << "sumwx2" << stats[3] << "\n";
          aString.SetString("foo");
        } } },
    { "consumer",
      {
        InputSpec{ "histos", "TST", "HISTOS" },
        InputSpec{ "string", "TST", "STRING" },
      },
      {},
      AlgorithmSpec{
        [](ProcessingContext& ctx) {
          // FIXME: for the moment we need to do the deserialization ourselves.
          //        this should probably be encoded in the serialization field
          //        of the DataHeader and done automatically by the framework
          auto h = ctx.inputs().get<TH1F*>("histos");
          if (h.get() == nullptr) {
            throw std::runtime_error("Missing output");
          }
          Double_t stats[4];
          h->GetStats(stats);
          LOG(INFO) << "sumw" << stats[0] << "\n"
                    << "sumw2" << stats[1] << "\n"
                    << "sumwx" << stats[2] << "\n"
                    << "sumwx2" << stats[3] << "\n";
          auto s = ctx.inputs().get<TObjString*>("string");

          LOG(INFO) << "String is " << s->GetString().Data();
        } } }
  };
}
