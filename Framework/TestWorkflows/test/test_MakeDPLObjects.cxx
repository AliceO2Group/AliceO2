// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include "Framework/runDataProcessing.h"
#include "Framework/DataRefUtils.h"
#include "Framework/Collection.h"
#include "Framework/ControlService.h"
#include <TH1F.h>

using namespace o2::framework;

struct XYZ {
  float x;
  float y;
  float z;
};

void defineDataProcessing(std::vector<DataProcessorSpec> &specs) {
  WorkflowSpec workflow = {
    DataProcessorSpec{
      "source",
      Inputs{},
      {
        OutputSpec{"TST", "HISTO"},
        OutputSpec{"TST", "POINT"},
        OutputSpec{"TST", "POINTS"}
      },
      AlgorithmSpec{
        [](ProcessingContext &ctx) {
          // A new message with 1 XYZ instance in it
          XYZ &x = ctx.allocator().make<XYZ>(OutputSpec{"TST", "POINT", 0});
          // A new message with a Collection<XYZ> with 1000 items
          Collection<XYZ> y = ctx.allocator().make<XYZ>(OutputSpec{"TST", "POINTS", 0}, 1000);
          // A new message with a TH1F inside
          auto h = ctx.allocator().make<TH1F>(OutputSpec{"TST", "HISTO"},
                                             "h", "test", 100, -10., 10.);
        }
      }
    },
    DataProcessorSpec{
      "dest",
      Inputs{
        InputSpec{"point", "TST", "POINT"},
        InputSpec{"points", "TST", "POINTS"},
        InputSpec{"histo", "TST", "HISTO"}
      },
      {},
      AlgorithmSpec{
        [](ProcessingContext &ctx) {
          // A new message with a TH1F inside
          auto h = ctx.inputs().get<TH1F>("histo");
          // A new message with 1 XYZ instance in it
          XYZ const &x = ctx.inputs().get<XYZ>("point");
          // A new message with a Collection<XYZ> with 1000 items
          auto ref = ctx.inputs().get("points");
          Collection<XYZ> c = DataRefUtils::as<XYZ>(ref);
          ctx.services().get<ControlService>().readyToQuit(true);
        }
      }
    }
  };

  specs.swap(workflow);
}
