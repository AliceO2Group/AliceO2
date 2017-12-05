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
#include "Framework/ControlService.h"
#include <TH1F.h>
#include <TNamed.h>
#include <gsl/gsl>

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
        OutputSpec{"TST", "POINTS"},
        OutputSpec{"TST", "VECTOR"},
        OutputSpec{"TST", "LINEARIZED"},
        OutputSpec{"TST", "OBJECT"}
      },
      AlgorithmSpec{
        [](ProcessingContext &ctx) {
          // A new message with 1 XYZ instance in it
          XYZ &x = ctx.allocator().make<XYZ>(OutputSpec{"TST", "POINT", 0});
          // A new message with a gsl::span<XYZ> with 1000 items
          gsl::span<XYZ> y = ctx.allocator().make<XYZ>(OutputSpec{"TST", "POINTS", 0}, 1000);
          y[0] = XYZ{1,2,3};
          y[999] = XYZ{1,2,3};
          // A new message with a TH1F inside
          auto h = ctx.allocator().make<TH1F>(OutputSpec{"TST", "HISTO"},
                                             "h", "test", 100, -10., 10.);
          // A snapshot for an std::vector
          std::vector<XYZ> v{1000};
          v[0] = XYZ{1,2,3};
          v[999] = XYZ{1,2,3};
          ctx.allocator().snapshot(OutputSpec{"TST", "VECTOR"}, v);
          v[999] = XYZ{2,3,4};

          // A snapshot for an std::vector of pointers to objects
          // simply make a vector of pointers, snapshot will include the latest
          // change, but not the one which is done after taking the snapshot
          std::vector<XYZ*> p;
          for (auto & i : v) p.push_back(&i);
          ctx.allocator().snapshot(OutputSpec{"TST", "LINEARIZED"}, p);
          v[999] = XYZ{3,4,5};

          TNamed named("named", "a named test object");
          ctx.allocator().snapshot(OutputSpec{"TST", "OBJECT"}, named);
        }
      }
    },
    DataProcessorSpec{
      "dest",
      Inputs{
        InputSpec{"point", "TST", "POINT"},
        InputSpec{"points", "TST", "POINTS"},
        InputSpec{"histo", "TST", "HISTO"},
        InputSpec{"vector", "TST", "VECTOR"},
        InputSpec{"linearized", "TST", "LINEARIZED"},
        InputSpec{"object", "TST", "OBJECT"},
      },
      {},
      AlgorithmSpec{
        [](ProcessingContext &ctx) {
          // A new message with a TH1F inside
          auto h = ctx.inputs().get<TH1F>("histo");
          // A new message with 1 XYZ instance in it
          XYZ const &x = ctx.inputs().get<XYZ>("point");
          // A new message with a gsl::span<XYZ> with 1000 items
          auto ref1 = ctx.inputs().get("points");
          gsl::span<XYZ> c = DataRefUtils::as<XYZ>(ref1);
          assert(c[0].x == 1);
          assert(c[0].y == 2);
          assert(c[0].z == 3);
          assert(c[999].x == 1);
          assert(c[999].y == 2);
          assert(c[999].z == 3);
          auto ref2 = ctx.inputs().get("vector");
          gsl::span<XYZ> c2 = DataRefUtils::as<XYZ>(ref2);
          assert(c2[0].x == 1);
          assert(c2[0].y == 2);
          assert(c2[0].z == 3);
          assert(c2[999].x == 1);
          assert(c2[999].y == 2);
          assert(c2[999].z == 3);
          auto ref3 = ctx.inputs().get("linearized");
          gsl::span<XYZ> c3 = DataRefUtils::as<XYZ>(ref3);
          assert(c3[0].x == 1);
          assert(c3[0].y == 2);
          assert(c3[0].z == 3);
          assert(c3[999].x == 2);
          assert(c3[999].y == 3);
          assert(c3[999].z == 4);
          ctx.services().get<ControlService>().readyToQuit(true);
          auto o = ctx.inputs().get<TNamed>("object");
          assert(strcmp(o->GetName(), "named") == 0 &&
                 strcmp(o->GetTitle(), "a named test object") == 0);
        }
      }
    }
  };

  specs.swap(workflow);
}
