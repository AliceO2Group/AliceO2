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
#include "Framework/AlgorithmSpec.h"
#include "Framework/ServiceRegistry.h"
#include "Framework/runDataProcessing.h"
#include "Framework/ControlService.h"
#include "Framework/ControlService.h"
#include "Framework/Logger.h"
#include "Framework/TableBuilder.h"
#include "Framework/TableConsumer.h"
#include <Monitoring/Monitoring.h>
#include <ROOT/RDataFrame.hxx>
#include <ROOT/RArrowDS.hxx>
#include <memory>

using namespace o2::framework;
using DataHeader = o2::header::DataHeader;

template class std::shared_ptr<arrow::Buffer>;

/// Example of how to use ROOT::RDataFrame using DPL.
WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  return WorkflowSpec{
    //
    DataProcessorSpec{
      "rdataframe_producer", //
      Inputs{},              //
      {
        OutputSpec{{"xz"}, "TES", "RFRAME"}, //
      },
      AlgorithmSpec{adaptStateless([](DataAllocator& outputs) {
        // We ask the framework for something which can build a Table
        auto& out = outputs.make<TableBuilder>(Output{"TES", "RFRAME"});
        // We use RDataFrame to create a few columns with 100 rows.
        // The final action is the one which allows the user to create the
        // output message.
        //
        // FIXME: bloat in the code I'd like to get rid of:
        //
        // * I need to specify the types for the columns
        // * I need to specify the names of the columns twice
        ROOT::RDataFrame rdf(100);
        auto t = rdf.Define("x", "1.f")
                   .Define("y", "2.f")
                   .Define("z", "x+y");
        t.ForeachSlot(out.persist<float, float>({"x", "z"}), {"x", "z"});
      })} //
    },    //
    DataProcessorSpec{
      "rdataframe_consumer", //
      {
        InputSpec{"xz", "TES", "RFRAME"}, //
      },                                  //
      Outputs{},                          //
      AlgorithmSpec{
        adaptStateless(
          [](InputRecord& inputs, ControlService& control) {
            /// This gets a table handle from the message.
            auto s = inputs.get<TableConsumer>("xz");

            /// From the handle, we construct the actual arrow table
            /// which is then used as a source for the RDataFrame.
            /// This is probably easy to change to a:
            ///
            /// auto rdf = ctx.inputs().get<RDataSource>("xz");
            auto table = s->asArrowTable();
            if (table->num_rows() != 100) {
              LOG(ERROR) << "Wrong number of entries for the arrow table" << table->num_rows();
            }

            if (table->num_columns() != 2) {
              LOG(ERROR) << "Wrong number of columns for the arrow table" << table->num_columns();
            }

            auto source = std::make_unique<ROOT::RDF::RArrowDS>(s->asArrowTable(), std::vector<std::string>{});
            ROOT::RDataFrame rdf(std::move(source));

            if (*rdf.Count() != 100) {
              LOG(ERROR) << "Wrong number of entries for the DataFrame" << *rdf.Count();
            }

            if (*rdf.Mean("z") - 3.f > 0.1f) {
              LOG(ERROR) << "Wrong average for z";
            }

            control.readyToQuit(QuitRequest::All);
          }) //
      }      //
    }        //
  };
}
