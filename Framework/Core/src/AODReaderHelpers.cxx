// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "Framework/AODReaderHelpers.h"
#include "Framework/RootTableBuilderHelpers.h"
#include "Framework/AlgorithmSpec.h"
#include <ROOT/RDataFrame.hxx>
#include <TFile.h>

namespace o2::framework::readers
{

AlgorithmSpec AODReaderHelpers::rootFileReaderCallback()
{
  auto callback = AlgorithmSpec{ [](InitContext& initCtx) {
    LOG(INFO) << "This is not a real device, merely a placeholder for external inputs";
    LOG(INFO) << "To be hidden / removed at some point.";
    std::shared_ptr<TFile> infile;
    try {
      auto filename = initCtx.options().get<std::string>("aod-file");
      infile = std::make_shared<TFile>(filename.c_str());
    } catch (...) {
      LOG(ERROR) << "Unable to open file";
    }
    return [infile](ProcessingContext& ctx) {
      if (infile.get() == nullptr || infile->IsOpen() == false) {
        LOG(ERROR) << "File not found: aod.root";
        return;
      }
      /// FIXME: Substitute here the actual data you want to convert for the AODReader
      {
        std::unique_ptr<TTreeReader> reader = std::make_unique<TTreeReader>("O2aod", infile.get());
        auto& trackParBuilder = ctx.outputs().make<TableBuilder>(Output{ "AOD", "TRACKPAR" });
        TTreeReaderValue<int> c0(*reader, "fVertexID");
        TTreeReaderValue<float> c1(*reader, "fX");
        TTreeReaderValue<float> c2(*reader, "fAlpha");
        TTreeReaderValue<float> c3(*reader, "fY");
        TTreeReaderValue<float> c4(*reader, "fZ");
        TTreeReaderValue<float> c5(*reader, "fSnp");
        TTreeReaderValue<float> c6(*reader, "fTgl");
        TTreeReaderValue<float> c7(*reader, "fSigned1Pt");
        RootTableBuilderHelpers::convertTTree(trackParBuilder, *reader,
                                              c0, c1, c2, c3, c4, c5, c6, c7);
      }

      {
        std::unique_ptr<TTreeReader> covReader = std::make_unique<TTreeReader>("O2aod", infile.get());
        TTreeReaderValue<float> c0(*covReader, "fCYY");
        TTreeReaderValue<float> c1(*covReader, "fCZY");
        TTreeReaderValue<float> c2(*covReader, "fCZZ");
        TTreeReaderValue<float> c3(*covReader, "fCSnpY");
        TTreeReaderValue<float> c4(*covReader, "fCSnpZ");
        TTreeReaderValue<float> c5(*covReader, "fCSnpSnp");
        TTreeReaderValue<float> c6(*covReader, "fCTglSnp");
        TTreeReaderValue<float> c7(*covReader, "fCTglTgl");
        TTreeReaderValue<float> c8(*covReader, "fC1PtY");
        TTreeReaderValue<float> c9(*covReader, "fC1PtZ");
        TTreeReaderValue<float> c10(*covReader, "fC1PtSnp");
        TTreeReaderValue<float> c11(*covReader, "fC1PtTgl");
        TTreeReaderValue<float> c12(*covReader, "fC1Pt21Pt2");
        auto& trackParCovBuilder = ctx.outputs().make<TableBuilder>(Output{ "AOD", "TRACKPARCOV" });
        RootTableBuilderHelpers::convertTTree(trackParCovBuilder, *covReader,
                                              c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12);
      }
      {
        std::unique_ptr<TTreeReader> extraReader = std::make_unique<TTreeReader>("O2aod", infile.get());
        TTreeReaderValue<float> c0(*extraReader, "fTPCinnerP");
        TTreeReaderValue<uint64_t> c1(*extraReader, "fFlags");
        TTreeReaderValue<unsigned char> c2(*extraReader, "fITSClusterMap");
        TTreeReaderValue<unsigned short> c3(*extraReader, "fTPCncls");
        TTreeReaderValue<unsigned char> c4(*extraReader, "fTRDntracklets");
        TTreeReaderValue<float> c5(*extraReader, "fITSchi2Ncl");
        TTreeReaderValue<float> c6(*extraReader, "fTPCchi2Ncl");
        TTreeReaderValue<float> c7(*extraReader, "fTRDchi2");
        // FIXME: not sure what is inside. Commenting for now
        // TTreeReaderValue<float> c8(*extraReader, "fTOFchi2fTOFchi2");
        TTreeReaderValue<float> c9(*extraReader, "fTPCsignal");
        TTreeReaderValue<float> c10(*extraReader, "fTRDsignal");
        TTreeReaderValue<float> c11(*extraReader, "fTOFsignal");
        TTreeReaderValue<float> c12(*extraReader, "fLength");
        auto& extraBuilder = ctx.outputs().make<TableBuilder>(Output{ "AOD", "TRACKEXTRA" });
        RootTableBuilderHelpers::convertTTree(extraBuilder, *extraReader,
                                              c0, c1, c2, c3, c4, c5, c6, c7, c9, c10, c11); // c8 is missing.
      }

    };
  } };

  return callback;
}

AlgorithmSpec AODReaderHelpers::fakeReaderCallback()
{
  return AlgorithmSpec{
    [](InitContext& setup) {
      return [](ProcessingContext& ctx) {
        /// We get the table builder for track param.
        auto& trackParBuilder = ctx.outputs().make<TableBuilder>(Output{ "AOD", "TRACKPAR" });
        auto& trackParCovBuilder = ctx.outputs().make<TableBuilder>(Output{ "AOD", "TRACKPARCOV" });
        // We use RDataFrame to create a few columns with 100 rows.
        // The final action is the one which allows the user to create the
        // output message.
        //
        // FIXME: bloat in the code I'd like to get rid of:
        //
        // * I need to specify the types for the columns
        // * I need to specify the names of the columns twice
        // * I should use the RDataFrame to read / convert from the ESD...
        //   Using dummy values for now.
        ROOT::RDataFrame rdf(100);
        auto trackParRDF = rdf.Define("mX", "1.f")
                             .Define("mAlpha", "2.f")
                             .Define("y", "3.f")
                             .Define("z", "4.f")
                             .Define("snp", "5.f")
                             .Define("tgl", "6.f")
                             .Define("qpt", "7.f");

        /// FIXME: think of the best way to include the non-diag elements.
        auto trackParCorRDF = rdf.Define("sigY", "1.f")
                                .Define("sigZ", "2.f")
                                .Define("sigSnp", "3.f")
                                .Define("sigTgl", "4.f")
                                .Define("sigQpt", "5.f");

        /// FIXME: we need to do some cling magic to hide all of this.
        trackParRDF.ForeachSlot(trackParBuilder.persist<float, float, float, float, float, float, float>(
                                  { "mX", "mAlpha", "y", "z", "snp", "tgl", "qpt" }),
                                { "mX", "mAlpha", "y", "z", "snp", "tgl", "qpt" });

        trackParCorRDF.ForeachSlot(trackParCovBuilder.persist<float, float, float, float, float>(
                                     { "sigY", "sigZ", "sigSnp", "sigTgl", "sigQpt" }),
                                   { "sigY", "sigZ", "sigSnp", "sigTgl", "sigQpt" });

      };
    }
  };
}

} // namespace o2::framework::readers
