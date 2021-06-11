// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
///
/// \brief Write tables to a root file.
/// \author
/// \since

#include "Framework/runDataProcessing.h"
#include "Framework/AnalysisTask.h"

/// This example is to be used together with the aodreader example.
/// aodwriter creates three tables and writes them to two sets of files.
/// aodreader reads these files and creates related tables. aodwriter takes an
/// aod file with tracks as input.
///
/// USAGE:
///
///   o2-analysistutorial-aodwriter --aod-file AO2D_ppK0starToyMC_v3.root --json-file writerConfiguration.json
///   ls -1 treResults*.root > resultFiles.txt
///   ls -1 unodueResults*.root >> resultFiles.txt
///   o2-analysistutorial-aodreader --json-file readerConfiguration.json
///
/// writerConfiguration.json:
/// {
///   "OutputDirector": {
///     "debugmode": true,
///     "resfile": "unodueResults",
///     "resfilemode": "RECREATE",
///     "ntfmerge": 1,
///     "OutputDescriptors": [
///       {
///         "table": "AOD/UNO/0",
///         "treename": "unotree"
///       },
///       {
///         "table": "AOD/DUE/0",
///         "columns": [
///           "due_1",
///           "due_3",
///           "due_5"
///         ],
///         "treename": "duetree"
///       },
///       {
///         "table": "AOD/TRE/0",
///         "filename": "treResults"
///       }
///     ]
///   }
/// }
///
/// readerConfiguration.json:
/// {
///   "InputDirector": {
///     "debugmode": true,
///     "resfiles": "@resultFiles.txt",
///     "fileregex": "(unodue)(.*)",
///     "InputDescriptors": [
///       {
///         "table": "AOD/EINS/0",
///         "treename": "unotree"
///       },
///       {
///         "table": "AOD/ZWEI/0",
///         "treename": "duetree"
///       },
///       {
///         "table": "AOD/DREI/0",
///         "treename": "TRE",
///         "fileregex": "(treResults)(.*)"
///       }
///     ]
///   }
/// }

namespace o2::aod
{
namespace uno
{
DECLARE_SOA_COLUMN_FULL(Eta, eta, float, "uno_1");
DECLARE_SOA_COLUMN_FULL(Phi, phi, float, "uno_2");
DECLARE_SOA_COLUMN_FULL(Mom, mom, double, "uno_3");
} // namespace uno

DECLARE_SOA_TABLE(Uno, "AOD", "UNO",
                  uno::Eta, uno::Phi, uno::Mom);

namespace due
{
DECLARE_SOA_COLUMN_FULL(Ok, ok, bool, "due_1");
DECLARE_SOA_COLUMN_FULL(Eta, eta, float, "due_2");
DECLARE_SOA_COLUMN_FULL(Phi, phi, float, "due_3");
DECLARE_SOA_COLUMN_FULL(Mom, mom, double, "due_4");
DECLARE_SOA_COLUMN_FULL(Pt, pt, double, "due_5");
} // namespace due

DECLARE_SOA_TABLE(Due, "AOD", "DUE",
                  due::Ok, due::Eta, due::Phi, due::Mom, due::Pt);

namespace tre
{
DECLARE_SOA_COLUMN_FULL(Eta, eta, float, "tre_1");
DECLARE_SOA_COLUMN_FULL(Phi, phi, float, "tre_2");
DECLARE_SOA_COLUMN_FULL(Mom, mom, double, "tre_3");
DECLARE_SOA_COLUMN_FULL(Id, id, int, "tre_4");
} // namespace tre

DECLARE_SOA_TABLE(Tre, "AOD", "TRE",
                  tre::Eta, tre::Phi, tre::Mom, tre::Id);

} // namespace o2::aod

using namespace o2;
using namespace o2::framework;

struct aodWriter {
  Produces<aod::Uno> table_uno;
  Produces<aod::Due> table_due;
  Produces<aod::Tre> table_tre;

  void init(InitContext&)
  {
    cnt = 0;
  }

  void process(aod::Tracks const& tracks)
  {
    for (auto& track : tracks) {
      bool ok = (cnt % 2) == 0;
      float eta = log(tan(0.25f * static_cast<float>(M_PI) - 0.5f * atan(track.tgl())));
      float phi = asin(track.snp()) + track.alpha() + static_cast<float>(M_PI);
      double mom = track.tgl();
      double pt = track.signed1Pt();
      int id = (int)cnt;

      table_uno(phi, eta, mom);
      table_due(ok, phi, eta, mom, pt);
      table_tre(phi, eta, mom, id);
      //LOGF(INFO, "Values (%i): (%i %f, %f, %f, %f, %i)", cnt, ok, eta, phi, mom, pt, id);
      cnt++;
    }

    LOGF(INFO, "ATask Processed %i data points from Tracks", cnt);
  }

  size_t cnt = 0;
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  return WorkflowSpec{
    adaptAnalysisTask<aodWriter>(cfgc),
  };
}
