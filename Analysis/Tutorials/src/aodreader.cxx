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
/// \brief Filling tables with data froma root file
/// \author
/// \since

#include "Framework/runDataProcessing.h"
#include "Framework/AnalysisTask.h"

/// This example is to be used together with the aodwriter example.
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
namespace eins
{
DECLARE_SOA_COLUMN_FULL(Eta, eta, float, "uno_1");
DECLARE_SOA_COLUMN_FULL(Mom, mom, double, "uno_3");
} // namespace eins

DECLARE_SOA_TABLE(Eins, "AOD", "EINS",
                  eins::Eta, eins::Mom);

namespace zwei
{
DECLARE_SOA_COLUMN_FULL(Ok, ok, bool, "due_1");
DECLARE_SOA_COLUMN_FULL(Phi, phi, float, "due_3");
DECLARE_SOA_COLUMN_FULL(Pt, pt, double, "due_5");
} // namespace zwei

DECLARE_SOA_TABLE(Zwei, "AOD", "ZWEI",
                  zwei::Ok, zwei::Phi, zwei::Pt);

namespace drei
{
DECLARE_SOA_COLUMN_FULL(Eta, eta, float, "tre_1");
DECLARE_SOA_COLUMN_FULL(Phi, phi, float, "tre_2");
DECLARE_SOA_COLUMN_FULL(Mom, mom, double, "tre_3");
} // namespace drei

DECLARE_SOA_TABLE(Drei, "AOD", "DREI",
                  drei::Eta, drei::Phi, drei::Mom);

} // namespace o2::aod

using namespace o2;
using namespace o2::framework;

struct aodReader {
  void process(aod::Eins const& unos, aod::Zwei const& dues, aod::Drei const& tres)
  {
    int cnt = 0;
    for (auto& uno : unos) {
      auto eta = uno.eta();
      auto mom = uno.mom();

      cnt++;
      LOGF(INFO, "Eins (%i): (%f, %f)", cnt, eta, mom);
    }
    LOGF(INFO, "ATask Processed %i data points from Eins", cnt);

    cnt = 0;
    for (auto& due : dues) {
      auto ok = due.ok();
      auto phi = due.phi();
      auto pt = due.pt();

      cnt++;
      LOGF(INFO, "Zwei (%i): (%i, %f, %f)", cnt, ok, phi, pt);
    }
    LOGF(INFO, "ATask Processed %i data points from Zwei", cnt);

    cnt = 0;
    for (auto& tre : tres) {
      auto eta = tre.eta();
      auto phi = tre.phi();
      auto mom = tre.mom();

      cnt++;
      LOGF(INFO, "Drei (%i): (%f, %f, %f)", cnt, eta, phi, mom);
    }
    LOGF(INFO, "ATask Processed %i data points from Drei", cnt);
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  return WorkflowSpec{
    adaptAnalysisTask<aodReader>(cfgc),
  };
}
