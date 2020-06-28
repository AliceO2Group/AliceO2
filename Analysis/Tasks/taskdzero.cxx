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
#include "Framework/AnalysisTask.h"
#include "Framework/AnalysisDataModel.h"
#include "Framework/ASoAHelpers.h"
#include "Analysis/SecondaryVertexHF.h"
#include "DetectorsVertexing/DCAFitterN.h"
#include "ReconstructionDataFormats/Track.h"

#include <TFile.h>
#include <TH1F.h>
#include <cmath>
#include <array>
#include <cstdlib>

using namespace o2;
using namespace o2::framework;
using namespace o2::framework::expressions;
using std::array;

struct TaskDzero {
  OutputObj<TH1F> hmass{TH1F("hmass", "2-track inv mass", 500, 0, 5.0)};
  OutputObj<TH1F> hptcand{TH1F("hptcand", "pt candidate", 100, 0, 10.0)};
  OutputObj<TH1F> hptprong0{TH1F("hptprong0", "pt prong0", 100, 0, 10.0)};
  OutputObj<TH1F> hptprong1{TH1F("hptprong1", "pt prong1", 100, 0, 10.0)};
  OutputObj<TH1F> hdeclength{TH1F("declength", "decay length", 100, 0., 1.0)};
  OutputObj<TH1F> hdeclengthxy{TH1F("declengthxy", "decay length xy", 100, 0., 1.0)};

  void process(aod::HfCandProng2 const& hfcandprong2s)
  {
    for (auto& hfcandprong2 : hfcandprong2s) {
      hmass->Fill(hfcandprong2.massD0());
      hmass->Fill(hfcandprong2.massD0bar());
      hptprong0->Fill(hfcandprong2.ptprong0());
      hptprong1->Fill(hfcandprong2.ptprong1());
      hptcand->Fill(hfcandprong2.pt());
      hdeclength->Fill(hfcandprong2.decaylength());
      hdeclengthxy->Fill(hfcandprong2.decaylengthxy());
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  return WorkflowSpec{
    adaptAnalysisTask<TaskDzero>("hf-taskdzero")};
}
