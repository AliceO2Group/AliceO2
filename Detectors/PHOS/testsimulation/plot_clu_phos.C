#if !defined(__CLING__) || defined(__ROOTCLING__)
#include <sstream>

#include <TStopwatch.h>
#include "TCanvas.h"
#include "TH2.h"
//#include "DataFormatsParameters/GRPObject.h"
#include "FairFileSource.h"
#include "FairLogger.h"
#include "FairRunAna.h"
//#include "FairRuntimeDb.h"
#include "FairParRootFileIo.h"
#include "FairSystemInfo.h"
#include "SimulationDataFormat/MCCompLabel.h"

#include "PHOSBase/Geometry.h"
#include "PHOSReconstruction/Cluster.h"
#endif

void plot_clu_phos(int ievent = 0, std::string inputfile = "o2clu.root")
{
  // macros to plot PHOS hits

  FairFileSource* fFileSource = new FairFileSource(inputfile);
  FairRootManager* mgr = FairRootManager::Instance();
  mgr->SetSource(fFileSource);
  mgr->InitSource();

  const std::vector<o2::phos::Cluster>* mHitsArray =
    mgr->InitObjectAs<const std::vector<o2::phos::Cluster>*>("PHSCluster");
  if (!mHitsArray) {
    cout << "PHOS hits not registered in the FairRootManager. Exiting ..." << endl;
    return;
  }
  mgr->ReadEvent(ievent);

  TH2D* vMod[5] = { 0 };

  o2::phos::Geometry* geom = new o2::phos::Geometry("PHOS");

  std::vector<o2::phos::Cluster>::iterator it;
  int relId[3];

  double posX, posZ;
  for (auto& it : *mHitsArray) {
    it.GetLocalPosition(posX, posZ);
    int mod = it.GetPHOSMod();
    if (!vMod[mod]) {
      vMod[mod] = new TH2D(Form("hMod%d", mod), Form("hMod%d", mod), 64, -64 * 2.25 / 2., 64 * 2.25 / 2., 56,
                           -56 * 2.25 / 2., 56. * 2.25 / 2.);
    }
    vMod[mod]->Fill(posX, posZ, it.GetEnergy());
  }

  TCanvas* c[5];
  for (int mod = 1; mod < 5; mod++) {
    c[mod] =
      new TCanvas(Form("HitInMod%d", mod), Form("PHOS hits in module %d", mod), 10 * mod, 0, 600 + 10 * mod, 400);
    if (vMod[mod])
      vMod[mod]->Draw("box");
  }
}