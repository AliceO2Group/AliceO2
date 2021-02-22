// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#if !defined(__CLING__) || defined(__ROOTCLING__)
#include <fstream>
#include <iostream>
#include <vector>
#include "TSystem.h"

#include "TROOT.h"

#include "GPUO2InterfaceRefit.h"
#include "TPCReconstruction/TPCFastTransformHelperO2.h"
#include "DataFormatsParameters/GRPObject.h"
#include "DataFormatsTPC/ClusterNative.h"
#include "DataFormatsTPC/ClusterNativeHelper.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "DataFormatsTPC/Constants.h"
#include "DataFormatsTPC/TrackTPC.h"
#include "DetectorsBase/GeometryManager.h"
#include "DetectorsBase/Propagator.h"
#include "DetectorsCommonDataFormats/NameConf.h"
#else
#pragma cling load("libO2TPCReconstruction")
#pragma cling load("libO2DataFormatsTPC")
#endif

using namespace o2::gpu;
using namespace o2::tpc;
using namespace o2::dataformats;
using namespace o2::base;
using namespace o2::track;
using namespace std;

int runTPCRefit(TString trackFile = "tpctracks.root", TString clusterFile = "tpc-native-clusters.root")
{
  GeometryManager::loadGeometry();
  Propagator::initFieldFromGRP(NameConf::getGRPFileName());
  const auto grp = o2::parameters::GRPObject::loadFrom("o2sim_grp.root");
  float bz = 5.00668f * grp->getL3Current() / 30000.;
  std::unique_ptr<TPCFastTransform> trans = std::move(TPCFastTransformHelperO2::instance()->create(0));
  auto* prop = Propagator::Instance();

  ClusterNativeAccess clusterIndex;
  std::vector<TrackTPC>* tracks = nullptr;
  std::vector<TPCClRefElem>* trackHitRefs = nullptr;

  TFile file(trackFile.Data());
  auto tree = (TTree*)file.Get("tpcrec");
  if (tree == nullptr) {
    std::cout << "Error getting tree\n";
    return 1;
  }
  tree->SetBranchAddress("TPCTracks", &tracks);
  tree->SetBranchAddress("ClusRefs", &trackHitRefs);

  ClusterNativeHelper::Reader tpcClusterReader{};
  tpcClusterReader.init(clusterFile.Data());
  std::unique_ptr<ClusterNative[]> clusterBuffer{};
  o2::tpc::ClusterNativeHelper::ConstMCLabelContainerViewWithBuffer clusterMCBuffer;
  const size_t maxEvent = tree->GetEntriesFast();
  if (maxEvent != tpcClusterReader.getTreeSize()) {
    std::cout << "Mismatch of entries in cluster and track file\n";
    return 1;
  }
  for (unsigned int iEvent = 0; iEvent < maxEvent; ++iEvent) {
    std::cout << "Event " << iEvent << " of " << maxEvent << "\n";
    tree->GetEntry(iEvent);
    memset(&clusterIndex, 0, sizeof(clusterIndex));
    tpcClusterReader.read(iEvent);
    int retVal = tpcClusterReader.fillIndex(clusterIndex, clusterBuffer, clusterMCBuffer);
    if (retVal < 0) {
      std::cout << "Error reading clusters (code " << retVal << ")\n";
      return 1;
    }
    GPUO2InterfaceRefit refit(&clusterIndex, trans.get(), bz, trackHitRefs->data(), nullptr, tracks, prop);
    //refit.setGPUTrackFitInProjections(false); // Enable full 3D fit without assuming y and Z are uncorrelated
    for (unsigned int i = 0; i < tracks->size(); i++) {
      TrackTPC trk = (*tracks)[i];
      refit.setTrackReferenceX(trk.getX());
      std::cout << "\nTrack " << i << "\n";
      std::cout << "Org track:\n";
      trk.print();
      std::cout << "Refitting as GPU track\n";
      int retval = refit.RefitTrackAsGPU(trk, false, true);
      if (retval < 0) {
        std::cout << "Refit as GPU track failed " << retval << "\n";
      } else {
        std::cout << "Succeeded using " << retval << " hits (chi2 = " << trk.getChi2() << ")\n";
        trk.print();
      }
      trk = (*tracks)[i];
      std::cout << "Refitting as TrackParCov track\n";
      retval = refit.RefitTrackAsTrackParCov(trk, false, true);
      if (retval < 0) {
        std::cout << "Refit as TrackParCov track failed " << retval << "\n";
      } else {
        std::cout << "Succeeded using " << retval << " hits (chi2 = " << trk.getChi2() << ")\n";
        trk.print();
      }
      trk = (*tracks)[i];
      TrackParCov trkX = trk;
      float chi2 = trk.getChi2();
      std::cout << "Refitting as TrackParCov track with TrackParCov input\n";
      retval = refit.RefitTrackAsTrackParCov(trkX, trk.getClusterRef(), trk.getTime0(), &chi2, false, true);
      if (retval < 0) {
        std::cout << "Refit as TrackParCov track failed " << retval << "\n";
      } else {
        std::cout << "Succeeded using " << retval << " hits (chi2 = " << chi2 << ")\n";
        trkX.print();
      }
    }
  }

  return 0;
}
