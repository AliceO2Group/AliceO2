// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file FemtoDreamTrackCuts.cxx
/// \brief Implementation of the FemtoDreamTrackCuts
/// \author Andi Mathis, TU München, andreas.mathis@ph.tum.de

#include "o2femtodream/FemtoDreamTrackSelection.h"

using namespace o2::analysis::femtoDream;

FemtoDreamTrackSelection::FemtoDreamTrackSelection()
  : mTrackCharge(1),
    mPtMin(0.f),
    mPtMax(999.f),
    mEtaMax(999.f),
    mTPCnClsMin(0),
    mTPCfClsMin(0.f),
    mTPCcRowMin(0),
    mTPCsClsRej(false),
    mDCAxyMax(999.f),
    mDCAzMax(999.f),
    mPIDnSigmaMax(999.f),
    mPIDmomTPC(0.7f),
    mPIDParticle(o2::track::PID::Pion),
    //mHistogramRegistry(nullptr),
    mDoQA(false)
{
}

FemtoDreamTrackSelection::FemtoDreamTrackSelection(int charge, float ptMin, float pTmax,
                                                   float etaMax, int tpcNcls, float tpcFcls,
                                                   int tpcNrows, bool tpcShareRej, float dcaXYMax,
                                                   float dcaZMax, float pidNsigmaMax, float pidTPCmom, o2::track::PID::ID part)
  : mTrackCharge(charge),
    mPtMin(ptMin),
    mPtMax(pTmax),
    mEtaMax(etaMax),
    mTPCnClsMin(tpcNcls),
    mTPCfClsMin(tpcFcls),
    mTPCcRowMin(tpcNrows),
    mTPCsClsRej(tpcShareRej),
    mDCAxyMax(dcaXYMax),
    mDCAzMax(dcaZMax),
    mPIDnSigmaMax(pidNsigmaMax),
    mPIDmomTPC(pidTPCmom),
    mPIDParticle(part),
    //mHistogramRegistry(nullptr),
    mDoQA(false)
{
}

void FemtoDreamTrackSelection::init() //HistogramRegistry* registry)
{
  // get minimal cuts
  std::sort(mTPCclsCut.begin(), mTPCclsCut.end());

  /* if (registry) {
    mHistogramRegistry = registry;
    mDoQA = true;
    mHistogramRegistry->add("TrackCuts/pThist", "; #it{p}_{T} (GeV/#it{c}); Entries", kTH1F, {{1000, 0, 10}});
    mHistogramRegistry->add("TrackCuts/etahist", "; #eta; Entries", kTH1F, {{1000, -1, 1}});
    mHistogramRegistry->add("TrackCuts/phihist", "; #phi; Entries", kTH1F, {{1000, 0, 2. * M_PI}});
    mHistogramRegistry->add("TrackCuts/tpcnclshist", "; TPC Cluster; Entries", kTH1F, {{163, 0, 163}});
    mHistogramRegistry->add("TrackCuts/tpcfclshist", "; TPC ratio findable; Entries", kTH1F, {{100, 0.5, 1.5}});
    mHistogramRegistry->add("TrackCuts/tpcnrowshist", "; TPC crossed rows; Entries", kTH1F, {{163, 0, 163}});
    mHistogramRegistry->add("TrackCuts/tpcnsharedhist", "; TPC shared clusters; Entries", kTH1F, {{163, 0, 163}});
    mHistogramRegistry->add("TrackCuts/dcaXYhistBefore", "; #it{p}_{T} (GeV/#it{c}); DCA_{xy} (cm)", kTH2F, {{100, 0, 10}, {301, -1.5, 1.5}});
    mHistogramRegistry->add("TrackCuts/dcaXYhist", "; #it{p}_{T} (GeV/#it{c}); DCA_{xy} (cm)", kTH2F, {{100, 0, 10}, {301, -1.5, 1.5}});
    mHistogramRegistry->add("TrackCuts/dcaZhist", "; #it{p}_{T} (GeV/#it{c}); DCA_{z} (cm)", kTH2F, {{100, 0, 10}, {301, -1.5, 1.5}});
    mHistogramRegistry->add("TrackCuts/tpcdEdx", "; #it{p} (GeV/#it{c}); TPC Signal", kTH2F, {{100, 0, 10}, {1000, 0, 1000}});
    mHistogramRegistry->add("TrackCuts/tofSignal", "; #it{p} (GeV/#it{c}); TOF Signal", kTH2F, {{100, 0, 10}, {1000, 0, 100e3}});
  } */
}

std::string FemtoDreamTrackSelection::getCutHelp()
{
  return "Charge; "
         "Min. pT (GeV/c); "
         "Max. pT (GeV/c); "
         "Max. eta; "
         "Min. TPC cluster; "
         "Min. TPC findable cluster fraction; "
         "Min. TPC crossed rows; "
         "Shared cluster rejection; "
         "Max. DCA to PV in xy (cm); "
         "Max. DCA to PV in z (cm); "
         "Max. nSigma value; "
         "Max. p for TPC-only PID (GeV/c); "
         "Particle species to select";
}

void FemtoDreamTrackSelection::printCuts()
{
  std::cout << "Debug information for FemtoDreamTrackSelection \n Charge: " << mTrackCharge << "\n Min. pT (GeV/c): " << mPtMin << "\n Max. pT (GeV/c): " << mPtMax << "\n Max. eta: " << mEtaMax << "\n Min. TPC cluster: " << mTPCnClsMin << "\n Min. TPC findable cluster fraction: " << mTPCfClsMin << "\n Min. TPC crossed rows: " << mTPCcRowMin << "\n Shared cluster rejection: " << mTPCsClsRej << "\n Max. DCA to PV in xy (cm): " << mDCAxyMax << "\n Max. DCA to PV in z (cm): " << mDCAzMax << "\n Max. nSigma value; " << mPIDnSigmaMax << "\n Max. p for TPC-only PID (GeV/c): " << mPIDmomTPC << "\n Particle species to select: " << mPIDParticle << "\n";
}
