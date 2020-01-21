#if !defined(__CLING__) || defined(__ROOTCLING__)
#include "TFile.h"
#include "TTree.h"
#include "TString.h"
#include "ITSMFTSimulation/Hit.h"
#include "TOFSimulation/Detector.h"
#include "EMCALBase/Hit.h"
#include "TRDSimulation/Detector.h" // For TRD Hit
#include "FT0Simulation/Detector.h" // for Fit Hit
#include "DataFormatsFV0/Hit.h"
#include "HMPIDBase/Hit.h"
#include "TPCSimulation/Point.h"
#include "PHOSBase/Hit.h"
#include "DataFormatsFDD/Hit.h"
#include "MCHSimulation/Hit.h"
#include "MIDSimulation/Hit.h"
#include "CPVBase/Hit.h"
#include "ZDCSimulation/Hit.h"
#include "SimulationDataFormat/MCEventHeader.h"
#endif

template <typename T>
void duplicate(TTree* tr, const char* brname, TTree* outtree, int factor)
{
  auto br = tr->GetBranch(brname);
  if (!br) {
    return;
  }
  auto entries = br->GetEntries();
  T* entrydata = nullptr;
  br->SetAddress(&entrydata);

  auto outbranch = outtree->Branch(brname, &entrydata);
  if (!outbranch) {
    std::cerr << "branch " << brname << " not created\n";
  }

  for (int i = 0; i < entries; ++i) {
    br->GetEntry(i);
    for (int i = 0; i < factor; ++i) {
      outbranch->Fill();
    }
  }
  outtree->SetEntries(entries * factor);
}

// we need to do something special for MCEventHeaders
template <>
void duplicate<o2::dataformats::MCEventHeader>(TTree* tr, const char* brname, TTree* outtree, int factor)
{
  auto br = tr->GetBranch(brname);
  if (!br) {
    return;
  }
  auto entries = br->GetEntries();
  o2::dataformats::MCEventHeader* entrydata = nullptr;
  br->SetAddress(&entrydata);

  auto outbranch = outtree->Branch(brname, &entrydata);
  if (!outbranch) {
    std::cerr << "branch " << brname << " not created\n";
  }

  int eventID = 1;
  for (int i = 0; i < entries; ++i) {
    br->GetEntry(i);
    for (int i = 0; i < factor; ++i) {
      entrydata->SetEventID(eventID++);
      outbranch->Fill();
    }
  }
  outtree->SetEntries(entries * factor);
}

template <typename T>
void duplicateV(TTree* tr, const char* brname, TTree* outtree, int factor)
{
  duplicate<std::vector<T>>(tr, brname, outtree, factor);
}

// need a special version for TPC since loop over sectors
void duplicateTPC(TTree* tr, TTree* newtree, int factor)
{
  for (int sector = 0; sector < 35; ++sector) {
    std::stringstream brnamestr;
    brnamestr << "TPCHitsShiftedSector" << sector;

    // call other duplicate function with correct type
    duplicateV<o2::tpc::HitGroup>(tr, brnamestr.str().c_str(), newtree, factor);
  }
}

// Macro to duplicate hit output of a simulation
// This might be useful to enlarge input for digitization or to engineer
// sequences of identical hit structures in order test pileup
void duplicateHits(const char* filename = "o2sim.root", const char* newfilename = "o2sim_duplicated.root", int factor = 2)
{
  TFile rf(filename, "OPEN");
  auto reftree = (TTree*)rf.Get("o2sim");

  TFile outfile(newfilename, "RECREATE");
  auto newtree = new TTree("o2sim", "o2sim");

  // NOTE: There might be a way in which this can be achieved
  // without explicit iteration over branches and type ... just by using TClasses

  // duplicate meta branches
  duplicate<o2::MCTrack>(reftree, "MCTrack", newtree, factor);
  duplicate<o2::dataformats::MCEventHeader>(reftree, "MCEventHeader.", newtree, factor);
  // TODO: fix EventIDs in the newly created MCEventHeaders

  duplicate<o2::TrackReference>(reftree, "TrackRefs", newtree, factor);
  duplicate<o2::dataformats::MCTruthContainer<o2::TrackReference>>(reftree, "IndexedTrackRefs", newtree, factor);

  // duplicating hits
  duplicateV<o2::itsmft::Hit>(reftree, "ITSHit", newtree, factor);
  duplicateV<o2::itsmft::Hit>(reftree, "MFTHit", newtree, factor);
  duplicateV<o2::tof::HitType>(reftree, "TOFHit", newtree, factor);
  duplicateV<o2::emcal::Hit>(reftree, "EMCHit", newtree, factor);
  duplicateV<o2::trd::HitType>(reftree, "TRDHit", newtree, factor);
  duplicateV<o2::phos::Hit>(reftree, "PHSHit", newtree, factor);
  duplicateV<o2::cpv::Hit>(reftree, "CPVHit", newtree, factor);
  duplicateV<o2::zdc::Hit>(reftree, "ZDCHit", newtree, factor);
  duplicateV<o2::ft0::HitType>(reftree, "FT0Hit", newtree, factor);
  duplicateV<o2::fv0::Hit>(reftree, "FV0Hit", newtree, factor);
  duplicateV<o2::fdd::Hit>(reftree, "FDDHit", newtree, factor);
  duplicateV<o2::hmpid::HitType>(reftree, "HMPHit", newtree, factor);
  duplicateV<o2::mid::Hit>(reftree, "MIDHit", newtree, factor);
  duplicateV<o2::mch::Hit>(reftree, "MCHHit", newtree, factor);
  duplicateTPC(reftree, newtree, factor);
  // duplicateACO(reftree);
  outfile.Write();
}
