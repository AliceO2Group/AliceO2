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
#include "DataFormatsHMP/Hit.h"
#include "TPCSimulation/Point.h"
#include "PHOSBase/Hit.h"
#include "DataFormatsFDD/Hit.h"
#include "MCHSimulation/Hit.h"
#include "MIDSimulation/Hit.h"
#include "DataFormatsCPV/Hit.h"
#include "DataFormatsZDC/Hit.h"
#include "SimulationDataFormat/MCEventHeader.h"
#include "SimulationDataFormat/MCTrack.h"
#include "SimulationDataFormat/TrackReference.h"
#include "DataFormatsParameters/GRPObject.h"
#include "DetectorsCommonDataFormats/DetectorNameConf.h"
#include "DetectorsCommonDataFormats/SimTraits.h"
#ifdef ENABLE_UPGRADES
// todo: put upgrade detectors?
#endif

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
    for (int j = 0; j < factor; ++j) {
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

TTree* getHitTree(o2::parameters::GRPObject const* grp, const char* filebase, o2::detectors::DetID detid, bool createnew = false)
{
  if (!grp) {
    std::cerr << "GRP is null\n";
    return nullptr;
  }
  if (!grp->isDetReadOut(detid)) {
    return nullptr;
  }
  std::string filename(o2::base::DetectorNameConf::getHitsFileName(detid, filebase).c_str());

  const char* mode = createnew ? "RECREATE" : "OPEN";

  // shamefully leaking memory as the TTree cannot live without the file...
  TFile* file = new TFile(filename.c_str(), mode);
  TTree* t = nullptr;
  if (createnew) {
    t = new TTree("o2sim", "o2sim");
  } else {
    t = (TTree*)file->Get("o2sim");
  }
  return t;
}

template <typename T>
struct HitContainer {
  using type = std::vector<T>;
};
template <>
struct HitContainer<o2::tpc::HitGroup> {
  using type = o2::tpc::HitGroup;
};

template <typename T>
void duplicateV(o2::parameters::GRPObject const* grp,
                const char* filebase, o2::detectors::DetID detid, const char* newfilebase, int factor)
{
  // open old file and extract tree
  auto tr = getHitTree(grp, filebase, detid);
  if (!tr) {
    return;
  }
  auto outtree = getHitTree(grp, newfilebase, detid, true);
  if (!outtree) {
    return;
  }

  // duplicate entries in all branches and copy
  auto branches = o2::detectors::SimTraits::DETECTORBRANCHNAMES[(int)detid];
  for (auto& br : branches) {
    duplicate<std::vector<T>>(tr, br.c_str(), outtree, factor);
  }
  outtree->SetEntries(tr->GetEntries() * factor);
  // write outtree
  outtree->Write();
}

TTree* getKinematicsTree(const char* filebase, bool createnew = false)
{
  // shamefully leaking memory as the TTree cannot live without the file...
  const char* mode = createnew ? "RECREATE" : "OPEN";
  TFile* file = new TFile(o2::base::NameConf::getMCKinematicsFileName(filebase).c_str(), mode);
  TTree* t = nullptr;
  if (createnew) {
    t = new TTree("o2sim", "o2sim");
  } else {
    t = (TTree*)file->Get("o2sim");
  }
  return t;
}

// Macro to duplicate hit output of a simulation
// This might be useful to enlarge input for digitization or to engineer
// sequences of identical hit structures in order test pileup
void duplicateHits(const char* filebase = "o2sim", const char* newfilebase = "o2sim_duplicated", int factor = 2)
{
  // NOTE: There might be a way in which this can be achieved
  // without explicit iteration over branches and type ... just by using TClasses

  // READ GRP AND ITERATE OVER DETECTED PARTS
  auto oldgrpfilename = o2::base::NameConf::getGRPFileName(filebase);
  auto grp = o2::parameters::GRPObject::loadFrom(oldgrpfilename.c_str());
  if (!grp) {
    std::cerr << "No GRP found; exiting\n";
    return;
  }
  // cp GRP for new sim files
  auto newgrpfilename = o2::base::NameConf::getGRPFileName(newfilebase);
  std::stringstream command;
  command << "cp " << oldgrpfilename << " " << newgrpfilename;
  system(command.str().c_str());

  // duplicate kinematics stuff
  auto kintree = getKinematicsTree(filebase);
  auto newkintree = getKinematicsTree(newfilebase, "RECREATE");
  // duplicate meta branches
  duplicate<std::vector<o2::MCTrack>>(kintree, "MCTrack", newkintree, factor);
  duplicate<o2::dataformats::MCEventHeader>(kintree, "MCEventHeader.", newkintree, factor);
  // TODO: fix EventIDs in the newly created MCEventHeaders
  duplicate<std::vector<o2::TrackReference>>(kintree, "TrackRefs", newkintree, factor);
  newkintree->Write();

  // duplicating hits
  using namespace o2::detectors;
  duplicateV<o2::itsmft::Hit>(grp, filebase, DetID::ITS, newfilebase, factor);
  duplicateV<o2::itsmft::Hit>(grp, filebase, DetID::MFT, newfilebase, factor);
  duplicateV<o2::tof::HitType>(grp, filebase, DetID::TOF, newfilebase, factor);
  duplicateV<o2::emcal::Hit>(grp, filebase, DetID::EMC, newfilebase, factor);
  duplicateV<o2::trd::Hit>(grp, filebase, DetID::TRD, newfilebase, factor);
  duplicateV<o2::phos::Hit>(grp, filebase, DetID::PHS, newfilebase, factor);
  duplicateV<o2::cpv::Hit>(grp, filebase, DetID::CPV, newfilebase, factor);
  duplicateV<o2::zdc::Hit>(grp, filebase, DetID::ZDC, newfilebase, factor);
  duplicateV<o2::ft0::HitType>(grp, filebase, DetID::FT0, newfilebase, factor);
  duplicateV<o2::fv0::Hit>(grp, filebase, DetID::FV0, newfilebase, factor);
  duplicateV<o2::fdd::Hit>(grp, filebase, DetID::FDD, newfilebase, factor);
  duplicateV<o2::hmpid::HitType>(grp, filebase, DetID::HMP, newfilebase, factor);
  duplicateV<o2::mid::Hit>(grp, filebase, DetID::MID, newfilebase, factor);
  duplicateV<o2::mch::Hit>(grp, filebase, DetID::MCH, newfilebase, factor);
  duplicateV<o2::tpc::HitGroup>(grp, filebase, DetID::TPC, newfilebase, factor);

#ifdef ENABLE_UPGRADES
  duplicateV<o2::itsmft::Hit>(grp, filebase, DetID::FT3, newfilebase, factor);
  duplicateV<o2::itsmft::Hit>(grp, filebase, DetID::FCT, newfilebase, factor);
#endif

  // duplicateACO(reftree);
  // outfile.Write();
}
