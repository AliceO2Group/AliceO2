// Macro to check functioning of stack
// Analyses kinematics and track references of a kinematics file

#if !defined(__CLING__) || defined(__ROOTCLING__)
#include "SimulationDataFormat/MCTrack.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "SimulationDataFormat/Stack.h"
#include "SimulationDataFormat/TrackReference.h"
#endif
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include "FairLogger.h"

void checkStack(const char* name = "o2sim.root")
{
  FairLogger::GetLogger()->SetLogScreenLevel("DEBUG");
  TFile f(name);

  LOG(DEBUG) << "Checking input file :" << f.GetPath();

  std::vector<o2::MCTrack>* mctracks = nullptr;
  auto tr = (TTree*)f.Get("o2sim");
  assert(tr);

  auto mcbr = tr->GetBranch("MCTrack");
  assert(mcbr);
  mcbr->SetAddress(&mctracks);

  std::vector<o2::TrackReference>* trackrefs = nullptr;
  auto refbr = tr->GetBranch("TrackRefs");
  assert(refbr);
  refbr->SetAddress(&trackrefs);

  o2::dataformats::MCTruthContainer<o2::TrackReference>* indexedtrackrefs = nullptr;
  auto irefbr = tr->GetBranch("IndexedTrackRefs");
  irefbr->SetAddress(&indexedtrackrefs);

  for (int i = 0; i < mcbr->GetEntries(); ++i) {
    mcbr->GetEntry(i);
    refbr->GetEntry(i);
    irefbr->GetEntry(i);
    LOG(DEBUG) << "-- Entry --" << i << "\n";
    LOG(DEBUG) << "Have " << mctracks->size() << " tracks \n";
    int ti = 0;

    // record tracks that left a hit in TPC
    // (we know that these tracks should then have a TrackRef)
    std::vector<int> trackidsinTPC;

    for (auto& t : *mctracks) {
      // check that mother indices are reasonable
      assert(ti > t.getMotherTrackId());
      if (t.leftTrace(o2::detectors::DetID::TPC)) {
        trackidsinTPC.emplace_back(ti);
      }
      LOG(DEBUG) << " track " << ti << "\t" << t.getMotherTrackId() << " hits " << t.hasHits() << "\n";
      ti++;
    }

    LOG(DEBUG) << "Have " << trackidsinTPC.size() << " tracks with hits in TPC\n";
    LOG(DEBUG) << "Have " << trackrefs->size() << " track refs \n";
    LOG(DEBUG) << "Have " << indexedtrackrefs->getIndexedSize() << " mapped tracks\n";
    LOG(DEBUG) << "Have " << indexedtrackrefs->getNElements() << " track refs\n";

    bool havereferences = trackrefs->size();
    if (havereferences) {
      for (auto& i : trackidsinTPC) {
        auto labels = indexedtrackrefs->getLabels(i);
        assert(labels.size() > 0);
        LOG(DEBUG) << " Track " << i << " has " << labels.size() << " TrackRefs "
                   << "\n";
        for (int j = 0; j < labels.size(); ++j) {
          LOG(DEBUG) << labels[j];
        }
      }
    }
  }
  LOG(INFO) << "STACK TEST SUCCESSFULL\n";
}
