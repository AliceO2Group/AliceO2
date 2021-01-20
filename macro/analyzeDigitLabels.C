#if !defined(__CLING__) || defined(__ROOTCLING__)
#include "TFile.h"
#include "TTree.h"
#include "TString.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/ConstMCTruthContainer.h"
#include "SimulationDataFormat/IOMCTruthContainerView.h"
#include "DataFormatsPHOS/MCLabel.h"
#include "DataFormatsFT0/MCLabel.h"
#include "DataFormatsFDD/MCLabel.h"
#include "DataFormatsFV0/MCLabel.h"
#include "ZDCSimulation/MCLabel.h"
#include "MIDSimulation/MCLabel.h"
#include "TRDBase/MCLabel.h"

#include <gsl/gsl> // for guideline support library; array_view
#include <unordered_map>
#endif

TString gPrefix("");

// see if multiple labels refer to pileup situation
template <typename Labels>
bool isPileup(Labels labels)
{
  std::unordered_map<int, int> events;
  for (auto& label : labels) {
    events[label.getEventID()] = 1;
  }
  return events.size() > 1;
}

// see if multiple labels refer to embedding situation
template <typename Labels>
bool isEmbedding(Labels labels)
{
  std::unordered_map<int, int> sources;
  for (auto& label : labels) {
    sources[label.getSourceID()] = 1;
  }
  return sources.size() > 1;
}

struct LabelStats {
  int NIndices = 0; // number of indexed digits
  int NLabels = 0;  // number of labels
  bool eventpileup = false;
  bool embedding = false;
  bool trackpileup = false;

  template <typename Labels>
  void addLabels(Labels labels)
  {
    NIndices++;
    NLabels += labels.size();
    if (labels.size() > 1) {
      eventpileup |= isPileup(labels);
      embedding |= isEmbedding(labels);
    }
  }

  void print()
  {
    std::cout << NIndices << " "
              << NLabels << " "
              << "pileup " << eventpileup << " "
              << "embedding " << embedding << " "
              << "\n";
  }
};

template <typename LabelType, typename Accumulator = LabelStats>
void analyse(TTree* tr, const char* brname, Accumulator& prop)
{
  if (!tr) {
    return;
  }
  auto br = tr->GetBranch(brname);
  if (!br) {
    return;
  }
  auto classname = br->GetClassName();
  auto entries = br->GetEntries();
  if (strcmp("o2::dataformats::IOMCTruthContainerView", classname) == 0) {
    o2::dataformats::IOMCTruthContainerView* io2 = nullptr;
    br->SetAddress(&io2);

    for (int i = 0; i < entries; ++i) {
      br->GetEntry(i);
      o2::dataformats::ConstMCTruthContainer<LabelType> labels;
      io2->copyandflatten(labels);

      for (int i = 0; i < (int)labels.getIndexedSize(); ++i) {
        prop.addLabels(labels.getLabels(i));
      }
    }
  } else {
    // standard MC truth container
    o2::dataformats::MCTruthContainer<LabelType>* labels = nullptr;
    br->SetAddress(&labels);
    for (int i = 0; i < entries; ++i) {
      br->GetEntry(i);
      for (int i = 0; i < (int)labels->getIndexedSize(); ++i) {
        prop.addLabels(labels->getLabels(i));
      }
    }
  }
};

// tof is special
template <typename LabelType, typename Accumulator = LabelStats>
void analyseTOF(TTree* tr, const char* brname, Accumulator& prop)
{
  auto br = tr->GetBranch(brname);
  if (!br) {
    return;
  }
  auto entries = br->GetEntries();
  std::vector<o2::dataformats::MCTruthContainer<LabelType>>* container = nullptr;
  br->SetAddress(&container);

  for (int i = 0; i < entries; ++i) {
    br->GetEntry(i);

    for (auto& labels : *container) {
      for (int i = 0; i < (int)labels.getIndexedSize(); ++i) {
        prop.addLabels(labels.getLabels(i));
      }
    }
  }
};

// need a special version for TPC since loop over sectors
void analyzeTPC(TTree* reftree)
{
  LabelStats result;
  for (int sector = 0; sector < 35; ++sector) {
    std::stringstream brnamestr;
    brnamestr << "TPCDigitMCTruth_" << sector;
    analyse<o2::MCCompLabel>(reftree, brnamestr.str().c_str(), result);
  }
  std::cout << gPrefix << " TPC ";
  result.print();
};

// do comparison for ITS
void analyzeITS(TTree* reftree)
{
  LabelStats result;
  analyse<o2::MCCompLabel>(reftree, "ITSDigitMCTruth", result);
  std::cout << gPrefix << " ITS ";
  result.print();
}

// do comparison for MFT
void analyzeMFT(TTree* reftree)
{
  LabelStats result;
  analyse<o2::MCCompLabel>(reftree, "MFTDigitMCTruth", result);
  std::cout << gPrefix << " MFT ";
  result.print();
}

// do comparison for EMC
void analyzeEMC(TTree* reftree)
{
  LabelStats result;
  analyse<o2::MCCompLabel>(reftree, "EMCALDigitMCTruth", result);
  std::cout << gPrefix << " EMC ";
  result.print();
}

// do comparison for PHS
void analyzePHS(TTree* reftree)
{
  LabelStats result;
  analyse<o2::phos::MCLabel>(reftree, "PHOSDigitMCTruth", result);
  std::cout << gPrefix << " PHS ";
  result.print();
}

// do comparison for CPV
void analyzeCPV(TTree* reftree)
{
  LabelStats result;
  analyse<o2::MCCompLabel>(reftree, "CPVDigitMCTruth", result);
  std::cout << gPrefix << " CPV ";
  result.print();
}

// do comparison for FT0
void analyzeFT0(TTree* reftree)
{
  LabelStats result;
  analyse<o2::ft0::MCLabel>(reftree, "FT0DIGITSMCTR", result);
  std::cout << gPrefix << " FT0 ";
  result.print();
}

// do comparison for FDD
void analyzeFDD(TTree* reftree)
{
  LabelStats result;
  analyse<o2::fdd::MCLabel>(reftree, "FDDDigitLabels", result);
  std::cout << gPrefix << " FDD ";
  result.print();
}

// do comparison for FV0
void analyzeFV0(TTree* reftree)
{
  LabelStats result;
  analyse<o2::fv0::MCLabel>(reftree, "FV0DigitLabels", result);
  std::cout << gPrefix << " FV0 ";
  result.print();
}

// do comparison for HMP
void analyzeHMP(TTree* reftree)
{
  LabelStats result;
  analyse<o2::MCCompLabel>(reftree, "HMPDigitLabels", result);
  std::cout << gPrefix << " HMP ";
  result.print();
}

// do comparison for ZDC
void analyzeZDC(TTree* reftree)
{
  LabelStats result;
  analyse<o2::zdc::MCLabel>(reftree, "ZDCDigitLabels", result);
  std::cout << gPrefix << " ZDC ";
  result.print();
}

// do comparison for MID
void analyzeMID(TTree* reftree)
{
  LabelStats result;
  analyse<o2::mid::MCLabel>(reftree, "MIDDigitMCLabels", result);
  std::cout << gPrefix << " MID ";
  result.print();
}

// do comparison for MID
void analyzeMCH(TTree* reftree)
{
  LabelStats result;
  analyse<o2::MCCompLabel>(reftree, "MCHMCLabels", result);
  std::cout << gPrefix << " MCH ";
  result.print();
}

// do comparison for MFT
void analyzeTOF(TTree* reftree)
{
  LabelStats result;
  analyseTOF<o2::MCCompLabel>(reftree, "TOFDigitMCTruth", result);
  std::cout << gPrefix << " TOF ";
  result.print();
}

// do comparison for MFT
void analyzeTRD(TTree* reftree)
{
  LabelStats result;
  analyse<o2::MCCompLabel>(reftree, "TRDMCLabels", result);
  std::cout << gPrefix << " TRD ";
  result.print();
}

// Simple macro to get basic mean properties of simulated digits.
//
// A prefix (such as a parameter) can be given which will be  prepended before each line of printout.
// This could be useful for plotting.
//
void analyzeDigitLabels(const char* filename, const char* detname = nullptr, const char* prefix = "")
{
  TFile rf(filename, "OPEN");
  auto reftree = (TTree*)rf.Get("o2sim");

  gPrefix = prefix;
  // should correspond to the same number as defined in DetID
  if (strcmp(detname, "ITS") == 0) {
    analyzeITS(reftree);
  }
  if (strcmp(detname, "MFT") == 0) {
    analyzeMFT(reftree);
  }
  if (strcmp(detname, "TPC") == 0) {
    analyzeTPC(reftree);
  }
  if (strcmp(detname, "TOF") == 0) {
    analyzeTOF(reftree);
  }
  if (strcmp(detname, "TRD") == 0) {
    analyzeTRD(reftree);
  }
  if (strcmp(detname, "EMC") == 0) {
    analyzeEMC(reftree);
  }
  if (strcmp(detname, "PHS") == 0) {
    analyzePHS(reftree);
  }
  if (strcmp(detname, "CPV") == 0) {
    analyzeCPV(reftree);
  }
  if (strcmp(detname, "FT0") == 0) {
    analyzeFT0(reftree);
  }
  if (strcmp(detname, "FV0") == 0) {
    analyzeFV0(reftree);
  }
  if (strcmp(detname, "FDD") == 0) {
    analyzeFDD(reftree);
  }
  if (strcmp(detname, "HMP") == 0) {
    analyzeHMP(reftree);
  }
  if (strcmp(detname, "MCH") == 0) {
    analyzeMCH(reftree);
  }
  if (strcmp(detname, "MID") == 0) {
    analyzeMID(reftree);
  }
  if (strcmp(detname, "ZDC") == 0) {
    analyzeZDC(reftree);
  }
  // analyzeACO(reftree);
}
