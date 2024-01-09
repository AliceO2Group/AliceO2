#if !defined(__CLING__) || defined(__ROOTCLING__)
#include <TTree.h>
#include <TChain.h>
#include <TFile.h>
#include <TStopwatch.h>
#include "Framework/Logger.h"
#include <vector>
#include <string>
#include <iomanip>
#include "ITSMFTReconstruction/ChipMappingITS.h"
#include "ITSMFTReconstruction/GBTWord.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "DataFormatsParameters/GRPObject.h"
#include "DataFormatsITSMFT/Digit.h"
#endif

#include "ITSMFTSimulation/MC2RawEncoder.h"
// demo macro the MC->raw conversion with new (variable page size) format
void setupLinks(o2::itsmft::MC2RawEncoder<o2::itsmft::ChipMappingITS>& m2r, const std::string& outPrefix);

void run_digi2rawVarPage_its(std::string outPrefix = "rawits",       // prefix of the output binary file
                             std::string inpName = "itsdigits.root", // name of the input ITS digits
                             int verbosity = 0,
                             std::string digTreeName = "o2sim",           // name of the digits tree
                             std::string digBranchName = "ITSDigit",      // name of the digits branch
                             std::string rofRecName = "ITSDigitROF",      // name of the ROF records branch
                             std::string inputGRP = "o2sim_grp.root",     // name of the simulated data GRP file
                             uint8_t ruSWMin = 0, uint8_t ruSWMax = 0xff, // seq.ID of 1st and last RU (stave) to convert
                             int superPageSizeInB = 1024 * 1024           // superpage in bytes
)
{
  TStopwatch swTot;
  swTot.Start();
  using ROFR = o2::itsmft::ROFRecord;
  using ROFRVEC = std::vector<o2::itsmft::ROFRecord>;

  ///-------> input
  TChain digTree(digTreeName.c_str());

  digTree.AddFile(inpName.c_str());

  std::vector<o2::itsmft::Digit> digiVec, *digiVecP = &digiVec;
  if (!digTree.GetBranch(digBranchName.c_str())) {
    LOG(fatal) << "Failed to find the branch " << digBranchName << " in the tree " << digTreeName;
  }
  digTree.SetBranchAddress(digBranchName.c_str(), &digiVecP);

  // ROF record entries in the digit tree
  ROFRVEC rofRecVec, *rofRecVecP = &rofRecVec;
  if (!digTree.GetBranch(rofRecName.c_str())) {
    LOG(fatal) << "Failed to find the branch " << rofRecName << " in the tree " << digTreeName;
  }
  digTree.SetBranchAddress(rofRecName.c_str(), &rofRecVecP);
  ///-------< input

  const auto grp = o2::parameters::GRPObject::loadFrom(inputGRP);

  o2::itsmft::MC2RawEncoder<o2::itsmft::ChipMappingITS> m2r;
  m2r.setVerbosity(verbosity);
  m2r.setContinuousReadout(grp->isDetContinuousReadOut(o2::detectors::DetID::ITS)); // must be set explicitly
  m2r.setDefaultSinkName(outPrefix + ".raw");
  m2r.setMinMaxRUSW(ruSWMin, ruSWMax);

  {
    // Attention: HBFUtils is a special singleton of ConfigurableParam type, cannot be set by detectors
    o2::raw::HBFUtils::updateFromString("HBFUtils.nHBFPerTF=256"); // this is default anyway
  }

  m2r.getWriter().setSuperPageSize(1024 * 1024);      // this is default anyway

  m2r.setVerbosity(0);

  setupLinks(m2r, outPrefix);
  //-------------------------------------------------------------------------------<<<<
  int lastTreeID = -1;
  long offs = 0, nEntProc = 0;
  for (int i = 0; i < digTree.GetEntries(); i++) {
    digTree.GetEntry(i);
    for (const auto& rofRec : rofRecVec) {
      int nDigROF = rofRec.getNEntries();
      if (verbosity) {
        LOG(info) << "Processing ROF:" << rofRec.getROFrame() << " with " << nDigROF << " entries";
        rofRec.print();
      }
      if (!nDigROF) {
        if (verbosity) {
          LOG(info) << "Frame is empty"; // ??
        }
        continue;
      }
      nEntProc++;
      auto dgs = nDigROF ? gsl::span<const o2::itsmft::Digit>(&digiVec[rofRec.getFirstEntry()], nDigROF) : gsl::span<const o2::itsmft::Digit>();
      m2r.digits2raw(dgs, rofRec.getBCData());
    }
  } // loop over multiple ROFvectors (in case of chaining)

  m2r.finalize(); // finish TF and flush data
  //
  swTot.Stop();
  swTot.Print();
}

void setupLinks(o2::itsmft::MC2RawEncoder<o2::itsmft::ChipMappingITS>& m2r, const std::string& outPrefix)
{
  //------------------------------------------------------------------------------->>>>
  // just as an example, we require here that the staves are read via 3 links, with partitioning according to lnkXB below
  // while OB staves use only 1 link.
  // Note, that if the RU container is not defined, it will be created automatically
  // during encoding.
  // If the links of the container are not defined, a single link readout will be assigned

  constexpr int MaxLinksPerRU = 3;
  constexpr int MaxLinksPerCRU = 16;
  const auto& mp = m2r.getMapping();
  int lnkAssign[3][MaxLinksPerRU] = {
    // requested link cabling for IB, MB and OB
    /* // uncomment this to have 1 link per RU
    {9, 0, 0}, // IB
    {16, 0, 0}, // MB
    {28, 0, 0} // OB
     */
    {3, 3, 3}, // IB
    {5, 5, 6}, // MB
    {9, 9, 10} // OB
  };

  // this is an arbitrary mapping
  int nCRU = 0, nRUtot = 0, nRU = 0, nLinks = 0;
  int linkID = 0, cruIDprev = -1, cruID = o2::detectors::DetID::ITS << 10; // this will be the lowest CRUID
  std::string outFileLink;

  for (int ilr = 0; ilr < mp.NLayers; ilr++) {
    int nruLr = mp.getNStavesOnLr(ilr);
    int ruType = mp.getRUType(nRUtot); // IB, MB or OB
    int* lnkAs = lnkAssign[ruType];
    // count requested number of links per RU
    int nlk = 0;
    for (int i = 3; i--;) {
      nlk += lnkAs[i] ? 1 : 0;
    }
    outFileLink = outPrefix + "_lr" + std::to_string(ilr) + ".raw";
    for (int ir = 0; ir < nruLr; ir++) {
      int ruID = nRUtot++;
      bool accept = !(ruID < m2r.getRUSWMin() || ruID > m2r.getRUSWMax()); // ignored RUs ?
      if (accept) {
        m2r.getCreateRUDecode(ruID); // create RU container
        nRU++;
      }
      int accL = 0;
      for (int il = 0; il < MaxLinksPerRU; il++) { // create links
        if (accept) {
          nLinks++;
          auto& ru = *m2r.getRUDecode(ruID);
          uint32_t lanes = mp.getCablesOnRUType(ru.ruInfo->ruType); // lanes patter of this RU
          ru.links[il] = m2r.addGBTLink();
          auto link = m2r.getGBTLink(ru.links[il]);
          link->lanes = lanes & ((0x1 << lnkAs[il]) - 1) << (accL);
          link->idInCRU = linkID;
          link->cruID = cruID;
          link->feeID = mp.RUSW2FEEId(ruID, il);
          link->endPointID = 0; // 0 or 1
          accL += lnkAs[il];
          if (m2r.getVerbosity()) {
            LOG(info) << "RU" << ruID << '(' << ir << " on lr " << ilr << ") " << link->describe()
                      << " -> " << outFileLink;
          }
          // register the link in the writer, if not done here, its data will be dumped to common default file
          m2r.getWriter().registerLink(link->feeID, link->cruID, link->idInCRU,
                                       link->endPointID, outFileLink);
          //
          if (cruIDprev != cruID) { // just to count used CRUs
            cruIDprev = cruID;
            nCRU++;
          }
        }
        if ((++linkID) >= MaxLinksPerCRU) {
          linkID = 0;
          ++cruID;
        }
      }
    }
    if (linkID) {
      linkID = 0; // we don't want to put links of different layers on the same CRU
      ++cruID;
    }
  }
  LOG(info) << "Distributed " << nLinks << " links on " << nRU << " RUs in " << nCRU << " CRUs";
}
