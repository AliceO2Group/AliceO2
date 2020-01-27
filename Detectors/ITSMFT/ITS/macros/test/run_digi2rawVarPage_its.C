#if !defined(__CLING__) || defined(__ROOTCLING__)
#include <TTree.h>
#include <TChain.h>
#include <TFile.h>
#include <TStopwatch.h>
#include <FairLogger.h>
#include <vector>
#include <string>
#include <iomanip>
#include "ITSMFTReconstruction/ChipMappingITS.h"
#include "ITSMFTReconstruction/GBTWord.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "DataFormatsParameters/GRPObject.h"
#include "DataFormatsITSMFT/Digit.h"
#include "ITSMFTSimulation/MC2RawEncoder.h"
#endif

// demo macro the MC->raw conversion with new (variable page size) format

void run_digi2rawVarPage_its(std::string outName = "rawits.bin",          // name of the output binary file
                             std::string inpName = "itsdigits.root",      // name of the input ITS digits
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
    LOG(FATAL) << "Failed to find the branch " << digBranchName << " in the tree " << digTreeName;
  }
  digTree.SetBranchAddress(digBranchName.c_str(), &digiVecP);

  // ROF record entries in the digit tree
  ROFRVEC rofRecVec, *rofRecVecP = &rofRecVec;
  if (!digTree.GetBranch(rofRecName.c_str())) {
    LOG(FATAL) << "Failed to find the branch " << rofRecName << " in the tree " << rofRecName;
  }
  digTree.SetBranchAddress(rofRecName.c_str(), &rofRecVecP);
  ///-------< input

  ///-------> output
  if (outName.empty()) {
    outName = "raw" + digBranchName + ".raw";
    LOG(INFO) << "Output file name is not provided, set to " << outName;
  }
  auto outFl = fopen(outName.c_str(), "wb");
  if (!outFl) {
    LOG(FATAL) << "failed to open raw data output file " << outName;
    ;
  } else {
    LOG(INFO) << "opened raw data output file " << outName;
  }
  ///-------< output

  const auto grp = o2::parameters::GRPObject::loadFrom(inputGRP);

  o2::itsmft::MC2RawEncoder<o2::itsmft::ChipMappingITS> m2r;
  m2r.setVerbosity(2);
  m2r.setOutFile(outFl);
  m2r.setContinuousReadout(grp->isDetContinuousReadOut(o2::detectors::DetID::ITS)); // must be set explicitly

  m2r.setMinMaxRUSW(ruSWMin, ruSWMax);

  //------------------------------------------------------------------------------->>>>
  // just as an example, we require here that the staves are read via 3 links, with partitioning according to lnkXB below
  // while OB staves use only 1 link.
  // Note, that if the RU container is not defined, it will be created automatically
  // during encoding.
  // If the links of the container are not defined, a single link readout will be assigned
  const auto& mp = m2r.getMapping();
  int lnkAssign[3][3] = {
    /* // uncomment this to have 1 link per RU
    {9, 0, 0}, // IB
    {16, 0, 0}, // MB
    {28, 0, 0} // OB
     */
    {3, 3, 3}, // IB
    {5, 5, 6}, // MB
    {9, 9, 10} // OB
  };
  for (int ir = m2r.getRUSWMin(); ir <= m2r.getRUSWMax(); ir++) {

    auto& ru = m2r.getCreateRUDecode(ir);                     // create RU container
    uint32_t lanes = mp.getCablesOnRUType(ru.ruInfo->ruType); // lanes patter of this RU
    int* lnkAs = lnkAssign[ru.ruInfo->ruType];
    int accL = 0;
    for (int il = 0; il < 3; il++) { // create links
      if (lnkAs[il]) {
        ru.links[il] = std::make_unique<o2::itsmft::GBTLink>();
        ru.links[il]->lanes = lanes & ((0x1 << lnkAs[il]) - 1) << (accL);
        ru.links[il]->id = il;
        ru.links[il]->cruID = ir;
        ru.links[il]->feeID = mp.RUSW2FEEId(ir, il);
        accL += lnkAs[il];
        LOG(INFO) << "RU " << std::setw(3) << ir << " on lr" << int(ru.ruInfo->layer)
                  << " : FEEId 0x" << std::hex << std::setfill('0') << std::setw(6) << ru.links[il]->feeID
                  << " reads lanes " << std::bitset<28>(ru.links[il]->lanes);
      }
    }
  }

  //-------------------------------------------------------------------------------<<<<
  int lastTreeID = -1;
  long offs = 0, nEntProc = 0;
  for (int i = 0; i < digTree.GetEntries(); i++) {
    digTree.GetEntry(i);
    for (const auto& rofRec : rofRecVec) {
      int nDigROF = rofRec.getNEntries();
      LOG(INFO) << "Processing ROF:" << rofRec.getROFrame() << " with " << nDigROF << " entries";
      rofRec.print();
      if (!nDigROF) {
        LOG(INFO) << "Frame is empty"; // ??
        continue;
      }
      nEntProc++;
      auto dgs = gsl::span<const o2::itsmft::Digit>(&digiVec[rofRec.getFirstEntry()], nDigROF);
      m2r.digits2raw(dgs, rofRec.getBCData());
    }
  } // loop over multiple ROFvectors (in case of chaining)

  m2r.finalize(); // finish TF and flush data
  //
  fclose(outFl);
  m2r.setOutFile(nullptr);
  //
  swTot.Stop();
  swTot.Print();
}
