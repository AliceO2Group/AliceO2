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
#include "ITSMFTReconstruction/PayLoadCont.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "DataFormatsITSMFT/Digit.h"
#endif

#include "ITSMFTReconstruction/RawPixelReader.h"

void run_digi2raw_its(std::string outName = "rawits.bin",                        // name of the output binary file
                      std::string inpName = "itsdigits.root",                    // name of the input ITS digits
                      std::string digTreeName = "o2sim",                         // name of the digits tree
                      std::string digBranchName = "ITSDigit",                    // name of the digits branch
                      std::string rofRecName = "ITSDigitROF",                    // name of the ROF records branch
                      uint8_t ruSWMin = 0, uint8_t ruSWMax = 0xff,               // seq.ID of 1st and last RU (stave) to convert
                      uint16_t superPageSize = o2::itsmft::NCRUPagesPerSuperpage // CRU superpage size, max = 256
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
    LOG(FATAL) << "Failed to find the branch " << rofRecName << " in the tree " << digTreeName;
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
  o2::itsmft::PayLoadCont outBuffer;
  ///-------< output

  o2::itsmft::RawPixelReader<o2::itsmft::ChipMappingITS> rawReader;
  rawReader.setPadding128(true);
  rawReader.setVerbosity(0);

  //------------------------------------------------------------------------------->>>>
  // just as an example, we require here that the staves are read via 3 links, with partitioning according to lnkXB below
  // while OB staves use only 1 link.
  // Note, that if the RU container is not defined, it will be created automatically
  // during encoding.
  // If the links of the container are not defined, a single link readout will be assigned
  const auto& mp = rawReader.getMapping();
  int lnkAssign[3][3] = {
    {3, 3, 3}, // IB
    {5, 5, 6}, // MB
    {9, 9, 10} // OB
  };
  for (int ir = 0; ir < mp.getNRUs(); ir++) {
    auto& ru = rawReader.getCreateRUDecode(ir);               // create RU container
    uint32_t lanes = mp.getCablesOnRUType(ru.ruInfo->ruType); // lanes patter of this RU
    int* lnkAs = lnkAssign[ru.ruInfo->ruType];
    int accL = 0;
    for (int il = 0; il < 3; il++) { // create links
      if (lnkAs[il]) {
        ru.links[il] = std::make_unique<o2::itsmft::GBTLink>();
        ru.links[il]->lanes = lanes & ((0x1 << lnkAs[il]) - 1) << (accL);
        ru.links[il]->id = il;
        ru.links[il]->feeID = mp.RUSW2FEEId(ir, il);
        accL += lnkAs[il];
        LOG(INFO) << "RU " << std::setw(3) << ir << " on lr" << int(ru.ruInfo->layer)
                  << " : FEEId 0x" << std::hex << std::setfill('0') << std::setw(6) << ru.links[il]->feeID
                  << " reads lanes " << std::bitset<28>(ru.links[il]->lanes);
      }
    }
  }

  //-------------------------------------------------------------------------------<<<<
  for (int i = 0; i < digTree.GetEntries(); i++) {
    digTree.GetEntry(i);

    for (const auto& rofRec : rofRecVec) {
      int rofEntry = rofRec.getFirstEntry();
      int nDigROF = rofRec.getNEntries();
      LOG(INFO) << "Processing ROF:" << rofRec.getROFrame() << " with " << nDigROF << " entries";
      rofRec.print();
      if (!nDigROF) {
        LOG(INFO) << "Frame is empty"; // ??
        continue;
      }
      int maxDigIndex = rofEntry + nDigROF;

      int nPagesCached = rawReader.digits2raw(digiVec, rofEntry, nDigROF, rofRec.getBCData(),
                                              ruSWMin, ruSWMax);
      LOG(INFO) << "Pages chached " << nPagesCached << " superpage: " << int(superPageSize);
      if (nPagesCached >= superPageSize) {
        int nPagesFlushed = rawReader.flushSuperPages(superPageSize, outBuffer);
        fwrite(outBuffer.data(), 1, outBuffer.getSize(), outFl); //write to file
        outBuffer.clear();
        LOG(INFO) << "Flushed " << nPagesFlushed << " CRU pages";
      }
    }
  } // loop over multiple ROFvectors (in case of chaining)

  // flush the rest
  int flushed = 0;
  do {
    flushed = rawReader.flushSuperPages(superPageSize, outBuffer, false);
    fwrite(outBuffer.data(), 1, outBuffer.getSize(), outFl); //write to file
    if (flushed) {
      LOG(INFO) << "Flushed final " << flushed << " CRU pages";
    }
    outBuffer.clear();
  } while (flushed);

  fclose(outFl);
  //
  swTot.Stop();
  swTot.Print();
}
