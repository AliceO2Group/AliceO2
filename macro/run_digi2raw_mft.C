#if !defined(__CLING__) || defined(__ROOTCLING__)
#include <TTree.h>
#include <TChain.h>
#include <TFile.h>
#include <TStopwatch.h>
#include <FairLogger.h>
#include <vector>
#include <string>
#include <iomanip>
#include "ITSMFTReconstruction/ChipMappingMFT.h"
#include "ITSMFTReconstruction/GBTWord.h"
#include "ITSMFTReconstruction/PayLoadCont.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "ITSMFTBase/Digit.h"
#endif

#include "ITSMFTReconstruction/RawPixelReader.h"

void run_digi2raw_mft(std::string outName = "rawmft.bin",                           // name of the output binary file
                      std::string inpName = "mftdigits.root",                       // name of the input MFT digits
                      std::string digTreeName = "o2sim",                            // name of the digits tree
                      std::string digBranchName = "MFTDigit",                       // name of the digits branch
                      std::string rofRecName = "MFTDigitROF",                       // name of the ROF records tree and its branch
                      uint8_t ruSWMin = 0, uint8_t ruSWMax = 0xff,                  // seq.ID of 1st and last RU (stave) to convert
                      uint8_t superPageSize = o2::itsmft::NCRUPagesPerSuperpage / 2 // CRU superpage size, max = 256
)
{
  TStopwatch swTot;
  swTot.Start();
  using ROFR = o2::itsmft::ROFRecord;
  using ROFRVEC = std::vector<o2::itsmft::ROFRecord>;

  ///-------> input
  TChain digTree(digTreeName.c_str());
  TChain rofTree(rofRecName.c_str());

  digTree.AddFile(inpName.c_str());
  rofTree.AddFile(inpName.c_str());

  std::vector<o2::itsmft::Digit> digiVec, *digiVecP = &digiVec;
  if (!digTree.GetBranch(digBranchName.c_str())) {
    LOG(FATAL) << "Failed to find the branch " << digBranchName << " in the tree " << digTreeName;
  }
  digTree.SetBranchAddress(digBranchName.c_str(), &digiVecP);

  // ROF record entries in the digit tree
  ROFRVEC rofRecVec, *rofRecVecP = &rofRecVec;
  if (!rofTree.GetBranch(rofRecName.c_str())) {
    LOG(FATAL) << "Failed to find the branch " << rofRecName << " in the tree " << rofRecName;
  }
  rofTree.SetBranchAddress(rofRecName.c_str(), &rofRecVecP);
  ///-------< input

  ///-------> output
  if (outName.empty()) {
    outName = "raw" + digBranchName + ".raw";
    LOG(INFO) << "Output file name is not provided, set to " << outName << FairLogger::endl;
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

  o2::itsmft::RawPixelReader<o2::itsmft::ChipMappingMFT> rawReader;
  rawReader.setPadding128(true);
  rawReader.setVerbosity(0);

  //------------------------------------------------------------------------------->>>>
  // just as an example, we require here that the IB staves are read via 3 links,
  // while OB staves use only 1 link.
  // Note, that if the RU container is not defined, it will be created automatically
  // during encoding.
  // If the links of the container are not defined, a single link readout will be assigned
  const auto& mp = rawReader.getMapping();
  LOG(INFO) << "Number of RUs = " << mp.getNRUs();
  for (int ir = 0; ir < mp.getNRUs(); ir++) {
    auto& ru = rawReader.getCreateRUDecode(ir);               // create RU container
    uint32_t lanes = mp.getCablesOnRUType(ru.ruInfo->ruType); // lanes patter of this RU
    ru.links[0] = std::make_unique<o2::itsmft::GBTLink>();
    ru.links[0]->lanes = lanes; // single link reads all lanes
    LOG(INFO) << "RU " << std::setw(3) << ir << " type " << int(ru.ruInfo->ruType) << " on lr" << int(ru.ruInfo->layer)
              << " : FEEId 0x" << std::hex << std::setfill('0') << std::setw(6) << mp.RUSW2FEEId(ir, int(ru.ruInfo->layer))
              << " reads lanes " << std::bitset<25>(ru.links[0]->lanes);
  }

  //-------------------------------------------------------------------------------<<<<
  int lastTreeID = -1;
  long offs = 0, nEntProc = 0;
  for (int i = 0; i < rofTree.GetEntries(); i++) {
    rofTree.GetEntry(i);
    if (rofTree.GetTreeNumber() > lastTreeID) { // this part is needed for chained input
      if (lastTreeID > 0) {                     // new chunk, increase the offset
        offs += digTree.GetTree()->GetEntries();
      }
      lastTreeID = rofTree.GetTreeNumber();
    }

    for (const auto& rofRec : rofRecVec) {
      auto rofEntry = rofRec.getROFEntry();
      int nDigROF = rofRec.getNROFEntries();
      LOG(INFO) << "Processing ROF:" << rofRec.getROFrame() << " with " << nDigROF << " entries";
      if (!nDigROF) {
        LOG(INFO) << "Frame is empty"; // ??
        continue;
      }
      if (rofEntry.getEvent() != digTree.GetReadEntry() + offs || !nEntProc) {
        digTree.GetEntry(rofEntry.getEvent() + offs); // read tree entry containing needed ROF data
        nEntProc++;
      }
      int digIndex = rofEntry.getIndex(); // needed ROF digits start from this one
      int maxDigIndex = digIndex + nDigROF;
      LOG(INFO) << "BV===== digIndex " << digIndex << " maxDigIndex " << maxDigIndex << "\n";

      int nPagesCached = rawReader.digits2raw(digiVec, rofEntry.getIndex(), nDigROF, rofRec.getBCData(),
                                              ruSWMin, ruSWMax);

      if (nPagesCached >= superPageSize) {
        int nPagesFlushed = rawReader.flushSuperPages(superPageSize, outBuffer);
        fwrite(outBuffer.data(), 1, outBuffer.getSize(), outFl); //write to file
        outBuffer.clear();
        LOG(INFO) << "Flushed " << nPagesFlushed << " CRU pages";
      }
      //printf("BV===== stop after the first ROF!\n");
      //break;
    }
  } // loop over multiple ROFvectors (in case of chaining)

  // flush the rest
  int flushed = 0;
  do {
    flushed = rawReader.flushSuperPages(o2::itsmft::NCRUPagesPerSuperpage, outBuffer);
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
