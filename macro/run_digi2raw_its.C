#if !defined(__CLING__) || defined(__ROOTCLING__)
#include <TTree.h>
#include <TChain.h>
#include <TFile.h>
#include <TStopwatch.h>
#include <FairLogger.h>
#include <vector>
#include <string>

#include "ITSMFTReconstruction/ChipMappingITS.h"
#include "ITSMFTReconstruction/GBTWord.h"
#include "ITSMFTReconstruction/PayLoadCont.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "ITSMFTBase/Digit.h"
#include "ITSMFTReconstruction/RawPixelReader.h"
#endif

void run_digi2raw_its(std::string outName = "rawits.bin",         // name of the output binary file
                      std::string inpName = "itsdigits.root",     // name of the input ITS digits
                      std::string digTreeName = "o2sim",          // name of the digits tree
                      std::string digBranchName = "ITSDigit",     // name of the digits branch
                      std::string rofRecName = "ITSDigitROF",     // name of the ROF records tree and its branch
                      uint8_t ruSWMin = 0, uint8_t ruSWMax = 0xff // seq.ID of 1st and last RU (stave) to convert
)
{
  TStopwatch swTot;
  swTot.Start();
  using ROFR = o2::ITSMFT::ROFRecord;
  using ROFRVEC = std::vector<o2::ITSMFT::ROFRecord>;

  ///-------> input
  TChain digTree(digTreeName.c_str());
  TChain rofTree(rofRecName.c_str());

  digTree.AddFile(inpName.c_str());
  rofTree.AddFile(inpName.c_str());

  std::vector<o2::ITSMFT::Digit> digiVec, *digiVecP = &digiVec;
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
  o2::ITSMFT::PayLoadCont outBuffer;
  ///-------< output

  o2::ITSMFT::RawPixelReader<o2::ITSMFT::ChipMappingITS> rawReader;

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

      rawReader.digits2raw(digiVec, rofEntry.getIndex(), nDigROF, rofRec.getBCData(), outBuffer, ruSWMin, ruSWMax);

      fwrite(outBuffer.data(), 1, outBuffer.getSize(), outFl); //write to file
      outBuffer.clear();
    }
  } // loop over multiple ROFvectors (in case of chaining)

  fclose(outFl);
  //
  swTot.Stop();
  swTot.Print();
}
