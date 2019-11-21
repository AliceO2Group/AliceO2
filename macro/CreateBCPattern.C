#if !defined(__CLING__) || defined(__ROOTCLING__)
#include "CommonDataFormat/BunchFilling.h"
#include "CommonConstants/LHCConstants.h"
#include <TFile.h>
#include <string>
#endif

#include "FairLogger.h"

void CreateBCPattern(const std::string& outFileName = "bcPattern.root", const string& objName = "")
{
  // example of interacting BC pattern creation

  o2::BunchFilling pattern;

  // create 16 trains spaced by 96 BCs, with 48 interacting BCs per train
  // and 50 ns spacing between the BCs of the train and starting at BC 20
  pattern.setBCTrains(16, 96, 48, 2, 20);

  // add extra train of 6 bunches with 25 ns spacing in the very end
  pattern.setBCTrain(6, 1, o2::constants::lhc::LHCMaxBunches - 6);

  // add isolated bunchs at slots 1,3,5
  pattern.setBC(1);
  pattern.setBC(3);
  pattern.setBC(5);
  //
  pattern.print();

  if (!outFileName.empty()) {
    std::string nm = objName.empty() ? objName : o2::BunchFilling::Class()->GetName();
    LOG(INFO) << "Storing pattern with name " << nm << " in a file " << outFileName;
    TFile outf(outFileName.c_str(), "update");
    outf.WriteObjectAny(&pattern, pattern.Class(), nm.c_str());
    outf.Close();
  }
}
