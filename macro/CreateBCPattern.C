#if !defined(__CLING__) || defined(__ROOTCLING__)
#include "CommonDataFormat/BunchFilling.h"
#include "CommonConstants/LHCConstants.h"
#include "UDFSParser.h"
#include <TFile.h>
#include <string>
#endif

#include <fairlogger/Logger.h>

using namespace std;

void CreateBCPattern(const std::string& outFileName = "bcPattern.root", const string& objName = "")
{
  // example of interacting BC pattern creation

  o2::BunchFilling pattern;

  // Note that all setXXX methods have an extra last parameter (def.=-1) to indicate to what object
  // the setting is to be applied: -1: interacting bunches pattern, 0/1: A/C beams.

  /* this is an old, but still functional way of setting bunch filling, here we are setting the interacting bunches only

  // create 16 trains spaced by 96 BCs, with 48 interacting BCs per train
  // and 50 ns spacing between the BCs of the train and starting at BC 20
  pattern.setBCTrains(16, 96, 48, 2, 20);

  // add extra train of 6 bunches with 25 ns spacing in the very end
  pattern.setBCTrain(6, 1, o2::constants::lhc::LHCMaxBunches - 6);

  // add isolated bunches at slots 1,3,5
  pattern.setBC(1);
  pattern.setBC(3);
  pattern.setBC(5);

  */

  // this is a new way of setting the bunch filling: we set separately A,C beams, then set the interacting bunches as their .AND.
  // the resulting interacting bunches pattern will be the same as commented above, but we get also extra A, C non-interacting bunches
  pattern.setBCFilling("3(LH) 7(LH) 16(48(HL) 48(HL)) 233(HL) 6H", 0); // A Beam
  pattern.setBCFilling("3(LH) 7(HL) 16(48(HL) 48(LH)) 233(LH) 6H", 1); // B Beam
  pattern.setInteractingBCsFromBeams();

  pattern.print();

  if (!outFileName.empty()) {
    std::string nm = !objName.empty() ? objName : o2::BunchFilling::Class()->GetName();
    LOG(info) << "Storing pattern with name " << nm << " in a file " << outFileName;
    TFile outf(outFileName.c_str(), "update");
    outf.WriteObjectAny(&pattern, pattern.Class(), nm.c_str());
    outf.Close();
  }
}

// this version of CreateBCPattern takes a filling scheme definition file (*.csv from
// https://lpc.web.cern.ch/cgi-bin/filling_schemes.py) as input
void CreateBCPatternFromFile(const std::string& inFileName, const std::string& outFileName = "bcPattern.root", const string& objName = "")
{
  o2::BunchFilling pattern;

  // the UDFSParser allows to read and interpret the filling scheme definition file (*.csv)
  // the buckets are converted to P2BCs
  UDFSParser fsparser = UDFSParser(inFileName.data());  
  
  // 1. method to fill pattern
  // loop over all possible P2BCs and select colliding BCs
  // with this method only the interacting bunches are specified
  for (auto bcnum{0}; bcnum < o2::constants::lhc::LHCMaxBunches; bcnum++) {
    if (fsparser.isP2BCBB(bcnum)) {
      pattern.setBC(bcnum);
    }
  }
  pattern.print();
  
  // 2. method to fill pattern
  // one can also use the pattern string from fsparser.patternString to set the
  // filling schemes of both beams
  LOGF(info, "\n");
  LOGF(info, "patternString A-beam %s", fsparser.patternString(0));
  LOGF(info, "patternString C-beam %s\n", fsparser.patternString(1));

  pattern.setBCFilling(fsparser.patternString(0), 0); // A Beam
  pattern.setBCFilling(fsparser.patternString(1), 1); // C Beam
  pattern.setInteractingBCsFromBeams();
  
  // the printout of both pattern filling methods should be the same
  LOGF(info, "\n");
  pattern.print();
  
  // create the bcPattern.root file
  if (!outFileName.empty()) {
    std::string nm = !objName.empty() ? objName : o2::BunchFilling::Class()->GetName();
    LOGF(info, "Storing pattern with name %s in a file %s", nm, outFileName);
    TFile outf(outFileName.c_str(), "update");
    outf.WriteObjectAny(&pattern, pattern.Class(), nm.c_str());
    outf.Close();
  }
}
