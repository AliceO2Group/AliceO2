// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

///////////////////////////////////////////////////////////////////////////////
//                                                                           //
//  ChamberCalibrations class :                                                     //
//    This essentially replicates:  4 caldet, 1 calpad, and                  //
//    Most of the functionality inherint in those classes is present         //
//    with some additional functionality to clean up the interface.          //
///////////////////////////////////////////////////////////////////////////////

#include <TMath.h>
#include <TH1F.h>
#include <TH2F.h>
#include <TRobustEstimator.h>
#include <sstream>

#include "TRDBase/TRDGeometry.h"
#include "TRDBase/ChamberCalibrations.h"
#include "fairlogger/Logger.h"
using namespace o2::trd;

//___________________________________________________________________________________
int ChamberCalibrations::loadReferenceCalibrations(int run2number)
{
  //dont change any code here without changing the corresponding writing code in
  //AliRoot!
  //reference calibrations are pulled in for run 297595
  //filename is calibrations-run%d  text file, because why complicate my life.
  //  std::string filename="calibrations-run"+run2runumber;
  //ifstream inputfile(filename);
  //now read in the file.
  /* if(run2run==-1){
  //load the defaults the simplest way from run 297595.
  #include "../testdata/ChamberVdrift-run297595";
  #include "../testdata/ChamberExB-run297595";
  #include "../testdata/ChamberGainFactor-run297595";
  #include "../testdata/ChamberT0-run297595";

  }
*/
}
