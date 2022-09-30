// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \author R+Preghenella - January 2020

#include "Generators/GeneratorPythia6.h"
#include <fairlogger/Logger.h>
#include "FairPrimaryGenerator.h"
#include <iostream>
#include <iomanip>
#include "TPythia6.h"

namespace o2
{
namespace eventgen
{

/*****************************************************************/
/*****************************************************************/

GeneratorPythia6::GeneratorPythia6() : GeneratorTGenerator("ALICEo2", "ALICEo2 Pythia6 Generator")
{
  /** default constructor **/

  setTGenerator(TPythia6::Instance());

  mInterface = reinterpret_cast<void*>(TPythia6::Instance());
  mInterfaceName = "pythia6";
}

/*****************************************************************/

GeneratorPythia6::GeneratorPythia6(const Char_t* name, const Char_t* title) : GeneratorTGenerator(name, title)
{
  /** constructor **/

  setTGenerator(TPythia6::Instance());

  mInterface = reinterpret_cast<void*>(TPythia6::Instance());
  mInterfaceName = "pythia6";
}

/*****************************************************************/

Bool_t GeneratorPythia6::Init()
{
  /** init **/

  /** init base class **/
  Generator::Init();

  /** configuration file **/
  if (!mConfig.empty()) {
    std::ifstream fin(mConfig.c_str());
    if (!fin.is_open()) {
      LOG(fatal) << "cannot open configuration file: " << mConfig;
      return kFALSE;
    }
    /** process configuration file **/
    std::string whitespace = " \t\f\v\n\r";
    std::string comment = "#";
    for (std::string line; getline(fin, line);) {
      /** remove comments **/
      line = line.substr(0, line.find_first_of(comment));
      if (line.size() == 0) {
        continue;
      }
      /** remove leading/trailing whitespaces **/
      const auto line_begin = line.find_first_not_of(whitespace);
      const auto line_end = line.find_last_not_of(whitespace);
      if (line_begin == std::string::npos ||
          line_end == std::string::npos) {
        continue;
      }
      const auto line_range = line_end - line_begin + 1;
      line = line.substr(line_begin, line_range);
      if (line.size() == 0) {
        continue;
      }
      /** process command **/
      readString(line.c_str());
    }
    fin.close();
  }

  /** tune **/
  TPythia6::Instance()->Pytune(mTune);

  /** initialize **/
  TPythia6::Instance()->Initialize(mFrame.c_str(), mBeam.c_str(), mTarget.c_str(), mWin);

  /** success **/
  return kTRUE;
}

/*****************************************************************/
/*****************************************************************/

} /* namespace eventgen */
} /* namespace o2 */
