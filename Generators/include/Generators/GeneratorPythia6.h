// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \author R+Preghenella - January 2020

#ifndef ALICEO2_EVENTGEN_GENERATORPYTHIA6_H_
#define ALICEO2_EVENTGEN_GENERATORPYTHIA6_H_

#include "Generators/GeneratorTGenerator.h"
#include "TPythia6.h"

namespace o2
{
namespace eventgen
{

/*****************************************************************/
/*****************************************************************/

class GeneratorPythia6 : public GeneratorTGenerator
{

 public:
  /** default constructor **/
  GeneratorPythia6();
  /** constructor **/
  GeneratorPythia6(const Char_t* name, const Char_t* title = "ALICEo2 Pythia6 Generator");
  /** destructor **/
  ~GeneratorPythia6() override = default;

  /** Initialize the generator if needed **/
  Bool_t Init() override;

  /** setters **/
  void setConfig(std::string val) { mConfig = val; };
  void setTune(int val) { mTune = val; };
  void setFrame(std::string val) { mFrame = val; };
  void setBeam(std::string val) { mBeam = val; };
  void setTarget(std::string val) { mTarget = val; };
  void setWin(double val) { mWin = val; };

  /** methods **/
  void readString(std::string val) { TPythia6::Instance()->Pygive(val.c_str()); };

 protected:
  /** copy constructor **/
  GeneratorPythia6(const GeneratorPythia6&);
  /** operator= **/
  GeneratorPythia6& operator=(const GeneratorPythia6&);

  /** configuration **/
  std::string mConfig;
  int mTune = 350;

  /** initialization members **/
  std::string mFrame = "CMS";
  std::string mBeam = "p";
  std::string mTarget = "p";
  double mWin = 14000.;

  ClassDefOverride(GeneratorPythia6, 1);

}; /** class GeneratorPythia6 **/

/*****************************************************************/
/*****************************************************************/

} // namespace eventgen
} // namespace o2

#endif /* ALICEO2_EVENTGEN_GENERATORPYTHIA8_H_ */
