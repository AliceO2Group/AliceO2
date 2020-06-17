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

#ifndef ALICEO2_EVENTGEN_GENERATORPYTHIA8_H_
#define ALICEO2_EVENTGEN_GENERATORPYTHIA8_H_

#include "Generators/Generator.h"
#include "Pythia8/Pythia.h"

namespace o2
{
namespace eventgen
{

/*****************************************************************/
/*****************************************************************/

class GeneratorPythia8 : public Generator
{

 public:
  /** default constructor **/
  GeneratorPythia8();
  /** constructor **/
  GeneratorPythia8(const Char_t* name, const Char_t* title = "ALICEo2 Pythia8 Generator");
  /** destructor **/
  ~GeneratorPythia8() override = default;

  /** Initialize the generator if needed **/
  Bool_t Init() override;

  /** setters **/
  void setConfig(std::string val) { mConfig = val; };
  void setHooksFileName(std::string val) { mHooksFileName = val; };
  void setHooksFuncName(std::string val) { mHooksFuncName = val; };

  /** methods **/
  bool readString(std::string val) { return mPythia.readString(val, true); };
  bool readFile(std::string val) { return mPythia.readFile(val, true); };

 protected:
  /** copy constructor **/
  GeneratorPythia8(const GeneratorPythia8&);
  /** operator= **/
  GeneratorPythia8& operator=(const GeneratorPythia8&);

  /** methods to override **/
  Bool_t generateEvent() override;
  Bool_t importParticles() override;

  /** methods that can be overridded **/
  void updateHeader(FairMCEventHeader* eventHeader) override;

  /** Pythia8 **/
  Pythia8::Pythia mPythia; //!

  /** configuration **/
  std::string mConfig;
  std::string mHooksFileName;
  std::string mHooksFuncName;

  ClassDefOverride(GeneratorPythia8, 1);

}; /** class GeneratorPythia8 **/

/*****************************************************************/
/*****************************************************************/

} // namespace eventgen
} // namespace o2

#endif /* ALICEO2_EVENTGEN_GENERATORPYTHIA8_H_ */
