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

  /** methods to override **/
  Bool_t generateEvent() override;
  Bool_t importParticles() override { return importParticles(mPythia.event); };

  /** setters **/
  void setConfig(std::string val) { mConfig = val; };
  void setHooksFileName(std::string val) { mHooksFileName = val; };
  void setHooksFuncName(std::string val) { mHooksFuncName = val; };
  void setUserHooks(Pythia8::UserHooks* hooks)
  {
#if PYTHIA_VERSION_INTEGER < 8300
    mPythia.setUserHooksPtr(hooks);
#else
    mPythia.setUserHooksPtr(std::shared_ptr<Pythia8::UserHooks>(hooks));
#endif
  }

  /** methods **/
  bool readString(std::string val) { return mPythia.readString(val, true); };
  bool readFile(std::string val) { return mPythia.readFile(val, true); };

  /** utilities **/
  void getNcoll(int& nColl)
  {
    getNcoll(mPythia.info, nColl);
  };
  void getNpart(int& nPart)
  {
    getNpart(mPythia.info, nPart);
  };
  void getNpart(int& nProtonProj, int& nNeutronProj, int& nProtonTarg, int& nNeutronTarg)
  {
    getNpart(mPythia.info, nProtonProj, nNeutronProj, nProtonTarg, nNeutronTarg);
  };
  void getNremn(int& nProtonProj, int& nNeutronProj, int& nProtonTarg, int& nNeutronTarg)
  {
    getNremn(mPythia.event, nProtonProj, nNeutronProj, nProtonTarg, nNeutronTarg);
  };

 protected:
  /** copy constructor **/
  GeneratorPythia8(const GeneratorPythia8&);
  /** operator= **/
  GeneratorPythia8& operator=(const GeneratorPythia8&);

  /** methods that can be overridded **/
  void updateHeader(o2::dataformats::MCEventHeader* eventHeader) override;

  /** internal methods **/
  Bool_t importParticles(Pythia8::Event& event);

  /** utilities **/
  void selectFromAncestor(int ancestor, Pythia8::Event& inputEvent, Pythia8::Event& outputEvent);
  void getNcoll(const Pythia8::Info& info, int& nColl);
  void getNpart(const Pythia8::Info& info, int& nPart);
  void getNpart(const Pythia8::Info& info, int& nProtonProj, int& nNeutronProj, int& nProtonTarg, int& nNeutronTarg);
  void getNremn(const Pythia8::Event& event, int& nProtonProj, int& nNeutronProj, int& nProtonTarg, int& nNeutronTarg);

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
