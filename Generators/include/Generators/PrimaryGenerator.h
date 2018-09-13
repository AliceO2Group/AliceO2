// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \author R+Preghenella - June 2017

#ifndef ALICEO2_EVENTGEN_PRIMARYGENERATOR_H_
#define ALICEO2_EVENTGEN_PRIMARYGENERATOR_H_

#include "FairPrimaryGenerator.h"

class TFile;
class TTree;
class TString;
class FairMCEventHeader;

namespace o2
{
namespace dataformats
{
class GeneratorHeader;
}
}

using o2::dataformats::GeneratorHeader;

namespace o2
{
namespace eventgen
{

/*****************************************************************/
/*****************************************************************/

class PrimaryGenerator : public FairPrimaryGenerator
{

 public:
  /** default constructor **/
  PrimaryGenerator();
  /** destructor **/
  ~PrimaryGenerator() override;

  /** Public method GenerateEvent
To be called at the beginning of each event from FairMCApplication.
Generates an event vertex and calls the ReadEvent methods from the
registered generators.
*@param pStack The particle stack
*@return kTRUE if successful, kFALSE if not
**/
  Bool_t GenerateEvent(FairGenericStack* pStack) override;

  /** Public method AddHeader
Adding a generator header to the MC event header.
To be called after all tracks have been added.
  **/
  void addHeader(GeneratorHeader* header);

  /** set interaction diamond position **/
  void setInteractionDiamond(const Double_t* xyz, const Double_t* sigmaxyz, Bool_t smear = kTRUE);
  void setInteractionDiamond(const FairMCEventHeader* event);

  /** Public embedding methods **/
  Bool_t embedInto(TString fname);

 protected:
  /** copy constructor **/
  PrimaryGenerator(const PrimaryGenerator&);
  /** operator= **/
  PrimaryGenerator& operator=(const PrimaryGenerator&);

  /** embedding members **/
  TFile* mEmbedFile = nullptr;
  TTree* mEmbedTree = nullptr;
  Int_t mEmbedEntries = 0;
  Int_t mEmbedCounter = 0;
  FairMCEventHeader* mEmbedEvent = nullptr;

  ClassDefOverride(PrimaryGenerator, 1);

}; /** class PrimaryGenerator **/

/*****************************************************************/
/*****************************************************************/

} /* namespace eventgen */
} /* namespace o2 */

#endif /* ALICEO2_EVENTGEN_PRIMARYGENERATOR_H_ */
