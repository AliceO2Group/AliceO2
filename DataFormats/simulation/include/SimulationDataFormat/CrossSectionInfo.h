// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \author R+Preghenella - August 2017

#ifndef ALICEO2_DATAFORMATS_CROSSSECTIONINFO_H_
#define ALICEO2_DATAFORMATS_CROSSSECTIONINFO_H_

#include "SimulationDataFormat/GeneratorInfo.h"

namespace o2
{
namespace dataformats
{

/*****************************************************************/
/*****************************************************************/

class CrossSectionInfo : public GeneratorInfo
{

 public:
  /** default constructor **/
  CrossSectionInfo();
  /** copy constructor **/
  CrossSectionInfo(const CrossSectionInfo& rhs);
  /** operator= **/
  CrossSectionInfo& operator=(const CrossSectionInfo& rhs);
  /** destructor **/
  ~CrossSectionInfo() override;

  /** getters **/
  Double_t getCrossSection() const { return mCrossSection; };
  Double_t getCrossSectionError() const { return mCrossSectionError; };
  Long64_t getAcceptedEvents() const { return mAcceptedEvents; };
  Long64_t getAttemptedEvents() const { return mAttemptedEvents; };

  /** setters **/
  void setCrossSection(Double_t val) { mCrossSection = val; };
  void setCrossSectionError(Double_t val) { mCrossSectionError = val; };
  void setAcceptedEvents(Long64_t val) { mAcceptedEvents = val; };
  void setAttemptedEvents(Long64_t val) { mAttemptedEvents = val; };

  /** methods **/
  void Print(Option_t* opt = "") const override;
  void Reset() override;

  /** statics **/
  static std::string keyName() { return "cross-section"; };

 protected:
  /** data members **/
  Double_t mCrossSection;      // Generated cross-section
  Double_t mCrossSectionError; // Generated cross-section error
  Long64_t mAcceptedEvents;    // The number of events generated so far
  Long64_t mAttemptedEvents;   // The number of events attempted so far

  ClassDefOverride(CrossSectionInfo, 1);

}; /** class CrossSectionInfo **/

/*****************************************************************/
/*****************************************************************/

} /* namespace dataformats */
} /* namespace o2 */

#endif /* ALICEO2_DATAFORMATS_CROSSSECTIONINFO_H_ */
