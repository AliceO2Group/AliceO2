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
#include <string>

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
  double getCrossSection() const { return mCrossSection; };
  double getCrossSectionError() const { return mCrossSectionError; };
  long int getAcceptedEvents() const { return mAcceptedEvents; };
  long int getAttemptedEvents() const { return mAttemptedEvents; };

  /** setters **/
  void setCrossSection(double val) { mCrossSection = val; };
  void setCrossSectionError(double val) { mCrossSectionError = val; };
  void setAcceptedEvents(long int val) { mAcceptedEvents = val; };
  void setAttemptedEvents(long int val) { mAttemptedEvents = val; };

  /** methods **/
  void print() const override;
  void reset() override;

  /** statics **/
  static std::string keyName() { return "cross-section"; };

 protected:
  /** data members **/
  double mCrossSection;      // Generated cross-section
  double mCrossSectionError; // Generated cross-section error
  double mAcceptedEvents;    // The number of events generated so far
  double mAttemptedEvents;   // The number of events attempted so far

}; /** class CrossSectionInfo **/

/*****************************************************************/
/*****************************************************************/

} /* namespace dataformats */
} /* namespace o2 */

#endif /* ALICEO2_DATAFORMATS_CROSSSECTIONINFO_H_ */
