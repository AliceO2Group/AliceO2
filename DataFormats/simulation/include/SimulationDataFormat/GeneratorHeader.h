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

#ifndef ALICEO2_DATAFORMATS_GENERATORHEADER_H_
#define ALICEO2_DATAFORMATS_GENERATORHEADER_H_

#include "TNamed.h"
#include <map>

namespace o2
{
namespace dataformats
{

class GeneratorInfo;
class CrossSectionInfo;
class HeavyIonInfo;

/*****************************************************************/
/*****************************************************************/

class GeneratorHeader : public TNamed
{

 public:
  /** default constructor **/
  GeneratorHeader();
  /** constructor **/
  GeneratorHeader(const Char_t* name, const Char_t* title = "ALICEo2 Generator Header");
  /** copy constructor **/
  GeneratorHeader(const GeneratorHeader& rhs);
  /** operator= **/
  GeneratorHeader& operator=(const GeneratorHeader& rhs);
  /** destructor **/
  ~GeneratorHeader() override;

  /** getters **/
  Int_t getTrackOffset() const { return mTrackOffset; };
  Int_t getNumberOfTracks() const { return mNumberOfTracks; };
  Int_t getNumberOfAttempts() const { return mNumberOfAttempts; };
  CrossSectionInfo* getCrossSectionInfo() const;
  HeavyIonInfo* getHeavyIonInfo() const;

  /** setters **/
  void setTrackOffset(Int_t val) { mTrackOffset = val; };
  void setNumberOfTracks(Int_t val) { mNumberOfTracks = val; };
  void setNumberOfAttempts(Int_t val) { mNumberOfAttempts = val; };

  /** methods **/
  void Print(Option_t* opt = "") const override;
  virtual void Reset();
  void removeGeneratorInfo(const std::string& key)
  {
    if (mInfo.count(key))
      mInfo.erase(key);
  };
  CrossSectionInfo* addCrossSectionInfo();
  void removeCrossSectionInfo();
  HeavyIonInfo* addHeavyIonInfo();
  void removeHeavyIonInfo();

 protected:
  /** data members **/
  Int_t mTrackOffset;
  Int_t mNumberOfTracks;
  Int_t mNumberOfAttempts;
  std::map<std::string, GeneratorInfo*> mInfo;

  ClassDefOverride(GeneratorHeader, 1);

}; /** class GeneratorHeader **/

/*****************************************************************/
/*****************************************************************/

} /* namespace dataformats */
} /* namespace o2 */

#endif /* ALICEO2_DATAFORMATS_GENERATORHEADER_H_ */
