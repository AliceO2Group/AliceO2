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

#include <string>
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

class GeneratorHeader
{

 public:
  /** default constructor **/
  GeneratorHeader();
  /** constructor **/
  GeneratorHeader(const std::string& name);
  /** copy constructor **/
  GeneratorHeader(const GeneratorHeader& rhs);
  /** operator= **/
  GeneratorHeader& operator=(const GeneratorHeader& rhs);
  /** destructor **/
  ~GeneratorHeader();

  /** getters **/
  const std::string& getName() const { return mName; };
  int getTrackOffset() const { return mTrackOffset; };
  int getNumberOfTracks() const { return mNumberOfTracks; };
  int getNumberOfAttempts() const { return mNumberOfAttempts; };
  CrossSectionInfo* getCrossSectionInfo() const;
  HeavyIonInfo* getHeavyIonInfo() const;

  /** setters **/
  void setName(const std::string& val) { mName = val; };
  void setTrackOffset(int val) { mTrackOffset = val; };
  void setNumberOfTracks(int val) { mNumberOfTracks = val; };
  void setNumberOfAttempts(int val) { mNumberOfAttempts = val; };

  /** methods **/
  void print() const;
  void reset();
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
  std::string mName;
  int mTrackOffset;
  int mNumberOfTracks;
  int mNumberOfAttempts;
  std::map<std::string, GeneratorInfo*> mInfo;

}; /** class GeneratorHeader **/

/*****************************************************************/
/*****************************************************************/

} /* namespace dataformats */
} /* namespace o2 */

#endif /* ALICEO2_DATAFORMATS_GENERATORHEADER_H_ */
