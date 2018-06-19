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

#include "SimulationDataFormat/GeneratorHeader.h"
#include "SimulationDataFormat/CrossSectionInfo.h"
#include "SimulationDataFormat/HeavyIonInfo.h"
#include <iostream>

namespace o2
{
namespace dataformats
{

/*****************************************************************/
/*****************************************************************/

GeneratorHeader::GeneratorHeader()
  : TNamed("ALICEo2", "ALICEo2 Generator Header"), mTrackOffset(0), mNumberOfTracks(0), mNumberOfAttempts(0), mInfo()
{
  /** default constructor **/
}

/*****************************************************************/

GeneratorHeader::GeneratorHeader(const Char_t* name, const Char_t* title)
  : TNamed(name, title), mTrackOffset(0), mNumberOfTracks(0), mNumberOfAttempts(0), mInfo()
{
  /** constructor **/
}

/*****************************************************************/

GeneratorHeader::GeneratorHeader(const GeneratorHeader& rhs)
  : TNamed(rhs),
    mTrackOffset(rhs.mTrackOffset),
    mNumberOfTracks(rhs.mNumberOfTracks),
    mNumberOfAttempts(rhs.mNumberOfAttempts),
    mInfo(rhs.mInfo)
{
  /** copy constructor **/
}

/*****************************************************************/

GeneratorHeader& GeneratorHeader::operator=(const GeneratorHeader& rhs)
{
  /** operator= **/

  if (this == &rhs)
    return *this;
  TNamed::operator=(rhs);
  mTrackOffset = rhs.mTrackOffset;
  mNumberOfTracks = rhs.mNumberOfTracks;
  mNumberOfAttempts = rhs.mNumberOfAttempts;
  mInfo = rhs.mInfo;
  return *this;
}

/*****************************************************************/

GeneratorHeader::~GeneratorHeader() { /** default destructor **/}

/*****************************************************************/

void GeneratorHeader::Reset()
{
  /** reset **/

  mTrackOffset = 0;
  mNumberOfTracks = 0;
  mNumberOfAttempts = 0;
  for (auto& info : mInfo)
    info.second->reset();
}

/*****************************************************************/

void GeneratorHeader::Print(Option_t* opt) const
{
  /** print **/

  auto name = GetName();
  auto offset = getTrackOffset();
  auto ntracks = getNumberOfTracks();
  std::cout << ">> generator: " << name << " | tracks: " << offset << " -> " << offset + ntracks - 1 << std::endl;
  for (auto const& info : mInfo)
    info.second->print();
}

/*****************************************************************/

CrossSectionInfo* GeneratorHeader::getCrossSectionInfo() const
{
  /** get cross-section info **/

  std::string key = CrossSectionInfo::keyName();
  if (!mInfo.count(key))
    return nullptr;
  return dynamic_cast<CrossSectionInfo*>(mInfo.at(key));
}

/*****************************************************************/

CrossSectionInfo* GeneratorHeader::addCrossSectionInfo()
{
  /** add cross-section info **/

  std::string key = CrossSectionInfo::keyName();
  if (!mInfo.count(key))
    mInfo[key] = new CrossSectionInfo();
  return static_cast<CrossSectionInfo*>(mInfo.at(key));
}

/*****************************************************************/

void GeneratorHeader::removeCrossSectionInfo()
{
  /** remove cross-section info **/

  removeGeneratorInfo(CrossSectionInfo::keyName());
}

/*****************************************************************/

HeavyIonInfo* GeneratorHeader::getHeavyIonInfo() const
{
  /** get heavy-ion info **/

  std::string key = HeavyIonInfo::keyName();
  if (!mInfo.count(key))
    return nullptr;
  return dynamic_cast<HeavyIonInfo*>(mInfo.at(key));
}

/*****************************************************************/

HeavyIonInfo* GeneratorHeader::addHeavyIonInfo()
{
  /** add cross-section info **/

  std::string key = HeavyIonInfo::keyName();
  if (!mInfo.count(key))
    mInfo[key] = new HeavyIonInfo();
  return static_cast<HeavyIonInfo*>(mInfo.at(key));
}

/*****************************************************************/

void GeneratorHeader::removeHeavyIonInfo()
{
  /** remove cross-section info **/

  removeGeneratorInfo(HeavyIonInfo::keyName());
}

/*****************************************************************/
/*****************************************************************/

} /* namespace dataformats */
} /* namespace o2 */

ClassImp(o2::dataformats::GeneratorHeader)
