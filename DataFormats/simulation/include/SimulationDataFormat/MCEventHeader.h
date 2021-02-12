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

#ifndef ALICEO2_DATAFORMATS_MCEVENTHEADER_H_
#define ALICEO2_DATAFORMATS_MCEVENTHEADER_H_

#include "FairMCEventHeader.h"
#include "SimulationDataFormat/MCEventStats.h"
#include "CommonUtils/RootSerializableKeyValueStore.h"
#include <string>
#include <Framework/Logger.h>

namespace o2
{
namespace dataformats
{

class GeneratorHeader;

/*****************************************************************/
/*****************************************************************/

// AliceO2 specialization of EventHeader class
class MCEventHeader : public FairMCEventHeader
{

 public:
  MCEventHeader() = default;
  MCEventHeader(const MCEventHeader& rhs) = default;
  MCEventHeader& operator=(const MCEventHeader& rhs) = default;
  ~MCEventHeader() override = default;

  /** setters **/
  void setEmbeddingFileName(std::string const& value) { mEmbeddingFileName = value; };
  void setEmbeddingEventIndex(Int_t value) { mEmbeddingEventIndex = value; };
  int getEmbeddedIndex() const { return mEmbeddingEventIndex; }

  /** methods to handle stored information **/

  void clearInfo()
  {
    mEventInfo.clear();
  };

  template <typename T>
  void putInfo(std::string const& key, T const& value)
  {
    mEventInfo.put<T>(key, value);
  };

  bool hasInfo(std::string const& key) const
  {
    return mEventInfo.has(key);
  }

  template <typename T>
  const T& getInfo(std::string const& key, bool& isvalid) const
  {
    o2::utils::RootSerializableKeyValueStore::GetState state;
    auto& ref = mEventInfo.getRef<T>(key, state);
    isvalid = (state == o2::utils::RootSerializableKeyValueStore::GetState::kOK);
    if (!isvalid) {
      LOG(WARNING) << "problem retrieving info '" << key << "': " << o2::utils::RootSerializableKeyValueStore::getStateString(state);
    }
    return ref;
  };

  /// prints a summary of info keys/types attached to this header
  void printInfo() const
  {
    mEventInfo.print();
  }

  /** methods **/
  virtual void Reset();

  MCEventStats& getMCEventStats() { return mEventStats; }

 protected:
  std::string mEmbeddingFileName;
  Int_t mEmbeddingEventIndex = 0;

  // store a view global properties that this event
  // had in the current simulation (which can be used quick filtering/searching)
  MCEventStats mEventStats{};
  o2::utils::RootSerializableKeyValueStore mEventInfo;

  ClassDefOverride(MCEventHeader, 3);

}; /** class MCEventHeader **/

/*****************************************************************/
/*****************************************************************/

} /* namespace dataformats */
} /* namespace o2 */

#endif /* ALICEO2_DATAFORMATS_MCEVENTHEADER_H_ */
