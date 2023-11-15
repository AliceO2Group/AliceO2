// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
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

/** Common keys for information in MC event header */
struct MCInfoKeys {
  /** @{
@name HepMC3 heavy-ion fields */
  static constexpr const char* impactParameter = "Bimpact";
  static constexpr const char* nPart = "Npart";
  static constexpr const char* nPartProjectile = "Npart_proj";
  static constexpr const char* nPartTarget = "Npart_targ";
  static constexpr const char* nColl = "Ncoll";
  static constexpr const char* nCollHard = "Ncoll_hard";
  static constexpr const char* nCollNNWounded = "NColl_NNw";
  static constexpr const char* nCollNWoundedN = "NColl_NwN";
  static constexpr const char* nCollNWoundedNwounded = "NColl_NwNW";
  static constexpr const char* planeAngle = "eventPsi";
  static constexpr const char* sigmaInelNN = "sigmaInelNN";
  static constexpr const char* centrality = "centrality";
  static constexpr const char* nSpecProjectileProton = "Nspec_proj_p";
  static constexpr const char* nSpecProjectileNeutron = "Nspec_proj_n";
  static constexpr const char* nSpecTargetProton = "Nspec_targ_p";
  static constexpr const char* nSpecTargetNeutron = "Nspec_targ_n";
  /** @} */
  /** @{
@name HepMC3 PDF information

In principle a header can have many of these.  In that case,
each set should be prefixed with "_<X>" where "<X>" is a
serial number.
  */
  static constexpr const char* pdfParton1Id = "pdf_parton_1_id";
  static constexpr const char* pdfParton2Id = "pdf_parton_2_id";
  static constexpr const char* pdfX1 = "pdf_x1";
  static constexpr const char* pdfX2 = "pdf_x2";
  static constexpr const char* pdfScale = "pdf_scale";
  static constexpr const char* pdfXF1 = "pdf_par_x1";
  static constexpr const char* pdfXF2 = "pdf_par_x2";
  static constexpr const char* pdfCode1 = "pdf_lhc_1_id";
  static constexpr const char* pdfCode2 = "pdf_lhc_2_id";
  /** @} */
  /** @{
@name HepMC3 cross-section information

In principle we can have one cross section per weight. In that
case, each should be post-fixed by "_<X>" where "<X>" is a
serial number.  These should then matcht possible names of
weights.
  */
  static constexpr const char* acceptedEvents = "accepted_events";
  static constexpr const char* attemptedEvents = "attempted_events";
  static constexpr const char* xSection = "cross_section";
  static constexpr const char* xSectionError = "cross_section_error";
  /** @} */
  /** @{
@name Common fields */
  static constexpr const char* generator = "generator";
  static constexpr const char* generatorVersion = "version";
  static constexpr const char* processName = "processName";
  static constexpr const char* processCode = "processCode";
  static constexpr const char* weight = "weight";
  /** @} */
};

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
      LOG(warning) << "problem retrieving info '" << key << "': " << o2::utils::RootSerializableKeyValueStore::getStateString(state);
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

  void setDetId2HitBitLUT(std::vector<int> const& v) { mDetId2HitBitIndex = v; }
  std::vector<int> const& getDetId2HitBitLUT() const { return mDetId2HitBitIndex; }

  /// create a standalone ROOT file/tree with only the MCHeader branch
  static void extractFileFromKinematics(std::string_view kinefilename, std::string_view targetfilename);

 protected:
  std::string mEmbeddingFileName;
  Int_t mEmbeddingEventIndex = 0;

  // store a few global properties that this event
  // had in the current simulation (which can be used quick filtering/searching)
  MCEventStats mEventStats{};
  std::vector<int> mDetId2HitBitIndex;
  o2::utils::RootSerializableKeyValueStore mEventInfo;

  ClassDefOverride(MCEventHeader, 4);

}; /** class MCEventHeader **/

/*****************************************************************/
/*****************************************************************/

} /* namespace dataformats */
} /* namespace o2 */

#endif /* ALICEO2_DATAFORMATS_MCEVENTHEADER_H_ */
