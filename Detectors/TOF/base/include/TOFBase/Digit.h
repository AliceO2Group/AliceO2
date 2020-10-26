// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_TOF_DIGIT_H_
#define ALICEO2_TOF_DIGIT_H_

#include <iosfwd>
#include "Rtypes.h"
#include "TOFBase/Geo.h"
#include "CommonDataFormat/RangeReference.h"
#include "CommonDataFormat/InteractionRecord.h"
#include <gsl/span>

#include <boost/serialization/base_object.hpp> // for base_object

namespace o2
{
namespace tof
{
/// \class Digit
/// \brief TOF digit implementation
class Digit
{
 public:
  Digit() = default;

  Digit(Int_t channel, Int_t tdc, Int_t tot, uint64_t bc, Int_t label = -1, uint32_t triggerorbit = 0, uint16_t triggerbunch = 0);
  Digit(Int_t channel, Int_t tdc, Int_t tot, uint32_t orbit, uint16_t bc, Int_t label = -1, uint32_t triggerorbit = 0, uint16_t triggerbunch = 0);
  ~Digit() = default;

  /// Get global ordering key made of
  static ULong64_t getOrderingKey(Int_t channel, uint64_t bc, Int_t /*tdc*/)
  {
    return ((static_cast<ULong64_t>(bc) << 18) + channel); // channel in the least significant bits; then shift by 18 bits (which cover the total number of channels) to write the BC number
  }
  ULong64_t getOrderingKey()
  {
    return getOrderingKey(mChannel, mIR.toLong(), mTDC);
  }

  Int_t getChannel() const { return mChannel; }
  void setChannel(Int_t channel) { mChannel = channel; }

  uint16_t getTDC() const { return mTDC; }
  void setTDC(uint16_t tdc) { mTDC = tdc; }

  uint16_t getTOT() const { return mTOT; }
  void setTOT(uint16_t tot) { mTOT = tot; }

  uint64_t getBC() const { return mIR.toLong(); }
  void setBC(uint64_t bc) { mIR.setFromLong(bc); }

  Int_t getLabel() const { return mLabel; }
  void setLabel(Int_t label) { mLabel = label; }

  void printStream(std::ostream& stream) const;

  void merge(Int_t tdc, Int_t tot);

  void getPhiAndEtaIndex(int& phi, int& eta) const;

  Bool_t isUsedInCluster() const { return mIsUsedInCluster; }

  void setIsUsedInCluster() { mIsUsedInCluster = kTRUE; }

  Int_t getElectronicIndex() const { return mElectronIndex; }
  void setElectronicIndex(Int_t ind) { mElectronIndex = ind; }
  Int_t getElCrateIndex() const { return Geo::getCrateFromECH(mElectronIndex); } // to be derived from mElectronIndex
  Int_t getElTRMIndex() const { return Geo::getTRMFromECH(mElectronIndex); }     // to be derived from mElectronIndex
  Int_t getElChainIndex() const { return Geo::getChainFromECH(mElectronIndex); } // to be derived from mElectronIndex
  Int_t getElTDCIndex() const { return Geo::getTDCFromECH(mElectronIndex); }     // to be derived from mElectronIndex
  Int_t getElChIndex() const { return Geo::getTDCChFromECH(mElectronIndex); }    // to be derived from mElectronIndex

  void setCalibratedTime(Double_t time) { mCalibratedTime = time; }
  double getCalibratedTime() const { return mCalibratedTime; }

  void setIsProblematic(bool flag) { mIsProblematic = flag; }
  bool isProblematic() const { return mIsProblematic; }

  void setTriggerOrbit(uint32_t value) { mTriggerOrbit = value; }
  uint32_t getTriggerOrbit() const { return mTriggerOrbit; }
  void setTriggerBunch(uint16_t value) { mTriggerBunch = value; }
  uint16_t getTriggerBunch() const { return mTriggerBunch; }

 private:
  friend class boost::serialization::access;

  Int_t mChannel;          ///< TOF channel index
  uint16_t mTDC;           ///< TDC bin number
  uint16_t mTOT;           ///< TOT bin number
  InteractionRecord mIR{0, 0}; ///< InteractionRecord (orbit and bc) when digit occurs
  Int_t mLabel;            ///< Index of the corresponding entry in the MC label array
  Double_t mCalibratedTime; //!< time of the digits after calibration (not persistent; it will be filled during clusterization)
  Int_t mElectronIndex;    //!/< index in electronic format
  uint32_t mTriggerOrbit = 0;    //!< orbit id of trigger event // RS: orbit must be 32bits long
  uint16_t mTriggerBunch = 0;    //!< bunch id of trigger event
  Bool_t mIsUsedInCluster;       //!/< flag to declare that the digit was used to build a cluster
  Bool_t mIsProblematic = false; //!< flag to tell whether the channel of the digit was problemati; not persistent; default = ok

  ClassDefNV(Digit, 4);
};

std::ostream& operator<<(std::ostream& stream, const Digit& dig);

struct ReadoutWindowData {
  // 1st entry and number of entries in the full vector of digits
  // for given trigger (or BC or RO frame)
  o2::dataformats::RangeReference<int, int> ref;
  o2::dataformats::RangeReference<int, int> refDiagnostic;

  // crate info for diagnostic patterns
  int mNdiaCrate[72] = {0};

  InteractionRecord mFirstIR{0, 0};

  const InteractionRecord& getBCData() const { return mFirstIR; }

  void addedDiagnostic(int crate) { mNdiaCrate[crate]++; }
  void setDiagnosticInCrate(int crate, int val) { mNdiaCrate[crate] = val; }
  int getDiagnosticInCrate(int crate) const { return mNdiaCrate[crate]; }

  void setBCData(int orbit, int bc)
  {
    mFirstIR.orbit = orbit;
    mFirstIR.bc = bc;
  }
  void setBCData(InteractionRecord& src)
  {
    mFirstIR.orbit = src.orbit;
    mFirstIR.bc = src.bc;
  }
  void SetBC(int bc) { mFirstIR.bc = bc; }
  void SetOrbit(int orbit) { mFirstIR.orbit = orbit; }

  gsl::span<const Digit> getBunchChannelData(const gsl::span<const Digit> tfdata) const
  {
    // extract the span of channel data for this readout window from the whole TF data
    return ref.getEntries() ? gsl::span<const Digit>(&tfdata[ref.getFirstEntry()], ref.getEntries()) : gsl::span<const Digit>();
  }

  ReadoutWindowData() = default;
  ReadoutWindowData(int first, int ne)
  {
    ref.setFirstEntry(first);
    ref.setEntries(ne);
    refDiagnostic.setFirstEntry(0);
    refDiagnostic.setEntries(0);
  }

  int first() const { return ref.getFirstEntry(); }
  int size() const { return ref.getEntries(); }
  int firstDia() const { return refDiagnostic.getFirstEntry(); }
  int sizeDia() const { return refDiagnostic.getEntries(); }

  void setFirstEntry(int first) { ref.setFirstEntry(first); }
  void setNEntries(int ne) { ref.setEntries(ne); }
  void setFirstEntryDia(int first) { refDiagnostic.setFirstEntry(first); }
  void setNEntriesDia(int ne) { refDiagnostic.setEntries(ne); }

  ClassDefNV(ReadoutWindowData, 3);
};

} // namespace tof
} // namespace o2
#endif
