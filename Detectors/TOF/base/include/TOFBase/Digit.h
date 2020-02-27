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

  Digit(Int_t channel, Int_t tdc, Int_t tot, Int_t bc, Int_t label = -1, Int_t triggerorbit = 0, Int_t triggerbunch = 0);
  ~Digit() = default;

  /// Get global ordering key made of
  static ULong64_t getOrderingKey(Int_t channel, Int_t bc, Int_t /*tdc*/)
  {
    return ((static_cast<ULong64_t>(bc) << 18) + channel); // channel in the least significant bits; then shift by 18 bits (which cover the total number of channels) to write the BC number
  }

  Int_t getChannel() const { return mChannel; }
  void setChannel(Int_t channel) { mChannel = channel; }

  uint16_t getTDC() const { return mTDC; }
  void setTDC(uint16_t tdc) { mTDC = tdc; }

  uint16_t getTOT() const { return mTOT; }
  void setTOT(uint16_t tot) { mTOT = tot; }

  Int_t getBC() const { return mBC; }
  void setBC(Int_t bc) { mBC = bc; }

  Int_t getLabel() const { return mLabel; }
  void setLabel(Int_t label) { mLabel = label; }

  void printStream(std::ostream& stream) const;

  void merge(Int_t tdc, Int_t tot);

  void getPhiAndEtaIndex(int& phi, int& eta);

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

  void setTriggerOrbit(int value) { mTriggerOrbit = value; }
  int getTriggerOrbit() const { return mTriggerOrbit; }
  void setTriggerBunch(int value) { mTriggerBunch = value; }
  int getTriggerBunch() const { return mTriggerBunch; }

 private:
  friend class boost::serialization::access;

  Double_t mCalibratedTime; //!< time of the digits after calibration (not persistent; it will be filled during clusterization)
  Int_t mChannel;          ///< TOF channel index
  uint16_t mTDC;           ///< TDC bin number
  uint16_t mTOT;           ///< TOT bin number
  Int_t mBC;               ///< Bunch Crossing
  Int_t mLabel;            ///< Index of the corresponding entry in the MC label array
  Int_t mElectronIndex;    //!/< index in electronic format
  uint16_t mTriggerOrbit = 0;    //!< orbit id of trigger event
  uint16_t mTriggerBunch = 0;    //!< bunch id of trigger event
  Bool_t mIsUsedInCluster;       //!/< flag to declare that the digit was used to build a cluster
  Bool_t mIsProblematic = false; //!< flag to tell whether the channel of the digit was problemati; not persistent; default = ok

  ClassDefNV(Digit, 2);
};

std::ostream& operator<<(std::ostream& stream, const Digit& dig);

struct ReadoutWindowData {
  // 1st entry and number of entries in the full vector of digits
  // for given trigger (or BC or RO frame)
  o2::dataformats::RangeReference<int, int> ref;
  gsl::span<const Digit> getBunchChannelData(const gsl::span<const Digit> tfdata) const
  {
    // extract the span of channel data for this readout window from the whole TF data
    if (!ref.getEntries())
      return gsl::span<const Digit>(nullptr, ref.getEntries());
    return gsl::span<const Digit>(&tfdata[ref.getFirstEntry()], ref.getEntries());
  }

  ReadoutWindowData() = default;
  ReadoutWindowData(int first, int ne)
  {
    ref.setFirstEntry(first);
    ref.setEntries(ne);
  }

  int first() const { return ref.getFirstEntry(); }
  int size() const { return ref.getEntries(); }

  ClassDefNV(ReadoutWindowData, 1);
};

} // namespace tof
} // namespace o2
#endif
