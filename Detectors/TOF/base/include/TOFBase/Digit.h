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

  Digit(Int_t channel, Int_t tdc, Int_t tot, Int_t bc, Int_t label = -1);
  ~Digit() = default;

  /// Get global ordering key made of
  static ULong64_t getOrderingKey(Int_t channel, Int_t bc, Int_t /*tdc*/)
  {
    return ((static_cast<ULong64_t>(bc) << 18) + channel); // channel in the least significant bits; then shift by 18 bits (which cover the total number of channels) to write the BC number
  }

  Int_t getChannel() const { return mChannel; }
  void setChannel(Int_t channel) { mChannel = channel; }

  Int_t getTDC() const { return mTDC; }
  void setTDC(Int_t tdc) { mTDC = tdc; }

  Int_t getTOT() const { return mTOT; }
  void setTOT(Int_t tot) { mTOT = tot; }

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

 private:
  friend class boost::serialization::access;

  Int_t mChannel;          ///< TOF channel index
  Int_t mTDC;              ///< TDC bin number
  Int_t mTOT;              ///< TOT bin number
  Int_t mBC;               ///< Bunch Crossing
  Int_t mLabel;            ///< Index of the corresponding entry in the MC label array
  Bool_t mIsUsedInCluster; //!/< flag to declare that the digit was used to build a cluster
  Int_t mElectronIndex;    //!/< index in electronic format
  Double_t mCalibratedTime;      //!< time of the digits after calibration (not persistent; it will be filled during clusterization)
  Bool_t mIsProblematic = false; //!< flag to tell whether the channel of the digit was problemati; not persistent; default = ok

  ClassDefNV(Digit, 1);
};

std::ostream& operator<<(std::ostream& stream, const Digit& dig);

struct ReadoutWindowData {
  // 1st entry and number of entries in the full vector of digits 
  // for given trigger (or BC or RO frame)
  int firstEntry;
  int nEntries;
  gsl::span<const Digit> getBunchChannelData(const gsl::span<const Digit> tfdata) const
  {
    // extract the span of channel data for this readout window from the whole TF data
    if(!nEntries) return gsl::span<const Digit>(nullptr, nEntries);
    return gsl::span<const Digit>(&tfdata[firstEntry], nEntries);
  }

  ReadoutWindowData() = default;
  ReadoutWindowData(int first, int ne)
  {
    firstEntry = first;
    nEntries = ne ;
  }

  int first() const {return firstEntry;}
  int size() const {return nEntries;}

  ClassDefNV(ReadoutWindowData, 1);
};

} // namespace tof
} // namespace o2
#endif
