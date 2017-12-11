// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file ClusterNative.h
/// \brief Class of a TPC cluster in TPC-native coordinates (row, time)
/// \author David Rohr
#ifndef ALICEO2_DATAFORMATSTPC_CLUSTERNATIVE_H
#define ALICEO2_DATAFORMATSTPC_CLUSTERNATIVE_H

#include <vector>
#include "TPCBase/Constants.h"

namespace o2 { class MCCompLabel; namespace dataformats { template <class T> class MCTruthContainer; }}

namespace o2 { namespace DataFormat { namespace TPC {

struct ClusterNative {
  static constexpr float mScaleTimePacked = 64.; //< ~50 is needed for 0.1mm precision, but leads to float rounding artifacts around 20ms
  static constexpr float mScalePadPacked = 64.;  //< ~60 is needed for 0.1mm precision, but power of two avoids rounding
  static constexpr float mScaleSigmaTimePacked = 20.;
  static constexpr float mScaleSigmaPadPacked = 25.;
    
  uint32_t mTimeFlagsPacked;    //< Contains the time in the lower 24 bits in a packed format, contains the flags in the upper 8 bits
  uint16_t mPadPacked;          //< Contains the pad in a packed format
  uint8_t mSigmaTimePacked;    //< Sigma of the time in packed format
  uint8_t mSigmaPadPacked;     //< Sigma of the pad in packed format
  uint16_t mQMax;               //< QMax of the cluster
  uint16_t mQTot;               //< Total charge of the cluster
  
  uint8_t getFlags() const {return mTimeFlagsPacked >> 24;}
  uint32_t getTimePacked() const {return mTimeFlagsPacked & 0xFFFFFF;}
  void setTimePackedFlags(uint32_t timePacked, uint8_t flags) {mTimeFlagsPacked = (timePacked & 0xFFFFFF) | (uint32_t) flags << 24;}
  void setTimePacked(uint32_t timePacked) {mTimeFlagsPacked = (timePacked & 0xFFFFFF) | (mTimeFlagsPacked & 0xFF000000);}
  void setFlags(uint8_t flags) {mTimeFlagsPacked = (mTimeFlagsPacked & 0xFFFFFF) | ((uint32_t) flags << 24);}
  
  float getTime() const {return (mTimeFlagsPacked & 0xFFFFFF) / mScaleTimePacked;}
  void setTime(float time) {mTimeFlagsPacked = (((uint32_t) (time * mScaleTimePacked + 0.5)) & 0xFFFFFF) | (mTimeFlagsPacked & 0xFF000000);}
  void setTimeFlags(float time, uint8_t flags) {mTimeFlagsPacked = (((uint32_t) (time * mScaleTimePacked + 0.5)) & 0xFFFFFF) | ((uint32_t) flags << 24);}
  float getPad() const {return mPadPacked / mScalePadPacked;}
  void setPad(float pad) {mPadPacked = (uint16_t) (pad * mScalePadPacked + 0.5);}
  float getSigmaTime() const {return mSigmaTimePacked / mScaleSigmaTimePacked;}
  void setSigmaTime(float sigmaTime) {uint32_t tmp = sigmaTime * mScaleSigmaTimePacked + 0.5; if (tmp > 0xFF) tmp = 0xFF; mSigmaTimePacked = tmp;}
  float getSigmaPad() const {return mSigmaPadPacked / mScaleSigmaPadPacked;}
  void setSigmaPad(float sigmaPad) {uint32_t tmp = sigmaPad * mScaleSigmaPadPacked + 0.5; if (tmp > 0xFF) tmp = 0xFF; mSigmaPadPacked = tmp;}
};

struct ClusterNativeContainer
{
  static bool sortComparison(const ClusterNative& a, const ClusterNative& b) {if (a.getTimePacked() != b.getTimePacked()) return(a.getTimePacked() < b.getTimePacked()); else return(a.mPadPacked < b.mPadPacked);}

  uint8_t mSector;
  uint8_t mGlobalPadRow;
  std::vector<ClusterNative> mClusters;
};

//This is an index struct to access TPC clusters inside sectors and rows. It shall not own the data, but jus point to the data inside a buffer.
struct ClusterNativeAccessFullTPC
{
  ClusterNative* mClusters[o2::TPC::Constants::MAXSECTOR][o2::TPC::Constants::MAXGLOBALPADROW];
  unsigned int mNClusters[o2::TPC::Constants::MAXSECTOR][o2::TPC::Constants::MAXGLOBALPADROW];
  o2::dataformats::MCTruthContainer<o2::MCCompLabel>* mClustersMCTruth[o2::TPC::Constants::MAXSECTOR][o2::TPC::Constants::MAXGLOBALPADROW];
};

}}}
#endif
