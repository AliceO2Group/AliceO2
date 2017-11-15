// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file ClusterHardware.h
/// \brief Class of a TPC cluster as produced by the hardware cluster finder (needs a postprocessing step to convert into ClusterNative
/// \author David Rohr
#ifndef ALICEO2_DATAFORMATSTPC_CLUSTERHARDWARE_H
#define ALICEO2_DATAFORMATSTPC_CLUSTERHARDWARE_H

#include <cstdint>

namespace o2 { namespace DataFormat { namespace TPC{

struct ClusterHardware { //Temporary draft of hardware clusters. The ...Pre members are yet to be defined, and will most likely either be floats or fixed point integers.
  float mPadPre;                //< Quantity needed to compute the pad
  float mTimePre;               //< Quantity needed to compute the time
  float mSigmaPad2Pre;          //< Quantity needed to compute the sigma^2 of the pad
  float mSigmaTime2Pre;         //< Quantity needed to compute the sigma^2 of the time
  uint16_t mQMax;               //< QMax of the cluster
  uint16_t mQTot;               //< Total charge of the cluster
  uint8_t mRow;                 //< Row of the cluster (local, needs to add PadRegionInfo::getGlobalRowOffset
  uint8_t mFlags;               //< Flags of the cluster
  
  float getPad() const {return mPadPre / mQTot;}
  float getTimeLocal() const {return mTimePre / mQTot;} //Returns the local time, not taking into accound the time bin offset of the container
  float getSigmaPad2() const {return (mSigmaPad2Pre - mPadPre * mPadPre) / (mQTot * mQTot);}
  float getSigmaTime2() const {return (mSigmaTime2Pre - mTimePre * mTimePre) / (mQTot * mQTot);}
};

struct ClusterHardwareContainer { //Temporary struct to hold a set of hardware clusters, prepended by an RDH, and a short header with metadata, to be replace
                                  //The total structure is supposed to use up to 8 kb (like a readout block, thus it can hold up to 339 clusters ((8192 - 40) / 24)
    uint64_t mRDH[8];                 //< 8 * 64 bit RDH (raw data header)
    uint32_t mTimeBinOffset;          //< Time offset in timebins since beginning of the time frame
    uint16_t mNumberOfClusters;       //< Number of clusters in this 8kb structure
    uint16_t mCRU;                    //< CRU of the cluster
    ClusterHardware mClusters[0];     //< Clusters
};

}}}

#endif
