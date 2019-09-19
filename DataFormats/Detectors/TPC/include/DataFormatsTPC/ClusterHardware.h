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
/// \brief Class of a TPC cluster as produced by the hardware cluster finder (needs a postprocessing step to convert
/// into ClusterNative
/// \author David Rohr
#ifndef ALICEO2_DATAFORMATSTPC_CLUSTERHARDWARE_H
#define ALICEO2_DATAFORMATSTPC_CLUSTERHARDWARE_H

#include <cstdint>

namespace o2
{
namespace tpc
{
struct ClusterHardware { // Draft of hardware clusters in bit-packed format.
  // padPre: word 0, bits 0 - 19 (20 bit, two-complement, 4 fixed-point-bits) - Quantity needed to compute the pad.
  // padPeak: word 0, bits 20-27 (8 bit, unsigned integer) - Offset of padPre.
  // timePre: word 1, bits 0-19 (20 bit, two-complement, 4 fixed-point-bits) - Quantity needed to compute the time.
  // timePeak: word 1, bits 20-28 (9 bit, unsigned integer) - Offset of timePre.
  // sigmaPadPre: word 2, bits 0-19 (20 bit, unsigned, 4 fixed-point-bits) - Quantity needed to compute the sigma^2 of
  // the pad.
  // sigmaTimePre: word 3, bits 0-19 (20 bit, unsigned, 4 fixed-point-bits) - Quantity needed to compute the
  // sigma^2 of the time.
  // qMax: word 2, bits 20-30 (11 bit, 1 fixed-point-bit) - QMax of the cluster.
  // qTot: word 4, bits 0-18 (19 bit, 4 fixed-point-bits) - Total charge of the cluster.
  // row: word 3, bits 20-24 (5 bit, unsigned integer) - Row of the cluster (local, needs to add
  // PadRegionInfo::getGlobalRowOffset)
  // flags: word 4, bits 19-26 (8 bit) up to 8 bit flag field.
  // remaining bits: reserved, must be 0!, could be used for additional bits later.
  uint32_t word0; //< word0 of binary hardware cluster
  uint32_t word1; //< word1 of binary hardware cluster
  uint32_t word2; //< word2 of binary hardware cluster
  uint32_t word3; //< word3 of binary hardware cluster
  uint32_t word4; //< word4 of binary hardware cluster

  float getQTotFloat() const
  {
    unsigned int qTotInt = word4 & 0x7FFFF;
    return (qTotInt / 16.f);
  }

  int getQTot() const { return (getQTotFloat() + 0.5); }

  int getQMax() const
  {
    int qmaxint = (word2 & 0x7FF00000) >> 20;
    return (qmaxint / 2.0 + 0.5);
  }

  int getRow() const { return ((word3 & 0x1F00000) >> 20); }

  int getFlags() const { return ((word4 & 0x7F80000) >> 19); }

  float getPadPre() const
  {
    int padPreInt = word0 & 0xFFFFF;
    if (padPreInt & 0x80000) {
      padPreInt |= 0xFFF00000;
    }
    return (padPreInt / 16.f);
  }

  /// @return Returns the pad position of the cluster.
  /// note that the pad position is defined on the left side of the pad.
  /// the pad position from clusters are calculated in HwClusterer::hwClusterProcessor()
  /// around the centre of gravity around the left side of the pad.
  /// i.e. the center of the first pad has pad position zero.
  /// To get the corresponding local Y coordinate of the cluster:
  /// Y = (pad_position - 0.5 * (n_pads - 1)) * padWidth
  /// example:
  /// the pad position is for example 12.4 (pad_position = 12.4).
  /// there are 66 pads in the first pad row (n_pads = 66).
  /// the pad width for pads in the first padrow is 4.16mm (padWidth = 4.16mm).
  /// Y = (12.4 - 0.5 * (66 - 1)) * 4.16mm = -83.616mm
  float getPad() const
  {
    int padPeak = (word0 & 0xFF00000) >> 20;
    return (getPadPre() / getQTotFloat() + padPeak);
  }

  float getTimeLocalPre() const
  {
    int timePreInt = word1 & 0xFFFFF;
    if (timePreInt & 0x80000) {
      timePreInt |= 0xFFF00000;
    }
    return (timePreInt / 16.f);
  }

  int getTimePeak() const
  {
    return (word1 & 0x1FF00000) >> 20;
  }

  float getTimeLocal() const // Returns the local time, not taking into account the time bin offset of the container
  {
    int timePeak = getTimePeak();
    return (getTimeLocalPre() / getQTotFloat() + timePeak);
  }

  float getSigmaPad2() const
  {
    int sigmaPad2PreInt = word2 & 0xFFFFF;
    float sigmaPad2Pre = sigmaPad2PreInt / 16.f;
    return sigmaPad2Pre / getQTotFloat() - (getPadPre() * getPadPre()) / (getQTotFloat() * getQTotFloat());
  }

  float getSigmaTime2() const
  {
    int sigmaTime2PreInt = word3 & 0xFFFFF;
    float sigmaTime2Pre = sigmaTime2PreInt / 16.f;
    return sigmaTime2Pre / getQTotFloat() - (getTimeLocalPre() * getTimeLocalPre()) / (getQTotFloat() * getQTotFloat());
  }

  void setCluster(float pad, float time, float sigmaPad2, float sigmaTime2, float qMax, float qTot, int row, int flags)
  {
    int max = qMax * 2.f;
    int tot = qTot * 16.f;
    qTot = tot / 16.f;
    int padPeak = pad + 0.5;
    pad -= padPeak;
    int timePeak = time + 0.5;
    time -= timePeak;
    pad *= qTot;
    time *= qTot;
    int p = pad * 16.f;
    int t = time * 16.f;
    pad = p / 16.f;
    time = t / 16.f;
    sigmaPad2 = (sigmaPad2 + pad / qTot * pad / qTot) * qTot;
    sigmaTime2 = (sigmaTime2 + time / qTot * time / qTot) * qTot;
    int sp = sigmaPad2 * 16.f;
    int st = sigmaTime2 * 16.f;
    word0 = (p & 0xFFFFF) | ((padPeak & 0xFF) << 20);
    word1 = (t & 0xFFFFF) | ((timePeak & 0x1FF) << 20);
    word2 = (sp & 0xFFFFF) | ((max & 0x7FF) << 20);
    word3 = (st & 0xFFFFF) | ((row & 0x1F) << 20);
    word4 = (tot & 0x7FFFF) | ((flags & 0xFF) << 19);
  }

  void setCluster(int padPeak, int timePeak, int pPre, int tPre, int sigmaPad2Pre, int sigmaTime2Pre, int qMax, int qTot, int row, int flags)
  {
    word0 = (pPre & 0xFFFFF) | ((padPeak & 0xFF) << 20);
    word1 = (tPre & 0xFFFFF) | ((timePeak & 0x1FF) << 20);
    word2 = (sigmaPad2Pre & 0xFFFFF) | ((qMax & 0x7FF) << 20);
    word3 = (sigmaTime2Pre & 0xFFFFF) | ((row & 0x1F) << 20);
    word4 = (qTot & 0x7FFFF) | ((flags & 0xFF) << 19);
  }
};

struct ClusterHardwareContainer { // Temporary struct to hold a set of hardware clusters, prepended by an RDH, and a
                                  // short header with metadata, to be replace
  // The total structure is supposed to use up to 8 kb (like a readout block, thus it can hold up to 406 clusters ((8192
  // - 40) / 20)
  uint64_t mRDH[8];            //< 8 * 64 bit RDH (raw data header)
  uint32_t timeBinOffset;      //< Time offset in timebins since beginning of the time frame
  uint16_t numberOfClusters;   //< Number of clusters in this 8kb structure
  uint16_t CRU;                //< CRU of the cluster
  ClusterHardware clusters[0]; //< Clusters
};
} // namespace tpc
} // namespace o2

#endif
