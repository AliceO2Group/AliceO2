// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file KrCluster.h
/// \brief Struct for Krypton and X-ray clusters
/// \author Philip Hauer <hauer@hiskp.uni-bonn.de>

#ifndef ALICEO2_TPC_KrCluster_H_
#define ALICEO2_TPC_KrCluster_H_

namespace o2
{
namespace tpc
{

struct KrCluster {
 public:
  float totCharge = 0; ///< Total charge of the cluster (ADC counts)
  float maxCharge = 0; ///< Maximum charge of the cluster (ADC counts)
  int size = 0;        ///< Size of the cluster (TPCDigits)
  float meanPad = 0;   ///< Center of gravity (Pad number)
  float meanRow = 0;   ///< Center of gravity (Row number)
  float meanTime = 0;  ///< Center of gravity (Time)
  float sigmaPad = 0;  ///< RMS of cluster in pad direction
  float sigmaRow = 0;  ///< RMS of cluster in row direction
  float sigmaTime = 0; ///< RMS of cluster in time direction
  int sector = 0;      ///< Sector number

  /// Used to set all Cluster variables to zero.
  void reset()
  {
    totCharge = 0;
    maxCharge = 0;
    size = 0;
    meanPad = 0;
    meanRow = 0;
    meanTime = 0;
    sigmaPad = 0;
    sigmaRow = 0;
    sigmaTime = 0;
    sector = 0;
  }

  ClassDefNV(KrCluster, 1);
};

} // namespace tpc
} // namespace o2

#endif
