// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   ClusterizerDevice.h
/// \brief  Cluster reconstruction device for MID
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   24 October 2016

#ifndef O2_MID_CLUSTERIZERDEVICE_H
#define O2_MID_CLUSTERIZERDEVICE_H

#include "FairMQDevice.h"
#include "Clusterizer.h"

namespace o2
{
namespace mid
{
/// Clusterizing device for MID
class ClusterizerDevice : public FairMQDevice
{
 public:
  ClusterizerDevice();
  ~ClusterizerDevice() override = default;

  ClusterizerDevice(const ClusterizerDevice&) = delete;
  ClusterizerDevice& operator=(const ClusterizerDevice&) = delete;
  ClusterizerDevice(ClusterizerDevice&&) = delete;
  ClusterizerDevice& operator=(ClusterizerDevice&&) = delete;

 protected:
  bool handleData(FairMQMessagePtr&, int);
  void InitTask() override;

 private:
  Clusterizer mClusterizer; ///< Clustering algorithm
};
} // namespace mid
} // namespace o2

#endif /* O2_MID_CLUSTERIZERDEVICE_H */
