// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file DigitalCurrentClusterIntegrator.h
/// \brief Decoder to convert TPC ClusterHardware to ClusterNative
/// \author David Rohr
#ifndef ALICEO2_TPC_DIGITALCURRENTCLUSTERINTEGRATOR_H_
#define ALICEO2_TPC_DIGITALCURRENTCLUSTERINTEGRATOR_H_

#include <vector>
#include <memory>

#include "DataFormatsTPC/ClusterNative.h"
#include "DataFormatsTPC/Constants.h"
#include "TPCBase/Mapper.h"

namespace o2 { namespace tpc {

//This class contains an array or all TPC pads (in selected sectors and rows), and can integrated the charges of clusters
class DigitalCurrentClusterIntegrator
{
public:
  DigitalCurrentClusterIntegrator() = default;
  ~DigitalCurrentClusterIntegrator() = default;
  
  void initRow(int sector, int row) {
    if (mIntegratedCurrents[sector][row] == nullptr) {
      int nPads = o2::tpc::Mapper::instance().getNumberOfPadsInRowSector(row);
      mIntegratedCurrents[sector][row].reset(new unsigned long long int[nPads]);
      memset(&mIntegratedCurrents[sector][row][0], 0, nPads * sizeof(mIntegratedCurrents[sector][row][0]));
    }
  }
  void integrateCluster(int sector, int row, float pad, unsigned int charge) {
    int ipad = ipad + 0.5;
    if (ipad < 0) ipad = 0;
    int maxPad = o2::tpc::Mapper::instance().getNumberOfPadsInRowSector(row);
    if (ipad >= maxPad) ipad = maxPad - 1;
    mIntegratedCurrents[sector][row][ipad] += charge;
  }
  void clear(); //Clear all currents to 0
  void reset(); //Free all allocated current buffers

private:
  std::unique_ptr<unsigned long long int[]> mIntegratedCurrents[Constants::MAXSECTOR][Constants::MAXGLOBALPADROW];
};

}}
#endif
