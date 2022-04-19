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

#include "TPCCalibration/IDCGroup.h"
#include "CommonUtils/TreeStreamRedirector.h"
#include "TPCCalibration/IDCDrawHelper.h"
#include "TPCCalibration/IDCContainer.h"
#include "TPCBase/Mapper.h"
#include "TPCBase/CalDet.h"
#include "TFile.h"
#include <numeric>

void o2::tpc::IDCGroup::dumpToTree(const char* outname) const
{
  o2::utils::TreeStreamRedirector pcstream(outname, "RECREATE");
  pcstream.GetFile()->cd();
  for (unsigned int integrationInterval = 0; integrationInterval < getNIntegrationIntervals(); ++integrationInterval) {
    for (unsigned int irow = 0; irow < mRows; ++irow) {
      for (unsigned int ipad = 0; ipad < mPadsPerRow[irow]; ++ipad) {
        float idc = (*this)(irow, ipad, integrationInterval);
        pcstream << "idcs"
                 << "row=" << irow
                 << "pad=" << ipad
                 << "IDC=" << idc
                 << "\n";
      }
    }
  }
  pcstream.Close();
}

void o2::tpc::IDCGroup::draw(const unsigned int integrationInterval, const std::string filename) const
{
  std::function<float(const unsigned int, const unsigned int, const unsigned int, const unsigned int)> idcFunc = [this, integrationInterval](const unsigned int, const unsigned int region, const unsigned int irow, const unsigned int pad) {
    return this->getValUngrouped(irow, pad, integrationInterval);
  };

  IDCDrawHelper::IDCDraw drawFun;
  drawFun.mIDCFunc = idcFunc;
  const std::string zAxisTitle = IDCDrawHelper::getZAxisTitle(IDCType::IDC);
  IDCDrawHelper::drawSector(drawFun, mRegion, mRegion + 1, 0, zAxisTitle, filename);
}

void o2::tpc::IDCGroup::dumpToFile(const char* outFileName, const char* outName) const
{
  TFile fOut(outFileName, "UPDATE");
  fOut.WriteObject(this, outName);
  fOut.Close();
}

float o2::tpc::IDCGroup::getValUngroupedGlobal(unsigned int ugrow, unsigned int upad, unsigned int integrationInterval) const
{
  return mIDCsGrouped[getIndexUngrouped(Mapper::getLocalRowFromGlobalRow(ugrow), upad, integrationInterval)];
}
