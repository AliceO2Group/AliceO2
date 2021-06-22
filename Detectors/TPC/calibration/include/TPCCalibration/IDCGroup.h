// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file IDCGroup.h
/// \brief class for storing grouped IDCs
/// \author Matthias Kleiner <mkleiner@ikf.uni-frankfurt.de>

#ifndef ALICEO2_TPC_IDCGROUP_H_
#define ALICEO2_TPC_IDCGROUP_H_

#include <vector>
#include "Rtypes.h"
#include "TPCCalibration/IDCGroupHelperRegion.h"

namespace o2::tpc
{

/// Class to hold grouped IDC values for one CRU for one TF

class IDCGroup : public IDCGroupHelperRegion
{
 public:
  /// constructor
  /// \param groupPads number of pads in pad direction which will be grouped
  /// \param groupRows number of pads in row direction which will be grouped
  /// \param groupLastRowsThreshold minimum number of pads in row direction for the last group in row direction
  /// \param groupLastPadsThreshold minimum number of pads in pad direction for the last group in pad direction
  /// \param region region of the TPC
  IDCGroup(const unsigned char groupPads = 4, const unsigned char groupRows = 4, const unsigned char groupLastRowsThreshold = 2, const unsigned char groupLastPadsThreshold = 2, const unsigned int region = 0)
    : IDCGroupHelperRegion{groupPads, groupRows, groupLastRowsThreshold, groupLastPadsThreshold, region}, mIDCsGrouped(getNIDCsPerIntegrationInterval()){};

  /// extend the size of the grouped and averaged IDC values corresponding to the number of integration intervals. This has to be called befor filling values!
  /// without using this function the object can hold only one integration interval
  /// \param nIntegrationIntervals number of ontegration intervals for which teh IDCs are stored
  void resize(const unsigned int nIntegrationIntervals) { mIDCsGrouped.resize(getNIDCsPerIntegrationInterval() * nIntegrationIntervals); }

  /// \return returns the stored value
  /// \param glrow local row of the grouped IDCs
  /// \param pad pad number of the grouped IDCs
  /// \param integrationInterval integration interval
  float operator()(unsigned int glrow, unsigned int pad, unsigned int integrationInterval) const { return mIDCsGrouped[getIndex(glrow, pad, integrationInterval)]; }

  /// \return returns the stored value
  /// \param glrow local row of the grouped IDCs
  /// \param pad pad number of the grouped IDCs
  /// \param integrationInterval integration interval
  float& operator()(unsigned int glrow, unsigned int pad, unsigned int integrationInterval) { return mIDCsGrouped[getIndex(glrow, pad, integrationInterval)]; }

  /// \return returns the stored value for local ungrouped pad row and ungrouped pad
  /// \param ulrow local row in region of the ungrouped IDCs
  /// \param upad pad number of the ungrouped IDCs
  /// \param integrationInterval integration interval
  float& setValUngrouped(unsigned int ulrow, unsigned int upad, unsigned int integrationInterval) { return mIDCsGrouped[getIndexUngrouped(ulrow, upad, integrationInterval)]; }

  /// \return returns the stored value for local ungrouped pad row and ungrouped pad
  /// \param ulrow local row in region of the ungrouped IDCs
  /// \param upad pad number of the ungrouped IDCs
  /// \param integrationInterval integration interval
  float getValUngrouped(unsigned int ulrow, unsigned int upad, unsigned int integrationInterval) const { return mIDCsGrouped[getIndexUngrouped(ulrow, upad, integrationInterval)]; }

  /// \return returns the stored value for local ungrouped pad row and ungrouped pad
  /// \param ugrow global row of the ungrouped IDCs
  /// \param upad pad number of the ungrouped IDCs
  /// \param integrationInterval integration interval
  float getValUngroupedGlobal(unsigned int ugrow, unsigned int upad, unsigned int integrationInterval) const;

  /// \return returns grouped and averaged IDC values
  const auto& getData() const { return mIDCsGrouped; }

  /// \return returns number of stored integration intervals
  unsigned int getNIntegrationIntervals() const { return mIDCsGrouped.size() / getNIDCsPerIntegrationInterval(); }

  /// dump the IDCs to a tree
  /// \param outname name of the output file
  void dumpToTree(const char* outname = "IDCGroup.root") const;

  /// dump object to disc
  /// \param outFileName name of the output file
  /// \param outName name of the object in the output file
  void dumpToFile(const char* outFileName = "IDCGroup.root", const char* outName = "IDCGroup") const;

  /// draw grouped IDCs
  /// \param integrationInterval integration interval for which the IDCs will be drawn
  /// \param filename name of the output file. If empty the canvas is drawn.
  void draw(const unsigned int integrationInterval = 0, const std::string filename = "IDCsGrouped.pdf") const;

  /// calculate and return 1D-IDCs for this CRU
  std::vector<float> get1DIDCs() const;

 private:
  std::vector<float> mIDCsGrouped{}; ///< grouped and averaged IDC values for n integration intervals for one CRU

  ClassDefNV(IDCGroup, 1)
};

} // namespace o2::tpc

#endif
