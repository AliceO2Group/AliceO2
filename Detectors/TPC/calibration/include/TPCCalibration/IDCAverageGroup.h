// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file IDCAverageGroup.h
/// \brief class for averaging and grouping of IDCs
/// \author Matthias Kleiner <mkleiner@ikf.uni-frankfurt.de>

#ifndef ALICEO2_IDCAVERAGEGROUP_H_
#define ALICEO2_IDCAVERAGEGROUP_H_

#include <vector>
#include "TPCCalibration/IDCGroup.h"
#include "TPCBase/Mapper.h"
#include "Rtypes.h"

namespace o2::utils
{
class TreeStreamRedirector;
}

namespace o2::tpc
{

/// class for averaging and grouping IDCs
/// usage:
/// 1. Define grouping parameters
/// const int region = 3;
/// IDCAverageGroup idcaverage(6, 4, 3, 2, region);
/// 2. set the ungrouped IDCs for one CRU
/// const int nIntegrationIntervals = 3;
/// std::vector<float> idcsungrouped(nIntegrationIntervals*Mapper::PADSPERREGION[region], 11.11); // vector containing IDCs for one region
/// idcaverage.setIDCs(idcsungrouped)
/// 3. perform the averaging and grouping
/// idcaverage.processIDCs();
/// 4. draw IDCs
/// idcaverage.drawUngroupedIDCs(0)
/// idcaverage.drawGroupedIDCs(0)

class IDCAverageGroup
{
 public:
  /// constructor
  /// \param groupPads number of pads in pad direction which will be grouped
  /// \param groupRows number of pads in row direction which will be grouped
  /// \param groupLastRowsThreshold minimum number of pads in row direction for the last group in row direction
  /// \param groupLastPadsThreshold minimum number of pads in pad direction for the last group in pad direction
  /// \param region region of the TPC
  IDCAverageGroup(const unsigned char groupPads = 4, const unsigned char groupRows = 4, const unsigned char groupLastRowsThreshold = 2, const unsigned char groupLastPadsThreshold = 2, const unsigned int region = 0, const Sector sector = Sector{0})
    : mIDCsGrouped{groupPads, groupRows, groupLastRowsThreshold, groupLastPadsThreshold, region}, mSector{sector} {}

  /// set the IDCs which will be averaged and grouped
  /// \param idcs vector containing the IDCs
  void setIDCs(const std::vector<float>& idcs);

  /// set the IDCs which will be averaged and grouped using move operator
  /// \param IDCs vector containing the IDCs
  void setIDCs(std::vector<float>&& idcs);

  /// \return returns number of integration intervalls stored in this object
  unsigned int getNIntegrationIntervals() const { return mIDCsUngrouped.size() / Mapper::PADSPERREGION[mIDCsGrouped.getRegion()]; }

  /// grouping and averaging of IDCs
  void processIDCs();

  /// \return returns grouped IDC object
  const auto& getIDCGroup() const { return mIDCsGrouped; }

  /// dump object to disc
  /// \param outFileName name of the output file
  /// \param outName name of the object in the output file
  void dumpToFile(const char* outFileName = "IDCAverageGroup.root", const char* outName = "IDCAverageGroup") const;

  /// draw ungrouped IDCs
  /// \param integrationInterval integration interval for which the IDCs will be drawn
  /// \param filename name of the output file. If empty the canvas is drawn.
  void drawUngroupedIDCs(const unsigned int integrationInterval = 0, const std::string filename = "IDCsUngrouped.pdf") const;

  /// draw grouped IDCs
  /// \param integrationInterval integration interval for which the IDCs will be drawn
  /// \param filename name of the output file. If empty the canvas is drawn.
  void drawGroupedIDCs(const unsigned int integrationInterval = 0, const std::string filename = "IDCsGrouped.pdf") const { mIDCsGrouped.draw(integrationInterval, filename); }

  /// \return returns the stored ungrouped IDC value for local ungrouped pad row and ungrouped pad
  /// \param ulrow ungrouped local row in region
  /// \param upad ungrouped pad in pad direction
  /// \param integrationInterval integration interval for which the IDCs will be returned
  float getUngroupedIDCValLocal(const unsigned int ulrow, const unsigned int upad, const unsigned int integrationInterval) const { return mIDCsUngrouped[getUngroupedIndex(ulrow, upad, integrationInterval)]; }

  /// \return returns the stored ungrouped IDC value for global ungrouped pad row and ungrouped pad
  /// \param ugrow ungrouped global row
  /// \param upad ungrouped pad in pad direction
  /// \param integrationInterval integration interval for which the IDCs will be returned
  float getUngroupedIDCValGlobal(const unsigned int ugrow, const unsigned int upad, const unsigned int integrationInterval) const { return mIDCsUngrouped[getUngroupedIndexGlobal(ugrow, upad, integrationInterval)]; }

  /// \return returns the stored ungrouped IDC value for local pad number
  /// \param localPadNumber local pad number for region
  /// \param integrationInterval integration interval for which the IDCs will be returned
  float getUngroupedIDCVal(const unsigned int localPadNumber, const unsigned int integrationInterval) const { return mIDCsUngrouped[localPadNumber + integrationInterval * Mapper::PADSPERREGION[mIDCsGrouped.getRegion()]]; }

  /// \return returns the stored grouped IDC value for local ungrouped pad row and ungrouped pad
  /// \param ulrow local row in region of the ungrouped IDCs
  /// \param upad pad number of the ungrouped IDCs
  /// \param integrationInterval integration interval
  float getGroupedIDCValLocal(unsigned int ulrow, unsigned int upad, unsigned int integrationInterval) const { return mIDCsGrouped.getValUngrouped(ulrow, upad, integrationInterval); }

  /// \return returns the stored grouped IDC value for local ungrouped pad row and ungrouped pad
  /// \param ugrow global ungrouped row
  /// \param upad pad number of the ungrouped IDCs
  /// \param integrationInterval integration interval
  float getGroupedIDCValGlobal(unsigned int ugrow, unsigned int upad, unsigned int integrationInterval) const { return mIDCsGrouped.getValUngroupedGlobal(ugrow, upad, integrationInterval); }

  /// get the number of threads used for some of the calculations
  static int getNThreads() { return sNThreads; }

  /// \return returns sector of which the IDCs are averaged and grouped
  Sector getSector() const { return mSector; }

  /// \return returns ungrouped IDCs
  const auto& getIDCsUngrouped() const { return mIDCsGrouped; }

  /// \return returns region
  unsigned int getRegion() const { return mIDCsGrouped.getRegion(); }

  /// set the number of threads used for some of the calculations
  static void setNThreads(const int nThreads) { sNThreads = nThreads; }

  /// for debugging: creating debug tree
  /// \param nameTree name of the output file
  void createDebugTree(const char* nameTree) const;

  /// for debugging: creating debug tree for integrated IDCs for all objects which are in the same file
  /// \param nameTree name of the output file
  /// \param filename name of the input file containing all objects
  static void createDebugTreeForAllCRUs(const char* nameTree, const char* filename);

 private:
  inline static int sNThreads{1};      ///< number of threads which are used during the calculations
  std::vector<float> mIDCsUngrouped{}; ///< integrated ungrouped IDC values per pad
  IDCGroup mIDCsGrouped{};             ///< grouped and averaged IDC values
  const Sector mSector{};              ///< sector of averaged and grouped IDCs (used for debugging)

  /// \return returns index to data from ungrouped pad and row
  /// \param ulrow ungrouped local row in region
  /// \param upad ungrouped pad in pad direction
  unsigned int getUngroupedIndex(const unsigned int ulrow, const unsigned int upad, const unsigned int integrationInterval) const { return integrationInterval * Mapper::PADSPERREGION[mIDCsGrouped.getRegion()] + Mapper::OFFSETCRULOCAL[mIDCsGrouped.getRegion()][ulrow] + upad; }

  /// \return returns index to data from ungrouped pad and row
  /// \param ugrow ungrouped global row
  /// \param upad ungrouped pad in pad direction
  unsigned int getUngroupedIndexGlobal(const unsigned int ugrow, const unsigned int upad, const unsigned int integrationInterval) const { return integrationInterval * Mapper::PADSPERREGION[mIDCsGrouped.getRegion()] + Mapper::OFFSETCRUGLOBAL[ugrow] + upad; }

  /// called from createDebugTreeForAllCRUs()
  static void createDebugTree(const IDCAverageGroup& idcavg, o2::utils::TreeStreamRedirector& pcstream);

  ClassDefNV(IDCAverageGroup, 1)
};

} // namespace o2::tpc

#endif
