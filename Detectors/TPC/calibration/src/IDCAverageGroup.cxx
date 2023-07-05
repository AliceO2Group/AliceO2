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

#include "TPCCalibration/IDCAverageGroup.h"
#include "TPCCalibration/IDCAverageGroupBase.h"
#include "TPCCalibration/IDCAverageGroupHelper.h"
#include "TPCCalibration/IDCDrawHelper.h"
#include "CommonUtils/TreeStreamRedirector.h"
#include "TPCBase/Mapper.h"
#include "CommonConstants/MathConstants.h"

// root includes
#include "TFile.h"
#include "TKey.h"
#include "TPCBase/Painter.h"
#include "TH2Poly.h"
#include "TCanvas.h"
#include "TLatex.h"
#include "TStyle.h"
#include "Framework/Logger.h"

#if (defined(WITH_OPENMP) || defined(_OPENMP)) && !defined(__CLING__)
#include <omp.h>
#else
static inline int omp_get_thread_num() { return 0; }
#endif

template <>
void o2::tpc::IDCAverageGroup<o2::tpc::IDCAverageGroupCRU>::init()
{
  unsigned int maxValues = 0;
  for (unsigned int i = 0; i < Mapper::NREGIONS; ++i) {
    const unsigned int maxGroup = (this->mIDCsGrouped.getGroupRows() + this->mIDCsGrouped.getGroupLastRowsThreshold()) * (this->mIDCsGrouped.getGroupPads() + this->mIDCsGrouped.getGroupLastPadsThreshold() + Mapper::ADDITIONALPADSPERROW[i].back());
    if (maxGroup > maxValues) {
      maxValues = maxGroup;
    }
  }

  for (auto& rob : this->mRobustAverage) {
    rob.reserve(maxValues);
  }

  // init weights
  const float sigmaEdge = 1.f;
  this->mWeightsPad.reserve(mOverlapPads);
  for (int i = 0; i < mOverlapPads; ++i) {
    const float groupPadsHalf = this->mIDCsGrouped.getGroupPads() / 2.f;
    const float sigmaPad = groupPadsHalf / sigmaEdge; // assume 3-sigma at the edge of the last pad
    this->mWeightsPad.emplace_back(normal_dist(groupPadsHalf + i, sigmaPad));
  }

  this->mWeightsRow.reserve(mOverlapRows);
  for (int i = 0; i < mOverlapRows; ++i) {
    const float groupRowsHalf = this->mIDCsGrouped.getGroupRows() / 2.f;
    const float sigmaRow = groupRowsHalf / sigmaEdge; // assume 3-sigma at the edge of the last pad
    this->mWeightsRow.emplace_back(normal_dist(groupRowsHalf + i, sigmaRow));
  }
}

template <>
void o2::tpc::IDCAverageGroup<o2::tpc::IDCAverageGroupTPC>::init()
{
  unsigned int maxValues = 0;
  for (unsigned int i = 0; i < Mapper::NREGIONS; ++i) {
    const unsigned int maxGroup = (this->mIDCGroupHelperSector.getGroupingParameter().getGroupRows(i) + this->mIDCGroupHelperSector.getGroupingParameter().getGroupLastRowsThreshold(i)) * (this->mIDCGroupHelperSector.getGroupingParameter().getGroupPads(i) + this->mIDCGroupHelperSector.getGroupingParameter().getGroupLastPadsThreshold(i) + Mapper::ADDITIONALPADSPERROW[i].back());
    if (maxGroup > maxValues) {
      maxValues = maxGroup;
    }
  }

  for (auto& rob : this->mRobustAverage) {
    rob.reserve(maxValues);
  }

  // init weights
  for (unsigned int region = 0; region < Mapper::NREGIONS; ++region) {
    const float sigmaEdge = 1.f; /// TODO make configurable
    this->mWeightsPad[region].reserve(mOverlapPads);
    for (unsigned int i = 0; i < mOverlapPads; ++i) {
      const float groupPadsHalf = this->mIDCGroupHelperSector.getGroupingParameter().getGroupPads(i) / 2.f;
      const float sigmaPad = groupPadsHalf / sigmaEdge; // assume 3-sigma at the edge of the last pad
      this->mWeightsPad[i].emplace_back(normal_dist(groupPadsHalf + i, sigmaPad));
    }

    this->mWeightsRow[region].reserve(mOverlapRows);
    for (unsigned int i = 0; i < mOverlapRows; ++i) {
      const float groupRowsHalf = this->mIDCGroupHelperSector.getGroupingParameter().getGroupRows(i) / 2.f;
      const float sigmaRow = groupRowsHalf / sigmaEdge; // assume 3-sigma at the edge of the last pad
      this->mWeightsRow[i].emplace_back(normal_dist(groupRowsHalf + i, sigmaRow));
    }
  }
}

template <class Type>
float o2::tpc::IDCAverageGroup<Type>::normal_dist(const float x, const float sigma)
{
  const float fac = x / sigma;
  return std::exp(-fac * fac / 2);
}

template <>
void o2::tpc::IDCAverageGroup<o2::tpc::IDCAverageGroupCRU>::processIDCs(const CalDet<PadFlags>* padStatusFlags)
{
  std::vector<IDCAverageGroupHelper<IDCAverageGroupCRU>> idcStruct(sNThreads, IDCAverageGroupHelper<IDCAverageGroupCRU>{this->mIDCsGrouped, this->mWeightsPad, this->mWeightsRow, this->mIDCsUngrouped, this->mRobustAverage, this->getCRU()});
#pragma omp parallel for num_threads(sNThreads)
  for (unsigned int integrationInterval = 0; integrationInterval < this->getNIntegrationIntervals(); ++integrationInterval) {
    const unsigned int threadNum = omp_get_thread_num();
    idcStruct[threadNum].set(threadNum, integrationInterval);
    loopOverGroups(idcStruct[threadNum], padStatusFlags);
  }
}

template <>
void o2::tpc::IDCAverageGroup<o2::tpc::IDCAverageGroupTPC>::processIDCs(const CalDet<PadFlags>* padStatusFlags)
{
  std::vector<IDCAverageGroupHelper<IDCAverageGroupTPC>> idcStruct(sNThreads, IDCAverageGroupHelper<IDCAverageGroupTPC>{this->mIDCsGrouped, this->mWeightsPad, this->mWeightsRow, this->mIDCsUngrouped, this->mRobustAverage, this->mIDCGroupHelperSector});
  for (int thread = 0; thread < sNThreads; ++thread) {
    idcStruct[thread].setThreadNum(thread);
  }

  const int cruStart = (mSide == Side::A) ? 0 : CRU::MaxCRU / 2;
  const int cruEnd = (mSide == Side::A) ? CRU::MaxCRU / 2 : CRU::MaxCRU;

#pragma omp parallel for num_threads(sNThreads)
  for (unsigned int i = cruStart; i < cruEnd; ++i) {
    const unsigned int threadNum = omp_get_thread_num();
    const CRU cru(i);
    idcStruct[threadNum].setCRU(cru);
    for (unsigned int integrationInterval = 0; integrationInterval < this->getNIntegrationIntervals(); ++integrationInterval) {
      idcStruct[threadNum].setIntegrationInterval(integrationInterval);
      loopOverGroups(idcStruct[threadNum], padStatusFlags);
    }
  }
}

template <class Type>
void o2::tpc::IDCAverageGroup<Type>::drawGrouping(const std::string filename)
{
  const auto& mapper = Mapper::instance();
  const float xMin = 83.65f;
  const float xMax = 247.7f;
  const float yMin = -51;
  const float yMax = 49;
  TH2Poly* poly = o2::tpc::painter::makeSectorHist("hSector", "Sector;#it{x} (cm);#it{y} (cm)", xMin, xMax, yMin, yMax);
  poly->GetXaxis()->SetTickLength(0.01f);

  poly->SetContour(255);
  gStyle->SetNumberContours(255);

  TCanvas can("can", "can", 2000, 1400);
  can.SetRightMargin(0.01f);
  can.SetLeftMargin(0.06f);
  can.SetTopMargin(0.04f);
  can.cd();
  poly->SetTitle(0);
  poly->GetYaxis()->SetTickSize(0.002f);
  poly->GetYaxis()->SetTitleOffset(0.7f);
  poly->SetStats(0);
  poly->Draw("col");

  int sumIDCs = 0;
  for (unsigned int i = 0; i < Mapper::NREGIONS; ++i) {
    if constexpr (std::is_same_v<Type, IDCAverageGroupCRU>) {
      IDCAverageGroupHelper<IDCAverageGroupDraw> idcStruct(this->mIDCsGrouped.getGroupPads(), this->mIDCsGrouped.getGroupRows(), this->mIDCsGrouped.getGroupLastRowsThreshold(), this->mIDCsGrouped.getGroupLastPadsThreshold(), this->mIDCsGrouped.getGroupPadsSectorEdges(), i, Mapper::PADSPERREGION[i], mapper.getPadRegionInfo(i), *poly);
      loopOverGroups(idcStruct);
      const int nidcs = this->mIDCsGrouped.getNIDCsPerIntegrationInterval();
      sumIDCs += nidcs;
      drawGroupingInformations(i, this->mIDCsGrouped.getGroupPads(), this->mIDCsGrouped.getGroupRows(), this->mIDCsGrouped.getGroupLastRowsThreshold(), this->mIDCsGrouped.getGroupLastPadsThreshold(), mOverlapRows, mOverlapPads, nidcs, this->mIDCsGrouped.getGroupPadsSectorEdges());
    } else {
      IDCAverageGroupHelper<IDCAverageGroupDraw> idcStruct(this->mIDCGroupHelperSector.getGroupingParameter().getGroupPads(i), this->mIDCGroupHelperSector.getGroupingParameter().getGroupRows(i), this->mIDCGroupHelperSector.getGroupingParameter().getGroupLastRowsThreshold(i), this->mIDCGroupHelperSector.getGroupingParameter().getGroupLastPadsThreshold(i), this->mIDCGroupHelperSector.getGroupingParameter().getGroupPadsSectorEdges(), i, Mapper::PADSPERREGION[i], mapper.getPadRegionInfo(i), *poly);
      loopOverGroups(idcStruct);
      const int nidcs = this->mIDCGroupHelperSector.getNIDCs(i);
      sumIDCs += nidcs;
      drawGroupingInformations(i, this->mIDCGroupHelperSector.getGroupingParameter().getGroupPads(i), this->mIDCGroupHelperSector.getGroupingParameter().getGroupRows(i), this->mIDCGroupHelperSector.getGroupingParameter().getGroupLastRowsThreshold(i), this->mIDCGroupHelperSector.getGroupingParameter().getGroupLastPadsThreshold(i), mOverlapRows, mOverlapPads, nidcs, this->mIDCGroupHelperSector.getGroupingParameter().getGroupPadsSectorEdges());
    }
  }

  painter::drawSectorLocalPadNumberPoly(kBlack);
  painter::drawSectorInformationPoly(kRed, kRed);

  TLatex lat;
  lat.SetTextColor(kBlack);
  lat.SetTextSize(0.02f);
  lat.SetTextAlign(12);
  const float posYInf = -44.5f;
  const float offsx = 1;
  lat.DrawLatex(xMin + offsx, posYInf, "nPads | nRows | nLastPads | nLastRows");

  lat.SetTextColor(kGreen + 2);
  lat.DrawLatex(mapper.getPadRegionInfo(4).getRadiusFirstRow(), posYInf, "nPadsSectorEdge | overlapRows | overlapPads");

  lat.SetTextColor(kBlack);
  lat.DrawLatex(xMin + offsx, 47.2f, "IDCs");

  // ToDo add compression factor from root
  const int dataRate = sumIDCs * Mapper::NSECTORS * sizeof(short) * 1000 / (1024 * 1024); // approximate data rate for IDCDelta: 'number of values per sector' * 'number of sectors' * 'sizeof datatype' * '1000 objects per second' / '1000000: to mega byte'
  lat.DrawLatex(xMin + offsx, 50.5f, fmt::format("IDCDelta data rate (short): {} MB/s    IDCs per sector: {}", dataRate, sumIDCs).data());

  if constexpr (std::is_same_v<Type, IDCAverageGroupCRU>) {
    const std::string outName = filename.empty() ? fmt::format("grouping_rows-{}_pads-{}_rowThr-{}_padThr-{}_ovRows-{}_ovPads-{}_edge-{}.pdf", this->mIDCsGrouped.getGroupPads(), this->mIDCsGrouped.getGroupRows(), this->mIDCsGrouped.getGroupLastRowsThreshold(), this->mIDCsGrouped.getGroupLastPadsThreshold(), mOverlapRows, mOverlapPads, this->mIDCsGrouped.getGroupPadsSectorEdges()) : filename;
    can.SaveAs(outName.data());
  } else {
    std::string sgrRows = {"_"};
    std::string sgrPads = {"_"};
    std::string sgrRowsTh = {"_"};
    std::string sgrPadsTh = {"_"};
    if (filename.empty()) {
      for (unsigned int i = 0; i < Mapper::NREGIONS; ++i) {
        const int grRows = this->mIDCGroupHelperSector.getGroupingParameter().getGroupRows(i);
        sgrRows += fmt::format("{}_", grRows);
        const int grPads = this->mIDCGroupHelperSector.getGroupingParameter().getGroupPads(i);
        sgrPads += fmt::format("{}_", grPads);
        const int grRowsTh = this->mIDCGroupHelperSector.getGroupingParameter().getGroupLastRowsThreshold(i);
        sgrRowsTh += fmt::format("{}_", grRowsTh);
        const int grPadsTh = this->mIDCGroupHelperSector.getGroupingParameter().getGroupLastPadsThreshold(i);
        sgrPadsTh += fmt::format("{}_", grPadsTh);
      }
    }
    const std::string outName = filename.empty() ? fmt::format("grouping_rows{}pads{}rowThr{}padThr{}ovRows-{}_ovPads-{}_edge-{}.pdf", sgrRows, sgrPads, sgrRowsTh, sgrPadsTh, mOverlapRows, mOverlapPads, this->mIDCGroupHelperSector.getGroupingParameter().getGroupPadsSectorEdges()) : filename;
    can.SaveAs(outName.data());
  }
  delete poly;
}

template <class Type>
void o2::tpc::IDCAverageGroup<Type>::drawGroupingInformations(const int region, const int grPads, const int grRows, const int groupLastRowsThreshold, const int groupLastPadsThreshold, const int overlapRows, const int overlapPads, const int nIDCs, const int groupPadsSectorEdges) const
{
  const o2::tpc::Mapper& mapper = Mapper::instance();

  TLatex lat;
  lat.SetTextColor(kBlack);
  lat.SetTextSize(0.02f);
  lat.SetTextAlign(12);

  const float radius = mapper.getPadRegionInfo(region).getRadiusFirstRow();

  // draw grouping parameter
  lat.DrawLatex(radius, -47, fmt::format("{} | {} | {} | {}", grPads, grRows, groupLastRowsThreshold, groupLastPadsThreshold).data());

  lat.SetTextColor(kGreen + 2);
  lat.DrawLatex(radius, -49, fmt::format("{} | {} | {}", groupPadsSectorEdges, overlapRows, overlapPads).data());

  // draw number of grouped pads
  lat.SetTextColor(kBlack);
  const float radiusNext = region == 9 ? 247.f : mapper.getPadRegionInfo(region + 1).getRadiusFirstRow();
  lat.DrawLatex((radius + radiusNext) / 2, 47.2f, Form("%i", nIDCs));
}

template <class Type>
template <class LoopType>
void o2::tpc::IDCAverageGroup<Type>::loopOverGroups(IDCAverageGroupHelper<LoopType>& idcStruct, const CalDet<PadFlags>* padStatusFlags)
{
  const unsigned int region = idcStruct.getRegion();
  const int groupRows = idcStruct.getGroupRows();
  const int groupPads = idcStruct.getGroupPads();
  const int lastRow = idcStruct.getLastRow();
  const int groupPadsSectorEdges = idcStruct.getTotalGroupPadsSectorEdges();
  unsigned int rowGrouped = 0;
  const bool applyWeights = mOverlapRows && mOverlapPads;

  if (groupPadsSectorEdges) {
    const auto groupingType = idcStruct.getEdgePadGroupingType();
    const bool groupRowsEdge = groupingType == EdgePadGroupingMethod::ROWS;
    const int groupedPads = idcStruct.getGroupedPadsSectorEdges();
    const int endrow = groupRowsEdge ? lastRow + 1 : Mapper::ROWSPERREGION[region];
    const int stepRow = groupRowsEdge ? groupRows : 1;
    for (int ulrow = 0; ulrow < endrow; ulrow += stepRow) {
      const bool bNotLastrow = ulrow != lastRow;

      if constexpr (std::is_same_v<LoopType, IDCAverageGroupDraw>) {
        idcStruct.mCol = ulrow / stepRow;
      }

      for (int iYLocalSide = 0; iYLocalSide < 2; ++iYLocalSide) {
        int pad = 0;
        for (int padGroup = 0; padGroup < groupedPads; ++padGroup) {
          const int nPadsPerGroup = idcStruct.getPadsInGroupSectorEdges(padGroup);
          for (int padInGroup = 0; padInGroup < nPadsPerGroup; ++padInGroup) {
            const int endRow = (groupRowsEdge && (ulrow + stepRow >= Mapper::ROWSPERREGION[region] || !bNotLastrow)) ? (Mapper::ROWSPERREGION[region] - ulrow) : stepRow; // last row in this group
            for (int iRowMerge = 0; iRowMerge < endRow; ++iRowMerge) {
              const int irow = ulrow + iRowMerge;
              const int ungroupedPad = !iYLocalSide ? pad : Mapper::PADSPERROW[region][irow] - pad - 1;
              const int padInRegion = Mapper::OFFSETCRULOCAL[region][irow] + ungroupedPad;

              if constexpr (std::is_same_v<LoopType, IDCAverageGroupCRU> || std::is_same_v<LoopType, IDCAverageGroupTPC>) {
                if (padStatusFlags) {
                  const auto flag = padStatusFlags->getCalArray(idcStruct.getCRU()).getValue(padInRegion);
                  if ((flag & PadFlags::flagSkip) == PadFlags::flagSkip) {
                    continue;
                  }
                }
                idcStruct.addValue(padInRegion, 1);
              } else {
                const GlobalPadNumber padNum = o2::tpc::Mapper::getGlobalPadNumber(irow, ungroupedPad, region);
                drawLatex(idcStruct, padNum, padInRegion, true);
              }
            }
            ++pad;
          }
          if constexpr (std::is_same_v<LoopType, IDCAverageGroupCRU> || std::is_same_v<LoopType, IDCAverageGroupTPC>) {
            const int ungroupedPad = !iYLocalSide ? pad - 1 : Mapper::PADSPERROW[region][ulrow] - pad;
            idcStruct.setSectorEdgeIDC(ulrow, ungroupedPad);
            idcStruct.clearRobustAverage();
          } else {
            ++idcStruct.mGroupCounter;
            ++idcStruct.mCol;
          }
        }
      }
    }
  }

  // loop over ungrouped row
  for (int iRow = 0; iRow <= lastRow; iRow += groupRows) {
    const bool bNotLastrow = iRow != lastRow;

    // the sectors is divide in to two parts around ylocal=0 to get the same simmetric grouping around ylocal=0
    for (int iYLocalSide = 0; iYLocalSide < 2; ++iYLocalSide) {
      if constexpr (std::is_same_v<LoopType, IDCAverageGroupDraw>) {
        idcStruct.mCol = region + iRow / groupRows + iYLocalSide;
      }
      unsigned int padGrouped = iYLocalSide ? idcStruct.getPadsPerRow(rowGrouped) / 2 : idcStruct.getPadsPerRow(rowGrouped) / 2 - 1; // grouped pad in pad direction
      const int nPadsStart = Mapper::PADSPERROW[region][iRow] / 2;                                                                   // first ungrouped pad in pad direction
      const int nPadsEnd = idcStruct.getLastPad(iRow) + nPadsStart;                                                                  // last grouped pad in pad direction

      // loop over ungrouped pads
      for (int iPad = nPadsStart; iPad <= nPadsEnd; iPad += groupPads) {
        if constexpr (std::is_same_v<LoopType, IDCAverageGroupCRU> || std::is_same_v<LoopType, IDCAverageGroupTPC>) {
          idcStruct.clearRobustAverage();
        }

        const int startRow = ((iRow - mOverlapRows) < 0) ? 0 : -mOverlapRows;                                                                                                          // first row in this group
        const int endRow = ((iRow + groupRows + mOverlapRows) >= Mapper::ROWSPERREGION[region] || !bNotLastrow) ? (Mapper::ROWSPERREGION[region] - iRow) : (mOverlapRows + groupRows); // last row in this group
        for (int iRowMerge = startRow; iRowMerge < endRow; ++iRowMerge) {
          const bool bOverlapRowRight = iRowMerge >= groupRows;
          const unsigned int ungroupedRow = iRow + iRowMerge;
          const int offsPad = static_cast<int>(Mapper::ADDITIONALPADSPERROW[region][ungroupedRow]) - static_cast<int>(Mapper::ADDITIONALPADSPERROW[region][iRow]); // offset due to additional pads in pad direction in the current row compared to the first row in the group

          const bool lastPad = iPad == nPadsEnd;
          const int padEnd = lastPad ? (static_cast<int>(Mapper::PADSPERROW[region][ungroupedRow]) - iPad - groupPadsSectorEdges) : (groupPads + offsPad + mOverlapPads); // last ungrouped pad in pad direction
          const int padStart = offsPad - mOverlapPads;                                                                                                                    // first ungrouped pad in pad direction

          for (int ipadMerge = padStart; ipadMerge < padEnd; ++ipadMerge) {
            const unsigned int ungroupedPad = iYLocalSide ? (iPad + ipadMerge) : Mapper::PADSPERROW[region][ungroupedRow] - (iPad + ipadMerge) - 1;
            const unsigned int padInRegion = Mapper::OFFSETCRULOCAL[region][ungroupedRow] + ungroupedPad;

            // averaging and grouping
            if constexpr (std::is_same_v<LoopType, IDCAverageGroupCRU> || std::is_same_v<LoopType, IDCAverageGroupTPC>) {
              // check status flag
              if (padStatusFlags) {
                const auto flag = padStatusFlags->getCalArray(idcStruct.getCRU()).getValue(padInRegion);
                if ((flag & PadFlags::flagSkip) == PadFlags::flagSkip) {
                  continue;
                }
              }

              // set weight for outer pads which are not in the main group
              if (applyWeights) {
                float weight = 1;
                if (iRowMerge < 0) {
                  // everything on the left border
                  const int relPosRow = std::abs(iRowMerge);
                  if (ipadMerge < offsPad) {
                    const int relPosPad = std::abs(ipadMerge - offsPad);
                    weight = idcStruct.getWeight(relPosRow, relPosPad);
                  } else if (!lastPad && ipadMerge >= (groupPads + offsPad)) {
                    const int relPosPad = std::abs(1 + ipadMerge - (groupPads + offsPad));
                    weight = idcStruct.getWeight(relPosRow, relPosPad);
                  } else {
                    weight = idcStruct.getWeightRow(relPosRow);
                  }
                } else if (bNotLastrow && bOverlapRowRight) {
                  const int relPosRow = std::abs(1 + iRowMerge - (groupRows));
                  if (ipadMerge < offsPad) {
                    const int relPosPad = std::abs(ipadMerge - offsPad);
                    weight = idcStruct.getWeight(relPosRow, relPosPad);
                  } else if (!lastPad && ipadMerge >= (groupPads + offsPad)) {
                    const int relPosPad = std::abs(1 + ipadMerge - (groupPads + offsPad));
                    weight = idcStruct.getWeight(relPosRow, relPosPad);
                  } else {
                    weight = idcStruct.getWeightRow(relPosRow);
                  }
                } else if (ipadMerge < offsPad) {
                  // bottom
                  const int relPadPos = std::abs(ipadMerge - offsPad);
                  weight = idcStruct.getWeightPad(relPadPos);
                } else if (!lastPad && ipadMerge >= (groupPads + offsPad)) {
                  const int relPadPos = std::abs(1 + ipadMerge - (groupPads + offsPad));
                  weight = idcStruct.getWeightPad(relPadPos);
                } else {
                }
                idcStruct.addValue(padInRegion, weight);
              } else {
                idcStruct.addValue(padInRegion);
              }

            } else {
              // drawing the grouping
              const GlobalPadNumber padNum = o2::tpc::Mapper::getGlobalPadNumber(ungroupedRow, ungroupedPad, region);
              const bool fillNotPoly = iRowMerge < 0 || (bNotLastrow && bOverlapRowRight) || (ipadMerge < offsPad) || (!lastPad && ipadMerge >= (groupPads + offsPad));
              drawLatex(idcStruct, padNum, padInRegion, !fillNotPoly, idcStruct.mColors.size());
            }
          }
        }
        if constexpr (std::is_same_v<LoopType, IDCAverageGroupCRU> || std::is_same_v<LoopType, IDCAverageGroupTPC>) {
          idcStruct.setGroupedIDC(rowGrouped, padGrouped, applyWeights);
        } else {
          ++idcStruct.mGroupCounter;
          ++idcStruct.mCol;
        }
        iYLocalSide ? ++padGrouped : --padGrouped;
      }
    }
    ++rowGrouped;
  }
}

template <class Type>
void o2::tpc::IDCAverageGroup<Type>::dumpToFile(const char* outFileName, const char* outName) const
{
  TFile fOut(outFileName, "RECREATE");
  fOut.WriteObject(this, outName);
  fOut.Close();
}

template <class Type>
bool o2::tpc::IDCAverageGroup<Type>::setFromFile(const char* fileName, const char* name)
{
  TFile inpf(fileName, "READ");
  using Temp = IDCAverageGroup<Type>;
  Temp* idcAverageGroupTmp{nullptr};
  idcAverageGroupTmp = reinterpret_cast<Temp*>(inpf.GetObjectChecked(name, Temp::Class()));

  if (!idcAverageGroupTmp) {
    LOGP(error, "Failed to load {} from {}", name, inpf.GetName());
    return false;
  }
  if constexpr (std::is_same_v<Type, IDCAverageGroupCRU>) {
    this->setIDCs(idcAverageGroupTmp->getIDCsUngrouped());
  } else {
    this->setIDCs(idcAverageGroupTmp->getIDCsUngrouped(), idcAverageGroupTmp->getSide());
  }

  delete idcAverageGroupTmp;
  return true;
}

template <>
void o2::tpc::IDCAverageGroup<o2::tpc::IDCAverageGroupCRU>::createDebugTree(const char* nameFile)
{
  o2::utils::TreeStreamRedirector pcstream(nameFile, "RECREATE");
  pcstream.GetFile()->cd();
  IDCAverageGroupHelper<IDCAverageGroupCRU> idcStruct(this->mIDCsGrouped, this->mWeightsPad, this->mWeightsRow, this->mIDCsUngrouped, this->mRobustAverage, this->getCRU());
  for (unsigned int integrationInterval = 0; integrationInterval < this->getNIntegrationIntervals(); ++integrationInterval) {
    idcStruct.set(0, integrationInterval);
    createDebugTree(idcStruct, pcstream);
  }
  pcstream.Close();
}

template <>
void o2::tpc::IDCAverageGroup<o2::tpc::IDCAverageGroupTPC>::createDebugTree(const char* nameFile)
{
  IDCAverageGroupHelper<IDCAverageGroupTPC> idcStruct(this->mIDCsGrouped, this->mWeightsPad, this->mWeightsRow, this->mIDCsUngrouped, this->mRobustAverage, this->mIDCGroupHelperSector);
  o2::utils::TreeStreamRedirector pcstream(nameFile, "RECREATE");
  pcstream.GetFile()->cd();
  for (unsigned int iCRU = 0; iCRU < CRU::MaxCRU; ++iCRU) {
    const CRU cru(iCRU);
    idcStruct.setCRU(cru);
    for (unsigned int integrationInterval = 0; integrationInterval < this->getNIntegrationIntervals(); ++integrationInterval) {
      idcStruct.setIntegrationInterval(integrationInterval);
      createDebugTree(idcStruct, pcstream);
    }
  }
  pcstream.Close();
}

template <>
void o2::tpc::IDCAverageGroup<o2::tpc::IDCAverageGroupCRU>::createDebugTreeForAllCRUs(const char* nameFile, const char* filename)
{
  o2::utils::TreeStreamRedirector pcstream(nameFile, "RECREATE");
  pcstream.GetFile()->cd();
  TFile fInp(filename, "READ");

  for (TObject* keyAsObj : *fInp.GetListOfKeys()) {
    const auto key = dynamic_cast<TKey*>(keyAsObj);
    LOGP(info, "Key name: {} Type: {}", key->GetName(), key->GetClassName());

    if (std::strcmp(o2::tpc::IDCAverageGroup<IDCAverageGroupCRU>::Class()->GetName(), key->GetClassName()) != 0) {
      LOGP(info, "skipping object. wrong class.");
      continue;
    }

    IDCAverageGroup<IDCAverageGroupCRU>* idcavg = (IDCAverageGroup<IDCAverageGroupCRU>*)fInp.Get(key->GetName());
    IDCAverageGroupHelper<IDCAverageGroupCRU> idcStruct(idcavg->mIDCsGrouped, idcavg->mWeightsPad, idcavg->mWeightsRow, idcavg->mIDCsUngrouped, idcavg->mRobustAverage, idcavg->getCRU());
    for (unsigned int integrationInterval = 0; integrationInterval < idcavg->getNIntegrationIntervals(); ++integrationInterval) {
      idcStruct.set(0, integrationInterval);
      createDebugTree(idcStruct, pcstream);
    }
    delete idcavg;
  }
  pcstream.Close();
}

template <class Type>
void o2::tpc::IDCAverageGroup<Type>::createDebugTree(const IDCAverageGroupHelper<Type>& idcStruct, o2::utils::TreeStreamRedirector& pcstream)
{
  const Mapper& mapper = Mapper::instance();
  unsigned int cru = idcStruct.getCRU();
  const CRU cruTmp(cru);
  unsigned int sector = cruTmp.sector();
  unsigned int region = idcStruct.getRegion();
  unsigned int integrationInterval = idcStruct.getIntegrationInterval();

  const unsigned long padsPerCRU = Mapper::PADSPERREGION[region];
  std::vector<unsigned int> vRow(padsPerCRU);
  std::vector<unsigned int> vPad(padsPerCRU);
  std::vector<float> vXPos(padsPerCRU);
  std::vector<float> vYPos(padsPerCRU);
  std::vector<float> vGlobalXPos(padsPerCRU);
  std::vector<float> vGlobalYPos(padsPerCRU);
  std::vector<float> idcsPerIntegrationInterval(padsPerCRU);        // idcs for one time bin
  std::vector<float> groupedidcsPerIntegrationInterval(padsPerCRU); // idcs for one time bin
  std::vector<float> invPadArea(padsPerCRU);

  for (unsigned int iPad = 0; iPad < padsPerCRU; ++iPad) {
    const GlobalPadNumber globalNum = Mapper::GLOBALPADOFFSET[region] + iPad;
    const auto& padPosLocal = mapper.padPos(globalNum);
    vRow[iPad] = padPosLocal.getRow();
    vPad[iPad] = padPosLocal.getPad();
    vXPos[iPad] = mapper.getPadCentre(padPosLocal).X();
    vYPos[iPad] = mapper.getPadCentre(padPosLocal).Y();
    invPadArea[iPad] = Mapper::INVPADAREA[region];
    const GlobalPosition2D globalPos = mapper.LocalToGlobal(LocalPosition2D(vXPos[iPad], vYPos[iPad]), cruTmp.sector());
    vGlobalXPos[iPad] = globalPos.X();
    vGlobalYPos[iPad] = globalPos.Y();
    idcsPerIntegrationInterval[iPad] = idcStruct.getUngroupedIDCVal(iPad);
    groupedidcsPerIntegrationInterval[iPad] = idcStruct.getGroupedIDCValGlobal(vRow[iPad], vPad[iPad]);
  }

  pcstream << "tree"
           << "cru=" << cru
           << "sector=" << sector
           << "region=" << region
           << "integrationInterval=" << integrationInterval
           << "IDCUngrouped.=" << idcsPerIntegrationInterval
           << "IDCGrouped.=" << groupedidcsPerIntegrationInterval
           << "invPadArea.=" << invPadArea
           << "pad.=" << vPad
           << "row.=" << vRow
           << "lx.=" << vXPos
           << "ly.=" << vYPos
           << "gx.=" << vGlobalXPos
           << "gy.=" << vGlobalYPos
           << "\n";
}

template <class Type>
void o2::tpc::IDCAverageGroup<Type>::drawLatex(IDCAverageGroupHelper<IDCAverageGroupDraw>& idcStruct, const GlobalPadNumber padNum, const unsigned int padInRegion, const bool fillPoly, const int colOffs) const
{
  // drawing the grouping
  static auto coords = o2::tpc::painter::getPadCoordinatesSector();
  auto coordinate = coords[padNum];

  const float yPos = (coordinate.yVals[0] + coordinate.yVals[2]) / 2;
  const float xPos = (coordinate.xVals[0] + coordinate.xVals[2]) / 2;
  const int nCountDraw = idcStruct.mCountDraw[padInRegion]++;
  const float offsX = (nCountDraw % 2) * 0.6f * idcStruct.mPadInf.getPadHeight();
  const float offsY = (nCountDraw / 2) * 0.2f * idcStruct.mPadInf.getPadWidth();
  const float xPosDraw = xPos - 0.3f * idcStruct.mPadInf.getPadHeight() + offsX;
  const float yPosDraw = yPos - 0.4f * idcStruct.mPadInf.getPadWidth() + offsY;

  TLatex latex;
  latex.SetTextFont(63);
  latex.SetTextSize(1);

  const int col = idcStruct.mCol % idcStruct.mColors.size();
  const char* groupText = Form("#bf{#color[%d]{%i}}", col + 1, idcStruct.mGroupCounter);
  latex.DrawLatex(xPosDraw, yPosDraw, groupText);
  if (fillPoly) {
    idcStruct.mPoly.Fill(xPos, yPos, idcStruct.mColors[col] + colOffs);
  }
}

template class o2::tpc::IDCAverageGroup<o2::tpc::IDCAverageGroupCRU>;
template class o2::tpc::IDCAverageGroup<o2::tpc::IDCAverageGroupTPC>;
template void o2::tpc::IDCAverageGroup<o2::tpc::IDCAverageGroupCRU>::loopOverGroups(IDCAverageGroupHelper<o2::tpc::IDCAverageGroupCRU>&, const CalDet<PadFlags>*);
template void o2::tpc::IDCAverageGroup<o2::tpc::IDCAverageGroupTPC>::loopOverGroups(IDCAverageGroupHelper<o2::tpc::IDCAverageGroupTPC>&, const CalDet<PadFlags>*);
template void o2::tpc::IDCAverageGroup<o2::tpc::IDCAverageGroupCRU>::loopOverGroups(IDCAverageGroupHelper<o2::tpc::IDCAverageGroupDraw>&, const CalDet<PadFlags>*);
template void o2::tpc::IDCAverageGroup<o2::tpc::IDCAverageGroupTPC>::loopOverGroups(IDCAverageGroupHelper<o2::tpc::IDCAverageGroupDraw>&, const CalDet<PadFlags>*);
