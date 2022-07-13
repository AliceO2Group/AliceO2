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

#include "EMCALBase/Geometry.h"
#include "EMCALBase/TriggerMappingV2.h"
#include "EMCALBase/TriggerMappingErrors.h"

ClassImp(o2::emcal::TriggerMappingV2);

using namespace o2::emcal;

TriggerMappingV2::TriggerMappingV2()
{
  reset_arrays();
  init_TRU_offset();
  init_SM_offset();
}

TriggerMappingV2::TriggerMappingV2(o2::emcal::Geometry* geo) : mGeometry(geo)
{
  reset_arrays();
  init_TRU_offset();
  init_SM_offset();
}

void TriggerMappingV2::reset_arrays()
{
  std::fill(mTRUFastOROffsetX.begin(), mTRUFastOROffsetX.end(), 0);
  std::fill(mTRUFastOROffsetY.begin(), mTRUFastOROffsetY.end(), 0);
  std::fill(mNFastORInTRUPhi.begin(), mNFastORInTRUPhi.end(), 0);
  std::fill(mNFastORInTRUEta.begin(), mNFastORInTRUEta.end(), 0);

  std::fill(mSMFastOROffsetX.begin(), mSMFastOROffsetX.end(), 0);
  std::fill(mSMFastOROffsetY.begin(), mSMFastOROffsetY.end(), 0);
  std::fill(mNFastORInSMPhi.begin(), mNFastORInSMPhi.end(), 0);
  std::fill(mNFastORInSMEta.begin(), mNFastORInSMEta.end(), 0);

  std::fill(mNModuleInEMCALPhi.begin(), mNModuleInEMCALPhi.end(), 0);
}

void TriggerMappingV2::init_TRU_offset()
{
  mTRUFastOROffsetX[0] = 0;
  mTRUFastOROffsetY[0] = 0;
  IndexTRU currentTRU = 0;

  enum class TRUType_t {
    STANDARD_TRU,
    EXT_TRU
  };

  for (IndexSupermodule supermoduleID = 0; supermoduleID < SUPERMODULES; supermoduleID++) {
    auto smtype = getSupermoduleType(supermoduleID);
    auto isCside = isSupermoduleOnCSide(supermoduleID);
    TRUType_t trutype = TRUType_t::STANDARD_TRU;

    //===================
    // TRU ieta/iphi size
    auto nTRU_inSM = TRUSSUPERMODULE;
    auto nTRU_inSM_phi = TRUSPHISM;
    auto nTRU_inSM_eta = TRUSETASM;
    auto nModule_inTRU_phi = FASTORSPHITRU;
    auto nModule_inTRU_eta = FASTORSETATRU;
    if (smtype == EMCAL_THIRD || smtype == DCAL_EXT) {
      nTRU_inSM = static_cast<unsigned int>(static_cast<float>(nTRU_inSM) / 3.);
      nTRU_inSM_eta = static_cast<unsigned int>(static_cast<float>(nTRU_inSM_eta) / 3.);
      nModule_inTRU_phi = static_cast<unsigned int>(static_cast<float>(nModule_inTRU_phi) / 3.);
      nModule_inTRU_eta = nModule_inTRU_eta * 3;
    }

    //===================
    // TRU ieta/iphi offset calculation
    for (IndexTRU truInSupermodule = 0; truInSupermodule < nTRU_inSM; truInSupermodule++) {
      mNFastORInTRUPhi[currentTRU] = nModule_inTRU_phi;
      mNFastORInTRUEta[currentTRU] = nModule_inTRU_eta;
      if (isCside) {
        mTRUIsCside.set(currentTRU, true);
      } else {
        mTRUIsCside.set(currentTRU, false);
      }

      if ((currentTRU + 1) >= ALLTRUS) {
        break;
      }

      trutype = TRUType_t::STANDARD_TRU;
      if (truInSupermodule == nTRU_inSM - 1 && isCside) { // last TRU in SM
        trutype = TRUType_t::EXT_TRU;                     // right
      }

      // calculate offset for the next TRU in supermodule (if any)
      switch (trutype) {
        case TRUType_t::STANDARD_TRU:
          mTRUFastOROffsetX[currentTRU + 1] = mTRUFastOROffsetX[currentTRU] + nModule_inTRU_eta;
          mTRUFastOROffsetY[currentTRU + 1] = mTRUFastOROffsetY[currentTRU];
          break;
        case TRUType_t::EXT_TRU:
          mTRUFastOROffsetX[currentTRU + 1] = 0;
          mTRUFastOROffsetY[currentTRU + 1] = mTRUFastOROffsetY[currentTRU] + nModule_inTRU_phi;
          break;
      };
      currentTRU++;
    } // TRU loop
  }   // SM loop
}

/// Initialize mapping offsets of SM (add more description)
void TriggerMappingV2::init_SM_offset()
{
  mSMFastOROffsetX[0] = 0;
  mSMFastOROffsetY[0] = 0;
  mNModuleInEMCALPhi[0] = 0;
  int iB = 0;

  EMCALSMType currentSMtype = NOT_EXISTENT;
  for (IndexSupermodule supermoduleID = 0; supermoduleID < SUPERMODULES; supermoduleID++) {
    auto smtype = getSupermoduleType(supermoduleID);
    auto isCside = isSupermoduleOnCSide(supermoduleID);

    IndexRowPhi nModule_inSM_phi = FASTORSPHISM;
    IndexColumnEta nModule_inSM_eta = FASTORSETASM;

    if (smtype == EMCAL_THIRD || smtype == DCAL_EXT) {
      nModule_inSM_phi = static_cast<IndexRowPhi>(static_cast<float>(nModule_inSM_phi) / 3.);
    }

    mNFastORInSMPhi[supermoduleID] = nModule_inSM_phi;
    mNFastORInSMEta[supermoduleID] = nModule_inSM_eta;

    if (!isCside) {
      if (currentSMtype == smtype) {
        mNModuleInEMCALPhi[iB] += nModule_inSM_phi;
      } else {
        mNModuleInEMCALPhi[iB + 1] = mNModuleInEMCALPhi[iB] + nModule_inSM_phi;
        iB++;
      }
      currentSMtype = smtype;
    }

    if ((supermoduleID + 1) >= SUPERMODULES) {
      break;
    }

    // initialize offsets for next supermodule (if any)
    if (isCside) { // right SM
      mSMFastOROffsetX[supermoduleID + 1] = 0;
      mSMFastOROffsetY[supermoduleID + 1] = mSMFastOROffsetY[supermoduleID] + nModule_inSM_phi;
    } else { // left SM
      mSMFastOROffsetX[supermoduleID + 1] = mSMFastOROffsetX[supermoduleID] + nModule_inSM_eta;
      mSMFastOROffsetY[supermoduleID + 1] = mSMFastOROffsetY[supermoduleID];
    }
  } // SM loop
}

TriggerMappingV2::IndexFastOR TriggerMappingV2::getAbsFastORIndexFromIndexInTRU(IndexTRU truIndex, unsigned int positionInTRU) const
{
  if (truIndex >= ALLTRUS) {
    throw TRUIndexException(truIndex);
  }
  if (positionInTRU >= FASTORSTRU) {
    throw FastORIndexException(positionInTRU);
  }

  // invert index on C-side
  IndexFastOR fastorIndexInverted = (mTRUIsCside.test(truIndex)) ? (FASTORSTRU - positionInTRU - 1) : positionInTRU;
  IndexColumnEta columnInTRU = mTRUFastOROffsetX[truIndex] + IndexFastOR(fastorIndexInverted / mNFastORInTRUPhi[truIndex]);
  IndexRowPhi rowInTRU = mTRUFastOROffsetY[truIndex] + mNFastORInTRUPhi[truIndex] - 1 - IndexFastOR(fastorIndexInverted % mNFastORInTRUPhi[truIndex]);

  IndexFastOR id = rowInTRU * FASTORSETA + columnInTRU;
  id = rotateAbsFastOrIndexEtaToPhi(id);

  return id;
}

TriggerMappingV2::IndexFastOR TriggerMappingV2::getAbsFastORIndexFromPositionInTRU(IndexTRU truIndex, IndexColumnEta etaColumn, IndexRowPhi phiRow) const
{
  if (truIndex >= ALLTRUS) {
    throw TRUIndexException(truIndex);
  }
  if (etaColumn > mNFastORInTRUEta[truIndex] - 1 ||
      phiRow > mNFastORInTRUPhi[truIndex] - 1) {
    throw FastORPositionExceptionTRU(truIndex, etaColumn, phiRow);
  }

  IndexColumnEta etatmp = etaColumn; // XXX
  IndexRowPhi phitmp = phiRow;       // XXX

  // unsigned int etatmp = ( mTRUIsCside[ iTRU])? (fnFastORInTRUEta[truIndex] - 1 - etaColumn) : etaColumn  ;
  // unsigned int phitmp = (!mTRUIsCside.test(truIndex))? (fnFastORInTRUPhi[truIndex] - 1 - phiRow) : phiRow  ;

  IndexColumnEta x = mTRUFastOROffsetX[truIndex] + etatmp;
  IndexRowPhi y = mTRUFastOROffsetY[truIndex] + phitmp;

  IndexFastOR id = y * FASTORSETA + x;
  id = rotateAbsFastOrIndexEtaToPhi(id);
  return id;
}

TriggerMappingV2::IndexFastOR TriggerMappingV2::getAbsFastORIndexFromPositionInSupermodule(IndexSupermodule supermoduleID, IndexColumnEta etaColumn, IndexRowPhi phiRow) const
{
  if (supermoduleID >= SUPERMODULES) {
    throw SupermoduleIndexException(supermoduleID, SUPERMODULES);
  }
  if (
    etaColumn >= mNFastORInSMEta[supermoduleID] ||
    phiRow >= mNFastORInSMPhi[supermoduleID]) {
    throw FastORPositionExceptionSupermodule(supermoduleID, etaColumn, phiRow);
  }

  // Int_t iEtatmp = (GetSMIsCside(iSM) && GetSMType(iSM) == kDCAL_Standard)?(iEta + 8):iEta ;
  // Int_t x = fSMFastOROffsetX[iSM] + iEtatmp ;

  IndexColumnEta x = mSMFastOROffsetX[supermoduleID] + etaColumn;
  IndexRowPhi y = mSMFastOROffsetY[supermoduleID] + phiRow;

  IndexFastOR id = y * FASTORSETA + x;
  id = rotateAbsFastOrIndexEtaToPhi(id);
  return id;
}

TriggerMappingV2::IndexFastOR TriggerMappingV2::getAbsFastORIndexFromPositionInEMCAL(IndexColumnEta etaColumn, IndexRowPhi phiRow) const
{
  if (
    etaColumn >= FASTORSETA ||
    phiRow >= FASTORSPHI) {
    throw FastORPositionExceptionEMCAL(etaColumn, phiRow);
  }

  IndexFastOR id = phiRow * FASTORSETA + etaColumn;
  id = rotateAbsFastOrIndexEtaToPhi(id);
  return id;
}

TriggerMappingV2::IndexFastOR TriggerMappingV2::getAbsFastORIndexFromPHOSSubregion(unsigned int phosRegionID) const
{
  if (phosRegionID > 35) {
    throw PHOSRegionException(phosRegionID);
  }

  IndexColumnEta absColumnEta = 16 + 4 * IndexColumnEta(phosRegionID % 4);
  IndexRowPhi absRowPhi = 64 + 4 * IndexRowPhi(phosRegionID / 4);

  return getAbsFastORIndexFromPositionInEMCAL(absColumnEta, absRowPhi);
}

std::tuple<TriggerMappingV2::IndexTRU, TriggerMappingV2::IndexFastOR> TriggerMappingV2::getTRUFromAbsFastORIndex(IndexFastOR fastOrAbsID) const
{
  IndexFastOR convertedFastorIndex = rotateAbsFastOrIndexPhiToEta(fastOrAbsID);

  auto fastorInfo = getInfoFromAbsFastORIndex(convertedFastorIndex);
  return std::make_tuple(fastorInfo.mTRUID, fastorInfo.mFastORIDTRU);
}

std::tuple<TriggerMappingV2::IndexTRU, TriggerMappingV2::IndexColumnEta, TriggerMappingV2::IndexRowPhi> TriggerMappingV2::getPositionInTRUFromAbsFastORIndex(IndexFastOR fastOrAbsID) const
{

  IndexFastOR convertedFastorIndex = rotateAbsFastOrIndexPhiToEta(fastOrAbsID);

  auto fastorInfo = getInfoFromAbsFastORIndex(convertedFastorIndex);
  return std::make_tuple(fastorInfo.mTRUID, fastorInfo.mColumnEtaTRU, fastorInfo.mRowPhiTRU);
}

std::tuple<TriggerMappingV2::IndexSupermodule, TriggerMappingV2::IndexColumnEta, TriggerMappingV2::IndexRowPhi> TriggerMappingV2::getPositionInSupermoduleFromAbsFastORIndex(IndexFastOR fastOrAbsID) const
{
  IndexFastOR convertedFastorIndex = rotateAbsFastOrIndexPhiToEta(fastOrAbsID);

  auto fastorInfo = getInfoFromAbsFastORIndex(convertedFastorIndex);
  return std::make_tuple(fastorInfo.mSupermoduleID, fastorInfo.mColumnEtaSupermodule, fastorInfo.mRowPhiSupermodule);
}

std::tuple<TriggerMappingV2::IndexColumnEta, TriggerMappingV2::IndexRowPhi> TriggerMappingV2::getPositionInEMCALFromAbsFastORIndex(IndexFastOR fastorAbsID) const
{
  if (fastorAbsID >= ALLFASTORS) {
    throw FastORIndexException(fastorAbsID);
  }

  IndexFastOR convertedFastorIndex = rotateAbsFastOrIndexPhiToEta(fastorAbsID);
  TriggerMappingV2::IndexColumnEta column = convertedFastorIndex % FASTORSETA;
  TriggerMappingV2::IndexRowPhi row = convertedFastorIndex / FASTORSETA;
  return std::make_tuple(column, row);
}

TriggerMappingV2::IndexFastOR TriggerMappingV2::getAbsFastORIndexFromCellIndex(IndexCell cellindex) const
{
  if (!mGeometry) {
    throw GeometryNotSetException();
  }
  auto [supermoduleID, moduleId, cellPhiModule, cellEtaModule] = mGeometry->GetCellIndex(cellindex);

  auto [cellPhiSupermodule, cellEtaSupermodule] = mGeometry->GetCellPhiEtaIndexInSModule(supermoduleID, moduleId, cellPhiModule, cellEtaModule);

  // ietam:0-31  for DCAL Cside
  auto cellEta = cellEtaSupermodule, cellPhi = cellPhiSupermodule;
  if (getSupermoduleType(supermoduleID) == DCAL_STANDARD && (supermoduleID % 2) == 1) {
    auto [cellPhiShifted, cellEtaShifted] = mGeometry->ShiftOfflineToOnlineCellIndexes(supermoduleID, cellPhiSupermodule, cellEtaSupermodule);
    cellEta = cellEtaShifted;
    cellPhi = cellPhiShifted;
  }

  // ietam:16-47 for DCAL Cside
  IndexRowPhi moduleRow = static_cast<IndexRowPhi>(cellPhi / 2);
  IndexColumnEta moduleColumn = static_cast<IndexColumnEta>(cellEta / 2);
  return getAbsFastORIndexFromPositionInSupermodule(supermoduleID, moduleColumn, moduleRow);
}

std::array<TriggerMappingV2::IndexCell, 4> TriggerMappingV2::getCellIndexFromAbsFastORIndex(IndexFastOR fastORAbsID) const
{
  if (!mGeometry) {
    throw GeometryNotSetException();
  }
  auto [supermoduleID, etaColumnSM, phiRowSM] = getPositionInSupermoduleFromAbsFastORIndex(fastORAbsID);
  IndexColumnEta cellEtaColumnSupermodule = 2 * etaColumnSM;
  IndexRowPhi cellPhiRowSupermodule = 2 * phiRowSM;

  // Shift index in case the module is a DCAL standard C-side module
  if (getSupermoduleType(supermoduleID) == DCAL_STANDARD) {
    if (supermoduleID % 2 == 1) {
      auto [cellPhiRowShifted, cellEtaColumnShifted] = mGeometry->ShiftOnlineToOfflineCellIndexes(supermoduleID, cellPhiRowSupermodule, cellEtaColumnSupermodule);
      cellEtaColumnSupermodule = cellEtaColumnShifted;
      cellPhiRowSupermodule = cellPhiRowShifted;
    }
  }
  std::array<IndexCell, 4> cells = {{static_cast<IndexCell>(mGeometry->GetAbsCellIdFromCellIndexes(supermoduleID, cellPhiRowSupermodule, cellEtaColumnSupermodule)),
                                     static_cast<IndexCell>(mGeometry->GetAbsCellIdFromCellIndexes(supermoduleID, cellPhiRowSupermodule, cellEtaColumnSupermodule + 1)),
                                     static_cast<IndexCell>(mGeometry->GetAbsCellIdFromCellIndexes(supermoduleID, cellPhiRowSupermodule + 1, cellEtaColumnSupermodule)),
                                     static_cast<IndexCell>(mGeometry->GetAbsCellIdFromCellIndexes(supermoduleID, cellPhiRowSupermodule + 1, cellEtaColumnSupermodule + 1))}};
  return cells;
}

TriggerMappingV2::IndexTRU TriggerMappingV2::convertTRUIndexSTUtoTRU(IndexTRU truIndexSTU, DetType_t detector) const
{
  if ((truIndexSTU > 31 && detector == DetType_t::DET_EMCAL) || (truIndexSTU > 13 && detector == DetType_t::DET_DCAL)) {
    throw TRUIndexException(truIndexSTU);
  }

  if (detector == DetType_t::DET_EMCAL) {
    return truIndexSTU;
  } else {
    return 32 + ((int)(truIndexSTU / 4) * 6) + ((truIndexSTU % 4 < 2) ? (truIndexSTU % 4) : (truIndexSTU % 4 + 2));
  }
}

TriggerMappingV2::IndexTRU TriggerMappingV2::convertTRUIndexTRUtoSTU(IndexTRU truIndexTRU) const
{
  if (truIndexTRU < 32) {
    return truIndexTRU;
  } else {
    IndexTRU truIndexSTU = truIndexTRU;
    if (truIndexSTU >= 48) {
      truIndexSTU -= 2;
    }
    if (truIndexSTU >= 42) {
      truIndexSTU -= 2;
    }
    if (truIndexSTU >= 36) {
      truIndexSTU -= 2;
    }
    truIndexSTU -= 32;

    return truIndexSTU;
  }
}

TriggerMappingV2::IndexTRU TriggerMappingV2::getTRUIndexFromOnlineHardareAddree(int hardwareAddress, unsigned int ddlID, unsigned int supermoduleID) const
{
  // 1/3 SMs

  if (supermoduleID == 10) {
    return 30;
  }
  if (supermoduleID == 11) {
    return 31;
  }
  if (supermoduleID == 18) {
    return 50;
  }
  if (supermoduleID == 19) {
    return 51;
  }

  // Standard EMCal/DCal SMs

  unsigned short branch = (hardwareAddress >> 11) & 0x1; // 0/1

  IndexTRU truIndex = ((ddlID << 1) | branch) - 1; // 0..2

  truIndex = (supermoduleID % 2) ? 2 - truIndex : truIndex;

  if (supermoduleID < 10) {
    truIndex += 3 * supermoduleID; // EMCal
  } else {
    truIndex += (3 * supermoduleID - 4); // DCal
  }

  if (truIndex >= ALLTRUS) {
    throw TRUIndexException(truIndex);
  }

  return truIndex;
}

std::array<unsigned int, 4> TriggerMappingV2::getFastORIndexFromL0Index(IndexTRU truIndex, IndexFastOR l0index, int l0size) const
{
  if (l0size <= 0 || l0size > 4) {
    throw L0sizeInvalidException(l0size);
  }

  int motif[4];
  motif[0] = 0;
  motif[2] = 1;
  motif[1] = mNFastORInTRUPhi[truIndex];
  motif[3] = mNFastORInTRUPhi[truIndex] + 1;

  std::array<unsigned int, 4> fastorIndex;
  std::fill(fastorIndex.begin(), fastorIndex.end(), 0);
  switch (l0size) {
    case 1: // Cosmic trigger
      fastorIndex[0] = getAbsFastORIndexFromIndexInTRU(truIndex, l0index);
      break;
    case 4: // Standard L0 patch
      for (int index = 0; index < 4; index++) {
        IndexFastOR fastorInTRU = mNFastORInTRUPhi[truIndex] * int(l0index / (mNFastORInTRUPhi[truIndex] - 1)) + (l0index % (mNFastORInTRUPhi[truIndex] - 1)) + motif[index];
        fastorIndex[index] = getAbsFastORIndexFromIndexInTRU(truIndex, fastorInTRU);
      }
      break;
    default:
      break;
  }

  return fastorIndex;
}
TriggerMappingV2::FastORInformation TriggerMappingV2::getInfoFromAbsFastORIndex( // conv from A
  IndexFastOR fastOrAbsID) const
{
  if (fastOrAbsID >= ALLFASTORS) {
    throw FastORIndexException(fastOrAbsID);
  }
  unsigned int fastorIndexTRU;

  IndexFastOR convertedFastorIndex = rotateAbsFastOrIndexEtaToPhi(fastOrAbsID);

  IndexTRU truIndex = convertedFastorIndex / FASTORSTRU;
  fastorIndexTRU = convertedFastorIndex % FASTORSTRU;
  if (truIndex >= ALLTRUS) {
    throw TRUIndexException(truIndex);
  }

  IndexColumnEta etaColumnTRU = fastorIndexTRU / mNFastORInTRUPhi[truIndex];
  IndexRowPhi phiRowTRU = fastorIndexTRU % mNFastORInTRUPhi[truIndex];
  fastorIndexTRU = mNFastORInTRUPhi[truIndex] * ((mTRUIsCside[truIndex]) ? (mNFastORInTRUEta[truIndex] - 1 - etaColumnTRU) : etaColumnTRU) + ((!mTRUIsCside[truIndex]) ? (mNFastORInTRUPhi[truIndex] - 1 - phiRowTRU) : phiRowTRU);

  IndexColumnEta etaColumnGlobal = fastOrAbsID % FASTORSETA;
  IndexRowPhi rowPhiGlobal = fastOrAbsID / FASTORSETA;

  Int_t idtmp = (rowPhiGlobal < mNModuleInEMCALPhi[2]) ? fastOrAbsID : (fastOrAbsID + FASTORSTRU * 4);

  IndexSupermodule supermoduleID = 2 * (int)(idtmp / (2 * FASTORSETASM * FASTORSPHISM)) + (int)(mTRUIsCside[truIndex]);
  if (supermoduleID >= SUPERMODULES) {
    throw SupermoduleIndexException(supermoduleID, SUPERMODULES);
  }

  IndexColumnEta etaColumnSupermodule = etaColumnGlobal % mNFastORInSMEta[supermoduleID];
  IndexRowPhi phiRowSupermodule = convertedFastorIndex % mNFastORInSMPhi[supermoduleID];
  return {
    truIndex,
    fastorIndexTRU,
    etaColumnTRU,
    phiRowTRU,
    supermoduleID,
    etaColumnSupermodule,
    phiRowSupermodule};
}

TriggerMappingV2::IndexFastOR TriggerMappingV2::rotateAbsFastOrIndexEtaToPhi(IndexFastOR fastorIndexInEta) const
{
  Int_t det_phi = int(fastorIndexInEta / FASTORSETA);
  Int_t nModule_inSM_phi = FASTORSPHISM; // number of modules in current supermodule
  Int_t fastOrIndexInPhi = 0;
  // Calculate FastOR offset relative to previous SM type
  for (int i = 1; i < 5; i++) {
    if (det_phi < mNModuleInEMCALPhi[i]) {
      fastOrIndexInPhi = FASTORSETA * mNModuleInEMCALPhi[i - 1];
      if (i == 2 || i == 4) {
        nModule_inSM_phi /= 3;
      }
      break;
    }
  }
  // fastOrIndexInPhi := Number of FastORs of the previous range with same SM type

  Int_t fastorInSMType = fastorIndexInEta - fastOrIndexInPhi;
  Int_t sectorInSMType = (int)(fastorInSMType / (FASTORSETA * nModule_inSM_phi));
  Int_t fastOrInSector = (int)(fastorInSMType % (FASTORSETA * nModule_inSM_phi));

  fastOrIndexInPhi += sectorInSMType * (FASTORSETA * nModule_inSM_phi); // Add back number of FastORs in previous tracking sectors of the same type
  // rotate arrangement from eta to phi
  fastOrIndexInPhi += (int)(fastOrInSector % FASTORSETA) * nModule_inSM_phi; // Add full colums in sector
  fastOrIndexInPhi += (int)(fastOrInSector / FASTORSETA);                    // Add FastORs in the last column

  return fastOrIndexInPhi;
}

TriggerMappingV2::IndexFastOR TriggerMappingV2::rotateAbsFastOrIndexPhiToEta(IndexFastOR fastOrIndexInPhi) const
{
  Int_t det_phi = int(fastOrIndexInPhi / FASTORSETA);
  Int_t fastorIndexInEta = 0;
  Int_t nModule_inSM_phi = FASTORSPHISM;
  // Calculate FastOR offset relative to previous SM type
  for (int i = 1; i < 5; i++) {
    if (det_phi < mNModuleInEMCALPhi[i]) {
      fastorIndexInEta = FASTORSETA * mNModuleInEMCALPhi[i - 1];
      if (i == 2 || i == 4) {
        nModule_inSM_phi /= 3;
      }
      break;
    }
  }
  // fastorIndexInEta := Number of FastORs of the previous range with same SM type

  Int_t fastorInSMType = fastOrIndexInPhi - fastorIndexInEta;
  Int_t sectorInSMType = (int)(fastorInSMType / (FASTORSETA * nModule_inSM_phi));
  Int_t fastOrInSector = (int)(fastorInSMType % (FASTORSETA * nModule_inSM_phi));

  Int_t columnInSector = fastOrInSector / nModule_inSM_phi;
  Int_t rowInSector = fastOrInSector % nModule_inSM_phi;

  fastorIndexInEta += sectorInSMType * (FASTORSETA * nModule_inSM_phi); // Add back number of FastORs in previous tracking sectors of the same type
  // rotate arrangement from phi to eta
  fastorIndexInEta += rowInSector * FASTORSETA + columnInSector;

  return fastorIndexInEta;
}

std::tuple<TriggerMappingV2::IndexTRU, TriggerMappingV2::IndexFastOR> TriggerMappingV2::convertFastORIndexSTUtoTRU(IndexTRU truIndexSTU, IndexFastOR fastOrIndexSTU, DetType_t detector) const
{
  if (fastOrIndexSTU >= FASTORSTRU) {
    throw FastORIndexException(fastOrIndexSTU);
  }
  IndexTRU truIndexTRU = convertTRUIndexSTUtoTRU(truIndexSTU, detector);
  IndexColumnEta etaSTU = fastOrIndexSTU % mNFastORInTRUEta[truIndexTRU];
  IndexRowPhi phiSTU = fastOrIndexSTU / mNFastORInTRUEta[truIndexTRU];

  // Rotate position on C-side as indices in TRU scheme are rotated on C-side with respect to A-side
  IndexColumnEta etaTRU = (mTRUIsCside[truIndexTRU]) ? (mNFastORInTRUEta[truIndexTRU] - etaSTU - 1) : etaSTU;
  IndexRowPhi phiTRU = (mTRUIsCside[truIndexTRU]) ? phiSTU : (mNFastORInTRUPhi[truIndexTRU] - phiSTU - 1);
  IndexFastOR fastorIndexTRU = etaTRU * mNFastORInTRUPhi[truIndexTRU] + phiTRU;

  return std::make_tuple(truIndexTRU, fastorIndexTRU);
}

std::tuple<TriggerMappingV2::IndexTRU, TriggerMappingV2::IndexColumnEta, TriggerMappingV2::IndexRowPhi> TriggerMappingV2::convertFastORPositionSTUtoTRU(IndexTRU truIndexSTU, IndexColumnEta truEtaSTU, IndexRowPhi truPhiSTU, DetType_t detector) const
{
  auto truIndexTRU = convertTRUIndexSTUtoTRU(truIndexSTU, detector);
  if (truEtaSTU >= mNFastORInTRUEta[truIndexTRU] || truPhiSTU >= mNFastORInTRUPhi[truIndexTRU]) {
    throw FastORPositionExceptionTRU(truIndexTRU, truEtaSTU, truPhiSTU);
  }
  // Rotate position on C-side as indices in TRU scheme are rotated on C-side with respect to A-side
  IndexColumnEta truEtaTRU = (mTRUIsCside[truIndexTRU]) ? (mNFastORInTRUEta[truIndexTRU] - truEtaSTU - 1) : truEtaSTU;
  IndexRowPhi truPhiTRU = (mTRUIsCside[truIndexTRU]) ? truPhiSTU : (mNFastORInTRUPhi[truIndexTRU] - truPhiSTU - 1);

  return std::make_tuple(truIndexTRU, truEtaTRU, truPhiTRU);
}

std::tuple<TriggerMappingV2::IndexTRU, TriggerMappingV2::IndexFastOR> TriggerMappingV2::convertFastORIndexTRUtoSTU(IndexTRU truIndexTRU, IndexFastOR fastorIndexTRU) const
{
  if (truIndexTRU >= FASTORSTRU) {
    throw FastORIndexException(truIndexTRU);
  }
  IndexColumnEta etaTRU = fastorIndexTRU / mNFastORInTRUPhi[truIndexTRU];
  IndexRowPhi phiTRU = fastorIndexTRU % mNFastORInTRUPhi[truIndexTRU];
  // Rotate position on C-side as indices in TRU scheme are rotated on C-side with respect to A-side
  IndexColumnEta etaSTU = (mTRUIsCside[truIndexTRU]) ? (mNFastORInTRUEta[truIndexTRU] - etaTRU - 1) : etaTRU;
  IndexRowPhi phiSTU = (mTRUIsCside[truIndexTRU]) ? phiTRU : (mNFastORInTRUPhi[truIndexTRU] - phiTRU - 1);

  IndexTRU truIndexSTU = convertTRUIndexTRUtoSTU(truIndexTRU);
  IndexFastOR fastorIndexSTU = phiSTU * mNFastORInTRUEta[truIndexTRU] + etaSTU;

  return std::tuple(truIndexSTU, fastorIndexSTU);
}

std::tuple<TriggerMappingV2::IndexTRU, TriggerMappingV2::IndexColumnEta, TriggerMappingV2::IndexRowPhi> TriggerMappingV2::convertFastORPositionTRUtoSTU(IndexTRU truIndexTRU, IndexColumnEta truEtaTRU, IndexRowPhi truPhiTRU) const
{
  IndexTRU truIndexSTU = convertTRUIndexTRUtoSTU(truIndexTRU);
  if (truEtaTRU >= mNFastORInTRUEta[truIndexTRU] || truPhiTRU >= mNFastORInTRUPhi[truIndexTRU]) {
    throw FastORPositionExceptionTRU(truIndexTRU, truEtaTRU, truPhiTRU);
  }
  // Rotate position on C-side as indices in TRU scheme are rotated on C-side with respect to A-side
  IndexColumnEta truEtaSTU = (mTRUIsCside[truIndexTRU]) ? (mNFastORInTRUEta[truIndexTRU] - truEtaTRU - 1) : truEtaTRU;
  IndexRowPhi truPhiSTU = (mTRUIsCside[truIndexTRU]) ? truPhiTRU : (mNFastORInTRUPhi[truIndexTRU] - truPhiTRU - 1);
  return std::make_tuple(truIndexSTU, truEtaSTU, truPhiSTU);
}