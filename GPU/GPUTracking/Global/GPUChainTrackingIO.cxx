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

/// \file GPUChainTrackingIO.cxx
/// \author David Rohr

#include "GPUChainTracking.h"
#include "GPUReconstructionIO.h"
#include "GPUTPCClusterData.h"
#include "GPUTPCGMMergedTrack.h"
#include "GPUTPCGMMergedTrackHit.h"
#include "GPUTPCTrack.h"
#include "GPUTPCHitId.h"
#include "GPUTRDTrackletWord.h"
#include "GPUTRDRecoParam.h"
#include "AliHLTTPCClusterMCData.h"
#include "GPUTPCMCInfo.h"
#include "GPUTRDTrack.h"
#include "GPUTRDTracker.h"
#include "AliHLTTPCRawCluster.h"
#include "GPUTRDTrackletLabels.h"
#include "GPUQA.h"
#include "GPULogging.h"
#include "GPUReconstructionConvert.h"
#include "GPUMemorySizeScalers.h"
#include "GPUTrackingInputProvider.h"
#include "TPCZSLinkMapping.h"
#include "GPUTriggerOutputs.h"
#include "GPUConstantMem.h"

#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "GPUTPCClusterStatistics.h"
#include "DataFormatsTPC/ZeroSuppression.h"
#include "GPUHostDataTypes.h"
#include "DataFormatsTPC/Digit.h"
#include "CalibdEdxContainer.h"

#include "TPCFastTransformPOD.h"
#include "CorrectionMapsHelper.h"

using namespace o2::gpu;

#include "GPUO2DataTypes.h"

using namespace o2::tpc;
using namespace o2::trd;
using namespace o2::dataformats;

static constexpr uint32_t DUMP_HEADER_SIZE = 4;
static constexpr char DUMP_HEADER[DUMP_HEADER_SIZE + 1] = "CAv1";

GPUChainTracking::InOutMemory::InOutMemory() = default;
GPUChainTracking::InOutMemory::~InOutMemory() = default;
GPUChainTracking::InOutMemory::InOutMemory(GPUChainTracking::InOutMemory&&) = default;
GPUChainTracking::InOutMemory& GPUChainTracking::InOutMemory::operator=(GPUChainTracking::InOutMemory&&) = default; // NOLINT: False positive in clang-tidy

void GPUChainTracking::DumpData(const char* filename, const GPUTrackingInOutPointers* ioPtrs)
{
  FILE* fp = fopen(filename, "w+b");
  if (fp == nullptr) {
    return;
  }
  if (ioPtrs == nullptr) {
    ioPtrs = &mIOPtrs;
  }
  fwrite(DUMP_HEADER, 1, DUMP_HEADER_SIZE, fp);
  fwrite(&GPUReconstruction::geometryType, sizeof(GPUReconstruction::geometryType), 1, fp);
  DumpData(fp, ioPtrs->clusterData, ioPtrs->nClusterData, InOutPointerType::CLUSTER_DATA);
  DumpData(fp, ioPtrs->rawClusters, ioPtrs->nRawClusters, InOutPointerType::RAW_CLUSTERS);
  if (ioPtrs->clustersNative) {
    if (DumpData(fp, &ioPtrs->clustersNative->clustersLinear, &ioPtrs->clustersNative->nClustersTotal, InOutPointerType::CLUSTERS_NATIVE)) {
      fwrite(&ioPtrs->clustersNative->nClusters[0][0], sizeof(ioPtrs->clustersNative->nClusters[0][0]), NSECTORS * GPUCA_ROW_COUNT, fp);
      if (ioPtrs->clustersNative->clustersMCTruth) {
        const auto& buffer = ioPtrs->clustersNative->clustersMCTruth->getBuffer();
        std::pair<const char*, size_t> tmp = {buffer.data(), buffer.size()};
        DumpData(fp, &tmp.first, &tmp.second, InOutPointerType::CLUSTER_NATIVE_MC);
      }
    }
  }
  if (ioPtrs->tpcPackedDigits) {
    if (DumpData(fp, ioPtrs->tpcPackedDigits->tpcDigits, ioPtrs->tpcPackedDigits->nTPCDigits, InOutPointerType::TPC_DIGIT) && ioPtrs->tpcPackedDigits->tpcDigitsMC) {
      const char* ptrs[NSECTORS];
      size_t sizes[NSECTORS];
      for (uint32_t i = 0; i < NSECTORS; i++) {
        if (ioPtrs->tpcPackedDigits->tpcDigitsMC->v[i]) {
          const auto& buffer = ioPtrs->tpcPackedDigits->tpcDigitsMC->v[i]->getBuffer();
          ptrs[i] = buffer.data();
          sizes[i] = buffer.size();
        } else {
          ptrs[i] = nullptr;
          sizes[i] = 0;
        }
      }
      DumpData(fp, ptrs, sizes, InOutPointerType::TPC_DIGIT_MC);
    }
  }
  if (ioPtrs->tpcZS) {
    size_t total = 0;
    for (int32_t i = 0; i < NSECTORS; i++) {
      for (uint32_t j = 0; j < GPUTrackingInOutZS::NENDPOINTS; j++) {
        for (uint32_t k = 0; k < ioPtrs->tpcZS->sector[i].count[j]; k++) {
          total += ioPtrs->tpcZS->sector[i].nZSPtr[j][k];
        }
      }
    }
    std::vector<std::array<char, TPCZSHDR::TPC_ZS_PAGE_SIZE>> pages(total);
    char* ptr = pages[0].data();
    GPUTrackingInOutZS::GPUTrackingInOutZSCounts counts;
    total = 0;
    for (int32_t i = 0; i < NSECTORS; i++) {
      for (uint32_t j = 0; j < GPUTrackingInOutZS::NENDPOINTS; j++) {
        for (uint32_t k = 0; k < ioPtrs->tpcZS->sector[i].count[j]; k++) {
          memcpy(&ptr[total * TPCZSHDR::TPC_ZS_PAGE_SIZE], ioPtrs->tpcZS->sector[i].zsPtr[j][k], ioPtrs->tpcZS->sector[i].nZSPtr[j][k] * TPCZSHDR::TPC_ZS_PAGE_SIZE);
          counts.count[i][j] += ioPtrs->tpcZS->sector[i].nZSPtr[j][k];
          total += ioPtrs->tpcZS->sector[i].nZSPtr[j][k];
        }
      }
    }
    total *= TPCZSHDR::TPC_ZS_PAGE_SIZE;
    if (DumpData(fp, &ptr, &total, InOutPointerType::TPC_ZS)) {
      fwrite(&counts, sizeof(counts), 1, fp);
    }
  }
  if (ioPtrs->tpcCompressedClusters) {
    if (ioPtrs->tpcCompressedClusters->ptrForward) {
      throw std::runtime_error("Cannot dump non-flat compressed clusters");
    }
    char* ptr = (char*)ioPtrs->tpcCompressedClusters;
    size_t size = ioPtrs->tpcCompressedClusters->totalDataSize;
    DumpData(fp, &ptr, &size, InOutPointerType::TPC_COMPRESSED_CL);
  }
  if (ioPtrs->settingsTF) {
    uint32_t n = 1;
    DumpData(fp, &ioPtrs->settingsTF, &n, InOutPointerType::TF_SETTINGS);
  }
  DumpData(fp, ioPtrs->sectorTracks, ioPtrs->nSectorTracks, InOutPointerType::SECTOR_OUT_TRACK);
  DumpData(fp, ioPtrs->sectorClusters, ioPtrs->nSectorClusters, InOutPointerType::SECTOR_OUT_CLUSTER);
  DumpData(fp, &ioPtrs->mcLabelsTPC, &ioPtrs->nMCLabelsTPC, InOutPointerType::MC_LABEL_TPC);
  DumpData(fp, &ioPtrs->mcInfosTPC, &ioPtrs->nMCInfosTPC, InOutPointerType::MC_INFO_TPC);
  DumpData(fp, &ioPtrs->mcInfosTPCCol, &ioPtrs->nMCInfosTPCCol, InOutPointerType::MC_INFO_TPC);
  DumpData(fp, &ioPtrs->mergedTracks, &ioPtrs->nMergedTracks, InOutPointerType::MERGED_TRACK);
  DumpData(fp, &ioPtrs->mergedTrackHits, &ioPtrs->nMergedTrackHits, InOutPointerType::MERGED_TRACK_HIT);
  DumpData(fp, &ioPtrs->trdTracks, &ioPtrs->nTRDTracks, InOutPointerType::TRD_TRACK);
  DumpData(fp, &ioPtrs->trdTracklets, &ioPtrs->nTRDTracklets, InOutPointerType::TRD_TRACKLET);
  if (ioPtrs->trdSpacePoints) {
    DumpData(fp, &ioPtrs->trdSpacePoints, &ioPtrs->nTRDTracklets, InOutPointerType::TRD_SPACEPOINT);
  }
  DumpData(fp, &ioPtrs->trdTriggerTimes, &ioPtrs->nTRDTriggerRecords, InOutPointerType::TRD_TRIGGERRECORDS);
  DumpData(fp, &ioPtrs->trdTrackletIdxFirst, &ioPtrs->nTRDTriggerRecords, InOutPointerType::TRD_TRIGGERRECORDS);
  DumpData(fp, &ioPtrs->trdTrigRecMask, &ioPtrs->nTRDTriggerRecords, InOutPointerType::TRD_TRIGGERRECORDS);
  fclose(fp);
}

int32_t GPUChainTracking::ReadData(const char* filename)
{
  ClearIOPointers();
  FILE* fp = fopen(filename, "rb");
  if (fp == nullptr) {
    return (1);
  }

  char buf[DUMP_HEADER_SIZE + 1] = "";
  size_t r = fread(buf, 1, DUMP_HEADER_SIZE, fp);
  if (strncmp(DUMP_HEADER, buf, DUMP_HEADER_SIZE)) {
    GPUError("Invalid file header");
    fclose(fp);
    return -1;
  }
  GeometryType geo;
  r = fread(&geo, sizeof(geo), 1, fp);
  if (geo != GPUReconstruction::geometryType) {
    GPUError("File has invalid geometry (%s v.s. %s)", GPUReconstruction::GEOMETRY_TYPE_NAMES[(int32_t)geo], GPUReconstruction::GEOMETRY_TYPE_NAMES[(int32_t)GPUReconstruction::geometryType]);
    fclose(fp);
    return 1;
  }
  GPUTPCClusterData* ptrClusterData[NSECTORS];
  ReadData(fp, mIOPtrs.clusterData, mIOPtrs.nClusterData, mIOMem.clusterData, InOutPointerType::CLUSTER_DATA, ptrClusterData);
  AliHLTTPCRawCluster* ptrRawClusters[NSECTORS];
  ReadData(fp, mIOPtrs.rawClusters, mIOPtrs.nRawClusters, mIOMem.rawClusters, InOutPointerType::RAW_CLUSTERS, ptrRawClusters);
  int32_t nClustersTotal = 0;
  mIOMem.clusterNativeAccess.reset(new ClusterNativeAccess);
  if (ReadData<ClusterNative>(fp, &mIOMem.clusterNativeAccess->clustersLinear, &mIOMem.clusterNativeAccess->nClustersTotal, &mIOMem.clustersNative, InOutPointerType::CLUSTERS_NATIVE)) {
    r = fread(&mIOMem.clusterNativeAccess->nClusters[0][0], sizeof(mIOMem.clusterNativeAccess->nClusters[0][0]), NSECTORS * GPUCA_ROW_COUNT, fp);
    mIOMem.clusterNativeAccess->setOffsetPtrs();
    mIOPtrs.clustersNative = mIOMem.clusterNativeAccess.get();
    std::pair<const char*, size_t> tmp = {nullptr, 0};
    if (ReadData(fp, &tmp.first, &tmp.second, &mIOMem.clusterNativeMC, InOutPointerType::CLUSTER_NATIVE_MC)) {
      mIOMem.clusterNativeMCView = std::make_unique<ConstMCLabelContainerView>(gsl::span<const char>(tmp.first, tmp.first + tmp.second));
      mIOMem.clusterNativeAccess->clustersMCTruth = mIOMem.clusterNativeMCView.get();
    }
  }
  mIOMem.digitMap.reset(new GPUTrackingInOutDigits);
  if (ReadData(fp, mIOMem.digitMap->tpcDigits, mIOMem.digitMap->nTPCDigits, mIOMem.tpcDigits, InOutPointerType::TPC_DIGIT)) {
    mIOPtrs.tpcPackedDigits = mIOMem.digitMap.get();
    const char* ptrs[NSECTORS];
    size_t sizes[NSECTORS];
    if (ReadData(fp, ptrs, sizes, mIOMem.tpcDigitsMC, InOutPointerType::TPC_DIGIT_MC)) {
      mIOMem.tpcDigitMCMap = std::make_unique<GPUTPCDigitsMCInput>();
      mIOMem.tpcDigitMCView.reset(new ConstMCLabelContainerView[NSECTORS]);
      for (uint32_t i = 0; i < NSECTORS; i++) {
        if (sizes[i]) {
          mIOMem.tpcDigitMCView.get()[i] = gsl::span<const char>(ptrs[i], ptrs[i] + sizes[i]);
          mIOMem.tpcDigitMCMap->v[i] = mIOMem.tpcDigitMCView.get() + i;
        } else {
          mIOMem.tpcDigitMCMap->v[i] = nullptr;
        }
      }
      mIOMem.digitMap->tpcDigitsMC = mIOMem.tpcDigitMCMap.get();
    }
  }
  const char* ptr;
  size_t total;
  char* ptrZSPages;
  if (ReadData(fp, &ptr, &total, &mIOMem.tpcZSpagesChar, InOutPointerType::TPC_ZS, &ptrZSPages)) {
    GPUTrackingInOutZS::GPUTrackingInOutZSCounts counts;
    r = fread(&counts, sizeof(counts), 1, fp);
    mIOMem.tpcZSmeta.reset(new GPUTrackingInOutZS);
    mIOMem.tpcZSmeta2.reset(new GPUTrackingInOutZS::GPUTrackingInOutZSMeta);
    total = 0;
    for (int32_t i = 0; i < NSECTORS; i++) {
      for (uint32_t j = 0; j < GPUTrackingInOutZS::NENDPOINTS; j++) {
        mIOMem.tpcZSmeta2->ptr[i][j] = &ptrZSPages[total * TPCZSHDR::TPC_ZS_PAGE_SIZE];
        mIOMem.tpcZSmeta->sector[i].zsPtr[j] = &mIOMem.tpcZSmeta2->ptr[i][j];
        mIOMem.tpcZSmeta2->n[i][j] = counts.count[i][j];
        mIOMem.tpcZSmeta->sector[i].nZSPtr[j] = &mIOMem.tpcZSmeta2->n[i][j];
        mIOMem.tpcZSmeta->sector[i].count[j] = 1;
        total += counts.count[i][j];
      }
    }
    mIOPtrs.tpcZS = mIOMem.tpcZSmeta.get();
  }
  if (ReadData(fp, &ptr, &total, &mIOMem.tpcCompressedClusters, InOutPointerType::TPC_COMPRESSED_CL)) {
    mIOPtrs.tpcCompressedClusters = (const o2::tpc::CompressedClustersFlat*)ptr;
  }
  uint32_t n;
  ReadData(fp, &mIOPtrs.settingsTF, &n, &mIOMem.settingsTF, InOutPointerType::TF_SETTINGS);
  ReadData(fp, mIOPtrs.sectorTracks, mIOPtrs.nSectorTracks, mIOMem.sectorTracks, InOutPointerType::SECTOR_OUT_TRACK);
  ReadData(fp, mIOPtrs.sectorClusters, mIOPtrs.nSectorClusters, mIOMem.sectorClusters, InOutPointerType::SECTOR_OUT_CLUSTER);
  ReadData(fp, &mIOPtrs.mcLabelsTPC, &mIOPtrs.nMCLabelsTPC, &mIOMem.mcLabelsTPC, InOutPointerType::MC_LABEL_TPC);
  ReadData(fp, &mIOPtrs.mcInfosTPC, &mIOPtrs.nMCInfosTPC, &mIOMem.mcInfosTPC, InOutPointerType::MC_INFO_TPC);
  ReadData(fp, &mIOPtrs.mcInfosTPCCol, &mIOPtrs.nMCInfosTPCCol, &mIOMem.mcInfosTPCCol, InOutPointerType::MC_INFO_TPC);
  ReadData(fp, &mIOPtrs.mergedTracks, &mIOPtrs.nMergedTracks, &mIOMem.mergedTracks, InOutPointerType::MERGED_TRACK);
  ReadData(fp, &mIOPtrs.mergedTrackHits, &mIOPtrs.nMergedTrackHits, &mIOMem.mergedTrackHits, InOutPointerType::MERGED_TRACK_HIT);
  ReadData(fp, &mIOPtrs.trdTracks, &mIOPtrs.nTRDTracks, &mIOMem.trdTracks, InOutPointerType::TRD_TRACK);
  ReadData(fp, &mIOPtrs.trdTracklets, &mIOPtrs.nTRDTracklets, &mIOMem.trdTracklets, InOutPointerType::TRD_TRACKLET);
  uint32_t dummy = 0;
  ReadData(fp, &mIOPtrs.trdSpacePoints, &dummy, &mIOMem.trdSpacePoints, InOutPointerType::TRD_SPACEPOINT);
  ReadData(fp, &mIOPtrs.trdTriggerTimes, &mIOPtrs.nTRDTriggerRecords, &mIOMem.trdTriggerTimes, InOutPointerType::TRD_TRIGGERRECORDS);
  ReadData(fp, &mIOPtrs.trdTrackletIdxFirst, &mIOPtrs.nTRDTriggerRecords, &mIOMem.trdTrackletIdxFirst, InOutPointerType::TRD_TRIGGERRECORDS);
  ReadData(fp, &mIOPtrs.trdTrigRecMask, &mIOPtrs.nTRDTriggerRecords, &mIOMem.trdTrigRecMask, InOutPointerType::TRD_TRIGGERRECORDS);

  size_t fptr = ftell(fp);
  fseek(fp, 0, SEEK_END);
  size_t fend = ftell(fp);
  fclose(fp);
  if (fptr != fend) {
    GPUError("Error reading data file, reading incomplete");
    return 1;
  }
  (void)r;
  for (uint32_t i = 0; i < NSECTORS; i++) {
    for (uint32_t j = 0; j < mIOPtrs.nClusterData[i]; j++) {
      ptrClusterData[i][j].id = nClustersTotal++;
      if ((uint32_t)ptrClusterData[i][j].amp >= 25 * 1024) {
        GPUError("Invalid cluster charge, truncating (%d >= %d)", (int32_t)ptrClusterData[i][j].amp, 25 * 1024);
        ptrClusterData[i][j].amp = 25 * 1024 - 1;
      }
    }
    for (uint32_t j = 0; j < mIOPtrs.nRawClusters[i]; j++) {
      if ((uint32_t)mIOMem.rawClusters[i][j].GetCharge() >= 25 * 1024) {
        GPUError("Invalid raw cluster charge, truncating (%d >= %d)", (int32_t)ptrRawClusters[i][j].GetCharge(), 25 * 1024);
        ptrRawClusters[i][j].SetCharge(25 * 1024 - 1);
      }
      if ((uint32_t)mIOPtrs.rawClusters[i][j].GetQMax() >= 1024) {
        GPUError("Invalid raw cluster charge max, truncating (%d >= %d)", (int32_t)ptrRawClusters[i][j].GetQMax(), 1024);
        ptrRawClusters[i][j].SetQMax(1024 - 1);
      }
    }
  }

  return (0);
}

void GPUChainTracking::DumpSettings(const char* dir)
{
  std::string f;
  if (processors()->calibObjects.fastTransform != nullptr) {
    f = dir;
    f += "tpctransform.dump";
    DumpStructToFile(processors()->calibObjects.fastTransform, f.c_str());
  }
  if (processors()->calibObjects.fastTransformHelper != nullptr) {
    f = dir;
    f += "tpctransformhelper.dump";
    DumpStructToFile(processors()->calibObjects.fastTransformHelper, f.c_str());
  }
  if (processors()->calibObjects.tpcPadGain != nullptr) {
    f = dir;
    f += "tpcpadgaincalib.dump";
    DumpStructToFile(processors()->calibObjects.tpcPadGain, f.c_str());
  }
  if (processors()->calibObjects.tpcZSLinkMapping != nullptr) {
    f = dir;
    f += "tpczslinkmapping.dump";
    DumpStructToFile(processors()->calibObjects.tpcZSLinkMapping, f.c_str());
  }
  if (processors()->calibObjects.dEdxCalibContainer != nullptr) {
    f = dir;
    f += "dEdxCalibContainer.dump";
    DumpFlatObjectToFile(processors()->calibObjects.dEdxCalibContainer, f.c_str());
  }
  if (processors()->calibObjects.matLUT != nullptr) {
    f = dir;
    f += "matlut.dump";
    DumpFlatObjectToFile(processors()->calibObjects.matLUT, f.c_str());
  }
  if (processors()->calibObjects.trdGeometry != nullptr) {
    f = dir;
    f += "trdgeometry.dump";
    DumpStructToFile(processors()->calibObjects.trdGeometry, f.c_str());
  }
  if (processors()->calibObjects.trdRecoParam != nullptr) {
    f = dir;
    f += "trdrecoparam.dump";
    DumpStructToFile(processors()->calibObjects.trdRecoParam, f.c_str());
  }
}

void GPUChainTracking::ReadSettings(const char* dir)
{
  std::string f;
  f = dir;
  f += "tpctransform.dump";
  mTPCFastTransformU = ReadStructFromFile<TPCFastTransformPOD>(f.c_str());
  processors()->calibObjects.fastTransform = mTPCFastTransformU.get();
  f = dir;
  f += "tpctransformhelper.dump";
  mTPCFastTransformHelperU = ReadStructFromFile<CorrectionMapsHelper>(f.c_str());
  if ((processors()->calibObjects.fastTransformHelper = mTPCFastTransformHelperU.get())) {
    mTPCFastTransformHelperU->setCorrMap(mTPCFastTransformU.get());
  }
  f = dir;
  f += "tpcpadgaincalib.dump";
  mTPCPadGainCalibU = ReadStructFromFile<TPCPadGainCalib>(f.c_str());
  processors()->calibObjects.tpcPadGain = mTPCPadGainCalibU.get();
  f = dir;
  f += "tpczslinkmapping.dump";
  mTPCZSLinkMappingU = ReadStructFromFile<TPCZSLinkMapping>(f.c_str());
  processors()->calibObjects.tpcZSLinkMapping = mTPCZSLinkMappingU.get();
  f = dir;
  f += "dEdxCalibContainer.dump";
  mdEdxCalibContainerU = ReadFlatObjectFromFile<o2::tpc::CalibdEdxContainer>(f.c_str());
  processors()->calibObjects.dEdxCalibContainer = mdEdxCalibContainerU.get();
  f = dir;
  f += "matlut.dump";
  mMatLUTU = ReadFlatObjectFromFile<o2::base::MatLayerCylSet>(f.c_str());
  processors()->calibObjects.matLUT = mMatLUTU.get();
  f = dir;
  f += "trdgeometry.dump";
  mTRDGeometryU = ReadStructFromFile<o2::trd::GeometryFlat>(f.c_str());
  processors()->calibObjects.trdGeometry = mTRDGeometryU.get();
  f = dir;
  f += "trdrecoparam.dump";
  mTRDRecoParamU = ReadStructFromFile<GPUTRDRecoParam>(f.c_str());
  processors()->calibObjects.trdRecoParam = mTRDRecoParamU.get();
}
