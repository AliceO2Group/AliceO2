// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file AliHLTTPCClusterStatComponent.cxx
/// \author David Rohr

#include "AliCDBEntry.h"
#include "AliCDBManager.h"
#include "AliEventInfo.h"
#include "AliGRPObject.h"
#include "AliGeomManager.h"
#include "AliHLTExternalTrackParam.h"
#include "AliHLTGlobalBarrelTrack.h"
#include "GPUParam.h"
#include "AliHLTTPCClusterStatComponent.h"
#include "AliHLTTPCClusterTransformation.h"
#include "AliHLTTPCClusterXYZ.h"
#include "AliHLTTPCDataCompressionComponent.h"
#include "AliHLTTPCDefinitions.h"
#include "GPUTPCGMPropagator.h"
#include "GPUTPCGMPolynomialField.h"
#include "GPUTPCGMPolynomialFieldManager.h"
#include "GPUTPCGMTrackParam.h"
#include "AliHLTTPCGeometry.h"
#include "AliHLTTPCRawCluster.h"
#include "AliRawEventHeaderBase.h"
#include "AliRecoParam.h"
#include "AliRunInfo.h"
#include "AliTPCParam.h"
#include "AliTPCRecoParam.h"
#include "AliTPCTransform.h"
#include "AliTPCcalibDB.h"
#include <TGeoGlobalMagField.h>

using namespace GPUCA_NAMESPACE::gpu;

ClassImp(AliHLTTPCClusterStatComponent);

AliHLTTPCClusterStatComponent::AliHLTTPCClusterStatComponent()
  : AliHLTProcessor(), mSliceParam(nullptr), fTotal(0), fEdge(0), fSplitPad(0), fSplitTime(0), fSplitPadTime(0), fSplitPadOrTime(0), fAssigned(0), fCompressionStudy(0), fPrintClusters(0), fPrintClustersScaled(0), fDumpClusters(0), fAggregate(0), fSort(0), fEvent(0)
{
}

AliHLTTPCClusterStatComponent::~AliHLTTPCClusterStatComponent() {}

void AliHLTTPCClusterStatComponent::GetInputDataTypes(AliHLTComponentDataTypeList& list)
{
  list.push_back(AliHLTTPCDefinitions::fgkRawClustersDataType | kAliHLTDataOriginTPC);
  list.push_back(AliHLTTPCDefinitions::fgkTPCReverseTransformInfoDataType);
  list.push_back(AliHLTTPCDefinitions::ClustersXYZDataType());
  list.push_back((kAliHLTDataTypeTrack | kAliHLTDataOriginTPC));
}

AliHLTComponentDataType AliHLTTPCClusterStatComponent::GetOutputDataType() { return kAliHLTDataTypeHistogram | kAliHLTDataOriginOut; }

void AliHLTTPCClusterStatComponent::GetOutputDataSize(unsigned long& constBase, double& inputMultiplier)
{
  constBase = 2000000;
  inputMultiplier = 0.0;
}

int AliHLTTPCClusterStatComponent::ProcessOption(TString option, TString value)
{
  int iResult = 0;

  if (option.EqualTo("print-clusters")) {
    fPrintClusters = 1;
  } else if (option.EqualTo("aggregate")) {
    fAggregate = 1;
  } else if (option.EqualTo("sort")) {
    fSort = 1;
  } else if (option.EqualTo("print-clusters-scaled")) {
    fPrintClustersScaled = 1;
  } else if (option.EqualTo("dump-clusters")) {
    fDumpClusters = 1;
  } else if (option.EqualTo("compression-study")) {
    fCompressionStudy = 1;
  } else {
    HLTError("invalid option: %s", value.Data());
    return -EINVAL;
  }
  return iResult;
}

int AliHLTTPCClusterStatComponent::DoInit(int argc, const char** argv)
{
  int iResult = 0;

  if (ProcessOptionString(GetComponentArgs()) < 0) {
    HLTFatal("wrong config string! %s", GetComponentArgs().c_str());
    return -EINVAL;
  }

  if (fDumpClusters) {
    if ((fp = fopen("clusters.dump", "w+b")) == nullptr) {
      return -1;
    }
  }

  AliTPCcalibDB* pCalib = AliTPCcalibDB::Instance();
  const AliMagF* field = (AliMagF*)TGeoGlobalMagField::Instance()->GetField();
  pCalib->SetExBField(field);
  AliCDBEntry* entry = AliCDBManager::Instance()->Get("GRP/GRP/Data");
  AliGRPObject tmpGRP, *pGRP = 0;
  pGRP = dynamic_cast<AliGRPObject*>(entry->GetObject());
  AliRunInfo runInfo(pGRP->GetLHCState(), pGRP->GetBeamType(), pGRP->GetBeamEnergy(), pGRP->GetRunType(), pGRP->GetDetectorMask());
  AliEventInfo evInfo;
  evInfo.SetEventType(AliRawEventHeaderBase::kPhysicsEvent);
  entry = AliCDBManager::Instance()->Get("TPC/Calib/RecoParam");
  TObject* recoParamObj = entry->GetObject();

  static AliRecoParam fOfflineRecoParam;
  if (dynamic_cast<TObjArray*>(recoParamObj)) {
    TObjArray* copy = (TObjArray*)(static_cast<TObjArray*>(recoParamObj)->Clone());
    fOfflineRecoParam.AddDetRecoParamArray(1, copy);
  } else if (dynamic_cast<AliDetectorRecoParam*>(recoParamObj)) {
    AliDetectorRecoParam* copy = (AliDetectorRecoParam*)static_cast<AliDetectorRecoParam*>(recoParamObj)->Clone();
    fOfflineRecoParam.AddDetRecoParam(1, copy);
  }
  fOfflineRecoParam.SetEventSpecie(&runInfo, evInfo, 0);
  AliTPCRecoParam* recParam = (AliTPCRecoParam*)fOfflineRecoParam.GetDetRecoParam(1);
  pCalib->GetTransform()->SetCurrentRecoParam(recParam);

  mSliceParam = new GPUParam();
  mSliceParam->SetDefaults(GetBz());

  return iResult;
}

int AliHLTTPCClusterStatComponent::DoDeinit()
{
  if (fDumpClusters) {
    fclose(fp);
  }
  delete mSliceParam;
  mSliceParam = nullptr;
  return 0;
}

void AliHLTTPCClusterStatComponent::TransformReverse(int slice, int row, float y, float z, float padtime[])
{
  AliTPCcalibDB* calib = AliTPCcalibDB::Instance();
  AliTPCParam* param = calib->GetParameters();

  float padWidth = 0;
  // float padLength = 0;
  float maxPad = 0;
  float sign = slice < NSLICES / 2 ? 1 : -1;
  float zwidth;

  int sector;
  int sectorrow;
  if (row < AliHLTTPCGeometry::GetNRowLow()) {
    sector = slice;
    sectorrow = row;
    maxPad = param->GetNPadsLow(sectorrow);
    // padLength = param->GetPadPitchLength(sector, sectorrow);
    padWidth = param->GetPadPitchWidth(sector);
  } else {
    sector = slice + NSLICES;
    sectorrow = row - AliHLTTPCGeometry::GetNRowLow();
    maxPad = param->GetNPadsUp(sectorrow);
    // padLength = param->GetPadPitchLength(sector, sectorrow);
    padWidth = param->GetPadPitchWidth(sector);
  }

  padtime[0] = y * sign / padWidth + 0.5 * maxPad;

  float xyzGlobal[2] = {param->GetPadRowRadii(sector, sectorrow), y};
  AliHLTTPCGeometry::Local2Global(xyzGlobal, slice);

  float time = z * sign * 1024.f / 250.f;
  padtime[1] = (1024.f - time);
}

void AliHLTTPCClusterStatComponent::TransformForward(int slice, int row, float pad, float time, float xyz[])
{
  AliTPCcalibDB* calib = AliTPCcalibDB::Instance();
  AliTPCParam* param = calib->GetParameters();

  float padWidth = 0;
  // float padLength = 0;
  float maxPad = 0;
  float sign = slice < NSLICES / 2 ? 1 : -1;
  float zwidth;

  int sector;
  int sectorrow;
  if (row < AliHLTTPCGeometry::GetNRowLow()) {
    sector = slice;
    sectorrow = row;
    maxPad = param->GetNPadsLow(sectorrow);
    // padLength = param->GetPadPitchLength(sector, sectorrow);
    padWidth = param->GetPadPitchWidth(sector);
  } else {
    sector = slice + NSLICES;
    sectorrow = row - AliHLTTPCGeometry::GetNRowLow();
    maxPad = param->GetNPadsUp(sectorrow);
    // padLength = param->GetPadPitchLength(sector, sectorrow);
    padWidth = param->GetPadPitchWidth(sector);
  }

  xyz[0] = param->GetPadRowRadii(sector, sectorrow);
  xyz[1] = (pad - 0.5 * maxPad) * padWidth * sign;

  float xyzGlobal[2] = {xyz[0], xyz[1]};
  AliHLTTPCGeometry::Local2Global(xyzGlobal, slice);

  xyz[2] = sign * (1024 - time) * 250.f / 1024.f;
}

static bool AliHLTTPCClusterStat_sorthelper(const AliHLTTPCRawCluster& a, const AliHLTTPCRawCluster& b)
{
  if (a.GetPadRow() < b.GetPadRow()) {
    return (true);
  }
  if (a.GetPadRow() > b.GetPadRow()) {
    return (false);
  }
  if (a.GetPad() < b.GetPad()) {
    return (true);
  }
  if (a.GetPad() > b.GetPad()) {
    return (false);
  }
  if (a.GetTime() < b.GetTime()) {
    return (true);
  }
  if (a.GetTime() > b.GetTime()) {
    return (false);
  }
  return (false);
}

int AliHLTTPCClusterStatComponent::DoEvent(const AliHLTComponentEventData& evtData, const AliHLTComponentBlockData* blocks, AliHLTComponentTriggerData& /*trigData*/, AliHLTUInt8_t* /*outputPtr*/, AliHLTUInt32_t& /*size*/, AliHLTComponentBlockDataList& /*outputBlocks*/)
{
  int iResult = 0;

  if (!IsDataEvent()) {
    return iResult;
  }

  if (!fAggregate) {
    fTotal = fEdge = fSplitPad = fSplitTime = fSplitPadTime = fSplitPadOrTime = 0;
  }
  int nBlocks = evtData.fBlockCnt;

  AliHLTTPCRawClusterData* clustersArray[NSLICES][NPATCHES];
  AliHLTTPCClusterXYZData* clustersTransformedArray[NSLICES][NPATCHES];
  AliHLTTPCTrackHelperStruct* clustersTrackIDArray[NSLICES][NPATCHES];
  memset(clustersArray, 0, NSLICES * NPATCHES * sizeof(void*));
  memset(clustersTransformedArray, 0, NSLICES * NPATCHES * sizeof(void*));
  memset(clustersTrackIDArray, 0, NSLICES * NPATCHES * sizeof(void*));

  AliHLTTracksData* tracks = nullptr;

  float bz = GetBz();

  AliTPCcalibDB* pCalib = AliTPCcalibDB::Instance();
  AliTPCParam* tpcParam = pCalib->GetParameters();
  tpcParam->Update();
  tpcParam->ReadGeoMatrices();
  AliTPCTransform* transform = pCalib->GetTransform();
  const AliTPCRecoParam* rec = transform->GetCurrentRecoParam();
  transform->SetCurrentTimeStamp(GetTimeStamp());

  for (int ndx = 0; ndx < nBlocks; ndx++) {
    const AliHLTComponentBlockData* iter = blocks + ndx;

    if (iter->fDataType == (AliHLTTPCDefinitions::fgkRawClustersDataType | kAliHLTDataOriginTPC)) {
      int slice = AliHLTTPCDefinitions::GetMinSliceNr(iter->fSpecification);
      int patch = AliHLTTPCDefinitions::GetMinPatchNr(iter->fSpecification);

      clustersArray[slice][patch] = (AliHLTTPCRawClusterData*)(iter->fPtr);
    }

    if (iter->fDataType == AliHLTTPCDefinitions::ClustersXYZDataType()) {
      int slice = AliHLTTPCDefinitions::GetMinSliceNr(iter->fSpecification);
      int patch = AliHLTTPCDefinitions::GetMinPatchNr(iter->fSpecification);

      clustersTransformedArray[slice][patch] = (AliHLTTPCClusterXYZData*)(iter->fPtr);
      if (clustersTransformedArray[slice][patch]->fCount) {
        clustersTrackIDArray[slice][patch] = new AliHLTTPCTrackHelperStruct[clustersTransformedArray[slice][patch]->fCount];
        memset(clustersTrackIDArray[slice][patch], 0, clustersTransformedArray[slice][patch]->fCount * sizeof(AliHLTTPCTrackHelperStruct));
        for (int i = 0; i < clustersTransformedArray[slice][patch]->fCount; i++) {
          clustersTrackIDArray[slice][patch][i].fID = -1;
        }
      }
    }

    if (iter->fDataType == (kAliHLTDataTypeTrack | kAliHLTDataOriginTPC)) {
      tracks = (AliHLTTracksData*)iter->fPtr;
    }
  }

  if (fCompressionStudy) {
    if (tracks == nullptr) {
      HLTError("Tracks missing");
      return (0);
    }
  }

  double residualBarrelTrackY = 0, residualBarrelTrackZ = 0, residualExternalTrackY = 0, residualExternalTrackZ = 0, residualBacktransformPad = 0, residualBacktransformTime = 0;
  double residualBarrelTrackYabs = 0, residualBarrelTrackZabs = 0, residualExternalTrackYabs = 0, residualExternalTrackZabs = 0, residualBacktransformPadabs = 0, residualBacktransformTimeabs = 0;
  double residualFitTrackY = 0, residualFitTrackZ = 0, residualFitTrackYabs = 0, residualFitTrackZabs = 0, residualTrackRawPad = 0, residualTrackRawTime = 0, residualTrackRawPadabs = 0, residualTrackRawTimeabs = 0;
  int nClusterTracks = 0, nClusters = 0, nClusterTracksRaw = 0;

  const AliHLTUInt8_t* pCurrent = reinterpret_cast<const AliHLTUInt8_t*>(tracks->fTracklets);
  if (fCompressionStudy) {
    GPUTPCGMPropagator prop;
    const float kRho = 1.025e-3;  // 0.9e-3;
    const float kRadLen = 29.532; // 28.94;
    prop.SetMaxSinPhi(.999);
    prop.SetMaterial(kRadLen, kRho);
    GPUTPCGMPolynomialField field;
    int err = GPUTPCGMPolynomialFieldManager::GetPolynomialField(field);
    if (err != 0) {
      HLTError("Can not initialize polynomial magnetic field");
      return -1;
    }
    prop.SetPolynomialField(&field);
    for (unsigned i = 0; i < tracks->fCount; i++) {
      const AliHLTExternalTrackParam* track = reinterpret_cast<const AliHLTExternalTrackParam*>(pCurrent);
      if (track->fNPoints == 0) {
        continue;
      }

      AliHLTGlobalBarrelTrack btrack(*track);
      btrack.CalculateHelixParams(bz);

      AliExternalTrackParam etrack(btrack);

      GPUTPCGMTrackParam ftrack;
      float falpha;

      int hitsUsed = 0;
      float averageCharge = 0;
      float averageQMax = 0;
      AliHLTTPCTrackHelperStruct* hitIndexCache[1024];
      for (int ip = 0; ip < track->fNPoints; ip++) {
        int clusterID = track->fPointIDs[ip];
        int slice = AliHLTTPCGeometry::CluID2Slice(clusterID);
        int patch = AliHLTTPCGeometry::CluID2Partition(clusterID);
        int index = AliHLTTPCGeometry::CluID2Index(clusterID);

        if (clustersTrackIDArray[slice][patch][index].fID != -1) {
          HLTDebug("Already assigned hit %d of track %d, skipping", ip, i);
          continue;
        }

        if (index > clustersArray[slice][patch]->fCount) {
          HLTError("Cluster index out of range");
          continue;
        }

        AliHLTTPCRawCluster& cluster = clustersArray[slice][patch]->fClusters[index];
        AliHLTTPCClusterXYZ& clusterTransformed = clustersTransformedArray[slice][patch]->fClusters[index];

        int padrow = AliHLTTPCGeometry::GetFirstRow(patch) + cluster.GetPadRow();
        float x = AliHLTTPCGeometry::Row2X(padrow);
        float y = 0.0;
        float z = 0.0;

        float xyz[3];
        if (1) // Use forward (exact reverse-reverse) transformation of raw cluster (track fit in distorted coordinates)
        {
          TransformForward(slice, padrow, cluster.GetPad(), cluster.GetTime(), xyz);
        } else { // Correct cluster coordinates using correct transformation
          xyz[0] = x;
          xyz[1] = clusterTransformed.fY;
          xyz[2] = clusterTransformed.fZ;
        }

        float alpha = slice;
        if (alpha > NSLICES / 2) {
          alpha -= NSLICES / 2;
        }
        if (alpha > NSLICES / 4) {
          alpha -= NSLICES / 2;
        }
        alpha = (alpha + 0.5f) * M_PI / 9.f;
        btrack.CalculateCrossingPoint(x, alpha /* Better use btrack.GetAlpha() ?? */, y, z);

        etrack.Propagate(alpha, x, bz);

        if (ip == 0) {
          ftrack.Par()[0] = xyz[1];
          ftrack.Par()[1] = xyz[2];
          for (int k = 2; k < 5; k++) {
            ftrack.Par()[k] = etrack.GetParameter()[k];
          }
          ftrack.SetX(xyz[0]);
          falpha = alpha;

          prop.SetTrack(&ftrack, falpha);
          ftrack.ResetCovariance();
          bool inFlyDirection = 1;
          prop.PropagateToXAlpha(xyz[0], falpha, inFlyDirection);
        } else {
          bool inFlyDirection = 0;
          prop.PropagateToXAlpha(xyz[0], alpha, inFlyDirection);
        }

        nClusterTracks++;
        residualBarrelTrackYabs += fabsf(clusterTransformed.fY - y);
        residualBarrelTrackZabs += fabsf(clusterTransformed.fZ - z);
        residualExternalTrackYabs += fabsf(clusterTransformed.fY - (float)etrack.GetY());
        residualExternalTrackZabs += fabsf(clusterTransformed.fZ - (float)etrack.GetZ());
        residualBarrelTrackY += clusterTransformed.fY - y;
        residualBarrelTrackZ += clusterTransformed.fZ - z;
        residualExternalTrackY += clusterTransformed.fY - etrack.GetY();
        residualExternalTrackZ += clusterTransformed.fZ - etrack.GetZ();
        residualFitTrackY += clusterTransformed.fY - ftrack.GetY();
        residualFitTrackZ += clusterTransformed.fZ - ftrack.GetZ();
        residualFitTrackYabs += fabsf(clusterTransformed.fY - ftrack.GetY());
        residualFitTrackZabs += fabsf(clusterTransformed.fZ - ftrack.GetZ());

        // Show residuals wrt track position
        // HLTImportant("Residual %d btrack %f %f etrack %f %f ftrack %f %f", padrow, clusterTransformed.fY - y, clusterTransformed.fZ - z,
        // clusterTransformed.fY - etrack.GetY(), clusterTransformed.fZ - etrack.GetZ(),
        // clusterTransformed.fY - ftrack.GetY(), clusterTransformed.fZ - ftrack.GetZ());

        float padtime[2];
        TransformReverse(slice, padrow, ftrack.GetY(), ftrack.GetZ(), padtime);

        // Check forward / backward transformation
        /*float xyzChk[3];
                TransformForward(slice, padrow, padtime[0], padtime[1], xyzChk);
                HLTImportant("BackwardForward Residual %f %f %f: %f %f", ftrack.GetX(), ftrack.GetY(), ftrack.GetZ(), ftrack.GetY() - xyzChk[1], ftrack.GetZ() - xyzChk[2]);*/

        // Show residual wrt to raw cluster position
        // HLTImportant("Raw Cluster Residual %d (%d/%d) %d: %f %f (%f %f)", i, ip, track->fNPoints, padrow, cluster.GetPad() - padtime[0], cluster.GetTime() - padtime[1], clusterTransformed.fY - ftrack.GetY(), clusterTransformed.fZ - ftrack.GetZ());
        if (fabsf(cluster.GetPad() - padtime[0]) > 5 || fabsf(cluster.GetTime() - padtime[1]) > 5) {
          break;
        }

        if (ip != 0) {
          clustersTrackIDArray[slice][patch][index].fResidualPad = cluster.GetPad() - padtime[0];
          clustersTrackIDArray[slice][patch][index].fResidualTime = cluster.GetTime() - padtime[1];
          clustersTrackIDArray[slice][patch][index].fFirstHit = 0;

          residualTrackRawPad += cluster.GetPad() - padtime[0];
          residualTrackRawTime += cluster.GetTime() - padtime[1];
          residualTrackRawPadabs += fabsf(cluster.GetPad() - padtime[0]);
          residualTrackRawTimeabs += fabsf(cluster.GetTime() - padtime[1]);
          nClusterTracksRaw++;
        } else {
          clustersTrackIDArray[slice][patch][index].fResidualPad = cluster.GetPad();
          clustersTrackIDArray[slice][patch][index].fResidualTime = cluster.GetTime();
          clustersTrackIDArray[slice][patch][index].fFirstHit = 1;
        }
        clustersTrackIDArray[slice][patch][index].fID = i;
        clustersTrackIDArray[slice][patch][index].fTrack = track;
        if (hitsUsed >= 1024) {
          HLTFatal("hitIndex cache exceeded");
        }
        hitIndexCache[hitsUsed] = &clustersTrackIDArray[slice][patch][index];
        hitsUsed++;
        averageCharge += cluster.GetCharge();
        averageQMax += cluster.GetQMax();

        if (ip != 0) {
          int rowType = padrow < 64 ? 0 : (padrow < 128 ? 2 : 1);
          prop.Update(xyz[1], xyz[2], rowType, *mSliceParam, 0, false, false);
        }
      }
      if (hitsUsed) {
        averageCharge /= hitsUsed;
        averageQMax /= hitsUsed;
      }
      for (int ip = 0; ip < hitsUsed; ip++) {
        hitIndexCache[ip]->fAverageQMax = averageQMax;
        hitIndexCache[ip]->fAverageQTot = averageCharge;
      }
      pCurrent += sizeof(AliHLTExternalTrackParam) + track->fNPoints * sizeof(UInt_t);
    }
  }

  for (unsigned int is = 0; is < NSLICES; is++) {
    for (unsigned int ip = 0; ip < NPATCHES; ip++) {
      AliHLTTPCRawClusterData* clusters = clustersArray[is][ip];
      AliHLTTPCClusterXYZData* clustersTransformed = clustersTransformedArray[is][ip];
      int firstRow = AliHLTTPCGeometry::GetFirstRow(ip);

      if (clusters == nullptr) {
        HLTDebug("Clusters missing for slice %d patch %d\n", is, ip);
        continue;
      }
      if (fCompressionStudy && (clustersTransformed == nullptr || clusters->fCount != clustersTransformed->fCount)) {
        HLTError("Cluster cound not equal");
        continue;
      }

      AliHLTTPCRawCluster* sortedClusters;
      if (fSort) {
        if (fCompressionStudy) {
          HLTFatal("Cannot sort when compressionstudy is enabled");
        }
        sortedClusters = new AliHLTTPCRawCluster[clusters->fCount];
        memcpy(sortedClusters, clusters->fClusters, sizeof(AliHLTTPCRawCluster) * clusters->fCount);
        std::sort(sortedClusters, sortedClusters + clusters->fCount, AliHLTTPCClusterStat_sorthelper);
      }

      for (UInt_t iCluster = 0; iCluster < clusters->fCount; iCluster++) {
        AliHLTTPCRawCluster& cluster = clusters->fClusters[iCluster];
        AliHLTTPCClusterXYZ& clusterTransformed = clustersTransformed->fClusters[iCluster];
        static AliHLTTPCTrackHelperStruct tmp;
        AliHLTTPCTrackHelperStruct& clusterTrack = fCompressionStudy ? clustersTrackIDArray[is][ip][iCluster] : tmp;

        if (fCompressionStudy) {
          int row = cluster.GetPadRow() + firstRow;

          float xyz[3];
          TransformForward(is, row, cluster.GetPad(), cluster.GetTime(), xyz);

          /*float xyzOrig[3], xyzLocGlob[3];
                    {
                        int sector = AliHLTTPCGeometry::GetNRowLow() ? is : is + NSLICES;
                        int sectorrow = AliHLTTPCGeometry::GetNRowLow() ? row : row - AliHLTTPCGeometry::GetNRowLow();

                        Double_t xx[] = {(double) sectorrow, cluster.GetPad(), cluster.GetTime()};
                        transform->Transform(xx, &sector, 0, 1);

                        Double_t yy[] = {(double) sectorrow, cluster.GetPad(), cluster.GetTime()};
                        transform->Local2RotatedGlobal(sector, yy);
                        for (int k = 0; k < 3; k++)
                        {
                            xyzOrig[k] = xx[k];
                            xyzLocGlob[k] = yy[k];
                        }
                    }*/

          float padtime[2];
          TransformReverse(is, row, clusterTransformed.fY, clusterTransformed.fZ, padtime);

          nClusters++;
          residualBacktransformPadabs += fabsf(cluster.GetPad() - padtime[0]);
          residualBacktransformTimeabs += fabsf(cluster.GetTime() - padtime[1]);
          residualBacktransformPad += cluster.GetPad() - padtime[0];
          residualBacktransformTime += cluster.GetTime() - padtime[1];
        }

        fTotal++;
        if (cluster.GetFlagEdge()) {
          fEdge++;
        }
        if (cluster.GetFlagSplitPad()) {
          fSplitPad++;
        }
        if (cluster.GetFlagSplitTime()) {
          fSplitTime++;
        }
        if (cluster.GetFlagSplitAny()) {
          fSplitPadOrTime++;
        }
        if (cluster.GetFlagSplitPad() && cluster.GetFlagSplitTime()) {
          fSplitPadTime++;
        }

        AliHLTTPCRawCluster& cluster2 = fSort ? sortedClusters[iCluster] : cluster;

        if (fPrintClusters) {
          HLTImportant("Event %d Slice %d, Patch %d, Row %d, Pad %.2f, Time %.2f, SPad %.2f, STime %.2f, QMax %d, QTot %d, SplitPad %d, SplitTime %d, Edge %d, TrackId %d, ResPad %.2f ResTime %.2f AvgQTot %d AvgQMax %d", fEvent, is, ip, (int)cluster2.GetPadRow(), cluster2.GetPad(),
                       cluster2.GetTime(), cluster2.GetSigmaPad2(), cluster2.GetSigmaTime2(), (int)cluster2.GetQMax(), (int)cluster2.GetCharge(), (int)cluster2.GetFlagSplitPad(), (int)cluster2.GetFlagSplitTime(), (int)cluster2.GetFlagEdge(), (int)clusterTrack.fID,
                       clusterTrack.fResidualPad, clusterTrack.fResidualTime, (int)clusterTrack.fAverageQTot, (int)clusterTrack.fAverageQMax);
        }

        if (fCompressionStudy && clusterTrack.fID == -1) {
          PrintDumpClustersScaled(is, ip, cluster, clusterTransformed, clusterTrack);
        }
      }
      if (fSort) {
        delete[] sortedClusters;
      }
    }
  }

  if (fDumpClusters || fPrintClustersScaled) {
    const AliHLTUInt8_t* pCurrent = reinterpret_cast<const AliHLTUInt8_t*>(tracks->fTracklets);
    for (unsigned i = 0; i < tracks->fCount; i++) {
      const AliHLTExternalTrackParam* track = reinterpret_cast<const AliHLTExternalTrackParam*>(pCurrent);
      for (int ip = 0; ip < track->fNPoints; ip++) {
        int clusterID = track->fPointIDs[ip];
        int slice = AliHLTTPCGeometry::CluID2Slice(clusterID);
        int patch = AliHLTTPCGeometry::CluID2Partition(clusterID);
        int index = AliHLTTPCGeometry::CluID2Index(clusterID);

        AliHLTTPCRawCluster& cluster = clustersArray[slice][patch]->fClusters[index];
        AliHLTTPCClusterXYZ& clusterTransformed = clustersTransformedArray[slice][patch]->fClusters[index];
        AliHLTTPCTrackHelperStruct& clusterTrack = clustersTrackIDArray[slice][patch][index];

        if (clusterTrack.fID == i) {
          PrintDumpClustersScaled(slice, patch, cluster, clusterTransformed, clusterTrack);
        }
      }
      pCurrent += sizeof(AliHLTExternalTrackParam) + track->fNPoints * sizeof(UInt_t);
    }
  }

  for (unsigned int is = 0; is < NSLICES; is++) {
    for (unsigned int ip = 0; ip < NPATCHES; ip++) {
      if (clustersTrackIDArray[is][ip]) {
        delete[] clustersTrackIDArray[is][ip];
      }
    }
  }

  int total = fTotal == 0 ? 1 : fTotal;
  fAssigned += nClusterTracks;
  HLTImportant("Total %d Assigned %d (%2.0f\%) SplitPad %d (%2.0f\%) SplitTime %d (%2.0f\%) SplitPadTime %d (%2.0f\%) SplitPadOrTime %d (%2.0f\%) Edge %d (%2.0f\%)", fTotal, fAssigned, (float)fAssigned / (float)total * 100.f, fSplitPad, (float)fSplitPad / (float)total * 100.f, fSplitTime,
               (float)fSplitTime / (float)total * 100.f, fSplitPadTime, (float)fSplitPadTime / (float)total * 100.f, fSplitPadOrTime, (float)fSplitPadOrTime / (float)total * 100.f, fEdge, (float)fEdge / (float)total * 100.f);

  if (nClusterTracks) {
    residualBarrelTrackY /= nClusterTracks;
    residualBarrelTrackZ /= nClusterTracks;
    residualExternalTrackY /= nClusterTracks;
    residualExternalTrackZ /= nClusterTracks;
    residualBarrelTrackYabs /= nClusterTracks;
    residualBarrelTrackZabs /= nClusterTracks;
    residualExternalTrackYabs /= nClusterTracks;
    residualExternalTrackZabs /= nClusterTracks;
    residualFitTrackYabs /= nClusterTracks;
    residualFitTrackZabs /= nClusterTracks;
    residualFitTrackY /= nClusterTracks;
    residualFitTrackZ /= nClusterTracks;
  }
  if (nClusterTracksRaw) {
    residualTrackRawPadabs /= nClusterTracksRaw;
    residualTrackRawTimeabs /= nClusterTracksRaw;
    residualTrackRawPad /= nClusterTracksRaw;
    residualTrackRawTime /= nClusterTracksRaw;
  }
  if (nClusters) {
    residualBacktransformPad /= nClusters;
    residualBacktransformTime /= nClusters;
    residualBacktransformPadabs /= nClusters;
    residualBacktransformTimeabs /= nClusters;
  }

  if (fCompressionStudy) {
    HLTImportant("Average Res: BarrelTr %f %f, ExtlTr %f %f, FitTr %f %f BackTr %f %f TrkRaw %f %f", residualBarrelTrackY, residualBarrelTrackZ, residualExternalTrackY, residualExternalTrackZ, residualFitTrackY, residualFitTrackZ, residualBacktransformPad, residualBacktransformTime,
                 residualTrackRawPad, residualTrackRawTime);
    HLTImportant("Average Abs Res: BarrelTr %f %f, ExtTr %f %f, FitTr %f %f BackTr %f %f TrkRaw %f %f", residualBarrelTrackYabs, residualBarrelTrackZabs, residualExternalTrackYabs, residualExternalTrackZabs, residualFitTrackYabs, residualFitTrackZabs, residualBacktransformPadabs,
                 residualBacktransformTimeabs, residualTrackRawPadabs, residualTrackRawTimeabs);
  }

  fEvent++;

  return iResult;
}

void AliHLTTPCClusterStatComponent::PrintDumpClustersScaled(int is, int ip, AliHLTTPCRawCluster& cluster, AliHLTTPCClusterXYZ& clusterTransformed, AliHLTTPCClusterStatComponent::AliHLTTPCTrackHelperStruct& clusterTrack)
{
  AliHLTUInt64_t pad64 = 0;
  if (!isnan(cluster.GetPad())) {
    pad64 = (AliHLTUInt64_t)round(cluster.GetPad() * AliHLTTPCDefinitions::fgkClusterParameterDefinitions[AliHLTTPCDefinitions::kPad].fScale);
  }

  AliHLTUInt64_t time64 = 0;
  if (!isnan(cluster.GetTime())) {
    time64 = (AliHLTUInt64_t)round(cluster.GetTime() * AliHLTTPCDefinitions::fgkClusterParameterDefinitions[AliHLTTPCDefinitions::kTime].fScale);
  }

  AliHLTUInt64_t sigmaPad64 = 0;
  if (!isnan(cluster.GetSigmaPad2())) {
    sigmaPad64 = (AliHLTUInt64_t)round(cluster.GetSigmaPad2() * AliHLTTPCDefinitions::fgkClusterParameterDefinitions[AliHLTTPCDefinitions::kSigmaY2].fScale);
  }

  AliHLTUInt64_t sigmaTime64 = 0;
  if (!isnan(cluster.GetSigmaTime2())) {
    sigmaTime64 = (AliHLTUInt64_t)round(cluster.GetSigmaTime2() * AliHLTTPCDefinitions::fgkClusterParameterDefinitions[AliHLTTPCDefinitions::kSigmaZ2].fScale);
  }

  if (sigmaPad64 >= (unsigned)1 << AliHLTTPCDefinitions::fgkClusterParameterDefinitions[AliHLTTPCDefinitions::kSigmaY2].fBitLength) {
    sigmaPad64 = (1 << AliHLTTPCDefinitions::fgkClusterParameterDefinitions[AliHLTTPCDefinitions::kSigmaY2].fBitLength) - 1;
  }
  if (sigmaTime64 >= (unsigned)1 << AliHLTTPCDefinitions::fgkClusterParameterDefinitions[AliHLTTPCDefinitions::kSigmaZ2].fBitLength) {
    sigmaTime64 = (1 << AliHLTTPCDefinitions::fgkClusterParameterDefinitions[AliHLTTPCDefinitions::kSigmaZ2].fBitLength) - 1;
  }

  AliHLTUInt64_t pad64res = 0;
  pad64res = (AliHLTUInt64_t)round(clusterTrack.fResidualPad * AliHLTTPCDefinitions::fgkClusterParameterDefinitions[AliHLTTPCDefinitions::kPad].fScale);

  AliHLTUInt64_t time64res = 0;
  time64res = (AliHLTUInt64_t)round(clusterTrack.fResidualTime * AliHLTTPCDefinitions::fgkClusterParameterDefinitions[AliHLTTPCDefinitions::kTime].fScale);

  if (fDumpClusters) {
    int dumpVals[16] = {fEvent,
                        (int)is,
                        (int)ip,
                        (int)cluster.GetPadRow(),
                        (int)pad64,
                        (int)time64,
                        (int)sigmaPad64,
                        (int)sigmaTime64,
                        (int)cluster.GetQMax(),
                        (int)cluster.GetCharge(),
                        (int)(cluster.GetFlagEdge() * 4 + cluster.GetFlagSplitPad() * 2 + cluster.GetFlagSplitTime()),
                        (int)clusterTrack.fID,
                        (int)pad64res,
                        (int)time64res,
                        (int)clusterTrack.fAverageQTot,
                        (int)clusterTrack.fAverageQMax};
    fwrite(dumpVals, sizeof(int), 16, fp);
  }

  if (fPrintClustersScaled) {
    HLTImportant("Event %d Slice %d, Patch %d, Row %d, Pad %d, Time %d, SPad %d, STime %d, QMax %d, QTot %d, SplitPad %d, SplitTime %d, Edge %d, TrackID %d, PadRes %d, TimeRes %d AvgTot %d AvgMax %d", fEvent, is, ip, (int)cluster.GetPadRow(), (int)pad64, (int)time64, (int)sigmaPad64,
                 (int)sigmaTime64, (int)cluster.GetQMax(), (int)cluster.GetCharge(), (int)cluster.GetFlagSplitPad(), (int)cluster.GetFlagSplitTime(), (int)cluster.GetFlagEdge(), (int)clusterTrack.fID, (int)pad64res, (int)time64res, (int)clusterTrack.fAverageQTot,
                 (int)clusterTrack.fAverageQMax);
  }
}
