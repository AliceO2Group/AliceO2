#include "AliGPUConstantMem.h"
#include "AliGPUTPCSettings.h"
#include "GPUITSFitterKernels.h"

#include "ITStracking/Constants.h"
#include "ITStracking/MathUtils.h"
#include "ITStracking/Road.h"
#include "ITStracking/Cluster.h"
#include "ITStracking/Cell.h"
#include "CommonConstants/MathConstants.h"

using namespace o2::ITS;
using namespace o2;

GPUd() bool GPUITSFitterKernel::fitTrack(GPUITSFitter& Fitter, AliGPUTPCGMPropagator& prop, GPUITSTrack& track, int start, int end, int step)
{
  for (int iLayer{ start }; iLayer != end; iLayer += step) {
    if (track.mClusters[iLayer] == Constants::ITS::UnusedIndex) {
      continue;
    }
    const TrackingFrameInfo& trackingHit = Fitter.trackingFrame()[iLayer][track.mClusters[iLayer]];

	if (prop.PropagateToXAlpha(trackingHit.xTrackingFrame, trackingHit.alphaTrackingFrame, step > 0))
	  return false;

	if (prop.Update(trackingHit.positionTrackingFrame[0], trackingHit.positionTrackingFrame[1], 0, false, trackingHit.covarianceTrackingFrame[0], trackingHit.covarianceTrackingFrame[2]))
      return false;

    /*const float xx0 = (iLayer > 2) ? 0.008f : 0.003f; // Rough layer thickness //FIXME
    constexpr float radiationLength = 9.36f;          // Radiation length of Si [cm]
    constexpr float density = 2.33f;                  // Density of Si [g/cm^3]
    if (!track.correctForMaterial(xx0, xx0 * radiationLength * density, true))
      return false;*/
  }
  return true;
}

template <>
GPUd() void GPUITSFitterKernel::Thread<0>(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() AliGPUTPCSharedMemory &smem, workerType &workers)
{
	GPUITSFitter &Fitter = workers.itsFitter;
	AliGPUTPCGMMerger &Merger = workers.tpcMerger;

	AliGPUTPCGMPropagator prop;
	prop.SetPolynomialField(Merger.pField());
	prop.SetMaxSinPhi(GPUCA_MAX_SIN_PHI);
	prop.SetToyMCEventsFlag(0);
	prop.SetFitInProjections(1);
	float bz = -5.f; //FIXME

	for (int iRoad = get_global_id(0); iRoad < Fitter.NumberOfRoads(); iRoad += get_global_size(0))
	{
		Road& road = Fitter.roads()[iRoad];
		int clusters[7] = {Constants::ITS::UnusedIndex, Constants::ITS::UnusedIndex, Constants::ITS::UnusedIndex, Constants::ITS::UnusedIndex, Constants::ITS::UnusedIndex, Constants::ITS::UnusedIndex, Constants::ITS::UnusedIndex};
		int lastCellLevel = Constants::ITS::UnusedIndex;
		CA_DEBUGGER(int nClusters = 2);

		for (int iCell{0}; iCell < Constants::ITS::CellsPerRoad; ++iCell)
		{
			const int cellIndex = road[iCell];
			if (cellIndex == Constants::ITS::UnusedIndex)
			{
				continue;
			}
			else
			{
				clusters[iCell] = Fitter.cells()[iCell][cellIndex].getFirstClusterIndex();
				clusters[iCell + 1] = Fitter.cells()[iCell][cellIndex].getSecondClusterIndex();
				clusters[iCell + 2] = Fitter.cells()[iCell][cellIndex].getThirdClusterIndex();
				lastCellLevel = iCell;
				CA_DEBUGGER(nClusters++);
			}
		}

		CA_DEBUGGER(roadCounters[nClusters - 4]++);

		if (lastCellLevel == Constants::ITS::UnusedIndex)
			continue;

		/// From primary vertex context index to event index (== the one used as input of the tracking code)
		for (int iC{0}; iC < 7; iC++)
		{
			if (clusters[iC] != Constants::ITS::UnusedIndex)
			{
				clusters[iC] = Fitter.clusters()[iC][clusters[iC]].clusterId;
			}
		}
		/// Track seed preparation. Clusters are numbered progressively from the outermost to the innermost.
		const auto &cluster1 = Fitter.trackingFrame()[lastCellLevel + 2][clusters[lastCellLevel + 2]];
		const auto &cluster2 = Fitter.trackingFrame()[lastCellLevel + 1][clusters[lastCellLevel + 1]];
		const auto &cluster3 = Fitter.trackingFrame()[lastCellLevel][clusters[lastCellLevel]];

		GPUITSTrack temporaryTrack;
		{
			const float ca = std::cos(cluster3.alphaTrackingFrame), sa = std::sin(cluster3.alphaTrackingFrame);
			const float x1 = cluster1.xCoordinate * ca + cluster1.yCoordinate * sa;
			const float y1 = -cluster1.xCoordinate * sa + cluster1.yCoordinate * ca;
			const float z1 = cluster1.zCoordinate;
			const float x2 = cluster2.xCoordinate * ca + cluster2.yCoordinate * sa;
			const float y2 = -cluster2.xCoordinate * sa + cluster2.yCoordinate * ca;
			const float z2 = cluster2.zCoordinate;
			const float x3 = cluster3.xTrackingFrame;
			const float y3 = cluster3.positionTrackingFrame[0];
			const float z3 = cluster3.positionTrackingFrame[1];

			const float crv = MathUtils::computeCurvature(x1, y1, x2, y2, x3, y3);
			const float x0 = MathUtils::computeCurvatureCentreX(x1, y1, x2, y2, x3, y3);
			const float tgl12 = MathUtils::computeTanDipAngle(x1, y1, x2, y2, z1, z2);
			const float tgl23 = MathUtils::computeTanDipAngle(x2, y2, x3, y3, z2, z3);

			const float r2 = CAMath::Sqrt(cluster2.xCoordinate * cluster2.xCoordinate + cluster2.yCoordinate * cluster2.yCoordinate);
			const float r3 = CAMath::Sqrt(cluster3.xCoordinate * cluster3.xCoordinate + cluster3.yCoordinate * cluster3.yCoordinate);
			const float fy = 1. / (r2 - r3);
			const float &tz = fy;
			const float cy = (MathUtils::computeCurvature(x1, y1, x2, y2 + Constants::ITS::Resolution, x3, y3) - crv) /
			                 (Constants::ITS::Resolution * bz * constants::math::B2C) *
			                 20.f; // FIXME: MS contribution to the cov[14] (*20 added)
			constexpr float s2 = Constants::ITS::Resolution * Constants::ITS::Resolution;

			temporaryTrack.X() = cluster3.xTrackingFrame;
			temporaryTrack.Y() = y3;
			temporaryTrack.Z() = z3;
			temporaryTrack.SinPhi() = crv * (x3 - x0);
			temporaryTrack.DzDs() = 0.5f * (tgl12 + tgl23);
			temporaryTrack.QPt() = CAMath::Abs(bz) < constants::math::Almost0 ? constants::math::Almost0 : crv / (bz * constants::math::B2C);
			temporaryTrack.ZOffset() = 0;
			temporaryTrack.Cov()[0] = s2;
			temporaryTrack.Cov()[1] = 0.f;
			temporaryTrack.Cov()[2] = s2;
			temporaryTrack.Cov()[3] = s2 * fy;
			temporaryTrack.Cov()[4] = 0.f;
			temporaryTrack.Cov()[5] = s2 * fy * fy;
			temporaryTrack.Cov()[6] = 0.f;
			temporaryTrack.Cov()[7] = s2 * tz;
			temporaryTrack.Cov()[8] = 0.f;
			temporaryTrack.Cov()[9] = s2 * tz * tz;
			temporaryTrack.Cov()[10] = s2 * cy;
			temporaryTrack.Cov()[11] = 0.f;
			temporaryTrack.Cov()[12] = s2 * fy * cy;
			temporaryTrack.Cov()[13] = 0.f;
			temporaryTrack.Cov()[14] = s2 * cy * cy;
			temporaryTrack.SetChi2(0);
			temporaryTrack.SetNDF(-5);
			
			prop.SetTrack(&temporaryTrack, cluster3.alphaTrackingFrame);
		}

		for (size_t iC = 0; iC < 7; ++iC)
		{
			temporaryTrack.mClusters[iC] = clusters[iC];
		}
		bool fitSuccess = fitTrack(Fitter, prop, temporaryTrack, Constants::ITS::LayersNumber - 4, -1, -1);
		if (!fitSuccess)
			continue;
		CA_DEBUGGER(fitCounters[nClusters - 4]++);
		temporaryTrack.ResetCovariance();
		fitSuccess = fitTrack(Fitter, prop, temporaryTrack, 0, Constants::ITS::LayersNumber, 1);
		if (!fitSuccess)
			continue;
		CA_DEBUGGER(backpropagatedCounters[nClusters - 4]++);
		for (int k = 0;k < 5;k++)
		{
			temporaryTrack.mOuterParam.fP[k] = temporaryTrack.Par()[k];
		}
		for (int k = 0;k < 15;k++)
		{
			temporaryTrack.mOuterParam.fC[k] = temporaryTrack.Cov()[k];
		}
		temporaryTrack.mOuterParam.fX = temporaryTrack.X();
		temporaryTrack.mOuterParam.fAlpha = prop.GetAlpha();
		temporaryTrack.ResetCovariance();
		fitSuccess = fitTrack(Fitter, prop, temporaryTrack, Constants::ITS::LayersNumber - 1, -1, -1);
		if (!fitSuccess)
			continue;
		CA_DEBUGGER(refitCounters[nClusters - 4]++);
		int trackId = CAMath::AtomicAdd(&Fitter.NumberOfTracks(), 1);
		Fitter.tracks()[trackId] = temporaryTrack;
	}
}
