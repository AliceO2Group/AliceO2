// **************************************************************************
// * This file is property of and copyright by the ALICE HLT Project        *
// * All rights reserved.                                                   *
// *                                                                        *
// * Primary Authors:                                                       *
// *     Copyright 2009       Matthias Kretz <kretz@kde.org>                *
// *                                                                        *
// * Permission to use, copy, modify and distribute this software and its   *
// * documentation strictly for non-commercial purposes is hereby granted   *
// * without fee, provided that the above copyright notice appears in all   *
// * copies and that both the copyright notice and this permission notice   *
// * appear in the supporting documentation. The authors make no claims     *
// * about the suitability of this software for any purpose. It is          *
// * provided "as is" without express or implied warranty.                  *
// **************************************************************************

#include "AliGPUCAParam.h"
#include "AliGPUTPCClusterData.h"
#include "AliGPUTPCGPUConfig.h"
#include "AliGPUTPCHit.h"
#include "AliGPUTPCSliceData.h"
#include "AliGPUReconstruction.h"
#include <iostream>
#include <string.h>
#include "cmodules/vecpod.h"

// calculates an approximation for 1/sqrt(x)
// Google for 0x5f3759df :)
static inline float fastInvSqrt(float _x)
{
	// the function calculates fast inverse sqrt
	union
	{
		float f;
		int i;
	} x = {_x};
	const float xhalf = 0.5f * x.f;
	x.i = 0x5f3759df - (x.i >> 1);
	x.f = x.f * (1.5f - xhalf * x.f * x.f);
	return x.f;
}

inline void AliGPUTPCSliceData::CreateGrid(AliGPUTPCRow *row, const float2 *data, int ClusterDataHitNumberOffset)
{
	// grid creation
	if (row->NHits() <= 0)
	{ // no hits or invalid data
		// grid coordinates don't matter, since there are no hits
		row->fGrid.CreateEmpty();
		return;
	}

	float yMin = 1.e3f;
	float yMax = -1.e3f;
	float zMin = 1.e3f;
	float zMax = -1.e3f;
	for (int i = ClusterDataHitNumberOffset; i < ClusterDataHitNumberOffset + row->fNHits; ++i)
	{
		const float y = data[i].x;
		const float z = data[i].y;
		if (yMax < y) yMax = y;
		if (yMin > y) yMin = y;
		if (zMax < z) zMax = z;
		if (zMin > z) zMin = z;
	}

	float dz = zMax - zMin;
	float tfFactor = 1.;
	if (dz > 270.)
	{
		tfFactor = dz / 250.;
		dz = 250.;
	}
	const float norm = fastInvSqrt(row->fNHits / tfFactor);
	row->fGrid.Create(yMin, yMax, zMin, zMax,
	                  CAMath::Max((yMax - yMin) * norm, 2.f),
	                  CAMath::Max(dz * norm, 2.f));
}

inline int AliGPUTPCSliceData::PackHitData(AliGPUTPCRow *const row, const AliGPUTPCHit* binSortedHits)
{
	// hit data packing
	static const float maxVal = (((long long int) 1 << CAMath::Min(24, sizeof(cahit) * 8)) - 1); //Stay within float precision in any case!
	static const float packingConstant = 1.f / (maxVal - 2.);
	const float y0 = row->fGrid.YMin();
	const float z0 = row->fGrid.ZMin();
	const float stepY = (row->fGrid.YMax() - y0) * packingConstant;
	const float stepZ = (row->fGrid.ZMax() - z0) * packingConstant;
	const float stepYi = 1.f / stepY;
	const float stepZi = 1.f / stepZ;

	row->fHy0 = y0;
	row->fHz0 = z0;
	row->fHstepY = stepY;
	row->fHstepZ = stepZ;
	row->fHstepYi = stepYi;
	row->fHstepZi = stepZi;

	for (int hitIndex = 0; hitIndex < row->fNHits; ++hitIndex)
	{
		// bin sorted index!
		const int globalHitIndex = row->fHitNumberOffset + hitIndex;
		const AliGPUTPCHit &hh = binSortedHits[hitIndex];
		const float xx = ((hh.Y() - y0) * stepYi) + .5;
		const float yy = ((hh.Z() - z0) * stepZi) + .5;
		if (xx < 0 || yy < 0 || xx > maxVal || yy > maxVal)
		{
			std::cout << "!!!! hit packing error!!! " << xx << " " << yy << " (" << maxVal << ")" << std::endl;
			return 1;
		}
		// HitData is bin sorted
		fHitData[globalHitIndex].x = (cahit) xx;
		fHitData[globalHitIndex].y = (cahit) yy;
	}
	return 0;
}

void AliGPUTPCSliceData::InitializeRows(const AliGPUCAParam &p)
{
	// initialisation of rows
	for (int i = 0; i < GPUCA_ROW_COUNT + 1; ++i)
	{
		new(&fRows[i]) AliGPUTPCRow;
	}
	for (int i = 0; i < GPUCA_ROW_COUNT; ++i)
	{
		fRows[i].fX = p.RowX[i];
		fRows[i].fMaxY = CAMath::Tan(p.DAlpha / 2.) * fRows[i].fX;
	}
}

void AliGPUTPCSliceData::SetClusterData(const AliGPUTPCClusterData *data)
{
	fClusterData = data;
	int hitMemCount = GPUCA_ROW_COUNT * sizeof(GPUCA_GPU_ROWALIGNMENT) + data->NumberOfClusters();
	const unsigned int kVectorAlignment = 256;
	fNumberOfHitsPlusAlign = AliGPUReconstruction::nextMultipleOf<(kVectorAlignment > sizeof(GPUCA_GPU_ROWALIGNMENT) ? kVectorAlignment : sizeof(GPUCA_GPU_ROWALIGNMENT)) / sizeof(int)>(hitMemCount);
	fNumberOfHits = data->NumberOfClusters();
}

void* AliGPUTPCSliceData::SetPointersInput(void* mem)
{
	const int firstHitInBinSize = (23 + sizeof(GPUCA_GPU_ROWALIGNMENT) / sizeof(int)) * GPUCA_ROW_COUNT + 4 * fNumberOfHits + 3;
	AliGPUReconstruction::computePointerWithAlignment(mem, fHitData, fNumberOfHitsPlusAlign);
	AliGPUReconstruction::computePointerWithAlignment(mem, fFirstHitInBin, firstHitInBinSize);
	return mem;
}

void* AliGPUTPCSliceData::SetPointersScratch(void* mem)
{
	AliGPUReconstruction::computePointerWithAlignment(mem, fLinkUpData, fNumberOfHitsPlusAlign);
	AliGPUReconstruction::computePointerWithAlignment(mem, fLinkDownData, fNumberOfHitsPlusAlign);
	AliGPUReconstruction::computePointerWithAlignment(mem, fHitWeights, fNumberOfHitsPlusAlign);
	return mem;
}

void* AliGPUTPCSliceData::SetPointersScratchHost(void* mem)
{
	AliGPUReconstruction::computePointerWithAlignment(mem, fClusterDataIndex, fNumberOfHitsPlusAlign);
	return mem;
}

void* AliGPUTPCSliceData::SetPointersRows(void* mem)
{
	AliGPUReconstruction::computePointerWithAlignment(mem, fRows, GPUCA_ROW_COUNT + 1);
	return mem;
}

void AliGPUTPCSliceData::RegisterMemoryAllocation()
{
	mMemoryResInput = mRec->RegisterMemoryAllocation(this, &AliGPUTPCSliceData::SetPointersInput, AliGPUMemoryResource::MEMORY_INPUT, "SliceInput");
	mMemoryResScratch = mRec->RegisterMemoryAllocation(this, &AliGPUTPCSliceData::SetPointersScratch, AliGPUMemoryResource::MEMORY_SCRATCH, "SliceLinks");
	mMemoryResScratchHost = mRec->RegisterMemoryAllocation(this, &AliGPUTPCSliceData::SetPointersScratchHost, AliGPUMemoryResource::MEMORY_SCRATCH_HOST, "SliceIds");
	mMemoryResRows = mRec->RegisterMemoryAllocation(this, &AliGPUTPCSliceData::SetPointersRows, AliGPUMemoryResource::MEMORY_PERMANENT, "SliceRows");
}

int AliGPUTPCSliceData::InitFromClusterData()
{
	////////////////////////////////////
	// 0. sort rows
	////////////////////////////////////
	
	fMaxZ = 0.f;

	float2 *YZData = new float2[fNumberOfHits];
	int *tmpHitIndex = new int[fNumberOfHits];

	int RowOffset[GPUCA_ROW_COUNT];
	int NumberOfClustersInRow[GPUCA_ROW_COUNT];
	memset(NumberOfClustersInRow, 0, GPUCA_ROW_COUNT * sizeof(NumberOfClustersInRow[0]));
	fFirstRow = GPUCA_ROW_COUNT;
	fLastRow = 0;

	for (int i = 0; i < fNumberOfHits; i++)
	{
		const int tmpRow = fClusterData->RowNumber(i);
		NumberOfClustersInRow[tmpRow]++;
		if (tmpRow > fLastRow) fLastRow = tmpRow;
		if (tmpRow < fFirstRow) fFirstRow = tmpRow;
	}
	int tmpOffset = 0;
	for (int i = fFirstRow; i <= fLastRow; i++)
	{
		if ((long long int) NumberOfClustersInRow[i] >= ((long long int) 1 << (sizeof(calink) * 8)))
		{
			printf("Too many clusters in row %d for row indexing (%d >= %lld), indexing insufficient\n", i, NumberOfClustersInRow[i], ((long long int) 1 << (sizeof(calink) * 8)));
			return (1);
		}
		if (NumberOfClustersInRow[i] >= (1 << 24))
		{
			printf("Too many clusters in row %d for hit id indexing (%d >= %d), indexing insufficient\n", i, NumberOfClustersInRow[i], 1 << 24);
			return (1);
		}
		RowOffset[i] = tmpOffset;
		tmpOffset += NumberOfClustersInRow[i];
	}

	{
		int RowsFilled[GPUCA_ROW_COUNT];
		memset(RowsFilled, 0, GPUCA_ROW_COUNT * sizeof(int));
		for (int i = 0; i < fNumberOfHits; i++)
		{
			float2 tmp;
			tmp.x = fClusterData->Y(i);
			tmp.y = fClusterData->Z(i);
			if (fabs(tmp.y) > fMaxZ) fMaxZ = fabs(tmp.y);
			int tmpRow = fClusterData->RowNumber(i);
			int newIndex = RowOffset[tmpRow] + (RowsFilled[tmpRow])++;
			YZData[newIndex] = tmp;
			tmpHitIndex[newIndex] = i;
		}
	}
	if (fFirstRow == GPUCA_ROW_COUNT) fFirstRow = 0;

	////////////////////////////////////
	// 2. fill HitData and FirstHitInBin
	////////////////////////////////////

	const int numberOfRows = fLastRow - fFirstRow + 1;
	for (int rowIndex = 0; rowIndex < fFirstRow; ++rowIndex)
	{
		AliGPUTPCRow &row = fRows[rowIndex];
		row.fGrid.CreateEmpty();
		row.fNHits = 0;
		row.fFullSize = 0;
		row.fHitNumberOffset = 0;
		row.fFirstHitInBinOffset = 0;

		row.fHy0 = 0.f;
		row.fHz0 = 0.f;
		row.fHstepY = 1.f;
		row.fHstepZ = 1.f;
		row.fHstepYi = 1.f;
		row.fHstepZi = 1.f;
	}
	for (int rowIndex = fLastRow + 1; rowIndex < GPUCA_ROW_COUNT + 1; ++rowIndex)
	{
		AliGPUTPCRow &row = fRows[rowIndex];
		row.fGrid.CreateEmpty();
		row.fNHits = 0;
		row.fFullSize = 0;
		row.fHitNumberOffset = 0;
		row.fFirstHitInBinOffset = 0;

		row.fHy0 = 0.f;
		row.fHz0 = 0.f;
		row.fHstepY = 1.f;
		row.fHstepZ = 1.f;
		row.fHstepYi = 1.f;
		row.fHstepZi = 1.f;
	}

	vecpod<AliGPUTPCHit> binSortedHits(fNumberOfHits + sizeof(GPUCA_GPU_ROWALIGNMENT));

	int gridContentOffset = 0;
	int hitOffset = 0;

	int binCreationMemorySize = 103 * 2 + fNumberOfHits;
	vecpod<calink> binCreationMemory(binCreationMemorySize);

	for (int rowIndex = fFirstRow; rowIndex <= fLastRow; ++rowIndex)
	{
		AliGPUTPCRow &row = fRows[rowIndex];
		row.fNHits = NumberOfClustersInRow[rowIndex];
		row.fHitNumberOffset = hitOffset;
		hitOffset += AliGPUReconstruction::nextMultipleOf<sizeof(GPUCA_GPU_ROWALIGNMENT) / sizeof(unsigned short)>(NumberOfClustersInRow[rowIndex]);

		row.fFirstHitInBinOffset = gridContentOffset;

		CreateGrid(&row, YZData, RowOffset[rowIndex]);
		const AliGPUTPCGrid &grid = row.fGrid;
		const int numberOfBins = grid.N();
		if ((long long int) numberOfBins >= ((long long int) 1 << (sizeof(calink) * 8)))
		{
			printf("Too many bins in row %d for grid (%d >= %lld), indexing insufficient\n", rowIndex, numberOfBins, ((long long int) 1 << (sizeof(calink) * 8)));
			delete[] YZData;
			delete[] tmpHitIndex;
			return (1);
		}

		int binCreationMemorySizeNew = numberOfBins * 2 + 6 + row.fNHits + sizeof(GPUCA_GPU_ROWALIGNMENT) / sizeof(unsigned short) * numberOfRows + 1;
		if (binCreationMemorySizeNew > binCreationMemorySize)
		{
			binCreationMemorySize = binCreationMemorySizeNew;
			binCreationMemory.resize(binCreationMemorySize);
		}

		calink* c = binCreationMemory.data();          // number of hits in all previous bins
		calink* bins = c + numberOfBins + 3;           // cache for the bin index for every hit in this row, 3 extra empty bins at the end!!!
		calink* filled = bins + row.fNHits;            // counts how many hits there are per bin

		for (unsigned int bin = 0; bin < row.fGrid.N() + 3; ++bin)
		{
			filled[bin] = 0; // initialize filled[] to 0
		}

		for (int hitIndex = 0; hitIndex < row.fNHits; ++hitIndex)
		{
			const int globalHitIndex = RowOffset[rowIndex] + hitIndex;
			const calink bin = row.fGrid.GetBin(YZData[globalHitIndex].x, YZData[globalHitIndex].y);

			bins[hitIndex] = bin;
			++filled[bin];
		}

		calink n = 0;
		for (int bin = 0; bin < numberOfBins + 3; ++bin)
		{
			c[bin] = n;
			n += filled[bin];
		}

		for (int hitIndex = 0; hitIndex < row.fNHits; ++hitIndex)
		{
			const calink bin = bins[hitIndex];
			--filled[bin];
			const calink ind = c[bin] + filled[bin]; // generate an index for this hit that is >= c[bin] and < c[bin + 1]
			const int globalBinsortedIndex = row.fHitNumberOffset + ind;
			const int globalHitIndex = RowOffset[rowIndex] + hitIndex;

			// allows to find the global hit index / coordinates from a global bin sorted hit index
			fClusterDataIndex[globalBinsortedIndex] = tmpHitIndex[globalHitIndex];
			binSortedHits[ind].SetY(YZData[globalHitIndex].x);
			binSortedHits[ind].SetZ(YZData[globalHitIndex].y);
		}

		if (PackHitData(&row, binSortedHits.data()))
		{
			delete[] YZData;
			delete[] tmpHitIndex;
			return (1);
		}

		for (int i = 0; i < numberOfBins; ++i)
		{
			fFirstHitInBin[row.fFirstHitInBinOffset + i] = c[i]; // global bin-sorted hit index
		}
		const calink a = c[numberOfBins];
		// grid.N is <= row.fNHits
		const int nn = numberOfBins + grid.Ny() + 3;
		for (int i = numberOfBins; i < nn; ++i)
		{
			fFirstHitInBin[row.fFirstHitInBinOffset + i] = a;
		}

		row.fFullSize = nn;
		gridContentOffset += nn;

		//Make pointer aligned
		gridContentOffset = AliGPUReconstruction::nextMultipleOf<sizeof(GPUCA_GPU_ROWALIGNMENT) / sizeof(calink)>(gridContentOffset);
	}

	delete[] YZData;
	delete[] tmpHitIndex;

	return (0);
}

void AliGPUTPCSliceData::ClearHitWeights()
{
	// clear hit weights
#ifdef ENABLE_VECTORIZATION
	const int v0(Zero);
	const int *const end = fHitWeights + fNumberOfHits;
	for (int *mem = fHitWeights; mem < end; mem += v0.Size)
	{
		v0.store(mem);
	}
#else
	for (int i = 0; i < fNumberOfHitsPlusAlign; ++i)
	{
		fHitWeights[i] = 0;
	}
#endif
}
