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

#include "AliGPUTPCClusterData.h"
#include "AliTPCCommonMath.h"
#include <algorithm>
#include "AliHLTArray.h"
#include "AliGPUTPCGPUConfig.h"

AliGPUTPCClusterData::~AliGPUTPCClusterData()
{
	if(fAllocated) free(fData);
}

void AliGPUTPCClusterData::StartReading( int sliceIndex, int guessForNumberOfClusters )
{
	// Start reading of event - initialisation
	fSliceIndex = sliceIndex;
	fNumberOfClusters = 0;
	Allocate(CAMath::Max( 64, guessForNumberOfClusters ));
}

template <class T> void AliGPUTPCClusterData::WriteEventVector(const T* const &data, std::ostream &out) const
{
	unsigned i;
	i = fNumberOfClusters;
	out.write((const char*) &i, sizeof(i));
	out.write((const char*) data, i * sizeof(T));
}

template <class T> void AliGPUTPCClusterData::ReadEventVector(T* &data, std::istream &in, int MinSize, bool addData)
{
	int i;
	in.read((char*) &i, sizeof(i));
	int currentClusters = addData ? fNumberOfClusters : 0;
	fNumberOfClusters = currentClusters + i;
	Allocate(CAMath::Max(MinSize, fNumberOfClusters));
	in.read((char*) (data + currentClusters), i * sizeof(T));
}

void AliGPUTPCClusterData::WriteEvent(std::ostream &out) const
{
	WriteEventVector<Data>(fData, out);
}

void AliGPUTPCClusterData::ReadEvent(std::istream &in, bool addData)
{
	ReadEventVector<Data>(fData, in, 64, addData);
}

void AliGPUTPCClusterData::Allocate(int number)
{
	int newnumber;
	if (fAllocated)
	{
		if (number < fAllocated) return;
		newnumber = CAMath::Max(number, 2 * fAllocated);
		fData = (Data*) realloc(fData, newnumber * sizeof(Data));
	}
	else
	{
		fData = (Data*) malloc(number * sizeof(Data));
		newnumber = number;
	}
	fAllocated = newnumber;
}

void AliGPUTPCClusterData::SetClusterData(int sl, int n, const Data* d)
{
	if (fAllocated) free(fData);
	fAllocated = 0;
	fSliceIndex = sl;
	fNumberOfClusters = n;
	fData = (Data*) d;
}
