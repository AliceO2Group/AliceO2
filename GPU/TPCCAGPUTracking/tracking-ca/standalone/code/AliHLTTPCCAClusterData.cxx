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

#include "AliHLTTPCCAClusterData.h"
#include "AliHLTTPCCAMath.h"
#include <algorithm>
#include "AliHLTArray.h"
#include "AliHLTTPCCAGPUConfig.h"
#include <stdio.h>

AliHLTTPCCAClusterData::~AliHLTTPCCAClusterData()
{
	if(fAllocated) free(fData);
}

void AliHLTTPCCAClusterData::StartReading( int sliceIndex, int guessForNumberOfClusters )
{
  // Start reading of event - initialisation
  fSliceIndex = sliceIndex;
  fNumberOfClusters = 0;
  Allocate(CAMath::Max( 64, guessForNumberOfClusters ));
}

template <class T> void AliHLTTPCCAClusterData::WriteEventVector(const T* const &data, std::ostream &out) const
{
	unsigned i;
	i = fNumberOfClusters;
	out.write((char*) &i, sizeof(i));
	out.write((char*) data, i * sizeof(T));
}

template <class T> void AliHLTTPCCAClusterData::ReadEventVector(T* &data, std::istream &in, int MinSize)
{
	int i;
	in.read((char*) &i, sizeof(i));
	fNumberOfClusters = i;
	Allocate(CAMath::Max(MinSize, fNumberOfClusters));
	in.read((char*) data, i * sizeof(T));
}

void AliHLTTPCCAClusterData::WriteEvent(std::ostream &out) const
{
	WriteEventVector<Data>(fData, out);
}

void AliHLTTPCCAClusterData::ReadEvent(std::istream &in)
{
	ReadEventVector<Data>(fData, in, 64);
}

void AliHLTTPCCAClusterData::Allocate(int number)
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
	if (fData == NULL)
	{
		fprintf(stderr, "Memory allocation error\n");
		exit(1);
	}
	fAllocated = newnumber;
}
