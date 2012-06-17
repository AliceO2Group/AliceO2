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

void AliHLTTPCCAClusterData::StartReading( int sliceIndex, int guessForNumberOfClusters )
{
  // Start reading of event - initialisation

  fSliceIndex = sliceIndex;
  fData.clear();
  fData.reserve( CAMath::Max( 64, guessForNumberOfClusters ) );
}


void AliHLTTPCCAClusterData::FinishReading()
{

}

template <class T> void AliHLTTPCCAClusterData::WriteEventVector(const std::vector<T> &data, std::ostream &out) const
{
	AliHLTResizableArray<T> tmpData(data.size());
	unsigned i;
	for (i = 0;i < data.size();i++)
	{
		tmpData[i] = data[i];
	}
	i = data.size();
	out.write((char*) &i, sizeof(i));
	out.write((char*) &tmpData[0], i * sizeof(T));
}

template <class T> void AliHLTTPCCAClusterData::ReadEventVector(std::vector<T> &data, std::istream &in, int MinSize)
{
	int i;
	in.read((char*) &i, sizeof(i));
	data.reserve(AliHLTTPCCAMath::Max(MinSize, i));
	data.resize(i);
	AliHLTResizableArray<T> tmpData(i);
	in.read((char*) &tmpData[0], i * sizeof(T));
	for (int j = 0;j < i;j++)
	{
#ifdef HLTCA_STANDALONE
		if (tmpData[j].fRow < 0 || tmpData[j].fRow >= HLTCA_ROW_COUNT)
		{
			exit(1);
		}
#endif
		data[j] = tmpData[j];
	}
}

void AliHLTTPCCAClusterData::WriteEvent(std::ostream &out) const
{
	WriteEventVector<Data>(fData, out);
}

void AliHLTTPCCAClusterData::ReadEvent(std::istream &in)
{
    fData.clear();
	ReadEventVector<Data>(fData, in, 64);
}

