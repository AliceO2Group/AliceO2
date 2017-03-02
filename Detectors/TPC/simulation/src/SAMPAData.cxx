/// \file SAMPAData.cxx
/// \author Sebastian Klewin

#include "TPCSimulation/SAMPAData.h"

using namespace AliceO2::TPC;

SAMPAData::SAMPAData()
  : SAMPAData(-1)
{}

SAMPAData::SAMPAData(int id)
  : mID(id)
  , mData(32,0)
{
}

SAMPAData::SAMPAData(int id, std::vector<int>* data)
  : mID(id)
{
  if (data->size() != 32)
    LOG(ERROR) << "Vector does not contain 32 elements." << FairLogger::endl;

  mData = *data;
}

SAMPAData::~SAMPAData()
{}

std::ostream& SAMPAData::Print(std::ostream& output) const
{
  for (int i = 0; i < 32; ++i)
  {
    output << "Channel " << i << ": " << mData[i] << std::endl;
  }
  return output;
}
