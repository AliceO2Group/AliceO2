// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \author R+Preghenella - August 2017

#include "SimulationDataFormat/MCEventHeader.h"
#include "SimulationDataFormat/GeneratorHeader.h"
#include "FairRootManager.h"

namespace o2
{
namespace dataformats
{

/*****************************************************************/
/*****************************************************************/

MCEventHeader::MCEventHeader()
  : FairMCEventHeader(), mGeneratorHeaders(), mEmbeddingFileName(), mEmbeddingEventCounter(-1)
{
  /** default constructor **/
}

/*****************************************************************/

MCEventHeader::MCEventHeader(const MCEventHeader& rhs)
  : FairMCEventHeader(rhs),
    mGeneratorHeaders(rhs.mGeneratorHeaders),
    mEmbeddingFileName(rhs.mEmbeddingFileName),
    mEmbeddingEventCounter(rhs.mEmbeddingEventCounter)
{
  /** copy constructor **/
}

/*****************************************************************/

MCEventHeader& MCEventHeader::operator=(const MCEventHeader& rhs)
{
  /** operator= **/

  if (this == &rhs)
    return *this;
  FairMCEventHeader::operator=(rhs);
  mGeneratorHeaders = rhs.mGeneratorHeaders;
  mEmbeddingFileName = rhs.mEmbeddingFileName;
  mEmbeddingEventCounter = rhs.mEmbeddingEventCounter;
  return *this;
}

/*****************************************************************/

MCEventHeader::~MCEventHeader() { /** default destructor **/}

/*****************************************************************/

void MCEventHeader::Reset()
{
  /** reset **/

  mGeneratorHeaders.clear();
  mEmbeddingFileName = "";
  mEmbeddingEventCounter = -1;
  FairMCEventHeader::Reset();
}

/*****************************************************************/

void MCEventHeader::Print(Option_t* opt) const
{
  /** print **/

  auto eventId = GetEventID();
  std::cout << "> event-id: " << eventId << " | xyz: (" << GetX() << ", " << GetY() << ", " << GetZ() << ")"
            << " | N.primaries: " << GetNPrim() << std::endl;
  for (auto const& header : mGeneratorHeaders)
    header->print();
}

/*****************************************************************/

void MCEventHeader::addHeader(GeneratorHeader* header)
{
  /** add header **/

  mGeneratorHeaders.push_back(new GeneratorHeader(*header));
}

/*****************************************************************/
/*****************************************************************/

} /* namespace dataformats */
} /* namespace o2 */

ClassImp(o2::dataformats::MCEventHeader)
