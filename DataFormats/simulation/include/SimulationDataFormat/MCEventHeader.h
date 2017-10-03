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

#ifndef ALICEO2_DATAFORMATS_MCEVENTHEADER_H_
#define ALICEO2_DATAFORMATS_MCEVENTHEADER_H_

#include "FairMCEventHeader.h"
#include <vector>

namespace o2
{
namespace dataformats
{

class GeneratorHeader;

/*****************************************************************/
/*****************************************************************/

class MCEventHeader : public FairMCEventHeader
{

 public:
  /** default constructor **/
  MCEventHeader();
  /** copy constructor **/
  MCEventHeader(const MCEventHeader& rhs);
  /** operator= **/
  MCEventHeader& operator=(const MCEventHeader& rhs);
  /** destructor **/
  ~MCEventHeader() override;

  /** getters **/
  const std::vector<GeneratorHeader*>& GeneratorHeaders() const { return mGeneratorHeaders; };

  /** setters **/
  void setEmbeddingFileName(TString value) { mEmbeddingFileName = value; };
  void setEmbeddingEventCounter(Int_t value) { mEmbeddingEventCounter = value; };

  /** methods **/
  void Print(Option_t* opt = "") const override;
  virtual void Reset();
  virtual void addHeader(GeneratorHeader* header);

 protected:
  std::vector<GeneratorHeader*> mGeneratorHeaders;
  TString mEmbeddingFileName;
  Int_t mEmbeddingEventCounter;

  ClassDefOverride(MCEventHeader, 1);

}; /** class MCEventHeader **/

/*****************************************************************/
/*****************************************************************/

} /* namespace dataformats */
} /* namespace o2 */

#endif /* ALICEO2_DATAFORMATS_MCEVENTHEADER_H_ */
