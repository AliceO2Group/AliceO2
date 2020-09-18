// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "Generators/GeneratorTGenerator.h"
#include "FairLogger.h"
#include "FairPrimaryGenerator.h"
#include "TGenerator.h"
#include "TClonesArray.h"
#include "TParticle.h"

namespace o2
{
namespace eventgen
{

/*****************************************************************/
/*****************************************************************/

GeneratorTGenerator::GeneratorTGenerator() : Generator("ALICEo2", "ALICEo2 TGenerator Generator"),
                                             mTGenerator(nullptr),
                                             mCloneParticles(nullptr)
{
  /** default constructor **/

  mInterface = reinterpret_cast<void*>(mTGenerator);
  mInterfaceName = "tgenerator";
  mCloneParticles = new TClonesArray("TParticle");
  mCloneParticles->SetOwner(kTRUE);
}

/*****************************************************************/

GeneratorTGenerator::GeneratorTGenerator(const Char_t* name, const Char_t* title) : Generator(name, title),
                                                                                    mTGenerator(nullptr),
                                                                                    mCloneParticles(nullptr)
{
  /** constructor **/

  mInterface = reinterpret_cast<void*>(mTGenerator);
  mInterfaceName = "tgenerator";
  mCloneParticles = new TClonesArray("TParticle");
  mCloneParticles->SetOwner(kTRUE);
}

/*****************************************************************/

GeneratorTGenerator::~GeneratorTGenerator()
{
  /** destructor **/

  delete mCloneParticles;
}

/*****************************************************************/

Bool_t
  GeneratorTGenerator::generateEvent()
{
  /** generate event **/

  mTGenerator->GenerateEvent();

  /** success **/
  return kTRUE;
}

/*****************************************************************/

Bool_t
  GeneratorTGenerator::importParticles()
{
  /** import particles **/

  mTGenerator->ImportParticles(mCloneParticles, "All");
  auto nparticles = mCloneParticles->GetEntries();
  for (Int_t iparticle = 0; iparticle < nparticles; iparticle++) {
    auto particle = (TParticle*)mCloneParticles->At(iparticle);
    mParticles.push_back(*particle);
  }

  /** success **/
  return kTRUE;
}

/*****************************************************************/

Bool_t
  GeneratorTGenerator::Init()
{
  /** init **/

  /** init base class **/
  Generator::Init();

  if (!mTGenerator) {
    LOG(FATAL) << "No TGenerator inteface assigned" << std::endl;
    return kFALSE;
  }

  /** success **/
  return kTRUE;
}

/*****************************************************************/
/*****************************************************************/

} /* namespace eventgen */
} /* namespace o2 */
