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
#include "Generators/TriggerTGenerator.h"
#include "SimulationDataFormat/GeneratorHeader.h"
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

GeneratorTGenerator::GeneratorTGenerator()
  : Generator("ALICEo2", "ALICEo2 TGenerator Generator"), mGenerator(nullptr), mParticles(nullptr), mImportParticles("Final")
{
  /** default constructor **/
}

/*****************************************************************/

GeneratorTGenerator::GeneratorTGenerator(const Char_t* name, const Char_t* title)
  : Generator(name, title), mGenerator(nullptr), mParticles(nullptr), mImportParticles("Final")
{
  /** constructor **/
}

/*****************************************************************/

GeneratorTGenerator::~GeneratorTGenerator()
{
  /** default destructor **/

  if (mParticles)
    delete mParticles;
}

/*****************************************************************/

Bool_t GeneratorTGenerator::generateEvent()
{
  /** generate event **/

  mGenerator->GenerateEvent();
  mGenerator->ImportParticles(mParticles, mImportParticles.c_str());

  /** success **/
  return kTRUE;
}

/*****************************************************************/

Bool_t GeneratorTGenerator::boostEvent(Double_t boost)
{
  /** boost event **/

  LOG(WARNING) << "Boost not implemented yet" << std::endl;
  return kTRUE;
}

/*****************************************************************/

Bool_t GeneratorTGenerator::triggerFired(Trigger* trigger) const
{
  /** trigger event **/

  auto aTrigger = dynamic_cast<TriggerTGenerator*>(trigger);
  if (!aTrigger) {
    LOG(ERROR) << "Incompatile trigger for HepMC interface" << std::endl;
    return kFALSE;
  }
  return aTrigger->triggerEvent(mParticles, mGenerator);
}

/*****************************************************************/

Bool_t GeneratorTGenerator::addTracks(FairPrimaryGenerator* primGen) const
{
  /** add tracks **/

  /* loop over particles */
  Int_t nParticles = mParticles->GetEntries();
  TParticle* particle = nullptr;
  for (Int_t iparticle = 0; iparticle < nParticles; iparticle++) {
    particle = (TParticle*)mParticles->At(iparticle);
    if (!particle)
      continue;
    primGen->AddTrack(particle->GetPdgCode(), particle->Px(), particle->Py(), particle->Pz(), particle->Vx(),
                      particle->Vy(), particle->Vz(), particle->GetMother(0), particle->GetStatusCode() == 1,
                      particle->Energy(), particle->T(), particle->GetWeight());
  }

  /** success **/
  return kTRUE;
}

/*****************************************************************/

Bool_t GeneratorTGenerator::addHeader(PrimaryGenerator* primGen) const
{
  /** add header **/

  /** success **/
  return Generator::addHeader(primGen);
}

/*****************************************************************/

Bool_t GeneratorTGenerator::Init()
{
  /** init **/

  if (!mGenerator) {
    LOG(ERROR) << "No TGenerator inteface assigned" << std::endl;
    return kFALSE;
  }

  /** array of generated particles **/
  mParticles = new TClonesArray("TParticle");
  mParticles->SetOwner(kTRUE);

  /** success **/
  return kTRUE;
}

/*****************************************************************/
/*****************************************************************/

} /* namespace eventgen */
} /* namespace o2 */
