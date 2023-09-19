// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \author R+Preghenella - August 2017

#include "SimulationDataFormat/MCUtils.h"
#include "Generators/GeneratorHepMC.h"
#include "Generators/GeneratorHepMCParam.h"
#include "SimulationDataFormat/MCEventHeader.h"
#include "HepMC3/ReaderAscii.h"
#include "HepMC3/ReaderAsciiHepMC2.h"
#include "HepMC3/GenEvent.h"
#include "HepMC3/GenParticle.h"
#include "HepMC3/GenVertex.h"
#include "HepMC3/FourVector.h"
#include "HepMC3/Version.h"
#include "TParticle.h"
#include "TSystem.h"

#include <cstdlib>
#include <sys/types.h> // POSIX only
#include <sys/stat.h>  // POISX only
#include <cstdio>

#include <fairlogger/Logger.h>
#include "FairPrimaryGenerator.h"
#include <cmath>

namespace o2
{
namespace eventgen
{

/*****************************************************************/
/*****************************************************************/

GeneratorHepMC::GeneratorHepMC()
  : Generator("ALICEo2", "ALICEo2 HepMC Generator"), mStream(), mFileName(), mVersion(3), mReader(nullptr), mEvent(nullptr)
{
  /** default constructor **/

  mEvent = new HepMC3::GenEvent();
  mInterface = reinterpret_cast<void*>(mEvent);
  mInterfaceName = "hepmc";
}

/*****************************************************************/

GeneratorHepMC::GeneratorHepMC(const Char_t* name, const Char_t* title)
  : Generator(name, title), mStream(), mFileName(), mVersion(3), mReader(nullptr), mEvent(nullptr)
{
  /** constructor **/

  mEvent = new HepMC3::GenEvent();
  mInterface = reinterpret_cast<void*>(mEvent);
  mInterfaceName = "hepmc";
}

/*****************************************************************/

GeneratorHepMC::~GeneratorHepMC()
{
  /** default destructor **/

  if (mStream.is_open()) {
    mStream.close();
  }
  if (mReader) {
    mReader->close();
    delete mReader;
  }
  if (mEvent) {
    delete mEvent;
  }
}

/*****************************************************************/

Bool_t GeneratorHepMC::generateEvent()
{
  /** generate event **/

  /** clear and read event **/
  mEvent->clear();
  mReader->read_event(*mEvent);
  if (mReader->failed()) {
    LOG(error) << "Failed to read one event from input";
    return kFALSE;
  }
  /** set units to desired output **/
  mEvent->set_units(HepMC3::Units::GEV, HepMC3::Units::MM);

  /** success **/
  return kTRUE;
}

/*****************************************************************/

Bool_t GeneratorHepMC::importParticles()
{
  /** import particles **/

  /** loop over particles **/
  auto particles = mEvent->particles();
  for (int i = 0; i < particles.size(); ++i) {

    /** get particle information **/
    auto particle = particles.at(i);
    auto pdg = particle->pid();
    auto st = particle->status();
    auto momentum = particle->momentum();
    auto vertex = particle->production_vertex()->position();
    auto parents = particle->parents();   // less efficient than via vertex
    auto children = particle->children(); // less efficient than via vertex

    /** get momentum information **/
    auto px = momentum.x();
    auto py = momentum.y();
    auto pz = momentum.z();
    auto et = momentum.t();

    /** get vertex information **/
    auto vx = vertex.x();
    auto vy = vertex.y();
    auto vz = vertex.z();
    auto vt = vertex.t();

    /** get mother information **/
    auto m1 = parents.empty() ? -1 : parents.front()->id() - 1;
    auto m2 = parents.empty() ? -1 : parents.back()->id() - 1;

    /** get daughter information **/
    auto d1 = children.empty() ? -1 : children.front()->id() - 1;
    auto d2 = children.empty() ? -1 : children.back()->id() - 1;

    /** add to particle vector **/
    mParticles.push_back(TParticle(pdg, st, m1, m2, d1, d2, px, py, pz, et, vx, vy, vz, vt));
    o2::mcutils::MCGenHelper::encodeParticleStatusAndTracking(mParticles.back(), st == 1);

  } /** end of loop over particles **/

  /** success **/
  return kTRUE;
}

namespace
{
void putAttributeInfo(o2::dataformats::MCEventHeader* eventHeader,
                      const std::string& name,
                      const std::shared_ptr<HepMC3::Attribute>& a)
{
  if (auto* p = dynamic_cast<HepMC3::IntAttribute*>(a.get()))
    eventHeader->putInfo<int>(name, p->value());
  if (auto* p = dynamic_cast<HepMC3::LongAttribute*>(a.get()))
    eventHeader->putInfo<int>(name, p->value());
  if (auto* p = dynamic_cast<HepMC3::FloatAttribute*>(a.get()))
    eventHeader->putInfo<float>(name, p->value());
  if (auto* p = dynamic_cast<HepMC3::DoubleAttribute*>(a.get()))
    eventHeader->putInfo<float>(name, p->value());
  if (auto* p = dynamic_cast<HepMC3::StringAttribute*>(a.get()))
    eventHeader->putInfo<std::string>(name, p->value());
  if (auto* p = dynamic_cast<HepMC3::CharAttribute*>(a.get()))
    eventHeader->putInfo<char>(name, p->value());
  if (auto* p = dynamic_cast<HepMC3::LongLongAttribute*>(a.get()))
    eventHeader->putInfo<int>(name, p->value());
  if (auto* p = dynamic_cast<HepMC3::LongDoubleAttribute*>(a.get()))
    eventHeader->putInfo<float>(name, p->value());
  if (auto* p = dynamic_cast<HepMC3::UIntAttribute*>(a.get()))
    eventHeader->putInfo<int>(name, p->value());
  if (auto* p = dynamic_cast<HepMC3::ULongAttribute*>(a.get()))
    eventHeader->putInfo<int>(name, p->value());
  if (auto* p = dynamic_cast<HepMC3::ULongLongAttribute*>(a.get()))
    eventHeader->putInfo<int>(name, p->value());
  if (auto* p = dynamic_cast<HepMC3::BoolAttribute*>(a.get()))
    eventHeader->putInfo<bool>(name, p->value());
}
} // namespace

/*****************************************************************/

void GeneratorHepMC::updateHeader(o2::dataformats::MCEventHeader* eventHeader)
{
  /** update header **/
  using Key = o2::dataformats::MCInfoKeys;

  eventHeader->putInfo<std::string>(Key::generator, "hepmc");
  eventHeader->putInfo<int>(Key::generatorVersion, HEPMC3_VERSION_CODE);

  auto xSection = mEvent->cross_section();
  auto pdfInfo = mEvent->pdf_info();
  auto hiInfo = mEvent->heavy_ion();

  // Set default cross-section
  if (xSection) {
    eventHeader->putInfo<float>(Key::xSection, xSection->xsec());
    eventHeader->putInfo<float>(Key::xSectionError, xSection->xsec_err());
    eventHeader->putInfo<int>(Key::acceptedEvents,
                              xSection->get_accepted_events());
    eventHeader->putInfo<int>(Key::attemptedEvents,
                              xSection->get_attempted_events());
  }

  // Set weights and cross sections
  size_t iw = 0;
  for (auto w : mEvent->weights()) {
    std::string post = (iw > 0 ? "_" + std::to_string(iw) : "");
    eventHeader->putInfo<float>(Key::weight + post, w);
    if (xSection) {
      eventHeader->putInfo<float>(Key::xSection, xSection->xsec(iw));
      eventHeader->putInfo<float>(Key::xSectionError, xSection->xsec_err(iw));
    }
    iw++;
  }

  // Set the PDF information
  if (pdfInfo) {
    eventHeader->putInfo<int>(Key::pdfParton1Id, pdfInfo->parton_id[0]);
    eventHeader->putInfo<int>(Key::pdfParton2Id, pdfInfo->parton_id[1]);
    eventHeader->putInfo<float>(Key::pdfX1, pdfInfo->x[0]);
    eventHeader->putInfo<float>(Key::pdfX2, pdfInfo->x[1]);
    eventHeader->putInfo<float>(Key::pdfScale, pdfInfo->scale);
    eventHeader->putInfo<float>(Key::pdfXF1, pdfInfo->xf[0]);
    eventHeader->putInfo<float>(Key::pdfXF2, pdfInfo->xf[1]);
    eventHeader->putInfo<float>(Key::pdfCode1, pdfInfo->pdf_id[0]);
    eventHeader->putInfo<float>(Key::pdfCode2, pdfInfo->pdf_id[1]);
  }

  // Set heavy-ion information
  if (hiInfo) {
    eventHeader->putInfo<int>(Key::impactParameter,
                              hiInfo->impact_parameter);
    eventHeader->putInfo<int>(Key::nPart,
                              hiInfo->Npart_proj + hiInfo->Npart_targ);
    eventHeader->putInfo<int>(Key::nPartProjectile, hiInfo->Npart_proj);
    eventHeader->putInfo<int>(Key::nPartTarget, hiInfo->Npart_targ);
    eventHeader->putInfo<int>(Key::nColl, hiInfo->Ncoll);
    eventHeader->putInfo<int>(Key::nCollHard, hiInfo->Ncoll_hard);
    eventHeader->putInfo<int>(Key::nCollNNWounded,
                              hiInfo->N_Nwounded_collisions);
    eventHeader->putInfo<int>(Key::nCollNWoundedN,
                              hiInfo->Nwounded_N_collisions);
    eventHeader->putInfo<int>(Key::nCollNWoundedNwounded,
                              hiInfo->Nwounded_Nwounded_collisions);
    eventHeader->putInfo<int>(Key::planeAngle,
                              hiInfo->event_plane_angle);
    eventHeader->putInfo<int>(Key::sigmaInelNN,
                              hiInfo->sigma_inel_NN);
    eventHeader->putInfo<int>(Key::centrality, hiInfo->centrality);
    eventHeader->putInfo<int>(Key::nSpecProjectileProton, hiInfo->Nspec_proj_p);
    eventHeader->putInfo<int>(Key::nSpecProjectileNeutron, hiInfo->Nspec_proj_n);
    eventHeader->putInfo<int>(Key::nSpecTargetProton, hiInfo->Nspec_targ_p);
    eventHeader->putInfo<int>(Key::nSpecTargetNeutron, hiInfo->Nspec_targ_n);
  }

  for (auto na : mEvent->attributes()) {
    std::string name = na.first;
    if (name == "GenPdfInfo" ||
        name == "GenCrossSection" ||
        name == "GenHeavyIon")
      continue;

    for (auto ia : na.second) {
      int no = ia.first;
      auto at = ia.second;
      std::string post = (no == 0 ? "" : std::to_string(no));

      putAttributeInfo(eventHeader, name + post, at);
    }
  }
}

/*****************************************************************/

Bool_t GeneratorHepMC::Init()
{
  /** init **/

  /** init base class **/
  Generator::Init();

  std::string filename = gSystem->ExpandPathName(mFileName.c_str());

  // If a EG command line is given, then we make a fifo on a temporary
  // file, and directs the EG to write to that fifo.  We will then set
  // up the HepMC3 reader to read from that fifo.
  //
  //    o2-sim -g hepmc --configKeyValues "HepMC.progCmd=<cmd>" ...
  //
  // where <cmd> is the command line to run an event generator.  The
  // event generator should output HepMC event records to standard
  // output.  Nothing else, but the HepMC event record may be output
  // to standard output.  If the EG has other output to standard
  // output, then a filter can be set-up.  For example
  //
  //    crmc -n 3 -o hepmc3 -c /optsw/inst/etc/crmc.param -f /dev/stdout \
  //      | sed -n 's/^\(HepMC::\|[EAUWVP] \)/\1/p'
  //
  // What's more, the event generator program must accept the command
  // line argument `-n NEVENTS` to set the number of events to
  // produce.
  //
  // Perhaps we should consider a way to set a seed on the EG.  It
  // could be another configuration parameter.  Of course, if the EG
  // program accepts a seed option, say `-s SEED`, then one could
  // simply pass
  //
  //     -s \$RANDOM
  //
  // to as part of the command line in `progCmd`.
  //
  // All of this can conviniently be achieved via a wrapper script
  // around the actual EG program.
  if (not mProgCmd.empty()) {
    // Set filename to be a temporary name
    // Should perhaps use
    //
    //   TString base("xxxxxx");
    //   auto fp = gSystem->TempFileName(base);
    //   fclose(fp);
    //
    filename = std::tmpnam(nullptr);

    // Make a fifo
    int ret = mkfifo(filename.c_str(), 0600);
    if (ret != 0) {
      LOG(fatal) << "Failed to make fifo \"" << filename << "\"";
      return false;
    }

    // Build command line, rediret stdout to our fifo and put
    // in the background.
    std::string cmd =
      mProgCmd +
      " -n " + std::to_string(mNEvents) +
      " > " + filename + " &";
    LOG(info) << "EG command line is \"" << cmd << "\"";

    ret = std::system(cmd.c_str());
    if (ret != 0) {
      LOG(fatal) << "Failed to spawn \"" << cmd << "\"";
      return false;
    }
  }
  /** open file **/
  mStream.open(filename);
  if (!mStream.is_open()) {
    LOG(fatal) << "Cannot open input file: " << filename << std::endl;
    return kFALSE;
  }

  LOG(info) << "Set up reader to read from \"" << filename << "\"" << std::endl;
  /** create reader according to HepMC version **/
  switch (mVersion) {
    case 2:
      mStream.close();
      mReader = new HepMC3::ReaderAsciiHepMC2(filename);
      break;
    case 3:
      mReader = new HepMC3::ReaderAscii(mStream);
      break;
    default:
      LOG(fatal) << "Unsupported HepMC version: " << mVersion << std::endl;
      return kFALSE;
  }

  // skip events at the beginning
  if (!mReader->failed()) {
    LOGF(info, "%i events to skip.", mEventsToSkip);
    for (auto ind = 0; ind < mEventsToSkip; ind++) {
      if (!generateEvent()) {
        LOGF(error, "The file %s only contains %i events!", mFileName, ind);
        break;
      }
    }
  }

  /** success **/
  return !mReader->failed();
}

/*****************************************************************/
/*****************************************************************/

} /* namespace eventgen */
} /* namespace o2 */
