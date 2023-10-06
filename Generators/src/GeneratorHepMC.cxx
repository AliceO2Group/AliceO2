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
#include "SimConfig/SimConfig.h"
#include "HepMC3/ReaderFactory.h"
#include "HepMC3/GenEvent.h"
#include "HepMC3/GenParticle.h"
#include "HepMC3/GenVertex.h"
#include "HepMC3/FourVector.h"
#include "HepMC3/Version.h"
#include "TParticle.h"

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
  : GeneratorHepMC("ALICEo2", "ALICEo2 HepMC Generator")
{
}

/*****************************************************************/

GeneratorHepMC::GeneratorHepMC(const Char_t* name, const Char_t* title)
  : Generator(name, title)
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
  LOG(info) << "Destructing GeneratorHepMC";
  if (mReader)
    mReader->close();
  if (mEvent)
    delete mEvent;
  removeTemp();
}

/*****************************************************************/
void GeneratorHepMC::setup(const GeneratorFileOrCmdParam& param0,
                           const GeneratorHepMCParam& param,
                           const conf::SimConfig& config)
{
  GeneratorFileOrCmd::setup(param0, config);
  setEventsToSkip(param.eventsToSkip);
}

/*****************************************************************/

Bool_t GeneratorHepMC::generateEvent()
{
  LOG(debug) << "Generating an event";
  /** generate event **/
  int tries = 0;
  do {
    LOG(debug) << " try # " << ++tries;
    if (not mReader and not makeReader())
      return false;

    /** clear and read event **/
    mEvent->clear();
    mReader->read_event(*mEvent);
    if (not mReader->failed()) {
      /** set units to desired output **/
      mEvent->set_units(HepMC3::Units::GEV, HepMC3::Units::MM);
      LOG(debug) << "Read one event " << mEvent->event_number();
      return true;
    }
  } while (true);

  /** failure **/
  return false;
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
    eventHeader->putInfo<int>(Key::pdfCode1, pdfInfo->pdf_id[0]);
    eventHeader->putInfo<int>(Key::pdfCode2, pdfInfo->pdf_id[1]);
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

bool GeneratorHepMC::makeReader()
{
  // Reset the reader smart pointer
  LOG(debug) << "Reseting the reader";
  mReader.reset();

  // Check that we have any file names left
  if (mFileNames.size() < 1) {
    LOG(debug) << "No more files to read, return false";
    return false;
  }

  // If we have file names left, pop the top of the list (LIFO)
  auto filename = mFileNames.front();
  mFileNames.pop_front();

  LOG(debug) << "Next file to read: \"" << filename << "\" "
             << mFileNames.size() << " left";

  if (not mCmd.empty()) {
    // For FIFO reading, we assume straight ASCII output always.
    // Unfortunately, the HepMC3::deduce_reader `stat`s the filename
    // which isn't supported on a FIFO, so we have to use the reader
    // directly.
    LOG(info) << "Creating ASCII reader of " << filename;
    mReader.reset(new HepMC3::ReaderAscii(filename));
  } else {
    LOG(info) << "Deduce a reader of " << filename;
    mReader = HepMC3::deduce_reader(filename);
  }

  bool ret = bool(mReader) and not mReader->failed();
  LOG(info) << "Reader is " << mReader.get() << " " << ret;
  return ret;
}

/*****************************************************************/

Bool_t GeneratorHepMC::Init()
{
  /** init **/

  /** init base class **/
  Generator::Init();

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
  // What's more, the event generator program _must_ accept the
  // following command line argument
  //
  //    `-n NEVENTS` to set the number of events to produce.
  //
  // Optionally, the command line should also accept
  //
  //    `-s SEED`   to set the random number seed
  //    `-b FM`     to set the maximum impact parameter to sample
  //    `-o OUTPUT` to set the output file name
  //
  // All of this can conviniently be achieved via a wrapper script
  // around the actual EG program.
  if (not mCmd.empty()) {
    // Set filename to be a temporary name
    if (not makeTemp())
      return false;

    // Make a fifo
    if (not makeFifo())
      return false;

    // Build command line, rediret stdout to our fifo and put
    std::string cmd = makeCmdLine();
    LOG(debug) << "EG command line is \"" << cmd << "\"";

    // Execute the command line
    if (not executeCmdLine(cmd)) {
      LOG(fatal) << "Failed to spawn \"" << cmd << "\"";
      return false;
    }
  } else {
    // If no command line was given, ensure that all files are present
    // on the system.  Note, in principle, HepMC3 can read from remote
    // files
    //
    //    root://           XRootD served
    //    http[s]://        Web served
    //    gsidcap://        DCap served
    //
    // These will all be handled in HepMC3 via ROOT's TFile protocol
    // and the files are assumed to contain a TTree named
    // `hepmc3_tree` and that tree has the branches
    //
    //    `hepmc3_event`  with object of type `HepMC3::GenEventData`
    //    `GenRunInfo`    with object of type `HepMC3::GenRunInfoData`
    //
    // where the last branch is optional.
    //
    // However, here we will assume system local files.  If _any_ of
    // the listed files do not exist, then we fail.
    if (not ensureFiles())
      return false;
  }

  // Create reader for current (first) file
  return true;
}

/*****************************************************************/
/*****************************************************************/

} /* namespace eventgen */
} /* namespace o2 */
