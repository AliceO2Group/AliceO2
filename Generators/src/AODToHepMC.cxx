// Copyright 2023-2099 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @author Christian Holm Christensen <cholm@nbi.dk>

#include "Generators/AODToHepMC.h"
#include <TMCProcess.h>
namespace o2
{
namespace eventgen
{
// -------------------------------------------------------------------
void AODToHepMC::init()
{

  mOnlyGen = configs.onlyGen;
  mUseTree = configs.useTree;
  mPrecision = configs.precision;
  mRecenter = configs.recenter;
  enableDump(configs.dump);

  LOG(debug) << "\n"
             << "=== o2::rivet::Converter ===\n"
             << "  Dump to output:        " << configs.dump << "\n"
             << "  Only generated tracks: " << mOnlyGen << "\n"
             << "  Use tree store:        " << mUseTree << "\n"
             << "  Output precision:      " << mPrecision;
}
// -------------------------------------------------------------------
void AODToHepMC::process(Header const& collision,
                         Tracks const& tracks)
{
  LOG(debug) << "--- Processing track information";
  mBeams.clear();
  mOrphans.clear();
  mParticles.clear();
  mVertices.clear();
  mEvent.clear();

  makeHeader(collision);
  makeParticles(tracks);
  makeEvent(collision, tracks);
}
// -------------------------------------------------------------------
void AODToHepMC::process(Header const& collision,
                         XSections const& xsections,
                         PdfInfos const& pdfs,
                         HeavyIons const& heavyions)
{
  LOG(debug) << "--- Processing auxiliary information";
  makeXSection(xsections);
  makePdfInfo(pdfs);
  makeHeavyIon(heavyions, collision);
}
// ===================================================================
void AODToHepMC::makeEvent(Header const& collision,
                           Tracks const& tracks)
{
  LOG(debug) << "Event # " << mEvent.event_number() << " "
             << "(# " << mEventNo << " processed)"
             << "\n"
             << "# of particles " << mParticles.size() << " "
             << "(" << tracks.size() << " input tracks)"
             << "\n"
             << "# of orphans   " << mOrphans.size() << "\n"
             << "# of beams     " << mBeams.size() << "\n"
             << "# of vertexes  " << mVertices.size();
  if (mUseTree) {
    mEvent.reserve(mParticles.size() + mBeams.size(), mVertices.size());
    mEvent.add_tree(mBeams);
  } else {
    if (mOrphans.size() > 0) {
      if (mBeams.size() > 0) {
        LOG(debug) << "Event has " << mOrphans.size() << " orphans "
                   << "out of " << mParticles.size() << " total, "
                   << "and explicit beams " << mBeams.size();
      } else {
        LOG(warning) << "Event has " << mOrphans.size() << " orphans "
                     << "out of " << mParticles.size() << " total, "
                     << "but no beams";
      }
      // Get collision IP
      FourVector v(collision.posX(),
                   collision.posY(),
                   collision.posZ(),
                   collision.t());
      auto ip = std::make_shared<HepMC3::GenVertex>(v);
      mEvent.add_vertex(ip);
      for (auto p : mOrphans) {
        ip->add_particle_out(p);
      }
      // If we have no beam particles, try to make them
      if (mBeams.size() == 0) {
        makeBeams(collision, ip);
      }
    }
  }
  // Flesh out the tracks based on daughter information.
  fleshOut(tracks);
  if (mWriter) {
    // If we have a writer, then dump event to output file
    mWriter->write_event(mEvent);
  }
}
// -------------------------------------------------------------------
void AODToHepMC::makeHeader(Header const& header)
{
  int eventNo = header.bcId();
  if (eventNo == mLastBC) {
    eventNo = mEventNo;
  }

  // Set the event number
  mEvent.set_event_number(eventNo);
  mEvent.weights().push_back(header.weight());
  LOG(debug) << "Event # " << mEvent.event_number()
             << " (BC: " << header.bcId()
             << " serial: " << mEventNo
             << " last: " << mLastBC << ") "
             << " w/weight " << mEvent.weights().front();
  // Store last seen BC
  mLastBC = header.bcId();
  // Increase internal counter of event
  mEventNo++;
}
// -------------------------------------------------------------------
void AODToHepMC::makeXSection(XSections const& xsections)
{
  if (not mCrossSec) {
    // If we do not have a cross-sections object, create it
    mCrossSec = std::make_shared<HepMC3::GenCrossSection>();
  }

  mEvent.set_cross_section(mCrossSec);
  mCrossSec->set_cross_section(0.f, 0.f, 0, 0);

  if (xsections.size() <= 0) {
    // If we have no info, skip the rest
    return;
  }

  XSection xsection = xsections.iteratorAt(0);
  mCrossSec->set_cross_section(xsection.xsectGen(),
                               xsection.xsectErr(),
                               xsection.accepted(),
                               xsection.attempted());
}
// -------------------------------------------------------------------
void AODToHepMC::makePdfInfo(PdfInfos const& pdfs)
{
  if (not mPdf) {
    // If we do not have a Parton Distribution Function object, create it
    mPdf = std::make_shared<HepMC3::GenPdfInfo>();
  }

  mEvent.set_pdf_info(mPdf);
  mPdf->set(0, 0, 0.f, 0.f, 0.f, 0.f, 0.f, 0, 0);

  if (pdfs.size() <= 0) {
    // If we have no PDF info, skip the rest
    return;
  }

  PdfInfo pdf = pdfs.iteratorAt(0);
  mPdf->set(pdf.id1(),
            pdf.id2(),
            pdf.x1(),
            pdf.x2(),
            pdf.scalePdf(),
            pdf.pdf1(),
            pdf.pdf2(),
            pdf.pdfId1(),
            pdf.pdfId2());
}
// -------------------------------------------------------------------
void AODToHepMC::makeHeavyIon(HeavyIons const& heavyions,
                              Header const& header)
{
  if (not mIon) {
    // Generate heavy ion element if it doesn't exist
    mIon = std::make_shared<HepMC3::GenHeavyIon>();
  }

  mEvent.set_heavy_ion(mIon);
  mIon->impact_parameter = header.impactParameter();
  mIon->event_plane_angle = 0.f;
  mIon->Ncoll_hard = 0;
  mIon->Npart_proj = 0;
  mIon->Npart_targ = 0;
  mIon->Nspec_proj_n = 0;
  mIon->Nspec_proj_p = 0;
  mIon->Nspec_targ_n = 0;
  mIon->Nspec_targ_p = 0;
  mIon->Ncoll = 0;
  mIon->N_Nwounded_collisions = 0;
  mIon->Nwounded_N_collisions = 0;
  mIon->Nwounded_Nwounded_collisions = 0;
  mIon->sigma_inel_NN = 0.f;
  mIon->centrality = 0.f;
#ifndef HEPMC3_NO_DEPRECATED
  // Deprecated interface with a single eccentricity
  mIon->eccentricity = 0.f;
#else
  // Newer interface that stores multiple orders of eccentricities.
  mIon->eccentricities[1] = 0.f;
#endif

  if (heavyions.size() <= 0) {
    // If we have no heavy-ion information, skip the rest
    return;
  }

  HeavyIon heavyion = heavyions.iteratorAt(0);
  float r = 1;
  // We need to calculate the ratio projectile to target participants
  // so that we may break up the number of spectators and so on. This
  // is because the AOD HepMC3HeavyIons table does not store the
  // relevant information directly.
  if (heavyion.npartProj() < heavyion.npartTarg() and
      heavyion.npartTarg() > 0) {
    r = heavyion.npartProj() / heavyion.npartTarg();
  } else if (heavyion.npartTarg() < heavyion.npartProj() and
             heavyion.npartProj() > 0) {
    r = heavyion.npartTarg() / heavyion.npartProj();
    r = (1 - r);
  }

  // Heavy ion parameters.  Note that number of projectile/target
  // proton/neutrons are set by the ratio calculated above.
  mIon->impact_parameter = heavyion.impactParameter();
  mIon->event_plane_angle = heavyion.eventPlaneAngle();
  mIon->Ncoll_hard = heavyion.ncollHard();
  mIon->Npart_proj = heavyion.npartProj();
  mIon->Npart_targ = heavyion.npartTarg();
  mIon->Nspec_proj_n = heavyion.spectatorNeutrons() * (1 - r);
  mIon->Nspec_proj_p = heavyion.spectatorProtons() * (1 - r);
  mIon->Nspec_targ_n = heavyion.spectatorNeutrons() * r;
  mIon->Nspec_targ_p = heavyion.spectatorProtons() * r;
  mIon->Ncoll = heavyion.ncoll();
  mIon->N_Nwounded_collisions = heavyion.nNwoundedCollisions();
  mIon->Nwounded_N_collisions = heavyion.nwoundedNCollisions();
  mIon->Nwounded_Nwounded_collisions = heavyion.nwoundedNwoundedCollisions();
  mIon->sigma_inel_NN = heavyion.sigmaInelNN();
  mIon->centrality = heavyion.centrality();
#ifndef HEPMC3_NO_DEPRECATED
  mIon->eccentricity = heavyion.eccentricity();
#else
  mIon->eccentricities[1] = heavyion.eccentricity();
#endif
}
// -------------------------------------------------------------------
void AODToHepMC::makeParticles(Tracks const& tracks)
{
  for (auto track : tracks) {
    makeParticleRecursive(track, tracks);
  }
}
// -------------------------------------------------------------------
AODToHepMC::ParticlePtr AODToHepMC::getParticle(Track const& ref) const
{
  auto iter = mParticles.find(ref.globalIndex());
  if (iter == mParticles.end()) {
    return nullptr;
  }

  return iter->second;
}
// -------------------------------------------------------------------
bool AODToHepMC::isIgnored(Track const& track) const
{
  bool fromEG = track.producedByGenerator();
  if (!fromEG and mOnlyGen) {
    LOG(debug) << " Track # " << track.globalIndex()
               << " from transport, ignored";
    return true;
  }
  return false;
}
// -------------------------------------------------------------------
AODToHepMC::ParticlePtr AODToHepMC::makeParticleRecursive(Track const& track,
                                                          Tracks const& tracks,
                                                          bool force)
{
  ParticlePtr particle = getParticle(track);

  // Check if we already have the particle, and if so, return it
  if (particle) {
    return particle;
  }

  // Make this particle and store it
  int motherStatus = 0;
  particle = makeParticle(track, motherStatus, force);
  if (not particle) {
    return nullptr;
  }

  // Store mapping from index to particle
  mParticles[track.globalIndex()] = particle;

  // Generate mother particles, recurses down tree
  ParticleVector mothers;
  VertexPtr vout;
  for (auto mtrack : track.mothers_as<Tracks>()) {
    auto mother = makeParticleRecursive(mtrack, tracks, true);
    // If mother not found, continue
    if (not mother) {
      continue;
    }

    // Overrride mother status based on production mechanism of daughter
    if (motherStatus != 0) {
      mother->set_status(motherStatus);
    }

    mothers.push_back(mother);
    // Update the production vertex if not set already
    if (not vout) {
      vout = mother->end_vertex();
    }
  }

  // If we have no out vertex, and the particle isn't a beam
  // particle, and we have mother particles, then create the
  // out-going vertex.
  if (not vout and
      particle->status() != 4 and
      mothers.size() > 0) {
    vout = makeVertex(track);
    mVertices.push_back(vout);

    // If mothers do not have any end-vertex, add them to the found
    // vertex.
    for (auto mother : mothers) {
      if (not mother->end_vertex()) {
        vout->add_particle_in(mother);
      }
    }
  }

  // If we got a out-going vertex, add this particle to that
  if (vout) {
    vout->add_particle_out(particle);
  }

  // If this is a beam particle, add to them
  if (particle->status() == 4) {
    mBeams.push_back(particle);
  }
  // if if there no mothers, and this is not beam, then make
  // this an orphan.
  else if (mothers.size() <= 0) {
    mOrphans.push_back(particle);
  }
  // return the particle
  return particle;
}
// -------------------------------------------------------------------
AODToHepMC::ParticlePtr AODToHepMC::makeParticle(const Track& track,
                                                 Int_t& motherStatus,
                                                 bool force) const
{
  Int_t pdg = track.pdgCode();
  int hepMCstatus = track.getHepMCStatusCode();
  int egStatus = track.getGenStatusCode();
  int transport = track.getProcess();
  bool fromEG = track.producedByGenerator();

  // Do not generate particle if it is not from generator and we are
  // asked not to make other particles.  Note, if a particle has this
  // as one of it's mothers, we will make it despite the flag.
  if (not force and mOnlyGen and !fromEG) {
    return nullptr;
  }

  FourVector p(track.px(),
               track.py(),
               track.pz(),
               track.e());

  // Possibly update mother status depending on the production
  // mechanism of this child.
  motherStatus = 0; // 200 + uni; // Mother unknown status code
  switch (transport) {
    case kPDecay:
      motherStatus = 2;
      break; // Mother decayed!
  }
  int state = hepMCstatus < 0 ? 200 + transport : hepMCstatus;

  ParticlePtr g = std::make_shared<HepMC3::GenParticle>(p, pdg, state);
  // g->set_generated_mass(track.GetMass());

  return g;
}
// -------------------------------------------------------------------
void AODToHepMC::fleshOut(Tracks const& tracks)
{
  for (auto track : tracks) {
    fleshOutParticle(track, tracks);
  }
}
// -------------------------------------------------------------------
void AODToHepMC::fleshOutParticle(Track const& track, Tracks const& tracks)
{
  // If we are only propagating generated tracks, then we need
  // not process transported tracks
  if (isIgnored(track)) {
    return;
  }

  // Check that we can find the track in our map
  auto particle = getParticle(track);
  if (not particle) {
    LOG(warning) << "No particle at " << track.globalIndex() << " in map";
    return;
  }

  auto endVtx = particle->end_vertex();

  // If endVtx is null, then the particle has no end vertex.  This can
  // be because the particle is truly a leaf particle (final-state,
  // including after transport), or because the particle wasn't
  // properly attached to the event.  Since we cannot be sure, in the
  // first case, that the status flag accurately reflects the
  // sitation, we instead investigate the actual daughters.
  //
  // We check even if the particle has an end vertex and that all
  // daughters have the same production vertex.
  int svId = particle->id();
  VertexPtr candidate = endVtx;
  ParticleVector headLess;
  for (auto dtrack : track.daughters_as<Tracks>()) {
    // Check that the daughther is generated by EG.  If not, and we
    // only store generated tracks, then go on to the next daughter
    // (or return?).
    if (isIgnored(dtrack)) {
      continue;
    }

    auto daughter = getParticle(dtrack);
    if (not daughter) {
      LOG(warning) << "Daughter " << dtrack.globalIndex()
                   << " of " << track.globalIndex()
                   << " not found in map!";
      continue;
    }

    // We get the production vertex of the daughter.  If there's no
    // production vertex, then the daughter is deemed "head-less", and
    // we will attach it to the end vertex of the mother, if one
    // exists or is found below.
    auto prodVtx = daughter->production_vertex();
    if (not prodVtx) {
      headLess.push_back(daughter);
      continue;
    }

    // If the mother has an end vertex, but it doesn't match the
    // production vertex of the daughter, then this daughter does not
    // belong to that mother.  This comes about because O2 encodes
    // daughters as a range - a la HEPEVT, which requires a specific
    // ordering of particles so that all daughters of a vertex are
    // consequitive.  Since we decide to trust the mother information,
    // rather than the daughter information, we will simply disregard
    // such daughters.
    //
    // This check may not be needed
    if (endVtx and endVtx->id() != prodVtx->id()) {
      continue;
    }

    // If we have a current candidate end vertex, but it doesn't match
    // the production vertex of the daughter, then we give a warning.
    //
    // This check may not be needed
    if (candidate and prodVtx->id() != candidate->id()) {
      LOG(warning) << "Production vertex of daughter " << daughter->id()
                   << " of " << particle->id()
                   << " is not the same as previously found from "
                   << svId;
      continue;
    }
    candidate = prodVtx;
    svId = daughter->id();
  }
  if (not endVtx) {
    // Give warning for decayed or beam particles without and
    // end vertex
    if (not candidate and
        (particle->status() == 4 or
         particle->status() == 2)) {
      // Only warn for beam and decayed particles
      LOG(warning) << "Particle " << track.globalIndex()
                   << " (" << particle->id() << ")"
                   << " w/status " << particle->status()
                   << " does not have an end-vertex, "
                   << "nor was any found from its daughters";
    }

    // If we have a candidate, set the end vertex
    if (candidate) {
      endVtx = candidate;
      endVtx->add_particle_in(particle);
    }
  }

  // If we have head-less daughters, add them here
  if (endVtx and headLess.size() > 0) {
    for (auto daughter : headLess) {
      endVtx->add_particle_out(daughter);
    }
  }
}
// -------------------------------------------------------------------
void AODToHepMC::enableDump(const std::string& dump)
{
  if (not dump.empty() and mWriter) {
    return;
  }

  if (dump.empty() and not mWriter) {
    return;
  }

  if (not dump.empty()) {
    LOG(debug) << "o2::rivet::Converter: Open output HepMC file " << dump;
    mOutput = new std::ofstream(dump.c_str());
    mWriter = std::make_shared<WriterAscii>(*mOutput);
    mWriter->set_precision(mPrecision);
  } else {
    LOG(debug) << "o2::river::Converter\n"
               << "*********************************\n"
               << "Closing output HepMC file\n"
               << "*********************************";
    mWriter.reset();
    if (mOutput) {
      mOutput->close();
    }

    delete mOutput;
    mOutput = nullptr;
  }
}
// -------------------------------------------------------------------
AODToHepMC::VertexPtr AODToHepMC::makeVertex(const Track& track)
{
  FourVector v(track.vx(),
               track.vy(),
               track.vz(),
               track.vt());
  auto vtx = std::make_shared<HepMC3::GenVertex>(v);
  if (not mUseTree) {
    mEvent.add_vertex(vtx);
  }
  return vtx;
}
// -------------------------------------------------------------------
void AODToHepMC::recenter(Header const& collision)
{
  FourVector ip(collision.posX(),
                collision.posY(),
                collision.posZ(),
                collision.t());

  for (auto vertex : mEvent.vertices()) {
    vertex->set_position(vertex->position() - ip);
  }
}

// -------------------------------------------------------------------
} /* namespace eventgen */
} /* namespace o2 */
//
// EOF
//
