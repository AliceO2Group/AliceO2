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

#include "../Framework/Core/src/ArrowSupport.h"
#include "Framework/AnalysisTask.h"
#include "Monitoring/Monitoring.h"
#include "Framework/CommonDataProcessors.h"
#include "SimulationDataFormat/MCEventHeader.h"
#include "SimulationDataFormat/MCTrack.h"

#include "HepMC3/GenEvent.h"
#include "HepMC3/GenParticle.h"
#include "HepMC3/GenVertex.h"
#include "HepMC3/ReaderAscii.h"
#include "HepMC3/ReaderAsciiHepMC2.h"

#include "Framework/runDataProcessing.h"

using namespace o2::framework;

struct O2simHepmcPublisher {
  Configurable<std::string> hepmcFileName{"hepmc", "input.hepmc", "name of the input file with HepMC events"};
  Configurable<int> aggregate{"aggregate-timeframe", 300, "Number of events to put in a timeframe"};
  Configurable<int> maxEvents{"nevents", -1, "Maximum number of events to convert"};
  Configurable<bool> hepmcv2{"v2", false, "If the input is HepMCv2"};

  int eventCounter = 0;
  int tfCounter = 0;
  std::shared_ptr<HepMC3::Reader> hepMCReader;
  bool eos = false;
  std::vector<o2::MCTrack> mcTracks;

  void init(o2::framework::InitContext& /*ic*/)
  {
    if (hepmcv2) {
      hepMCReader = std::make_shared<HepMC3::ReaderAsciiHepMC2>((std::string)hepmcFileName);
    } else {
      hepMCReader = std::make_shared<HepMC3::ReaderAscii>((std::string)hepmcFileName);
    }
    if (hepMCReader->failed()) {
      LOGP(fatal, "Cannot open HEPMC kine file {}", (std::string)hepmcFileName);
    }
    // allocate the memory upfront to prevent reallocations later
    mcTracks.reserve(1e3 * aggregate);
  }

  void run(o2::framework::ProcessingContext& pc)
  {
    HepMC3::GenEvent event;
    for (auto i = 0; i < (int)aggregate; ++i) {
      // read next entry
      hepMCReader->read_event(event);
      if (hepMCReader->failed()) {
        LOGP(warn, "Failed to read from HEPMC input file");
        eos = true;
        break;
      }

      // create O2 MCHeader and MCtracks vector out of HEPMC event
      o2::dataformats::MCEventHeader mcHeader;
      mcHeader.SetEventID(event.event_number());
      mcHeader.SetVertex(event.event_pos().px(), event.event_pos().py(), event.event_pos().pz());
      auto xsecInfo = event.cross_section();
      if (xsecInfo != nullptr) {
        mcHeader.putInfo("Accepted", (uint64_t)xsecInfo->get_accepted_events());
        mcHeader.putInfo("Attempted", (uint64_t)xsecInfo->get_attempted_events());
        mcHeader.putInfo("XsectGen", (float)xsecInfo->xsec());
        mcHeader.putInfo("XsectErr", (float)xsecInfo->xsec_err());
      }
      auto scale = event.attribute<HepMC3::DoubleAttribute>("event_scale");
      if (scale != nullptr) {
        mcHeader.putInfo("PtHard", (float)scale->value());
      }
      auto nMPI = event.attribute<HepMC3::IntAttribute>("mpi");
      if (nMPI != nullptr) {
        mcHeader.putInfo("MPI", nMPI->value());
      }
      auto sid = event.attribute<HepMC3::IntAttribute>("signal_process_id");
      if (sid != nullptr) {
        mcHeader.putInfo("ProcessId", sid->value());
      }
      auto pdfInfo = event.pdf_info();
      if (pdfInfo != nullptr) {
        mcHeader.putInfo("Id1", pdfInfo->parton_id[0]);
        mcHeader.putInfo("Id2", pdfInfo->parton_id[1]);
        mcHeader.putInfo("PdfId1", pdfInfo->pdf_id[0]);
        mcHeader.putInfo("PdfId2", pdfInfo->pdf_id[1]);
        mcHeader.putInfo("X1", (float)pdfInfo->x[0]);
        mcHeader.putInfo("X2", (float)pdfInfo->x[1]);
        mcHeader.putInfo("scale", (float)pdfInfo->scale);
        mcHeader.putInfo("Pdf1", (float)pdfInfo->xf[0]);
        mcHeader.putInfo("Pdf2", (float)pdfInfo->xf[1]);
      }
      auto heavyIon = event.heavy_ion();
      if (heavyIon != nullptr) {
        mcHeader.putInfo("NcollHard", heavyIon->Ncoll_hard);
        mcHeader.putInfo("NpartProj", heavyIon->Npart_proj);
        mcHeader.putInfo("NpartTarg", heavyIon->Npart_targ);
        mcHeader.putInfo("Ncoll", heavyIon->Ncoll);
        mcHeader.putInfo("NNwoundedCollisions", heavyIon->N_Nwounded_collisions);
        mcHeader.putInfo("NwoundedNCollisions", heavyIon->Nwounded_N_collisions);
        mcHeader.putInfo("NwoundedNwoundedCollisions", heavyIon->Nwounded_Nwounded_collisions);
        mcHeader.putInfo("SpectatorNeutrons", heavyIon->spectator_neutrons);
        mcHeader.putInfo("SpectatorProtons", heavyIon->spectator_protons);
        mcHeader.putInfo("ImpactParameter", (float)heavyIon->impact_parameter);
        mcHeader.putInfo("EventPlaneAngle", (float)heavyIon->event_plane_angle);
        mcHeader.putInfo("Eccentricity", (float)heavyIon->eccentricity);
        mcHeader.putInfo("SigmaInelNN", (float)heavyIon->sigma_inel_NN);
        mcHeader.putInfo("Centrality", (float)heavyIon->centrality);
      }

      auto particles = event.particles();
      for (auto const& particle : particles) {
        auto parents = particle->parents();
        auto has_parents = parents.size() > 0;
        auto children = particle->children();
        auto has_children = children.size() > 0;
        auto p = particle->momentum();
        auto v = particle->production_vertex();
        mcTracks.emplace_back(
          particle->pid(),
          has_parents ? parents.front()->id() : -1, has_parents ? parents.back()->id() : -1,
          has_children ? children.front()->id() : -1, has_children ? children.back()->id() : -1,
          p.px(), p.py(), p.pz(),
          v->position().x(), v->position().y(), v->position().z(),
          v->position().t(), 0);
      }

      // add to the message
      pc.outputs().snapshot(Output{"MC", "MCHEADER", 0, Lifetime::Timeframe}, mcHeader);
      pc.outputs().snapshot(Output{"MC", "MCTRACKS", 0, Lifetime::Timeframe}, mcTracks);
      mcTracks.clear();
      ++eventCounter;
    }

    // report number of TFs injected for the rate limiter to work
    pc.services().get<o2::monitoring::Monitoring>().send(o2::monitoring::Metric{(uint64_t)tfCounter, "df-sent"}.addTag(o2::monitoring::tags::Key::Subsystem, o2::monitoring::tags::Value::DPL));
    ++tfCounter;
    if (eos || (maxEvents > 0 && eventCounter == maxEvents)) {
      pc.services().get<ControlService>().endOfStream();
      pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  auto spec = adaptAnalysisTask<O2simHepmcPublisher>(cfgc);
  spec.outputs.emplace_back("MC", "MCHEADER", 0, Lifetime::Timeframe);
  spec.outputs.emplace_back("MC", "MCTRACKS", 0, Lifetime::Timeframe);
  spec.requiredServices.push_back(o2::framework::ArrowSupport::arrowBackendSpec());
  spec.algorithm = CommonDataProcessors::wrapWithRateLimiting(spec.algorithm);
  return {spec};
}
