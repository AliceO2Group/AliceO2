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
using namespace o2::dataformats;

struct O2simHepmcPublisher {
  Configurable<std::string> hepmcFileName{"hepmc", "input.hepmc", "name of the input file with HepMC events"};
  Configurable<int> aggregate{"aggregate-timeframe", 300, "Number of events to put in a timeframe"};
  Configurable<int> maxEvents{"nevents", -1, "Maximum number of events to convert"};
  Configurable<bool> hepmcv2{"v2", false, "If the input is HepMCv2"};

  int eventCounter = 0;
  int tfCounter = 0;
  std::shared_ptr<HepMC3::Reader> hepMCReader;
  bool eos = false;

  std::vector<o2::pmr::vector<o2::MCTrack>*> mctracks_vector;
  std::vector<o2::dataformats::MCEventHeader*> mcheader_vector;

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
    mctracks_vector.reserve(aggregate);
    mcheader_vector.reserve(aggregate);
  }

  void run(o2::framework::ProcessingContext& pc)
  {
    HepMC3::GenEvent event;
    auto batch = maxEvents > 0 ? std::min((int)aggregate, (int)maxEvents - eventCounter) : (int)aggregate;
    for (auto i = 0; i < batch; ++i) {
      mctracks_vector.push_back(&pc.outputs().make<o2::pmr::vector<o2::MCTrack>>(Output{"MC", "MCTRACKS", 0}));
      auto& mctracks = mctracks_vector.back();
      mcheader_vector.push_back(&pc.outputs().make<o2::dataformats::MCEventHeader>(Output{"MC", "MCHEADER", 0}));
      auto& mcheader = mcheader_vector.back();
      // read next entry
      hepMCReader->read_event(event);
      if (hepMCReader->failed()) {
        LOGP(warn, "Failed to read from HEPMC input file");
        eos = true;
        break;
      }

      // create O2 MCHeader and MCtracks vector out of HEPMC event
      mcheader->SetEventID(event.event_number());
      mcheader->SetVertex(event.event_pos().px(), event.event_pos().py(), event.event_pos().pz());
      auto xsecInfo = event.cross_section();
      if (xsecInfo != nullptr) {
        mcheader->putInfo(MCInfoKeys::acceptedEvents, (uint64_t)xsecInfo->get_accepted_events());
        mcheader->putInfo(MCInfoKeys::attemptedEvents, (uint64_t)xsecInfo->get_attempted_events());
        mcheader->putInfo(MCInfoKeys::xSection, (float)xsecInfo->xsec());
        mcheader->putInfo(MCInfoKeys::xSectionError, (float)xsecInfo->xsec_err());
      }
      auto scale = event.attribute<HepMC3::DoubleAttribute>(MCInfoKeys::eventScale);
      if (scale != nullptr) {
        mcheader->putInfo(MCInfoKeys::eventScale, (float)scale->value());
      }
      auto nMPI = event.attribute<HepMC3::IntAttribute>(MCInfoKeys::mpi);
      if (nMPI != nullptr) {
        mcheader->putInfo(MCInfoKeys::mpi, nMPI->value());
      }
      auto sid = event.attribute<HepMC3::IntAttribute>(MCInfoKeys::processCode);
      auto scode = event.attribute<HepMC3::IntAttribute>(MCInfoKeys::processID); // default pythia8 hepmc3 interface uses signal_process_id
      if (sid != nullptr) {
        mcheader->putInfo(MCInfoKeys::processCode, sid->value());
      } else if (scode != nullptr) {
        mcheader->putInfo(MCInfoKeys::processCode, scode->value());
      }
      auto pdfInfo = event.pdf_info();
      if (pdfInfo != nullptr) {
        mcheader->putInfo(MCInfoKeys::pdfParton1Id, pdfInfo->parton_id[0]);
        mcheader->putInfo(MCInfoKeys::pdfParton2Id, pdfInfo->parton_id[1]);
        mcheader->putInfo(MCInfoKeys::pdfCode1, pdfInfo->pdf_id[0]);
        mcheader->putInfo(MCInfoKeys::pdfCode2, pdfInfo->pdf_id[1]);
        mcheader->putInfo(MCInfoKeys::pdfX1, (float)pdfInfo->x[0]);
        mcheader->putInfo(MCInfoKeys::pdfX2, (float)pdfInfo->x[1]);
        mcheader->putInfo(MCInfoKeys::pdfScale, (float)pdfInfo->scale);
        mcheader->putInfo(MCInfoKeys::pdfXF1, (float)pdfInfo->xf[0]);
        mcheader->putInfo(MCInfoKeys::pdfXF2, (float)pdfInfo->xf[1]);
      }
      auto heavyIon = event.heavy_ion();
      if (heavyIon != nullptr) {
        mcheader->putInfo(MCInfoKeys::nCollHard, heavyIon->Ncoll_hard);
        mcheader->putInfo(MCInfoKeys::nPartProjectile, heavyIon->Npart_proj);
        mcheader->putInfo(MCInfoKeys::nPartTarget, heavyIon->Npart_targ);
        mcheader->putInfo(MCInfoKeys::nColl, heavyIon->Ncoll);
        mcheader->putInfo(MCInfoKeys::nCollNNWounded, heavyIon->N_Nwounded_collisions);
        mcheader->putInfo(MCInfoKeys::nCollNWoundedN, heavyIon->Nwounded_N_collisions);
        mcheader->putInfo(MCInfoKeys::nCollNWoundedNwounded, heavyIon->Nwounded_Nwounded_collisions);
        mcheader->putInfo(MCInfoKeys::nSpecProjectileNeutron, heavyIon->Nspec_proj_n);
        mcheader->putInfo(MCInfoKeys::nSpecProjectileProton, heavyIon->Nspec_proj_p);
        mcheader->putInfo(MCInfoKeys::nSpecTargetNeutron, heavyIon->Nspec_targ_n);
        mcheader->putInfo(MCInfoKeys::nSpecTargetProton, heavyIon->Nspec_targ_p);
        mcheader->putInfo(MCInfoKeys::impactParameter, (float)heavyIon->impact_parameter);
        mcheader->putInfo(MCInfoKeys::planeAngle, (float)heavyIon->event_plane_angle);
        mcheader->putInfo("eccentricity", (float)heavyIon->eccentricity);
        mcheader->putInfo(MCInfoKeys::sigmaInelNN, (float)heavyIon->sigma_inel_NN);
        mcheader->putInfo(MCInfoKeys::centrality, (float)heavyIon->centrality);
      }

      auto particles = event.particles();
      for (auto const& particle : particles) {
        auto parents = particle->parents();
        auto has_parents = parents.size() > 0;
        auto children = particle->children();
        auto has_children = children.size() > 0;
        auto p = particle->momentum();
        auto v = particle->production_vertex();
        mctracks->emplace_back(
          particle->pid(),
          has_parents ? parents.front()->id() : -1, has_parents ? parents.back()->id() : -1,
          has_children ? children.front()->id() : -1, has_children ? children.back()->id() : -1,
          p.px(), p.py(), p.pz(),
          v->position().x(), v->position().y(), v->position().z(),
          v->position().t(), 0);
      }
      ++eventCounter;
    }

    // report number of TFs injected for the rate limiter to work
    ++tfCounter;
    pc.services().get<o2::monitoring::Monitoring>().send(o2::monitoring::Metric{(uint64_t)tfCounter, "df-sent"}.addTag(o2::monitoring::tags::Key::Subsystem, o2::monitoring::tags::Value::DPL));
    if (eos || (maxEvents > 0 && eventCounter >= maxEvents)) {
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
