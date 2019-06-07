o2_add_library(Steer
               SOURCES
               src/O2MCApplication.cxx
               src/InteractionSampler.cxx
               src/HitProcessingManager.cxx
               PUBLIC_LINK_LIBRARIES
               O2::CommonDataFormat
               O2::SimulationDataFormat
               O2::ITSMFTSimulation
               O2::TPCSimulation)

o2_target_root_dictionary(Steer
                          HEADERS
                          include/Steer/InteractionSampler.h
                          include/Steer/HitProcessingManager.h
                          include/Steer/O2RunSim.h
                          include/Steer/O2MCApplication.h
                          include/Steer/O2MCApplicationBase.h)

o2_add_test(InteractionSampler
            PUBLIC_LINK_LIBRARIES
            O2::Steer
            SOURCES
            test/testInteractionSampler.cxx
            LABELS
            steer)

o2_add_test(HitProcessingManager
            PUBLIC_LINK_LIBRARIES
            O2::Steer
            SOURCES
            test/testHitProcessingManager.cxx
            LABELS
            steer)

add_subdirectory(DigitizerWorkflow)
