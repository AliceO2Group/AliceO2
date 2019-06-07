o2_add_test(simulation
            COMPONENT_NAME mchsimulation
            SOURCES testDigitMerging.cxx DigitMerging.cxx testGeometry.cxx
                    testDigitization.cxx testResponse.cxx
            PUBLIC_LINK_LIBRARIES O2::MCHSimulation
            LABELS "mch;slow" CONFIGURATIONS RelWithDebInfo)

if(benchmark_FOUND)
  o2_add_executable(
    digit-merging
    COMPONENT_NAME mch
    SOURCES benchDigitMerging.cxx DigitMerging.cxx
    PUBLIC_LINK_LIBRARIES O2::MCHSimulation benchmark::benchmark)

endif()
