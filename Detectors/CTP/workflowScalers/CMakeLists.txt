o2_add_executable(
                  proxy
                  COMPONENT_NAME ctp
                  SOURCES src/ctp-proxy.cxx
                  PUBLIC_LINK_LIBRARIES O2::DCStestWorkflow
                  O2::DataFormatsCTP)
o2_add_executable(
                  qc-proxy
                  COMPONENT_NAME ctp
                  SOURCES src/ctp-qc-proxy.cxx
                  PUBLIC_LINK_LIBRARIES O2::DCStestWorkflow
                  O2::DataFormatsCTP)
o2_add_executable(
                  ccdb-orbit
                  COMPONENT_NAME ctp
                  SOURCES src/ctp-ccdb-orbit.cxx
                  PUBLIC_LINK_LIBRARIES O2::DataFormatsCTP
                  Boost::program_options)
