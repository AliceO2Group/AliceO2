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

#include "Mocking.h"
#include <catch_amalgamated.hpp>
#include "../src/O2ControlHelpers.h"
#include "../src/DeviceSpecHelpers.h"
#include "../src/SimpleResourceManager.h"
#include "../src/ComputingResourceHelpers.h"
#include "Framework/DataAllocator.h"
#include "Framework/DeviceControl.h"
#include "Framework/DeviceSpec.h"
#include "Framework/ProcessingContext.h"
#include "Framework/WorkflowSpec.h"
#include "Framework/DriverConfig.h"
#include "Framework/O2ControlParameters.h"

#include <sstream>

using namespace o2::framework;

namespace
{
WorkflowSpec defineDataProcessing()
{
  return {{.name = "A",                                                          //
           .outputs = Outputs{OutputSpec{"TST", "A1"}, OutputSpec{"TST", "A2"}}, // A1 will be consumed twice, A2 is dangling
           .algorithm = AlgorithmSpec{},                                         //
           .options = {ConfigParamSpec{"channel-config", VariantType::String,    // raw input channel
                                       "name=into_dpl,type=pull,method=connect,address=ipc:///tmp/pipe-into-dpl,transport=shmem,rateLogging=10,rcvBufSize=789",
                                       {"Out-of-band channel config"}}}},
          {.name = "B", // producer, no inputs
           .outputs = Outputs{OutputSpec{"TST", "B1"}},
           .metadata = {{ecs::cpuKillThreshold, "3.0"}}},
          {.name = "C", // first consumer of A1, consumer of B1
           .inputs = {InputSpec{"y", "TST", "A1"}, InputSpec{"y", "TST", "B1"}},
           .metadata = {{ecs::privateMemoryKillThresholdMB, "5000"}}},
          {.name = "D", // second consumer of A1
           .inputs = Inputs{InputSpec{"x", "TST", "A1"}},
           .options = {ConfigParamSpec{"a-param", VariantType::Int, 1, {"A parameter which should not be escaped"}},
                       ConfigParamSpec{"b-param", VariantType::String, "", {"a parameter which will be escaped"}},
                       ConfigParamSpec{"c-param", VariantType::String, "foo;bar", {"another parameter which will be escaped"}},
                       ConfigParamSpec{"channel-config", VariantType::String, // raw output channel
                                       "name=outta_dpl,type=push,method=bind,address=ipc:///tmp/pipe-outta-dpl,transport=shmem,rateLogging=10",
                                       {"Out-of-band channel config"}}}}};
}

char* strdiffchr(const char* s1, const char* s2)
{
  while (*s1 && *s1 == *s2) {
    s1++;
    s2++;
  }
  return (*s1 == *s2) ? nullptr : (char*)s1;
}

} // namespace
const auto expectedWorkflow = R"EXPECTED(name: testwf
vars:
  dpl_command: >-
    o2-exe --abdf -defg 'asdf fdsa' | o2-exe-2 -b --zxcv "asdf zxcv"
defaults:
  monitoring_dpl_url: "no-op://"
  user: "flp"
  fmq_rate_logging: 0
  shm_segment_size: 10000000000
  shm_throw_bad_alloc: false
  session_id: default
  resources_monitoring: 15
roles:
  - name: "A"
    connect:
    - name: into_dpl
      type: pull
      transport: shmem
      target: "::into_dpl-{{ it }}"
      rateLogging: "{{ fmq_rate_logging }}"
      rcvBufSize: 789
    task:
      load: testwf-A
  - name: "B"
    connect:
    task:
      load: testwf-B
  - name: "C"
    connect:
    - name: from_A_to_C
      type: pull
      transport: shmem
      target: "{{ Parent().Path }}.A:from_A_to_C"
      rateLogging: "{{ fmq_rate_logging }}"
      sndBufSize: 1
      rcvBufSize: 1
    - name: from_B_to_C
      type: pull
      transport: shmem
      target: "{{ Parent().Path }}.B:from_B_to_C"
      rateLogging: "{{ fmq_rate_logging }}"
      sndBufSize: 1
      rcvBufSize: 1
    task:
      load: testwf-C
  - name: "D"
    connect:
    - name: from_C_to_D
      type: pull
      transport: shmem
      target: "{{ Parent().Path }}.C:from_C_to_D"
      rateLogging: "{{ fmq_rate_logging }}"
      sndBufSize: 1
      rcvBufSize: 1
    bind:
    - name: outta_dpl
      type: push
      transport: shmem
      addressing: ipc
      rateLogging: "{{ fmq_rate_logging }}"
      global: "outta_dpl-{{ it }}"
    task:
      load: testwf-D
)EXPECTED";

const std::vector expectedTasks{
  R"EXPECTED(name: A
defaults:
  log_task_stdout: none
  log_task_stderr: none
  exit_transition_timeout: 15
  _module_cmdline: >-
    source /etc/profile.d/modules.sh && MODULEPATH={{ modulepath }} module load O2 QualityControl Control-OCCPlugin &&
    {{ dpl_command }} | bcsadc/foo
  _plain_cmdline: >-
    source /etc/profile.d/o2.sh && {{ len(extra_env_vars)>0 ? 'export ' + extra_env_vars + ' &&' : '' }} {{ dpl_command }} | bcsadc/foo
control:
  mode: "fairmq"
wants:
  cpu: 0.01
  memory: 1
bind:
  - name: from_A_to_C
    type: push
    transport: shmem
    addressing: ipc
    rateLogging: "{{ fmq_rate_logging }}"
    sndBufSize: 1
    rcvBufSize: 1
command:
  shell: true
  stdout: "{{ log_task_stdout }}"
  stderr: "{{ log_task_stderr }}"
  env:
    - O2_DETECTOR={{ detector }}
    - O2_PARTITION={{ environment_id }}
    - HOME=/tmp
  user: "{{ user }}"
  value: "{{ len(modulepath)>0 ? _module_cmdline : _plain_cmdline }}"
  arguments:
    - "-b"
    - "--exit-transition-timeout"
    - "'{{ exit_transition_timeout }}'"
    - "--monitoring-backend"
    - "'{{ monitoring_dpl_url }}'"
    - "--session"
    - "'{{ session_id }}'"
    - "--infologger-severity"
    - "'{{ infologger_severity }}'"
    - "--infologger-mode"
    - "'{{ infologger_mode }}'"
    - "--driver-client-backend"
    - "'stdout://'"
    - "--shm-segment-size"
    - "'{{ shm_segment_size }}'"
    - "--shm-throw-bad-alloc"
    - "'{{ shm_throw_bad_alloc }}'"
    - "--resources-monitoring"
    - "'{{ resources_monitoring }}'"
    - "--id"
    - "'A'"
    - "--shm-monitor"
    - "'false'"
    - "--log-color"
    - "'false'"
    - "--no-batch"
    - "--bad-alloc-attempt-interval"
    - "'50'"
    - "--bad-alloc-max-attempts"
    - "'1'"
    - "--channel-prefix"
    - "''"
    - "--early-forward-policy"
    - "'never'"
    - "--io-threads"
    - "'1'"
    - "--jobs"
    - "'1'"
    - "--severity"
    - "'info'"
    - "--shm-allocation"
    - "'rbtree_best_fit'"
    - "--shm-mlock-segment"
    - "'false'"
    - "--shm-mlock-segment-on-creation"
    - "'false'"
    - "--shm-no-cleanup"
    - "'false'"
    - "--shm-segment-id"
    - "'0'"
    - "--shm-zero-segment"
    - "'false'"
    - "--stacktrace-on-signal"
    - "'simple'"
    - "--timeframes-rate-limit"
    - "'0'"
)EXPECTED",
  R"EXPECTED(name: B
defaults:
  log_task_stdout: none
  log_task_stderr: none
  exit_transition_timeout: 15
  _module_cmdline: >-
    source /etc/profile.d/modules.sh && MODULEPATH={{ modulepath }} module load O2 QualityControl Control-OCCPlugin &&
    {{ dpl_command }} | foo
  _plain_cmdline: >-
    source /etc/profile.d/o2.sh && {{ len(extra_env_vars)>0 ? 'export ' + extra_env_vars + ' &&' : '' }} {{ dpl_command }} | foo
control:
  mode: "fairmq"
wants:
  cpu: 0.01
  memory: 1
limits:
  cpu: 3.0
bind:
  - name: from_B_to_C
    type: push
    transport: shmem
    addressing: ipc
    rateLogging: "{{ fmq_rate_logging }}"
    sndBufSize: 1
    rcvBufSize: 1
command:
  shell: true
  stdout: "{{ log_task_stdout }}"
  stderr: "{{ log_task_stderr }}"
  env:
    - O2_DETECTOR={{ detector }}
    - O2_PARTITION={{ environment_id }}
    - HOME=/tmp
  user: "{{ user }}"
  value: "{{ len(modulepath)>0 ? _module_cmdline : _plain_cmdline }}"
  arguments:
    - "-b"
    - "--exit-transition-timeout"
    - "'{{ exit_transition_timeout }}'"
    - "--monitoring-backend"
    - "'{{ monitoring_dpl_url }}'"
    - "--session"
    - "'{{ session_id }}'"
    - "--infologger-severity"
    - "'{{ infologger_severity }}'"
    - "--infologger-mode"
    - "'{{ infologger_mode }}'"
    - "--driver-client-backend"
    - "'stdout://'"
    - "--shm-segment-size"
    - "'{{ shm_segment_size }}'"
    - "--shm-throw-bad-alloc"
    - "'{{ shm_throw_bad_alloc }}'"
    - "--resources-monitoring"
    - "'{{ resources_monitoring }}'"
    - "--id"
    - "'B'"
    - "--shm-monitor"
    - "'false'"
    - "--log-color"
    - "'false'"
    - "--no-batch"
    - "--bad-alloc-attempt-interval"
    - "'50'"
    - "--bad-alloc-max-attempts"
    - "'1'"
    - "--channel-prefix"
    - "''"
    - "--early-forward-policy"
    - "'never'"
    - "--io-threads"
    - "'1'"
    - "--jobs"
    - "'1'"
    - "--severity"
    - "'info'"
    - "--shm-allocation"
    - "'rbtree_best_fit'"
    - "--shm-mlock-segment"
    - "'false'"
    - "--shm-mlock-segment-on-creation"
    - "'false'"
    - "--shm-no-cleanup"
    - "'false'"
    - "--shm-segment-id"
    - "'0'"
    - "--shm-zero-segment"
    - "'false'"
    - "--stacktrace-on-signal"
    - "'simple'"
    - "--timeframes-rate-limit"
    - "'0'"
)EXPECTED",
  R"EXPECTED(name: C
defaults:
  log_task_stdout: none
  log_task_stderr: none
  exit_transition_timeout: 15
  _module_cmdline: >-
    source /etc/profile.d/modules.sh && MODULEPATH={{ modulepath }} module load O2 QualityControl Control-OCCPlugin &&
    {{ dpl_command }} | foo
  _plain_cmdline: >-
    source /etc/profile.d/o2.sh && {{ len(extra_env_vars)>0 ? 'export ' + extra_env_vars + ' &&' : '' }} {{ dpl_command }} | foo
control:
  mode: "fairmq"
wants:
  cpu: 0.01
  memory: 1
limits:
  memory: 5000
bind:
  - name: from_C_to_D
    type: push
    transport: shmem
    addressing: ipc
    rateLogging: "{{ fmq_rate_logging }}"
    sndBufSize: 1
    rcvBufSize: 1
command:
  shell: true
  stdout: "{{ log_task_stdout }}"
  stderr: "{{ log_task_stderr }}"
  env:
    - O2_DETECTOR={{ detector }}
    - O2_PARTITION={{ environment_id }}
    - HOME=/tmp
  user: "{{ user }}"
  value: "{{ len(modulepath)>0 ? _module_cmdline : _plain_cmdline }}"
  arguments:
    - "-b"
    - "--exit-transition-timeout"
    - "'{{ exit_transition_timeout }}'"
    - "--monitoring-backend"
    - "'{{ monitoring_dpl_url }}'"
    - "--session"
    - "'{{ session_id }}'"
    - "--infologger-severity"
    - "'{{ infologger_severity }}'"
    - "--infologger-mode"
    - "'{{ infologger_mode }}'"
    - "--driver-client-backend"
    - "'stdout://'"
    - "--shm-segment-size"
    - "'{{ shm_segment_size }}'"
    - "--shm-throw-bad-alloc"
    - "'{{ shm_throw_bad_alloc }}'"
    - "--resources-monitoring"
    - "'{{ resources_monitoring }}'"
    - "--id"
    - "'C'"
    - "--shm-monitor"
    - "'false'"
    - "--log-color"
    - "'false'"
    - "--no-batch"
    - "--bad-alloc-attempt-interval"
    - "'50'"
    - "--bad-alloc-max-attempts"
    - "'1'"
    - "--channel-prefix"
    - "''"
    - "--early-forward-policy"
    - "'never'"
    - "--io-threads"
    - "'1'"
    - "--jobs"
    - "'1'"
    - "--severity"
    - "'info'"
    - "--shm-allocation"
    - "'rbtree_best_fit'"
    - "--shm-mlock-segment"
    - "'false'"
    - "--shm-mlock-segment-on-creation"
    - "'false'"
    - "--shm-no-cleanup"
    - "'false'"
    - "--shm-segment-id"
    - "'0'"
    - "--shm-zero-segment"
    - "'false'"
    - "--stacktrace-on-signal"
    - "'simple'"
    - "--timeframes-rate-limit"
    - "'0'"
)EXPECTED",
  R"EXPECTED(name: D
defaults:
  log_task_stdout: none
  log_task_stderr: none
  exit_transition_timeout: 15
  _module_cmdline: >-
    source /etc/profile.d/modules.sh && MODULEPATH={{ modulepath }} module load O2 QualityControl Control-OCCPlugin &&
    {{ dpl_command }} | foo
  _plain_cmdline: >-
    source /etc/profile.d/o2.sh && {{ len(extra_env_vars)>0 ? 'export ' + extra_env_vars + ' &&' : '' }} {{ dpl_command }} | foo
control:
  mode: "fairmq"
wants:
  cpu: 0.01
  memory: 1
bind:
  - name: outta_dpl
    type: push
    transport: shmem
    addressing: ipc
    rateLogging: "{{ fmq_rate_logging }}"
    global: "outta_dpl-{{ it }}"
command:
  shell: true
  stdout: "{{ log_task_stdout }}"
  stderr: "{{ log_task_stderr }}"
  env:
    - O2_DETECTOR={{ detector }}
    - O2_PARTITION={{ environment_id }}
    - HOME=/tmp
  user: "{{ user }}"
  value: "{{ len(modulepath)>0 ? _module_cmdline : _plain_cmdline }}"
  arguments:
    - "-b"
    - "--exit-transition-timeout"
    - "'{{ exit_transition_timeout }}'"
    - "--monitoring-backend"
    - "'{{ monitoring_dpl_url }}'"
    - "--session"
    - "'{{ session_id }}'"
    - "--infologger-severity"
    - "'{{ infologger_severity }}'"
    - "--infologger-mode"
    - "'{{ infologger_mode }}'"
    - "--driver-client-backend"
    - "'stdout://'"
    - "--shm-segment-size"
    - "'{{ shm_segment_size }}'"
    - "--shm-throw-bad-alloc"
    - "'{{ shm_throw_bad_alloc }}'"
    - "--resources-monitoring"
    - "'{{ resources_monitoring }}'"
    - "--id"
    - "'D'"
    - "--shm-monitor"
    - "'false'"
    - "--log-color"
    - "'false'"
    - "--no-batch"
    - "--bad-alloc-attempt-interval"
    - "'50'"
    - "--bad-alloc-max-attempts"
    - "'1'"
    - "--channel-prefix"
    - "''"
    - "--early-forward-policy"
    - "'never'"
    - "--io-threads"
    - "'1'"
    - "--jobs"
    - "'1'"
    - "--severity"
    - "'info'"
    - "--shm-allocation"
    - "'rbtree_best_fit'"
    - "--shm-mlock-segment"
    - "'false'"
    - "--shm-mlock-segment-on-creation"
    - "'false'"
    - "--shm-no-cleanup"
    - "'false'"
    - "--shm-segment-id"
    - "'0'"
    - "--shm-zero-segment"
    - "'false'"
    - "--stacktrace-on-signal"
    - "'simple'"
    - "--timeframes-rate-limit"
    - "'0'"
    - "--a-param"
    - "'1'"
    - "--b-param"
    - "''"
    - "--c-param"
    - "'foo;bar'"
)EXPECTED"};

TEST_CASE("TestO2ControlDump")
{
  auto workflow = defineDataProcessing();
  std::ostringstream ss{""};
  auto configContext = makeEmptyConfigContext();
  auto channelPolicies = makeTrivialChannelPolicies(*configContext);
  std::vector<DeviceSpec> devices;
  std::vector<ComputingResource> resources{ComputingResourceHelpers::getLocalhostResource()};
  SimpleResourceManager rm(resources);
  auto completionPolicies = CompletionPolicy::createDefaultPolicies();
  auto callbacksPolicies = CallbacksPolicy::createDefaultPolicies();
  DeviceSpecHelpers::dataProcessorSpecs2DeviceSpecs(workflow, channelPolicies, completionPolicies, callbacksPolicies, devices, rm, "workflow-id", *configContext, true);
  std::vector<DeviceControl> controls;
  std::vector<DeviceExecution> executions;
  controls.resize(devices.size());
  executions.resize(devices.size());
  CommandInfo commandInfo{R"(o2-exe --abdf -defg 'asdf fdsa' | o2-exe-2 -b --zxcv "asdf zxcv")"};

  std::vector<ConfigParamSpec> workflowOptions = {
    ConfigParamSpec{"jobs", VariantType::Int, 1, {"number of producer jobs"}}};

  std::vector<DataProcessorInfo> dataProcessorInfos = {
    {
      {"A", "bcsadc/foo", {}, workflowOptions},
      {"B", "foo", {}, workflowOptions},
      {"C", "foo", {}, workflowOptions},
      {"D", "foo", {}, workflowOptions},
    }};

  DriverConfig driverConfig{
    .batch = false,
  };
  DeviceSpecHelpers::prepareArguments(false, false, false, 8080,
                                      driverConfig,
                                      dataProcessorInfos,
                                      devices, executions, controls,
                                      "workflow-id");

  dumpWorkflow(ss, devices, executions, commandInfo, "testwf", "");

  REQUIRE(strdiffchr(ss.str().data(), expectedWorkflow) == strdiffchr(expectedWorkflow, ss.str().data()));
  REQUIRE(ss.str() == expectedWorkflow);

  REQUIRE(devices.size() == executions.size());
  REQUIRE(devices.size() == expectedTasks.size());
  for (size_t di = 0; di < devices.size(); ++di) {
    auto& spec = devices[di];
    auto& expected = expectedTasks[di];

    SECTION("Device " + std::string(spec.name))
    {
      ss.str({});
      ss.clear();
      dumpTask(ss, devices[di], executions[di], devices[di].name, "");
      REQUIRE(strdiffchr(ss.str().data(), expected) == strdiffchr(expected, ss.str().data()));
      REQUIRE(ss.str() == expected);
    }
  }
}
