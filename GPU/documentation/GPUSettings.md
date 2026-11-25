The settings of the GPU reconstruction are defined once in macro lists, from which the structs, the command line options of the standalone benchmark, the O2 `ConfigurableParam`s, the help, and the RTC code are generated (`GPU/GPUTracking/utils/qconfig*`).

# Settings lists

- [GPUSettingsList.template](https://github.com/AliceO2Group/AliceO2/blob/dev/GPU/GPUTracking/Definitions/GPUSettingsList.template): all settings of the reconstruction, available in O2 and in the standalone benchmark: `rec` (reconstruction parameters, in GPU constant memory, with the sub-structs `rec.tpc`, `rec.trd`, `rec.dyn`), `proc` (processing, host only, with `proc.rtc`, `proc.rtctech`, `proc.param`, `proc.nn`, `proc.scaling`, `proc.debug`, `proc.backend`), `QA`, and `global` (O2 workflow settings).
- [GPUSettingsListDisplay.template](https://github.com/AliceO2Group/AliceO2/blob/dev/GPU/GPUTracking/Definitions/GPUSettingsListDisplay.template): event display settings (`display`, with `light`, `heavy`, `renderer`, `vulkan`, `internal`).
- [GPUSettingsListStandalone.template](https://github.com/AliceO2Group/AliceO2/blob/dev/GPU/GPUTracking/Definitions/GPUSettingsListStandalone.template): options only for the standalone benchmark (`configStandalone`, e.g. `--runs`, `--gpuType`, `--syncAsync`), not available in O2.
- [GPUDisplayControl.h](https://github.com/AliceO2Group/AliceO2/blob/dev/GPU/GPUTracking/display/GPUDisplayControl.h): display state without command line options (`GPUDisplayOptions`, saved / reset with the display settings, and `GPUDisplayControl`, transient control state).

CMake generates `GPUSettingsList*.h` from the templates into `include_gpu_onthefly` in the build folder, sorting the members of each struct by alignment. `rec` structs must not contain `bool` (CMake error), since they are copied to the GPU: use `int8_t` / `uint8_t`.

# Option names

- Standalone benchmark: `--[PREFIX][name]`, with the prefix of the block (`BeginSubConfig(..., "PREFIX", ...)`), e.g. `--RECTPCrejectQPtB5 20`, `--PROCdebugLevel 1`, `--RTCenable 1`, `--QAnoMC 1`. Options with an explicit name have it in addition, e.g. `--debug` / `-d` for `proc.debugLevel`. Standalone-only options have no prefix. `./ca --help`, `--helpall`, and `--[PREFIX]help` list the options, `--printSettings` prints all values.
- O2: each block is an O2 `ConfigurableParam` named `GPU_[o2prefix]` (last argument of `BeginSubConfig`), set via `--configKeyValues`, e.g. `--configKeyValues "GPU_rec_tpc.rejectQPtB5=20;GPU_proc.debugLevel=1;GPU_proc_rtc.enable=1"`.

# Macros

```
BeginSubConfig(StructName, instance, parent, "PREFIX", 'p', "description", o2prefix)
AddOption(name, type, default, "optname", 'o', "help", modifiers...)
EndConfig()
```

- `AddOption`: a setting with command line option and O2 key. `optname` `""` uses no extra name, `'o'` is an optional short name (0 for none).
- `AddVariable(name, type, default)`: struct member without option, e.g. for values derived from options (`GPUParam::UpdateSettings`). Any type is allowed.
- `AddOptionArray(name, type, count, (v0, v1, ...), ...)`: fixed-size array, the values are given space-separated on the command line, or as `GPU_x.name[i]=...` in O2.
- `AddOptionVec(name, type, ...)`: `std::vector`, standalone only (not in O2, not in device structs).
- `AddOptionSet(name, type, value, "optname", 'o', "help")`: additional option which sets an existing member to a fixed value, e.g. `-c` sets `runGPU = 0`.
- `AddSubConfig(Struct, instance)`, `AddHelp`, `AddHelpAll`, `AddShortcut`, `AddCommand`, `AddHelpText`, `AddCustomCPP` (code added to the struct).
- Entries must not end with `;`.

Supported types of options: `bool`, `int8_t` ... `int64_t`, `uint8_t` ... `uint64_t`, `float`, `double`, `std::string` (not for `...RTC` entries).

## RTC and non-RTC

`AddOptionRTC`, `AddVariableRTC`, `AddOptionArrayRTC` mark values read in GPU device code. With run time compilation and `--RTCoptConstexpr`, their current values are compiled in as `static constexpr` constants (see [run-time-compilation.md](https://github.com/AliceO2Group/AliceO2/blob/dev/GPU/documentation/run-time-compilation.md)). Otherwise they behave like the non-RTC versions. Use them only for values used in device code.

## Modifiers

Additional arguments after the help text:
- `def(x)`: value if the option is given without argument (e.g. `--display` = `--display 1`).
- `min(x)`, `max(x)`: range checked by the command line parser (standalone only).
- `set(x)`: the option takes no argument and sets `x`.
- `message("text %s")`: printed by the parser when the option is set, `%s` is the value.
- `qcheck(expression)`: checked at the initialization of `GPUReconstruction`, in O2 and standalone, if the value differs from the default; the initialization fails if `expression` is false. `val` is the value of the option, `get<T>()` gives another passed settings object, e.g. `qcheck(get<GPUSettingsDeviceBackend>().deviceType == gpudatatypes::DeviceType::OCL)`. The objects are passed in `GPUConfigDump::checkConfig`.

# Validation and derived values

- `GPUParam::UpdateSettings`: values derived from options.
- `GPUChainTracking::ValidateSettings`: consistency checks.
- `GPUChainTracking::ApplySyncSettings`: differences between synchronous and asynchronous reconstruction.
