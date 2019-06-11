# ------------------------------------------------------------------------------
# O2_FORMAT
function(O2_FORMAT _output input prefix suffix)

  # DevNotes - input should be put in quotes or the complete list does not get
  # passed to the function
  set(format)
  foreach(arg ${input})
    set(item ${arg})
    if(prefix)
      string(REGEX MATCH "^${prefix}" pre ${arg})
    endif(prefix)
    if(suffix)
      string(REGEX MATCH "${suffix}$" suf ${arg})
    endif(suffix)
    if(NOT pre)
      set(item "${prefix}${item}")
    endif(NOT pre)
    if(NOT suf)
      set(item "${item}${suffix}")
    endif(NOT suf)
    list(APPEND format ${item})
  endforeach(arg)
  set(${_output} ${format} PARENT_SCOPE)

endfunction()
