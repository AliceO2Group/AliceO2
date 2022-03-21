#!/bin/bash

o2Warnings=(
'pointer-sign'
'override-init'
'catch-value'
'pessimizing-move'
'reorder'
'delete-non-virtual-dtor'
'deprecated-copy'
'redundant-move'
'overloaded-virtual'
'address'
'bool-compare'
'bool-operation'
'char-subscripts'
'comment'
'enum-compare'
'format'
'format-overflow'
'format-truncation'
'int-in-bool-context'
'init-self'
'logical-not-parentheses'
'maybe-uninitialized'
'memset-elt-size'
'memset-transposed-args'
'misleading-indentation'
'missing-attributes'
'multistatement-macros'
'narrowing'
'nonnull'
'nonnull-compare'
'openmp-simd'
'parentheses'
'restrict'
'return-type'
'sequence-point'
'sign-compare'
'sizeof-pointer-div'
'sizeof-pointer-memaccess'
'strict-aliasing'
'strict-overflow'
'switch'
'tautological-compare'
'trigraphs'
'uninitialized'
'unused-label'
'unused-value'
'unused-variable'
'volatile-register-var'
'zero-length-bounds'
'unused-but-set-variable'
'stringop-truncation'
'clobbered'
'cast-function-type'
'empty-body'
'ignored-qualifiers'
'implicit-fallthrough'
'missing-field-initializers'
'sign-compare'
'string-compare'
'type-limits'
'uninitialized'
'shift-negative-value'
	)

logFile=${1}
warningsFile="$(dirname "${logFile}" )/warnings"

grep "warning:" ${logFile} | sort | uniq > ${warningsFile}

nTotalWarnings="$(cat ${warningsFile} | grep -c "warning:")"

printf "Total warnings: ${nTotalWarnings}\n" 
printf "##################################\n"
for warning in ${o2Warnings[@]}; do
	nWarnings=$(cat ${warningsFile} | grep -c "${warning}")
	printf '%-30s:\t%6s\t%8s\n' "${warning}" "${nWarnings}" "$(python3 -c "print(\"{:>.1%} \".format(${nWarnings} / ${nTotalWarnings}))")"
done
