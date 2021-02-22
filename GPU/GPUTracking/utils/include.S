	.global FILENAMEMOD
	.global FILENAMEMOD_size
	.section .data
FILENAMEMOD:
	.incbin "FILENAMENORMAL"
1:
	.byte 0
FILENAMEMOD_size:
	.int 1b - FILENAMEMOD
