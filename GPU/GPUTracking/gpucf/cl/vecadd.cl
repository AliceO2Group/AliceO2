kernel
void vecadd(global int *A,
            global int *B,
            global int *C)
{
    int idx = get_global_id(0);
    C[idx] = A[idx] + B[idx];
}
