/****************************************************************************
 * cuda-skyline.cu - Parallel implementation of the skyline operator in cuda
 * 
 * by Giovanni Tentelli giovannni.tentelli@studio.unibo.it 0000921144
 * 
 * This program is a partial modification of skyline.c original implementation
 * of the skyline operator by Moreno Marzolla
 * 
 * Copyright (C) 2024 Moreno Marzolla
 * 
  * --------------------------------------------------------------------------
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * --------------------------------------------------------------------------
 * 
 * Questo programma calcola lo skyline di un insieme di punti in D
 * dimensioni letti da standard input.
 * 
 * Per compilare:
 * 
 *      nvcc cuda-skyline.cu -o cuda-skyline
 * 
 * Per eseguire il programma:
 * 
 *       ./cuda-skyline < input > output
 * 
 ****************************************************************************/

#if _XOPEN_SOURCE < 600
#define _XOPEN_SOURCE 600
#endif

#define BLKDIM 32
#define MAX_D 200

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "hpc.h"

typedef struct {
    float *P;   /* coordinates P[i][j] of point i               */
    int N;      /* Number of points (rows of matrix P)          */
    int D;      /* Number of dimensions (columns of matrix P)   */
} points_t;

/**
 * Read input from stdin. Input format is:
 *
 * d [other ignored stuff]
 * N
 * p0,0 p0,1 ... p0,d-1
 * p1,0 p1,1 ... p1,d-1
 * ...
 * pn-1,0 pn-1,1 ... pn-1,d-1
 *
 */
__host__ void read_input( points_t *points )
{
    char buf[1024];
    int N, D;
    float *P;

    if (1 != scanf("%d", &D)) {
        fprintf(stderr, "FATAL: can not read the dimension\n");
        exit(EXIT_FAILURE);
    }
    assert(D >= 2);
    if (NULL == fgets(buf, sizeof(buf), stdin)) { /* ignore rest of the line */
        fprintf(stderr, "FATAL: can not read the first line\n");
        exit(EXIT_FAILURE);
    }
    if (1 != scanf("%d", &N)) {
        fprintf(stderr, "FATAL: can not read the number of points\n");
        exit(EXIT_FAILURE);
    }
    P = (float*)malloc( D * N * sizeof(*P) );
    assert(P);
    for (int i=0; i<N; i++) {
        for (int k=0; k<D; k++) {
            if (1 != scanf("%f", &(P[i*D + k]))) {
                fprintf(stderr, "FATAL: failed to get coordinate %d of point %d\n", k, i);
                exit(EXIT_FAILURE);
            }
        }
    }
    points->P = P;
    points->N = N;
    points->D = D;
}

__host__ void free_points( points_t* points )
{
    free(points->P);
    points->P = NULL;
    points->N = points->D = -1;
}

/* Returns 1 if |p| dominates |q| */
__device__ int dominates( const float * p, const float * q, int D )
{
    int dominates = 0; 
    for (int i = 0; i < D; i++) {    
        if (p[i] < q[i]) {
            return 0;
        } else if (p[i] > q[i]){
            dominates = 1;
        }
    }
    return dominates ? 1 : 0;
}


/**
 * CUDA kernel to compute the skyline of a set of points.
 * Each thread determines whether a point in a D-dimensional space is part of the skyline.
 * A point is part of the skyline if it is not dominated by any other point.
 *
 * - N The number of points.
 * - D The number of dimensions.
 * - P Input array of N points of dimension D (size: N * D).
 * - s Output array indicating if a point is in the skyline (1 if true, 0 otherwise).
 * - r Counter for number of points in the compurted skyline.
 */
__global__ void skyline( const int N, const int D, const float *P, int *s, int *r)
{
    // Each thread is responsible for one point (index j).
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if ( j >= N ) return;

    // Copy point j into local memory (faster than repeated global reads).
    float local_P [MAX_D];
    for (int k = 0; k < D; ++k) {
        local_P[k] = P[j*D + k];
    }

    // Initially assume all the points are part of the skyline.
    int on_skyline = 1;

    for (int i=0; i<N; i++) {
        // If there exists a point i that dominates j, then j is not in the skyline.
        if ( dominates( &(P[i*D]), local_P, D ) ) {
            on_skyline = 0;
            break; // No need to continue checking.
        }
    }

    /*  Transfer from local to global memory. */
    s[j] = on_skyline;
    
    /*Decrement the counter for every point not part of the skyline. */
    __shared__ int block_counter;
    if (threadIdx.x == 0) block_counter = 0;
    __syncthreads();

    if (!on_skyline) atomicAdd(&block_counter, 1);
    __syncthreads();

    if (threadIdx.x == 0) atomicSub(r, block_counter);
}

/**
 * Print the coordinates of points belonging to the skyline `s` to
 * standard ouptut. `s[i] == 1` if point `i` belongs to the skyline.
 * The output format is the same as the input format, so that this
 * program can process its own output.
 */
__host__ void print_skyline( const points_t *points, const int *s, const int r )
{
    const int D = points->D;
    const int N = points->N;
    const float *P = points->P;

    printf("%d\n", D);
    printf("%d\n", r);
    for (int i=0; i<N; i++) {
        if ( s[i] ) {
            for (int k=0; k<D; k++) {
                printf("%f ", P[i*D + k]);
            }
            printf("\n");
        }
    }
}

int main( int argc, char* argv[] )
{
    points_t points;

    if (argc != 1) {
        fprintf(stderr, "Usage: %s < input_file > output_file\n", argv[0]);
        return EXIT_FAILURE;
    }

    read_input(&points);
    int *s = (int*)malloc(points.N * sizeof(*s));
    assert(s);
    int r = points.N;

    //For ease of use in cuda, N, D and P are dissected into separate pointers
    float *d_P;
    int *d_s; int *d_r;

    size_t size_of_P = points.N * points.D * sizeof(float);
    size_t size_of_s = points.N * sizeof(int);
    size_t size_of_r = sizeof(int);

    cudaSafeCall( cudaMalloc((void **) &d_P, size_of_P) );
    cudaSafeCall( cudaMalloc((void **) &d_s, size_of_s) );
    cudaSafeCall( cudaMalloc((void **) &d_r, size_of_r) );

    cudaSafeCall(  cudaMemcpy(d_P, points.P, size_of_P, cudaMemcpyHostToDevice) );
    cudaSafeCall( cudaMemcpy(d_r, &r, sizeof(int), cudaMemcpyHostToDevice) );

    dim3 grid( (points.N+BLKDIM-1)/BLKDIM );
    dim3 block(BLKDIM);

    cudaEvent_t start, stop;
    float elapsedTime;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start,0);
    skyline<<< grid, block >>> ( points.N, points.D, d_P, d_s, d_r );
    cudaCheckError();
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&elapsedTime, start,stop); // calolo del tempo in millisecondi 

    cudaSafeCall( cudaMemcpy(s, d_s, size_of_s, cudaMemcpyDeviceToHost) );
    cudaSafeCall( cudaMemcpy(&r, d_r, size_of_r, cudaMemcpyDeviceToHost) );
    cudaFree(d_P); cudaFree(d_s); cudaFree(d_r);

    print_skyline(&points, s, r);

    double time = (double)elapsedTime / 1000;
    double num_ops = (double)points.N * (double)points.N * (double)points.D;
    double ops_per_sec = num_ops / time;
    double gops = ops_per_sec / 1e9;

    fprintf(stderr, "\n\t%d points\n", points.N);
    fprintf(stderr, "\t%d dimensions\n", points.D);
    fprintf(stderr, "\t%d points in skyline\n\n", r);
    fprintf(stderr, "Execution time (s) %f\n", time);
    fprintf(stderr, "Throughput: %.2f GOps/sec\n", gops);

    free_points(&points);
    free(s);
    return EXIT_SUCCESS;
}