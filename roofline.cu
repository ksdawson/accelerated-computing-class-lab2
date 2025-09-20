// roofline.cu
//
// Empirically generate a roofline curve:
// - Sweeps arithmetic intensity (FLOPs per byte) by performing many FMAs per memory access.
// - Measures time using CUDA events and prints CSV: intensity,gflops,bandwidth_gBps,kernel_type
//
// Build:
//   nvcc -O3 roofline.cu -o roofline
//
// Run:
//   ./roofline
//

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>

#define CHECK(call)                                                   \
do {                                                                   \
    cudaError_t err = call;                                            \
    if (err != cudaSuccess) {                                          \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1);                                                       \
    }                                                                  \
} while (0)

// Parameterizable kernel:
// Each thread:
//   - loads one float from global memory
//   - performs `iters` iterations of an FMA-like sequence (r = r * a + b)
//     we will consider 2 FLOPs per iteration (mul+add).
//   - writes the result back to global memory.
//
// This yields bytes_per_thread = 4 (load) + 4 (store) = 8 bytes.
// Arithmetic intensity (FLOPs / byte) = (2 * iters) / 8 = iters / 4
// We compute iters to hit a desired intensity.

__global__ void ai_kernel(const float * __restrict__ in, float * __restrict__ out, int iters, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // Load once
    float r = in[idx];

    // Prevent compiler from optimizing the loop away
    // we use constants not known at compile time (but simplify).
    float a = 1.000001f;
    float b = 0.999999f;

    // Each iteration: one multiply + one add (count as 2 flops).
    #pragma unroll 4
    for (int i = 0; i < iters; ++i) {
        r = r * a + b; // FMA (or mul + add)
    }

    // Write result back
    out[idx] = r;
}

// Pure memory copy kernel (read and write sequentially) to estimate bandwidth.
__global__ void copy_kernel(const float * __restrict__ in, float * __restrict__ out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    // Strided loop to cover entire buffer if needed
    for (int i = idx; i < N; i += stride) {
        float v = in[i];
        out[i] = v;
    }
}

// Pure compute kernel: keep everything in registers, no global memory accesses
// Perform many FMAs to measure peak GFLOPS.
__global__ void compute_kernel(float * __restrict__ buf, int iters, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // initialize a few registers to chain operations
    float r0 = 1.0f + idx;
    float r1 = 1.5f + idx;
    float r2 = 2.0f + idx;
    float r3 = 2.5f + idx;

    float a = 1.000001f;
    float b = 0.999999f;

    for (int i = 0; i < iters; ++i) {
        // 4 FMAs per loop: 8 FLOPs (if counting mul+add separately)
        r0 = r0 * a + b;
        r1 = r1 * a + b;
        r2 = r2 * a + b;
        r3 = r3 * a + b;
    }

    // reduce and write to memory to avoid optimizing away
    buf[idx] = r0 + r1 + r2 + r3;
}

int main(int argc, char** argv) {
    // Size and kernel configs
    const int N = 1 << 24; // ~16.77M floats ~ 64 MB per array
    const int threads_per_block = 256;
    const int blocks = (N + threads_per_block - 1) / threads_per_block;

    printf("# roofline empirical measurement\n");
    printf("# N = %d elements (~%.2f MB per array)\n", N, (N * sizeof(float)) / (1024.0f * 1024.0f));

    // Allocate host pinned memory for faster host->device copies (optional)
    float *h_in = nullptr;
    float *h_out = nullptr;
    CHECK(cudaMallocHost(&h_in, N * sizeof(float)));
    CHECK(cudaMallocHost(&h_out, N * sizeof(float)));

    // Initialize input
    for (int i = 0; i < N; ++i) {
        h_in[i] = (float)(i & 255) * 0.001f + 1.0f;
        h_out[i] = 0.0f;
    }

    // Allocate device arrays
    float *d_in = nullptr, *d_out = nullptr;
    CHECK(cudaMalloc(&d_in, N * sizeof(float)));
    CHECK(cudaMalloc(&d_out, N * sizeof(float)));

    // Copy once
    CHECK(cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemset(d_out, 0, N * sizeof(float)));

    // Warmup
    ai_kernel<<<blocks, threads_per_block>>>(d_in, d_out, 1, N);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    // Prepare CUDA events for timing
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    // Sweep intensities: powers of two from 0.5 to 16384 (FLOPs/byte)
    // But because our bytes per thread = 8, and flops per iter = 2, intensity = (2 * iters) / 8 = iters / 4
    // So iters = 4 * intensity. We'll pick a set of intensities and compute iters accordingly.
    std::vector<float> intensities;
    // Add some low values, powers of two, and high values
    std::vector<float> base = {0.125f, 0.25f, 0.5f, 1.f};
    for (float b : base) intensities.push_back(b);
    for (int p = 0; p <= 14; ++p) { // 2^0 .. 2^14 = 16384
        intensities.push_back((float)(1 << p));
        intensities.push_back((float)(1 << p) * 3.0f / 2.0f); // a 1.5x point
    }
    // unique & sort
    std::sort(intensities.begin(), intensities.end());
    intensities.erase(std::unique(intensities.begin(), intensities.end()), intensities.end());

    // Print CSV header
    printf("intensity(FLOP_per_byte),gflops,bandwidth_GBps,kernel_type\n");

    // Helper lambda to measure ai kernel at a requested intensity
    auto run_ai = [&](float target_intensity) {
        // bytes per thread: 4 load + 4 store = 8
        const float bytes_per_thread = 8.0f;
        // FLOPs per iteration: we count 2 (mul + add) per loop iteration
        const int FLOPS_PER_ITER = 2;

        // iters needed per thread (float), then clamp to integer >= 0
        // intensity = (FLOPS_PER_ITER * iters) / bytes_per_thread
        // => iters = intensity * bytes_per_thread / FLOPS_PER_ITER
        double iters_d = (double)target_intensity * (double)bytes_per_thread / (double)FLOPS_PER_ITER;
        int iters = std::max(0, (int)std::ceil(iters_d));

        // ensure at least 1 iteration for non-zero intensity
        if (target_intensity > 0.0f && iters == 0) iters = 1;

        // total FLOPs and bytes for the whole kernel
        // total_flops = N_threads * iters * FLOPS_PER_ITER
        long long total_threads = (long long)N;
        long long total_flops = total_threads * (long long)iters * (long long)FLOPS_PER_ITER;
        long long total_bytes = total_threads * (long long) (int) bytes_per_thread;

        // run kernel and time
        CHECK(cudaEventRecord(start));
        ai_kernel<<<blocks, threads_per_block>>>(d_in, d_out, iters, N);
        CHECK(cudaGetLastError());
        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));
        float ms = 0.f;
        CHECK(cudaEventElapsedTime(&ms, start, stop)); // ms

        double secs = ms * 1e-3;
        double gflops = (double)total_flops / secs / 1e9;
        double bandwidth_gbps = (double)total_bytes / secs / 1e9; // GB/s

        // print CSV line
        printf("%.6e,%.6f,%.6f,ai_kernel\n", target_intensity, gflops, bandwidth_gbps);
    };

    // Run the sweep
    for (float I : intensities) {
        run_ai(I);
    }

    // Measure pure memory bandwidth (copy kernel)
    {
        // Memory traffic per element: load 4 bytes + store 4 bytes = 8 bytes.
        // We'll time the copy kernel which iterates with strided loops to cover buffer.
        CHECK(cudaEventRecord(start));
        copy_kernel<<<blocks, threads_per_block>>>(d_in, d_out, N);
        CHECK(cudaGetLastError());
        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));
        float ms = 0.f;
        CHECK(cudaEventElapsedTime(&ms, start, stop));
        double secs = ms * 1e-3;
        double total_bytes = (double)N * 8.0; // bytes
        double bandwidth = total_bytes / secs / 1e9;
        // For copy kernel intensity ~= 0 (very small)
        printf("%.6e,%.6f,%.6f,copy_kernel\n", 0.0, 0.0, bandwidth);
    }

    // Measure pure compute (peak GFLOPS) using compute_kernel
    {
        // choose large iters to saturate compute
        int iters = 16384; // tuneable
        long long total_threads = (long long)N;
        // compute kernel does 4 FMAs per iteration * 2 FLOPs per FMA = 8 FLOPs per iteration
        const int flops_per_iter = 8;
        long long total_flops = total_threads * (long long)iters * flops_per_iter;

        CHECK(cudaEventRecord(start));
        compute_kernel<<<blocks, threads_per_block>>>(d_out, iters, N);
        CHECK(cudaGetLastError());
        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));
        float ms = 0.f;
        CHECK(cudaEventElapsedTime(&ms, start, stop));
        double secs = ms * 1e-3;
        double gflops = (double)total_flops / secs / 1e9;
        // Compute-only: bandwidth ~ 0
        printf("%.6e,%.6f,%.6f,compute_kernel\n", 1e6, gflops, 0.0); // 1e6 intensity as placeholder
    }

    // Cleanup
    CHECK(cudaFree(d_in));
    CHECK(cudaFree(d_out));
    CHECK(cudaFreeHost(h_in));
    CHECK(cudaFreeHost(h_out));
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));

    return 0;
}