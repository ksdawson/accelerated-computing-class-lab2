// Optional arguments:
//  -r <img_size>
//  -b <max iterations>
//  -i <implementation: {"scalar", "vector", "vector_ilp", "vector_multicore",
//  "vector_multicore_multithread", "vector_multicore_multithread_ilp", "all"}>

#include <cmath>
#include <cstdint>
#include <immintrin.h>
#include <pthread.h>

constexpr float window_zoom = 1.0 / 10000.0f;
constexpr float window_x = -0.743643887 - 0.5 * window_zoom;
constexpr float window_y = 0.131825904 - 0.5 * window_zoom;
constexpr uint32_t default_max_iters = 2000;

// Vector dimensions: 16x1, 8x2, 4x4, 2x4, 1x16
constexpr uint8_t VECTOR_SIZE = 16; // 16 32bit ints/floats in an AVX-512 vector
constexpr uint8_t VECTOR_WIDTH = 4; // Tuning parameter
constexpr uint8_t VECTOR_HEIGHT = 4; // Tuning parameter

// Thread dimensions
constexpr uint8_t THREADS_PER_CORE_SINGLE_THREAD = 1;
constexpr uint8_t THREADS_PER_CORE_MULTI_THREAD = 2; // Tuning parameter

// ILP dimensions
constexpr uint8_t NO_ILP = 1;
constexpr uint8_t ILP_HEIGHT = 2; // Tuning parameter
constexpr uint8_t ILP_WIDTH = 2; // Tuning parameter
constexpr uint8_t ILP_SIZE = ILP_HEIGHT * ILP_WIDTH;

// Vindexes for writing to memory
__m512i _1b16_vindex = _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0); // add img_size 0 to 0
__m512i _2b8_vindex = _mm512_set_epi32(7, 6, 5, 4, 3, 2, 1, 0, 7, 6, 5, 4, 3, 2, 1, 0); // add img_size 0 to 1
__m512i _4b4_vindex = _mm512_set_epi32(3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0); // add img_size 0 to 3
__m512i _8b2_vindex = _mm512_set_epi32(1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0); // add img_size 0 to 7
__m512i _16b1_vindex = _mm512_set1_epi32(0); // add img_size 0 to 15

// Memory functions
void write_row_vector_to_memory(uint32_t *out, __m512i vector, uint64_t i, uint64_t j, uint32_t img_size) {
    uint32_t *mem_addr = out + i * img_size + j;
    _mm512_storeu_si512(mem_addr, vector);
}
void write_tile_vector_to_memory(uint32_t *out, __m512i vector, uint64_t i, uint64_t j, uint32_t img_size, __m512i vindex) {
    uint32_t *mem_addr = out + i * img_size + j;
    _mm512_i32scatter_epi32(mem_addr, vindex, vector, 4);
}

// Vector constants
__m512 _4P0_VECTOR = _mm512_set1_ps(4.0f);
__m512i _1_VECTOR = _mm512_set1_epi32(1);

// We represent each pixel vector as a struct of all its relevant vectors
struct PixelVector {
    __m512 cy_vector;
    __m512 cx_vector;
    __m512 x2_vector;
    __m512 y2_vector;
    __m512 w_vector;
    __m512i iters_vector;
    __m512 x2y2_vector;
    __mmask16 while_condition_mask;
};

// Main function
template <uint8_t ILP_SIZE>
void mandelbrot_cpu_vector_helper(uint32_t img_size, uint32_t max_iters, uint32_t *out, // Image params
    uint32_t rows_start, uint32_t rows_end, uint32_t cols_start, uint32_t cols_end, // Multi core/thread params
    uint8_t block_vector_height=NO_ILP, uint8_t block_vector_width=NO_ILP // ILP params
) {
    // Vector constants
    __m512i max_iters_vector = _mm512_set1_epi32(max_iters);

    // Select the appropriate vindex for writing to memory
    __m512i vindex;
    if (VECTOR_WIDTH == 16) {
        vindex = _1b16_vindex;
    } else if (VECTOR_WIDTH == 8) {
        __m512i voffset = _mm512_set_epi32(img_size, img_size, img_size, img_size, img_size, img_size, img_size, img_size, 0, 0, 0, 0, 0, 0, 0, 0);
        vindex = _mm512_add_epi32(_2b8_vindex, voffset);
    } else if (VECTOR_WIDTH == 4) {
        __m512i voffset = _mm512_set_epi32(3*img_size, 3*img_size, 3*img_size, 3*img_size, 2*img_size, 2*img_size, 2*img_size, 2*img_size, img_size, img_size, img_size, img_size, 0, 0, 0, 0);
        vindex = _mm512_add_epi32(_4b4_vindex, voffset);
    } else if (VECTOR_WIDTH == 2) {
        __m512i voffset = _mm512_set_epi32(7*img_size, 7*img_size, 6*img_size, 6*img_size, 5*img_size, 5*img_size, 4*img_size, 4*img_size, 3*img_size, 3*img_size, 2*img_size, 2*img_size, img_size, img_size, 0, 0);
        vindex = _mm512_add_epi32(_8b2_vindex, voffset);
    } else if (VECTOR_WIDTH == 1) {
        __m512i voffset = _mm512_set_epi32(15*img_size, 14*img_size, 13*img_size, 12*img_size, 11*img_size, 10*img_size, 9*img_size, 8*img_size, 7*img_size, 6*img_size, 5*img_size, 4*img_size, 3*img_size, 2*img_size, img_size, 0);
        vindex = _mm512_add_epi32(_16b1_vindex, voffset);
    } else {
        // Incorrectly set parameters
        return;
    }

    // Block dimensions (pixels)
    uint64_t block_height = block_vector_height * VECTOR_HEIGHT;
    uint64_t block_width = block_vector_width * VECTOR_WIDTH;

    // We represent each pixel vector as a struct of all its relevant vectors
    PixelVector vector_block[ILP_SIZE];

    for (uint64_t block_i = rows_start / block_height; block_i < rows_end / block_height; ++block_i) {
        for (uint64_t block_j = cols_start / block_width; block_j < cols_end / block_width; ++block_j) {
            // Initialize our block of vectors
            for (uint8_t v = 0; v < ILP_SIZE; ++v) {
                // Vector coordinates
                uint8_t vector_i = v / block_vector_width;
                uint8_t vector_j = v % block_vector_width;
                // Pixel coordinates
                uint64_t pixel_i = block_i * block_height + vector_i * VECTOR_HEIGHT;
                uint64_t pixel_j = block_j * block_width + vector_j * VECTOR_WIDTH;
                // Get the vector
                auto &vector = vector_block[v];
                // Set the cy and cx vectors
                float cy_vector_vals[VECTOR_SIZE];
                float cx_vector_vals[VECTOR_SIZE];
                for (uint64_t vi = 0; vi < VECTOR_HEIGHT; ++vi) {
                    for (uint64_t vj = 0; vj < VECTOR_WIDTH; ++vj) {
                        cy_vector_vals[vi * VECTOR_WIDTH + vj] = (float(pixel_i + vi) / float(img_size)) * window_zoom + window_y;
                        cx_vector_vals[vi * VECTOR_WIDTH + vj] = (float(pixel_j + vj) / float(img_size)) * window_zoom + window_x;
                    }
                }
                vector.cy_vector = _mm512_loadu_ps(cy_vector_vals);
                vector.cx_vector = _mm512_loadu_ps(cx_vector_vals);
                // Set the other state vectors
                vector.x2_vector = _mm512_set1_ps(0.0f);
                vector.y2_vector = _mm512_set1_ps(0.0f);
                vector.w_vector = _mm512_set1_ps(0.0f);
                vector.iters_vector = _mm512_set1_epi32(0);
                vector.x2y2_vector = _mm512_set1_ps(0.0f);
                vector.while_condition_mask = 65535; // All vector lanes are true
            }

            // Inner loop math
            bool while_condition = true;
            uint32_t tot_iters = 0;
            while (while_condition && tot_iters < max_iters) {
                // Reset while condition
                while_condition = false;
                // Iterate over vectors in the block, interleaving the instructions to unlock ILP
                #pragma unroll
                for (uint8_t v = 0; v < ILP_SIZE; ++v) {
                    // Get the vector
                    auto &vector = vector_block[v];
                    // Skip vector if it's already done
                    if (vector.while_condition_mask == 0) {
                        // Do we lose some ILP here when we finish some vectors early?
                        continue;
                    }
                    // Update x2, y2, and w for all pixels in the vector
                    __m512 x_vector = _mm512_add_ps(_mm512_sub_ps(vector.x2_vector, vector.y2_vector), vector.cx_vector);
                    __m512 y_vector =_mm512_add_ps(_mm512_sub_ps(vector.w_vector, vector.x2y2_vector), vector.cy_vector);
                    vector.x2_vector = _mm512_mul_ps(x_vector, x_vector);
                    vector.y2_vector = _mm512_mul_ps(y_vector, y_vector);
                    vector.x2y2_vector = _mm512_add_ps(vector.x2_vector, vector.y2_vector);
                    __m512 z_vector = _mm512_add_ps(x_vector, y_vector);
                    vector.w_vector = _mm512_mul_ps(z_vector, z_vector);
                    // Only update iters for the pixels that are not done yet
                    vector.iters_vector = _mm512_mask_add_epi32(vector.iters_vector, vector.while_condition_mask, vector.iters_vector, _1_VECTOR);
                    // Update the vector while mask
                    vector.while_condition_mask = _mm512_cmp_ps_mask(vector.x2y2_vector, _4P0_VECTOR, _MM_CMPINT_LE);
                    // Update block while condition
                    while_condition = while_condition || (vector.while_condition_mask > 0);
                }
                ++tot_iters;
            }

            // Write results
            for (uint8_t v = 0; v < ILP_SIZE; ++v) {
                // Vector coordinates
                uint8_t vector_i = v / block_vector_width;
                uint8_t vector_j = v % block_vector_width;
                // Pixel coordinates
                uint64_t pixel_i = block_i * block_height + vector_i * VECTOR_HEIGHT;
                uint64_t pixel_j = block_j * block_width + vector_j * VECTOR_WIDTH;
                // Write vector to memory
                write_tile_vector_to_memory(out, vector_block[v].iters_vector, pixel_i, pixel_j, img_size, vindex);
            }
        }
    }
}

// CPU Scalar Mandelbrot set generation.
// Based on the "optimized escape time algorithm" in
// https://en.wikipedia.org/wiki/Plotting_algorithms_for_the_Mandelbrot_set
void mandelbrot_cpu_scalar(uint32_t img_size, uint32_t max_iters, uint32_t *out) {
    for (uint64_t i = 0; i < img_size; ++i) {
        for (uint64_t j = 0; j < img_size; ++j) {
            float cx = (float(j) / float(img_size)) * window_zoom + window_x;
            float cy = (float(i) / float(img_size)) * window_zoom + window_y;

            float x2 = 0.0f;
            float y2 = 0.0f;
            float w = 0.0f;
            uint32_t iters = 0;
            while (x2 + y2 <= 4.0f && iters < max_iters) {
                float x = x2 - y2 + cx;
                float y = w - (x2 + y2) + cy;
                x2 = x * x;
                y2 = y * y;
                float z = x + y;
                w = z * z;
                ++iters;
            }

            // Write result.
            out[i * img_size + j] = iters;
        }
    }
}

uint32_t ceil_div(uint32_t a, uint32_t b) { return (a + b - 1) / b; }

/// <--- your code here --->

// OPTIONAL: Uncomment this block to include your CPU vector implementation
// from Lab 1 for easy comparison.
//
// (If you do this, you'll need to update your code to use the new constants
// 'window_zoom', 'window_x', and 'window_y'.)

#define HAS_VECTOR_IMPL // <~~ keep this line if you want to benchmark the vector kernel!

////////////////////////////////////////////////////////////////////////////////
// Vector

void mandelbrot_cpu_vector(uint32_t img_size, uint32_t max_iters, uint32_t *out) {
    mandelbrot_cpu_vector_helper<NO_ILP>(img_size, max_iters, out, // Image parameters
        0, img_size, 0, img_size // Block dimensions
    );
}

////////////////////////////////////////////////////////////////////////////////
// Vector + ILP

void mandelbrot_cpu_vector_ilp(uint32_t img_size, uint32_t max_iters, uint32_t *out) {
    mandelbrot_cpu_vector_helper<ILP_SIZE>(img_size, max_iters, out, // Image params
        0, img_size, 0, img_size, // Multi core/thread params
        ILP_HEIGHT, ILP_WIDTH // ILP params
    );
}

////////////////////////////////////////////////////////////////////////////////
// Vector + Multi-core

// Struct for thread args
typedef struct {
    uint32_t img_size;
    uint32_t max_iters;
    uint32_t *out;
    uint32_t rows_start;
    uint32_t rows_end;
    uint32_t cols_start;
    uint32_t cols_end;
    uint8_t block_vector_height;
    uint8_t block_vector_width;
} ThreadArgs;

// Worker function
template <uint8_t ILP>
void *worker(void *arg) {
    ThreadArgs *args = (ThreadArgs *)arg;
    mandelbrot_cpu_vector_helper<ILP>(
        args->img_size, args->max_iters, args->out,
        args->rows_start, args->rows_end,
        args->cols_start, args->cols_end,
        args->block_vector_height, args->block_vector_width
    );
    return NULL;
}

template <uint8_t THREADS_PER_CORE, uint8_t ILP>
void mandelbrot_cpu_vector_multithread_helper(uint32_t img_size, uint32_t max_iters, uint32_t *out, // Image params
    uint8_t block_vector_height=NO_ILP, uint8_t block_vector_width=NO_ILP // ILP params
) {
    // 8 cores in our CPU
    constexpr uint8_t num_cores = 8;
    constexpr uint8_t n = num_cores * THREADS_PER_CORE;

    // Struct to hold the thread args
    ThreadArgs thread_args[n];

    // Keep track of the threads
    pthread_t threads[n];

    // Block dimensions: 8x1, 4x2, 2x4, 1x8
    uint32_t block_rows = 1; // Tuning parameter
    uint32_t block_cols = n / block_rows;
    uint32_t block_width = img_size / block_cols;
    uint32_t block_height = img_size / block_rows;

    // Spawn a thread for each block
    for (uint64_t block_i = 0; block_i < block_rows; ++block_i) {
        // Iteration dimensions
        uint32_t rows_start = block_i * block_height;
        uint32_t rows_end = rows_start + block_height;
        for (uint64_t block_j = 0; block_j < block_cols; ++block_j) {
            // Iteration dimensions
            uint32_t cols_start = block_j * block_width;
            uint32_t cols_end = cols_start + block_width;

            // Thread params
            uint8_t thread_idx = block_i * block_cols + block_j;
            thread_args[thread_idx] = { img_size, max_iters, out, rows_start, rows_end, cols_start, cols_end, block_vector_height, block_vector_width };

            pthread_create(&threads[thread_idx], NULL, worker<ILP>, &thread_args[thread_idx]);
        }
    }

    // Wait for all the threads to finish
    for (uint8_t thread_idx = 0; thread_idx < n; ++thread_idx) {
        pthread_join(threads[thread_idx], NULL);
    }
}

void mandelbrot_cpu_vector_multicore(
    uint32_t img_size,
    uint32_t max_iters,
    uint32_t *out) {
    mandelbrot_cpu_vector_multithread_helper<THREADS_PER_CORE_SINGLE_THREAD, NO_ILP>(img_size, max_iters, out);
}

////////////////////////////////////////////////////////////////////////////////
// Vector + Multi-core + Multi-thread-per-core

void mandelbrot_cpu_vector_multicore_multithread(
    uint32_t img_size,
    uint32_t max_iters,
    uint32_t *out) {
    mandelbrot_cpu_vector_multithread_helper<THREADS_PER_CORE_MULTI_THREAD, NO_ILP>(img_size, max_iters, out);
}

////////////////////////////////////////////////////////////////////////////////
// Vector + Multi-core + Multi-thread-per-core + ILP

void mandelbrot_cpu_vector_multicore_multithread_ilp(
    uint32_t img_size,
    uint32_t max_iters,
    uint32_t *out) {
    mandelbrot_cpu_vector_multithread_helper<THREADS_PER_CORE_MULTI_THREAD, ILP_SIZE>(img_size, max_iters, out, ILP_HEIGHT, ILP_WIDTH);
}

/// <--- /your code here --->

////////////////////////////////////////////////////////////////////////////////
///          YOU DO NOT NEED TO MODIFY THE CODE BELOW HERE.                  ///
////////////////////////////////////////////////////////////////////////////////

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <sys/types.h>
#include <vector>

// Useful functions and structures.
enum MandelbrotImpl {
    SCALAR,
    VECTOR,
    VECTOR_ILP,
    VECTOR_MULTICORE,
    VECTOR_MULTICORE_MULTITHREAD,
    VECTOR_MULTICORE_MULTITHREAD_ILP,
    ALL
};

// Command-line arguments parser.
int ParseArgsAndMakeSpec(
    int argc,
    char *argv[],
    uint32_t *img_size,
    uint32_t *max_iters,
    MandelbrotImpl *impl) {
    char *implementation_str = nullptr;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-r") == 0) {
            if (i + 1 < argc) {
                *img_size = atoi(argv[++i]);
                if (*img_size % 32 != 0) {
                    std::cerr << "Error: Image width must be a multiple of 32"
                              << std::endl;
                    return 1;
                }
            } else {
                std::cerr << "Error: No value specified for -r" << std::endl;
                return 1;
            }
        } else if (strcmp(argv[i], "-b") == 0) {
            if (i + 1 < argc) {
                *max_iters = atoi(argv[++i]);
            } else {
                std::cerr << "Error: No value specified for -b" << std::endl;
                return 1;
            }
        } else if (strcmp(argv[i], "-i") == 0) {
            if (i + 1 < argc) {
                implementation_str = argv[++i];
                if (strcmp(implementation_str, "scalar") == 0) {
                    *impl = SCALAR;
                } else if (strcmp(implementation_str, "vector") == 0) {
                    *impl = VECTOR;
                } else if (strcmp(implementation_str, "vector_ilp") == 0) {
                    *impl = VECTOR_ILP;
                } else if (strcmp(implementation_str, "vector_multicore") == 0) {
                    *impl = VECTOR_MULTICORE;
                } else if (
                    strcmp(implementation_str, "vector_multicore_multithread") == 0) {
                    *impl = VECTOR_MULTICORE_MULTITHREAD;
                } else if (
                    strcmp(implementation_str, "vector_multicore_multithread_ilp") == 0) {
                    *impl = VECTOR_MULTICORE_MULTITHREAD_ILP;
                } else if (strcmp(implementation_str, "all") == 0) {
                    *impl = ALL;
                } else {
                    std::cerr << "Error: unknown implementation" << std::endl;
                    return 1;
                }
            } else {
                std::cerr << "Error: No value specified for -i" << std::endl;
                return 1;
            }
        } else {
            std::cerr << "Unknown flag: " << argv[i] << std::endl;
            return 1;
        }
    }
    std::cout << "Testing with image size " << *img_size << "x" << *img_size << " and "
              << *max_iters << " max iterations." << std::endl;

    return 0;
}

// Output image writers: BMP file header structure
#pragma pack(push, 1)
struct BMPHeader {
    uint16_t fileType{0x4D42};   // File type, always "BM"
    uint32_t fileSize{0};        // Size of the file in bytes
    uint16_t reserved1{0};       // Always 0
    uint16_t reserved2{0};       // Always 0
    uint32_t dataOffset{54};     // Start position of pixel data
    uint32_t headerSize{40};     // Size of this header (40 bytes)
    int32_t width{0};            // Image width in pixels
    int32_t height{0};           // Image height in pixels
    uint16_t planes{1};          // Number of color planes
    uint16_t bitsPerPixel{24};   // Bits per pixel (24 for RGB)
    uint32_t compression{0};     // Compression method (0 for uncompressed)
    uint32_t imageSize{0};       // Size of raw bitmap data
    int32_t xPixelsPerMeter{0};  // Horizontal resolution
    int32_t yPixelsPerMeter{0};  // Vertical resolution
    uint32_t colorsUsed{0};      // Number of colors in the color palette
    uint32_t importantColors{0}; // Number of important colors
};
#pragma pack(pop)

void writeBMP(const char *fname, uint32_t img_size, const std::vector<uint8_t> &pixels) {
    uint32_t width = img_size;
    uint32_t height = img_size;

    BMPHeader header;
    header.width = width;
    header.height = height;
    header.imageSize = width * height * 3;
    header.fileSize = header.dataOffset + header.imageSize;

    std::ofstream file(fname, std::ios::binary);
    file.write(reinterpret_cast<const char *>(&header), sizeof(header));
    file.write(reinterpret_cast<const char *>(pixels.data()), pixels.size());
}

std::vector<uint8_t> iters_to_colors(
    uint32_t img_size,
    uint32_t max_iters,
    const std::vector<uint32_t> &iters) {
    uint32_t width = img_size;
    uint32_t height = img_size;
    uint32_t min_iters = max_iters;
    for (uint32_t i = 0; i < img_size; i++) {
        for (uint32_t j = 0; j < img_size; j++) {
            min_iters = std::min(min_iters, iters[i * img_size + j]);
        }
    }
    float log_iters_min = log2f(static_cast<float>(min_iters));
    float log_iters_range =
        log2f(static_cast<float>(max_iters) / static_cast<float>(min_iters));
    auto pixel_data = std::vector<uint8_t>(width * height * 3);
    for (uint32_t i = 0; i < height; i++) {
        for (uint32_t j = 0; j < width; j++) {
            uint32_t iter = iters[i * width + j];

            uint8_t r = 0, g = 0, b = 0;
            if (iter < max_iters) {
                auto log_iter = log2f(static_cast<float>(iter)) - log_iters_min;
                auto intensity = static_cast<uint8_t>(log_iter * 222 / log_iters_range);
                r = 32;
                g = 32 + intensity;
                b = 32;
            }

            auto index = (i * width + j) * 3;
            pixel_data[index] = b;
            pixel_data[index + 1] = g;
            pixel_data[index + 2] = r;
        }
    }
    return pixel_data;
}

// Benchmarking macros and configuration.
static constexpr size_t kNumOfOuterIterations = 10;
static constexpr size_t kNumOfInnerIterations = 1;
#define BENCHPRESS(func, ...) \
    do { \
        std::cout << std::endl << "Running " << #func << " ...\n"; \
        std::vector<double> times(kNumOfOuterIterations); \
        for (size_t i = 0; i < kNumOfOuterIterations; ++i) { \
            auto start = std::chrono::high_resolution_clock::now(); \
            for (size_t j = 0; j < kNumOfInnerIterations; ++j) { \
                func(__VA_ARGS__); \
            } \
            auto end = std::chrono::high_resolution_clock::now(); \
            times[i] = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start) \
                           .count() / \
                kNumOfInnerIterations; \
        } \
        std::sort(times.begin(), times.end()); \
        std::stringstream sstream; \
        sstream << std::fixed << std::setw(6) << std::setprecision(2) \
                << times[0] / 1'000'000; \
        std::cout << "  Runtime: " << sstream.str() << " ms" << std::endl; \
    } while (0)

double difference(
    uint32_t img_size,
    uint32_t max_iters,
    std::vector<uint32_t> &result,
    std::vector<uint32_t> &ref_result) {
    int64_t diff = 0;
    for (uint32_t i = 0; i < img_size; i++) {
        for (uint32_t j = 0; j < img_size; j++) {
            diff +=
                abs(int(result[i * img_size + j]) - int(ref_result[i * img_size + j]));
        }
    }
    return diff / double(img_size * img_size * max_iters);
}

void dump_image(
    const char *fname,
    uint32_t img_size,
    uint32_t max_iters,
    const std::vector<uint32_t> &iters) {
    // Dump result as an image.
    auto pixel_data = iters_to_colors(img_size, max_iters, iters);
    writeBMP(fname, img_size, pixel_data);
}

// Main function.
// Compile with:
//  g++ -march=native -O3 -Wall -Wextra -o mandelbrot mandelbrot_cpu.cc
int main(int argc, char *argv[]) {
    // Get Mandelbrot spec.
    uint32_t img_size = 1024;
    uint32_t max_iters = default_max_iters;
    enum MandelbrotImpl impl = ALL;
    if (ParseArgsAndMakeSpec(argc, argv, &img_size, &max_iters, &impl))
        return -1;

    // Allocate memory.
    std::vector<uint32_t> result(img_size * img_size);
    std::vector<uint32_t> ref_result(img_size * img_size);

    // Compute the reference solution
    mandelbrot_cpu_scalar(img_size, max_iters, ref_result.data());

    // Test the desired kernels.
    if (impl == SCALAR || impl == ALL) {
        memset(result.data(), 0, sizeof(uint32_t) * img_size * img_size);
        BENCHPRESS(mandelbrot_cpu_scalar, img_size, max_iters, result.data());
        dump_image("out/mandelbrot_cpu_scalar.bmp", img_size, max_iters, result);
    }

#ifdef HAS_VECTOR_IMPL
    if (impl == VECTOR || impl == ALL) {
        memset(result.data(), 0, sizeof(uint32_t) * img_size * img_size);
        BENCHPRESS(mandelbrot_cpu_vector, img_size, max_iters, result.data());
        dump_image("out/mandelbrot_cpu_vector.bmp", img_size, max_iters, result);

        std::cout << "  Correctness: average output difference from reference = "
                  << difference(img_size, max_iters, result, ref_result) << std::endl;
    }
#endif

    if (impl == VECTOR_ILP || impl == ALL) {
        memset(result.data(), 0, sizeof(uint32_t) * img_size * img_size);
        BENCHPRESS(mandelbrot_cpu_vector_ilp, img_size, max_iters, result.data());
        dump_image("out/mandelbrot_cpu_vector_ilp.bmp", img_size, max_iters, result);

        std::cout << "  Correctness: average output difference from reference = "
                  << difference(img_size, max_iters, result, ref_result) << std::endl;
    }

    if (impl == VECTOR_MULTICORE || impl == ALL) {
        memset(result.data(), 0, sizeof(uint32_t) * img_size * img_size);
        BENCHPRESS(mandelbrot_cpu_vector_multicore, img_size, max_iters, result.data());
        dump_image(
            "out/mandelbrot_cpu_vector_multicore.bmp",
            img_size,
            max_iters,
            result);

        std::cout << "  Correctness: average output difference from reference = "
                  << difference(img_size, max_iters, result, ref_result) << std::endl;
    }

    if (impl == VECTOR_MULTICORE_MULTITHREAD || impl == ALL) {
        memset(result.data(), 0, sizeof(uint32_t) * img_size * img_size);
        BENCHPRESS(
            mandelbrot_cpu_vector_multicore_multithread,
            img_size,
            max_iters,
            result.data());
        dump_image(
            "out/mandelbrot_cpu_vector_multicore_multithread.bmp",
            img_size,
            max_iters,
            result);

        std::cout << "  Correctness: average output difference from reference = "
                  << difference(img_size, max_iters, result, ref_result) << std::endl;
    }

    if (impl == VECTOR_MULTICORE_MULTITHREAD_ILP || impl == ALL) {
        memset(result.data(), 0, sizeof(uint32_t) * img_size * img_size);
        BENCHPRESS(
            mandelbrot_cpu_vector_multicore_multithread_ilp,
            img_size,
            max_iters,
            result.data());
        dump_image(
            "out/mandelbrot_cpu_vector_multicore_multithread_ilp.bmp",
            img_size,
            max_iters,
            result);

        std::cout << "  Correctness: average output difference from reference = "
                  << difference(img_size, max_iters, result, ref_result) << std::endl;
    }

    return 0;
}
