/*
 * Copyright (C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 3. Neither the name of the copyright holder nor the names of its contributors
 *    may be used to endorse or promote products derived from this software without
 *    specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
 * INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
 * OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 */

#include "aoclda.h"
#include "aoclda_types.h"
#include "da_cblas.hh"
#include "da_error.hpp"
#include "macros.h"
#include "pairwise_distances.hpp"
#include <iostream>
#include <vector>
#include <chrono>

namespace ARCH {

/*
An important kernel used repeatedly in k-means and SVM computations, among others.
Given an m by k matrix X and an n by k matrix Y (both column major), computes the m by n matrix D, where
D_{ij} is the Euclidean distance between row i of X and row j of Y.
Computes the distance by forming the norms of the rows of X and Y and computing XY^T which is more efficient
Various options are available:
- the squared norms of the rows of X and Y can be supplied precomputed or not used at all:
    compute_X/Y_norms = 0: do not use at all - note this is risky if you want to set square to be false as the function doesn't test for negative outputs
    compute_X/Y_norms = 1: use precomputed versions
    compute_X/Y_norms = 2: compute them in this function
- the square of the Euclidean distances can be returned if square is set to true
- if X_is_Y is true then X and Y are taken to be the same matrix, so only X is referenced and syrk is used
  instead of gemm and only the upper triangle is referenced and stored. Need m=n, otherwise garbage will come out
*/
#if defined __AVX512F__
    #warning Using AVX512F
    #include <immintrin.h>
    void sqrt_(double* x, da_int len) {
        const da_int reg_cap = 8; // How many doubles can fit in the register
        // Vectorised square root
        #pragma omp parallel for shared(reg_cap, x, len)
        for (da_int i = 0; i <= len-reg_cap; i+=reg_cap) {
            // Load vector
            __m512d vec_x = _mm512_loadu_pd(x+i);

            // Generate mask to detect negative numbers
            __mmask8 k = _mm512_cmp_pd_mask(vec_x, _mm512_setzero_pd(), _CMP_GT_OQ);

            // Perform masked square root, setting negative numbers to 0
            vec_x = _mm512_maskz_sqrt_pd(k, vec_x);

            // Store vector
            _mm512_storeu_pd(x+i, vec_x);
        }
        // Do serial square root on the remainder
        da_int rem = len % reg_cap;
        for (da_int j = len-rem; j < len; ++j)
            x[j] = (x[j] > 0) ? std::sqrt(x[j]) : 0.0;
    }
    void sqrt_(float* x, da_int len) {
        const da_int reg_cap = 16; // How many floats can fit in the register
        #pragma omp parallel for shared(reg_cap, x, len)
        for (da_int i = 0; i <= len-reg_cap; i+=reg_cap) {
            // Load vector
            __m512 vec_x = _mm512_loadu_ps(x+i);

            // Generate mask to detect negative numbers
            __mmask16 k = _mm512_cmp_ps_mask(vec_x, _mm512_setzero_ps(), _CMP_GT_OQ);

            // Perform masked square root, setting negative numbers to 0
            vec_x = _mm512_maskz_sqrt_ps(k, vec_x);

            // Store vector
            _mm512_storeu_ps(x+i, vec_x);
        }
        // Do serial square root on the remainder and set negatives to 0
        da_int rem = len % reg_cap;
        for (da_int j = len-rem; j < len; ++j)
            x[j] = (x[j] > 0) ? std::sqrt(x[j]) : 0.0;
    }
#elif defined __AVX2__
    #warning Using AVX2
    #include <immintrin.h>
    void sqrt_(double* x, da_int len) {
        const da_int reg_cap = 4; // How many doubles can fit in the register
        #pragma omp parallel for shared(reg_cap, x, len)
        for (da_int i = 0; i <= len-reg_cap; i+=reg_cap) {
            // Load array into register
            __m256d vec_x = _mm256_loadu_pd(x+i);

            // Generate mask to detect negative numbers
            __m256d mask = _mm256_cmp_pd(vec_x, _mm256_setzero_pd(), _CMP_GT_OQ);

            // Perform standard square root
            __m256d vec_sqrtx = _mm256_sqrt_pd(vec_x);

            // Blend a zero vector and the square root vector using the mask
            vec_sqrtx = _mm256_blendv_pd(_mm256_setzero_pd(), vec_sqrtx, mask);

            // Store vector
            _mm256_storeu_pd(x+i, vec_sqrtx);
        }
        // Do serial square root on the remainder and set negatives to 0
        da_int rem = len % reg_cap;
        for (da_int j = len-rem; j < len; ++j)
            x[j] = (x[j] > 0) ? std::sqrt(x[j]) : 0.0;
    }
    void sqrt_(float* x, da_int len) {
        const da_int reg_cap = 8; // How many floats can fit in the register
        #pragma omp parallel for shared(reg_cap, x, len)
        for (da_int i = 0; i <= len-reg_cap; i+=reg_cap) {
            // Load array into register
            __m256 vec_x = _mm256_loadu_ps(x+i);

            // Generate mask to detect negative numbers
            __m256 mask = _mm256_cmp_ps(vec_x, _mm256_setzero_ps(), _CMP_GT_OQ);

            // Perform standard square root
            __m256 vec_sqrtx = _mm256_sqrt_ps(vec_x);

            // Blend a zero vector and the square root vector using the mask
            vec_sqrtx = _mm256_blendv_ps(_mm256_setzero_ps(), vec_sqrtx, mask);

            // Store vector
            _mm256_storeu_ps(x+i, vec_sqrtx);
        }
        // Do serial square root on the remainder and set negatives to 0
        da_int rem = len % reg_cap;
        for (da_int j = len-rem; j < len; ++j)
            x[j] = (x[j] > 0) ? std::sqrt(x[j]) : 0.0;
    }
#else
    #warning AVX unavailable
    template <typename T>
    void sqrt_(T* x, da_int len) {
        #pragma omp parallel for shared(reg_cap, x, len)
        for (da_int i = 0; i < len; i++) {
            x[i] = (x[i] > 0) ? std::sqrt(x[i]) : 0.0;
        }
    }
#endif

template <typename T>
void euclidean_distance(da_order order, da_int m, da_int n, da_int k, const T *X,
                        da_int ldx, const T *Y, da_int ldy, T *D, da_int ldd, T *X_norms,
                        da_int compute_X_norms, T *Y_norms, da_int compute_Y_norms,
                        bool square, bool X_is_Y) {

    CBLAS_ORDER cblas_order =
        (order == column_major) ? CBLAS_ORDER::CblasColMajor : CBLAS_ORDER::CblasRowMajor;

    std::chrono::duration<double> nonblas_duration1;
    std::chrono::duration<double> nonblas_duration2;
    std::chrono::duration<double> blas_duration;
    auto nonblas_t1 = std::chrono::high_resolution_clock::now();
    auto nonblas_t0 = std::chrono::high_resolution_clock::now();

    // If needed, compute the squared norms of the rows of X and Y
    if (compute_X_norms == 2) {
        for (da_int i = 0; i < m; i++) {
            X_norms[i] = 0.0;
        }
        if (order == column_major) {
            #pragma omp parallel for shared(X, ldx, k, m) reduction(+:X_norms[0:m])
            for (da_int j = 0; j < k; j++) {
                for (da_int i = 0; i < m; i++) {
                    X_norms[i] += X[i + j * ldx] * X[i + j * ldx];
                }
            }
        } else {
            #pragma omp parallel for shared(X, ldx, k, m) reduction(+:X_norms[0:m])
            for (da_int i = 0; i < m; i++) {
                for (da_int j = 0; j < k; j++) {
                    X_norms[i] += X[i * ldx + j] * X[i * ldx + j];
                }
            }
        }
    }

    if (compute_Y_norms == 2 && !(X_is_Y)) {
        for (da_int i = 0; i < n; i++) {
            Y_norms[i] = 0.0;
        }
        if (order == column_major) {
            #pragma omp parallel for shared(Y, ldy, k, n) reduction(+:Y_norms[0:n])
            for (da_int j = 0; j < k; j++) {
                for (da_int i = 0; i < n; i++) {
                    Y_norms[i] += Y[i + j * ldy] * Y[i + j * ldy];
                }
            }
        } else {
            #pragma omp parallel for shared(Y, ldy, k, n) reduction(+:Y_norms[0:n])
            for (da_int i = 0; i < n; i++) {
                for (da_int j = 0; j < k; j++) {
                    Y_norms[i] += Y[i * ldy + j] * Y[i * ldy + j];
                }
            }
        }
    }

    if (!X_is_Y) {
        // A few different cases to check depending on the boolean inputs

        if (compute_X_norms == 0 && compute_Y_norms == 0) {
            if (order == column_major) {
                for (da_int j = 0; j < n; j++) {
                    for (da_int i = 0; i < m; i++) {
                        D[i + j * ldd] = 0.0;
                    }
                }
            } else {
                for (da_int i = 0; i < m; i++) {
                    for (da_int j = 0; j < n; j++) {
                        D[i * ldd + j] = 0.0;
                    }
                }
            }
        } else if (compute_X_norms > 0 && compute_Y_norms == 0) {
            if (order == column_major) {
                for (da_int j = 0; j < n; j++) {
                    for (da_int i = 0; i < m; i++) {
                        D[i + j * ldd] = X_norms[i];
                    }
                }
            } else {
                for (da_int i = 0; i < m; i++) {
                    for (da_int j = 0; j < n; j++) {
                        D[i * ldd + j] = X_norms[i];
                    }
                }
            }
        } else if (compute_X_norms == 0 && compute_Y_norms > 0) {
            if (order == column_major) {
                for (da_int j = 0; j < n; j++) {
                    for (da_int i = 0; i < m; i++) {
                        D[i + j * ldd] = Y_norms[j];
                    }
                }
            } else {
                for (da_int i = 0; i < m; i++) {
                    for (da_int j = 0; j < n; j++) {
                        D[i * ldd + j] = Y_norms[j];
                    }
                }
            }
        } else {
            if (order == column_major) {
                #pragma omp parallel for shared(D, X_norms, Y_norms, ldd, m, n)
                for (da_int j = 0; j < n; j++) {
                    for (da_int i = 0; i < m; i++) {
                        D[i + j * ldd] = X_norms[i] + Y_norms[j];
                    }
                }
            } else {
                #pragma omp parallel for shared(D, X_norms, Y_norms, ldd, m, n)
                for (da_int i = 0; i < m; i++) {
                    for (da_int j = 0; j < n; j++) {
                        D[i * ldd + j] = X_norms[i] + Y_norms[j];
                    }
                }
            }
        }

        nonblas_t1 = std::chrono::high_resolution_clock::now();
        nonblas_duration1 = nonblas_t1 - nonblas_t0;

        auto blas_t0 = std::chrono::high_resolution_clock::now();
        da_blas::cblas_gemm(cblas_order, CblasNoTrans, CblasTrans, m, n, k, -2.0, X, ldx,
                            Y, ldy, 1.0, D, ldd);
        auto blas_t1 = std::chrono::high_resolution_clock::now();
        blas_duration = blas_t1 - blas_t0;

        nonblas_t0 = std::chrono::high_resolution_clock::now();

        if (!square) {
            sqrt_(D, m*n);
        }
    } else {
        // Special case when computing upper triangle of symmetric distance matrix

        if (compute_X_norms == 0) {
            if (order == column_major) {
                #pragma omp parallel for shared(D, ldd, m)
                for (da_int j = 0; j < m; j++) {
                    for (da_int i = 0; i <= j; i++) {
                        D[i + j * ldd] = 0.0;
                    }
                }
            } else {
                #pragma omp parallel for shared(D, ldd, m)
                for (da_int i = 0; i < m; i++) {
                    for (da_int j = i; j < m; j++) {
                        D[i * ldd + j] = 0.0;
                    }
                }
            }
        } else {
            if (order == column_major) {
                #pragma omp parallel for shared(D, X_norms, ldd, m)
                for (da_int j = 0; j < m; j++) {
                    for (da_int i = 0; i <= j; i++) {
                        D[i + j * ldd] = X_norms[i] + X_norms[j];
                    }
                }
            } else {
                #pragma omp parallel for shared(D, X_norms, ldd, m)
                for (da_int i = 0; i < m; i++) {
                    for (da_int j = i; j < m; j++) {
                        D[i * ldd + j] = X_norms[i] + X_norms[j];
                    }
                }
            }
        }

        nonblas_t1 = std::chrono::high_resolution_clock::now();
        nonblas_duration1 = nonblas_t1 - nonblas_t0;

        auto blas_t0 = std::chrono::high_resolution_clock::now();
        da_blas::cblas_syrk(cblas_order, CblasUpper, CblasNoTrans, m, k, -2.0, X, ldx,
                            1.0, D, ldd);
        auto blas_t1 = std::chrono::high_resolution_clock::now();
        blas_duration = blas_t1 - blas_t0;

        nonblas_t0 = std::chrono::high_resolution_clock::now();

        // Ensure diagonal entries are precisely zero and perform square root if needed
        if (order == column_major) {
            #pragma omp parallel for shared(m, D, ldd)
            for (int i = 0; i < m; i++) {
                D[i + i * ldd] = 0.0;
                if (!square)
                    sqrt_(D + (i*ldd), i+1);
            }
        } else {
            #pragma omp parallel for shared(m, D, ldd)
            for (int i = 0; i < m; i++) {
                D[i + i * ldd] = 0.0;
                if (!square)
                    sqrt_(D + (i*ldd) + i, m-i);
            }
        }
    }

    nonblas_t1 = std::chrono::high_resolution_clock::now();
    nonblas_duration2 = nonblas_t1 - nonblas_t0;

    std::cout << "Non-Blas: " << nonblas_duration1.count() << " + " << nonblas_duration2.count()
    << " Blas: " << blas_duration.count() << std::endl;
}

template void euclidean_distance<float>(da_order order, da_int m, da_int n, da_int k,
                                        const float *X, da_int ldx, const float *Y,
                                        da_int ldy, float *D, da_int ldd, float *X_norms,
                                        da_int compute_X_norms, float *Y_norms,
                                        da_int compute_Y_norms, bool square, bool X_is_Y);

template void euclidean_distance<double>(da_order order, da_int m, da_int n, da_int k,
                                         const double *X, da_int ldx, const double *Y,
                                         da_int ldy, double *D, da_int ldd,
                                         double *X_norms, da_int compute_X_norms,
                                         double *Y_norms, da_int compute_Y_norms,
                                         bool square, bool X_is_Y);

namespace da_metrics {
namespace pairwise_distances {

template <typename T>
da_status euclidean(da_order order, da_int m, da_int n, da_int k, const T *X, da_int ldx,
                    const T *Y, da_int ldy, T *D, da_int ldd, bool square_distances) {
    da_status status = da_status_success;
    // Initialize X_is_Y.
    bool X_is_Y = false;
    // Allocate memory for compute_X_norms.
    std::vector<T> x_work, y_work;
    try {
        x_work.resize(m);
    } catch (std::bad_alloc const &) {
        return da_status_memory_error;
    }
    if (Y != nullptr) {
        try {
            y_work.resize(n);
        } catch (std::bad_alloc const &) {
            return da_status_memory_error;
        }
    } else {
        // Y is null pointer, set X_is_Y to true
        X_is_Y = true;
    }
    euclidean_distance(order, m, n, k, X, ldx, Y, ldy, D, ldd, x_work.data(), 2,
                       y_work.data(), 2, square_distances, X_is_Y);
    // If X_is_Y only the upper triangular part of the symmetric matrix D is computed in euclidean_distance.
    // Update the lower part accordingly before returning.
    if (X_is_Y) {
        if (order == column_major) {
            #pragma omp parallel for shared(D, m, ldd)
            for (da_int i = 0; i < m; i++)
                for (da_int j = 0; j < i; j++)
                    D[i + j * ldd] = D[j + i * ldd];
        } else {
            #pragma omp parallel for shared(D, m, ldd)
            for (da_int i = 0; i < m; i++)
                for (da_int j = 0; j < i; j++)
                    D[j + i * ldd] = D[i + j * ldd];
        }
    }

    return status;
}

template da_status euclidean<float>(da_order order, da_int m, da_int n, da_int k,
                                    const float *X, da_int ldx, const float *Y,
                                    da_int ldy, float *D, da_int ldd,
                                    bool square_distances);
template da_status euclidean<double>(da_order order, da_int m, da_int n, da_int k,
                                     const double *X, da_int ldx, const double *Y,
                                     da_int ldy, double *D, da_int ldd,
                                     bool square_distances);

} // namespace pairwise_distances
} // namespace da_metrics

} // namespace ARCH