/*
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
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
#include "da_std.hpp"
#include "pairwise_distances.hpp"
#include <omp.h>

namespace ARCH {
namespace da_metrics {
namespace pairwise_distances {

#if defined __AVX512F__
    #warning Using AVX512F
    #include <immintrin.h>
    // Perform Manhattan distance calculation on a single row of X and Y
    // Update a single entry in D
    // Return how many entries were not completed due to not fitting exactly in registers
    da_int manhattan_row(da_int k, const double *row_X, const double *row_Y, double *D) {
        const da_int reg_cap = 8; // How many doubles can fit in the register
        da_int rem = k % reg_cap; // How many items left at the end of the row

        // Go through the rows of X and Y
        for (da_int l = 0; l <= k-reg_cap; l+=reg_cap) {
            __m512d vec_x = _mm512_loadu_pd(row_X + l);
            __m512d vec_y = _mm512_loadu_pd(row_Y + l);

            // Full subtract and full abs. float abs only available in avx512
            vec_y = _mm512_sub_pd(vec_x, vec_y);
            vec_y = _mm512_abs_pd(vec_y);

            // Update D
            *D += _mm512_reduce_add_pd(vec_y);
        }

        return rem;
    }

    da_int manhattan_row(da_int k, const float *row_X, const float *row_Y, float *D) {
        const da_int reg_cap = 16; // How many doubles can fit in the register
        da_int rem = k % reg_cap; // How many items left at the end of the row

        // Go through the rows of X and Y
        for (da_int l = 0; l <= k-reg_cap; l+=reg_cap) {
            __m512 vec_x = _mm512_loadu_ps(row_X + l);
            __m512 vec_y = _mm512_loadu_ps(row_Y + l);

            // Full subtract and full abs. float abs only available in avx512
            vec_y = _mm512_sub_ps(vec_x, vec_y);
            vec_y = _mm512_abs_ps(vec_y);

            // Update D
            *D += _mm512_reduce_add_ps(vec_y);
        }

        return rem;
    }
#elif defined __AVX2__
    #warning Using AVX2
    #include <immintrin.h>
    da_int manhattan_row(da_int k, const double *row_X, const double *row_Y, double *D) {
        const da_int reg_cap = 4; // How many doubles can fit in the register
        da_int rem = k % reg_cap; // How many items left at the end of the row
        __m256d neg_ones = _mm256_set1_pd(-1.0); // Vector of -1.0s. Useful for negating

        double temp_vec[reg_cap]; // Helper array for storing the vector result before summing to D

        // Go through the rows of X and Y
        for (da_int l = 0; l <= k-reg_cap; l+=reg_cap) {
            __m256d vec_x = _mm256_loadu_pd(row_X + l);
            __m256d vec_y = _mm256_loadu_pd(row_Y + l);

            // Generate mask to detect negative numbers
            __m256d mask = _mm256_cmp_pd(vec_x, vec_y, _CMP_GT_OQ);
            
            // Full subtract. Reuse vec_x for the result
            vec_x = _mm256_sub_pd(vec_x, vec_y);

            // Negative version of the subtract. Reuse vec_y for the result
            vec_y = _mm256_mul_pd(vec_x, neg_ones);

            // Blend the positive results from each vector together with the mask
            vec_x = _mm256_blendv_pd(vec_y, vec_x, mask);

            // Update D
            _mm256_storeu_pd(temp_vec, vec_x);
            *D += temp_vec[0] + temp_vec[1] + temp_vec[2] + temp_vec[3];
        }

        return rem;
    }
    da_int manhattan_row(da_int k, const float *row_X, const float *row_Y, float *D) {
        const da_int reg_cap = 8; // How many floats can fit in the register
        da_int rem = k % reg_cap; // How many items left at the end of the row
        __m256 neg_ones = _mm256_set1_ps(-1.0); // Vector of -1.0s. Useful for negating

        float temp_vec[reg_cap]; // Helper array for storing the vector result before summing to D

        // Go through the rows of X and Y
        for (da_int l = 0; l <= k-reg_cap; l+=reg_cap) {
            __m256 vec_x = _mm256_loadu_ps(row_X + l);
            __m256 vec_y = _mm256_loadu_ps(row_Y + l);

            // Generate mask to detect negative numbers
            __m256 mask = _mm256_cmp_ps(vec_x, vec_y, _CMP_GT_OQ);
            
            // Full subtract. Reuse vec_x for the result
            vec_x = _mm256_sub_ps(vec_x, vec_y);

            // Negative version of the subtract. Reuse vec_y for the result
            vec_y = _mm256_mul_ps(vec_x, neg_ones);

            // Blend the positive results from each vector together with the mask
            vec_x = _mm256_blendv_ps(vec_y, vec_x, mask);

            // Update D
            _mm256_storeu_ps(temp_vec, vec_x);
            *D += temp_vec[0] + temp_vec[1] + temp_vec[2] + temp_vec[3]
                + temp_vec[4] + temp_vec[5] + temp_vec[6] + temp_vec[7];
        }

        return rem;
    }

#else
    #warning AVX unavailable
    template <typename T>
    inline da_int manhattan_row(da_int k, const T *row_X, const T *row_Y, T *D) {
        return k;
    }
#endif

template <typename T>
void manhattan_kernel(da_order order, da_int m, da_int n, da_int k, const T *X, da_int ldx,
                    const T *Y, da_int ldy, T *D, da_int ldd) {
    if (order == column_major) {
        // Go through the columns of D
        for (da_int j = 0; j < n; j++) {
            // Fill the column Dj with zeros
            da_std::fill(D + j * ldd, D + j * ldd + m, 0.0);
            // Go through the columns of both X and Y
            for (da_int l = 0; l < k; l++) {
                // Go through the rows of X (also updating the rows of D)
                for (da_int i = 0; i < m; i++) {
                    D[i + j * ldd] += std::abs(X[i + l * ldx] - Y[j + l * ldy]);
                }
            }
        }
    } else {
        da_int rem;
        // Go through the rows of D
        for (da_int i = 0; i < m; i++) {
            // Fill the row Di with zeros
            da_std::fill(D + i * ldd, D + i * ldd + n, 0.0);
            // Go through the columns of X and Y
            for (da_int j = 0; j < n; j++) {
                rem = manhattan_row(k, X + i*ldx, Y + j*ldy, D + i*ldd + j);
            }
            // Do any remaining elements serially
            for (da_int l = k-rem; l < k; l++)
                for (da_int j = 0; j < n; j++)
                    D[i * ldd + j] += std::abs(X[i * ldx + l] - Y[j * ldy + l]);
        }
    }
}

template <typename T>
da_status manhattan(da_order order, da_int m, da_int n, da_int k, const T *X, da_int ldx,
                    const T *Y, da_int ldy, T *D, da_int ldd) {
    da_status status = da_status_success;

    if (Y != nullptr) {
        // Block sizes
        da_int i_block_size = std::min(2048, m);
        da_int j_block_size = std::min(2048, n);
        #pragma omp parallel for shared(m, n, k, i_block_size, j_block_size, X, ldx, Y, ldy, D, ldd) collapse(2)
        for (da_int jj = 0; jj < n; jj+=j_block_size) {
            for (da_int ii = 0; ii < m; ii+=i_block_size) {
                int jb = std::min(n-jj, j_block_size);
                int ib = std::min(m-ii, i_block_size);

                if (order == column_major) {
                    manhattan_kernel(order, ib, jb, k,
                        X + ii, ldx,
                        Y + jj, ldy,
                        D + ii + jj*ldd, ldd);
                } else {
                    manhattan_kernel(order, ib, jb, k,
                        X + ii*ldx, ldx,
                        Y + jj*ldy, ldy,
                        D + jj + ii*ldd, ldd);
                }
            }
        }
    } else {
        // Symmetric case
        // Blocking is slower in the single thread case. This is a common case so it's worth accounting for
        // Also if the there is only one block, openmp only splits the blocks between threads (short and fat datasets)
        if (omp_get_max_threads() == 1 || m < 2048) {
            if (order == column_major) {
                // Go through the columns of D
                for (da_int j = 0; j < m; j++) {
                    // Fill the column Dj with zeros
                    da_std::fill(D + j * ldd, D + j * ldd + m, 0.0);
                    // Go through the columns of X
                    for (da_int l = 0; l < k; l++) {
                        // Go through the rows of X (also updating the rows of D)
                        for (da_int i = 0; i <= j; i++) {
                            D[i + j * ldd] += std::abs(X[i + l * ldx] - X[j + l * ldx]);
                        }
                    }
                }
                // Update the lower part accordingly before returning.
                for (da_int i = 0; i < m; i++)
                    for (da_int j = 0; j < i; j++)
                        D[i + j * ldd] = D[j + i * ldd];
            } else {
                da_int rem = 0;
                // Go through the rows of D
                for (da_int i = 0; i < m; i++) {
                    // Fill the row Di with zeros
                    da_std::fill(D + i * ldd, D + i * ldd + m, 0.0);
                    // Go through the columns of X
                    for (da_int j = i; j < m; j++) {
                        rem = manhattan_row(k, X + i*ldx, X + j*ldx, D + i*ldd + j);
                    }
                    // Do any remaining elements serially
                    for (da_int l = k-rem; l < k; l++)
                        for (da_int j = i; j < m; j++)
                            D[i * ldd + j] += std::abs(X[i * ldx + l] - X[j * ldx + l]);
                }
                // Update the lower part accordingly before returning.
                for (da_int i = 0; i < m; i++)
                    for (da_int j = 0; j < i; j++)
                        D[j + i * ldd] = D[i + j * ldd];
            }
        } else {
            // Blocking is required for effective parallelisation in this case
            // The larger the block size is relative to m, the less calculations are skipped.
            // It might be worth making the block size for the symmetric case a function of m.
            // Additionally, can't use omp collapse(2) in aocc/clang
            da_int block_size = std::min(2048, m);
            #pragma omp parallel for shared(m, k, block_size, X, ldx, D, ldd) collapse(2)
            for (da_int jj = 0; jj < m; jj+=block_size) {
                for (da_int ii = 0; ii <= jj; ii+=block_size) {
                    int jb = std::min(m-jj, block_size);
                    int ib = std::min(m-ii, block_size);

                    if (order == column_major) {
                        manhattan_kernel(order, ib, jb, k,
                            X + ii, ldx,
                            X + jj, ldx,
                            D + ii + jj*ldd, ldd);
                    } else {
                        manhattan_kernel(row_major, ib, jb, k,
                            X + ii*ldx, ldx,
                            X + jj*ldx, ldx,
                            D + jj + ii*ldd, ldd);
                    }
                }
            }

            // Update the lower triangle accordingly
            if (order == column_major) {
                #pragma omp parallel for shared(m,D,ldd) collapse(2)
                for (da_int i = 0; i < m; i++)
                    for (da_int j = 0; j < i; j++)
                        D[i + j * ldd] = D[j + i * ldd];
            } else {
                #pragma omp parallel for shared(m,D,ldd) collapse(2)
                for (da_int i = 0; i < m; i++)
                    for (da_int j = 0; j < i; j++)
                        D[j + i * ldd] = D[i + j * ldd];
            }
        }
    }

    return status;
}

template da_status manhattan<float>(da_order order, da_int m, da_int n, da_int k,
                                    const float *X, da_int ldx, const float *Y,
                                    da_int ldy, float *D, da_int ldd);

template da_status manhattan<double>(da_order order, da_int m, da_int n, da_int k,
                                     const double *X, da_int ldx, const double *Y,
                                     da_int ldy, double *D, da_int ldd);
} // namespace pairwise_distances
} // namespace da_metrics
} // namespace ARCH