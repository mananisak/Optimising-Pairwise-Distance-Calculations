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

namespace ARCH {
namespace da_metrics {
namespace pairwise_distances {

#if defined __AVX512F__
    #warning Using AVX512F
    #include <immintrin.h>
    // Perform Hassanat distance calculation on a single row of X and Y
    // Update a single entry in D
    // Return how many entries were not completed due to not fitting exactly in registers
    da_int hassanat_row(da_int k, const double *row_X, const double *row_Y, double *D) {
        const da_int reg_cap = 8; // How many doubles can fit in the register
        da_int rem = k % reg_cap; // How many items left at the end of the row

        // Go through the columns of both X and Y
        for (da_int l = 0; l <= k-reg_cap; l+=reg_cap) {
            __m512d vec_x_result = _mm512_loadu_pd(row_X + l);
            __m512d vec_y_diff = _mm512_loadu_pd(row_Y + l);
            __m512d vec_max = _mm512_max_pd(vec_x_result, vec_y_diff);
            // Three-way min between x, y and 0.0
            __m512d vec_min = _mm512_min_pd(vec_x_result, vec_y_diff);
            vec_min = _mm512_min_pd(vec_min, _mm512_setzero_pd());

            // Absolute difference. Reuse vec_y_diff
            vec_y_diff = _mm512_sub_pd(vec_x_result, vec_y_diff);
            vec_y_diff = _mm512_abs_pd(vec_y_diff);

            // Final hassanat calculation. Reuse vec_x_result for the result
            vec_x_result = _mm512_div_pd(
                                vec_y_diff,
                                _mm512_sub_pd(
                                    _mm512_add_pd(
                                        _mm512_set1_pd(1.0), 
                                        vec_max
                                    ),
                                    vec_min
                                )
                            );

            // Update D
            *D += _mm512_reduce_add_pd(vec_x_result);
        }

        return rem;
    }

    da_int hassanat_row(da_int k, const float *row_X, const float *row_Y, float *D) {
        const da_int reg_cap = 16; // How many floats can fit in the register
        da_int rem = k % reg_cap; // How many items left at the end of the row

        // Go through the columns of both X and Y
        for (da_int l = 0; l <= k-reg_cap; l+=reg_cap) {
            __m512 vec_x_result = _mm512_loadu_ps(row_X + l);
            __m512 vec_y_diff = _mm512_loadu_ps(row_Y + l);
            __m512 vec_max = _mm512_max_ps(vec_x_result, vec_y_diff);
            // Three-way min between x, y and 0.0
            __m512 vec_min = _mm512_min_ps(vec_x_result, vec_y_diff);
            vec_min = _mm512_min_ps(vec_min, _mm512_setzero_ps());

            // Absolute difference. Reuse vec_y_diff
            vec_y_diff = _mm512_sub_ps(vec_x_result, vec_y_diff);
            vec_y_diff = _mm512_abs_ps(vec_y_diff);

            // Final hassanat calculation. Reuse vec_x_result for the result
            // D = diff/(1+max(x,y)-min(x,y,0))
            vec_x_result = _mm512_div_ps(
                                vec_y_diff,
                                _mm512_sub_ps(
                                    _mm512_add_ps(
                                        _mm512_set1_ps(1.0),
                                        vec_max
                                    ),
                                    vec_min
                                )
                            );

            // Update D
            *D += _mm512_reduce_add_ps(vec_x_result);
        }

        return rem;
    }
#elif defined __AVX2__
    #warning Using AVX2
    #include <immintrin.h>
    // Perform Hassanat distance calculation on a single row of X and Y
    // Update a single entry in D
    // Return how many entries were not completed due to not fitting exactly in registers
    da_int hassanat_row(da_int k, const double *row_X, const double *row_Y, double *D) {
        const da_int reg_cap = 4; // How many doubles can fit in the register
        da_int rem = k % reg_cap; // How many items left at the end of the row
        double temp_vec[reg_cap]; // Helper array for storing the vector result before summing to D

        // Go through the columns of both X and Y
        for (da_int l = 0; l <= k-reg_cap; l+=reg_cap) {
            __m256d vec_x_result = _mm256_loadu_pd(row_X + l);
            __m256d vec_y_diff = _mm256_loadu_pd(row_Y + l);
            __m256d vec_max = _mm256_max_pd(vec_x_result, vec_y_diff);
            // Three-way min between x, y and 0.0
            __m256d vec_min = _mm256_min_pd(vec_x_result, vec_y_diff);
            vec_min = _mm256_min_pd(vec_min, _mm256_setzero_pd());

            // Absolute difference. Store in vec_y_diff and use vec_x_result as a helper
            // Generate mask to detect negative numbers
            __m256d mask = _mm256_cmp_pd(vec_x_result, vec_y_diff, _CMP_GT_OQ);
            
            // Full subtract
            vec_y_diff = _mm256_sub_pd(vec_x_result, vec_y_diff);

            // Negative version of the subtract
            vec_x_result = _mm256_mul_pd(vec_y_diff, _mm256_set1_pd(-1.0));

            // Blend the positive results from each vector together with the mask
            vec_y_diff = _mm256_blendv_pd(vec_x_result, vec_y_diff, mask);

            // Final hassanat calculation. Reuse vec_x_result for the result
            vec_x_result = _mm256_div_pd(
                                vec_y_diff,
                                _mm256_sub_pd(
                                    _mm256_add_pd(
                                        _mm256_set1_pd(1.0),
                                        vec_max
                                    ),
                                    vec_min
                                )
                            );

            // Update D
            _mm256_storeu_pd(temp_vec, vec_x_result);
            *D += temp_vec[0] + temp_vec[1] + temp_vec[2] + temp_vec[3];
        }

        return rem;
    }

    da_int hassanat_row(da_int k, const float *row_X, const float *row_Y, float *D) {
        const da_int reg_cap = 8; // How many floats can fit in the register
        da_int rem = k % reg_cap; // How many items left at the end of the row
        float temp_vec[reg_cap]; // Helper array for storing the vector result before summing to D

        // Go through the columns of both X and Y
        for (da_int l = 0; l <= k-reg_cap; l+=reg_cap) {
            __m256 vec_x_result = _mm256_loadu_ps(row_X + l);
            __m256 vec_y_diff = _mm256_loadu_ps(row_Y + l);
            __m256 vec_max = _mm256_max_ps(vec_x_result, vec_y_diff);
            // Three-way min between x, y and 0.0
            __m256 vec_min = _mm256_min_ps(vec_x_result, vec_y_diff);
            vec_min = _mm256_min_ps(vec_min, _mm256_setzero_ps());

            // Absolute difference. Store in vec_y_diff and use vec_x_result as a helper
            // Generate mask to detect negative numbers
            __m256 mask = _mm256_cmp_ps(vec_x_result, vec_y_diff, _CMP_GT_OQ);
            
            // Full subtract
            vec_y_diff = _mm256_sub_ps(vec_x_result, vec_y_diff);

            // Negative version of the subtract
            vec_x_result = _mm256_mul_ps(vec_y_diff, _mm256_set1_ps(-1.0));

            // Blend the positive results from each vector together with the mask
            vec_y_diff = _mm256_blendv_ps(vec_x_result, vec_y_diff, mask);

            // Final hassanat calculation. Reuse vec_x_result for the result
            vec_x_result = _mm256_div_ps(
                                vec_y_diff,
                                _mm256_sub_ps(
                                    _mm256_add_ps(
                                        _mm256_set1_ps(1.0),
                                        vec_max
                                    ),
                                    vec_min
                                )
                            );

            // Update D
            _mm256_storeu_ps(temp_vec, vec_x_result);
            *D += temp_vec[0] + temp_vec[1] + temp_vec[2] + temp_vec[3]
                + temp_vec[4] + temp_vec[5] + temp_vec[6] + temp_vec[7];
        }

        return rem;
    }
#else
    #warning AVX unavailable
    template <typename T>
    inline da_int hassanat_row(da_int k, const T *row_X, const T *row_Y, T *D) {
        return k;
    }
#endif

template <typename T>
void hassanat_kernel(da_order order, da_int m, da_int n, da_int k, const T *X, da_int ldx,
                    const T *Y, da_int ldy, T *D, da_int ldd) {
    const T eps = std::numeric_limits<T>::epsilon();

    T min, max, diff; // Helper variables to make the hassanat equation more readable

    if (order == column_major) {
        // Go through the columns of D
        for (da_int j = 0; j < n; j++) {
            // Fill the column Dj with zeros
            da_std::fill(D + j * ldd, D + j * ldd + m, 0.0);
            // Go through the columns of both X and Y
            for (da_int l = 0; l < k; l++) {
                // Go through the rows of X (also updating the rows of D)
                for (da_int i = 0; i < m; i++) {
                    min = std::min(std::min(X[i + l * ldx],Y[j + l * ldy]), eps); // min between x, y and 0
                    max = std::max(X[i + l * ldx],Y[j + l * ldy]);
                    diff = std::abs(X[i + l * ldx]-Y[j + l * ldy]);

                    D[i + j * ldd] += diff/(1 + max - min);
                }
            }
        }
    } else {
        da_int rem;
        // Go through the rows of D
        for (da_int i = 0; i < m; i++) {
            // Fill the row Di with zeros
            da_std::fill(D + i * ldd, D + i * ldd + n, 0.0);
            // Go through the columns of X (also updating the columns of D)
            for (da_int j = 0; j < n; j++) {
                rem = hassanat_row(k, X + i*ldx, Y + j*ldy, D + i*ldd + j);
            }
            // Do any remaining elements serially
            for (da_int l = k-rem; l < k; l++) {
                for (da_int j = 0; j < n; j++) {
                    min = std::min(std::min(X[i * ldx + l],Y[j * ldy + l]), eps);
                    max = std::max(X[i * ldx + l],Y[j * ldy + l]);
                    diff = std::abs(X[i * ldx + l]-Y[j * ldy + l]);

                    D[i * ldd + j] += diff/(1 + max - min);
                }
            }
        }
        // Go through the rows of D
        // for (da_int i = 0; i < m; i++) {
        //     // Fill the row Di with zeros
        //     da_std::fill(D + i * ldd, D + i * ldd + n, 0.0);
        //     // Go through the columns of both X and Y
        //     for (da_int l = 0; l < k; l++) {
        //         // Go through the columns of X (also updating the columns of D)
        //         for (da_int j = 0; j < n; j++) {
        //             // T x = X[i * ldx + l];
        //             // T y = Y_new[j * ldy + l];
        //             T min = std::min(X[i * ldx + l],Y[j * ldy + l]);
        //             T max = std::max(X[i * ldx + l],Y[j * ldy + l]);

        //             if (min >= 0) {
        //                 D[i * ldd + j] += std::abs(X[i * ldx + l]-Y[j * ldy + l])/(1 + max);
        //             } else {
        //                 D[i * ldd + j] += std::abs(X[i * ldx + l]-Y[j * ldy + l])/(1 + max - min);
        //             }
        //         }
        //     }
        // }
    }
}

template <typename T>
da_status hassanat(da_order order, da_int m, da_int n, da_int k, const T *X, da_int ldx,
                    const T *Y, da_int ldy, T *D, da_int ldd) {
    da_status status = da_status_success;

    const T *Y_new = Y;
    if (Y == nullptr) {
        n = m;
        ldy = ldx;
        Y_new = X;
    }
    
    // Block sizes
    da_int i_block_size = std::min(2048, m);
    da_int j_block_size = std::min(2048, n);
    
    #pragma omp parallel for shared(m, n, k, i_block_size, j_block_size, X, ldx, Y_new, ldy, D, ldd) collapse(2)
    for (da_int jj = 0; jj < n; jj+=j_block_size) {
        for (da_int ii = 0; ii < m; ii+=i_block_size) {
            int jb = std::min(n-jj, j_block_size);
            int ib = std::min(m-ii, i_block_size);

            if (order == column_major) {
                hassanat_kernel(order, ib, jb, k, 
                    X + ii, ldx,
                    Y_new + jj, ldy, 
                    D + ii + jj*ldd, ldd);
            } else {
                hassanat_kernel(order, ib, jb, k,
                    X + ii*ldx, ldx,
                    Y_new + jj*ldy, ldy,
                    D + jj + ii*ldd, ldd);
            }
        }
    }

    return status;
}

template da_status hassanat<float>(da_order order, da_int m, da_int n, da_int k,
                                    const float *X, da_int ldx, const float *Y,
                                    da_int ldy, float *D, da_int ldd);

template da_status hassanat<double>(da_order order, da_int m, da_int n, da_int k,
                                     const double *X, da_int ldx, const double *Y,
                                     da_int ldy, double *D, da_int ldd);
} // namespace pairwise_distances
} // namespace da_metrics
} // namespace ARCH