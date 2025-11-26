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

// #define AMD_LIBM_VEC_EXTERNAL_H
#define AMD_LIBM_VEC_EXPERIMENTAL

#include "amdlibm.h"
#include "amdlibm_vec.h"

namespace ARCH {
namespace da_metrics {
namespace pairwise_distances {

#if defined __AVX512F__
    #warning Using AVX512F
    #include <immintrin.h>
    // Perform Manhattan distance calculation on a single row of X and Y
    // Update a single entry in D
    // Return how many entries were not completed due to not fitting exactly in registers
    da_int minkowski_row(da_int k, double p, const double *row_X, const double *row_Y, double *D) {
        const da_int reg_cap = 8; // How many doubles can fit in the register
        da_int rem = k % reg_cap; // How many items left at the end of the row
        __m512d vec_p = _mm512_set1_pd(p);

        // Go through the rows of Y
        for (da_int l = 0; l <= k-reg_cap; l+=reg_cap) {
            __m512d vec_x = _mm512_loadu_pd(row_X + l);
            __m512d vec_y = _mm512_loadu_pd(row_Y + l);

            // Full subtract, full abs and then power p
            vec_y = _mm512_sub_pd(vec_x, vec_y);
            vec_y = _mm512_abs_pd(vec_y);
            vec_y = amd_vrd8_pow(vec_y, vec_p);

            // Update D
            *D += _mm512_reduce_add_pd(vec_y);
        }

        return rem;
    }

    da_int minkowski_row(da_int k, float p, const float *row_X, const float *row_Y, float *D) {
        const da_int reg_cap = 16; // How many doubles can fit in the register
        da_int rem = k % reg_cap; // How many items left at the end of the row
        __m512 vec_p = _mm512_set1_ps(p);

        // Go through the rows of Y
        for (da_int l = 0; l <= k-reg_cap; l+=reg_cap) {
            __m512 vec_x = _mm512_loadu_ps(row_X + l);
            __m512 vec_y = _mm512_loadu_ps(row_Y + l);

            // Full subtract, full abs and then power p
            vec_y = _mm512_sub_ps(vec_x, vec_y);
            vec_y = _mm512_abs_ps(vec_y);
            vec_y = amd_vrs16_powf(vec_y, vec_p);

            // Update D
            *D += _mm512_reduce_add_ps(vec_y);
        }

        return rem;
    }

    // Perform the final pth root on the distance matrix. Similar to the square root in euclidean
    void minkowski_root(double* x, da_int len, double invp) {
        const da_int reg_cap = 8; // How many doubles can fit in the register
        __m512d vec_invp = _mm512_set1_pd(invp);
        
        #pragma omp parallel for shared(reg_cap, x, len, vec_invp)
        for (da_int i = 0; i <= len-reg_cap; i+=reg_cap) {
            // Load vectors
            __m512d vec_x = _mm512_loadu_pd(x+i);

            // Pth root
            vec_x = amd_vrd8_pow(vec_x, vec_invp);

            // Store vector
            _mm512_storeu_pd(x+i, vec_x);
        }
        // Do serial pth root on the remainder
        da_int rem = len % reg_cap;
        for (da_int j = len-rem; j < len; ++j)
            x[j] = amd_pow(x[j], invp);
    }
    void minkowski_root(float* x, da_int len, float invp) {
        const da_int reg_cap = 16; // How many floats can fit in the register
        __m512 vec_invp = _mm512_set1_ps(invp);
        
        #pragma omp parallel for shared(reg_cap, x, len, vec_invp)
        for (da_int i = 0; i <= len-reg_cap; i+=reg_cap) {
            // Load vector
            __m512 vec_x = _mm512_loadu_ps(x+i);

            // Perform pth root
            vec_x = amd_vrs16_powf(vec_x, vec_invp);

            // Store vector
            _mm512_storeu_ps(x+i, vec_x);
        }
        // Do serial pth root on the remainder
        da_int rem = len % reg_cap;
        for (da_int j = len-rem; j < len; ++j)
            x[j] = amd_powf(x[j], invp);
    }
#elif defined __AVX2__
    #warning Using AVX2
    #include <immintrin.h>
    // Perform Manhattan distance calculation on a single row of X and Y
    // Update a single entry in D
    // Return how many entries were not completed due to not fitting exactly in registers
    da_int minkowski_row(da_int k, double p, const double *row_X, const double *row_Y, double *D) {
        const da_int reg_cap = 4; // How many doubles can fit in the register
        da_int rem = k % reg_cap; // How many items left at the end of the row
        __m256d neg_ones = _mm256_set1_pd(-1.0); // Vector of -1.0s. Useful for negating
        __m256d vec_p = _mm256_set1_pd(p);

        double temp_vec[reg_cap]; // Helper array for storing the vector result before summing to D

        // Go through the rows of Y
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

            // Full power p
            vec_x = amd_vrd4_pow(vec_x, vec_p);

            // Update D
            _mm256_storeu_pd(temp_vec, vec_x);
            *D += temp_vec[0] + temp_vec[1] + temp_vec[2] + temp_vec[3];
        }

        return rem;
    }

    da_int minkowski_row(da_int k, float p, const float *row_X, const float *row_Y, float *D) {
        const da_int reg_cap = 8; // How many doubles can fit in the register
        da_int rem = k % reg_cap; // How many items left at the end of the row
        __m256 neg_ones = _mm256_set1_ps(-1.0); // Vector of -1.0s. Useful for negating
        __m256 vec_p = _mm256_set1_ps(p);

        float temp_vec[reg_cap]; // Helper array for storing the vector result before summing to D

        // Go through the rows of Y
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

            // Full power p
            vec_x = amd_vrs8_powf(vec_x, vec_p);

            // Update D
            _mm256_storeu_ps(temp_vec, vec_x);
            *D += temp_vec[0] + temp_vec[1] + temp_vec[2] + temp_vec[3]
                + temp_vec[4] + temp_vec[5] + temp_vec[6] + temp_vec[7];
        }

        return rem;
    }

    // Perform the final pth root on the distance matrix. Similar to the square root in euclidean
    void minkowski_root(double* x, da_int len, double invp) {
        const da_int reg_cap = 4; // How many doubles can fit in the register
        __m256d vec_invp = _mm256_set1_pd(invp);
        
        #pragma omp parallel for shared(reg_cap, x, len, vec_invp)
        for (da_int i = 0; i <= len-reg_cap; i+=reg_cap) {
            // Load vectors
            __m256d vec_x = _mm256_loadu_pd(x+i);

            // Pth root
            vec_x = amd_vrd4_pow(vec_x, vec_invp);

            // Store vector
            _mm256_storeu_pd(x+i, vec_x);
        }
        // Do serial pth root on the remainder
        da_int rem = len % reg_cap;
        for (da_int j = len-rem; j < len; ++j)
            x[j] = amd_pow(x[j], invp);
    }
    void minkowski_root(float* x, da_int len, float invp) {
        const da_int reg_cap = 8; // How many floats can fit in the register
        __m256 vec_invp = _mm256_set1_ps(invp);
        
        #pragma omp parallel for shared(reg_cap, x, len, vec_invp)
        for (da_int i = 0; i <= len-reg_cap; i+=reg_cap) {
            // Load vector
            __m256 vec_x = _mm256_loadu_ps(x+i);

            // Perform pth root
            vec_x = amd_vrs8_powf(vec_x, vec_invp);

            // Store vector
            _mm256_storeu_ps(x+i, vec_x);
        }
        // Do serial pth root on the remainder
        da_int rem = len % reg_cap;
        for (da_int j = len-rem; j < len; ++j)
            x[j] = amd_powf(x[j], invp);
    }
#else
    #warning AVX unavailable
    template <typename T>
    inline da_int minkowski_row(da_int k, T p, const T *row_X, const T *row_Y, T *D) {
        return k;
    }

    template <typename T>
    void minkowski_root(T* x, da_int len, T invp) {
        #pragma omp parallel for shared(x, len, invp)
        for (da_int j = 0; j < len; ++j)
            x[j] = std::pow(x[j], invp);
    }
#endif

// Row major minkowski kernel
template <typename T>
void minkowski_kernel(da_int m, da_int n, da_int k, const T *X, da_int ldx,
                    const T *Y, da_int ldy, T *D, da_int ldd, T p) {
    T elementwise = 0.0;
    da_int rem = 0.0;
    // Go through the rows of D
    for (da_int i = 0; i < m; i++) {
        // Fill the row Di with zeros
        da_std::fill(D + i * ldd, D + i * ldd + n, 0.0);
        // Go through the columns of X (also updating the columns of D)
        for (da_int j = 0; j < n; j++) {
            // Vectorised minkowski
            rem = minkowski_row(k, p, X + i*ldx, Y + j*ldy, D + i*ldd + j);

            // Do any remaining elements serially
            for (da_int l = k-rem; l < k; l++) {
                elementwise = std::abs(X[i * ldx + l] - Y[j * ldy + l]);
                elementwise = std::pow(elementwise, p);
                D[i * ldd + j] += elementwise;
            }
        }
    }
}

template <typename T>
da_status minkowski(da_order order, da_int m, da_int n, da_int k, const T *X, da_int ldx,
                    const T *Y, da_int ldy, T *D, da_int ldd, T p) {
    da_status status = da_status_success;

    T invp = 1.0 / p;

    const T *X_new = X;
    const T *Y_new = Y;
    T *D_new = D;
    da_int ldd_new = ldd;
    // Create temporary vector X_row
    std::vector<T> X_row, D_row;
    if (order == column_major) {
        try {
            X_row.resize(m * k);
            D_row.resize(m * n);
        } catch (std::bad_alloc const &) {
            return da_status_memory_error;
        }
        // Transpose X so that the data is stored in row major order
        da_blas::omatcopy('T', m, k, 1.0, X_new, ldx, X_row.data(), k);
        D_new = D_row.data();
        X_new = X_row.data();
        ldx = k;
        ldd_new = n;
    }

    if (Y != nullptr) {
        // Create temporary vector Y_row
        std::vector<T> Y_row;
        if (order == column_major) {
            try {
                Y_row.resize(n * k);
            } catch (std::bad_alloc const &) {
                return da_status_memory_error;
            }
            // Transpose X and Y so that the data is stored in row major order
            da_blas::omatcopy('T', n, k, 1.0, Y_new, ldy, Y_row.data(), k);
            Y_new = Y_row.data();
            ldy = k;
        }

        da_int i_block_size = std::min(100, m);
        da_int j_block_size = std::min(100, n);
        #pragma omp parallel for shared(m, n, k, p, i_block_size, j_block_size, X_new, ldx, Y_new, ldy, D_new, ldd_new) collapse(2)
        for (da_int jj = 0; jj < n; jj+=j_block_size) {
            for (da_int ii = 0; ii < m; ii+=i_block_size) {
                int jb = std::min(n-jj, j_block_size);
                int ib = std::min(m-ii, i_block_size);

                minkowski_kernel(ib, jb, k,
                    X_new + ii*ldx, ldx,
                    Y_new + jj*ldy, ldy,
                    D_new + jj + ii*ldd_new, ldd_new, p);
            }
        }

        // Perform final pth root on D
        minkowski_root(D_new, m*n, invp);
    } else {
        // Symmetric case
        da_int block_size = std::min(100, m);
        #pragma omp parallel for shared(m, k, p, block_size, X_new, ldx, D_new, ldd_new) //collapse (2) //Can't collapse with aocc
        for (da_int jj = 0; jj < m; jj+=block_size) {
            for (da_int ii = 0; ii <= jj; ii+=block_size) {
                int jb = std::min(m-jj, block_size);
                int ib = std::min(m-ii, block_size);

                minkowski_kernel(ib, jb, k,
                    X_new + ii*ldx, ldx,
                    X_new + jj*ldx, ldx,
                    D_new + jj + ii*ldd_new, ldd_new, p);
            }
        }

        // Update the lower triangle accordingly
        #pragma omp parallel for shared(m, invp, D_new, ldd_new)
        for (int i = 0; i < m; i++)
            for (da_int j = 0; j < i; j++)
                D_new[j + i * ldd_new] = D_new[i + j * ldd_new];

        // minkowski_root(D_new + (i*ldd_new) + i, m-i, invp);
        minkowski_root(D_new, m*m, invp);
    }

    if (order == column_major) {
        // Transpose D to return data in column major order
        da_blas::omatcopy('T', n, m, 1.0, D_new, ldd_new, D, ldd);
    }

    return status;
}

template da_status minkowski<float>(da_order order, da_int m, da_int n, da_int k,
                                    const float *X, da_int ldx, const float *Y,
                                    da_int ldy, float *D, da_int ldd, float p);

template da_status minkowski<double>(da_order order, da_int m, da_int n, da_int k,
                                     const double *X, da_int ldx, const double *Y,
                                     da_int ldy, double *D, da_int ldd, double p);
} // namespace pairwise_distances
} // namespace da_metrics
} // namespace ARCH