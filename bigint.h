#pragma once

#include <stdint.h>
#include <stddef.h>
#include <gmp.h>

#ifdef __cplusplus
class BigInt {
public:
    mpz_t z;
    int _initialized;
    
    BigInt() {
        mpz_init(z);
        _initialized = 0x600DB00B;
    }
    
    ~BigInt() {
        mpz_clear(z);
    }
    
    BigInt(const BigInt& other) {
        mpz_init_set(z, other.z);
        _initialized = 0x600DB00B;
    }
    
    BigInt& operator=(const BigInt& other) {
        if (this != &other) {
            mpz_set(z, other.z);
        }
        return *this;
    }
};

extern "C" {
#else
typedef struct BigInt { mpz_t z; int _initialized; } BigInt;
#endif

void bigint_clear(BigInt *a);
void bigint_destroy(BigInt *a);  /* Free GMP memory — call before scope exit */
void bigint_copy(BigInt *dst, const BigInt *src);
int bigint_is_zero(const BigInt *a);
int bigint_cmp(const BigInt *a, const BigInt *b);

void bigint_add(BigInt *result, const BigInt *a, const BigInt *b);
void bigint_sub(BigInt *result, const BigInt *a, const BigInt *b);
void bigint_mul(BigInt *result, const BigInt *a, const BigInt *b);

void bigint_shl1(BigInt *a);
void bigint_shr1(BigInt *a);
int  bigint_get_bit(const BigInt *a, uint32_t bit_index);
void bigint_set_bit(BigInt *a, uint32_t bit_index);
void bigint_clr_bit(BigInt *a, uint32_t bit_index);
uint32_t bigint_bitlen(const BigInt *a);

void bigint_set_u64(BigInt *a, uint64_t val);
uint64_t bigint_to_u64(const BigInt *a);

void bigint_div_mod(const BigInt *dividend, const BigInt *divisor,
                    BigInt *quotient, BigInt *remainder);
void bigint_gcd(BigInt *result, const BigInt *a, const BigInt *b);
void bigint_pow_mod(BigInt *result, const BigInt *base,
                    const BigInt *exp, const BigInt *mod);

int  bigint_from_decimal(BigInt *a, const char *str);
void bigint_to_decimal(char *buf, size_t bufsize, const BigInt *a);

#ifdef __cplusplus
}
#endif
