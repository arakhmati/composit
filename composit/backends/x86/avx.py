import codegen as c


def _mm256_load_ps(*args):
    return c.invoke(
        c.Identifier("_mm256_load_ps"),
        *args,
    )


def _mm256_fmadd_ps(*args):
    return c.invoke(
        c.Identifier("_mm256_fmadd_ps"),
        *args,
    )
