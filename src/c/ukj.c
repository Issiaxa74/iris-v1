/* ukj.c — lector del formato UKJ v0.1 en C
 * - Abre el archivo con mmap (solo lectura): permite cargar UNA ruta sin leer
 *   todo el archivo al disco (base del límite de RAM de Iris).
 * - Parsea cabecera y tabla de rutas, verifica checksum y carga una ruta.
 * Compilar: gcc -O2 -o ukj_reader.exe ukj.c
 * Uso:      ukj_reader.exe <archivo.ukj> [indice_ruta]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#ifdef _WIN32
#  include <windows.h>
#else
#  include <fcntl.h>
#  include <unistd.h>
#  include <sys/mman.h>
#  include <sys/stat.h>
#endif

#define MAX_NIVELES 16
#define MAX_NIVEL_LEN 128

typedef struct {
    uint32_t id;
    uint16_t n_niveles;
    char     niveles[MAX_NIVELES][MAX_NIVEL_LEN];
    uint64_t offset;
    uint64_t nbytes;
    uint32_t dim;
} UkJRuta;

typedef struct {
    uint8_t*  map;      /* mmap completo */
    size_t    size;
    uint32_t  version;
    char      quant[256];
    uint32_t  n_rutas;
    uint32_t  flags;
    UkJRuta*  rutas;
} UkJ;

static uint16_t rd_u16(const uint8_t* p) { uint16_t v; memcpy(&v, p, 2); return v; }
static uint32_t rd_u32(const uint8_t* p) { uint32_t v; memcpy(&v, p, 4); return v; }
static uint64_t rd_u64(const uint8_t* p) { uint64_t v; memcpy(&v, p, 8); return v; }

static int mmap_file(const char* path, uint8_t** out, size_t* size) {
#ifdef _WIN32
    HANDLE fh = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL,
                            OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (fh == INVALID_HANDLE_VALUE) return -1;
    LARGE_INTEGER sz;
    if (!GetFileSizeEx(fh, &sz)) { CloseHandle(fh); return -1; }
    HANDLE mh = CreateFileMappingA(fh, NULL, PAGE_READONLY, 0, 0, NULL);
    if (!mh) { CloseHandle(fh); return -1; }
    void* view = MapViewOfFile(mh, FILE_MAP_READ, 0, 0, 0);
    CloseHandle(mh); CloseHandle(fh);
    if (!view) return -1;
    *out = (uint8_t*)view; *size = (size_t)sz.QuadPart;
    return 0;
#else
    int fd = open(path, O_RDONLY);
    if (fd < 0) return -1;
    struct stat st;
    if (fstat(fd, &st) < 0) { close(fd); return -1; }
    void* m = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (m == MAP_FAILED) return -1;
    *out = (uint8_t*)m; *size = (size_t)st.st_size;
    return 0;
#endif
}

static int parse_ukj(const char* path, UkJ* uk) {
    memset(uk, 0, sizeof *uk);
    if (mmap_file(path, &uk->map, &uk->size) != 0) return -1;
    const uint8_t* p = uk->map;
    const uint8_t* end = uk->map + uk->size;

    if (uk->size < 24 || memcmp(p, "UKJ\0", 4) != 0) return -2;
    uk->version = rd_u32(p + 4);
    uint8_t qlen = p[8];
    if (1 + (size_t)qlen > 255) return -3;
    memcpy(uk->quant, p + 9, qlen); uk->quant[qlen] = 0;
    uk->n_rutas = rd_u32(p + 9 + qlen);
    uk->flags   = rd_u32(p + 13 + qlen);
    p += 17 + qlen;

    uk->rutas = calloc(uk->n_rutas, sizeof(UkJRuta));
    if (!uk->rutas) return -4;
    for (uint32_t i = 0; i < uk->n_rutas; i++) {
        UkJRuta* r = &uk->rutas[i];
        r->id = rd_u32(p);          p += 4;
        r->n_niveles = rd_u16(p);   p += 2;
        if (r->n_niveles > MAX_NIVELES) return -5;
        for (uint16_t j = 0; j < r->n_niveles; j++) {
            uint16_t l = rd_u16(p); p += 2;
            if (l >= MAX_NIVEL_LEN || p + l > end) return -6;
            memcpy(r->niveles[j], p, l); r->niveles[j][l] = 0;
            p += l;
        }
        r->offset = rd_u64(p);      p += 8;
        r->nbytes = rd_u64(p);      p += 8;
        r->dim    = rd_u32(p);      p += 4;
        if (r->offset + r->nbytes > uk->size) return -7;
    }
    return 0;
}

/* Carga UNA ruta a memoria (lo que en Iris será la única porción en RAM) */
static int load_route(const UkJ* uk, uint32_t idx, float** buf, size_t* n_floats) {
    if (idx >= uk->n_rutas) return -1;
    const UkJRuta* r = &uk->rutas[idx];
    float* f = malloc((size_t)r->nbytes);
    if (!f) return -2;
    memcpy(f, uk->map + r->offset, (size_t)r->nbytes);
    *buf = f; *n_floats = (size_t)(r->nbytes / 4);
    return 0;
}

/* checksum v0.1: suma mod 2^64 de todos los bytes menos los últimos 8 */
static int verify_checksum(const UkJ* uk) {
    uint64_t sum = 0;
    for (size_t i = 0; i + 8 < uk->size; i++) sum += uk->map[i];
    uint64_t stored = rd_u64(uk->map + uk->size - 8);
    return sum == stored;
}

int main(int argc, char** argv) {
    const char* path = argc > 1 ? argv[1] : "test.ukj";
    uint32_t idx = argc > 2 ? (uint32_t)strtoul(argv[2], NULL, 10) : 0;
    UkJ uk;
    int rc = parse_ukj(path, &uk);
    if (rc != 0) { fprintf(stderr, "error parse %d\n", rc); return 1; }

    printf("version=%u quant=%s rutas=%u checksum=%s\n",
           uk.version, uk.quant, uk.n_rutas,
           verify_checksum(&uk) ? "OK" : "FALLO");
    for (uint32_t i = 0; i < uk.n_rutas; i++) {
        const UkJRuta* r = &uk.rutas[i];
        printf("ruta %u (", r->id);
        for (uint16_t j = 0; j < r->n_niveles; j++)
            printf("%s%s", j ? ">" : "", r->niveles[j]);
        printf(") offset=%llu nbytes=%llu dim=%u\n",
               (unsigned long long)r->offset,
               (unsigned long long)r->nbytes, r->dim);
    }

    float* f; size_t n;
    if (load_route(&uk, idx, &f, &n) == 0) {
        printf("carga parcial ruta %u: %zu floats [%.1f %.1f %.1f %.1f]\n",
               idx, n, f[0], f[1], f[2], f[3]);
        free(f);
    }
    free(uk.rutas);
    return 0;
}
