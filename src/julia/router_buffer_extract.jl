# router_buffer_extract.jl — Iris F4d: muestras del buffer F4b para clases debiles
# Uso: julia router_buffer_extract.jl [dir_chunks_txt] [salida.tsv]   (desde iris/julia)
#
# Entrada: chunks DESCOMPRIMIDOS del buffer F4b (jsonl, campo "text"), en el dir
#          temporal opencode/buffer_txt con nombres tipo c4_es__c4_es_000000.jsonl.gz.jsonl
# Salida:  router_data_f4d.tsv = 674 (f4c) + muestras nuevas por ORIGEN de dataset:
#          c4_es -> lenguaje_general.web_es (~100), wikipedia_es ->
#          humanidades.enciclopedia_general (~60).
# Reglas: determinista (2 pasadas: cuenta + stride), min 200 chars, cap 900 (chop),
#         dedup exacto contra f4c y entre muestras nuevas, sin dependencias externas
#         (jobj/_limpia/read_tsv vienen de router_extractor.jl). No toca el nucleo.

using Printf
include(joinpath(@__DIR__, "router_extractor.jl"))

const MIN_CHARS = 200
const CAP = 300   # F4d: extractos cortos (cabecera); los 900 chars diluian el
                  # registro web en el promedio de embeddings (test: web_es 1/17)

# (prefijo de archivo, etiqueta, muestras por archivo)
const FUENTES = [
    ("c4_es__c4_es_",            "lenguaje_general.web_es",          25),
    ("wikipedia_es__wikipedia_", "humanidades.enciclopedia_general", 10),
]

chop900(s) = length(s) > CAP ? chop(s; tail = length(s) - CAP) : s

function valida(line)
    isempty(strip(line)) && return nothing
    d = jobj(line)
    tx = _limpia(get(d, "text", ""))
    (length(tx) >= MIN_CHARS) ? chop900(tx) : nothing
end

function main()
    raiz = normpath(@__DIR__, "..", "..")
    tmpdir = length(ARGS) >= 1 ? abspath(ARGS[1]) :
             get(ENV, "IRIS_BUFFER_TXT", joinpath(raiz, "datos", "buffer_txt"))
    outpath = length(ARGS) >= 2 ? abspath(ARGS[2]) : joinpath(raiz, "data", "router_data_f4d.tsv")

    base = joinpath(raiz, "data", "router_data_f4c.tsv")
    isfile(base) || error("falta $base; usa el TSV incluido o genera el tuyo")
    isdir(tmpdir) || error("no existe el directorio de chunks: $tmpdir`n" *
                           "descomprime chunks del buffer F4b o pasa la ruta como primer argumento")

    rows = read_tsv(base)
    n_f4c = length(rows)
    @printf("base f4c: %d muestras\n", n_f4c)

    # dedup: claves de los textos existentes, cap 900 Y sus primeros 300
    # (una muestra nueva corta puede ser prefijo de una larga del f4c)
    vistos = Set{String}()
    for (lab, tx) in rows
        push!(vistos, chop900(tx))
        push!(vistos, chop(tx; tail = max(0, length(tx) - CAP)))
    end

    files = sort(filter(f -> isfile(f), readdir(tmpdir; join = true)))
    isempty(files) && error("no hay chunks en $tmpdir")
    @printf("chunks: %d\n", length(files))

    por_label = Dict{String,Int}()
    nuenas = 0
    for (pref, lab, objetivo) in FUENTES
        mine = filter(f -> occursin(pref, basename(f)), files)
        isempty(mine) && @printf("(sin chunks de %s)\n", pref)
        for f in mine
            # pasada 1: contar lineas validas
            nvalid = 0
            for line in eachline(f)
                linea = strip(line)
                isempty(linea) && continue
                tx = try valida(linea) catch; nothing end
                tx === nothing || (nvalid += 1)
            end
            nvalid == 0 && @printf("(sin validas en %s)\n", basename(f))
            stride = max(1, cld(nvalid, objetivo))
            # pasada 2: tomar cada stride-esima valida
            iv = 0; cogidas = 0
            for line in eachline(f)
                linea = strip(line)
                isempty(linea) && continue
                tx = try valida(linea) catch; nothing end
                tx === nothing && continue
                iv += 1
                (iv % stride == 0) || continue
                cogidas < objetivo || break
                tx in vistos && continue
                push!(vistos, tx)
                push!(rows, (lab, tx))
                por_label[lab] = get(por_label, lab, 0) + 1
                nuenas += 1
                cogidas += 1
            end
            @printf("  %-42s validas %5d stride %4d cogidas %3d\n",
                    basename(f), nvalid, stride, cogidas)
        end
    end

    # control: 2 campos, sin tab/salto internos
    for (lab, tx) in rows
        isempty(tx) && error("texto vacio en $lab")
        findfirst('\t', tx) === nothing || error("tab interno en $lab")
        findfirst('\n', tx) === nothing || error("salto interno en $lab")
    end

    open(outpath, "w") do io
        for (lab, tx) in rows
            println(io, lab, '\t', tx)
        end
    end

    @printf("nuevas: %d -> total %d en %s\n", nuenas, length(rows), outpath)
    println("nuevas por etiqueta:")
    for k in sort(collect(keys(por_label)); by = x -> (-por_label[x], x))
        @printf("  %-36s %4d\n", k, por_label[k])
    end
    println("OK extractor buffer")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
