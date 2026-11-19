# router_extractor.jl — Iris F4c: convierte el destilado (JSONL) en muestras del router
# Uso: julia router_extractor.jl [dir_destilado] [salida.tsv]   (desde iris/julia)
#
# Entrada: Shared/IRIS_DESTILADO/descargas/*.jsonl (213 pares prompt/response con
#          ruta explicita tipo "humanidades->historia->tema->modo", separador U+2192).
# Salida:  router_data_f4c.tsv = 248 muestras originales de F3 + 2 muestras por par
#          (el prompt y la respuesta), etiqueta = primeros 2 niveles de la ruta
#          (mundo.submundo), la misma granularidad del TSV de F3.
# Reglas: determinista, sin dependencias externas (parser JSON minimal propio),
#         no toca router_data.tsv original ni el resto del nucleo.

using Printf

const FLECHA = string(Char(0x2192))   # separador de rutas del destilado

# ----------------------------- JSON minimal -----------------------------
# Suficiente para objetos planos con claves string; salta numeros/bools/objetos.

_skipws(s, i) = begin
    n = ncodeunits(s)
    while i <= n
        b = codeunit(s, i)
        (b == UInt8(' ') || b == UInt8('\t') || b == UInt8('\r') || b == UInt8('\n')) || break
        i += 1
    end
    return i
end

_hex4(s, i) = parse(UInt32, s[(i + 1):(i + 4)]; base = 16)

_utf8(cp) = Vector{UInt8}(codeunits(string(Char(cp))))

function _jstring(s::AbstractString, i::Int)
    codeunit(s, i) == UInt8('"') || error("json: se esperaba '\"' en $i")
    buf = UInt8[]
    i += 1
    n = ncodeunits(s)
    while i <= n
        b = codeunit(s, i)
        if b == UInt8('"')
            return String(buf), i + 1
        elseif b == UInt8('\\')
            e = codeunit(s, i + 1)
            if e == UInt8('u')
                cp = _hex4(s, i + 1)
                adv = 6
                if 0xD800 <= cp <= 0xDBFF  # surrogado alto: lee el bajo
                    codeunit(s, i + 6) == UInt8('\\') || error("json: surrogado sin pareja")
                    codeunit(s, i + 7) == UInt8('u') || error("json: surrogado sin pareja")
                    lo = _hex4(s, i + 7)
                    cp = 0x10000 + (cp - 0xD800) * 0x400 + (lo - 0xDC00)
                    adv = 12
                end
                append!(buf, _utf8(cp))
                i += adv
            else
                c = e == UInt8('n') ? '\n' : e == UInt8('t') ? '\t' :
                    e == UInt8('r') ? '\r' : e == UInt8('b') ? '\b' :
                    e == UInt8('f') ? '\f' : e == UInt8('"') ? '"' :
                    e == UInt8('\\') ? '\\' : e == UInt8('/') ? '/' :
                    error("json: escape desconocido \\$(Char(e))")
                append!(buf, codeunits(string(c)))
                i += 2
            end
        else
            push!(buf, b)
            i += 1
        end
    end
    error("json: string sin cerrar")
end

function _jskip(s::AbstractString, i::Int)
    n = ncodeunits(s)
    b = codeunit(s, i)
    if b == UInt8('"')
        _, i = _jstring(s, i)
        return i
    elseif b == UInt8('{') || b == UInt8('[')
        cerra = b == UInt8('{') ? UInt8('}') : UInt8(']')
        depth = 0
        while i <= n
            c = codeunit(s, i)
            if c == UInt8('"')
                _, i = _jstring(s, i)
                continue
            end
            (c == UInt8('{') || c == UInt8('[')) && (depth += 1)
            (c == cerra || c == UInt8('}')) && (depth -= 1)
            i += 1
            depth == 0 && return i
        end
        error("json: valor compuesto sin cerrar")
    else
        while i <= n
            c = codeunit(s, i)
            (c == UInt8(',') || c == UInt8('}') || c == UInt8(']') ||
             c == UInt8(' ') || c == UInt8('\n') || c == UInt8('\r')) && break
            i += 1
        end
        return i
    end
end

"""Objeto JSON plano -> Dict{String,String} (solo valores string; el resto se salta)."""
function jobj(s::AbstractString)
    d = Dict{String,String}()
    i = _skipws(s, 1)
    codeunit(s, i) == UInt8('{') || error("json: se esperaba '{'")
    i += 1
    while true
        i = _skipws(s, i)
        b = codeunit(s, i)
        b == UInt8('}') && break
        b == UInt8('"') || error("json: se esperaba clave '\"'")
        k, i = _jstring(s, i)
        i = _skipws(s, i)
        codeunit(s, i) == UInt8(':') || error("json: se esperaba ':'")
        i = _skipws(s, i + 1)
        if codeunit(s, i) == UInt8('"')
            v, i = _jstring(s, i)
            d[k] = v
        else
            i = _jskip(s, i)
        end
        i = _skipws(s, i)
        c = codeunit(s, i)
        c == UInt8(',') && (i += 1)
        c == UInt8('}') && break
        (c == UInt8(',') || c == UInt8('}')) || error("json: se esperaba ',' o '}'")
    end
    return d
end

# ----------------------------- extractor -----------------------------

"""Etiqueta del router = primeros 2 niveles de la ruta (mundo.submundo)."""
function label_de(ruta::AbstractString)
    parts = split(ruta, FLECHA)
    length(parts) < 2 && return nothing
    a, b = strip(parts[1]), strip(parts[2])
    (isempty(a) || isempty(b)) && return nothing
    return string(a, '.', b)
end

"""Limpia texto para fila TSV: saltos y tabs a espacio, espacios multiples colapsados."""
function _limpia(t::AbstractString)
    t = replace(t, '\r' => ' ', '\n' => ' ', '\t' => ' ')
    t = replace(t, r" {2,}" => " ")
    return strip(t)
end

function read_tsv(path)
    rows = Vector{Tuple{String,String}}()
    for line in eachline(path)
        line = strip(line)
        isempty(line) && continue
        ix = findfirst('\t', line)
        ix === nothing && continue
        push!(rows, (String(line[1:prevind(line, ix)]), String(line[nextind(line, ix):end])))
    end
    return rows
end

function main()
    srcdir = length(ARGS) >= 1 ? ARGS[1] :
             normpath(@__DIR__, "..", "..", "..", "..", "Shared", "IRIS_DESTILADO", "descargas")
    outpath = length(ARGS) >= 2 ? ARGS[2] : joinpath(@__DIR__, "router_data_f4c.tsv")
    origpath = joinpath(@__DIR__, "router_data.tsv")

    isfile(origpath) || error("falta $origpath (248 muestras F3); no se incluye en el repo")
    isdir(srcdir) || error("no existe el directorio del destilado: $srcdir (no se incluye en el repo)")

    rows = read_tsv(origpath)
    n_orig = length(rows)
    @printf("muestras originales (F3): %d\n", n_orig)

    files = sort(filter(f -> endswith(f, ".jsonl"), readdir(srcdir; join = true)))
    isempty(files) && error("no hay .jsonl en $srcdir")
    @printf("fuentes destilado: %d archivo(s)\n", length(files))

    pares = 0; anadidas = 0
    por_label = Dict{String,Int}()
    for f in files
        for line in eachline(f)
            line = strip(line)
            isempty(line) && continue
            d = jobj(line)
            ruta = get(d, "route", "")
            isempty(ruta) && continue
            lab = label_de(ruta)
            lab === nothing && continue
            get(d, "lang", "es") != "es" && continue
            prompt = _limpia(get(d, "prompt", ""))
            resp = _limpia(get(d, "response", ""))
            isempty(prompt) && isempty(resp) && continue
            pares += 1
            if !isempty(prompt)
                push!(rows, (lab, prompt))
                por_label[lab] = get(por_label, lab, 0) + 1
                anadidas += 1
            end
            if !isempty(resp)
                length(resp) > 900 && (resp = chop(resp; tail = length(resp) - 900))
                push!(rows, (lab, resp))
                por_label[lab] = get(por_label, lab, 0) + 1
                anadidas += 1
            end
        end
    end

    # control: cada fila con exactamente 2 campos y sin saldos internos
    for (lab, tx) in rows
        isempty(lab) && error("etiqueta vacia")
        isempty(tx) && error("texto vacio en $lab")
        findfirst('\t', tx) === nothing || error("tab interno en $lab: $(first(tx, 40))")
        findfirst('\n', tx) === nothing || error("salto interno en $lab: $(first(tx, 40))")
    end

    open(outpath, "w") do io
        for (lab, tx) in rows
            println(io, lab, '\t', tx)
        end
    end

    @printf("pares destilado usados: %d -> %d muestras nuevas\n", pares, anadidas)
    @printf("total: %d muestras en %s\n", length(rows), outpath)
    println("muestras nuevas por etiqueta:")
    ks = sort(collect(keys(por_label)); by = x -> (-por_label[x], x))
    for k in ks
        @printf("  %-28s %4d\n", k, por_label[k])
    end
    println("OK extractor")
end


if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
