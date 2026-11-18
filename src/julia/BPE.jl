# BPE.jl — Tokenizer propio de Iris (BPE por bytes, formato binario iris.tok)
# Entrenamiento: BPE.train(texts, n_merges)  →  Tokenizer
# Uso:           BPE.encode(t, texto) -> Vector{UInt32};  BPE.decode(t, ids) -> String
# Persistencia:  BPE.save(path, t) / BPE.load_tok(path)

module BPE

export Tokenizer, train, encode, decode, save, load_tok, vocab_size

struct Tokenizer
    merges::Vector{Tuple{UInt32,UInt32}}
    bytes_of::Vector{Vector{UInt8}}   # bytes_of[id+1] = bytes del token id
end

vocab_size(t::Tokenizer) = UInt32(256 + length(t.merges))

"""Entrena BPE por bytes sobre una lista de textos."""
function train(texts::Vector{String}, n_merges::Int)
    seqs = [UInt32.(codeunits(String(t))) for t in texts]
    bytes_of = [[UInt8(i)] for i in 0:0xff]  # ids 0..255
    merges = Tuple{UInt32,UInt32}[]

    for _ in 1:n_merges
        counts = Dict{Tuple{UInt32,UInt32},Int}()
        for s in seqs
            for i in 1:length(s)-1
                k = (s[i], s[i+1])
                counts[k] = get(counts, k, 0) + 1
            end
        end
        isempty(counts) && break
        best = argmax(counts)
        counts[best] <= 1 && break
        new_id = UInt32(256 + length(merges))
        push!(bytes_of, vcat(bytes_of[best[1]+1], bytes_of[best[2]+1]))
        push!(merges, best)
        a, b = best
        for s in seqs
            j = 1
            out = Vector{UInt32}()
            while j <= length(s)
                if j < length(s) && s[j] == a && s[j+1] == b
                    push!(out, new_id); j += 2
                else
                    push!(out, s[j]); j += 1
                end
            end
            empty!(s); append!(s, out)
        end
    end
    Tokenizer(merges, bytes_of)
end

"""Codifica texto a tokens aplicando las fusiones en orden de entrenamiento.
La fusión i-ésima (base 1) produce el id 256+i-1."""
function encode(t::Tokenizer, text::AbstractString)
    s = UInt32.(codeunits(String(text)))
    for (i, (a, b)) in enumerate(t.merges)
        new_id = UInt32(256 + i - 1)
        j = 1
        out = Vector{UInt32}()
        while j <= length(s)
            if j < length(s) && s[j] == a && s[j+1] == b
                push!(out, new_id); j += 2
            else
                push!(out, s[j]); j += 1
            end
        end
        s = out
    end
    s
end

"""Decodifica tokens a texto (los primeros 256 ids son bytes directos)."""
function decode(t::Tokenizer, ids::Vector{UInt32})
    io = IOBuffer()
    for id in ids
        if id < 256
            write(io, UInt8(id))
        else
            write(io, t.bytes_of[id + 1])
        end
    end
    String(take!(io))
end

const MAGIC = UInt8[0x49, 0x52, 0x49, 0x53, 0x54, 0x4F, 0x4B, 0x00]  # "IRISTOK\0"

function save(path::AbstractString, t::Tokenizer)
    open(path, "w") do io
        write(io, MAGIC)
        write(io, UInt32(1))
        write(io, UInt32(length(t.merges)))
        for (a, b) in t.merges
            write(io, a); write(io, b)
        end
    end
    path
end

function load_tok(path::AbstractString)
    raw = read(path)
    @assert length(raw) >= 16 && raw[1:8] == MAGIC "no es un iris.tok"
    version = reinterpret(UInt32, raw[9:12])[1]
    @assert version == 1
    n = reinterpret(UInt32, raw[13:16])[1]
    merges = Tuple{UInt32,UInt32}[]
    bytes_of = [[UInt8(i)] for i in 0:0xff]
    off = 17
    for i in 1:n
        a = reinterpret(UInt32, raw[off:off+3])[1]
        b = reinterpret(UInt32, raw[off+4:off+7])[1]
        push!(merges, (a, b))
        push!(bytes_of, vcat(bytes_of[a+1], bytes_of[b+1]))
        off += 8
    end
    @assert off == length(raw) + 1 "iris.tok corrupto"
    Tokenizer(merges, bytes_of)
end

end # module
