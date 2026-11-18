include(joinpath(@__DIR__, "UKJ.jl"))
using .UKJ

rutas = [
    UKJ.Ruta(UInt32(1), ["informatica", "programacion"], collect(Float32, 1.0f0:64.0f0), UInt32(64)),
    UKJ.Ruta(UInt32(2), ["matematicas", "calculo"], collect(Float32, 100.0f0:1.0f0:163.0f0), UInt32(64)),
    UKJ.Ruta(UInt32(3), ["sft"], collect(Float32, 1000.0f0:1.0f0:1063.0f0), UInt32(64)),
]
UKJ.write_ukj("test.ukj", "p4-one", rutas)
println("escrito test.ukj ", filesize("test.ukj"), " bytes")
