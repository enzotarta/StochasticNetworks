using SpecialFunctions
using AutoGrad
import AutoGrad: Broadcasted
# import Knet:broadcast_func
using Knet

const F = Float32
H(x) = erfc(x / F(√2)) / 2
GH(x) = 2 / erfcx(x/F(√2)) / F(√(2π))


function broadcast_func(f)
    bf = Symbol("broadcast#", lstrip(string(f), '.'))
    if isdefined(Knet, bf)
        eval(Expr(:import, :Knet, bf))
    elseif isdefined(AutoGrad, bf)
        eval(Expr(:import, :AutoGrad, bf))
    else
        f = Symbol(f)
        if isdefined(Base, f)
            eval(Expr(:import, :Base, f))
        end
        @eval begin
            $bf(x...) = broadcast($f, x...)
            $f(x::Broadcasted...) = $bf(getval.(x)...) |> Broadcasted
        end
    end
    bf
end

logH(x) = log(H(x))
blogH = broadcast_func(logH)
@eval $blogH(x::KnetArray) = log.(H.(x))
@primitive logH(x),dy,y  (@. -dy*GH(x))
@eval @primitive $blogH(x),dy,y  (@. -dy*GH(x))

hardtanh(x) = relu(2 - relu(1-x)) - 1

fSign(x) = sign(x)
fSign_back(x) = (1+sign(1-abs(x))) / 2
bfSign = broadcast_func(fSign)
@eval $bfSign(x::KnetArray) = sign.(x)
@primitive fSign(x),dy,y  (@. dy*fSign_back(x))
@eval @primitive $bfSign(x),dy,y  (@. dy*fSign_back(x))

fTheta(x) = (1 +sign(x))/2
fTheta_back(x) = (1+sign(1-abs(x))) / 4
bfTheta = broadcast_func(fTheta)
@eval $bfTheta(x::KnetArray) = (1 .+ sign.(x)) ./ 4
@primitive fTheta(x),dy,y  (@. dy*fTheta_back(x))
@eval @primitive $bfTheta(x),dy,y  (@. dy*fTheta_back(x))

function clip(w, ϵ = 0)
    ϵ1 = F(1 - ϵ)
    w = relu.(w .+ ϵ1) .- ϵ1
    w = -relu.( ϵ1 .- w) .+ ϵ1
    return w
end

binreg(w) = mean((1 .- w) .* (1 .+ w))

function loadmnist(M=60_000, Mtst=10_000; fashion=false, preprocess=false)
    if fashion
        info("Loading FashionMNIST...")
        xtrn, ytrn = FashionMNIST.traindata(1:M)
        xtst, ytst = FashionMNIST.testdata(1:Mtst)
    else
        info("Loading MNIST...")
        xtrn, ytrn = MNIST.traindata(1:M)
        xtst, ytst = MNIST.testdata(1:Mtst)
    end
    ytst[ytst .== 0] .= 10
    ytrn[ytrn .== 0] .= 10
    if preprocess
        xtst .= (xtst .- mean(xtst, 3)) ./ (std(xtst,3) .+ F(1e-5))
        xtrn .= (xtrn .- mean(xtrn, 3)) ./ (std(xtrn,3) .+ F(1e-5))
    end
    xtrn = convert(Array{Float32}, xtrn)
    ytrn = convert(Array{Int}, ytrn)
    xtst = convert(Array{Float32}, xtst)
    ytst = convert(Array{Int}, ytst)
    return xtrn, ytrn, xtst, ytst
end


function loadcifar10(M=50000, Mtst=10000; preprocess=false)
    info("Loading CIFAR10...")
    xtrn, ytrn = CIFAR10.traindata()
    xtst, ytst = CIFAR10.testdata()
    ytst[ytst .== 0] .= 10
    ytrn[ytrn .== 0] .= 10
    if preprocess
        xtrn .= (xtrn .- mean(xtrn, 4)) ./ (std(xtrn,4) .+ F(1e-5))
        xtst .= (xtst .- mean(xtst, 4)) ./ (std(xtst,4) .+ F(1e-5))
    end
    xtrn = convert(Array{Float32}, xtrn)
    ytrn = convert(Array{Int}, ytrn)
    xtst = convert(Array{Float32}, xtst)
    ytst = convert(Array{Int}, ytst)
    return xtrn[:,:,:,1:M], ytrn[1:M], xtst[:,:,:,1:Mtst], ytst[1:Mtst]
end

setlr!(opt, lr) = for o in opt; o.lr =lr; end

# function findindices{T<:Integer}(y, a::Vector{T})
#     n = length(a)
#     indices = Vector{Int}(n)
#     y1 = size(y,1)
#     y2 = div(length(y),y1)
#     if n != y2; throw(DimensionMismatch()); end
#     @inbounds for j=1:n
#         indices[j] = (j-1)*y1 + a[j]
#     end
#     return indices
# end

function onehot(a::Vector, K)
    y = zeros(Int, K, length(a))
    @inbounds for i=1:length(a)
        y[a[i], i] = 1
    end
    y
end

function onehot!(y, a::Vector)
    y[:] = 0
    @inbounds for i=1:length(a)
        y[a[i], i] = 1
    end
    y
end

percentage(x) = round(x*100, 2)

mutable struct BatchMoments
    μ
    σ
    momentum
end
BatchMoments(; momentum=0.9f0) = BatchMoments(nothing, nothing, momentum)

function Base.push!(b::BatchMoments, μ, σ)
    if b.μ != nothing
        b.μ = b.momentum .* b.μ .+ (1 - b.momentum) .* μ
        b.σ = b.momentum .* b.σ .+ (1 - b.momentum) .* σ
    else
        b.μ = μ
        b.σ = σ
    end
end

getmoments(bm::BatchMoments) = bm.μ != nothing ? (bm.μ, bm.σ) : (0, 1)

import AutoGrad: getval, Rec

# Batch Normalization Layer
# works both for convolutional and fully connected layers
function batchnorm(w, x, bmom::BatchMoments; ϵ=Float32(1e-5))
    training = w isa Rec
    if training
        nd = ndims(x)
        d = nd == 2 ? (2,) : 
            nd == 4 ? (1,2,4) : error("wrong dimension")

        μ = mean(x, d)
        σ = sqrt.(ϵ .+ mean((x .- μ).*(x .- μ), d))
        push!(bmom, getval(μ), getval(σ))
    else
        μ, σ = getmoments(bmom)
    end
    return @. w[1] * (x - μ) / σ + w[2]
end
