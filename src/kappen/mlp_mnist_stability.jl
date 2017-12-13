using MLDatasets
using Knet
using SpecialFunctions
const F = Float32
include("../../utility/common.jl")

function predict(w, x; clip=false, pdrop=0.5, input_do = 0.0)
    x = mat(x)
    scale = w[1] isa Rec ? 1-pdrop : 1
    x = dropout(x, input_do; training= w[1] isa Rec ? true : false)
    if clip
      for i=1:2:length(w)-2
        x = sign.(sign.(w[i])*x .+ w[i+1])
      end
      x =  sign.(w[end-1])*x .+ w[end]
    	return x
    else
        for i=1:2:length(w)-3
            N = size(w[i], 2)
            μ = w[i]*x .+ w[i+1]
            σ = i== 1 ? (1 .- w[1].*w[1]) * (x.*x) :
                        N/scale .- (w[i].*w[i]) * (x.*x)
            x = @.  2H(-μ / √σ) - 1
            x = dropout(x, pdrop)
        end
        N = size(w[end-1], 2)
        μ = w[end-1]*x .+ w[end]
        σ = N/scale .- (w[end-1] .* w[end-1]) * (x .* x)
        x = @. μ/√σ
        return x
    end
end

function loss(w, x, y; clip = false, pdrop=0.0, input_do = 0.0, ρ=0.5, ϵ=0.000001)
  ŷ = predict(w, x; clip=clip, pdrop=pdrop, input_do = input_do)
  ŷ= H.(-ŷ)
  nm = onehot!(similar(ŷ), y)
  ŷ = ŷ ./(1.0 + ϵ - ŷ)
  ŷ -= ρ .* ŷ .* nm
	ŷ = ŷ ./ maximum(ŷ, 1)
  s = _sum_loss(nm .* ŷ) - _sum_loss(ŷ)
  return -s
end

function _sum_loss(ŷ)
  sum(log.(sum(ŷ, 1)))
end

function build_net(; atype=Array{F})
    w = [
      xavier(801, 28*28),
      zeros(801, 1),

      xavier(801, 801),
      zeros(801, 1),

      xavier(801, 801),
      zeros(801, 1),

      xavier(10, 801),
      zeros(10, 1),
    ]
    return map(a->convert(atype,a), w)
end

function main(xtrn, ytrn, xtst, ytst;
        seed = -1,
        batchsize = 200,
        lr = 1e-3,
        epochs = 200,
        reportname = "",
        out_bn = false,
        infotime = 1,  # report every `infotime` epochs
        atype = gpu() >= 0 ? KnetArray{F} : Array{F},
        verb = 2,
        pdrop = 0.5,
        ρ = 0.5,
        input_do = 0.25
        )

    info("using ", atype)
    seed > 0 && srand(seed)

    w = build_net(atype=atype)
    opt = [Adam(lr=lr) for _=1:length(w)]

    acctrn = 0
    acctst = 0

    report(epoch) = begin
            dtrn = minibatch(xtrn, ytrn, batchsize; xtype=atype)
            dtst = minibatch(xtst, ytst, batchsize; xtype=atype)
            acctrn = accuracy(w, dtrn, (w,x)->predict(w,x,clip=true))
            acctst = accuracy(w, dtst, (w,x)->predict(w,x,clip=true))
            println((:epoch, epoch,
                :trn, accuracy(w, dtrn, (w,x)->predict(w,x)) |> percentage,
                :trn_clip, acctrn  |> percentage,
                :tst, accuracy(w, dtst, (w,x)->predict(w,x)) |> percentage,
                :tst_clip, acctst  |> percentage
            ))
            if reportname != ""
                f = open(reportname, "a")
                print(f, epoch, "\t", acctrn, "\t", acctst, "\t")
                for lay = 1:2:length(w)
                    print(f, vecnorm(w[lay])/√length(w[lay]), "\t")
                end
                println(f)
                close(f)
            end

            if verb > 1
                for i=1:2:length(w)
                    n, m = length(w[i]), length(w[i+1])
                    print(" layer $(i÷2+1): W-norm $(round(vecnorm(w[i])/√n, 3))")
                    print(" Θ1-norm $(round(vecnorm(w[i+1])/√m, 3))\n")
                end
            end
        end

    report(0); tic()
    @time for epoch=1:epochs
        for (x, y) in  minibatch(xtrn, ytrn, batchsize, shuffle=true, xtype=atype)
            dw = grad(loss)(w, x, y; pdrop=pdrop, input_do = input_do, ρ = ρ)
            for i=1:2:length(w)
                dw[i] = (1 - w[i] .* w[i]) .* dw[i]
            end
            update!(w, dw, opt)
            for i=1:2:length(w)
                w[i] = clip(w[i], 1e-2)
            end
        end
        (epoch % infotime == 0) && (report(epoch); toc(); tic())
        # acctrn == 100 && break
    end; toq()
    println("# FINAL RESULT acccuracy: train=$acctrn test=$acctst")
    return w
end
