using MLDatasets
using Knet

const F = Float32
include("../../utility/common.jl")

function predict(w, x, bmom; clip=false, pdrop=0.5, input_do = 0.0)
    x = mat(x)
    scale = w[1] isa Rec ? 1-pdrop : 1
    x = dropout(x, input_do; training= w[1] isa Rec ? true : false)
    if clip
        for i=1:2:length(w)-3
            x = sign.(sign.(w[i])*x .+ w[i+1])
        end
        x =  sign.(w[end-2])*x
        return x .+ w[end]
    else

        for i=1:2:length(w)-3
            N = size(w[i], 2)
            μ = w[i]*x .+ w[i+1]
            σ = i== 1 ? (1 .- w[1].*w[1]) * (x.*x) : 
                        N/scale .- (w[i].*w[i]) * (x.*x)
            x = @.  2H(-μ / √σ) - 1
            x = dropout(x, pdrop)
        end
        N = size(w[end-2], 2)
        μ = w[end-2]*x .+ w[end]
        σ = N/scale .- (w[end-2] .* w[end-2]) * (x .* x)
        return @. μ / √σ
    end
end

losslogH(y) = -sum(logH.(-y)) / size(y, 2)

function loss(w, x, y, bmom; pdrop=0.5, input_do = 0.0)
    ŷ = predict(w, x, bmom; pdrop=pdrop, input_do = input_do)
    y = onehot!(similar(ŷ), y)
    return losslogH((2 .* y .- 1) .* ŷ)
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
      ones(10, 1),
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
        infotime = 1,  # report every `infotime` epochs
        atype = gpu() >= 0 ? KnetArray{F} : Array{F},
        verb = 2,
        pdrop = 0.5,
        input_do = 0.0
        )

    info("using ", atype)
    seed > 0 && srand(seed)

    w = build_net(atype=atype)
    opt = [Adam(lr=lr) for _=1:length(w)]
    bmom = BatchMoments() # eventually only on softmax
    
    acctrn = 0
    acctst = 0
    
    
    
    report(epoch) = begin
            dtst = minibatch(xtst, ytst, batchsize; xtype=atype)
            dtrn = minibatch(xtrn, ytrn, batchsize; xtype=atype)
        
            acctrn = accuracy(w, dtrn, (w,x)->predict(w,x,bmom,clip=true))
            acctst = accuracy(w, dtst, (w,x)->predict(w,x,bmom,clip=true)) 
            println((:epoch, epoch,
                :trn, accuracy(w, dtrn, (w,x)->predict(w,x,bmom)) |> percentage,
                :trn_clip, acctrn  |> percentage,
                :tst, accuracy(w, dtst, (w,x)->predict(w,x,bmom)) |> percentage,
                :tst_clip, acctst  |> percentage
            ))
						if reportname != ""
                f = open(reportname, "a")
                print(f, epoch, "\t", acctrn, "\t", acctst, "\t")
                for lay = 1:2:length(w)-2
                    print(f, vecnorm(w[lay])/√length(w[lay]), "\t")
                end
                println(f)
                close(f)
            end

            if verb > 1
                for i=1:2:length(w)-2
                    n, m = length(w[i]), length(w[i+1])
                    print(" layer $(i÷2+1): W-norm $(round(vecnorm(w[i])/√n, 3))")
                    print(" Θ1-norm $(round(vecnorm(w[i+1])/√m, 3))\n")
                end
            end    
        end

    report(0); tic()
    @time for epoch=1:epochs
        for (x, y) in  minibatch(xtrn, ytrn, batchsize, shuffle=true, xtype=atype)
            dw = grad(loss)(w, x, y, bmom; pdrop=pdrop, input_do = input_do)
            for i=1:2:length(w)-2
                dw[i] = (1 - w[i] .* w[i]) .* dw[i]
            end
            update!(w, dw, opt)
            for i=1:2:length(w)-2
                w[i] = clip(w[i], 1e-2)
            end
        end
        (epoch % infotime == 0) && (report(epoch); toc(); tic())
        # acctrn == 100 && break
    end; toq()
    println("# FINAL RESULT acccuracy: train=$acctrn test=$acctst")
    return w
end
