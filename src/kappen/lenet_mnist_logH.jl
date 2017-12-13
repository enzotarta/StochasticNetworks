using MLDatasets
using Knet

const F = Float32
include("../../utility/common.jl")


function predict(w, x, bmom; clip=false, pdrop=0.5, input_do = 0.0)
    i = 1
    x = reshape(x, 28, 28, 1, length(x)÷(28*28))
    x = dropout(x, input_do; training= w[1] isa Rec ? true : false)
    if clip 
        x = conv4(sign.(w[i]), x; padding=0) 
        x = pool(x, mode=1)
        x = sign.(x .+ w[i+1])
        i += 2    
    
        x = conv4(sign.(w[i]), x; padding=0) 
        x = pool(x, mode=1)
        x = sign.(x .+ w[i+1])
        i += 2    
    
        x = mat(x)
        
        x = sign.(w[i]) * x .+ w[i+1]
        x = sign.(x)
        i += 2 

        return sign.(w[i]) * x .+ w[i+2]
    else 
        scale = w[1] isa Rec ? 1-pdrop : 1
        
        μ = conv4(w[1], x) .+ w[2]
        σ² = conv4(1 .- w[1] .* w[1], x .* x)
        μ = pool(μ, mode=1)
        σ² = pool(σ², mode=1)
        x = @.  2H(-μ / √σ²) - 1

        N = prod(size(w[3], 1,2,3))
        μ = conv4(w[3], x) .+ w[4]
        σ² = N .- conv4(w[3].*w[3], x .* x) 
        μ = pool(μ, mode=1)
        σ² = pool(σ², mode=1)
        x = @.  2H(-μ / √σ²) - 1

        x = mat(x)
        x = dropout(x, pdrop)

        N = size(w[5], 2)
        μ = w[5]*x .+ w[6]
        σ² = N/scale .- (w[5] .* w[5]) * (x.*x)
        x = @.  2H(-μ / √σ²) - 1
        x = dropout(x, pdrop)
        
        N = size(w[7], 2)
        μ = w[7]*x .+ w[9]
        σ² = N/scale .- (w[7] .* w[7]) * (x.*x)
        return @. μ / √σ²
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
      xavier(5,5,1,20),
      zeros(1,1,20,1),
      
      xavier(5,5,20,50),
      zeros(1,1,50,1),
      
      xavier(500,800),
      zeros(500,1),
      
      xavier(10,500),
      ones(10,1),
      zeros(10,1),
    ]
    return map(a->convert(atype,a), w)
end

function main(xtrn, ytrn, xtst, ytst;
        seed = -1,
        batchsize = 200,
        lr = 1e-3,
        epochs = 200,
        infotime = 1,  # report every `infotime` epochs
        reportname = "",
        atype = gpu() >= 0 ? KnetArray{F} : Array{F},
        verb = 2,
        pdrop = 0.5,
        input_do = 0.0
        )

    info("using ", atype)
    seed > 0 && srand(seed)

    w = build_net(atype=atype)
    opt = [Adam(lr=lr) for _=1:length(w)]
    bmom = [BatchMoments() for _=1:(length(w)÷3)]
    
    acctrn = 0
    acctst = 0
    
    report(epoch) = begin
            dtrn = minibatch(xtrn, ytrn, batchsize; xtype=atype)
            dtst = minibatch(xtst, ytst, batchsize; xtype=atype)
            acctrn = accuracy(w, dtrn, (w,x)->predict(w,x, bmom, clip=true))
            acctst = accuracy(w, dtst, (w,x)->predict(w,x, bmom, clip=true)) 
            println((:epoch, epoch,
                :trn, accuracy(w, dtrn, (w,x)->predict(w,x, bmom)) |> percentage,
                :trn_clip, acctrn  |> percentage,
                :tst, accuracy(w, dtst, (w,x)->predict(w,x,bmom))  |> percentage,
                :tst_clip, acctst  |> percentage
            ))

            if reportname != ""
                open(reportname, "a") do f 
                    print(f, epoch, "\t", acctrn, "\t", acctst, "\t")
                    for lay = 1:2:length(w)-2
                        print(f, vecnorm(w[lay])/√length(w[lay]), "\t", mean(1 - w[lay].*w[lay]), "\t")
                    end
                    println(f)
                end
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
        # epoch == 10 && setlr!(opt, lr/=10)
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
    println("# FINAL RESULT acccuracy: train=$acctrn test=$acctst" , )
    return w
end
