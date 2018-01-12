using MLDatasets
using Knet

const F = Float32
include("../../utility/common.jl")

# all cnn are "same" 3x3  
NET = [128,128, 'M', 256, 256, 'M',
  512, 512,  'M', 'F', [1024]]


function predict(w, x, bmom; clip=false, pdrop=0.5, input_do = 0.0)
    i = 1
    x = reshape(x, 32, 32, 3, length(x)÷(32*32*3))
 
    for idx = 1:length(NET)
        NET[idx] == 'F' && break
        if NET[idx] != 'M'
            μ = conv4(w[i], x; padding=1)
            σ² = conv4(1 .- w[i] .* w[i], x .* x .+ 0.00000001; padding=1)
            if NET[idx+1] == 'M'
              μ = pool(μ, mode=1)
        			σ² = pool(σ², mode=1)
            end
            #x = batchnorm(w[i+1:i+2], x, bmom[i÷3+1])
            x = @.  2H(-μ / √σ²) - 1
            i += 3
        end
    end
    x = mat(x)
    for f in NET[end]
        μ = w[i] * x
        σ² = (1 .- w[i] .* w[i]) * (x .* x)
        #x = batchnorm(w[i+1:i+2], x, bmom[i÷3+1])
        x = @.  2H(-μ / √σ²) - 1
        i += 3    
    end

    μ = w[i] * x
    σ² = (1 .- w[i] .* w[i]) * (x .* x)
    x = @.  2H(-μ / √σ²) - 1
    #x = batchnorm(w[i+1:i+2], x, bmom[i÷3+1])
    i+= 3
    
    return x
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
